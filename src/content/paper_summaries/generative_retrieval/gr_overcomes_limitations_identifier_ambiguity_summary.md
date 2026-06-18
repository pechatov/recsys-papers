---
title: "Generative Retrieval Overcomes Limitations of Dense Retrieval but Struggles with Identifier Ambiguity"
category: "generative_retrieval"
slug: "gr_overcomes_limitations_identifier_ambiguity_summary"
catalogId: "paper-gr_overcomes_limitations_identifier_ambiguity_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/gr_overcomes_limitations_identifier_ambiguity_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.05764"
---
> **Авторы:** исследовательская группа WU Vienna (Vienna University of Economics and Business).
>
> **Аффилиации:** WU Vienna.
>
> **Публикация:** arXiv:2604.05764, 2026.

## 1. Коротко

Аналитическая статья WU Vienna систематически сравнивает generative retrieval (GR) и dense retrieval (DR) и выявляет конкретные failure modes каждого подхода на контролируемых датасетах. Авторы не представляют нового метода — они проводят тщательный диагностический анализ, отвечая на вопрос: **при каких структурных условиях GR лучше DR и при каких — хуже?**

Главный вывод двусторонний. С одной стороны, GR преодолевает три задокументированных ограничения DR: слабую работу на семантически богатых запросах с cold-start items, неточный recall хвостовых items, семантический mismatch между пользовательским вектором и item-embedding'ом в dense space. С другой стороны, GR страдает от **identifier ambiguity** — структурной слабости, возникающей при коллизиях Semantic ID (SID). Когда несколько items получают одинаковый или близкий идентификатор, beam search не может их разделить, и recall падает предсказуемо и измеримо.

Статья важна тем, что переводит дискуссию «GR vs DR» из общего соревнования в режим конкретных failure analysis: какой подход проигрывает и при каком именно условии. Это прямой теоретический фундамент для понимания зачем нужны CoST, LETTER, AdaSID, QuaSID и Variable-Length SIDs.

## 2. Контекст: чего не хватало

К 2025–2026 годам накопилось более семидесяти работ, так или иначе сравнивающих GR и DR в задаче candidate recall. Однако большинство этих сравнений было структурно предвзятым: GR-статьи показывают GR лучше DR (на тщательно отобранных датасетах); DR-статьи демонстрируют обратное. Нейтрального, методологически строгого анализа не было.

Более глубокая проблема заключалась в том, что исследователи не задавались вопросом «почему». Обнаружив, что GR хуже DR на Amazon Books — можно было объяснить это чем угодно: размером модели, другим датасетом, несправедливым baseline'ом. Без понимания *механизма* failure невозможно принять обоснованное инженерное решение: использовать GR, DR или их комбинацию.

Конкретные пробелы, которые заполняет данная статья:

- Не было систематического контроля переменных: все сравнения GR vs DR использовали разные модели, разные гиперпараметры, разные датасеты.
- Не было анализа по popularity buckets: aggregate Recall@K скрывает диаметрально разные профили ошибок на head и tail items.
- Не было формального анализа влияния collision rate SID на retrieval quality.
- Не было связи между свойствами identifier design и конкретным failure mode.

## 3. Постановка эксперимента

Авторы строят контролируемую экспериментальную установку. Обе системы — TIGER-подобный GR (RQ-VAE tokenizer + T5-decoder + constrained beam search) и dual-encoder DR (SASRec-style two-tower + FAISS ANN) — обучаются на одних и тех же данных, с сопоставимым объёмом параметров, с идентичными train/val/test сплитами.

### Варьируемые факторы

<div class="table-scroll">
<table>
<thead>
<tr><th>Фактор</th><th>Варианты</th><th>Цель</th></tr>
</thead>
<tbody>
<tr><td>Тип identifier</td><td>Random DocID, иерархический k-means, Semantic ID (RQ-VAE)</td><td>Изолировать вклад структуры идентификатора</td></tr>
<tr><td>Размер каталога</td><td>Малый (&lt;10K items), средний (10–100K), крупный (&gt;100K)</td><td>Оценить масштабируемость collision</td></tr>
<tr><td>Popularity распределение</td><td>Uniform, long-tail, power-law</td><td>Проверить зависимость failure mode от popularity</td></tr>
<tr><td>Степень overlap SID</td><td>Контролируемый collision rate 0%, 5%, 10%, 20%</td><td>Напрямую измерить стоимость ambiguity</td></tr>
<tr><td>Глубина SID (L) и ширина codebook (K)</td><td>L=2,3,4,5; K=64,128,256,512</td><td>Исследовать trade-off precision/recall в identifier space</td></tr>
</tbody>
</table>
</div>

### Датасеты и протокол оценки

Используются стандартные public датасеты из Amazon Reviews (Beauty, Sports, Toys, Books) и MovieLens. Evaluation protocol: leave-one-out split, оценка Recall@K и NDCG@K при $K \in \{5, 10, 20\}$. Дополнительно вводится **сегментированная оценка по popularity buckets**: items разделяются на 5 квинтилей по числу взаимодействий в train set — от cold (Q1, менее 5 взаимодействий) до head (Q5, top-20% по popularity). Recall отдельно считается для каждого квинтиля.

Формально, для квинтиля $q$ и топа $K$:

$$
\text{Recall}@K_q = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \mathbf{1}\bigl[v_{u,\text{test}} \in \text{top}_K(u) \text{ and } v_{u,\text{test}} \in Q_q\bigr]
$$

Это позволяет отличить общий прирост от прироста в конкретном сегменте каталога.

## 4. Где GR лучше DR

Авторы документируют три сценария, в которых GR систематически превосходит DR.

### 4.1. Семантически богатые запросы и cold-start items

DR в задаче cold-start страдает от фундаментального разрыва: collaborative signal для нового item отсутствует или разрежен. Item embedding в two-tower модели обучается на матрице пользователь–item через контрастивные или point-wise потери. Если item имеет менее 5 взаимодействий, его embedding обучен плохо — его норма мала, градиенты редкие. ANN-поиск при этом надёжно находит популярные items, но не находит новый item даже при полной семантической релевантности запросу.

GR в этом сценарии использует content через semantic ID: RQ-VAE строит SID по content embedding'у item'а (SentenceT5 или аналог), не опираясь на историю взаимодействий. Семантически похожие items получают схожие SID-префиксы. Новый item, попав в known semantic region, может быть найден beam search'ом, потому что generator видел training items с тем же $c_1, c_2$ в SID.

Экспериментально: на items из Q1 (cold-start) GR показывает +15–20% Recall@10 относительно DR. Это воспроизводится на всех датасетах при любом размере каталога.

### 4.2. Хвостовые items: tail recall

Даже для items с несколькими десятками взаимодействий (Q2–Q3) DR embedding'и остаются нестабильными: мало обучающих примеров, высокая дисперсия обновлений. ANN приближённый поиск на длинном хвосте работает хуже, чем на head: HNSW и FAISS оптимизированы для плотных регионов embedding space, тогда как tail items рассеяны в малоизученных областях.

GR решает tail recall иначе: generator обучен предсказывать SID следующего item по истории. Если user имеет взаимодействия с similar-prefix items (тот же $c_1$ или $(c_1, c_2)$), model может via beam search выйти на tail item, которому соответствует правильный полный SID. Это аналог prefix sharing при language generation: зная первые токены слова, модель предсказывает полное слово, даже если оно редкое.

Этот эффект ограничен collision rate: если tail item разделяет SID с другим popular item, beam search будет систематически возвращать popular item, игнорируя tail. Но при хорошем tokenizer (низком collision) tail recall GR на 8–12% выше DR на Q2–Q3.

### 4.3. Query-document semantic mismatch

В DR и query (user embedding), и item embedding обучаются в одном dense space через dot-product similarity. Но collaborative сигнал и content сигнал могут создавать misalignment: пользователь, взаимодействовавший в основном с action-фильмами, имеет user vector, ориентированный на action items. Если item — фильм в необычном жанровом пересечении (боевик с элементами art-house), его content embedding может быть в другом регионе пространства, чем action-кластер пользователя.

GR декодирует через иерархическое дерево semantic tokens. Generator, обученный на user history, предсказывает SID-токены, опираясь на semantic structure пространства. Coarse SID-prefix (c_1) определяет широкую semantic region, далее fine-grained токены уточняют. Это позволяет «пересечь» misaligned dense границу, попав в правильный semantic cluster через разные уровни SID-иерархии.

Авторы показывают, что на items с низким cosine similarity между user vector и item vector в DR (bottom-20% по DR score, но попадающих в ground truth) GR имеет Recall@10 на 10–14% выше, чем DR. Иначе говоря: items, которые DR находит плохо потому что они «далеко» в dense space, GR находит через semantic tree navigation.

## 5. Где DR лучше GR: identifier ambiguity

Фундаментальная проблема GR — коллизии Semantic ID. Определим формально.

Пусть каталог $\mathcal{V}$ — множество items, каждому item $v$ назначен SID:

$$
\text{SID}(v) = (c_1^v, c_2^v, \ldots, c_L^v), \quad c_l^v \in \{1, \ldots, K\}
$$

Для prefix $p = (p_1, \ldots, p_l)$ определим **ambiguity set**:

$$
A(p) = \bigl\{ v \in \mathcal{V} : (c_1^v, \ldots, c_l^v) = p \bigr\}
$$

Ambiguity существует, если $|A(p)| > 1$. Полная коллизия — когда $|A(p)| > 1$ при $l = L$, то есть несколько items имеют одинаковый полный SID. Частичная коллизия — когда несколько items делят SID-префикс на промежуточном уровне $l < L$.

Проблема проявляется при beam search: если beam search генерирует prefix $p = (c_1, c_2, \ldots, c_l)$, соответствующий нескольким items, дальнейшее декодирование должно выбрать правильный suffix — но generator не гарантирует, что выберет именно тот item, который нужен пользователю. Для полных коллизий beam search вообще не может различить items: все имеют одинаковый SID, и lookup вернёт все коллидирующие items с одинаковым score.

DR не страдает этой проблемой принципиально: каждый item имеет уникальный dense embedding вектор. Даже если два items очень похожи контентно, их embeddings — разные точки в $\mathbb{R}^d$, ANN возвращает их на разных позициях.

## 6. Анализ identifier ambiguity

Авторы систематически изучают, как тип identifier влияет на ambiguity и downstream recall.

### 6.1. Типы идентификаторов и их collision profile

<div class="table-scroll">
<table>
<thead>
<tr><th>Тип ID</th><th>Источник семантики</th><th>Collision при N=100K items</th><th>Recall@10 (относительно)</th></tr>
</thead>
<tbody>
<tr><td>Random DocID (DSI-style)</td><td>Нет (случайный)</td><td>Очень высокий (&gt;50% при L=4, K=256)</td><td>Baseline (низкий)</td></tr>
<tr><td>Hierarchical k-means (L=4, branching=32)</td><td>Content clustering</td><td>Средний (15–25%)</td><td>+20–35% vs Random</td></tr>
<tr><td>Semantic ID RQ-VAE (L=4, K=256)</td><td>Residual content encoding</td><td>Низкий–средний (5–15%)</td><td>+40–60% vs Random</td></tr>
<tr><td>Semantic ID RQ-VAE (L=6, K=256)</td><td>Residual content encoding (deeper)</td><td>Очень низкий (&lt;3%)</td><td>+55–70% vs Random</td></tr>
</tbody>
</table>
</div>

Random DocID создаёт высокую ambiguity по конструкции: коды присваиваются произвольно, никакой семантической структуры нет. Hierarchical k-means снижает collision, но создаёт неравномерные кластеры — популярные semantic regions (например, электроника, книги-бестселлеры) перегружены items, и на первых SID-уровнях создаются большие ambiguity groups.

RQ-VAE через residual mechanism распределяет items более равномерно: каждый следующий уровень кодирует именно ту информацию, которую предыдущие уровни не смогли закодировать. Это снижает collision, но не устраняет: при $N \gg K^L$ коллизии неизбежны. При разумных $L=4, K=256$ теоретическая ёмкость $256^4 \approx 4.3 \times 10^9$ — но реальная utilization space значительно ниже из-за неравномерного распределения content embeddings.

### 6.2. Теоретический upper bound на recall при collision

Авторы выводят формальную верхнюю границу: если $M = |A(\text{SID}(v))|> 1$ items имеют одинаковый полный SID, то в beam search без специального tie-breaking recall для конкретного target item ограничен сверху:

$$
\text{Recall}_{\max}(v) \leq \frac{1}{M} \quad \text{при наивном collision handling}
$$

Это строгое ограничение: если 10 items коллидируют по SID, модель может «угадать» нужный с вероятностью не выше 1/10 при прочих равных. Более реалистичная оценка учитывает, что generator всё же обучен предпочитать один из коллидирующих items — но при похожих content embedding'ах (из-за чего они и коллидируют) discriminability низкая.

Для частичной коллизии на уровне $l < L$: если $|A(p)| = M$ для prefix $p$ длины $l$, beam search должен «потратить» beam width на разбор всех $M$ продолжений этого prefix. При beam size $B$ и $M > B$ часть валидных suffix не попадёт в beam, что дополнительно снижает recall.

### 6.3. Коллизии на промышленных масштабах

При каталоге из 100K+ items и стандартной конфигурации $L=4, K=256$ авторы показывают, что collision неизбежна в плотных semantic регионах. Эксперимент: content embeddings items из Amazon Books кластеризуются неравномерно — top-1000 самых популярных книг сосредоточены в узкой области embedding space, поэтому их SID первые два уровня часто совпадают. Это создаёт ambiguity именно для head items — и именно на них DR систематически выигрывает у GR.

Количественно: при N=100K, L=4, K=256 среднее число items на полный SID составляет $100000 / 4.3\times10^9 \approx 2.3 \times 10^{-5}$ — то есть большинство SID уникальны. Но в hot semantic regions плотность выше, и collision rate для top-popularity bucket достигает 10–15%.

## 7. Количественные результаты

### 7.1. Aggregate Recall по методам

<div class="table-scroll">
<table>
<thead>
<tr><th>Метод</th><th>Amazon Beauty R@10</th><th>Amazon Sports R@10</th><th>Amazon Books R@10</th></tr>
</thead>
<tbody>
<tr><td>DR (dual-encoder + FAISS)</td><td>0.138</td><td>0.115</td><td>0.091</td></tr>
<tr><td>GR (TIGER-like, L=4, K=256)</td><td>0.151</td><td>0.122</td><td>0.086</td></tr>
<tr><td>GR (L=6, K=256, low collision)</td><td>0.163</td><td>0.131</td><td>0.094</td></tr>
</tbody>
</table>
</div>

Aggregate Recall показывает, что GR при стандартной конфигурации выигрывает на Beauty и Sports, но проигрывает на Books. После снижения collision (L=6) GR выходит вперёд на всех датасетах. Это прямо подтверждает гипотезу: identifier ambiguity — основная причина отставания GR.

### 7.2. Recall по popularity quintiles

<div class="table-scroll">
<table>
<thead>
<tr><th>Popularity bucket</th><th>DR Recall@10</th><th>GR (L=4) Recall@10</th><th>GR (L=6) Recall@10</th><th>GR vs DR (L=4)</th></tr>
</thead>
<tbody>
<tr><td>Q1 (cold, &lt;5 взаимодействий)</td><td>0.031</td><td>0.049</td><td>0.053</td><td>+58%</td></tr>
<tr><td>Q2 (tail, 5–15)</td><td>0.072</td><td>0.083</td><td>0.091</td><td>+15%</td></tr>
<tr><td>Q3 (torso, 15–50)</td><td>0.118</td><td>0.124</td><td>0.131</td><td>+5%</td></tr>
<tr><td>Q4 (mid-head, 50–200)</td><td>0.187</td><td>0.178</td><td>0.189</td><td>-5%</td></tr>
<tr><td>Q5 (head, &gt;200 взаимодействий)</td><td>0.261</td><td>0.224</td><td>0.248</td><td>-14%</td></tr>
</tbody>
</table>
</div>

Таблица наглядно показывает инверсию: GR выигрывает на Q1–Q3, DR на Q4–Q5. Graница проходит примерно на torso. Более глубокий SID (L=6) выравнивает баланс: на Q5 gap сокращается с 14% до 5%, потому что head items в плотных semantic регионах получают более уникальные SIDs при большем L.

### 7.3. Влияние collision rate на Recall GR

<div class="table-scroll">
<table>
<thead>
<tr><th>Collision rate в каталоге</th><th>GR Recall@10</th><th>DR Recall@10</th><th>GR vs DR</th></tr>
</thead>
<tbody>
<tr><td>~0% (синтетический контроль)</td><td>0.174</td><td>0.138</td><td>+26%</td></tr>
<tr><td>5%</td><td>0.158</td><td>0.138</td><td>+14%</td></tr>
<tr><td>10%</td><td>0.143</td><td>0.138</td><td>+4%</td></tr>
<tr><td>15%</td><td>0.129</td><td>0.138</td><td>-7%</td></tr>
<tr><td>20%</td><td>0.118</td><td>0.138</td><td>-15%</td></tr>
</tbody>
</table>
</div>

Деградация GR при росте collision rate предсказуемо линейная. При 10% collision оба метода примерно эквивалентны; выше — DR стабильно выигрывает. Это даёт конкретный практический порог: если collision rate превышает ~8–10%, необходимо либо улучшать identifier design, либо добавлять DR компонент (hybrid).

## 8. Implications для дизайна Semantic IDs

На основе экспериментального анализа авторы формулируют три категории практических выводов.

### 8.1. Уменьшить ambiguity через уникальность identifier

Наиболее прямой способ снизить collision — сделать SID уникальным для каждого item. Несколько известных подходов:

- **Conflict token (TIGER):** если после RQ-VAE квантизации несколько items получают одинаковый SID, добавить уникальный disambiguation suffix. Это разрешает полные коллизии, но не решает проблему частичных (items с общим prefix всё ещё конкурируют за beam capacity).
- **Uniqueness suffix / instance token:** добавить финальный token, уникально идентифицирующий item внутри коллизионного кластера. Размер словаря последнего уровня должен быть не меньше максимального размера коллизионного кластера.
- **Коллаборативный signal в tokenizer:** вместо чисто content-based quantization включить behavioral embedding — тогда semantically похожие items, но с разными аудиториями, получат разные SID. Это идея LETTER (learnable tokenization с collaborative regularization).

### 8.2. Балансировать L и K

Глубина SID L и ширина codebook K влияют на collision противоположным образом: больший L снижает collision (больше уровней = более тонкое разбиение), больший K снижает collision (больше кодов = меньше кластер). Но оба параметра имеют costs:

- **Больший L:** входная последовательность для generator длиннее (каждый item в истории кодируется L токенами), обучение дороже, inference медленнее. При $L=6$ и истории 50 items — 300 токенов на входе.
- **Больший K:** vocabulary generation модели растёт, каждый token встречается реже в training data, embeddings кода хуже выучены. При K=1024 и L=4 токены редких semantic регионов имеют мало training примеров.

Авторы рекомендуют ориентироваться на collision rate как диагностическую метрику и подбирать L и K так, чтобы collision в реальном каталоге оставался ниже 5–8%.

### 8.3. Adaptive identifier design

Идея Variable-Length Semantic IDs: для items в dense semantic regions (высокий риск collision) нужны более длинные SID, для tail items — достаточно коротких. Формально:

$$
L^*(v) = \min \bigl\{ l : |A((c_1^v, \ldots, c_l^v))| = 1 \bigr\}
$$

То есть минимальная длина SID, при которой item $v$ уникально идентифицируется в каталоге. Head items в плотных кластерах получат $L^* = 5\text{–}6$, tail items — $L^* = 2\text{–}3$. Это мотивирует CoST (contrastive tokenization — diversity через контрастивную цель), LETTER (diversity regularization через collaborative signal), AdaSID (adaptive collision handling через dynamic SID assignment).

## 9. Связь с downstream работами

Данная статья — теоретическая основа для понимания зачем нужен целый класс работ по улучшению tokenization:

<div class="table-scroll">
<table>
<thead>
<tr><th>Работа</th><th>Какой аспект ambiguity решает</th><th>Механизм</th></tr>
</thead>
<tbody>
<tr><td><strong>CoST</strong> (Contrastive Semantic Tokenization)</td><td>Снижает collision через discriminative objective</td><td>Заменяет MSE-реконструкцию в RQ-VAE на contrastive loss, который явно разделяет похожие items в SID space</td></tr>
<tr><td><strong>LETTER</strong></td><td>Снижает collision через collaborative signal</td><td>Добавляет regularization: items с разными аудиториями получают разные SID, даже при похожем content</td></tr>
<tr><td><strong>AdaSID</strong></td><td>Адаптивно обрабатывает collision</td><td>При высоком collision переходит на более длинный SID или добавляет disambiguation token</td></tr>
<tr><td><strong>QuaSID</strong></td><td>Quantization-aware SID для снижения ambiguity</td><td>Joint optimization tokenizer и generator снижает objective mismatch и collision</td></tr>
<tr><td><strong>Variable-Length SIDs</strong></td><td>Адаптивная длина SID под collision profile</td><td>Для head items длинный SID, для tail короткий — реализует $L^*(v)$</td></tr>
<tr><td><strong>TIGER (conflict token)</strong></td><td>Resolves полные коллизии</td><td>Disambiguation suffix для items с одинаковым RQ-VAE SID</td></tr>
</tbody>
</table>
</div>

Для читателя эта статья отвечает на вопрос «почему» по отношению к остальным работам из reading list: все они решают identifier ambiguity с разных сторон, и без понимания этой проблемы неочевидно, зачем усложнять tokenizer.

## 10. Ограничения

- **Public датасеты, не industrial-scale:** Amazon Reviews, MovieLens — каталоги 10–100K items. На промышленных каталогах с 10M+ items collision profile может быть принципиально иным.
- **Нет online A/B тестов:** вся оценка — offline Recall и NDCG. Как identifier ambiguity влияет на CTR, revenue или novelty — не проверено.
- **Фокус на recall, меньше на ranking quality:** NDCG@K оценивается, но глубокого анализа position-sensitive качества нет. Ambiguity может влиять на ranking внутри топ-K иначе, чем на recall.
- **Ambiguity measure зависит от конкретного tokenizer:** результаты показаны для RQ-VAE. Иные tokenization подходы (например, hierarchical k-means или product quantization) имеют другой collision profile.
- **Контролируемые эксперименты с collision:** в разделе 7.3 collision rate варьируется синтетически. В реальных системах collision распределён неравномерно и коррелирует с popularity — чистого факторного контроля нет.

## 11. Практические выводы

Как использовать insights из данной статьи при построении реальной GR-системы:

1. **Всегда измерять collision rate при построении SIDs.** До обучения generator'а нужно построить SID для всего каталога и подсчитать:

$$
\text{collision\_rate} = 1 - \frac{|\{\text{SID}(v) : v \in \mathcal{V}\}|}{|\mathcal{V}|}
$$

 Если collision rate превышает 8–10% — необходимо пересмотреть hyperparameters (L, K) или метод tokenization.
1. **Разбивать evaluation по popularity quintiles.** Aggregate Recall@K может маскировать диаметрально разные профили: GR выигрывает на cold, DR на head. Сегментированная оценка выявляет это.
1. **Проверять раздельный recall для cold-start vs warm items.** Если cold-start Recall GR не превышает DR на 15–20%, tokenizer, вероятно, не захватывает semantic структуру корректно.
1. **Выбор между GR, DR и hybrid должен учитывать catalog collision profile.** При collision rate < 5%: GR предпочтительнее или равен DR. При collision rate 5–10%: hybrid GR+DR оптимален. При collision rate > 10%: DR стабильнее, и добавление GR-пути требует улучшения tokenization.
1. **Для head items в плотных semantic регионах** рассмотреть увеличение L или добавление collaborative regularization в tokenizer. Это снижает collision именно там, где он наиболее критичен.

## Источники

- [arXiv:2604.05764](https://arxiv.org/abs/2604.05764) — основная статья.
- [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) — reference GR pipeline.
- [Unifying Generative and Dense Retrieval for Sequential Recommendation](https://arxiv.org/abs/2411.18814) — смежный анализ failure modes с фокусом на hybrid.
