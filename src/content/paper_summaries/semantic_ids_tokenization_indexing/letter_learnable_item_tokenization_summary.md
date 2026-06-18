---
title: "LETTER: Learnable Item Tokenization for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "letter_learnable_item_tokenization_summary"
catalogId: "paper-letter_learnable_item_tokenization_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/letter_learnable_item_tokenization_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2405.07314"
---
Подробное саммари статьи:

> **Авторы:** Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, Tat-Seng Chua.
>
> **Аффилиации:** National University of Singapore; University of Science and Technology of China; Hong Kong Polytechnic University.
>
> **Публикация:** CIKM 2024, Boise, ID, USA.

## 1. Коротко: о чем статья

LETTER предлагает learnable item tokenizer для генеративных рекомендательных систем, который одновременно учитывает три свойства идеального идентификатора: иерархическую семантику, collaborative сигналы и разнообразие назначения кодов. Метод строится поверх RQ-VAE и добавляет к стандартной semantic regularization два новых компонента: contrastive collaborative regularization, выравнивающую квантованные эмбеддинги с CF-представлениями, и diversity regularization, борющуюся с дисбалансом назначения кодов. Дополнительно предлагается ranking-guided generation loss с регулируемой температурой для подавления hard negatives. LETTER инстанцируется на двух generative recommender'ах (TIGER и LC-Rec) и стабильно улучшает качество на трех датасетах.

## 2. Контекст

Генеративная рекомендация формулирует задачу next-item prediction как seq2seq: модель по истории пользователя авторегрессионно генерирует дискретные токены, кодирующие целевой item. Для этого каждый item должен получить уникальный идентификатор -- последовательность токенов из компактного vocabulary. Качество такой токенизации критически влияет на downstream-модель: если token space плохо отражает структуру каталога, даже мощный Transformer будет учиться на некорректном "языке item'ов".

К моменту выхода LETTER существовали три типа идентификаторов: числовые ID (атомарные строки вроде "28,128"), текстовые (title, описание) и codebook-based (RQ-VAE в TIGER). Числовые ID не несут семантической информации и плохо обобщаются на cold-start. Текстовые идентификаторы семантически богаты, но не содержат collaborative signal и не имеют иерархической структуры от coarse к fine-grained. Codebook-based подходы создают иерархические коды через residual quantization, но опираются только на текстовые эмбеддинги и страдают от дисбаланса назначения кодов.

## 3. Проблема и мотивация

### 3.1. Отсутствие collaborative сигналов в кодах

Авторы демонстрируют ключевую проблему: два item'а с похожим текстовым описанием, но разным поведением пользователей, получают похожие token sequences в RQ-VAE. Это означает, что пространство семантических кодов не различает item'ы, которые различны с точки зрения collaborative filtering. Формально, если item $i$ и item $j$ имеют близкие текстовые эмбеддинги $\mathbf{s}_i \approx \mathbf{s}_j$, но далекие CF-эмбеддинги $\mathbf{h}_i \neq \mathbf{h}_j$, стандартный RQ-VAE назначит им почти одинаковые коды, что затрудняет генеративному recommender'у различение этих item'ов.

<figure class="paper-figure">
  <img src="../../assets/letter/method.png" alt="LETTER method: misalignment between semantic and collaborative signals" width="720">
  <figcaption>Рисунок метода демонстрирует эту проблему misalignment: item'ы с похожим текстом, но разными CF patterns получают одинаковые коды в стандартном RQ-VAE. LETTER решает это, добавляя collaborative regularization, которая разводит такие item'ы в code space.</figcaption>
</figure>

### 3.2. Code assignment bias и generation bias

Авторы эмпирически показывают на датасете Instruments с TIGER: распределение назначенных кодов сильно неравномерно. Небольшое количество кодов первого уровня используется для большинства item'ов, при этом именно item'ы с популярными кодами чаще генерируются моделью. Это создает порочный круг: популярные коды получают больше training signal, генеративная модель лучше их выучивает, они еще чаще генерируются. Item'ы с редкими кодами, наоборот, недообучаются.

### 3.3. Три критерия идеального идентификатора

LETTER формализует три требования к item tokenization: (1) иерархическая семантика -- token sequence должна кодировать информацию от coarse к fine-grained; (2) collaborative signals -- коды должны отражать user-item interaction patterns, а не только текстовую похожесть; (3) разнообразие кодов -- распределение назначений должно быть достаточно равномерным для предотвращения generation bias.

## 4. Метод LETTER

<figure class="paper-figure">
  <img src="../../assets/letter/overview.png" alt="LETTER overview: semantic, collaborative and diversity regularization" width="720">
  <figcaption>Общая схема LETTER: semantic regularization через RQ-VAE сохраняет content meaning, collaborative regularization выравнивает quantized embeddings с CF embeddings, а diversity regularization борется с дисбалансом использования codebook. Три потока loss'ов объединяются в единую objective токенизатора.</figcaption>
</figure>

### 4.1. Semantic regularization (RQ-VAE)

Основой LETTER остается Residual Quantized VAE. Для каждого item извлекается semantic embedding $\mathbf{s}$ через pretrained extractor (LLaMA-7B), который сжимается encoder'ом в latent representation:

$$
\mathbf{z} = \mathrm{Encoder}(\mathbf{s})
$$

Далее $\mathbf{z}$ квантуется через $L$ уровней codebook'ов $\mathcal{C}_l = \{\mathbf{e}_i\}_{i=1}^{N}$, где $N$ -- размер codebook, $\mathbf{e}_i \in \mathbb{R}^d$. Residual quantization работает итеративно:

$$
c_l = \arg\min_i \|\mathbf{r}_{l-1} - \mathbf{e}_i\|^2, \quad \mathbf{r}_l = \mathbf{r}_{l-1} - \mathbf{e}_{c_l}
$$

где $\mathbf{r}_0 = \mathbf{z}$. Идентификатор item'а -- tuple $\tilde{i} = [c_1, c_2, \ldots, c_L]$, квантованный эмбеддинг $\hat{\mathbf{z}} = \sum_{l=1}^{L} \mathbf{e}_{c_l}$. Semantic loss:

$$
\mathcal{L}_{\mathrm{Sem}} =
\|\mathbf{s} - \hat{\mathbf{s}}\|^2
+ \sum_{l=1}^{L}
\left(
\|\mathrm{sg}[\mathbf{r}_{l-1}] - \mathbf{e}_{c_l}\|^2
+ \mu\|\mathbf{r}_{l-1} - \mathrm{sg}[\mathbf{e}_{c_l}]\|^2
\right)
$$

где $\mathrm{sg}[\cdot]$ -- stop-gradient, $\mu = 0.25$ -- commitment coefficient.

### 4.2. Collaborative regularization

Ключевой вклад LETTER -- выравнивание квантованного эмбеддинга $\hat{\mathbf{z}}$ с CF-эмбеддингом $\mathbf{h}$, полученным из обученной CF-модели (в экспериментах -- SASRec, 32-мерные эмбеддинги). Используется contrastive loss:

$$
\mathcal{L}_{\mathrm{CF}} =
-\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp(\langle\hat{\mathbf{z}}_i, \mathbf{h}_i\rangle)}
{\sum_{j=1}^{B}\exp(\langle\hat{\mathbf{z}}_i, \mathbf{h}_j\rangle)}
$$

где $\langle \cdot, \cdot \rangle$ -- inner product, $B$ -- batch size. Интуиция: item'ы с похожими collaborative patterns должны получать похожие code sequences. Contrastive objective заставляет квантованный эмбеддинг item'а быть ближе к его собственному CF-эмбеддингу, чем к CF-эмбеддингам других item'ов в batch'е.

### 4.3. Diversity regularization

Для борьбы с code assignment bias авторы кластеризуют code embeddings в $K$ групп через constrained K-means и применяют contrastive regularization:

$$
\mathcal{L}_{\mathrm{Div}} =
-\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp(\langle\mathbf{e}_{c_l}^i, \mathbf{e}_+\rangle)}
{\sum_{j=1}^{N-1}\exp(\langle\mathbf{e}_{c_l}^i, \mathbf{e}_j\rangle)}
$$

где $\mathbf{e}_+$ -- случайный эмбеддинг из того же кластера, $\mathbf{e}_j$ -- все остальные code embeddings. Эта функция потерь сближает эмбеддинги одного кластера и разводит эмбеддинги разных кластеров, создавая более равномерное распределение кодов в embedding space.

### 4.4. Итоговая objective токенизатора

$$
\mathcal{L}_{\mathrm{LETTER}} =
\mathcal{L}_{\mathrm{Sem}}
+ \alpha\,\mathcal{L}_{\mathrm{CF}}
+ \beta\,\mathcal{L}_{\mathrm{Div}}
$$

где $\alpha$ и $\beta$ управляют балансом между semantic, collaborative и diversity компонентами.

### 4.5. Ranking-guided generation loss

При обучении генеративного recommender'а стандартный cross-entropy заменяется на loss с регулируемой температурой:

$$
\mathcal{L}_{\mathrm{rank}} =
-\sum_{t=1}^{|y|}\log P_\theta(y_t|y_{<t}, x),
\quad
P_\theta(y_t|y_{<t}, x) =
\frac{\exp(p(y_t)/\tau)}{\sum_{v \in \mathcal{V}}\exp(p(v)/\tau)}
$$

Авторы доказывают, что уменьшение температуры $\tau$ увеличивает вес hard negatives в градиенте. Теоретически показывается связь с one-way partial AUC (OPAUC): минимизация $\mathcal{L}_{\mathrm{rank}}$ эквивалентна оптимизации DRO upper bound на OPAUC, что напрямую коррелирует с Recall@K и NDCG@K.

### 4.6. Пошаговый алгоритм LETTER

1. **Извлечь два пространства item features.** Semantic embedding $\mathbf{s}$ берется из LLaMA-7B по тексту item'а, collaborative embedding $\mathbf{h}$ -- из обученного SASRec на interaction history.
2. **Запустить RQ-VAE quantization.** Encoder строит latent $\mathbf{z}$, затем $L$ residual codebooks последовательно выбирают codes $c_1,\ldots,c_L$, а decoder восстанавливает semantic embedding.
3. **Добавить semantic reconstruction.** $\mathcal{L}_{\mathrm{Sem}}$ удерживает код от потери content meaning и сохраняет coarse-to-fine residual structure.
4. **Добавить collaborative contrastive alignment.** $\mathcal{L}_{\mathrm{CF}}$ притягивает quantized embedding item'а к его SASRec embedding и отталкивает от других batch embeddings.
5. **Добавить diversity regularization.** Code embeddings кластеризуются constrained K-means; $\mathcal{L}_{\mathrm{Div}}$ помогает не сконцентрировать assignments в малой части codebook.
6. **Обучить tokenizer objective.** Минимизируется $\mathcal{L}_{\mathrm{LETTER}} = \mathcal{L}_{\mathrm{Sem}} + \alpha\mathcal{L}_{\mathrm{CF}} + \beta\mathcal{L}_{\mathrm{Div}}$; после этого item -> SID mapping фиксируется для downstream GR.
7. **Обучить generator с ranking-guided loss.** TIGER/LC-Rec backend генерирует SID target, а temperature $\tau$ усиливает hard negatives, которые важнее для Recall/NDCG.

## 5. Экспериментальный setup

### 5.1. Датасеты

<div class="table-scroll">
<table>
<thead>
<tr><th>Датасет</th><th>Домен</th><th>Preprocessing</th></tr>
</thead>
<tbody>
<tr><td>Instruments</td><td>Amazon, музыкальные инструменты</td><td>5-core filtering, leave-one-out</td></tr>
<tr><td>Beauty</td><td>Amazon, косметика</td><td>5-core filtering, leave-one-out</td></tr>
<tr><td>Yelp</td><td>рестораны / бизнесы</td><td>5-core filtering, leave-one-out</td></tr>
</tbody>
</table>
</div>

Для всех датасетов история пользователя ограничена 20 item'ами. Semantic embeddings извлекаются из LLaMA-7B, CF embeddings -- из обученного SASRec (32 измерения).

### 5.2. Hyperparameters токенизатора

Codebook: $L = 4$ уровня, $N = 256$ code embeddings на уровень, размерность $d = 32$. Число кластеров для diversity: $K = 10$. Обучение: AdamW, lr = 1e-3, batch size = 1024, 20k epochs. $\mu = 0.25$. $\alpha$ ищется в $\{0.1, 0.02, 0.01, 0.001\}$, $\beta$ в $\{0.01, 0.001, 0.0001, 0.00001\}$. Оборудование: 4x NVIDIA RTX A5000.

### 5.3. Baselines

Традиционные: MF, Caser, HGN, BERT4Rec, LightGCN, SASRec. LLM-based с разными идентификаторами: BIGRec (текст), P5-TID (текст), P5-SemID (metadata), P5-CID (collaborative spectral clustering), TIGER (RQ-VAE), LC-Rec (codebook + alignment tasks). Метрики: Recall@5/10, NDCG@5/10.

## 6. Основные результаты

<div class="table-scroll">
<table>
<thead>
<tr><th>Модель</th><th>Inst R@10</th><th>Inst N@10</th><th>Beauty R@10</th><th>Beauty N@10</th><th>Yelp R@10</th><th>Yelp N@10</th></tr>
</thead>
<tbody>
<tr><td>SASRec</td><td>0.0947</td><td>0.0690</td><td>0.0588</td><td>0.0313</td><td>0.0296</td><td>0.0152</td></tr>
<tr><td>P5-CID</td><td>0.0987</td><td>0.0751</td><td>0.0597</td><td>0.0347</td><td>0.0347</td><td>0.0181</td></tr>
<tr><td>TIGER</td><td>0.1058</td><td>--</td><td>--</td><td>0.0331</td><td>0.0407</td><td>0.0213</td></tr>
<tr><td>LC-Rec</td><td>0.1006</td><td>0.0772</td><td>0.0642</td><td>0.0374</td><td>0.0359</td><td>0.0199</td></tr>
<tr><td><strong>LETTER-TIGER</strong></td><td><strong>0.1122</strong></td><td><strong>0.0831</strong></td><td><strong>0.0672</strong></td><td><strong>0.0364</strong></td><td><strong>0.0426</strong></td><td><strong>0.0231</strong></td></tr>
<tr><td><strong>LETTER-LC-Rec</strong></td><td><strong>0.1115</strong></td><td><strong>0.0854</strong></td><td><strong>0.0703</strong></td><td><strong>0.0418</strong></td><td><strong>0.0393</strong></td><td><strong>0.0211</strong></td></tr>
</tbody>
</table>
</div>

LETTER стабильно и значимо улучшает оба backend'а на всех трех датасетах. Важно отметить: P5-CID (collaborative IDs) обычно превосходит P5-SemID (semantic IDs), что подтверждает важность collaborative signal в идентификаторах. Codebook-based методы (TIGER, LC-Rec) в целом сильнее ID-based и текстовых подходов.

## 7. Ablation study

<div class="table-scroll">
<table>
<thead>
<tr><th>Вариант</th><th>Inst R@10</th><th>Inst N@10</th><th>Beauty R@10</th><th>Beauty N@10</th></tr>
</thead>
<tbody>
<tr><td>TIGER (baseline)</td><td>0.1058</td><td>--</td><td>--</td><td>0.0331</td></tr>
<tr><td>+ collaborative reg.</td><td>0.1078</td><td>0.0810</td><td>0.0660</td><td>0.0351</td></tr>
<tr><td>+ diversity reg.</td><td>0.1075</td><td>0.0809</td><td>0.0618</td><td>0.0335</td></tr>
<tr><td>+ collaborative + diversity</td><td>0.1092</td><td>0.0819</td><td>0.0672</td><td>0.0357</td></tr>
<tr><td><strong>LETTER-TIGER (full)</strong></td><td><strong>0.1122</strong></td><td><strong>0.0831</strong></td><td><strong>0.0672</strong></td><td><strong>0.0364</strong></td></tr>
</tbody>
</table>
</div>

Каждый компонент вносит независимый вклад. Collaborative regularization дает основной прирост; diversity regularization помогает меньше по метрикам, но критически важна для code utilization. Полная конфигурация с ranking-guided loss дает лучший результат.

### 7.1. Code utilization

Diversity regularization существенно увеличивает число используемых кодов первого уровня. Однако collaborative regularization сама по себе снижает code utilization (со 148 до 76 на Instruments), потому что стягивает item'ы в кластеры по CF-похожести. Diversity regularization компенсирует этот эффект, обеспечивая баланс.

### 7.2. Collaborative signal alignment

Авторы проводят два теста. Первый: замена SASRec item embeddings на квантованные LETTER embeddings в ranking-задаче -- LETTER значимо лучше TIGER (Beauty: R@10 0.0343 vs 0.0213). Второй: подсчет code similarity для item'ов с похожими CF-паттернами -- LETTER дает code overlap 0.2760 на Instruments vs 0.0849 у TIGER. Это подтверждает, что collaborative regularization действительно переносит CF-структуру в code space.

## 8. Hyperparameter analysis

На Instruments с LETTER-TIGER:

- **Длина идентификатора $L$.** Качество растет от $L=2$ до $L=4$, затем падает из-за накопления ошибок авторегрессии. Оптимум: $L=4$.
- **Размер codebook $N$.** Увеличение $N$ помогает до 256; при 512 может быть overfitting.
- **$\alpha$ (collaborative reg.).** Оптимально около 0.02. Слишком большое значение мешает semantic regularization.
- **$\beta$ (diversity reg.).** Даже маленькое значение $10^{-4}$ значимо помогает. Слишком большое -- мешает остальным компонентам.
- **Кластеры $K$.** Оптимум при $K=10$. Слишком большие кластеры дают недостаточно близкие embeddings; слишком маленькие -- слишком похожие.
- **Температура $\tau$.** Снижение от 1.2 до примерно 0.7 улучшает качество за счет акцента на hard negatives. Слишком низкая $\tau$ рискует подавить valid positives.

## 9. Теоретические результаты

Авторы доказывают две теоремы. Первая: градиент $\mathcal{L}_{\mathrm{rank}}$ эквивалентен hard negative mining loss с весами $w(p(v), \tau)$, пропорциональными $p(v)$ -- чем ниже $\tau$, тем больше вес hard negatives (item'ов с высоким predicted score). Вторая: $\mathcal{L}_{\mathrm{rank}}$ связан с DRO-формулировкой, которая является верхней оценкой на one-way partial AUC. Через OPAUC устанавливается прямая связь с Recall@K и NDCG@K:

$$
\text{Recall@K} =
\mathbb{I}\bigl(\mathrm{OPAUC}(K/(|\mathcal{V}|-1)) > 0\bigr)
$$

$$
\text{NDCG@K} =
\frac{\mathbb{I}(\mathrm{OPAUC}(K/(|\mathcal{V}|-1)) > 0)}
{\log_2\bigl((K-1)(1 - \mathrm{OPAUC}(K/(|\mathcal{V}|-1))) + 2\bigr)}
$$

## 10. Сильные стороны

- **Четкая формулировка проблемы.** LETTER правильно идентифицирует три ортогональных недостатка существующих tokenizer'ов и предлагает по одному механизму на каждый. Анализ code assignment bias с конкретными эмпирическими данными убедителен.
- **Modular design.** Токенизатор отделен от генеративного recommender'а и может быть подключен к разным backbone'ам (TIGER, LC-Rec). Ablation показывает, что улучшение стабильно переносится.
- **Теоретическое обоснование.** Связь ranking-guided loss с OPAUC и downstream метриками -- нетривиальный вклад, выходящий за рамки простого heuristic design.
- **Collaborative signal интеграция.** Тест с заменой SASRec embeddings и измерение code overlap -- сильное эмпирическое доказательство, что CF-информация действительно попадает в code space.

## 11. Слабые стороны и ограничения

- **Зависимость от pretrained CF модели.** LETTER требует обученного SASRec (или аналога) для получения CF embeddings. Качество collaborative regularization напрямую зависит от качества этой модели. Если CF-модель плоха или обучена на шумных данных, collaborative alignment может навредить.
- **Дорогое обучение токенизатора.** 20,000 epochs -- значительный training budget. Авторы не исследуют convergence speed и не сравнивают wall-clock time с конкурентами.
- **Гиперпараметры.** Метод вводит как минимум 5 важных гиперпараметров ($\alpha$, $\beta$, $K$, $\tau$, $L$), каждый из которых требует отдельной настройки. Перенос оптимальных значений на новый домен не гарантирован.
- **Нет прямого анализа коллизий.** Несмотря на обсуждение code assignment bias, нет метрик collision rate (число item'ов с одинаковым token tuple), что затрудняет понимание, насколько diversity regularization решает проблему неуникальности кодов.
- **Только academic datsets.** Все три датасета -- относительно небольшие public benchmarks. Нет проверки на industrial-scale каталоге с миллионами item'ов и real-time обновлениями.
- **Static tokenization.** LETTER обучает токенизатор один раз. Нет механизма обновления кодов при появлении новых item'ов или изменении collaborative patterns.

## 12. Практические выводы

LETTER показывает, что item tokenization -- не нейтральный preprocessing, а важная точка оптимизации. Три вывода для практика:

Во-первых, добавление CF signal в токенизатор может дать значимый прирост при минимальном изменении pipeline: достаточно обучить отдельную CF-модель и добавить contrastive loss.

Во-вторых, code utilization -- важная диагностика. Без diversity regularization collaborative alignment может ухудшить разнообразие кодов. Мониторинг entropy/perplexity codebook'ов и code assignment distribution обязателен.

В-третьих, temperature в generation loss -- простой, но эффективный механизм для усиления hard negative mining. Оптимальная температура зависит от датасета и должна подбираться на валидации.

## 13. Связь с соседними работами

LETTER занимает промежуточную позицию между чисто semantic (TIGER) и end-to-end (ETEGRec) подходами. В отличие от CoST, который заменяет reconstruction loss на contrastive, LETTER сохраняет reconstruction и добавляет CF alignment как дополнительную regularization. В отличие от ETEGRec, LETTER не связывает токенизатор с recommender'ом через совместное обучение -- это проще, но оставляет tokenizer-recommender mismatch. Концептуально LETTER является предшественником для работ вроде PIT и CoFiRec, которые идут дальше в интеграции collaborative signals и hierarchical structure.

## 14. Итог

LETTER -- важная работа, которая четко формулирует три требования к item identifier'ам (семантика, CF, diversity) и предлагает по одному компоненту на каждое. Метод прост в реализации, переносим между backend'ами и подтвержден ablation'ами. Главная ценность -- не столько конкретные числа, сколько framework мышления: tokenizer для генеративной рекомендации должен быть не просто "сжатием текстовых embeddings", а retrieval-aware модулем, учитывающим collaborative structure и code balance. Ограничения связаны с зависимостью от pretrained CF модели, дорогим обучением и отсутствием dynamic update механизма.
