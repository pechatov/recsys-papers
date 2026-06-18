---
title: "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
category: "semantic_ids_tokenization_indexing"
slug: "actions_speak_louder_than_words_trillion_parameter_sequential_summary"
catalogId: "paper-actions_speak_louder_than_words_trillion_parameter_sequential_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/actions_speak_louder_than_words_trillion_parameter_sequential_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2402.17152"
---
Подробное саммари статьи:

> **Авторы:** Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, Yinghai Lu, Yu Shi.
>
> **Аффилиации:** Meta AI.

## 1. Коротко: о чем статья

Статья предлагает переформулировать промышленные рекомендательные системы как задачу **sequential transduction** в рамках парадигмы Generative Recommenders (GR). Ключевой вклад -- архитектура **HSTU (Hierarchical Sequential Transduction Units)**, которая заменяет классические Deep Learning Recommendation Models (DLRMs) и демонстрирует масштабирование до **1.5 триллиона параметров** с деплоем в production на платформе Meta. HSTU работает в 5.3x-15.2x быстрее, чем FlashAttention2-based Transformers на последовательностях длиной 8192, и дает до +65.8% NDCG на публичных датасетах и +12.4% улучшение ключевых онлайн-метрик.

Это принципиально важная работа для контекста semantic ID исследований, потому что она показывает, что **действия пользователей (actions) сами по себе могут быть модальностью для генеративного моделирования**, без обязательного сжатия item'ов в compact semantic codes. Она ставит вопрос: нужны ли вообще semantic IDs, если можно масштабировать GR напрямую через действия?

## 2. Контекст: зачем это нужно

Промышленные рекомендательные системы исторически строились на DLRMs, которые комбинируют гетерогенные фичи: высококардинальные ID (liked items, followed creators), численные счетчики (decayed CTR по топикам), категориальные признаки (город запроса, язык) и другие сигналы. Эти модели обрабатывают десятки миллиардов действий пользователей ежедневно, но имеют фундаментальное ограничение: **качество не масштабируется с ростом compute**, в отличие от языковых и визуальных моделей, которые демонстрируют power-law scaling laws.

Одновременно развивается линия генеративного retrieval с semantic IDs (TIGER, DSI), где item'ы представляются дискретными кодами, и seq2seq модель генерирует коды следующего item'а. Но эти подходы работают на относительно небольших каталогах и не показаны в промышленном масштабе с миллиардами пользователей.

HSTU/GR от Meta предлагает третий путь: **не сжимать item'ы в semantic codes, а напрямую моделировать последовательность действий** как временной ряд в едином feature space, обрабатываемый мощным self-attention encoder'ом.

## 3. Проблема и мотивация

Авторы выделяют три ключевые проблемы, которые нужно решить для generative recommendation в промышленном масштабе:

**Отсутствие явной структуры в фичах.** В отличие от текста, где слова формируют последовательности, рекомендательные фичи гетерогенны: embedding'и, counters, ratios, cross-features с кардинальностью в миллиарды. Нет естественного "словаря" размером 100K, как в NLP.

**Нестационарный vocabulary масштаба миллиардов.** Каталог item'ов постоянно меняется: новые товары, удаленные посты, обновленный контент. В NLP словарь фиксирован; здесь vocabulary динамический и на порядки больше.

**Вычислительная стоимость.** Крупнейшие платформы обрабатывают на порядки больше токенов в день, чем LLM обрабатывает за месяцы обучения. Стандартный Transformer с softmax attention становится prohibitively expensive на длинных пользовательских историях (до $10^5$ действий).

## 4. От DLRM к Generative Recommenders: переформулировка задачи

### 4.1. Объединение гетерогенных фичей

GR консолидирует все фичи в единый временной ряд. Категориальные фичи (liked items, followed creators, языки, города) упорядочиваются хронологически: самый длинный ряд (обычно объединенные item engagement features) берется как основной, остальные ряды сжимаются путем сохранения только первого элемента каждого consecutive сегмента и вмерживаются в основной ряд.

Численные фичи (decayed counters, CTR по топикам) в традиционных DLRM обновляются с каждым взаимодействием, что делает полную секвенциализацию невозможной. Но авторы показывают, что категориальные фичи, по которым строятся агрегации, уже секвенциализированы в GR. Поэтому при достаточно мощной архитектуре и длинных последовательностях численные фичи могут быть удалены.

### 4.2. Ranking и Retrieval как sequential transduction

Формально: дана последовательность из $n$ токенов $x_0, x_1, \ldots, x_{n-1}$, где $x_i \in \mathbb{X}$, упорядоченных хронологически с метками времени $t_0, t_1, \ldots, t_{n-1}$. Sequential transduction отображает входы в выходы $y_0, y_1, \ldots, y_{n-1}$, где $y_i \in \mathbb{X} \cup \{\emptyset\}$.

Пусть $\Phi_i \in \mathbb{X}_c$ -- контент, показанный пользователю, а $a_i$ -- действие пользователя (клик, лайк, просмотр). Для **ranking** задача формулируется как предсказание действия $a_{i+1}$ с учетом всей истории контента и действий, включая текущий target $\Phi_{i+1}$ (target-aware формулировка). Для **retrieval** задача -- предсказание следующего контента $\Phi_{i+1}$, с которым пользователь позитивно взаимодействует.

### 4.3. Generative training

Традиционное impression-level обучение self-attention моделей имеет сложность:

$$
\sum_i n_i(n_i^2 d + n_i d_{\mathrm{ff}} d)
$$

При $N = \max_i n_i$ общая сложность составляет $O(N^3 d + N^2 d^2)$, что prohibitively expensive. Generative training снижает сложность на множитель $O(N)$ за счет амортизации стоимости encoder'а по нескольким целевым item'ам. При сэмплировании $i$-го пользователя с частотой $s_u(n_i) = 1/n_i$ сложность снижается до $O(N^2 d + N d^2)$. На практике это реализуется выпуском training examples в конце пользовательской сессии.

## 5. Архитектура HSTU

HSTU состоит из стека идентичных слоев с residual connections. Каждый слой содержит три подслоя.

### 5.1. Pointwise Projection

$$
U(X), V(X), Q(X), K(X) = \mathrm{Split}(\phi_1(f_1(X)))
$$

где $f_1(X) = W_1 X + b_1$ -- одиночный линейный слой, а $\phi_1$ -- SiLU активация.

### 5.2. Spatial Aggregation

$$
A(X) V(X) = \phi_2(Q(X) K(X)^\top + r_{ab}^{p,t}) V(X)
$$

где $r_{ab}^{p,t}$ -- relative attention bias, включающий позиционную ($p$) и временную ($t$) информацию, а $\phi_2$ -- также SiLU. Ключевое отличие от стандартного Transformer: **pointwise aggregated (не-softmax) attention**.

### 5.3. Pointwise Transformation

$$
Y(X) = f_2(\mathrm{Norm}(A(X)V(X)) \odot U(X))
$$

Элемент $\mathrm{Norm}(A(X)V(X)) \odot U(X)$ интерпретируется как вариант SwiGLU gating: attention-pooled features напрямую взаимодействуют с gate U(X), виртуально выполняя Mixture-of-Experts gating.

### 5.4. Почему не softmax attention

Авторы мотивируют отказ от softmax двумя аргументами. Во-первых, количество прошлых взаимодействий с похожим контентом напрямую указывает на интенсивность предпочтения пользователя; после softmax нормализации этот сигнал теряется. Во-вторых, softmax хуже работает с нестационарными vocabulary в streaming-настройках. Результаты на синтетических данных показывают разрыв до 44.7% HR@10 между softmax и pointwise attention.

<div class="table-scroll">
<table>
<tr><th>Архитектура</th><th>HR@10</th><th>HR@50</th></tr>
<tr><td>Transformers (softmax)</td><td>0.0442</td><td>0.2025</td></tr>
<tr><td>HSTU (без $r_{ab}^{p,t}$, softmax)</td><td>0.0617</td><td>0.2496</td></tr>
<tr><td>HSTU (без $r_{ab}^{p,t}$, pointwise)</td><td>0.0893</td><td>0.3170</td></tr>
</table>
</div>

## 6. Эффективность HSTU: Stochastic Length и память

### 6.1. Raggified attention и Stochastic Length

Распределение длин историй пользователей strongly skewed. HSTU выполняет raggified attention как grouped GEMMs разных размеров, что дает 2-5x throughput gains. Stochastic Length (SL) дополнительно увеличивает sparsity алгоритмически: для пользователей с длинными историями входная последовательность субсэмплируется до $N_c^{\alpha/2}$ с вероятностью $1 - N_c^{\alpha}/n_{c,j}^2$, что снижает сложность attention до $O(N^{\alpha} d)$ при $\alpha \in (1, 2]$.

<div class="table-scroll">
<table>
<tr><th>$\alpha$</th><th>Seq 1024</th><th>Seq 2048</th><th>Seq 4096</th><th>Seq 8192</th></tr>
<tr><td>1.6</td><td>71.5%</td><td>76.1%</td><td>80.5%</td><td>84.4%</td></tr>
<tr><td>1.8</td><td>40.2%</td><td>45.3%</td><td>54.1%</td><td>66.4%</td></tr>
<tr><td>2.0</td><td>3.1%</td><td>6.6%</td><td>29.1%</td><td>64.1%</td></tr>
</table>
</div>

При $\alpha=1.6$ последовательность длиной 4096 большую часть времени сжимается до 776 токенов (более 80% токенов удалено), при этом NE не деградирует более чем на 0.002.

### 6.2. Activation memory

В рекомендательных системах (в отличие от LLM) критичен большой batch size. HSTU минимизирует activation memory: **14d на слой** (bfloat16) против **33d на слой** у стандартного Transformer. Это позволяет масштабировать сеть до >2x большей глубины. Для embedding vocabulary размером 10B при dimension 512 используются rowwise AdamW optimizers с optimizer states на DRAM, снижая HBM usage с 12 до 2 байт на float.

## 7. M-FALCON: масштабирование inference

Для target-aware ranking нужно обработать $m$ кандидатов. M-FALCON (Microbatched-Fast Attention Leveraging Cacheable OperatioNs) обрабатывает $b_m$ кандидатов параллельно, модифицируя attention masks так, что cross-attention стоимость снижается с $O(b_m n^2 d)$ до $O((n + b_m)^2 d) = O(n^2 d)$ при $b_m \ll n$. Кандидаты разбиваются на $\lceil m/b_m \rceil$ microbatches с KV caching между forward passes. Результат: модель, **в 285 раз более вычислительно сложная**, работает с 1.5x-3x большим QPS при том же inference budget.

## 8. Эксперименты и результаты

### 8.1. Публичные датасеты (multi-pass, full-shuffle)

<div class="table-scroll">
<table>
<tr><th>Метод</th><th>Датасет</th><th>HR@10</th><th>NDCG@10</th><th>NDCG@200</th></tr>
<tr><td>SASRec</td><td>ML-1M</td><td>0.2853</td><td>0.1603</td><td>0.2498</td></tr>
<tr><td>HSTU</td><td>ML-1M</td><td>0.3097 (+8.6%)</td><td>0.1720 (+7.3%)</td><td>0.2606 (+4.3%)</td></tr>
<tr><td>HSTU-large</td><td>ML-1M</td><td>0.3294 (+15.5%)</td><td>0.1893 (+18.1%)</td><td>0.2771 (+10.9%)</td></tr>
<tr><td>SASRec</td><td>ML-20M</td><td>0.2906</td><td>0.1621</td><td>0.2521</td></tr>
<tr><td>HSTU-large</td><td>ML-20M</td><td>0.3567 (+22.8%)</td><td>0.2106 (+30.0%)</td><td>0.2971 (+17.9%)</td></tr>
<tr><td>SASRec</td><td>Books</td><td>0.0292</td><td>0.0156</td><td>0.0350</td></tr>
<tr><td>HSTU-large</td><td>Books</td><td>0.0469 (+60.6%)</td><td>0.0257 (+65.8%)</td><td>0.0508 (+45.1%)</td></tr>
</table>
</div>

HSTU-large использует 4x больше слоев и 2x больше attention heads. На датасете Books улучшение достигает **+65.8% NDCG@10**.

### 8.2. Промышленный streaming

Обучение на более чем 100B примеров на 64-256 H100 GPU. Для ranking используется Normalized Entropy (NE); снижение на 0.001 NE считается значимым и примерно эквивалентно ~0.5% улучшению top-line метрики для миллиардов пользователей.

<div class="table-scroll">
<table>
<tr><th>Архитектура</th><th>Retrieval (log pplx)</th><th>Ranking NE (E-Task)</th><th>Ranking NE (C-Task)</th></tr>
<tr><td>Transformers</td><td>4.069</td><td>NaN (loss explosions)</td><td>NaN</td></tr>
<tr><td>HSTU (softmax, без bias)</td><td>4.024</td><td>0.5067</td><td>0.7931</td></tr>
<tr><td>Transformer++</td><td>4.015</td><td>0.4945</td><td>0.7822</td></tr>
<tr><td>HSTU (полный)</td><td><strong>3.978</strong></td><td><strong>0.4937</strong></td><td><strong>0.7805</strong></td></tr>
</table>
</div>

Стандартные Transformers показали frequent loss explosions в ranking даже при пониженных learning rates. HSTU дает лучшее качество при 1.5-2x большей скорости и 50% меньшем HBM usage.

### 8.3. GR vs DLRM в production

<div class="table-scroll">
<table>
<tr><th>Метод</th><th>HR@100</th><th>HR@500</th><th>Online E-Task</th><th>Online C-Task</th></tr>
<tr><td>DLRM (baseline)</td><td>29.0%</td><td>55.5%</td><td>+0%</td><td>+0%</td></tr>
<tr><td>GR (content-based)</td><td>11.6%</td><td>18.8%</td><td>--</td><td>--</td></tr>
<tr><td>GR (interactions only)</td><td>35.6%</td><td>61.7%</td><td>--</td><td>--</td></tr>
<tr><td>GR (new source)</td><td>36.9%</td><td>62.4%</td><td>+6.2%</td><td>+5.0%</td></tr>
</table>
</div>

Для ranking GR дает **+12.4%** улучшение E-Task online metric. Критически важный результат: content-based GR (аналог semantic ID подхода) показывает HR@100 всего 11.6% против 29.0% у DLRM и 35.6% у action-based GR. Это прямой аргумент, что **высококардинальные user actions важнее content semantics** для промышленного retrieval.

### 8.4. Scaling laws

Авторы масштабируют GR через гиперпараметры HSTU (число слоев, длину последовательности, dimension embedding'ов, число attention heads) и сравнивают с DLRM-вариантами (Transformers, DHEN, DCN). Ключевые находки:

- В режиме низкого compute DLRM может превосходить GR благодаря handcrafted features.
- GR демонстрирует существенно лучшую масштабируемость с FLOPs, тогда как DLRM performance plateaus.
- GR масштабируется до **1.5 триллиона параметров** ; DLRM насыщается на ~200 миллиардах.
- Все основные метрики (HR@100, HR@500, NE) **масштабируются как power law от compute** через три порядка величины.
- Длина последовательности играет значительно более важную роль в GR, чем в language modeling.
- Общий compute важнее конкретных hyperparameters: в разумном диапазоне точные параметры модели менее критичны, чем суммарный training compute.

### 8.5. Пошаговая схема HSTU/GR

Алгоритм работы Meta GR можно читать как замену DLRM feature table на длинную action sequence, которую HSTU обрабатывает одним amortized encoder'ом.

1. **Сконсолидировать feature streams.** Категориальные события пользователя упорядочиваются по времени; самый длинный engagement stream становится основой, остальные categorical streams вмерживаются после сжатия consecutive segments.
1. **Удалить часть ручных numeric aggregations.** Decayed CTR/counters не подаются как отдельные DLRM-фичи, если последовательность содержит исходные categorical events, из которых мощная модель может восстановить паттерн.
1. **Сформировать training examples generative способом.** Вместо impression-level обучения на каждом candidate item encoder прогоняется по пользовательской истории и амортизирует cost на несколько target positions.
1. **Применить HSTU layer.** Pointwise projection строит $U,V,Q,K$, spatial aggregation считает SiLU attention без softmax с relative position/time bias, pointwise transformation смешивает aggregated signal через gate.
1. **Сжать длинные histories через Stochastic Length.** Для длинных sequences часть tokens subsample'ится по правилу, зависящему от $\alpha$, чтобы attention cost был управляемым.
1. **Для ranking использовать M-FALCON.** Target-aware candidates обрабатываются microbatches с mask/KV caching, чтобы стоимость не росла линейно как отдельный full forward на каждого кандидата.
1. **Разделить retrieval и ranking targets.** Retrieval предсказывает следующий content $\Phi_{i+1}$, ranking предсказывает action $a_{i+1}$ для заданного target content.
1. **Масштабировать и проверять power law.** Увеличиваются depth, dimensions, heads и sequence length; результат оценивается по HR/NE/online metrics и по scaling curve, а не по одной точке.

```
for each user:
    events = merge_chronological_categorical_streams(user_logs)
    events = stochastic_length_sample(events, alpha)

    X = embed(events)
    for layer in HSTU:
        U, V, Q, K = pointwise_projection(X)
        A = silu(Q @ K.T + relative_position_time_bias)
        X = pointwise_transform(norm(A @ V) * U)

    retrieval_loss = predict_next_content(X)
    ranking_loss = predict_next_action_with_target(X, candidates via M-FALCON)
    update model at scale
```

## 9. Сильные стороны

### 9.1. Реальная production валидация

В отличие от большинства работ по semantic IDs, HSTU развернут в production на платформе с миллиардами пользователей. Онлайн A/B тест с +12.4% на E-Task -- это очень сильный результат, который нельзя отбросить как artifact offline evaluation.

### 9.2. Первые scaling laws для рекомендаций

Статья впервые показывает, что quality-compute curve для рекомендательных систем может следовать power law, аналогично LLM. Это стратегически важно: если scaling laws работают, можно предсказывать качество по доступному compute без exhaustive experiments.

### 9.3. Радикальное упрощение feature engineering

GR заменяет сотни гетерогенных feature'ов единым sequential representation'ом действий. При этом, когда DLRM получает тот же ограниченный набор feature'ов, его качество существенно деградирует. Это показывает, что GR не просто использует те же feature'ы лучше, а действительно научился извлекать информацию из последовательной структуры действий.

### 9.4. Инженерные инновации

HSTU (pointwise attention, SiLU gating, raggified attention, Stochastic Length) и M-FALCON (microbatched inference с KV caching) -- это не теоретические предложения, а работающие инженерные решения, которые делают GR практически развертываемым. Факт, что 285x более сложная модель обслуживается с 1.5-3x большим QPS -- впечатляющий инженерный результат.

## 10. Ограничения и критический анализ

### 10.1. Это не работа по semantic tokenization

HSTU не предлагает semantic IDs и не использует text-based item representations. Для линии исследований semantic IDs это скорее **challenge paper**: он заставляет доказать, что semantic compression помогает масштабированию, а не мешает ему. Content-based GR (аналог SID подхода) показывает HR@100 = 11.6% против 35.6% у interaction-based GR.

### 10.2. Невоспроизводимость масштаба

1.5 триллиона параметров, 100B+ training examples, 64-256 H100 GPU -- это ресурсы, доступные единичным компаниям. Для академических лабораторий и большинства индустриальных команд воспроизведение невозможно. Scaling laws убедительны как демонстрация, но проверить их могут лишь немногие.

### 10.3. Production details абстрагированы

Абсолютные значения метрик не раскрыты (только относительные улучшения). Детали serving architecture, candidate selection, multi-stage pipeline integration описаны на высоком уровне. Это типично для industrial papers, но ограничивает понимание, где именно GR интегрируется и какие компоненты legacy system остаются.

### 10.4. Может не помочь каталогам с доминирующей content semantics

HSTU оптимизирован для сценариев с богатыми interaction histories. В доменах с сильным cold-start (новые товары, новый контент), где text/content semantics доминируют, action-native GR может уступать подходам с semantic IDs. Статья явно не исследует cold-start scenarios.

### 10.5. Метрики платформо-специфичны

Online metrics (E-Task, C-Task) -- это внутренние бизнес-метрики Meta, определение которых не раскрыто. Это затрудняет прямое сравнение с результатами других работ.

## 11. Практические выводы

### 11.1. HSTU как benchmark для масштабирования

Для любого исследователя semantic IDs полезно сопоставлять свой подход с HSTU-style baseline: если semantic tokenization не улучшает quality-compute curve по сравнению с action-native GR, то компрессия может быть bottleneck, а не преимуществом.

### 11.2. Actions как модальность

Авторы подчеркивают: "user actions represent an underexplored modality in generative modeling". Это открывает направление foundation models для рекомендаций, где единый feature space объединяет действия из разных доменов (рекомендации, поиск, реклама).

### 11.3. Длина последовательности -- ключевой параметр

В GR длина последовательности играет значительно большую роль, чем в language modeling. Это контрастирует с подходами semantic IDs, которые часто работают с короткими историями (10-50 item'ов). Если длинные истории дают основной прирост, то semantic IDs должны масштабироваться для очень длинных token sequences.

## 12. Связь с соседними работами

HSTU принципиально отличается от semantic ID работ тем, что не пытается сжать item representation в compact code. Но он напрямую связан с ними как counterpoint:

<div class="table-scroll">
<table>
<tr><th>Работа</th><th>Связь с HSTU</th></tr>
<tr><td>TIGER / DSI</td><td>HSTU показывает, что generative retrieval может работать без semantic IDs, на чистых actions</td></tr>
<tr><td>CoST</td><td>CoST улучшает tokenizer objective; HSTU ставит вопрос, нужен ли tokenizer вообще</td></tr>
<tr><td>ETEGRec</td><td>End-to-end alignment tokenizer/recommender; HSTU убирает tokenizer из pipeline</td></tr>
<tr><td>Scaling-view (2509.25522)</td><td>Показывает saturation SID-based GR при масштабировании; HSTU -- один из аргументов за это</td></tr>
<tr><td>MoC/LAMIA</td><td>Пытаются увеличить capacity semantic representation; HSTU решает проблему другим путем</td></tr>
</table>
</div>

## 13. Итоговая оценка

HSTU -- это стратегически важная работа для всего поля generative recommendation. Она устанавливает новую верхнюю границу того, что возможно при масштабировании action-native GR в промышленных условиях. Для исследователей semantic IDs это одновременно вызов и ориентир: semantic tokenization должна доказать, что она предоставляет преимущества (efficiency, cold-start, cross-domain transfer, interpretability), которые оправдывают information loss при компрессии.

Ключевой урок: **не все задачи рекомендаций требуют semantic compression**. Если interaction data достаточно богаты и compute доступен, action-native подход может быть сильнее. Semantic IDs наиболее ценны там, где данных мало (cold-start), каталог меняется быстро, или нужна cross-domain transferability -- именно те сценарии, которые HSTU не исследует.
