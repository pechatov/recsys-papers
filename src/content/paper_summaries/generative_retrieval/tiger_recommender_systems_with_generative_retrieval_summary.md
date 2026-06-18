---
title: "TIGER: Recommender Systems with Generative Retrieval"
category: "generative_retrieval"
slug: "tiger_recommender_systems_with_generative_retrieval_summary"
catalogId: "paper-tiger_recommender_systems_with_generative_retrieval_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/tiger_recommender_systems_with_generative_retrieval_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2305.05065"
---
> **Авторы:** Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Blake Anderson, Ed H. Chi, Maheswaran Sathiamoorthy.
>
> **Аффилиации:** Google DeepMind; University of Wisconsin-Madison.
>
> **Публикация:** NeurIPS 2023 (Advances in Neural Information Processing Systems, том 36). arXiv: 2305.05065.

## 1. Коротко: о чем статья

TIGER (Transformer Index for GEnerative Recommenders) — первая крупная работа, применившая парадигму **generative retrieval с semantic IDs** к задаче рекомендации. Авторы из Google DeepMind и UW-Madison предлагают радикально изменить архитектуру retrieval: вместо dense embedding lookup + ANN-поиска модель напрямую *генерирует* дискретный идентификатор следующего item'а в виде последовательности токенов.

Центральная идея: каждый item кодируется компактным **Semantic ID (SID)** — кортежем из $L$ дискретных кодов, полученным через Residual Quantization VAE (RQ-VAE) из текстового embedding'а item'а. Задача next-item recommendation формулируется как авторегрессионная генерация: дана история пользователя в виде последовательности SID прошлых item'ов — сгенерировать SID следующего item'а токен за токеном. Генератором служит T5-based encoder-decoder Transformer. Валидность сгенерированных последовательностей гарантируется constrained beam search через trie-структуру, построенную по множеству существующих SID.

Работа важна сразу по нескольким причинам. Это **первое применение** generative retrieval (GR) к рекомендации, доказавшее offline quality, сравнимое или превосходящее SASRec, BERT4Rec, P5 и другие сильные baseline'ы. Статья ввела в оборот понятие Semantic ID для recommendation и задала reference pipeline — RQ-VAE токенизация + T5 encoder-decoder + constrained beam search — который затем воспроизводился и модифицировался в десятках последующих работ.

## 2. Контекст и мотивация

### 2.1. Стандартная архитектура retrieve-then-rank

Промышленные рекомендательные системы традиционно работают по схеме **многоэтапной воронки**:

1. **Retrieval / Recall:** из каталога в миллионы item'ов быстро отбираются сотни кандидатов. Обычно используется two-tower model (user tower + item tower) с dot-product similarity, индексированной через приближённый поиск ближайших соседей (ANN): Faiss, ScaNN, HNSW и т.п.
1. **Pre-ranking и Ranking:** кандидаты переоцениваются более сложными моделями.
1. **Re-ranking и Filtering:** бизнес-правила, diversity, freshness.

Retrieval stage — узкое место по нескольким причинам:

- **Suboptimality ANN:** ANN-индекс не является частью дифференцируемой модели. Он оптимизирует скорость поиска, а не quality of recommendation. Граница между embedding learning и ANN-структурой непреодолима градиентным спуском.
- **Семантически бессмысленные ID:** item'ы идентифицируются случайными целочисленными ID, полученными хэшированием или последовательным счётчиком. Модель не может использовать структуру пространства ID: похожие item'ы имеют несвязанные ID.
- **Проблема cold-start:** новый item не имеет истории взаимодействий. Его embedding приходится строить только по content features, а ANN-поиск или коллаборативная фильтрация работают плохо.
- **Разрыв оптимизации:** embedding model и ANN-структура обучаются и обновляются независимо. Даже если embedding модель улучшилась, ANN-индекс нужно пересчитывать отдельно.

### 2.2. Generative retrieval как альтернатива

Generative retrieval (GR) предлагает объединить retrieval и generation: вместо *поиска* по индексу модель *генерирует* идентификатор нужного документа или item'а. Эта идея впервые появилась в NLP-области: DSI (Differentiable Search Index, Tay et al., 2022) генерировал textual docid документа, GENRE генерировал имя entity как строку символов. TIGER переносит эту парадигму в recommender systems, адаптируя её к специфике задачи: каталог содержит не тексты, а разнородные item'ы; item'ы описываются не только текстом, но и collaborative signal; история пользователя — это последовательность взаимодействий, а не query.

Ключевое преимущество GR для рекомендации: **единый end-to-end pipeline** без отдельного ANN-индекса. Генератор напрямую предсказывает следующий item через лингвистику item space, закодированного в дискретных токенах.

### 2.3. Почему нельзя напрямую использовать DSI/GENRE

В NLP-приложениях GR документу назначается identifier либо через хэш его текста, либо через числовую метку. Для рекомендации такие подходы неудобны:

- Каталог item'ов огромный и меняется постоянно (новые item'ы, устаревшие item'ы).
- Item'ы описываются мультимодально (текст, изображение, категории, поведенческие сигналы), и identifier должен отражать семантику по всем этим измерениям.
- Случайные или хэш-based ID не несут структурной информации: генератор не может экстраполировать знание о похожих item'ах.
- Generalization на cold-start item'ы невозможна, если ID непредсказуемо скачет.

TIGER решает эту проблему через **Semantic IDs**: дискретные иерархические идентификаторы, построенные таким образом, что семантически похожие item'ы получают частично совпадающие SID. Это позволяет модели разделять знание о related item'ах.

## 3. Ключевая идея и постановка задачи

### 3.1. Формализация

Пусть $\mathcal{U}$ — множество пользователей, $\mathcal{V}$ — каталог item'ов. Для каждого пользователя $u$ имеется хронологическая история взаимодействий:

$$
S_u = (v_1, v_2, \ldots, v_T)
$$

Каждый item $v \in \mathcal{V}$ кодируется Semantic ID — кортежем из $L$ дискретных токенов:

$$
\text{SID}(v) = (c_1^v, c_2^v, \ldots, c_L^v), \quad c_l^v \in \{1, \ldots, K\}
$$

где $K$ — размер codebook на каждом уровне, $L$ — число уровней RQ-VAE. История пользователя переводится в длинную последовательность токенов путём конкатенации (flattening) всех SID прошлых item'ов:

$$
X_u^{(t)} = \text{SID}(v_1) \oplus \text{SID}(v_2) \oplus \cdots \oplus \text{SID}(v_t)
$$

Задача рекомендации формулируется как **условная авторегрессионная генерация**:

$$
p\bigl(\text{SID}(v_{t+1}) \mid X_u^{(t)}\bigr) = \prod_{l=1}^{L} p\bigl(c_l^{v_{t+1}} \mid X_u^{(t)},\, c_1^{v_{t+1}}, \ldots, c_{l-1}^{v_{t+1}}\bigr)
$$

Генератор учится предсказывать токены SID целевого item'а по одному, используя историю пользователя как контекст и ранее сгенерированные токены SID как decoder prefix. На inference beam search с trie-ограничением гарантирует, что сгенерированный кортеж $(c_1, \ldots, c_L)$ соответствует хотя бы одному существующему item'у в каталоге.

### 3.2. Почему seq2seq — правильная абстракция

Формулировка рекомендации как seq2seq задачи нетривиальна. Альтернативы:

- **Autoregressive LM над item ID:** предсказывать следующий atomic item ID из vocabulary $|\mathcal{V}|$. Проблема: при больших каталогах vocabulary огромный, каждый ID изолирован, нет семантической структуры.
- **Dense retrieval:** предсказывать вектор, затем ANN. Нет единого дифференцируемого pipeline.
- **Seq2seq с текстовым title item'а:** генерировать заголовок текстом. Проблема: один заголовок может соответствовать нескольким item'ам; большой output vocabulary; медленная генерация.

Semantic ID как short tuple кодов — компромисс: короткая (L=4 токена) дискретная последовательность с семантической структурой, полученной из content embedding'ов. Это и есть главная архитектурная ставка TIGER.

## 4. Построение Semantic IDs через RQ-VAE

### 4.1. Откуда берётся content embedding item'а

Для каждого item из текстовых полей (название, описание, категория, brand) формируется один текстовый prompt:

```
Item title: <title>. Brand: <brand>. Category: <category>. Description: <description>.
```

Этот текст кодируется замороженной моделью **SentenceT5-large** (Ni et al., 2022), которая возвращает вектор размерности 768. Это фиксированный content embedding item'а, обозначим его $x_v \in \mathbb{R}^{768}$. SentenceT5 выбрана за качество sentence-level семантических представлений; авторы подчёркивают, что модель заморожена и не дообучается на данных рекомендаций.

Важное следствие: SID строится исключительно по контентным признакам, *без* коллаборативного сигнала. Это делает TIGER принципиально отличным от методов, где ID строятся по матрице user-item взаимодействий.

### 4.2. Архитектура RQ-VAE

<figure class="paper-figure">
  <img src="../../assets/tiger/tokenization_pipeline.png" alt="TIGER content embedding to semantic ID tokenization pipeline">
  <figcaption>Рисунок 1. TIGER строит SID из content information: item text/category/brand кодируются content encoder'ом, затем residual quantization переводит embedding в sequence semantic codes.</figcaption>
</figure>

Residual Quantization VAE (основанный на Lee et al., CVPR 2022) состоит из трёх компонентов:

- **Encoder $E$:** MLP, проецирующий $x_v \in \mathbb{R}^{768}$ в latent vector $z = E(x_v) \in \mathbb{R}^d$. В экспериментах $d = 64$.
- **Residual Quantizer:** набор $L$ codebook'ов $\mathcal{C}_1, \ldots, \mathcal{C}_L$, где $\mathcal{C}_l = \{e_k^l\}_{k=1}^{K} \subset \mathbb{R}^d$. Каждый codebook содержит $K$ векторов (code vectors).
- **Decoder $D$:** MLP, восстанавливающий $\hat{x}_v \in \mathbb{R}^{768}$ из суммы выбранных code vectors.

### 4.3. Алгоритм Residual Quantization

Получив latent $z = E(x_v)$, RQ-VAE проходит $L$ уровней:

На уровне $l = 1$: начальный residual $r_0 = z$. Находим ближайший code vector в первом codebook:

$$
c_1 = \arg\min_{k \in \{1,\ldots,K\}} \lVert r_0 - e_k^1 \rVert_2
$$

Вычисляем остаток:

$$
r_1 = r_0 - e_{c_1}^1
$$

На уровне $l \ge 2$: квантуем текущий residual $r_{l-1}$ через codebook $\mathcal{C}_l$:

$$
c_l = \arg\min_{k \in \{1,\ldots,K\}} \lVert r_{l-1} - e_k^l \rVert_2
$$

$$
r_l = r_{l-1} - e_{c_l}^l
$$

После всех $L$ уровней item получает Semantic ID:

$$
\text{SID}(v) = (c_1, c_2, \ldots, c_L)
$$

Квантованное представление:

$$
\hat{z} = \sum_{l=1}^{L} e_{c_l}^l
$$

Decoder восстанавливает content embedding: $\hat{x}_v = D(\hat{z})$.

Семантическая интерпретация иерархии: $c_1$ кодирует наиболее грубую семантическую категорию (крупный кластер item'ов), $c_2$ — более тонкое разбиение внутри кластера, и т.д. Item'ы с одинаковым $c_1$ семантически похожи; item'ы с одинаковыми $(c_1, c_2)$ ещё более похожи. Это **иерархическое покрытие item space** — ключевое свойство, отличающее SID от случайного хэша.

<figure class="paper-figure">
  <img src="../../assets/tiger/semantic_hierarchy_level1.png" alt="TIGER semantic hierarchy grouped by first SID token">
  <figcaption>Рисунок 2. Пример интерпретации первого SID token: крупные группы соответствуют широким товарным категориям, что объясняет transfer между похожими item'ами.</figcaption>
</figure>

<figure class="paper-figure">
  <img src="../../assets/tiger/semantic_hierarchy_prefixes.png" alt="TIGER semantic hierarchy grouped by SID prefixes">
  <figcaption>Рисунок 3. Более длинные SID prefixes уточняют category grouping. Это показывает, почему SID ведет себя как hierarchical identifier, а не как случайный item hash.</figcaption>
</figure>

### 4.4. Функция потерь RQ-VAE

RQ-VAE обучается минимизировать три слагаемых:

$$
\mathcal{L}_{\text{RQ-VAE}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{vq}} + \beta\,\mathcal{L}_{\text{commit}}
$$

Reconstruction loss (MSE между исходным и восстановленным embedding'ом):

$$
\mathcal{L}_{\text{recon}} = \lVert x_v - \hat{x}_v \rVert_2^2
$$

VQ loss (обучает code vectors двигаться к encoder output; stop-gradient на encoder'е, чтобы только codebook обновлялся):

$$
\mathcal{L}_{\text{vq}} = \sum_{l=1}^{L} \bigl\lVert \operatorname{sg}(r_{l-1}) - e_{c_l}^l \bigr\rVert_2^2
$$

Commitment loss (заставляет encoder output не убегать от выбранного code vector; stop-gradient на codebook, чтобы только encoder обновлялся):

$$
\mathcal{L}_{\text{commit}} = \sum_{l=1}^{L} \bigl\lVert r_{l-1} - \operatorname{sg}(e_{c_l}^l) \bigr\rVert_2^2
$$

где $\operatorname{sg}(\cdot)$ — stop-gradient оператор. Hyperparameter $\beta$ обычно задаётся равным 0.25. Для предотвращения codebook collapse code vectors инициализируются k-means центроидами из первого прохода по данным.

После обучения RQ-VAE параметры замораживаются: tokenizer не переобучается совместно с recommendation generator. Это двухстадийное разделение — одно из ключевых ограничений TIGER, устранённых в последующих работах (DIGER, ETEGRec).

### 4.5. Теоретическая ёмкость SID space

При $L$ уровнях и codebook size $K$ максимальное число уникальных SID равно $K^L$. В основной конфигурации TIGER: $L = 4$, $K = 256$, что даёт:

$$
256^4 = 4{,}294{,}967{,}296 \approx 4.3 \times 10^9
$$

Этого достаточно для каталогов в миллионы item'ов с большим запасом. Однако реальная утилизация SID space ограничена: большинство кодов используются редко, что создаёт риск коллизий в плотных семантических областях.

## 5. Generative Recommender на основе T5

### 5.1. Общая архитектура

В качестве генератора TIGER использует **T5-small** или **T5-base** — encoder-decoder Transformer из семейства T5 (Raffel et al., 2020). Это стандартная seq2seq архитектура с bidirectional encoder и causal decoder.

Ключевое переиспользование архитектуры T5 из NLP: авторы показывают, что text-to-text фреймворк хорошо обобщается на задачу рекомендации, если правильно выбрать vocabulary. Вместо BPE/WordPiece токенов текста vocabulary состоит из дискретных кодов SID.

### 5.2. Кодирование истории пользователя

История пользователя $S_u^{(t)} = (v_1, \ldots, v_t)$ переводится в последовательность токенов путём конкатенации SID всех item'ов:

$$
X_u^{(t)} = c_1^{v_1}, c_2^{v_1}, \ldots, c_L^{v_1},\; c_1^{v_2}, \ldots, c_L^{v_2},\; \ldots,\; c_1^{v_t}, \ldots, c_L^{v_t}
$$

При $L = 4$ история из 50 item'ов даёт последовательность из $4 \times 50 = 200$ токенов. Это умещается в стандартный контекст T5-small (до 512 токенов). Для более длинных историй авторы обрезают историю, оставляя последние $N_{\max}$ item'ов.

Важная деталь: порядок item'ов в истории сохраняется. Encoder T5 обрабатывает flattened последовательность с position embeddings, поэтому позиционная информация о том, какой item более ранний, а какой более поздний, неявно закодирована. Temporal ordering — ключевой сигнал для sequential recommendation.

### 5.3. Обучение: cross-entropy loss по токенам SID

Encoder обрабатывает историю $X_u^{(t)}$. Decoder авторегрессионно генерирует SID целевого item'а $v_{t+1}$. Training objective — cross-entropy loss по каждому токену SID:

$$
\mathcal{L}_{\text{gen}} = -\sum_{u} \sum_{t} \sum_{l=1}^{L} \log p_\theta\bigl(c_l^{v_{t+1}} \mid X_u^{(t)},\, c_1^{v_{t+1}}, \ldots, c_{l-1}^{v_{t+1}}\bigr)
$$

Все $L$ токенов SID предсказываются совместно в одном forward pass: decoder получает $c_1, \ldots, c_{l-1}$ как prefix и предсказывает $c_l$. Это стандартный teacher forcing в режиме seq2seq.

Vocabulary генератора состоит из кодов всех codebook'ов. В конфигурации $K=256, L=4$ это $4 \times 256 = 1024$ токена (в предположении раздельных codebook'ов на каждом уровне). Это *намного* меньше, чем vocabulary в item-level autoregressive модели с $|\mathcal{V}|$ в миллионы.

### 5.4. Constrained Beam Search через Trie

Без ограничений beam search может сгенерировать SID-кортеж, не соответствующий ни одному реальному item'у. Например, при $K=256, L=4$ пространство из $256^4 \approx 4.3 \times 10^9$ кортежей намного больше, чем каталог. Большинство кортежей — "несуществующие item'ы".

TIGER решает это через **prefix trie (бор) по всем SID каталога**. Trie — дерево, в котором каждый путь от корня до листа соответствует SID одного или нескольких item'ов. На каждом шаге декодирования beam search ограничивает следующий токен теми значениями, которые продолжают хотя бы один валидный SID-путь в trie.

Формально: пусть в beam на шаге $l$ имеется partial sequence $(c_1, \ldots, c_{l-1})$. Допустимое множество следующих токенов:

$$
\mathcal{A}(c_1, \ldots, c_{l-1}) = \bigl\{k : \exists v \in \mathcal{V},\; \text{SID}(v) \text{ начинается с } (c_1, \ldots, c_{l-1}, k)\bigr\}
$$

Вероятности токенов вне $\mathcal{A}$ обнуляются (или задаются $-\infty$ в log-space). Это гарантирует, что любая полная последовательность из beam соответствует существующему item'у. Constrained decoding — элегантное решение: не требует постпроцессинга или lookup failure handling.

При размере beam $B = 20$ на inference возвращается до 20 валидных кандидатов, которые ранжируются по log-probability декодера.

## 6. Полный TIGER Pipeline: взгляд на архитектуру

### 6.1. Этап 1: Tokenization (offline, один раз)

Этот этап выполняется один раз и заморожен на время обучения recommender'а:

1. Для каждого item $v \in \mathcal{V}$ формируется текстовый prompt из полей каталога.
1. SentenceT5-large кодирует prompt в вектор $x_v \in \mathbb{R}^{768}$.
1. Обученный RQ-VAE (encoder $E$ + $L$ codebook'ов) преобразует $x_v$ в SID $(c_1, c_2, c_3, c_4)$.
1. Строится таблица $\text{SID} \to \text{item\_id}$ и trie по всем SID.

### 6.2. Этап 2: Generator Training

1. Истории пользователей переводятся в SID-последовательности.
1. T5-based encoder-decoder обучается предсказывать SID следующего item'а.
1. Loss — cross-entropy по токенам; оптимизируется через standard AdamW.

### 6.3. Inference

1. История пользователя: flattened SID-последовательность → T5 encoder.
1. T5 decoder выполняет constrained beam search (beam size $B=20$) через trie.
1. Возвращаются top-$B$ SID-кортежей с наивысшей log-probability.
1. Через таблицу $\text{SID} \to \text{item\_id}$ SID-кортежи переводятся в item ID.
1. Если несколько item'ов имеют одинаковый SID (коллизия), они возвращаются все с одинаковым score.

## 7. Детали реализации и экспериментальная Setup

### 7.1. Датасеты

<div class="table-scroll">
<table>
<thead>
<tr><th>Dataset</th><th>#Users</th><th>#Items</th><th>#Interactions</th><th>Avg. seq len</th><th>Домен</th></tr>
</thead>
<tbody>
<tr><td>Amazon Beauty</td><td>22,363</td><td>12,101</td><td>198,502</td><td>8.9</td><td>Косметика</td></tr>
<tr><td>Amazon Sports</td><td>35,598</td><td>18,357</td><td>296,337</td><td>8.3</td><td>Спорт</td></tr>
<tr><td>Amazon Toys</td><td>19,412</td><td>11,924</td><td>167,597</td><td>8.6</td><td>Игрушки</td></tr>
<tr><td>Amazon Books</td><td>52,643</td><td>91,599</td><td>927,694</td><td>17.6</td><td>Книги</td></tr>
<tr><td>Steam</td><td>334,729</td><td>13,047</td><td>3,461,770</td><td>10.3</td><td>Видеоигры</td></tr>
</tbody>
</table>
</div>

Amazon данные — стандартные 5-core splits, где каждый user и item имеют не менее 5 взаимодействий. Books — более крупный и разреженный датасет. Steam — публичный датасет Steam игровой платформы. Preprocessing: хронологическая сортировка взаимодействий; leave-one-out split: последний item — тест, предпоследний — валидация, остальные — обучение.

### 7.2. Гиперпараметры RQ-VAE

- Content embedding: SentenceT5-large, размерность 768, заморожен.
- Encoder $E$: MLP с hidden layers $[512, 256, 128, 64]$, активация ReLU.
- Latent dimension: $d = 64$.
- Уровни residual quantization: $L = 4$.
- Размер codebook: $K = 256$ кодов на уровень.
- Commitment coefficient: $\beta = 0.25$.
- Инициализация codebook: k-means по item embedding'ам.
- Обучение RQ-VAE: 10,000 итераций, batch size 256, Adam.

### 7.3. Гиперпараметры T5 генератора

- Базовая архитектура: T5-small (77M параметров) или T5-base (250M).
- T5-small: 6 encoder layers, 6 decoder layers, hidden size 512, 8 heads.
- Input sequence: история последних $N_{\max} = 50$ item'ов, итого до $50 \times 4 = 200$ токенов.
- Target sequence: SID следующего item'а, $L = 4$ токена.
- Обучение: AdamW, learning rate $5 \times 10^{-4}$, batch size 256, до 200K шагов.
- Inference: constrained beam search, beam size $B = 20$.

### 7.4. Метрики оценки

Стандартный leave-one-out protocol для sequential recommendation:

- **Recall@K** : доля пользователей, у которых true next item входит в top-K рекомендаций.
- **NDCG@K** (Normalized Discounted Cumulative Gain): учитывает позицию true next item в ranked списке.
- Оцениваются при $K \in \{5, 10\}$.
- Ранжирование производится против всего каталога item'ов (full ranking, без negative sampling).

## 8. Результаты экспериментов

### 8.1. Сравнение с baseline методами

TIGER сравнивается с несколькими категориями baseline'ов:

- **Collaborative filtering:** BPR-MF, LightGCN.
- **Sequential models:** GRU4Rec, SASRec, BERT4Rec.
- **Content-based / text models:** ZESRec, FDSA, UniSRec.
- **Generative / LLM-based:** P5 (recommendation as text-to-text NLP).

Результаты на Amazon Beauty (Recall@10 и NDCG@10):

<div class="table-scroll">
<table>
<thead>
<tr><th>Метод</th><th>Тип</th><th>Recall@5</th><th>Recall@10</th><th>NDCG@5</th><th>NDCG@10</th></tr>
</thead>
<tbody>
<tr><td>BPR-MF</td><td>CF</td><td>0.0394</td><td>0.0619</td><td>0.0261</td><td>0.0335</td></tr>
<tr><td>SASRec</td><td>Sequential</td><td>0.0752</td><td>0.1060</td><td>0.0528</td><td>0.0631</td></tr>
<tr><td>BERT4Rec</td><td>Sequential</td><td>0.0637</td><td>0.0944</td><td>0.0437</td><td>0.0537</td></tr>
<tr><td>ZESRec</td><td>Content-based</td><td>0.0413</td><td>0.0637</td><td>0.0281</td><td>0.0355</td></tr>
<tr><td>UniSRec</td><td>Content-based</td><td>0.0525</td><td>0.0808</td><td>0.0360</td><td>0.0449</td></tr>
<tr><td>P5</td><td>Generative NLP</td><td>0.0390</td><td>0.0588</td><td>0.0262</td><td>0.0325</td></tr>
<tr><td><strong>TIGER (T5-small)</strong></td><td><strong>GR + SID</strong></td><td><strong>0.0818</strong></td><td><strong>0.1126</strong></td><td><strong>0.0587</strong></td><td><strong>0.0688</strong></td></tr>
</tbody>
</table>
</div>

На Beauty TIGER превосходит SASRec по Recall@10 (0.1126 vs 0.1060), что особенно примечательно, учитывая, что SASRec использует collaborative signal, а TIGER — только content features. Это один из ключевых результатов статьи.

### 8.2. Результаты на Amazon Sports

<div class="table-scroll">
<table>
<thead>
<tr><th>Метод</th><th>Recall@5</th><th>Recall@10</th><th>NDCG@5</th><th>NDCG@10</th></tr>
</thead>
<tbody>
<tr><td>SASRec</td><td>0.0470</td><td>0.0702</td><td>0.0322</td><td>0.0399</td></tr>
<tr><td>BERT4Rec</td><td>0.0378</td><td>0.0590</td><td>0.0250</td><td>0.0320</td></tr>
<tr><td>UniSRec</td><td>0.0361</td><td>0.0556</td><td>0.0245</td><td>0.0308</td></tr>
<tr><td>P5</td><td>0.0261</td><td>0.0420</td><td>0.0175</td><td>0.0227</td></tr>
<tr><td><strong>TIGER (T5-small)</strong></td><td><strong>0.0501</strong></td><td><strong>0.0730</strong></td><td><strong>0.0352</strong></td><td><strong>0.0429</strong></td></tr>
</tbody>
</table>
</div>

### 8.3. Результаты на Amazon Toys и Books

На Toys TIGER показывает Recall@10 ≈ 0.092, превосходя SASRec (≈ 0.082) и BERT4Rec (≈ 0.071). На Books, как самом крупном датасете, разрыв между методами меньше: TIGER достигает Recall@10 ≈ 0.052, тогда как SASRec ≈ 0.048.

Общий паттерн: **TIGER стабильно превосходит все content-based baseline'ы** (ZESRec, UniSRec, P5) на всех датасетах. Относительно SASRec TIGER либо превосходит (Beauty, Sports, Toys), либо показывает сравнимые результаты (Books). Это неожиданный результат, учитывая, что TIGER не использует collaborative signal при построении SID.

### 8.4. Результаты на Steam

Steam — наибольший датасет по числу пользователей (334K). На нём TIGER с T5-base достигает Recall@10 ≈ 0.118, что превосходит SASRec (≈ 0.101) и BERT4Rec (≈ 0.093). Большой датасет позволяет T5-base лучше использовать свою ёмкость.

### 8.5. T5-small vs T5-base

На крупных датасетах (Books, Steam) T5-base заметно лучше T5-small: Recall@10 растёт на 5–8%. На небольших датасетах (Beauty, Sports) разница менее значительна. Авторы отмечают, что generative retrieval масштабируется с размером модели.

## 9. Ablation Study

### 9.1. Semantic ID vs Random ID vs Textual ID

Ключевой ablation: что происходит, если заменить SID на другой тип идентификатора?

<div class="table-scroll">
<table>
<thead>
<tr><th>Тип ID</th><th>Описание</th><th>Recall@10 (Beauty)</th><th>NDCG@10 (Beauty)</th></tr>
</thead>
<tbody>
<tr><td><strong>Semantic ID (RQ-VAE)</strong></td><td>4 токена через RQ-VAE</td><td><strong>0.1126</strong></td><td><strong>0.0688</strong></td></tr>
<tr><td>Random ID</td><td>4 случайных токена из [0, K)</td><td>0.0341</td><td>0.0202</td></tr>
<tr><td>Textual Title ID</td><td>BPE токены названия item'а</td><td>0.0782</td><td>0.0478</td></tr>
<tr><td>Single Index ID</td><td>Один токен — порядковый номер item'а</td><td>0.0419</td><td>0.0248</td></tr>
</tbody>
</table>
</div>

Выводы:

- **Random ID катастрофически хуже SID** : Recall@10 = 0.0341 против 0.1126. Семантическая структура ID критически важна: без неё модель не может переносить знание между похожими item'ами.
- **Textual ID хуже SID** : несмотря на то что текстовые токены несут семантику, они длиннее (больше токенов на item), нестабильны (незначительные изменения названия меняют tokenization), и модель хуже обобщается. Кроме того, textual tokens не имеют иерархической структуры, аналогичной RQ-VAE.
- **Single Index ID хуже SID** : одного токена недостаточно для выражения иерархической семантики item'а.

### 9.2. RQ-VAE Tokenizer vs Альтернативные методы токенизации

<div class="table-scroll">
<table>
<thead>
<tr><th>Метод токенизации</th><th>Recall@10</th><th>NDCG@10</th></tr>
</thead>
<tbody>
<tr><td><strong>RQ-VAE (4 уровня)</strong></td><td><strong>0.1126</strong></td><td><strong>0.0688</strong></td></tr>
<tr><td>Hierarchical K-means (depth 4)</td><td>0.0981</td><td>0.0601</td></tr>
<tr><td>PQ (Product Quantization)</td><td>0.0876</td><td>0.0538</td></tr>
<tr><td>Direct text tokens (SentenceT5 + k-means)</td><td>0.0812</td><td>0.0497</td></tr>
</tbody>
</table>
</div>

RQ-VAE превосходит все альтернативы. Hierarchical k-means — ближайший конкурент: он тоже строит иерархическую кластеризацию, но RQ-VAE обеспечивает более компактную и информативную кодировку благодаря residual: каждый следующий уровень кодирует именно ту информацию, которую предыдущие уровни не смогли закодировать.

### 9.3. Влияние длины SID (число уровней L)

<div class="table-scroll">
<table>
<thead>
<tr><th>L (число уровней)</th><th>Recall@5</th><th>Recall@10</th><th>NDCG@10</th><th>Теоретич. ёмкость</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>0.0342</td><td>0.0510</td><td>0.0308</td><td>256</td></tr>
<tr><td>2</td><td>0.0621</td><td>0.0878</td><td>0.0534</td><td>65,536</td></tr>
<tr><td>3</td><td>0.0774</td><td>0.1055</td><td>0.0643</td><td>16,777,216</td></tr>
<tr><td><strong>4</strong></td><td><strong>0.0818</strong></td><td><strong>0.1126</strong></td><td><strong>0.0688</strong></td><td>4,294,967,296</td></tr>
<tr><td>5</td><td>0.0809</td><td>0.1112</td><td>0.0677</td><td>~10^12</td></tr>
</tbody>
</table>
</div>

Результаты: **Качество растёт с L и насыщается при L=4**. При L=1 модель имеет только 256 возможных SID — грубая кластеризация с большим числом коллизий. При L=4 ёмкость SID пространства намного превышает размер каталога, и коллизий мало. При L=5 качество почти не растёт, а длина input sequence увеличивается — это создаёт нежелательную нагрузку на encoder.

### 9.4. Влияние размера codebook K

<div class="table-scroll">
<table>
<thead>
<tr><th>K</th><th>Recall@10</th><th>NDCG@10</th></tr>
</thead>
<tbody>
<tr><td>64</td><td>0.1041</td><td>0.0635</td></tr>
<tr><td>128</td><td>0.1088</td><td>0.0661</td></tr>
<tr><td><strong>256</strong></td><td><strong>0.1126</strong></td><td><strong>0.0688</strong></td></tr>
<tr><td>512</td><td>0.1119</td><td>0.0682</td></tr>
<tr><td>1024</td><td>0.1105</td><td>0.0673</td></tr>
</tbody>
</table>
</div>

Качество растёт до K=256, затем немного снижается. При K=64 codebook capacity недостаточна: много коллизий, модель теряет различимость item'ов. При K=1024 появляются проблемы с codebook utilization (не все 1024 кода используются равномерно), а обучение constrained beam search усложняется.

### 9.5. Анализ cold-start обобщения

TIGER специально проверяется на **cold-start item'ах** — item'ах, отсутствующих в обучающей выборке. Эксперимент: из тестовой выборки выбираются пары (user, next_item), где next_item встречается в тесте первый раз. Сравнение:

<div class="table-scroll">
<table>
<thead>
<tr><th>Метод</th><th>Cold-start Recall@10</th><th>Warm Recall@10</th><th>Соотношение cold/warm</th></tr>
</thead>
<tbody>
<tr><td>SASRec</td><td>0.0312</td><td>0.1089</td><td>0.287</td></tr>
<tr><td>TIGER (Random ID)</td><td>0.0098</td><td>0.0352</td><td>0.278</td></tr>
<tr><td><strong>TIGER (Semantic ID)</strong></td><td><strong>0.0641</strong></td><td><strong>0.1148</strong></td><td><strong>0.558</strong></td></tr>
</tbody>
</table>
</div>

Ключевой результат: TIGER с Semantic ID сохраняет 55.8% качества на cold-start item'ах от warm-quality, тогда как SASRec теряет около 71% (сохраняет 28.7%). Это прямое следствие semantic sharing: новый item, схожий со старыми по content, получает похожий SID, и модель может генерировать его, используя обобщённые знания о семантическом регионе. TIGER с Random ID не даёт никакого преимущества — что подтверждает, что improvement целиком обеспечивается semantic structure SID, а не самой generative парадигмой.

## 10. Сильные стороны TIGER

### 10.1. Единый дифференцируемый pipeline без ANN-индекса

TIGER устраняет ANN-поиск как недифференцируемый внешний компонент. Весь retrieval реализован через параметры T5 генератора. Это теоретически открывает возможность сквозной оптимизации всей системы (хотя на практике tokenizer всё равно обучается отдельно).

### 10.2. Semantic sharing между похожими item'ами

Благодаря иерархической структуре SID модель разделяет параметры между item'ами с общим SID-префиксом. Токены c1, c2 отвечают за coarse семантические регионы; fine-grained различие — в c3, c4. Это в точности то, как language model разделяет знание между похожими словами.

### 10.3. Естественная поддержка cold-start через content embedding

Новый item немедленно получает SID через RQ-VAE из своих content features, без необходимости ждать накопления пользовательских взаимодействий. Если content item'а похож на существующие item'ы — он получает близкий SID и сразу может быть рекомендован генератором. Это принципиальное преимущество перед collaborative filtering методами.

### 11.4. Constrained decoding гарантирует валидность вывода

В отличие от dense retrieval, где ANN может вернуть вектор, не соответствующий реальному item'у, TIGER через trie гарантирует: каждый кандидат — существующий item. Нет post-hoc фильтрации "несуществующих" результатов.

### 11.5. Первое доказательство offline quality GR в рекомендации

До TIGER не было работы, убедительно показавшей, что generative retrieval конкурентоспособен в recommendation benchmarks. TIGER устанавливает baseline и опровергает скептицизм о том, что GR может быть полезен только в NLP-задачах.

### 11.6. Компактный vocabulary для генерации

При K=256, L=4 vocabulary из 1024 токенов (или $4 \times 256$) несопоставимо меньше, чем при генерации item ID напрямую (требует vocabulary $|\mathcal{V}|$ токенов). Это ускоряет обучение и делает generalization лучше: каждый code token встречается часто, что улучшает его embedding качество.

## 12. Ограничения TIGER

### 12.1. Двухстадийное обучение: objective mismatch

RQ-VAE оптимизируется под *reconstruction* content embedding'а, тогда как downstream генератор оптимизируется под *recommendation ranking*. Эти цели не совпадают. SID, хорошие с точки зрения реконструкции, могут быть неудобными для autoregressive generation. Этот mismatch формально доказан в работе DIGER (Fu et al., 2025): двухстадийное обучение оптимизирует в restricted subspace параметров tokenizer'а, и gap от joint оптимизации может быть произвольно большим.

### 12.2. SID коллизии при семантически близких item'ах

Если два item'а имеют очень похожие content embedding'и, RQ-VAE может назначить им одинаковый SID. В этом случае beam search возвращает оба item'а, но не может их различить по SID. Это особенно проблематично в плотных семантических областях (например, несколько версий одной книги). Collision rate напрямую ограничивает верхнюю границу Recall.

### 12.3. Inference дороже dense retrieval

ANN-поиск (Faiss, ScaNN) выполняется за единицы миллисекунд. Constrained beam search с T5-decoder — O(B × L) forward pass'ов через transformer, что занимает десятки-сотни миллисекунд. Для промышленных систем с требованием latency < 100ms это существенное ограничение. Параллельный beam search немного помогает, но масштабирование всё равно хуже ANN.

### 12.4. Обновление каталога требует переиндексации и потенциально переобучения

При добавлении новых item'ов в каталог:

- Новый item кодируется через RQ-VAE (быстро, не требует переобучения).
- Trie обновляется с новым SID (быстро).
- Но если новый item получает SID, который генератор "не видел" при обучении, quality для этого item'а снижена. Генератор нужно дообучить.
- При массовом обновлении каталога (например, при ежедневном добавлении тысяч новых item'ов) нужно регулярное переобучение generator'а.

### 12.5. Не тестировался в production

TIGER — исследовательская работа с offline evaluation. В production неизбежны дополнительные сложности: latency requirements, streaming updates каталога, A/B тестирование, business constraints, diversity requirements. Авторы честно это отмечают, но production deployment оставлен для будущих работ (первые шаги — в "Deploying Semantic ID-based GR for Large-Scale Recommendation", Google 2024).

### 12.6. Небольшие датасеты для нейронных систем

Amazon Beauty (22K users, 12K items) — относительно небольшой датасет. На нём SASRec и TIGER примерно эквивалентны. Были бы ли результаты такими же на датасете с 100M пользователями и 10M item'ами — не проверено.

### 12.7. Только текстовые content features

SentenceT5 кодирует только текст item'а. Изображения, ценовые сигналы, inventory status, freshness — не используются. Современные item embedding'и часто мультимодальны; TIGER архитектурно это не поддерживает.

### 12.8. Отсутствие collaborative signal в токенизации

SID строится только по content features, без учёта, с какими другими item'ами данный item покупается вместе, какие пользователи его смотрят, каковы его co-click паттерны. Это означает, что два item'а с похожим текстом, но противоположными аудиториями (разные возрастные группы, разные интересы), получат близкий SID — что может приводить к неоптимальным рекомендациям.

## 13. Связь с соседними работами

### 13.1. TIGER как основа последующих исследований

TIGER задал reference pipeline для целого направления исследований. Большинство последующих работ модифицируют один или несколько компонентов TIGER:

- **Tokenizer:** CoST (contrastive quantization), LETTER (learnable tokenization с collaborative signals), DIGER (differentiable SID), ETEGRec (end-to-end обучение), DAS (dual-aligned SID).
- **Generator:** GRACE (sparse attention для длинных историй), GLASS (long sequence modeling), STORE (streamlined pipeline).
- **Constrained decoding:** TrieRec (trie-aware transformers), работы по efficient constrained decoding.
- **Cold-start:** Better Generalization with Semantic IDs (Google, 2024 — production validation).

### 13.2. TIGER vs Actions Speak Louder Than Words (HSTU)

Конкурирующая парадигма от Meta (Zhai et al., 2024): вместо дискретных SID и generative approach — единая авторегрессионная модель (HSTU — Hierarchical Sequential Transduction Unit) над атомарными item ID. HSTU работает на production-scale (1.5T параметров, 10T токенов обучения). Эти работы представляют два принципиально разных взгляда: TIGER — "items as token sequences through semantic encoding", HSTU — "items as atomic tokens learned end-to-end from massive collaborative data".

## 14. Что читать после TIGER

### 14.1. Непосредственные продолжения и улучшения

1. **[LETTER: Learnable Item Tokenization for Generative Recommendation](https://arxiv.org/abs/2405.07314)** — коллаборативная регуляризация токенайзера: SID строится с учётом collaborative signals через SASRec-derived embeddings, не только из content features. Первое чёткое улучшение tokenization phase TIGER.
1. **[CoST: Contrastive Quantization based Semantic Tokenization](https://arxiv.org/abs/2404.14774)** — замена MSE-реконструкции в RQ-VAE на contrastive objective. Ключевой аргумент: retrieval нужна discrimination между item'ами, а не точная реконструкция embedding'а. На MIND +43% Recall@5 vs RQ-VAE.
1. **[ETEGRec: End-to-End Learnable Item Tokenization for Generative Recommendation](https://arxiv.org/abs/2406.04876)** — first attempt to train tokenizer и generator jointly через alternating optimization с distillation. Устраняет objective mismatch TIGER без катастрофического collapse (который происходит при наивном joint training).
1. **[DIGER: Differentiable Semantic ID for Generative Recommendation](https://arxiv.org/abs/2601.19711)** — теоретическое обоснование suboptimality двухстадийного обучения + решение через Gumbel noise и uncertainty decay. Доказывает, что STE приводит к codebook collapse, и предлагает DRIL.
1. **[Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations](https://arxiv.org/abs/2306.08121)** — Google production study, показывающий, что semantic IDs улучшают generalization в ranking модели на production scale. Валидирует ключевую идею TIGER в реальной системе.

### 14.2. Конкурирующие парадигмы

1. **[Actions Speak Louder Than Words: Trillion-Parameter Sequential Transducers for Generative Recommendation](https://arxiv.org/abs/2402.17152)** (Meta, NeurIPS 2024) — HSTU: атомарные item ID, масштабируемый autoregressive LM, production deployment. Контрапункт к TIGER: collaborative data в триллионах взаимодействий может заменить необходимость в semantic IDs.

### 14.3. Технические базы

1. **[Transformer Memory as a Differentiable Search Index (DSI)](https://arxiv.org/abs/2202.06991)** — пионерская работа GR для NLP retrieval, откуда TIGER черпает вдохновение. Генерация docid вместо scoring.
1. **[Autoregressive Image Generation using Residual Quantization (RQ-VAE)](https://arxiv.org/abs/2203.01941)** — источник RQ-VAE архитектуры. Необходимо для понимания residual quantization mechanics.
1. **[Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)** — базовая работа по vector quantization и straight-through estimator.
1. **[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](https://arxiv.org/abs/1910.10683)** — архитектура генератора TIGER.

### 14.4. Направления по конкретным ограничениям TIGER

- Если интересует устранение objective mismatch → DIGER, ETEGRec.
- Если интересует better tokenizer → CoST, LETTER, DAS, ASI.
- Если интересует cold-start → Better Generalization (Google), TIGER ablation раздел 9.5.
- Если интересует production deployment → Deploying Semantic ID-based GR (Google, 2024).
- Если интересует masштабирование → HSTU (Actions Speak Louder).

## 15. Практические выводы для адаптации TIGER

### 15.1. Что нужно реализовать

При воспроизведении или адаптации TIGER следующий checklist:

1. **Text embedding pipeline:** SentenceT5-large (или заменить на подходящую мультиязычную / доменную модель). Зафиксировать формат текстового prompt'а для каждого item. Сохранить embeddings — они используются только для RQ-VAE обучения.
1. **RQ-VAE:** Реализовать MLP encoder, $L$ codebook'ов размером $K$, decoder. Использовать k-means инициализацию codebook'ов. Контролировать balance кодов — dead codes (неиспользуемые коды) = проблема.
1. **SID assignment и trie:** После обучения RQ-VAE назначить SID каждому item'у. Построить trie для constrained decoding. Реализовать обработку коллизий (несколько item'ов с одинаковым SID).
1. **T5 Generator:** Адаптировать T5 vocabulary под SID токены. Flattening истории в правильном порядке. Правильный padding и attention masking.
1. **Constrained beam search:** Интегрировать trie с beam search в HuggingFace / custom decoder. Проверить, что все $B$ beam кандидатов valid.

### 15.2. Ключевые метрики для мониторинга

- **Collision rate:** доля item'ов, имеющих неуникальный SID. Если > 5% — нужно увеличить $L$ или $K$. Считается: $\text{collision\_rate} = 1 - |\text{unique SIDs}| / |\mathcal{V}|$.
- **Codebook utilization:** entropy распределения кодов на каждом уровне. Идеально — uniform: $H = \log_2 K$. Если entropy низкая — codebook collapse, и нужно улучшить инициализацию или добавить diversity loss.
- **Beam search validity rate:** доля queries, для которых beam search возвращает ровно $B$ валидных кандидатов. Если меньше — trie слишком ограничителен или модель генерирует несуществующие prefixes чаще, чем ожидалось.
- **Reconstruction quality (RQ-VAE):** cosine similarity между $x_v$ и $\hat{x}_v$. Должна быть > 0.95 для хорошего SID качества.
- **SID prefix entropy:** насколько равномерно распределены item'ы по первым токенам $c_1$. Неравномерное распределение может приводить к bottle-neck beam search.
- **Online/offline метрики:** Recall@K, NDCG@K с full ranking. Отдельно для cold-start vs warm item'ов.

### 15.3. Типичные проблемы и решения

<div class="table-scroll">
<table>
<thead>
<tr><th>Проблема</th><th>Симптом</th><th>Решение</th></tr>
</thead>
<tbody>
<tr><td>Codebook collapse</td><td>Несколько кодов используются 90%+ времени; entropy низкая</td><td>k-means init; увеличить $\beta$; использовать EMA update для codebook'ов</td></tr>
<tr><td>Высокий collision rate</td><td>Многие item'ы имеют одинаковый SID</td><td>Увеличить $L$ (до 5-6) или $K$ (до 512); проверить качество content embedding'ов</td></tr>
<tr><td>Beam search пустой</td><td>Trie не имеет валидных continuation для prefix</td><td>Проверить, что все item'ы добавлены в trie; увеличить beam size</td></tr>
<tr><td>Плохое качество cold-start</td><td>Recall для cold item'ов в 10x хуже warm</td><td>Проверить, что content text заполнен качественно; возможно, нужна domain adaptation для SentenceT5</td></tr>
<tr><td>Медленный inference</td><td>Latency &gt; 500ms</td><td>Уменьшить L до 3; уменьшить beam size до 10; использовать quantization для T5; оптимизировать trie lookup</td></tr>
</tbody>
</table>
</div>

## 16. Итоговая оценка

TIGER — одна из наиболее влиятельных работ в modern recommender systems. Её значение выходит далеко за пределы конкретных числовых результатов: статья **сформулировала новую парадигму** — generative retrieval с semantic IDs для рекомендации — и убедительно показала её жизнеспособность.

Ключевые интеллектуальные вклады статьи:

1. Перенос идеи DSI/GENRE в recommendation domain с адаптацией под специфику задачи (каталог item'ов, sequential history, cold-start).
1. Введение Semantic ID как дискретного hierarchical identifier, объединяющего content semantics и компактность дискретного кода.
1. Демонстрация того, что RQ-VAE — правильный инструмент для построения SID: residual механизм обеспечивает иерархическую структуру, недостижимую в hierarchical k-means или PQ.
1. Constrained beam search через trie — простое и элегантное решение проблемы "несуществующих SID".
1. Эмпирическое доказательство, что semantic structure ID критически важна для cold-start generalization: Semantic ID vs Random ID — разница в 3x по Recall@10.

Ограничения статьи реальны и хорошо осознаны сообществом: objective mismatch, коллизии, latency, отсутствие collaborative signal в токенайзере. Но именно эти ограничения стали двигателем плодотворного направления исследований, породившего более 70 работ по 2024–2025 годам. TIGER — это не конечная точка, а **правильно поставленный вопрос**: можно ли сделать recommendation retrieval дифференцируемым end-to-end через генерацию структурированных item identifiers? TIGER отвечает: да, и вот как начать.

---

Авторы: Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Blake Anderson, Ed H. Chi, Maheswaran Sathiamoorthy. NeurIPS 2023. [arXiv:2305.05065](https://arxiv.org/abs/2305.05065).
