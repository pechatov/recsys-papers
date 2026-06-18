---
title: "CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "cost_contrastive_quantization_semantic_tokenization_summary"
catalogId: "paper-cost_contrastive_quantization_semantic_tokenization_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/cost_contrastive_quantization_semantic_tokenization_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2404.14774"
---

> **Авторы:** Jieming Zhu, Mengqun Jin, Qijiong Liu, Zexuan Qiu, Zhenhua Dong, Xiu Li.
>
> **Аффилиации:** Huawei Technologies (China); University Town of Shenzhen; Tsinghua University.

## 1. Коротко: о чем статья

CoST предлагает улучшить этап **semantic tokenization** в генеративных рекомендательных системах. В таких системах item retrieval формулируется как задача генерации последовательности токенов: модель по истории пользователя генерирует не вектор для ANN-поиска, а дискретные semantic IDs целевого item'а. Это похоже на то, как языковая модель генерирует токены текста, только токены здесь кодируют товары, новости или другие объекты рекомендаций.

Главная идея статьи: стандартная RQ-VAE-токенизация, используемая в TIGER, обучается через **точную реконструкцию эмбеддинга item'а**, но для retrieval важнее не восстановить embedding один-в-один, а сохранить **относительные отношения между item'ами**: какой item ближе, какой дальше, какие объекты нужно различать в плотных областях пространства. CoST заменяет MSE-реконструкцию на **contrastive loss** между исходным embedding'ом item'а и его реконструкцией после квантования. То есть реконструкция должна быть ближе к своему исходному item'у, чем к реконструкциям других item'ов в batch'е.

В экспериментах CoST используется как drop-in замена RQ-VAE-токенизации внутри TIGER. На MIND авторы показывают до **+43.34% Recall@5** и **+43.76% NDCG@5** относительно reconstruction-based RQ-VAE. На Amazon Office улучшения тоже есть, но они заметно слабее на большем Office(L), особенно по Recall.

## 2. Контекст: зачем нужны semantic tokens

Типичный промышленный recommender часто работает по схеме **retrieve-then-rank**:

1. Retrieval stage быстро достает несколько сотен или тысяч кандидатов из большого каталога.
1. Ranking stage дорого и точно переоценивает кандидатов.

Классический retrieval обычно строится через:

- two-tower models: user tower и item tower, затем dot product / cosine similarity;
- graph-based retrieval: item-user graph, GNN или упрощенные graph embeddings;
- ANN index, например Faiss, для быстрого top-k поиска по векторному пространству.

У такого подхода есть несколько ограничений:

- retrieval score часто ограничен простой похожестью векторов, например dot product;
- embedding model и ANN index оптимизируются раздельно;
- ANN-инфраструктура является внешним недифференцируемым компонентом;
- модель retrieval не генерирует item напрямую, а только предоставляет вектор для поиска.

Генеративный retrieval предлагает альтернативу: представить item не как один атомарный ID и не только как dense vector, а как **последовательность дискретных semantic tokens**. Тогда задача next-item recommendation превращается в seq2seq:

```
история пользователя -> последовательность токенов следующего item'а
```

После генерации токенов модель делает lookup в таблице `semantic token sequence -&gt; item ID` и получает рекомендации.

## 3. Как выглядит generative recommendation pipeline

В статье используется framework, близкий к TIGER:

1. **Tokenization phase**

- Берутся текстовые поля item'а: категория, подкатегория, title, description, brand и т.п.
- Они превращаются в dense semantic embedding через pretrained text encoder, например Sentence-T5 или E5.
- Затем embedding дискретизируется в последовательность semantic tokens.

1. **Generation phase**

- История пользователя представляется как длинная последовательность semantic tokens всех прошлых item'ов.
- Encoder-decoder Transformer обучается предсказывать tokens следующего item'а.
- На inference используется beam search.
- Сгенерированные token sequences мапятся обратно в item IDs.

Авторы подчеркивают: качество первой фазы критично. Если токены плохо отражают item space, даже хороший Transformer дальше будет учиться на плохом дискретном представлении.

## 4. Мотивация CoST

### 4.1. Что не так с RQ-VAE в этой задаче

TIGER использует RQ-VAE, чтобы превратить item embedding в tuple кодов. RQ-VAE учится минимизировать reconstruction loss: исходный embedding `x` после encoder, residual quantization и decoder должен быть восстановлен как `x_hat`, близкий к `x` по MSE.

Это логично для задач вроде image/audio tokenization, где tokenization должна сохранить достаточно информации для восстановления сигнала. Но recommendation retrieval имеет другую цель. Там не обязательно идеально восстановить исходный embedding; важнее:

- различать похожие, но разные item'ы;
- сохранять локальную структуру соседства;
- избегать слипания популярных или семантически близких item'ов в одинаковые token sequences;
- строить token space, удобный для autoregressive generation.

Авторы утверждают, что reconstruction-based objective рассматривает item'ы слишком независимо. Каждый item пытаются восстановить сам по себе, без явного требования правильно организовать отношения между item'ами внутри batch'а или локального neighborhood.

### 4.2. Проблема плотных областей и long tail

В recommender data часто есть long-tail distribution:

- мало популярных item'ов имеют много взаимодействий;
- много редких item'ов имеют мало взаимодействий;
- текстовые embedding'и могут образовывать плотные кластеры вокруг популярных тем или категорий.

Если RQ-VAE реконструирует embedding'и независимо, в плотной области несколько похожих item'ов могут получить одинаковые или очень похожие semantic IDs. Для генеративного retrieval это опасно: модель может сгенерировать правильный coarse semantic region, но не различить конкретный target item.

CoST пытается решить именно эту проблему: не просто приблизить `x_hat` к `x`, а сделать так, чтобы `x_hat_i` был более похож на свой `x_i`, чем на другие `x_j` / `x_hat_j` из batch'а.

### 4.3. Главный сдвиг в постановке задачи

RQ-VAE спрашивает:

> Насколько точно я могу восстановить исходный embedding item'а из его дискретных кодов?

CoST спрашивает:

> Насколько хорошо я могу отличить правильную reconstruction данного item'а от reconstruction других item'ов?

Это важный концептуальный сдвиг: CoST делает semantic tokenization более retrieval-oriented.

## 5. Базовая механика residual quantization

CoST сохраняет архитектурный каркас RQ-VAE:

- encoder `E` ;
- набор codebooks;
- residual quantizer;
- decoder;
- straight-through estimator / quantization loss для обучения codebooks.

Пусть `x` - dense item embedding, например 768-dimensional Sentence-T5 vector.

### 5.1. Encoder

Сначала MLP encoder переводит `x` в более низкоразмерное latent representation:

$$
z = E(x)
$$

В экспериментах:

- входной embedding: 768 dimensions;
- encoder hidden layers: `[512, 256, 128]` ;
- latent dimension: 96.

### 5.2. Residual quantization

Есть `M` codebooks:

$$
C_i = \{e_i^k \mid k = 1, \ldots, K\}, \qquad i = 1, \ldots, M
$$

где:

- `M` - число уровней / codebooks;
- `K` - размер каждого codebook;
- `e_i^k` - code vector;
- выбранный индекс `c_i` становится semantic token на уровне `i` .

Процедура:

1. Начальный residual:

$$
r_0 = z
$$

1. В первом codebook выбирается ближайший code vector:

$$
c_1 = \arg\min_k \lVert r_0 - e_1^k \rVert_2
$$

1. Считается остаток:

$$
r_1 = r_0 - e_1^{c_1}
$$

1. Остаток квантуется вторым codebook.

1. Процесс повторяется `M` раз.

На выходе item получает tuple:

$$
(c_1, c_2, \ldots, c_M)
$$

В основном setup статьи:

- `M = 3` ;
- `K = 64` ;
- semantic ID каждого item'а - 3-token tuple;
- codebooks shared across three levels, согласно описанию авторов.

Теоретическая емкость token space при `K=64, M=3` равна:

$$
64^3 = 262{,}144
$$

Для датасетов статьи этого достаточно: MIND содержит 12,251 item, Office(L) - 37,347 item. Но важно, что большая теоретическая емкость не гарантирует хорошую фактическую утилизацию codebook'ов и отсутствие коллизий.

## 6. Reconstruction-based objective: что делает RQ-VAE

После residual quantization code vectors суммируются:

$$
\hat z = \sum_{i=1}^{M} e_i^{c_i}
$$

Затем decoder восстанавливает embedding:

$$
\hat x = D(\hat z)
$$

Классическая loss:

$$
\mathcal L_{\mathrm{re}} = \mathcal L_{\mathrm{mse}} + \mathcal L_{\mathrm{rq}}
$$

где:

$$
\mathcal L_{\mathrm{mse}} = \lVert x - \hat x \rVert_2^2
$$

`L_rq` - quantization/codebook loss со stop-gradient, аналогично VQ-VAE:

$$
\mathcal L_{\mathrm{rq}} = \sum_i \left\lVert \operatorname{sg}(r_{i-1}) - e_i^{c_i} \right\rVert_2^2 + \beta \left\lVert r_{i-1} - \operatorname{sg}(e_i^{c_i}) \right\rVert_2^2
$$

Смысл двух частей:

- первая часть двигает code vector к residual;
- вторая часть, commitment term, заставляет encoder output не убегать от выбранного code vector;
- `sg` означает stop-gradient;
- `beta` контролирует commitment.

Чтобы снизить риск codebook collapse, авторы используют k-means initialization: на первом training batch делают k-means, а centroids берут как начальные code vectors.

## 7. Contrastive quantization в CoST

CoST сохраняет `L_rq`, но заменяет MSE-реконструкцию на contrastive objective.

### 7.1. Positive pair

Для item `0` positive pair:

$$
(x_0, \hat x_0)
$$

где:

- `x_0` - исходный semantic embedding item'а;
- `x_hat_0` - embedding, восстановленный после encoder, residual quantization и decoder.

### 7.2. Negative examples

Negatives - reconstructed vectors других item'ов в batch'е:

$$
\{\hat x_i \mid i = 1, \ldots, K_{\mathrm{batch}}\}
$$

В тексте формула использует `K` как число других векторов в batch'е, что немного перегружает обозначения, потому что `K` уже используется как codebook size. По смыслу здесь речь о batch negatives.

### 7.3. Contrastive loss

Loss:

$$
\mathcal L_{\mathrm{cl}} = -\log \frac{\exp(\langle x_0,\hat x_0\rangle / \tau)} {\sum_{i=0}^{K_{\mathrm{batch}}} \exp(\langle x_0,\hat x_i\rangle / \tau)}
$$

где:

- `&lt;.,.&gt;` - cosine similarity;
- `tau` - temperature;
- positive pair должен получить максимальную вероятность среди reconstructed candidates.

Итоговая CoST objective:

$$
\mathcal L_{\mathrm{co}} = \alpha\,\mathcal L_{\mathrm{cl}} + \mathcal L_{\mathrm{rq}}
$$

В экспериментах:

- `alpha = 0.1` ;
- `beta = 0.25` ;
- `tau = 0.1` .

### 7.4. Интуиция

MSE говорит: "восстанови координаты embedding'а".

Contrastive loss говорит: "сохрани идентичность item'а относительно других item'ов".

Это лучше соответствует retrieval: модель должна выбрать правильный item среди множества кандидатов. Даже если `x_hat` не идеально совпадает с `x` по всем координатам, он полезен, если остается ближе к своему item'у, чем к чужим.

### 7.5. Почему это может улучшить semantic IDs

CoST может помочь по нескольким причинам:

- **Более явное разделение item'ов.** В batch'е каждый item конкурирует с другими, что снижает вероятность слипания reconstruction'ов.
- **Сохранение локального neighborhood.** Если два item'а близки, contrastive objective вынуждает модель аккуратнее различать их, а не просто приблизительно реконструировать оба.
- **Retrieval-aligned objective.** Рекомендательная задача ближе к ranking/classification среди candidates, чем к regression reconstruction.
- **Меньше зависимости от точной геометрии PLM embedding'а.** MSE слепо наследует все координатные особенности embedding space, включая шум. Contrastive loss фокусируется на относительной близости.

## 8. Как CoST интегрируется в TIGER

CoST не предлагает новый generative recommender целиком. Это важно: вклад статьи находится в **tokenization phase**, а не в Transformer generation phase.

Pipeline:

1. Для каждого item строится text embedding через Sentence-T5.
1. CoST обучает residual quantizer с contrastive objective.
1. Каждый item получает semantic token tuple.
1. Последовательности взаимодействий пользователя заменяются последовательностями semantic tokens.
1. TIGER-like encoder-decoder Transformer обучается next-item token generation.
1. На inference:

- decoder генерирует token tuple;
- beam search возвращает несколько token sequences;
- token-to-item lookup превращает sequences в item IDs.

Сильная сторона такой постановки: CoST можно рассматривать как относительно локальную замену RQ-VAE в существующих generative retrieval системах.

Слабая сторона: если generation architecture, beam search или token-to-item mapping имеют собственные ограничения, CoST их не решает.

## 9. Экспериментальная диагностика и интерпретация

### Figure 1: Framework of generative recommendation

<img src="../../assets/cost/cost_x1.png" alt="Figure 1: framework of generative recommendation, tokenization phase and generation phase">

Рисунок показывает две фазы:

- **Tokenization Phase**
- item corpus;
- PLM embeddings;
- contrastive quantization;
- каждому item назначается sequence из code tokens, например `(5, 9, 8)` .

- **Generation Phase**
- user history подается в Transformer encoder;
- Transformer decoder autoregressively генерирует semantic tokens следующего item;
- token sequence мапится обратно в item.

Почему рисунок полезен:

- Он фиксирует, что CoST работает до обучения recommendation Transformer.
- Он показывает, что semantic tokenization уменьшает vocabulary: вместо миллионов item IDs модель работает с тысячами code tokens.
- Он объясняет, почему ошибка токенизации потом распространяется на весь generation stage.

Практическая интерпретация: если tokenization делает плохие semantic IDs, Transformer будет учиться на "битом языке item'ов".

### Figure 2: Reconstructive vs Contrastive Quantization

<img src="../../assets/cost/cost_x2.png" alt="Figure 2: vector quantization workflow trained via reconstructive quantization and contrastive quantization">

Это главный методологический рисунок.

Левая часть:

- input проходит quantization;
- output/reconstruction сравнивается с input через reconstruction loss;
- каждый item фактически оптимизируется сам по себе.

Правая часть:

- input/output своего item'а образуют positive pair;
- outputs других item'ов выступают negatives;
- contrastive loss заставляет correct reconstruction быть ближе к own input, чем к чужим.

Почему рисунок важен:

- Он показывает единственное ключевое изменение CoST относительно RQ-VAE: objective меняется с coordinate-level reconstruction на pairwise discrimination.
- Он помогает понять, почему авторы говорят о neighborhood relationships.

### Figure 3: Temperature `tau` и число эпох

<img src="../../assets/cost/cost_x3.png" alt="Figure 3a: sensitivity analysis of temperature tau on MIND">

<img src="../../assets/cost/cost_x4.png" alt="Figure 3b: sensitivity analysis of training epochs on MIND">

Рисунок анализирует MIND:

- `tau` тестируется в `{0.1, 0.5, 1.0}` ;
- CoST обучается 20 и 100 эпох;
- отдельно показывается динамика при 20, 50, 100 эпохах.

Выводы авторов:

- качество чувствительно к temperature;
- при 20 эпохах лучший Recall@40 получается при одном `tau` , при 100 эпохах - при другом;
- увеличение числа эпох улучшает NDCG и Recall, loss снижается.

Критическое замечание:

- такой результат означает не только "CoST хорошо обучается", но и то, что метод требует настройки training budget и `tau` ;
- в production-постановке это может быть нетривиально, особенно при больших каталогах и частом обновлении item corpus.

### Figure 4: Sensitivity analysis по codebook hyperparameters

<img src="../../assets/cost/cost_x5.png" alt="Figure 4a: sensitivity analysis of codebook size K on MIND">

<img src="../../assets/cost/cost_x6.png" alt="Figure 4b: sensitivity analysis of number of codebooks M on MIND">

<img src="../../assets/cost/cost_x7.png" alt="Figure 4c: sensitivity analysis of code vector dimension d on MIND">

Рисунок изучает:

- codebook size `K` ;
- number of codebooks `M` ;
- embedding dimension `d` .

Основные выводы:

- увеличение `K` улучшает NDCG;
- увеличение `M` улучшает NDCG;
- увеличение `d` с 32 до 96 помогает;
- `d = 256` уже ухудшает качество, авторы связывают это с overfitting.

Интерпретация:

- большее token space помогает точнее представить item'ы;
- но слишком высокая latent capacity может переобучаться;
- tokenization quality зависит от компромисса между емкостью semantic ID space и learnability генеративной модели.

## 10. Таблицы и результаты

### Table 1: Dataset statistics

<div class="table-scroll">
<table class="results-table">
<thead>
<tr><th>Dataset</th><th>#Users</th><th>#Items</th><th>Seq_Len</th><th>Avg seq len</th><th>Median</th></tr>
</thead>
<tbody>
<tr><td>MIND</td><td>29,207</td><td>12,251</td><td>[15, 70]</td><td>25.06</td><td>22</td></tr>
<tr><td>Office(S)</td><td>2,868</td><td>14,618</td><td>[10, 20]</td><td>13.21</td><td>12</td></tr>
<tr><td>Office(L)</td><td>16,696</td><td>37,347</td><td>[5, 50]</td><td>8.38</td><td>12</td></tr>
</tbody>
</table>
</div>

Важные детали preprocessing:

- MIND: оставляют interactions с at least 15 users, sequence length capped at 70.
- Amazon Office: фильтруют users with fewer than 5 interactions.
- Office разделяют на small и large, чтобы посмотреть влияние числа item'ов.
- Для MIND используют category, sub-category, title.
- Для Amazon используют type, brand, title, category.
- Sentence-T5 дает 768-dimensional item embeddings.

Что бросается в глаза:

- Office(L) имеет больше item'ов, но средняя sequence length ниже, чем у MIND.
- Office(S) очень мал по users: 2,868, что делает результаты потенциально шумными.
- MIND выглядит наиболее сильным benchmark'ом в статье.

### Table 2: Results on MIND

<div class="table-scroll">
<table class="results-table">
<thead>
<tr><th>Method</th><th>Metric</th><th>@5</th><th>@10</th><th>@20</th><th>@40</th></tr>
</thead>
<tbody>
<tr><td>Random</td><td>NDCG</td><td>0.0201</td><td>0.0265</td><td>0.0327</td><td>0.0390</td></tr>
<tr><td>Random</td><td>Recall</td><td>0.0319</td><td>0.0519</td><td>0.0766</td><td>0.1075</td></tr>
<tr><td><code>L_re</code></td><td>NDCG</td><td>0.0363</td><td>0.0474</td><td>0.0594</td><td>0.0727</td></tr>
<tr><td><code>L_re</code></td><td>Recall</td><td>0.0560</td><td>0.0905</td><td>0.1384</td><td>0.2031</td></tr>
<tr><td><code>L_co</code></td><td>NDCG</td><td>0.0522</td><td>0.0663</td><td>0.0817</td><td>0.0975</td></tr>
<tr><td><code>L_co</code></td><td>Recall</td><td>0.0803</td><td>0.1241</td><td>0.1855</td><td>0.2625</td></tr>
<tr><td><code>L_re + L_co</code></td><td>NDCG</td><td>0.0444</td><td>0.0574</td><td>0.0710</td><td>0.0865</td></tr>
<tr><td><code>L_re + L_co</code></td><td>Recall</td><td>0.0677</td><td>0.1081</td><td>0.1621</td><td>0.2376</td></tr>
<tr><td>Improvement over <code>L_re</code></td><td>NDCG</td><td>43.76%</td><td>39.90%</td><td>37.52%</td><td>34.10%</td></tr>
<tr><td>Improvement over <code>L_re</code></td><td>Recall</td><td>43.34%</td><td>37.21%</td><td>34.05%</td><td>29.24%</td></tr>
</tbody>
</table>
</div>

Главный результат:

- pure contrastive CoST ( `L_co` ) лучше reconstruction baseline ( `L_re` ) на всех k;
- комбинация `L_re + L_co` хуже, чем pure `L_co` , но лучше `L_re` .

Это важная деталь: добавление reconstruction loss не помогает на MIND. Значит, точная реконструкция может не просто быть ненужной, а мешать retrieval-aligned tokenization.

### Table 3: Results on Amazon Office

<div class="table-scroll">
<table class="results-table">
<thead>
<tr><th>Method</th><th>Metric</th><th>Office(S) @10</th><th>Office(S) @20</th><th>Office(L) @10</th><th>Office(L) @20</th></tr>
</thead>
<tbody>
<tr><td><code>L_re</code></td><td>NDCG</td><td>0.0024</td><td>0.0032</td><td>0.0041</td><td>0.0053</td></tr>
<tr><td><code>L_re</code></td><td>Recall</td><td>0.0037</td><td>0.0068</td><td>0.0075</td><td>0.0123</td></tr>
<tr><td><code>L_co</code></td><td>NDCG</td><td>0.0034</td><td>0.0043</td><td>0.0047</td><td>0.0059</td></tr>
<tr><td><code>L_co</code></td><td>Recall</td><td>0.0060</td><td>0.0094</td><td>0.0079</td><td>0.0125</td></tr>
<tr><td><code>L_re + L_co</code></td><td>NDCG</td><td>0.0035</td><td>0.0042</td><td>0.0042</td><td>0.0052</td></tr>
<tr><td><code>L_re + L_co</code></td><td>Recall</td><td>0.0066</td><td>0.0096</td><td>0.0079</td><td>0.0119</td></tr>
<tr><td>Improvement</td><td>NDCG</td><td>43.80%</td><td>31.06%</td><td>15.11%</td><td>10.53%</td></tr>
<tr><td>Improvement</td><td>Recall</td><td>80.95%</td><td>41.03%</td><td>5.38%</td><td>1.46%</td></tr>
</tbody>
</table>
</div>

Интерпретация:

- на Office(S) gains большие, особенно Recall@10;
- на Office(L) gains существенно меньше;
- `L_re + L_co` лучше на Office(S), но не на Office(L);
- pure `L_co` выглядит устойчивее на большем наборе.

Критический момент:

- абсолютные значения метрик на Office очень низкие;
- относительные улучшения могут выглядеть впечатляюще из-за низкой базы;
- на Office(L) Recall improvement 1.46% at @20 практически скромный.

## 11. Алгоритм CoST пошагово

Ниже - реконструкция алгоритма в удобном виде.

### Input

- Item corpus `I` .
- Для каждого item доступны текстовые поля.
- Pretrained text encoder, например Sentence-T5.
- Hyperparameters:
- number of codebooks `M` ;
- codebook size `K` ;
- latent dimension `d` ;
- contrastive temperature `tau` ;
- quantization commitment `beta` ;
- contrastive weight `alpha` ;
- batch size.

### Training semantic tokenizer

1. **Build item text**

Для каждого item создать prompt:

```text Item category: &lt;category&gt;. Title: &lt;title&gt;. Description: &lt;description&gt;.```

Для Amazon Office вместо description могут использоваться type, brand, title, category.

1. **Encode text**

Получить dense vector:

```text x_i = PLM(text_i)```

1. **Project to latent**

```text z_i = E(x_i)```

1. **Residual quantization**

Для каждого уровня `m = 1..M`:

- найти nearest code vector в codebook `C_m` ;
- записать code index `c_{i,m}` ;
- обновить residual.

1. **Reconstruct**

```text z_hat_i = sum_m e_m^{c_{i,m}} x_hat_i = D(z_hat_i)```

1. **Compute batch contrastive loss**

Для каждого item в batch positive pair - `(x_i, x_hat_i)`, negatives - `x_hat_j` других item'ов.

1. **Compute quantization loss**

Использовать `L_rq`, чтобы обучать encoder/codebooks через straight-through estimator.

1. **Optimize**

```text L_co = alpha * L_cl + L_rq```

1. **Export semantic IDs**

После обучения для каждого item сохранить:

```text item_id -&gt; (c_1,..., c_M)```

### Training generative recommender

1. Заменить item history пользователя на token sequence.
1. Обучить TIGER-like encoder-decoder Transformer.
1. Использовать leave-one-out:

- last item: test;
- penultimate item: validation;
- previous items: train.

1. На inference использовать beam search.
1. Вернуть top-k item IDs через lookup.

## 12. Что именно является вкладом статьи

Вклад не в том, что авторы впервые используют semantic IDs. Это уже есть в DSI, TIGER, EAGER, semantic indexing работах.

Основные вклады:

1. **Постановка semantic tokenization как contrastive quantization.**

1. **Аргумент против чистой MSE-реконструкции.**

1. **Простая интеграция с существующим framework.**

1. **Эмпирический сигнал, что tokenization objective сильно влияет на итоговый generative recommender.**

1. **Sensitivity analysis по tokenization hyperparameters.**

## 13. Плюсы статьи и метода

### 13.1. Хорошая постановка проблемы

Статья правильно фокусируется на semantic tokenization как на отдельном bottleneck. В generative recommendation часто основное внимание уходит на Transformer/generator, но если item language плохой, downstream model ограничена с самого начала.

### 13.2. Objective лучше соответствует retrieval

Contrastive loss естественнее для retrieval, чем MSE. Retrieval - это выбор правильного item среди конкурентов, а не восстановление embedding coordinates. CoST делает quantizer ближе к ranking/discrimination objective.

### 13.3. Минимальное изменение архитектуры

CoST не требует менять весь recommender. Это удобно:

- можно заменить tokenizer;
- можно оставить generation phase прежней;
- можно сравнивать tokenization objectives достаточно чисто.

### 13.4. Хорошие результаты на MIND

На MIND gains крупные и последовательные по Recall/NDCG и по всем k. Это сильный аргумент, что reconstruction objective действительно может быть неоптимальным.

### 13.5. Полезный анализ hyperparameters

Figure 3 и Figure 4 показывают, что tokenization не является одноразовой технической деталью. Размер codebook, число codebooks, latent dimension и temperature реально меняют результат.

### 13.6. Метод концептуально простой

CoST легко объяснить:

- positive pair: original item embedding + its quantized reconstruction;
- negatives: reconstructions of other batch items;
- loss: InfoNCE-like contrastive objective.

Это делает работу полезной как building block для дальнейших исследований.

## 14. Минусы, спорные моменты и недоработки

### 14.1. Небольшой масштаб экспериментов

Статья короткая, всего около 6 страниц, и экспериментальная часть ограничена:

- MIND;
- Amazon Office(S);
- Amazon Office(L).

Нет проверки на действительно industrial-scale catalog с миллионами item'ов. Но именно в таких сценариях semantic tokenization и ANN replacement наиболее интересны.

### 14.2. Слабая baseline coverage

Сравнение в основном идет между:

- random hashing;
- reconstruction RQ-VAE ( `L_re` );
- contrastive CoST ( `L_co` );
- combination ( `L_re + L_co` ).

Не хватает широкого сравнения с:

- hierarchical k-means tokenization;
- EAGER-style behavior-semantic tokenization;
- TokenRec / learnable tokenization approaches;
- semantic IDs from collaborative signals;
- simple strong dense retrieval baselines under the same evaluation.

Авторы используют TIGER as baseline framework, но не показывают полную картину конкурирующих tokenization methods.

### 14.3. Нет подробного анализа коллизий semantic IDs

Мотивация говорит, что RQ-VAE может назначать одинаковые token sequences похожим item'ам в плотных областях. Но в статье почти нет прямых метрик:

- collision rate;
- codebook utilization;
- entropy по codebooks;
- distribution of token sequence frequencies;
- number of items per semantic ID;
- collision impact on Recall/NDCG.

Без этого трудно понять, улучшает ли CoST именно collision/neighborhood проблему или просто дает regularization effect.

### 14.4. Top-one contrastive objective может быть слишком узким

Сами авторы в conclusion признают: CoST фокусируется на top-one neighborhood alignment. Но recommendation часто требует top-k neighborhood structure:

- item может иметь несколько релевантных соседей;
- substitutable/complementary relations не сводятся к одному positive;
- batch negatives могут содержать false negatives.

Например, две почти одинаковые новости или два похожих офисных товара могут оказаться negatives друг для друга, хотя с точки зрения recommendation они близкие и взаимозаменяемые.

### 14.5. Batch negatives могут быть шумными

CoST использует другие item'ы в batch как negatives. Это стандартно для contrastive learning, но здесь есть риски:

- false negatives: похожие item'ы считаются отрицательными;
- batch composition сильно влияет на сигнал;
- при малом batch может быть недостаточно hard negatives;
- при большом каталоге batch negatives могут плохо аппроксимировать глобальное item space.

Статья не исследует:

- hard negative mining;
- cross-batch memory;
- debiased contrastive loss;
- sampled softmax over larger candidate pools;
- relation-aware positives.

### 14.6. Текстовые embedding'и не обязательно отражают recommendation semantics

CoST стартует с PLM embedding'ов текстовых полей. Это хорошо для cold-start и semantic indexing, но recommendation relevance часто зависит от:

- co-click / co-purchase behavior;
- user intent;
- popularity;
- temporal trends;
- complementarity, а не только semantic similarity;
- domain-specific taxonomy.

Авторы почти не интегрируют collaborative signals в tokenizer. В related work они упоминают EAGER, TokenRec, LETTER, но CoST сам остается в основном text-semantic tokenizer.

### 14.7. Неясно, насколько улучшение связано именно с "neighborhood relationships"

Contrastive objective действительно меняет pairwise geometry. Но статья не дает глубокого embedding-space анализа:

- nearest-neighbor preservation до/после quantization;
- trustworthiness / continuity метрики;
- visualization of token clusters;
- intra-category vs inter-category separation;
- ranking correlation между original PLM space и quantized reconstruction space.

Поэтому causal story убедительна, но не полностью доказана.

### 14.8. Не раскрыта production сторона

Generative retrieval обещает убрать ANN index, но production deployment требует ответов на вопросы:

- как часто переобучать tokenizer при появлении новых item'ов;
- как добавлять new item без полного retraining;
- как обрабатывать deleted/updated item;
- как поддерживать token-to-item table при коллизиях;
- как контролировать latency beam search;
- как гарантировать diversity и business constraints;
- как совмещать generated candidates с существующим retrieval stack.

Статья это не рассматривает.

### 14.9. Ограниченная проверка на cold-start

Поскольку CoST использует текстовые embeddings, он потенциально подходит для item cold-start. Но статья не делает отдельного cold-start split:

- новые item'ы;
- новые категории;
- sparse interaction items;
- zero-shot item insertion.

Это упущенная возможность.

### 14.10. Слишком мало ablation по loss design

Было бы полезно сравнить:

- cosine contrastive vs dot-product contrastive;
- symmetric contrastive loss;
- original-to-reconstructed vs reconstructed-to-original directions;
- positives from nearest neighbors;
- supervised contrastive loss по категориям;
- contrastive + reconstruction с разными weights;
- memory bank / queue.

В статье есть только `L_re`, `L_co`, `L_re + L_co`.

### 14.11. Результаты на Office(L) выглядят умеренно

На Office(L) улучшения:

- NDCG@10: +15.11%;
- NDCG@20: +10.53%;
- Recall@10: +5.38%;
- Recall@20: +1.46%.

Это значительно слабее, чем на MIND и Office(S). Возможные причины:

- более sparse user histories;
- больше item'ов;
- текстовые fields хуже описывают recommendation behavior;
- token space / hyperparameters не оптимальны;
- contrastive negatives становятся менее полезными при большем и разреженном каталоге.

Этот результат снижает уверенность, что CoST масштабируется линейно и устойчиво.

## 15. Что можно было бы добавить в идеальной версии статьи

1. **Codebook diagnostics**

- utilization per codebook;
- perplexity;
- dead codes;
- entropy;
- collision distribution.

1. **Token collision analysis**

- сколько item'ов получают одинаковый tuple;
- в каких категориях больше collisions;
- как collisions коррелируют с ошибками recommendation.

1. **Neighborhood preservation metrics**

- recall of nearest neighbors after quantization;
- rank correlation между original embedding space и quantized reconstruction space;
- category purity.

1. **Hard negative experiments**

- random in-batch negatives vs nearest-neighbor negatives;
- false negative mitigation;
- cross-batch memory.

1. **Collaborative semantic tokenization**

- добавить user-item behavior graph;
- использовать co-click positives;
- contrastive positives не только own reconstruction, но и behavior-neighbor item'ы.

1. **Cold-start evaluation**

- new item split;
- new category split;
- item with no interactions but text available.

1. **Industrial constraints**

- update strategy for new items;
- latency of beam search;
- memory footprint;
- online A/B-like proxy.

1. **Comparison with stronger tokenizers**

- hierarchical clustering;
- EAGER;
- LETTER;
- TokenRec;
- semantic ID ranking approaches.

## 16. Практические выводы для исследователя или инженера

### 16.1. Semantic tokenizer - это не preprocessing detail

CoST показывает, что tokenization objective может дать большую разницу в итоговом recommender quality. Поэтому semantic IDs нельзя воспринимать как нейтральный preprocessing.

### 16.2. Reconstruction loss не всегда правильная цель

Если downstream задача - retrieval, стоит оптимизировать tokenizer под discrimination/ranking, а не только под reconstruction.

### 16.3. Batch contrastive objective - хороший первый шаг

CoST прост и может быть baseline для новых tokenization ideas:

- contrastive semantic tokenizer;
- multimodal contrastive tokenizer;
- collaborative contrastive tokenizer;
- top-k neighborhood-aware tokenizer.

### 16.4. Но нужны диагностики

При реализации CoST-like подхода стоит обязательно логировать:

- codebook utilization;
- token collision rate;
- distribution of semantic IDs;
- nearest-neighbor preservation;
- downstream Recall/NDCG;
- invalid generated token sequences;
- beam search hit rate.

Без этих метрик можно получить downstream gain, но не понимать, почему он появился.

## 17. Возможная реализация в псевдокоде

```
# 1. Build text embeddings
texts = [format_item_text(item) for item in items]
X = sentence_t5.encode(texts)  # [num_items, 768]

# 2. Train CoST tokenizer
for batch in dataloader(X):
    x = batch  # [B, 768]

    z = encoder(x)  # [B, d]
    residual = z
    selected_codes = []
    selected_vectors = []

    for m in range(M):
        # nearest code vector in codebook m
        code_idx = nearest_code(residual, codebook[m])
        code_vec = codebook[m][code_idx]

        selected_codes.append(code_idx)
        selected_vectors.append(code_vec)
        residual = residual - code_vec

    z_hat = sum(selected_vectors)
    x_hat = decoder(z_hat)

    # cosine similarities between original x_i and reconstructed x_hat_j
    logits = cosine_similarity_matrix(x, x_hat) / tau
    labels = torch.arange(len(x))

    L_cl = cross_entropy(logits, labels)
    L_rq = residual_quantization_loss(...)

    loss = alpha * L_cl + L_rq
    loss.backward()
    optimizer.step()

# 3. Export semantic IDs
semantic_ids = {}
for item, x in zip(items, X):
    semantic_ids[item.id] = cost_tokenizer.encode(x)  # tuple(c1, ..., cM)
```

Важно: этот псевдокод передает идею, но не все детали straight-through estimator и stop-gradient.

### 17.1. Что должно получиться на выходе каждого этапа

1. **После text encoding:** для каждого item есть фиксированный PLM vector; если эти embeddings плохи, CoST не исправит отсутствие collaborative signal.
1. **После residual quantization:** каждый item имеет tuple codes $(c_1,\ldots,c_M)$, а codebook utilization не должен collapse'иться в несколько популярных codes.
1. **После contrastive training:** reconstructed representation своего item должна быть ближе к original representation, чем reconstructions других batch items; это проверяется similarity matrix diagnostics.
1. **После экспорта SID:** token-to-item lookup должен явно хранить collisions, потому что генератор может сгенерировать валидный tuple, который соответствует нескольким item'ам.
1. **После обучения TIGER:** улучшение Recall/NDCG нужно сверить с invalid SID rate, beam hit rate и распределением сгенерированных prefixes.

## 18. Спорные интерпретации

### 18.1. "CoST captures neighborhood relationships" - частично да, но формально ограниченно

Contrastive loss действительно использует other items in batch. Но positive pair - это только `(x_i, x_hat_i)`, а не `(x_i, x_neighbor)`. Поэтому CoST скорее учит **instance discrimination** и indirectly preserves neighborhood, чем напрямую моделирует top-k neighborhood graph.

Более сильная версия neighborhood modeling могла бы использовать:

- nearest neighbors in PLM space as positives;
- co-click neighbors as positives;
- category-aware supervised contrastive positives;
- multi-positive InfoNCE.

### 18.2. Улучшение может быть связано с better separation, а не semantic understanding

Если contrastive loss просто сильнее разводит item'ы в representation space, это может улучшить Recall/NDCG, но не обязательно означает, что semantic tokens стали "семантически" лучше. Они могли стать более discriminative, но менее interpretable.

### 18.3. Удаление MSE может вредить переносимости

MSE сохраняет исходное PLM space более буквально. Contrastive loss может исказить embedding geometry ради текущего dataset'а. Это хорошо для in-domain retrieval, но может хуже работать для:

- transfer learning;
- new domains;
- zero-shot item insertion;
- explanation/interpretable semantic IDs.

### 18.4. Batch negatives могут конфликтовать с recommendation semantics

В recommendation похожие item'ы не всегда должны быть "разведены". Иногда похожие item'ы - хорошие substitutes, и их близость полезна. CoST же на уровне instance discrimination заставляет каждый item отличаться от других batch items.

Это может помогать точному next-item prediction, но потенциально вредить diversity, substitutability и broad recall.

## 19. Как я бы развивал CoST дальше

1. **Top-k contrastive quantization**

- positive set включает nearest neighbors item'а;
- own reconstruction остается главным positive;
- neighborhood positives получают меньший вес.

1. **Behavior-aware CoST**

- positives берутся из co-click/co-purchase graph;
- text semantic similarity комбинируется с collaborative similarity.

1. **Debiased negatives**

- не считать item negative, если он слишком близок по category/text/behavior;
- использовать soft labels вместо hard one-hot labels.

1. **Adaptive token length**

- popular/dense regions получают более длинные semantic IDs;
- sparse regions получают короткие IDs.

1. **Multimodal CoST**

- text + image + metadata + behavior;
- contrastive objective между modalities и quantized reconstruction.

1. **Online update mechanism**

- incremental code assignment для новых item'ов;
- periodic codebook refresh;
- backward-compatible semantic IDs.

1. **Joint tokenizer-generator training**

- сейчас tokenizer обучается отдельно;
- можно попробовать end-to-end или alternating optimization с downstream generation loss.

## 20. Итоговая оценка

CoST - небольшая, но полезная статья. Ее ценность не в сложной архитектуре, а в ясном observation: **для генеративного recommendation semantic tokenization должна быть retrieval-aware**. RQ-VAE с MSE-реконструкцией пришел из мира generative modeling, где важно восстановление сигнала. Но в item retrieval цель другая: различить правильный item среди конкурентов и сохранить полезную структуру item space.

Метод CoST делает простой шаг в правильную сторону: заменяет reconstruction-centric objective на contrastive objective. Эксперименты на MIND показывают сильный эффект. При этом статья оставляет много открытых вопросов: масштабирование, коллизии, false negatives, collaborative signals, cold-start, production lifecycle.

Моя оценка: CoST стоит читать как **важный baseline и conceptual pivot** для работ по semantic IDs. Это не окончательное решение tokenization problem, но хороший аргумент, что следующие поколения generative recommenders должны проектировать tokenizer не как autoencoder, а как retrieval/ranking-aware модуль.

## 21. Что читать после CoST

### Базовые работы по generative retrieval и semantic IDs

1. **[TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)**

1. **[Transformer Memory as a Differentiable Search Index (DSI)](https://arxiv.org/abs/2202.06991)**

1. **[Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904)**

1. **[Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations](https://arxiv.org/abs/2306.08121)**

### Работы про semantic tokenization в recommender systems

1. **[EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration](https://arxiv.org/abs/2406.14017)**

1. **[Learnable Item Tokenization for Generative Recommendation (LETTER)](https://arxiv.org/abs/2405.07314)**

1. **[TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation](https://arxiv.org/abs/2406.10450)**

1. **[How to Index Item IDs for Recommendation Foundation Models](https://arxiv.org/abs/2305.06569)**

1. **[Discrete Semantic Tokenization for Deep CTR Prediction](https://arxiv.org/abs/2403.08206)**

### Quantization и contrastive quantization

1. **[Neural Discrete Representation Learning / VQ-VAE](https://arxiv.org/abs/1711.00937)**

1. **[Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941)**

1. **[SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312)**

1. **[Contrastive Quantization with Code Memory for Unsupervised Image Retrieval](https://arxiv.org/abs/2109.05205)**

1. **[Improved Vector Quantization for Dense Retrieval with Contrastive Distillation](https://sigir.org/sigir2023/program/accepted-papers/short-papers/)**

1. **[Vector Quantization for Recommender Systems: A Review and Outlook](https://arxiv.org/abs/2405.03110)**

### Более широкая линия generative recommendation

1. **[P5: Recommendation as Language Processing](https://arxiv.org/abs/2203.13366)**

1. **[Recent Advances in Generative Information Retrieval](https://doi.org/10.1145/3589335.3641239)**

1. **[Non-autoregressive Generative Models for Reranking Recommendation](https://arxiv.org/abs/2402.06871)**

### Что читать в первую очередь

Если цель - быстро продолжить именно тему CoST, я бы шел в таком порядке:

1. [TIGER](https://arxiv.org/abs/2305.05065) - чтобы понять baseline framework.
1. [VQ-VAE](https://arxiv.org/abs/1711.00937) / [RQ-VAE](https://arxiv.org/abs/2203.01941) - чтобы понять quantization mechanics.
1. [EAGER](https://arxiv.org/abs/2406.14017) - чтобы увидеть behavior-semantic collaboration.
1. [LETTER](https://arxiv.org/abs/2405.07314) или [TokenRec](https://arxiv.org/abs/2406.10450) - чтобы сравнить разные learnable tokenization strategies.
1. [Vector Quantization for Recommender Systems survey](https://arxiv.org/abs/2405.03110) - чтобы собрать общую карту области.
