---
title: "Closing the Performance Gap in Generative Recommenders with Collaborative Tokenization and Efficient Modeling"
category: "semantic_ids_tokenization_indexing"
slug: "closing_performance_gap_collaborative_tokenization_efficient_modeling_summary"
catalogId: "paper-closing_performance_gap_collaborative_tokenization_efficient_modeling_summary"
paperUrl: "https://arxiv.org/abs/2508.14910"
---
> **Авторы:** Simon Lepage, Jeremie Mary, David Picard.
>
> **Аффилиации:** CRITEO AI Lab; LIGM, École Nationale des Ponts et Chaussées.

## 1. Коротко: о чем статья

Статья атакует очень практичный вопрос: почему generative recommenders с semantic IDs часто выглядят красиво концептуально, но проигрывают хорошо настроенному ID-based SASRec? Авторы выделяют две причины.

Первая причина - **tokenization без collaborative signal**. TIGER-style pipeline строит semantic IDs из текстовых embeddings и RQ-VAE. Такие коды могут быть семантически интерпретируемыми, но они не обязательно отражают co-occurrence и transition patterns в user timelines.

Вторая причина - **неэффективная encoder-decoder архитектура**. В TIGER item history разворачивается в длинную token sequence, а decoder генерирует SID токен за токеном. Это дорого и не всегда нужно: timeline modeling и item-code decoding решают разные подзадачи.

Авторы предлагают два компонента: **COSETTE** для collaborative tokenization и **MARIUS** для более эффективной генеративной модели. Главный claim: сочетание collaborative tokenization и decoupled modeling сокращает или закрывает разрыв между generative recommenders и сильными ID-based baselines.

## 2. Контекст: важный baseline - SASRec++

Один из сильных аспектов статьи - авторы сначала усиливают SASRec baseline. Они показывают, что "обычный SASRec" легко недооценить: фильтрация, cross-entropy вместо BCE и crop/shuffle augmentation дают большой прирост.

<div class="table-scroll">
<table>
<thead>
<tr><th>SASRec variant</th><th>Beauty R@10</th><th>Sports R@10</th><th>Toys R@10</th></tr>
</thead>
<tbody>
<tr><td>SASRec BCE</td><td>5.48%</td><td>3.81%</td><td>5.86%</td></tr>
<tr><td>+ Filtering</td><td>6.33%</td><td>4.19%</td><td>6.49%</td></tr>
<tr><td>+ Cross-Entropy</td><td>8.70%</td><td>5.53%</td><td>8.84%</td></tr>
<tr><td>+ Crop &amp; Shuffle</td><td>9.73%</td><td>6.44%</td><td>9.94%</td></tr>
</tbody>
</table>
</div>

Этот baseline важен методологически. Если generative recommender сравнивать с недотюненным SASRec, можно переоценить пользу semantic IDs. Статья фактически говорит: performance gap надо закрывать против сильного ID-based reference, а не против слабой реализации.

## 3. Метод: COSETTE, collaborative and semantic tokenization

COSETTE сохраняет реконструктивную идею RQ-VAE, но добавляет collaborative objective. В стандартном reconstructive quantization tokenizer учится сжимать text embedding item'а и восстанавливать его. Это хорошо сохраняет semantic content, но не учитывает, какие item'ы совместно встречаются в user timelines.

COSETTE добавляет contrastive collaborative term: similarities в latent/tokenizer space должны быть согласованы с co-occurrence structure, извлеченной из batches user timelines. В результате tokenizer получает два вида давления:

- reconstruction loss сохраняет content semantics;
- collaborative loss сближает item'ы, которые реально соседствуют в пользовательском поведении.

<figure class="paper-figure">
  <img src="../../assets/cosette_marius/cosette_overview.png" alt="COSETTE overview with reconstruction and collaborative losses">
  <figcaption>Рисунок 1. COSETTE объединяет reconstruction objective и collaborative contrastive objective. Это делает tokenizer не только content-preserving, но и recommendation-aware.</figcaption>
</figure>

### 3.1. Почему collaborative tokenization важна

Если два item'а имеют похожие descriptions, но редко заменяют друг друга в поведении, content-only SID может поставить их слишком близко. И наоборот, товары с разным текстом могут быть поведенчески взаимозаменяемыми. COSETTE пытается внести именно этот второй тип знания в quantizer.

Авторы также обсуждают post-processing collisions. Для generative retrieval важно, чтобы SID path указывал на валидный item и не создавал слишком много неоднозначностей. Поэтому качество tokenizer оценивается не только reconstruction loss, но и downstream retrieval.

## 4. MARIUS: разделить timeline modeling и code decoding

MARIUS меняет архитектуру generative recommender. Вместо того чтобы один encoder-decoder Transformer обрабатывал плоскую последовательность всех SID tokens, MARIUS разделяет задачу на два уровня.

1. **Temporal Transformer** моделирует user timeline на уровне item positions.
1. **Depth Transformer** точечно генерирует SID tokens следующего item'а.
1. Item representation строится как fusion embeddings его code tokens.

Это снижает вычислительную стоимость: temporal attention работает по числу items $N$, а не по числу item-token positions $N L$.

<figure class="paper-figure">
  <img src="../../assets/cosette_marius/marius_overview.png" alt="MARIUS overview with temporal transformer and depth transformer">
  <figcaption>Рисунок 2. MARIUS разделяет временное моделирование истории и autoregressive decoding кодов item'а. Это ключевая архитектурная причина снижения inference cost.</figcaption>
</figure>

## 5. Complexity: где появляется выигрыш

Авторы сравнивают TIGER и MARIUS через $N$ - длину user sequence, $L$ - длину semantic ID, $T_E$ - число temporal/encoder layers, $T_D$ - число decoder/depth layers, $B$ - число generated items.

<div class="table-scroll">
<table>
<thead>
<tr><th>Model</th><th>Training complexity</th><th>Inference complexity</th></tr>
</thead>
<tbody>
<tr><td>TIGER</td><td>$(NL)^2 T_E + (N L^2 + L^2)T_D$</td><td>$(NL)^2 T_E + B(NL^2 + L^2)T_D$</td></tr>
<tr><td>MARIUS</td><td>$N^2T_E + N L^2T_D$</td><td>$N^2T_E + B L^2T_D$</td></tr>
</tbody>
</table>
</div>

Главная разница: TIGER платит attention cost по flattened token history, а MARIUS держит timeline на item-level. При длинных history и long SIDs эта разница становится существенной.

## 6. Эксперименты

Эксперименты построены на Amazon Reviews 2014 и 2023. Авторы сравнивают ID-based baselines, TIGER-style generative models, tokenizers RQ-VAE/CoST/LETTER/COSETTE и архитектуру MARIUS.

В tokenization ablation COSETTE дает лучший validation R@10 среди text-based quantization methods:

<div class="table-scroll">
<table>
<thead>
<tr><th>Model</th><th>Tokenizer</th><th>Video Games R@10</th><th>Movies &amp; TV R@10</th><th>Pet Supplies R@10</th></tr>
</thead>
<tbody>
<tr><td>TIGER</td><td>RQ-VAE</td><td>9.79%</td><td>5.03%</td><td>2.57%</td></tr>
<tr><td>TIGER</td><td>LETTER</td><td>10.33%</td><td>5.05%</td><td>2.48%</td></tr>
<tr><td>TIGER</td><td>COSETTE</td><td>10.89%</td><td>6.44%</td><td>3.30%</td></tr>
<tr><td>MARIUS</td><td>RQ-VAE</td><td>14.24%</td><td>9.16%</td><td>5.01%</td></tr>
<tr><td>MARIUS</td><td>COSETTE</td><td>15.02%</td><td>9.90%</td><td>5.32%</td></tr>
</tbody>
</table>
</div>

На больших Amazon 2023 категориях MARIUS обычно превосходит TIGER и часто конкурирует с SASRec++. Важно, что авторы не утверждают универсальное доминирование: на некоторых datasets сильный ID-based baseline остается очень конкурентным.

## 7. Что реально доказывает статья

Статья убедительно показывает, что performance gap generative recommenders нельзя списывать только на "генеративная постановка плохая". Gap возникает из конкретных инженерных решений: слабый tokenizer, отсутствие collaborative signal, слишком дорогая flattened архитектура и слабые baselines в предыдущих сравнениях.

COSETTE показывает, что tokenization objective matters. MARIUS показывает, что generative recommendation не обязан быть T5-style encoder-decoder по плоской token sequence. Вместе они дают практический blueprint: semantic IDs должны быть recommendation-aware, а модель должна уважать двухуровневую структуру "items in timeline" и "tokens inside item".

## 8. Сильные стороны

Сильная сторона работы - честная baseline discipline. Авторы явно усиливают SASRec и тем самым поднимают планку для собственных методов.

Второй плюс - разделение вклада tokenizer и model architecture. COSETTE можно использовать с TIGER и MARIUS; MARIUS можно тестировать с разными tokenizers. Это делает ablations содержательными.

Третий плюс - efficiency analysis. В recsys engineering качество без latency/cost не является полноценным результатом; MARIUS прямо оптимизирует computational shape.

## 9. Ограничения и открытые вопросы

Работа остается offline benchmark paper. Нет online A/B и нет production serving constraints вроде candidate deduplication, freshness, catalog churn, abuse filtering или business constraints.

COSETTE использует co-occurrence из timelines; в доменах с сильным exposure bias или noisy sessions collaborative objective может перенести bias в tokenizer. Нужны popularity-stratified и cold-start diagnostics.

MARIUS снижает cost, но сохраняет autoregressive decoding внутри item code depth. Для очень длинных semantic IDs может понадобиться parallel decoding или order-agnostic generation, как в RPG/SETRec.

## 10. Вывод

Эта статья полезна как sanity check для всей линии generative recommendation. Она показывает, что semantic-ID модели должны соревноваться с хорошо настроенными ID baselines, а не с учебными реализациями. Практический takeaway: если generative recommender проигрывает SASRec, сначала стоит проверить collaborative tokenization и архитектурную стоимость, прежде чем отвергать весь generative retrieval paradigm.
