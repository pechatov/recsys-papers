---
title: "Vector Quantization for Recommender Systems: A Review and Outlook"
category: "semantic_ids_tokenization_indexing"
slug: "vector_quantization_for_recommender_systems_review_and_outlook_summary"
catalogId: "paper-vector_quantization_for_recommender_systems_review_and_outlook_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/vector_quantization_for_recommender_systems_review_and_outlook_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2405.03110"
---
Подробное саммари статьи:

> **Авторы:** Qijiong Liu, Xiaoyu Dong, Jiaren Xiao, Nuo Chen, Hengchang Hu, Jieming Zhu, Chenxu Zhu, Tetsuya Sakai, Xiao-Ming Wu.
>
> **Аффилиации:** The Hong Kong Polytechnic University; National University of Singapore; Huawei Noah's Ark Lab; Waseda University.
>
> **Статус:** arXiv:2405.03110, версия v1 от 2024-05-06.
>
> **Первичный источник:** [arXiv](https://arxiv.org/abs/2405.03110).

## 1. Коротко: о чем обзор

Статья систематизирует **VQ4Rec** - применение vector quantization в recommender systems. Это не только обзор semantic IDs для generative recommendation. Авторы показывают, что VQ в рекомендациях используется в двух больших режимах: **efficiency-oriented** и **quality-oriented**.

Efficiency-oriented VQ применяют для space compression, similarity search и acceleration. Здесь VQ помогает хранить огромные embedding tables компактнее, ускорять MIPS/ANN search и сокращать стоимость long-sequence attention. Quality-oriented VQ применяют для feature enhancement, modality alignment и discrete tokenization. Именно в этой ветке находится TIGER/RQ-VAE и большая часть semantic-ID литературы.

Для generative recommendation главный вывод обзора: residual quantization стала базовым строительным блоком item semantic tokenization, но сама по себе она не решает вопросы codebook collapse, code quality, multimodal tokenization, user tokenization и alignment с LLM. Поэтому статья полезна как карта проблем вокруг tokenizer'ов, а не как аргумент "просто используем RQ-VAE".

## 2. Почему VQ снова стала важной

Vector quantization появилась задолго до современных recommender systems: классические VQ-методы использовались в signal processing для сжатия представлений. В IR большой всплеск был связан с product quantization для approximate nearest neighbor search. В recommender systems раннее применение авторы датируют music recommendation, но массовый интерес появился позже из-за трех факторов:

- **VQ-VAE.** Нейросетевой differentiable-enough bottleneck сделал VQ удобной для representation learning.
- **RQ-VAE.** Residual quantization дала coarse-to-fine codes, которые хорошо подходят для последовательной генерации item IDs.
- **LLM и generative recommendation.** Большие модели усилили потребность в compact discrete representations, которые можно генерировать как tokens.

Авторы отмечают, что рост VQ4Rec связан не только с качеством, но и с инженерной экономикой. Большие каталоги, длинные истории, multimodal item content и LLM inference требуют способов уменьшить storage, latency и search cost.

## 3. Базовые техники VQ

<figure class="paper-figure">
  <img src="../../assets/vq4rec/figure2_vq_techniques.png" alt="Standard, parallel, and sequential vector quantization techniques">
  <figcaption><strong>Figure 2.</strong> Три базовых режима VQ. Standard VQ заменяет весь vector одним code. Parallel VQ делит vector на subspaces и квантует их независимо. Sequential VQ/RQ квантует residual по уровням, что напрямую связано с semantic ID tuple в TIGER-like системах. Источник: Liu et al., arXiv:2405.03110.</figcaption>
</figure>

### 3.1. Standard VQ

Standard vector quantization заменяет каждый dense vector ближайшим prototype vector из codebook. Пусть есть матрица object vectors $E \in \mathbb{R}^{N \times D}$. Нужно найти codebook $C \in \mathbb{R}^{K \times D}$, где $K \ll N$, чтобы минимизировать расстояния от vectors до ближайших codes. Для item vector $e_i$ выбирается индекс:

$$
x = \arg\min_{j=1,\ldots,K} d(e_i, c_j), \qquad e_i \approx c_x.
$$

В рекомендациях это можно понимать как clustering item/user embeddings в конечный словарь. Проблема standard VQ - грубая аппроксимация: один code должен заменить весь $D$-dimensional vector.

### 3.2. Parallel VQ: PQ и OPQ

Product quantization делит high-dimensional vector на $M$ sub-vectors и квантует каждый subspace отдельным codebook. Вместо одного выбора из $K$ codes модель получает комбинацию codes из нескольких codebooks. При этом размер codebook storage растет умеренно, а effective representation space растет экспоненциально по числу segments.

PQ полезна для ANN/MIPS, потому что расстояния между query и codes можно precompute в lookup tables и быстро суммировать. Недостаток: uniform split dimensions игнорирует correlations между sub-vectors.

Optimized Product Quantization добавляет learnable rotation matrix $R$, чтобы повернуть пространство перед PQ и уменьшить зависимость между subspaces:

$$
E' = E R.
$$

После quantization результат можно вернуть в исходное пространство через $R^T$. OPQ пытается сделать subspaces более независимыми и тем самым улучшить quantization quality.

### 3.3. Sequential VQ: RQ и AQ

Sequential vector quantization использует несколько codebooks последовательно. Residual Quantization работает coarse-to-fine: первый codebook аппроксимирует vector, следующий квантует остаток, следующий квантует остаток от остатка. После $M$ уровней:

$$
e_i \approx \sum_{j=1}^{M} c^{(j)}_{x_j}.
$$

Это ключевая идея RQ-VAE и semantic IDs: item получает не один token, а tuple $(x_1, x_2, \ldots, x_M)$, где ранние уровни обычно несут более coarse signal, а поздние уточняют residual details.

Additive Quantization похожа тем, что суммирует codes из нескольких codebooks, но не ограничивается greedy residual nearest neighbor на каждом уровне. В AQ combinations ищутся через beam search, что потенциально лучше глобально, но сложнее.

### 3.4. Differentiable VQ и STE

Главная сложность VQ в neural networks - nearest-neighbor assignment не дифференцируем. После VQ-VAE стандартным workaround стал Straight-Through Estimator. В forward pass quantization выполняется жестко, а в backward pass gradient пропускается как будто операция была identity:

$$
\frac{\partial c_x}{\partial e_i} \approx I.
$$

Это позволяет обучать encoder/codebook внутри neural model, но приводит к известным проблемам. Главная - **codebook collapse**: значительная часть codes не используется или используется слишком редко. Обзор упоминает EMA updates и codebook reset как практические способы бороться с underutilization.

Авторы также выделяют finite scalar quantization (FSQ) как перспективную альтернативу: вместо nearest-neighbor codebook используется rounding по dimensions. На момент обзора FSQ еще не применялась в recommender systems, но выглядит потенциально полезной.

## 4. Taxonomies of VQ4Rec

Авторы предлагают несколько осей классификации.

### 4.1. По training phase

<figure class="paper-figure">
  <img src="../../assets/vq4rec/figure3_training_stages.png" alt="Vector quantization integration points in a recommender system: preprocessing, in-processing, and post-processing">
  <figcaption><strong>Figure 3.</strong> Где VQ может стоять в recommender pipeline. Pre-processing tokenizes inputs заранее, in-processing обучает VQ вместе с recommender model, post-processing квантует уже полученные user/item embeddings для search или compression. Это разные инженерные риски: freshness, end-to-end alignment и serving constraints отличаются.</figcaption>
</figure>

- **Pre-processing.** VQ применяется до recommender model: например, item features или user sequences превращаются в static quantized inputs.
- **In-processing.** VQ интегрирована в модель и обучается вместе с ней, выдавая dynamically quantized features.
- **Post-processing.** VQ применяется к embeddings, уже полученным recommender model, например для ускорения similarity search.

Это разделение важно для оценки рисков. Pre-processing tokenizer может устаревать отдельно от recommender. In-processing сложнее обучать, но потенциально лучше согласуется с objective. Post-processing удобен для ускорения, но редко меняет сами user/item representations.

### 4.2. По application scenario

Главная taxonomy статьи делит VQ4Rec на две ветки:

<figure class="paper-figure">
  <img src="../../assets/vq4rec/figure4_application_taxonomy.png" alt="VQ4Rec application taxonomy with efficiency-oriented and quality-oriented branches">
  <figcaption><strong>Figure 4.</strong> Application taxonomy обзора. Efficiency-oriented ветка использует VQ для space compression, similarity search и model acceleration. Quality-oriented ветка использует VQ для feature enhancement, modality alignment и discrete tokenization; semantic-ID работы находятся именно здесь, но не исчерпывают VQ4Rec.</figcaption>
</figure>

<div class="table-scroll">
<table>
<thead>
<tr><th>Ветка</th><th>Подзадачи</th><th>Типичный смысл VQ</th></tr>
</thead>
<tbody>
<tr><td>Efficiency-oriented</td><td>Space compression, similarity search, model acceleration</td><td>Сжать embeddings/caches, ускорить ANN/MIPS или attention</td></tr>
<tr><td>Quality-oriented</td><td>Feature enhancement, modality alignment, discrete tokenization</td><td>Сделать representation более структурированным, transferable или generatable</td></tr>
</tbody>
</table>
</div>

### 4.3. По target

Большинство работ квантуют item representations, потому что item features обычно стабильнее user preference, а catalogs часто огромны. Но обзор отдельно отмечает user quantization и совместную item/user quantization. User tokenization менее зрелая, но важна для personalized LLM/recommender alignment.

## 5. Efficiency-oriented VQ4Rec

### 5.1. Space compression

Embedding tables в больших recommender systems могут быть огромными. Авторы приводят ориентир: 1 миллиард пользователей с 64-dimensional vectors в FP32 требует около 238 GB памяти. VQ предлагает заменять dense embeddings компактными code combinations.

Примеры направлений:

- PQ-VAE получает discrete user representations из user-item interactions для быстрых CTR predictions.
- Federated/decentralized approaches вроде ReFRS используют VAE/VQ для user tokens.
- Recent residual-quantization approaches сжимают user history и item content в short tokens; авторы упоминают возможность около 100x space compression при caching tokens вместо embeddings.
- Другие работы применяют VQ напрямую к embedding tables, например differentiable VQ для item embeddings.

Ограничение: многие claims по compression еще недостаточно проверены на настоящих large-scale recommendation models с большими embedding dimensions.

### 5.2. Model acceleration

Long user histories делают attention дорогим: стандартный self-attention квадратичен по длине sequence. VQ можно использовать для clustering/condensing attention space. Обзор выделяет LISA: модель ускоряет inference для long-sequence recommendation через codeword histograms, пытаясь сохранить full contextual attention при стоимости ближе к sparse/linear approaches.

Авторы считают VQ-based acceleration перспективной для lifelong learning и long sequence features, но признают, что применение VQ именно для model optimization пока ограничено.

### 5.3. Similarity search

Similarity search - самый исторически зрелый use case VQ. Product quantization позволяет быстро оценивать approximate distances через precomputed distance tables. В recommender systems это касается item-item search и user-item search.

Обзор подчеркивает, что parallel и sequential quantization по-разному расширяют representation capacity: PQ делает это "горизонтально", через subspaces; RQ - "вертикально", через successive residual levels. Открытый вопрос - как эффективно совмещать оба подхода, чтобы одновременно получить fine segmentation и compact representation.

## 6. Quality-oriented VQ4Rec

### 6.1. Feature enhancement

VQ может усиливать representations в sparse/cold-start сценариях. Примеры из обзора:

- User interest clusters через VQ для cluster-level contrastive learning между inactive и active users.
- VQ + attention для выбора candidate combination patterns.
- CAGE-like подходы, где categorical code representations и entity embeddings обучаются вместе для ID-based recommendation.

Общая идея: VQ добавляет intermediate discrete structure, которая может сглаживать sparsity и связывать похожие сущности на уровне clusters/codes.

### 6.2. Modality alignment

В transferable/multimodal recommendation VQ может ослабить жесткую привязку между raw item ID и content representation. Например, item text/image/ID можно связать через compact code representations. Авторы обсуждают работы, которые используют product quantization или related VQ constraints, чтобы сохранить modality-specific patterns и улучшить alignment.

Ограничение текущей литературы: большинство подходов работают с двумя modalities. Для трех и более modalities нужны новые решения, потому что simple pairwise alignment плохо масштабируется и может терять complementary signals.

### 6.3. Discrete tokenization

Это центральная часть для semantic-ID generative recommendation. Традиционный atomic item ID не несет семантики и плохо переносится на cold-start. Более поздние document-retrieval-inspired подходы используют tree IDs или multi-layer k-means, но могут страдать semantic mismatch.

TIGER - ключевой пример embedding-level reconstruction pipeline:

1. Извлечь item embeddings из content, обычно text encoder'ом.
1. Дискретизировать embeddings через residual quantization/RQ-VAE.
1. Использовать полученные item tokens в sequential recommendation, где model генерирует следующий semantic ID.

Сильная сторона RQ - hierarchical/coarse-to-fine organization tokens. Поэтому TIGER стал foundational для следующих работ. LC-Rec развивает идею в сторону интеграции item tokens в large models. Отдельные работы пробуют text-level reconstruction: трактовать item tokenization как translation task и обучать tokenizer/decoder восстанавливать item content, а не только embedding.

Авторы отмечают, что multimodal и multi-domain item tokenization пока мало изучены, хотя именно там VQ может стать основой future foundational recommender models.

## 7. Future directions

### 7.1. Codebook collapse

Codebook collapse - главный технический риск. Если используется только малая часть codes, модель теряет capacity и diversity. Для RecSys это особенно опасно: разные items/users начинают попадать в слишком похожие codes, что ухудшает personalization и recommendation diversity. EMA и code reset помогают, но не закрывают проблему полностью.

### 7.2. Item discovery

В item tokenization theoretical code space обычно намного больше числа items. Это означает, что существуют unused code combinations. Авторы предлагают интересную перспективу: дать human-readable description для новых code combinations. В product recommendation это может подсказать merchant'ам новые продукты; в video recommendation - помочь платформе создавать personalized content по latent demand.

### 7.3. User tokenization

Большинство схем дискретизируют items. User tokenization остается более сложной из-за динамики предпочтений, но потенциально очень ценна для personalized LLM and recommendation models. Хорошие user tokens могли бы дать compact personalization signal без передачи полного history.

### 7.4. Multimodal tokenization и RS-LLM alignment

Современные semantic IDs в RecSys в основном text-based. Авторы считают multimodal tokenization важным направлением, потому что реальные items содержат text, images, video, audio и graph/context signals. Еще одна линия - alignment recommender objects с LLM: discretized item IDs можно fine-tune'ить в LLaMA-like models, но для foundational recommender нужны multi-domain skills и надежная связка item/user/content tokens.

### 7.5. Codebook quality evaluation

Сейчас качество codebook часто проверяется через downstream recommendation. Это дорого и медленно: нужно обучать recommender, запускать evaluation, сравнивать Recall/NDCG. Авторы выделяют отдельную проблему: нужны быстрые proxy metrics для code quality, которые сравнивают generated tokens с original inputs и предсказывают downstream usefulness.

### 7.6. Efficient large-scale recommender systems

VQ может стать частью efficiency toolkit рядом с distillation, model quantization, parameter-efficient tuning и linear attention. Но для больших моделей вроде LLaMA с embedding dimension порядка тысяч simple VQ может быть недостаточной. Авторы предлагают исследовать комбинации parallel quantization и linear attention.

## 8. Сильные стороны

- **Широкий охват VQ4Rec.** Обзор не ограничивается semantic IDs и включает storage/search/attention use cases.
- **Хорошая базовая часть.** Standard VQ, PQ/OPQ, RQ/AQ, STE и codebook collapse объяснены достаточно системно.
- **Полезная taxonomy.** Training phase, application scenario, VQ technique и quantization target помогают быстро классифицировать новые papers.
- **Прямое попадание в semantic-ID литературу.** TIGER/RQ-VAE помещены в более широкую историю VQ, что помогает понять, почему RQ стала стандартной starting point.

## 9. Ограничения

- **Survey ранней стадии поля.** Многие направления обозначены как promising, но пока имеют мало сильных сравнений.
- **Нет unified experiments.** Статья не пересравнивает методы; выводы зависят от результатов оригинальных papers.
- **Мало production деталей.** Freshness, online updates, serving constraints, monitoring codebook drift и rollback tokenizers почти не разобраны.
- **Quality metrics для tokenizer'ов остаются открытыми.** Авторы сами признают, что downstream evaluation дорогая, а reliable proxy metrics еще нужны.

## 10. Итог

Vector Quantization for Recommender Systems: A Review and Outlook - один из ключевых foundation обзоров для semantic IDs. Он объясняет, почему VQ полезна не только как compression trick, но и как способ создать learned discrete vocabulary для recommender systems.

Главный takeaway: RQ-VAE/TIGER - это лишь одна, quality-oriented ветка VQ4Rec. Вокруг нее есть более широкая область: PQ/OPQ для search, VQ для embedding compression, VQ-based attention acceleration, modality alignment, user tokenization и RS-LLM alignment. Поэтому при чтении новых semantic-ID papers нужно отдельно смотреть, какую именно роль играет VQ: сжатие, поиск, alignment, tokenization или downstream generation.
