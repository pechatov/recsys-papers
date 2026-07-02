---
title: "Unified Semantic and ID Representation Learning for Deep Recommenders"
category: "semantic_ids_tokenization_indexing"
slug: "unified_semantic_and_id_representation_learning_for_deep_recommenders_summary"
catalogId: "paper-unified_semantic_and_id_representation_learning_for_deep_summary"
paperUrl: "https://arxiv.org/abs/2502.16474"
---
> **Авторы:** Guanyu Lin, Zhigang Hua, Tao Feng, Shuang Yang, Bo Long, Jiaxuan You.
>
> **Аффилиации:** University of Illinois at Urbana-Champaign; Meta AI.
>
> **Источник:** arXiv:2502.16474v1 от 2025-02-23.

## 1. Коротко: о чем статья

Работа предлагает гибридную tokenization scheme для deep recommenders: не заменять ID tokens на semantic tokens полностью, а объединить их. ID tokens хорошо запоминают unique item attributes, но плохо generalize и занимают большой embedding space. Semantic tokens лучше шарят transferable characteristics между похожими items, но могут создавать item duplication/collisions и хуже различать очень похожие objects.

Unified Semantic and ID Representation Learning оставляет маленький low-dimensional ID token на item и добавляет semantic tokens, полученные через RQ-VAE из text embeddings. Дополнительно авторы комбинируют cosine similarity и Euclidean distance в quantization: cosine лучше decouple accumulated embeddings на первых слоях, Euclidean лучше distinguish unique items на финальном слое.

Главный результат: на Beauty, Sports и Toys метод улучшает baselines на 6-17% и уменьшает token size больше чем на 80% относительно full ID tokenization.

<figure class="paper-figure">
  <img src="../../assets/unified_semantic_id/framework.png" alt="Unified semantic and ID representation learning framework">
  <figcaption>Рисунок 1. Framework объединяет compressed ID token и semantic tokens: ID часть сохраняет item-specific signal, semantic RQ-VAE часть дает shared transferable structure.</figcaption>
</figure>

## 2. Мотивация: ID vs semantic tokens

ID tokens дают memorization. Для frequent items это полезно: model может выучить точный collaborative pattern конкретного item'а. Но ID embedding table растет линейно с catalog size, плохо переносится на cold-start и содержит redundancy: многие item IDs лежат близко в embedding space и могут быть compressed.

Semantic tokens дают generalization. Они группируют похожие items и позволяют sharing параметров. Но если два item'а слишком похожи в text embedding, semantic tokenizer может присвоить им одинаковые или почти одинаковые codes. Тогда model теряет способность различать конкретные items.

Статья делает pragmatic conclusion: semantic tokens и ID tokens complementary, поэтому лучше оставить небольшую ID-specific часть, а не выбирать только один тип.

## 3. Метод: unified tokenization

Для item создается sequence representation, объединяющая:

- semantic tokens из RQ-VAE quantization text description embedding;
- low-dimensional ID token, где размер ID embedding существенно меньше обычного full ID embedding.

В experiments ID dimension снижается с 64 до 8. Semantic part состоит из нескольких codebook levels, например `3 x 256 x 64`.

Итоговый recommender оптимизирует recommendation loss, RQ-VAE quantization loss и text reconstruction loss end-to-end. В качестве sequential backbone используется SASRec-like/S3-Rec framework в fine-tuning setup.

## 4. Unified distance function

Авторы отдельно анализируют cosine similarity и Euclidean distance для codebook selection.

Cosine similarity хорошо активирует codebooks на всех слоях, но хуже покрывает unique items: total coverage unique items 70.13%. Euclidean distance лучше различает unique items: coverage 92.67%, но first layer активирует только 5.86% codebook entries, то есть есть риск codebook underuse.

Unified approach использует cosine на первых слоях для decoupling accumulated embeddings и Euclidean на final layer для item distinction. В результате activation codebook достигает 100% across layers, а coverage unique items - 83.27%. Это компромисс между utilization и discriminability.

## 5. Эксперименты

Датасеты: Amazon Beauty, Sports and Outdoors, Toys and Games. Baselines: FM, GRU4Rec, Caser, SASRec, BERT4Rec, HGN.

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>Ours HIT@10</th><th>Ours NDCG@10</th><th>Lift</th></tr></thead>
<tbody>
<tr><td>Beauty</td><td>0.5318</td><td>0.3440</td><td>+12.22% HIT@10, +9.00% NDCG@10</td></tr>
<tr><td>Sports</td><td>0.5247</td><td>0.3168</td><td>+11.12% HIT@10, +10.42% NDCG@10</td></tr>
<tr><td>Toys</td><td>0.5456</td><td>0.3501</td><td>+17.01% HIT@10, +11.64% NDCG@10</td></tr>
</tbody>
</table>
</div>

Unified tokenization также лучше ID-only и semantic-only:

- Beauty: ID HIT@10 0.4654, Semantic 0.4956, Unified 0.5318;
- Sports: ID 0.4582, Semantic 0.4704, Unified 0.5247;
- Toys: ID 0.4603, Semantic 0.4644, Unified 0.5456.

Token size reduction значительный: 81.15% на Beauty, 85.41% на Sports, 81.06% на Toys относительно ID-only.

## 6. Codebook size и ID dimension

Увеличение codebook size не всегда помогает. На Sports peak наблюдается при codebook size 128: HR@10 0.5247, NDCG@10 0.3168. При 256, 512 и 1024 качество не растет и местами падает. Это согласуется с другими SID papers: larger codebook может ухудшить utilization и вызвать degeneration.

Анализ ID dimension показывает, что small ID part достаточно для unique item information. Улучшение shrink-ится при росте ID dimension, а после некоторого размера performance может падать. Это важный practical point: ID token нужен как residual uniqueness signal, а не как full replacement semantic part.

## 7. Сильные стороны

- Очень прагматичная постановка: не "semantic IDs вместо item IDs", а controlled combination.
- Есть явная token-size accounting, а не только accuracy.
- Unified cosine/Euclidean analysis полезен для всех RQ-VAE-style tokenizers.
- Результаты показывают, что маленький ID residual может компенсировать semantic collisions.

## 8. Ограничения и вопросы

Работа не является полноценным generative retrieval system: semantic/ID tokens используются в deep recommender, а не обязательно как autoregressive target IDs. Поэтому результаты нельзя напрямую переносить на beam-search GR.

Text embeddings строятся из item descriptions; для domains с бедным content semantic part может стать слабым.

Нет online evaluation и нет анализа catalog churn. Low-dimensional ID token для новых items все равно требует initialization/update strategy.

## 9. Вывод

Статья полезна как counterweight к радикальной замене item IDs. Главный takeaway: **semantic tokens хорошо дают sharing, но небольшой ID residual остается полезным для item-specific distinction**. Для production recommender это часто более безопасный путь, чем полностью отказаться от ID embeddings.
