---
title: "Making Transformer Decoders Better Differentiable Indexers"
category: "generative_retrieval"
slug: "making_transformer_decoders_better_differentiable_indexers_summary"
catalogId: "paper-making_transformer_decoders_better_differentiable_indexers_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/making_transformer_decoders_better_differentiable_indexers_summary.html"
generatedFromHtml: true
paperUrl: "https://openreview.net/forum?id=bePaRx0otZ"
---
Подробное саммари статьи:

> **Авторы:** Wuchao Li, Kai Zheng, Defu Lian, Qi Liu, Wentian Bao, Enyun Yu, Yang Song, Han Li, Kun Gai.
>
> **Аффилиации:** University of Science and Technology of China; Kuaishou; Independent.
>
> **Публикация:** ICLR 2025 Poster, OpenReview forum bePaRx0otZ.

## 1. Коротко

URI (Unified framework for Retrieval and Indexing) делает радикальный шаг для generative retrieval: индекс больше не строится отдельным clustering/RQ-VAE этапом, а обучается тем же Transformer decoder'ом, который затем будет retrieval model. Авторы называют decoder differentiable indexer: он одновременно учится распределять item'ы по token buckets и извлекать релевантные buckets по query/user.

Главная претензия к DSI/NCI/TIGER-like pipelines: они двухстадийны. Сначала item representations кластеризуются в евклидовом пространстве, затем decoder учится запоминать уже построенный индекс. URI переносит index construction в interactive query-item space, где учитываются не только item features, но и collaborative/relevance отношения между queries/users и items.

## 2. Контекст

В generative retrieval item/document представляется последовательностью token IDs. Обычно эти IDs получены заранее: hierarchical k-means, PQ/RQ-VAE, tree index, semantic tokens. Такой индекс может быть сбалансированным и семантически разумным, но он не оптимизирован напрямую под decoder и query-item relevance.

URI пытается закрыть именно этот optimization gap. Если decoder в любом случае будет генерировать bucket tokens, почему бы не использовать его output distributions для построения самих buckets? Так индекс становится не внешним артефактом, а частью обучения retriever'а.

## 3. Метод

### 3.1. APE: метрика качества индекса

Авторы вводят Average Partition Entropy (APE). Для query берутся его positive items и смотрится, насколько они сконцентрированы в buckets текущего индекса. Чем ниже entropy распределения positive items по buckets, тем лучше: query легче сгенерировать нужный bucket. APE можно считать сразу после index construction, без полного обучения retriever'а.

Важное caveat: APE осмысленен только при относительно сбалансированном индексе. Если все item'ы положить в один bucket, entropy будет минимальной, но retrieval станет бесполезным.

### 3.2. Decoder as indexer and retriever

URI использует один decoder, который получает representations от query encoder и item encoder и выдает distribution по k token buckets. Есть два симметричных шага, обучаемых одновременно:

- **E-step-like retrieval loss.** Query distribution должен быть близок к token distributions релевантных item'ов.
- **M-step-like indexing loss.** Item distribution должен соответствовать ожидаемому distribution релевантных queries.
- **Commitment loss.** Item distribution должен приближаться к one-hot, чтобы item получил конкретный token.
- **Balance loss.** Среднее распределение item'ов должно быть близко к uniform, чтобы не получить collapsed index.

Индекс обучается layer-by-layer. После сходимости на текущем уровне item'ам явно назначаются tokens с balanced assignment; затем следующий уровень учится с учетом уже назначенных предыдущих tokens. Memory reinforcement loss помогает decoder не забыть ранние уровни.

### 3.3. Ranker и inference

После retriever/indexer URI обучает ranker: query и item representations сравниваются inner product'ом, sampled softmax используется с hard negative mining из sibling nodes. На inference decoder через beam search генерирует top bucket sequences, union item'ов из этих buckets образует candidate set, а final top-k выбирается ranker'ом.

## 4. Эксперименты/результаты

Датасеты покрывают три retrieval-сценария: KuaiSAR для short-video search/document retrieval, Amazon Beauty для general recommendation, Amazon Toys and Games для sequential recommendation. Baselines: DSI, NCI, RecForest, TIGER, DR. Метрики: Recall@10/20, NDCG@10/20 и APE.

<div class="table-scroll">
<table>
<thead><tr><th>Датасет</th><th>URI</th><th>Лучший baseline</th><th>Комментарий</th></tr></thead>
<tbody>
<tr><td>KuaiSAR R@10</td><td>0.1793</td><td>DR w/ URI index 0.1124; DSI 0.0976</td><td>URI сильно выигрывает, особенно из-за token-anonymized text, где pretrained text features слабее.</td></tr>
<tr><td>Beauty R@10</td><td>0.0157</td><td>RecForest/TIGER w/ URI index 0.0129</td><td>Улучшение меньше, но стабильно.</td></tr>
<tr><td>Toys R@10</td><td>0.0573</td><td>DR w/ URI index 0.0523</td><td>URI лучший среди GR variants.</td></tr>
</tbody>
</table>
</div>

APE comparison показывает, что URI строит индексы с заметно меньшим APE, чем Random, K-means, VQ-VAE и RQ-VAE. Например, на KuaiSAR при k=32: URI 0.654 против K-means 0.915 и RQ-VAE 0.909. Ablation подтверждает вклад adaptive balance weight и hard negative mining.

## 5. Ограничения

- **Layer-by-layer training.** URI не генерирует multi-level tokens за один проход обучения; авторы прямо называют ускорение multi-level generation future work.
- **Дополнительная training complexity.** Есть overhead примерно O(2L|D|t), потому что query и item проходят decoder отдельно на каждом уровне.
- **Нужен balanced assignment.** Collapse предотвращается loss'ами и assignment algorithm; это усложняет реализацию.
- **Эксперименты сосредоточены на GR baselines.** Работа доказывает преимущество среди generative indexing approaches, но не является полной production comparison с сильными ANN/dense retrieval системами.

## 6. Как читать для GR/SID

URI - одна из ключевых работ для вопроса “должен ли SID tokenizer быть обучаемым вместе с retriever'ом?”. В отличие от TIGER/RQ-VAE, где semantic IDs строятся по item text embeddings, URI строит token assignments из query-item interaction space. Это ближе к recommender reality: похожесть item'ов определяется не только текстом, но и тем, какие users/queries связывают их вместе.

Для SID-проектирования полезны три идеи: APE как быстрая метрика индекса до полного обучения; balance+commitment losses против collapse; layer-wise token consistency, где одинаковые tokens на разных ветках имеют согласованную семантику. Это хорошая основа для сравнения с ELCRec и последующими learnable-tokenization подходами.
