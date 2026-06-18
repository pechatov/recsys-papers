---
title: "Towards Scalable Semantic Representation for Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "towards_scalable_semantic_representation_for_recommendation_summary"
catalogId: "paper-towards_scalable_semantic_representation_for_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/towards_scalable_semantic_representation_for_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2410.09560"
---
> **Авторы:** Taolin Zhang, Junwei Pan, Jinpeng Wang, Yaohua Zha, Tao Dai, Bin Chen, Ruisheng Luo, Xiaoxiang Deng, Yuan Wang, Ming Yue, Jie Jiang, Shu-Tao Xia.
>
> **Аффилиации:** Tsinghua University; Tencent Inc..
>
> **Первичный источник:** arXiv:2410.09560.

## коротко

MoC studies scalable semantic representation: multiple independent codebooks are used to preserve high-dimensional LLM semantics under small recommender embedding dimensions.

Ниже разбор сфокусирован на механике метода, objectives, экспериментальном setup, ablation conclusions и практических рисках внедрения.

- Single semantic embedding/code saturates under dimension compression.
- Multi-Embedding and RQ-VAE scale poorly in discriminability/robustness analysis.
- MoC uses Multi-Codebooks VQ-VAE at indexing stage.
- Downstream Mixture-of-Codes fuses code embeddings.
- Experiments show better scale-up performance on three public datasets.

## контекст

Работа находится в линии generative recommendation / semantic identifiers, где item представляется не только atomic ID, а дискретным, текстовым, многокодовым или генерируемым представлением.

Общий контекст: чем сильнее сжатие item semantics в короткий код, тем больше риск потерять информацию, которую downstream recommender уже не восстановит.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>Input</td><td>LLM item embeddings, often 4096-16384 dimensions</td></tr>
<tr><td>Constraint</td><td>recommender ID embedding dimensions are much smaller</td></tr>
<tr><td>Goal</td><td>scale semantic capacity without exploding downstream embedding tables</td></tr>
</table>
</div>

## проблема

A single code sequence or low-dimensional semantic embedding cannot span the complex structure of original LLM embedding space, causing information loss before recommendation.

Для оценки важно отделять три уровня: качество item representation, качество tokenizer/identifier и качество sequence/generative model. Многие papers выигрывают именно за счет улучшения одного уровня, а не всей системы сразу.

- Tokenizer/recommender mismatch: код удобен одному модулю и неудобен другому.
- Collisions или near-collisions ухудшают exact item retrieval.
- Long-tail items получают слабый сигнал и нестабильные IDs.
- Beam search может генерировать invalid или duplicate identifiers.
- Offline lift может не перенестись в production из-за drift, catalog churn и constraints.

## метод/архитектура

Метод вводит специальные компоненты поверх базового recommender/tokenizer pipeline. Важно, что авторы обычно стараются сохранить inference совместимым с существующей GR схемой или явно обсуждают trade-off inference cost.

Архитектурная идея раскрывается через следующие элементы.

- Multi-Codebooks VQ-VAE learns several independent discrete codebooks.
- Each item receives multiple codes, one per codebook/aspect.
- Mixture-of-Codes module fuses learnable code embeddings downstream.
- Analysis tracks discriminability and dimension robustness.
- Comparison includes Multi-Embedding and RQ-VAE variants.

## objective/алгоритм

Objective связывает representation learning, identifier learning и downstream recommendation/retrieval signal. В некоторых работах это explicit loss, в других — training schedule или iterative relabeling.

Для практической реализации важно логировать каждый компонент loss отдельно: общий Recall/NDCG не объясняет, какой механизм сработал.

- Indexing objective reconstructs/captures LLM embeddings through codebooks.
- Recommendation loss trains downstream fusion end-to-end.
- Scalability is evaluated through discriminability, robustness and performance metrics.
- Independent codebooks increase representational rank/capacity.
- Fusion module learns how much each code contributes.

## Детальный алгоритм MoC

MoC решает не задачу генерации одного SID sequence, а задачу сохранения большой LLM semantic capacity в компактных recommender embeddings. Один item получает несколько независимых codes, а downstream модель учится смешивать их как multiple aspects item semantics.

1. **Получить high-dimensional LLM embeddings.** Исходный item representation может иметь 4096-16384 dimensions, что слишком дорого для прямого использования в recommender embedding table.
1. **Обучить Multi-Codebooks VQ-VAE.** Вместо одного bottleneck используются несколько независимых codebooks. Каждый codebook квантует свой aspect/partition latent representation.
1. **Назначить несколько codes каждому item.** Item representation становится набором discrete codes, по одному из каждого codebook. Это повышает representational capacity без хранения полного LLM vector online.
1. **Экспортировать code embeddings.** Downstream recommender хранит embedding tables для codes, а не огромные dense item embeddings.
1. **Смешать codes через Mixture-of-Codes.** Fusion module (sum/attention/MLP-style) учится, какой codebook важнее для конкретной recommendation task.
1. **Обучить downstream recommender.** Recommendation loss оптимизирует fusion и model weights end-to-end, но indexing stage уже задал capacity.
1. **Диагностировать масштабируемость.** Помимо Recall/NDCG проверяются discriminability, robustness к малым dimensions, occupancy и redundancy между codebooks.

```
for item in catalog:
    x[item] = LLM_encoder(item.content)  # high-dimensional vector

multi_codes = train_multi_codebook_VQVAE(x, num_codebooks=C)
for item in catalog:
    codes[item] = [quantize_c(x[item], codebook_c) for c in range(C)]

for interaction in logs:
    code_vectors = [embedding_table[c][codes[item][c]] for c in range(C)]
    item_repr = mixture_of_codes(code_vectors)
    score = recommender(user_repr, item_repr)
    update(recommendation_loss(score, label))

monitor occupancy_per_codebook, redundancy_between_codebooks, Recall/NDCG
```

## эксперименты

Эксперименты в статье построены вокруг сравнения с классическими sequential recommenders, TIGER-like GR baselines, tokenizer variants или scaling baselines. Ниже перечислена конкретика setup.

При чтении результатов полезно проверять, совпадает ли inference format у baseline и нового метода: разные beam constraints, token lengths и collision handling могут сильно менять сравнение.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>Datasets</td><td>three public recommendation datasets</td></tr>
<tr><td>Baselines</td><td>MLP projection, single-code VQ, Multi-Embedding, RQ-VAE style approaches</td></tr>
<tr><td>Diagnostics</td><td>discriminability and dimension robustness</td></tr>
<tr><td>Core result</td><td>MoC scales better as semantic representation capacity grows</td></tr>
<tr><td>Deployment angle</td><td>more capacity without using full LLM embeddings online</td></tr>
</table>
</div>

## рисунки/таблицы

Рисунки и таблицы в статье полезны как operational checklist: они показывают, какие компоненты надо воспроизводить, а какие являются ablation-only.

Если статья недоступна как production code, именно captions и ablation tables часто дают лучшие подсказки для повторной реализации.

- Scalability analysis figure: baseline saturation.
- MoC architecture figure: multiple codebooks plus fusion.
- Performance tables across datasets.
- Dimension robustness plots.
- Discriminability comparisons.

## сильные стороны

Ниже — основные инженерные плюсы.

- **Явно формулирует capacity bottleneck.** Работа показывает, что single semantic embedding/code быстро насыщается при сжатии LLM semantics.
- **Чисто разделяет indexing и fusion.** Multi-Codebooks VQ-VAE отвечает за representation capacity, Mixture-of-Codes - за downstream usage.
- **Практична для малых embedding dimensions.** Метод рассчитан на recommender ограничения памяти, а не на полный online LLM embedding.
- **Есть диагностики вне Recall/NDCG.** Discriminability и dimension robustness помогают понять, сохраняет ли MoC структуру LLM space.
- **Можно менять fusion независимо.** Sum, attention или MLP fusion можно тестировать без переобучения всего item encoder-а.

## ограничения

Для каждого нового домена нужен отдельный audit: taxonomy, item text quality, freshness и user behavior distribution могут полностью изменить картину.

- **Independent codebooks могут дублировать информацию.** Без redundancy regularization несколько codebooks могут выучить почти один и тот же aspect.
- **Это не готовый autoregressive SID language.** Несколько параллельных codes удобны для representation layer, но не обязательно для sequential token decoding.
- **Lookup cost растет с числом codebooks.** Память меньше, чем у LLM vectors, но больше, чем у одного ID embedding.
- **Качество завязано на upstream LLM embeddings.** Если LLM space плохо отражает collaborative relevance, MoC сохранит не тот signal.
- **Disentanglement не гарантирован.** Нельзя автоматически считать codebooks "аспектами" без occupancy/redundancy analysis.

## как реализовать/проверять

Практический путь — начинать с сильного baseline и добавлять новый механизм как isolated intervention. Нельзя менять tokenizer, backbone, beam search и preprocessing одновременно, иначе lift невозможно интерпретировать.

Ниже — минимальный набор проверок перед доверием к результату.

- Measure mutual redundancy between codebooks.
- Try sum, attention and MLP fusion.
- Track memory overhead per item.
- Run robustness tests at multiple embedding dimensions.
- Compare against simply increasing embedding dimension.
- Audit codebook occupancy separately for each codebook.

## связь

Эта работа связана с соседними подходами тем, что пытается уменьшить разрыв между rich item semantics и компактным recommender-friendly представлением.

MoC connects to LAMIA and MTGRec through multi-identifier thinking: one item should not be forced into a single narrow code.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>CoST</td><td>retrieval-aware tokenizer loss</td></tr>
<tr><td>ETEGRec</td><td>end-to-end alignment tokenizer/recommender</td></tr>
<tr><td>MTGRec</td><td>multiple identifiers as pretraining augmentation</td></tr>
<tr><td>MoC/LAMIA</td><td>parallel/multi-aspect semantic representation</td></tr>
<tr><td>Scaling-view</td><td>diagnostics of SID capacity bottleneck</td></tr>
</table>
</div>

## итог

MoC is most compelling for ranker/retriever representation layers where semantic capacity matters more than strict single-sequence decoding.

Ключевая рекомендация: воспроизводить не только top-line metric, но и diagnostic metrics по кодам, collision, distribution, head/tail и latency.

- Хороший кандидат для controlled offline reproduction.
- Требует versioned SID/token maps при production использовании.
- Нужны ablations по каждому заявленному компоненту.
- Нужно проверять head/tail и cold-start отдельно.
- Нужно явно считать training и inference cost.
