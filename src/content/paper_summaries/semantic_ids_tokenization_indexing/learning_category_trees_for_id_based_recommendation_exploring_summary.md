---
title: "Learning Category Trees for ID-Based Recommendation: Exploring the Power of Differentiable Vector Quantization"
category: "semantic_ids_tokenization_indexing"
slug: "learning_category_trees_for_id_based_recommendation_exploring_summary"
catalogId: "paper-learning_category_trees_for_id_based_recommendation_exploring_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/learning_category_trees_for_id_based_recommendation_exploring_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2308.16761"
---
> **Авторы:** Qijiong Liu, Lu Fan, Jiaren Xiao, Jieming Zhu, Xiao-Ming Wu.
>
> **Аффилиации:** Hong Kong Polytechnic University; Huawei Noah's Ark Lab.
>
> **Первичный источник:** arXiv:2308.16761; DOI 10.1145/3589334.3645484.

## коротко

CAGE учит category trees для ID-only recommendation через cascaded differentiable vector quantization, без item text или готовых категорий.

Ниже разбор сфокусирован на механике метода, objectives, экспериментальном setup, ablation conclusions и практических рисках внедрения.

- Работает как pluggable module для sequential и non-sequential recommenders.
- Строит fine-to-coarse category path через cascaded codebooks.
- Использует STE для nearest-neighbor quantization.
- Проверен на list completion, CF и CTR prediction.
- Заявлен lift до 21.41% over SOTA в list completion scenarios.

## контекст

Работа находится в линии generative recommendation / semantic identifiers, где item представляется не только atomic ID, а дискретным, текстовым, многокодовым или генерируемым представлением.

Общий контекст: чем сильнее сжатие item semantics в короткий код, тем больше риск потерять информацию, которую downstream recommender уже не восстановит.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>Задачи</td><td>list completion, collaborative filtering, CTR prediction</td></tr>
<tr><td>Данные</td><td>Zhihu, Spotify, Goodreads, Amazon subsets, MovieLens, MIND</td></tr>
<tr><td>Выход</td><td>category-aware entity embedding, не generative SID</td></tr>
</table>
</div>

## проблема

В ID-only данных нет надежной category information; обычная VQ требует meaningful embeddings, но ID embeddings в начале обучения случайны.

Для оценки важно отделять три уровня: качество item representation, качество tokenizer/identifier и качество sequence/generative model. Многие papers выигрывают именно за счет улучшения одного уровня, а не всей системы сразу.

- Tokenizer/recommender mismatch: код удобен одному модулю и неудобен другому.
- Collisions или near-collisions ухудшают exact item retrieval.
- Long-tail items получают слабый сигнал и нестабильные IDs.
- Beam search может генерировать invalid или duplicate identifiers.
- Offline lift может не перенестись в production из-за drift, catalog churn и constraints.

## метод/архитектура

Метод вводит специальные компоненты поверх базового recommender/tokenizer pipeline. Важно, что авторы обычно стараются сохранить inference совместимым с существующей GR схемой или явно обсуждают trade-off inference cost.

Архитектурная идея раскрывается через следующие элементы.

- Cascaded quantizers Q(1)...Q(H) с decreasing granularity.
- Nearest neighbor links between entity/codebook layers.
- Code fusion: average pooling multi-level codes.
- Residual connection: z = e + alpha c_bar.
- Tree backpropagation через straight-through estimator.
- Separate user CAGE and item CAGE для non-sequential settings.

## objective/алгоритм

Objective связывает representation learning, identifier learning и downstream recommendation/retrieval signal. В некоторых работах это explicit loss, в других — training schedule или iterative relabeling.

Для практической реализации важно логировать каждый компонент loss отдельно: общий Recall/NDCG не объясняет, какой механизм сработал.

- Quantization loss: sum ||sg[c_prev] - c_i||^2.
- Commitment loss: sum ||c_prev - sg[c_i]||^2.
- CAGE loss: L_quant + beta L_commit.
- Recommendation loss берется из downstream model.
- Total training co-evolves entity embeddings and codebooks.

### Подробная схема алгоритма CAGE

CAGE не генерирует item ID в LLM. Он добавляет к обычным ID embeddings обучаемое category-tree представление, которое строится через каскадные VQ codebooks и обновляется вместе с downstream recommender'ом.

1. **Старт с entity embedding.** Для item или user берется текущий embedding $\mathbf{e}$ из базовой модели. В начале обучения он может быть почти случайным, поэтому дерево нельзя построить один раз offline.
1. **Каскадное квантование.** На первом уровне nearest codeword выбирается из $Q^{(1)}$; на следующем уровне квантуется representation предыдущего уровня. Так получается path от fine к coarse или наоборот, в зависимости от реализации layer ordering.
1. **Straight-through update.** Forward pass использует hard nearest-neighbor code, а backward pass пропускает градиент к entity embedding/codebook через STE.
1. **Собрать category representation.** Коды нескольких уровней усредняются или агрегируются в $\bar{\mathbf{c}}$, после чего смешиваются с исходным ID embedding через residual connection $\mathbf{z} = \mathbf{e} + \alpha \bar{\mathbf{c}}$.
1. **Передать в downstream model.** SASRec/CF/CTR backbone получает enhanced embedding $\mathbf{z}$ и оптимизирует свой обычный loss.
1. **Обновить tree через CAGE loss.** Quantization и commitment losses удерживают entity embeddings и codewords согласованными, пока recommendation loss сдвигает дерево к полезной для задачи структуре.

```
for batch in recommender_training:
    e = embedding_lookup(batch.entities)
    previous = e
    selected_codes = []

    for level in 1..H:
        c = nearest_codeword(previous, codebook[level])
        selected_codes.append(c)
        loss_quant += ||stopgrad(previous) - c||^2
        loss_commit += ||previous - stopgrad(c)||^2
        previous = straight_through(c, previous)

    c_bar = average(selected_codes)
    z = e + alpha * c_bar
    prediction = downstream_model(z, batch.context)
    loss = recommendation_loss(prediction, batch.labels) + omega * (loss_quant + beta * loss_commit)
    update(embeddings, codebooks, downstream_model)
```

## эксперименты

Эксперименты в статье построены вокруг сравнения с классическими sequential recommenders, TIGER-like GR baselines, tokenizer variants или scaling baselines. Ниже перечислена конкретика setup.

При чтении результатов полезно проверять, совпадает ли inference format у baseline и нового метода: разные beam constraints, token lengths и collision handling могут сильно менять сравнение.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>List completion</td><td>Zhihu, Spotify, Goodreads; baselines Caser, GRU4Rec, SASRec, BERT4Rec, CAR, FANS</td></tr>
<tr><td>CF</td><td>Amazon Toys, Kindle, Phones, Grocery, MovieLens 1M; BPRMF, NeuMF, CFKG, LightGCN</td></tr>
<tr><td>CTR</td><td>MIND small, MovieLens 100K; DeepFM, DCN, FiBiNET, FinalMLP</td></tr>
<tr><td>Metrics</td><td>NDCG@5/10, HR@5/10 and CTR metrics</td></tr>
<tr><td>Ablation</td><td>number of layers, entries, user/item CAGE, alpha/beta/omega weights</td></tr>
</table>
</div>

## рисунки/таблицы

Рисунки и таблицы в статье полезны как operational checklist: они показывают, какие компоненты надо воспроизводить, а какие являются ablation-only.

Если статья недоступна как production code, именно captions и ablation tables часто дают лучшие подсказки для повторной реализации.

- Illustration figure: learned category tree from ID embeddings.
- VQ comparison figure: three-stage VQ versus end-to-end CAGE.
- Completion/CF/CTR tables: broad validation.
- Layer/entry table: two-level trees often outperform one-level variants.
- Hyperparameter plots: omega_q and beta are especially important.

## сильные стороны

Сильные стороны CAGE связаны с тем, что он извлекает дискретную структуру даже из ID-only данных.

- **Не требует текстов, картинок или готовых категорий.** Это важно для доменов, где side content бедный, приватный или недоступный.
- **Дерево обучается вместе с recommender'ом.** В отличие от offline clustering, category structure может перестраиваться под list completion, CF или CTR objective.
- **Модуль легко встроить.** Residual fusion $e + \alpha c$ позволяет добавить CAGE поверх существующих embedding lookups без переписывания всего backbone.
- **Работает в нескольких family models.** Проверка на sequential, CF и CTR снижает риск, что метод является трюком одного benchmark'а.
- **Помогает sparse entity representations.** Shared codewords дают редким items/users дополнительный сигнал через соседей в learned tree.

## ограничения

Ограничения CAGE возникают из-за того, что learned categories полезны для модели, но не обязательно стабильны или интерпретируемы.

- **Категории могут быть неинтерпретируемыми.** Даже если они улучшают CTR/NDCG, это не значит, что tree можно показывать analyst'ам как taxonomy.
- **Стабильность taxonomy не гарантирована.** При retraining один и тот же item может сменить path, что усложняет monitoring и сравнение версий.
- **Гиперпараметры критичны.** Число уровней $H$, размер codebook'ов, $\alpha$, $\beta$ и вес CAGE loss меняют trade-off memorization vs sharing.
- **Cold-start без interactions не решен.** Если item вообще не имеет поведенческого сигнала и нет content features, CAGE неоткуда вывести осмысленный code path.
- **Это не полноценный generative retrieval method.** CAGE строит representation augmentation, но не решает constrained decoding, valid ID generation и beam search.

## как реализовать/проверять

Практический путь — начинать с сильного baseline и добавлять новый механизм как isolated intervention. Нельзя менять tokenizer, backbone, beam search и preprocessing одновременно, иначе lift невозможно интерпретировать.

Ниже — минимальный набор проверок перед доверием к результату.

- Wrap item/user embedding lookup with CAGE module.
- Track codebook occupancy by layer.
- Measure assignment churn across epochs.
- Run ablation one-layer versus two/three-layer.
- Check whether learned categories correlate with known labels where available.
- Tune alpha and beta per dataset.

## связь

Эта работа связана с соседними подходами тем, что пытается уменьшить разрыв между rich item semantics и компактным recommender-friendly представлением.

CAGE is a predecessor to semantic ID thinking: it shows differentiable VQ can create useful discrete structure even from IDs.

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

CAGE is best read as a representation augmentation paper, not a full GR paper. It is valuable because it removes the assumption that side content must exist before discrete hierarchy can help.

Ключевая рекомендация: воспроизводить не только top-line metric, но и diagnostic metrics по кодам, collision, distribution, head/tail и latency.

- Хороший кандидат для controlled offline reproduction.
- Требует versioned SID/token maps при production использовании.
- Нужны ablations по каждому заявленному компоненту.
- Нужно проверять head/tail и cold-start отдельно.
- Нужно явно считать training и inference cost.
