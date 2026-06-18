---
title: "Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion"
category: "semantic_ids_tokenization_indexing"
slug: "generate_what_you_prefer_reshaping_sequential_recommendation_via_summary"
catalogId: "paper-generate_what_you_prefer_reshaping_sequential_recommendation_via_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/generate_what_you_prefer_reshaping_sequential_recommendation_via_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2310.20453"
---
> **Авторы:** Zhengyi Yang, Jiancan Wu, Zhicai Wang, Yancheng Yuan, Xiang Wang, Xiangnan He.
>
> **Аффилиации:** University of Science and Technology of China; Hong Kong Polytechnic University; без индустриальной аффилиации в первичном источнике.
>
> **Первичный источник:** arXiv:2310.20453.

## коротко

DreamRec uses guided diffusion to generate a continuous oracle item representation from user history, avoiding negative sampling classification.

Ниже разбор сфокусирован на механике метода, objectives, экспериментальном setup, ablation conclusions и практических рисках внедрения.

- Reframes sequential recommendation as learning-to-generate.
- Transformer encoder provides history guidance.
- Forward process noises target item embedding.
- Reverse process denoises Gaussian noise into oracle item.
- Retrieval then finds nearest real items to generated oracle.

## контекст

Работа находится в линии generative recommendation / semantic identifiers, где item представляется не только atomic ID, а дискретным, текстовым, многокодовым или генерируемым представлением.

Общий контекст: чем сильнее сжатие item semantics в короткий код, тем больше риск потерять информацию, которую downstream recommender уже не восстановит.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>Paradigm</td><td>continuous generative recommendation</td></tr>
<tr><td>Avoided component</td><td>negative sampling during training</td></tr>
<tr><td>Inference</td><td>nearest-neighbor search over item embeddings</td></tr>
</table>
</div>

## проблема

Learning-to-classify with sampled negatives may learn a crude decision boundary and can dilute preference signals with easy or false negatives.

Для оценки важно отделять три уровня: качество item representation, качество tokenizer/identifier и качество sequence/generative model. Многие papers выигрывают именно за счет улучшения одного уровня, а не всей системы сразу.

- Tokenizer/recommender mismatch: код удобен одному модулю и неудобен другому.
- Collisions или near-collisions ухудшают exact item retrieval.
- Long-tail items получают слабый сигнал и нестабильные IDs.
- Beam search может генерировать invalid или duplicate identifiers.
- Offline lift может не перенестись в production из-за drift, catalog churn и constraints.

## метод/архитектура

Метод вводит специальные компоненты поверх базового recommender/tokenizer pipeline. Важно, что авторы обычно стараются сохранить inference совместимым с существующей GR схемой или явно обсуждают trade-off inference cost.

Архитектурная идея раскрывается через следующие элементы.

- Transformer history encoder.
- DDPM-style forward noising of target item representation.
- Conditional denoising network guided by history.
- Generated oracle item vector.
- Nearest-neighbor matching to concrete item catalog.

## objective/алгоритм

Objective связывает representation learning, identifier learning и downstream recommendation/retrieval signal. В некоторых работах это explicit loss, в других — training schedule или iterative relabeling.

Для практической реализации важно логировать каждый компонент loss отдельно: общий Recall/NDCG не объясняет, какой механизм сработал.

- Diffusion variational/denoising objective.
- Noise prediction or clean representation reconstruction per timestep.
- History-conditioned reverse process.
- No sampled negative classification loss.
- Similarity retrieval objective at inference.

### Подробная схема алгоритма DreamRec

DreamRec не строит discrete SID. Его алгоритм полезен как контраст к SID-подходам: генератор создает continuous oracle vector следующего item'а, а конкретный catalog item выбирается nearest-neighbor search по embedding table.

1. **Подготовить item embedding space.** Для каждого item хранится вектор $\mathbf{e}_i$. Качество всего метода ограничено тем, насколько это пространство отражает заменяемость и последовательные предпочтения.
1. **Закодировать историю.** Transformer encoder получает последовательность прошлых item'ов и строит condition vector $\mathbf{h}_u$, описывающий текущий user intent.
1. **Forward diffusion.** Target embedding следующего item'а $\mathbf{e}_{t+1}$ постепенно зашумляется до Gaussian noise на случайном timestep $s$.
1. **Conditional denoising.** Denoising network получает noisy vector, timestep и $\mathbf{h}_u$, затем предсказывает noise или clean embedding. Loss учит восстанавливать target representation без sampled negatives.
1. **Inference из шума.** Модель начинает с Gaussian vector и делает несколько reverse diffusion шагов, каждый раз conditioning on user history. На выходе получается oracle item vector $\hat{\mathbf{e}}$.
1. **Catalog projection.** $\hat{\mathbf{e}}$ не обязан совпадать с реальным item. Поэтому финальный candidate set строится через ANN/top-K nearest neighbors в item embedding table.

```
for user_sequence in training:
    history, target_item = split_last(user_sequence)
    h = transformer_encoder(history)
    x0 = item_embedding[target_item]

    s = sample_diffusion_step()
    noise = normal_like(x0)
    xs = sqrt(alpha_bar[s]) * x0 + sqrt(1 - alpha_bar[s]) * noise

    predicted = denoiser(xs, step=s, condition=h)
    loss = denoising_loss(predicted, noise_or_x0)
    update(transformer_encoder, denoiser)

for request in serving:
    h = transformer_encoder(request.history)
    x = gaussian_noise()
    for s in reversed(diffusion_steps):
        x = denoise_step(x, s, condition=h)
    candidates = ANN_search(item_embedding_table, query=x, top_k=K)
```

## эксперименты

Эксперименты в статье построены вокруг сравнения с классическими sequential recommenders, TIGER-like GR baselines, tokenizer variants или scaling baselines. Ниже перечислена конкретика setup.

При чтении результатов полезно проверять, совпадает ли inference format у baseline и нового метода: разные beam constraints, token lengths и collision handling могут сильно менять сравнение.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>Datasets</td><td>three benchmark sequential recommendation datasets</td></tr>
<tr><td>Baselines</td><td>GRU4Rec, Caser, SASRec, BERT4Rec and auxiliary-task methods</td></tr>
<tr><td>Metrics</td><td>Recall and NDCG</td></tr>
<tr><td>Ablations</td><td>history guidance, diffusion configuration, embedding choices</td></tr>
<tr><td>Code</td><td>paper states code/data are open-sourced at DreamRec repository</td></tr>
</table>
</div>

## рисунки/таблицы

Рисунки и таблицы в статье полезны как operational checklist: они показывают, какие компоненты надо воспроизводить, а какие являются ablation-only.

Если статья недоступна как production code, именно captions и ablation tables часто дают лучшие подсказки для повторной реализации.

- Figure 1: classify paradigm versus generate oracle paradigm.
- Diffusion process diagram.
- Performance comparison tables.
- Ablation of guidance and diffusion steps.
- Analysis of negative-sampling-free preference modeling.

## сильные стороны

Сильные стороны DreamRec в том, что он меняет саму форму обучения sequential recommender.

- **Убирает sampled-negative classification.** Модель не тратит обучение на easy negatives и не страдает от false negatives в sampled softmax setting.
- **Генерирует preference target в continuous space.** Oracle vector может описывать направление интереса пользователя до привязки к конкретному item, что полезно при плотном embedding space.
- **Простой смысл inference.** Reverse diffusion создает запрос к ANN index; это проще отлаживать, чем invalid SID sequences и constrained decoding.
- **Хороший counterfactual baseline для SID работ.** Если continuous generation выигрывает у SID, значит discrete compression действительно теряет полезную информацию.
- **Можно использовать сильные pretrained item embeddings.** Метод напрямую выигрывает от улучшения content/collaborative representation без смены decoding vocabulary.

## ограничения

Ограничения DreamRec в первую очередь связаны с тем, что continuous oracle vector нужно проецировать обратно на дискретный каталог.

- **ANN остается обязательным serving компонентом.** DreamRec не заменяет retrieval index генерацией валидного ID; он генерирует vector query к item table.
- **Latency зависит от числа denoising steps.** Greedy one-pass decoder SID-системы могут быть быстрее, если diffusion требует много reverse steps.
- **Oracle vector может попасть между item'ами.** Ближайший сосед не обязательно является лучшим recommendation, если embedding geometry плохо калибрована.
- **Нет компактного semantic identifier artifact.** Метод не дает token map, trie, prefix analysis или code utilization diagnostics.
- **Качество полностью завязано на embedding space.** Если item embeddings переобучены на popularity или плохо отражают substitutability, diffusion будет красиво генерировать неправильное пространство.

## как реализовать/проверять

Практический путь — начинать с сильного baseline и добавлять новый механизм как isolated intervention. Нельзя менять tokenizer, backbone, beam search и preprocessing одновременно, иначе lift невозможно интерпретировать.

Ниже — минимальный набор проверок перед доверием к результату.

- Build strong item embedding table first.
- Tune diffusion steps for latency/quality.
- Compare against sampled softmax baselines under equal candidates.
- Measure nearest-neighbor recall and oracle-vector calibration.
- Check whether generated vectors collapse to popular regions.
- Evaluate cold-start separately.

## связь

Эта работа связана с соседними подходами тем, что пытается уменьшить разрыв между rich item semantics и компактным recommender-friendly представлением.

DreamRec is a counterpoint to SID-based GR: it is generative but not discrete-token generative.

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

DreamRec is attractive if ANN serving is acceptable and negative sampling is the main pain; it is less aligned with systems trying to eliminate ANN via semantic token decoding.

Ключевая рекомендация: воспроизводить не только top-line metric, но и diagnostic metrics по кодам, collision, distribution, head/tail и latency.

- Хороший кандидат для controlled offline reproduction.
- Требует versioned SID/token maps при production использовании.
- Нужны ablations по каждому заявленному компоненту.
- Нужно проверять head/tail и cold-start отдельно.
- Нужно явно считать training и inference cost.
