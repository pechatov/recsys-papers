---
title: "Pre-training Generative Recommender with Multi-Identifier Item Tokenization"
category: "semantic_ids_tokenization_indexing"
slug: "mtgrec_multi_identifier_item_tokenization_summary"
catalogId: "paper-mtgrec_multi_identifier_item_tokenization_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/mtgrec_multi_identifier_item_tokenization_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2504.04400"
---
> **Авторы:** Bowen Zheng, Enze Liu, Zhongfu Chen, Zhongrui Ma, Yue Wang, Wayne Xin Zhao, Ji-Rong Wen.
>
> **Аффилиации:** Renmin University of China; Huawei Poisson Lab.
>
> **Первичный источник:** arXiv:2504.04400.

## коротко

MTGRec gives each item multiple semantically related identifiers from adjacent RQ-VAE checkpoints and uses curriculum pretraining over the resulting tokenized sequence groups.

Ниже разбор сфокусирован на механике метода, objectives, экспериментальном setup, ablation conclusions и практических рисках внедрения.

- Breaks strict one-item-one-identifier training assumption.
- Uses adjacent RQ-VAE checkpoints instead of independent tokenizers.
- Turns one user sequence into multiple token sequences.
- Uses data influence estimation for curriculum sampling.
- Fine-tunes with single tokenizer for unambiguous inference.

## контекст

Работа находится в линии generative recommendation / semantic identifiers, где item представляется не только atomic ID, а дискретным, текстовым, многокодовым или генерируемым представлением.

Общий контекст: чем сильнее сжатие item semantics в короткий код, тем больше риск потерять информацию, которую downstream recommender уже не восстановит.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>Tokenizer</td><td>RQ-VAE / TIGER++-style enhanced tokenizer</td></tr>
<tr><td>Data augmentation</td><td>multiple identifiers per item</td></tr>
<tr><td>Curriculum</td><td>first-order gradient approximation of data influence</td></tr>
</table>
</div>

## проблема

A single identifier can be suboptimal for low-frequency items and creates low-diversity token sequence data. But naive multiple tokenizers may produce unrelated/noisy IDs.

Для оценки важно отделять три уровня: качество item representation, качество tokenizer/identifier и качество sequence/generative model. Многие papers выигрывают именно за счет улучшения одного уровня, а не всей системы сразу.

- Tokenizer/recommender mismatch: код удобен одному модулю и неудобен другому.
- Collisions или near-collisions ухудшают exact item retrieval.
- Long-tail items получают слабый сигнал и нестабильные IDs.
- Beam search может генерировать invalid или duplicate identifiers.
- Offline lift может не перенестись в production из-за drift, catalog churn и constraints.

## метод/архитектура

Метод вводит специальные компоненты поверх базового recommender/tokenizer pipeline. Важно, что авторы обычно стараются сохранить inference совместимым с существующей GR схемой или явно обсуждают trade-off inference cost.

Архитектурная идея раскрывается через следующие элементы.

- RQ-VAE backbone with 3 codebooks of size 256 plus extra collision codebook in implementation.
- Adjacent final checkpoints as semantically relevant tokenizers.
- Multiple tokenized datasets/groups from same interaction sequences.
- Influence estimator based on validation gradient and training gradient.
- Dynamic sampler over tokenizer data groups.
- Final fine-tuning on one tokenizer for serving.

## objective/алгоритм

Objective связывает representation learning, identifier learning и downstream recommendation/retrieval signal. В некоторых работах это explicit loss, в других — training schedule или iterative relabeling.

Для практической реализации важно логировать каждый компонент loss отдельно: общий Recall/NDCG не объясняет, какой механизм сработал.

- RQ-VAE loss: reconstruction plus residual quantization commitment terms.
- Generative recommender NLL over target tokens.
- Data influence estimates validation loss change using first-order Taylor approximation.
- Curriculum updates sampling probabilities of groups.
- Pretrain on hybrid data, then fine-tune single-tokenizer data.

## Детальный алгоритм MTGRec

MTGRec использует нестабильность последних checkpoints tokenizer-а как полезную augmentation: один item получает несколько близких semantic identifiers, но на inference система возвращается к одному выбранному tokenizer-у, чтобы не создавать ambiguous serving.

1. **Обучить enhanced RQ-VAE tokenizer.** В реализации используется RQ-VAE с 3 codebooks size 256 и дополнительным collision codebook; также упоминаются whitening, deeper MLP и EMA как TIGER++-style улучшения.
1. **Сохранить последние checkpoints.** Берутся adjacent final checkpoints, а не независимые tokenizer-ы. Такие checkpoints дают разные, но семантически близкие views item identity.
1. **Перекодировать каталог каждым checkpoint.** Для каждого item получается несколько SID variants. Одна и та же user history превращается в несколько tokenized sequence groups.
1. **Сделать pretraining dataset из groups.** Вместо one-item-one-identifier модель видит несколько допустимых token traces для одного поведения, что увеличивает diversity training signal.
1. **Оценивать influence group sampling.** Для группы считается first-order approximation: насколько gradient на этой группе должен изменить validation loss. Sampling probability увеличивается для групп с положительным влиянием.
1. **Pretrain generative recommender curriculum-style.** Модель обучается на hybrid groups, где curriculum sampler постепенно выбирает более полезные tokenizer variants.
1. **Fine-tune на одном tokenizer-е.** Перед serving модель дообучается на выбранной single-tokenizer разметке, чтобы decoder генерировал однозначные SIDs.
1. **На inference использовать стандартный SID lookup.** Multi-identifier эффект остается в весах модели, но внешняя система видит один item-token map.

```
tokenizer_checkpoints = train_RQVAE_and_save_last_n()
groups = []
for ckpt in tokenizer_checkpoints:
    sid_map = encode_catalog(ckpt, items)
    groups.append(tokenize_user_sequences(logs, sid_map))

for pretrain_step in range(T):
    validation_grad = grad(loss(model, validation_batch))
    group_scores = {}
    for group in groups:
        train_grad = grad(loss(model, sample(group)))
        group_scores[group] = - dot(validation_grad, train_grad)
    group_probs = normalize_positive_influence(group_scores)
    batch = sample_group(groups, group_probs)
    update_model(batch)

fine_tune(model, groups[selected_serving_tokenizer])
serve with selected SID map only
```

## эксперименты

Эксперименты в статье построены вокруг сравнения с классическими sequential recommenders, TIGER-like GR baselines, tokenizer variants или scaling baselines. Ниже перечислена конкретика setup.

При чтении результатов полезно проверять, совпадает ли inference format у baseline и нового метода: разные beam constraints, token lengths и collision handling могут сильно менять сравнение.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>Datasets</td><td>Amazon 2023 Musical Instruments, Industrial and Scientific, Video Games</td></tr>
<tr><td>Baselines</td><td>Caser, HGN, GRU4Rec, BERT4Rec, SASRec, FMLP-Rec, HSTU, TIGER, LETTER, TIGER++</td></tr>
<tr><td>Metrics</td><td>Recall@5/10 and NDCG@5/10</td></tr>
<tr><td>Implementation</td><td>4 GPUs, batch size 256 per GPU, 200 pretraining epochs</td></tr>
<tr><td>Ablation</td><td>multi-identifier and curriculum both contribute</td></tr>
</table>
</div>

## рисунки/таблицы

Рисунки и таблицы в статье полезны как operational checklist: они показывают, какие компоненты надо воспроизводить, а какие являются ablation-only.

Если статья недоступна как production code, именно captions и ablation tables часто дают лучшие подсказки для повторной реализации.

- Framework figure: adjacent checkpoints and curriculum sampler.
- Main result table across three Amazon datasets.
- Ablation table on Instrument and Scientific.
- Sensitivity to number of tokenizers n.
- Implementation section: whitening, deeper MLP, EMA as TIGER++ tokenizer enhancements.

## сильные стороны

Ниже — основные инженерные плюсы.

- **Дешевый источник multiple identifiers.** Adjacent checkpoints уже возникают при обучении RQ-VAE, поэтому не нужно тренировать несколько независимых tokenizer-ов.
- **Использует uncertainty tokenizer-а как сигнал.** Небольшие изменения assignment около convergence превращаются в augmentation для rare/unstable items.
- **Curriculum не считает все variants равными.** Influence estimator отбирает группы, которые лучше помогают validation objective.
- **Serving остается простым.** После fine-tuning используется один tokenizer, поэтому нет multi-map ambiguity при beam decoding.
- **Сильные baselines.** Сравнение с TIGER, LETTER и TIGER++ делает вывод про pretraining augmentation более полезным.

## ограничения

Для каждого нового домена нужен отдельный audit: taxonomy, item text quality, freshness и user behavior distribution могут полностью изменить картину.

- **Зависимость от checkpoint trajectory.** Если последние RQ-VAE checkpoints почти одинаковы, augmentation слабая; если слишком разные, identifiers становятся noisy.
- **Influence estimation стоит градиентов.** Curriculum требует дополнительных gradient computations и усложняет distributed training.
- **Эффект multiple identifiers не виден напрямую на inference.** После single-tokenizer fine-tuning улучшение остается только в параметрах модели, поэтому диагностика сложнее.
- **Pipeline длиннее обычного TIGER.** Нужно хранить несколько token maps, группы последовательностей, sampling state и selected serving map.
- **Versioning все равно критичен.** Fine-tuned checkpoint должен строго соответствовать финальному tokenizer-у; иначе beam outputs будут мапиться неверно.

## как реализовать/проверять

Практический путь — начинать с сильного baseline и добавлять новый механизм как isolated intervention. Нельзя менять tokenizer, backbone, beam search и preprocessing одновременно, иначе lift невозможно интерпретировать.

Ниже — минимальный набор проверок перед доверием к результату.

- Save final n RQ-VAE checkpoints.
- Verify adjacent checkpoint IDs are semantically close, not random.
- Create separate tokenized sequence groups.
- Implement influence-based sampler and compare uniform sampling.
- Run fine-tune with selected single tokenizer.
- Measure performance versus n and tokenizer checkpoint choice.

## связь

Эта работа связана с соседними подходами тем, что пытается уменьшить разрыв между rich item semantics и компактным recommender-friendly представлением.

MTGRec is related to self-improvement and MoC: all use more than one view of item identity, but MTGRec uses training-time checkpoint diversity.

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

MTGRec is a practical pretraining method: it increases token sequence diversity while preserving simple single-tokenizer inference.

Ключевая рекомендация: воспроизводить не только top-line metric, но и diagnostic metrics по кодам, collision, distribution, head/tail и latency.

- Хороший кандидат для controlled offline reproduction.
- Требует versioned SID/token maps при production использовании.
- Нужны ablations по каждому заявленному компоненту.
- Нужно проверять head/tail и cold-start отдельно.
- Нужно явно считать training и inference cost.
