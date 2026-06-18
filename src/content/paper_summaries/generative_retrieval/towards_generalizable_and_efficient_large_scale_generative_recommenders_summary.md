---
title: "Towards Generalizable and Efficient Large-Scale Generative Recommenders"
category: "generative_retrieval"
slug: "towards_generalizable_and_efficient_large_scale_generative_recommenders_summary"
catalogId: "paper-towards_generalizable_and_efficient_large_scale_generative_recommenders_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/towards_generalizable_and_efficient_large_scale_generative_recommenders_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.23312"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Qiuling Xu, Ko-Jen Hsiao, Moumita Bhattacharya.
>
> **Аффилиации:** Netflix Research.
>
> **Источник:** [arXiv 2605.23312](https://arxiv.org/abs/2605.23312) · дата metadata: 2026-05-22.
>
> **Категория/теги:** generative recommendation, efficiency, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: Netflix case study о масштабировании GR backbone от 2M до 1B parameters в production-scale title recommendation.
- Алгоритм: Авторы используют offset scaling-law diagnostics, sampled softmax/projected decoding head, multi-token prediction for cached serving и semantic item towers с collaborative-embedding masking для cold start.
- Evidence: В one-week production-shadow evaluation на 1M users 1B backbone улучшает MRR на всех reported tasks, включая +22.5% на сложной Task A.
- Ограничение: Это закрытый production setup; absolute metrics и task details ограничены, а scale сам по себе не гарантирует transfer.
- Итог: Сильная инженерная статья: масштаб модели нужно читать вместе с headroom, retraining cost, serving latency и item freshness.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>In production, however, pre-training gains do not automatically translate into downstream application improvements: task headroom, repeated-training cost, serving latency, and item freshness all affect transfer.</td></tr>
<tr><th>Предложение авторов</th><td>Generative recommendation models can model user behavior as sequences of events and provide a shared backbone for multiple recommendation tasks.</td></tr>
<tr><th>Главный evidence claim</th><td>In production, however, pre-training gains do not automatically translate into downstream application improvements: task headroom, repeated-training cost, serving latency, and item freshness all affect transfer.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>cross-entropy, masking, softmax, sampled softmax</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Netflix</td></tr>
<tr><th>Metrics</th><td>NDCG, Hit, MRR, CTR, latency, MAP</td></tr>
<tr><th>Baselines</th><td>TIGER, CoST, HSTU</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Efficient Training and Inference: Efficiency is one of the main differences between a promising large recommender and an operational one. Unlike many language models, recommender models must be refreshed frequently to reflect changing preferences, seasonal effects, and catalog updates. This repeated-training requirement makes efficiency central even when vocabulary pressure is reduced...
1. **Ключевой модуль.** Efficient Training and Inference: Our pre-training datasets comprise 2 trillion behavior tokens per cycle--on par with major LLM pre-training corpora, but processed on a repeated refresh cadence. Alongside standard distributed-training optimizations, we make additional changes to encoding and decoding. Below, we focus on advances in decoding efficiency.
1. **Learning signal.** The Latency Issue in Next-Token Prediction: Large recommenders are rarely served in a purely synchronous way. To control latency and cost, many production systems cache user embeddings, candidate lists, or pre-ranked results, refreshing them after a fixed time interval or after sufficient new activity. This pattern is related to deferred retrieval systems such as RADAR jaspal2025radar, but it...

## 5. Objectives, formulas и training details

**Detected objective keywords:** cross-entropy, masking, softmax, sampled softmax.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `L = - _y_i Y w_i p_ (y_i x), w_i = r_i (- (2) t_i-t_ context).`
- `aligned z_i &= _ sem(e_i^ graph, e_i^ lang, e_i^ ann), \\ e_i^ ID &= cases e_i^ ID, & i V, \\ e^ OOV, & i V, cases \\ v_i &= _ (e_i^ ID, z_i), \\ s(u, i) &= g_ (h_u)^ v_i. aligned`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Scaling-law fits for three anonymized recommendation task categories: Task A captures lower-predictability long-horizon taste, Task B captures moderate-predictability short-horizon engagement, and Task C captures higher-predictability time- or availability-driven behavior. The x-axis counts backbone parameters, excluding embedding and decoding layers;...
- Task categories used for scaling-law analysis.
- Fit-error comparison for scaling-law models.
- Estimated training FLOPs per training token for a 6-layer transformer with hidden dimension 1024 and sequence length 512. Values include a fixed backbone term plus vocabulary-dependent decoding cost; lines compare full decoding with sampled-softmax and projected-head variants as the output vocabulary grows.
- Latency mismatch between next-token training and delayed cached serving. Title A is the immediate next event used as the training label, but after cached serving delay it may already be consumed; title B can then become the relevant serving-time target.
- Relative MRR degradation as cached outputs become stale. Delays are simulated by replaying evaluation at increasing serving horizons; task definitions follow Figure and
- MTP comparison across serving scenarios. Bars report relative MRR changes for different future-target window sizes under online serving with p95 latency below one second and cached serving with a 48-hour horizon; task labels use the taxonomy in
- Shared semantic title metadata for encoder events and decoder title representations. Blue denotes learned representations and computation, red denotes input-side event context, and green denotes reusable title metadata from knowledge-graph message passing, LLM2Vec language embeddings, and human annotation features.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Netflix</td></tr>
<tr><th>Metrics</th><td>NDCG, Hit, MRR, CTR, latency, MAP</td></tr>
<tr><th>Baselines</th><td>TIGER, CoST, HSTU</td></tr>
</tbody></table>
</div>

- Главная идея: Netflix case study о масштабировании GR backbone от 2M до 1B parameters в production-scale title recommendation.
- Evidence: В one-week production-shadow evaluation на 1M users 1B backbone улучшает MRR на всех reported tasks, включая +22.5% на сложной Task A.
- In production, however, pre-training gains do not automatically translate into downstream application improvements: task headroom, repeated-training cost, serving latency, and item freshness all affect transfer.
- We describe our experience scaling a generative recommender from 2M to 1B backbone parameters, excluding embedding and decoding layers, in a production-scale title recommendation setting.
- Across multiple downstream tasks, we observe task-dependent scaling behavior: some tasks approach an empirical ceiling within the observed scale range, while others continue to benefit from additional capacity.
- In a one-week production-shadow evaluation over 1M users, the 1B-backbone model achieves higher MRR than the 2M-backbone baseline across all reported tasks.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: Netflix case study о масштабировании GR backbone от 2M до 1B parameters в production-scale title recommendation.
- **Algorithmic/system:** Алгоритм: Авторы используют offset scaling-law diagnostics, sampled softmax/projected decoding head, multi-token prediction for cached serving и semantic item towers с collaborative-embedding masking для cold start.
- **Empirical:** Evidence: В one-week production-shadow evaluation на 1M users 1B backbone улучшает MRR на всех reported tasks, включая +22.5% на сложной Task A.
- **Practical:** Ограничение: Это закрытый production setup; absolute metrics и task details ограничены, а scale сам по себе не гарантирует transfer.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Netflix.
- Метрики: NDCG, Hit, MRR, CTR, latency, MAP.
- Baselines и parity settings: TIGER, CoST, HSTU.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Сильная инженерная статья: масштаб модели нужно читать вместе с headroom, retraining cost, serving latency и item freshness.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: Netflix case study о масштабировании GR backbone от 2M до 1B parameters в production-scale title recommendation.
- Алгоритм: Авторы используют offset scaling-law diagnostics, sampled softmax/projected decoding head, multi-token prediction for cached serving и semantic item towers с collaborative-embedding masking для cold start.
- Evidence: В one-week production-shadow evaluation на 1M users 1B backbone улучшает MRR на всех reported tasks, включая +22.5% на сложной Task A.
- Ограничение: Это закрытый production setup; absolute metrics и task details ограничены, а scale сам по себе не гарантирует transfer.
- Итог: Сильная инженерная статья: масштаб модели нужно читать вместе с headroom, retraining cost, serving latency и item freshness.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Generative recommendation models extend the sequence-modeling paradigm of language models to user behavior: a user's history is represented as a sequence of events, and the model is trained to predict future events. This formulation is appealing for...</td></tr>
<tr><td>Related Work</td><td>Generative recommenders. Generative retrieval casts recommendation as sequence generation over item or semantic identifiers, providing an alternative to nearest-neighbor retrieval over fixed item embeddings. TIGER introduced this formulation with semantic IDs...</td></tr>
<tr><td>Scaling Law in Recommendation</td><td>We first ask whether larger generative recommenders improve uniformly across downstream tasks. To reduce temporal distribution shift, we group user activities into three broad anonymized task categories that differ in predictability and serving sensitivity....</td></tr>
<tr><td>Efficient Training and Inference</td><td>Efficiency is one of the main differences between a promising large recommender and an operational one. Unlike many language models, recommender models must be refreshed frequently to reflect changing preferences, seasonal effects, and catalog updates. This...</td></tr>
<tr><td>Efficient Decoding</td><td>Even when SID or retrieval-based methods reduce the effective output space, direct item scoring remains a common and useful interface for recommendation models. For a vanilla transformer, this output layer can become a large part of training cost as the...</td></tr>
<tr><td>Efficient Encoding</td><td>Our event representation also differs from architectures that separately model heterogeneous action and feature streams. We compress the action, observation context, and item metadata for each user event into a single token-level representation. This choice...</td></tr>
<tr><td>Efficient Development</td><td>Finally, we use a small-to-large experiment funnel. Candidate changes are first evaluated on smaller backbones and scaled test sets designed to detect task-specific regressions. Only changes that pass this funnel are integrated into expensive...</td></tr>
<tr><td>Multi-Token Prediction (MTP)</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>The Latency Issue in Next-Token Prediction</td><td>Large recommenders are rarely served in a purely synchronous way. To control latency and cost, many production systems cache user embeddings, candidate lists, or pre-ranked results, refreshing them after a fixed time interval or after sufficient new activity....</td></tr>
<tr><td>Set-Valued Future Targets</td><td>A second mismatch is that many recommendation targets are set-like. In language modeling, token order is usually semantic; in recommendation, several future items may be similarly valid and their order may be partly arbitrary. If a user is likely to watch...</td></tr>
<tr><td>Unseen and Cold-Start Titles</td><td>Cold-start titles expose a failure mode that cannot be addressed by scaling alone. Mature titles can be represented by collaborative ID embeddings learned from interaction histories, but new or sparse titles provide little evidence for that channel. Prior...</td></tr>
<tr><td>Production-Shadow Evaluation</td><td>текст не извлечен; смотреть PDF/source</td></tr>
</tbody></table>
</div>
