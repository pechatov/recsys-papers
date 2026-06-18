---
title: "Efficient Dataset Selection for Continual Adaptation of Generative Recommenders"
category: "generative_retrieval"
slug: "efficient_dataset_selection_for_continual_adaptation_of_generative_recommenders_summary"
catalogId: "paper-efficient_dataset_selection_for_continual_adaptation_of_generative_recommenders_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/efficient_dataset_selection_for_continual_adaptation_of_generative_recommenders_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.07739"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Cathy Jiao, Juan Elenter, Praveen Ravichandran, Bernd Huber, Joseph Cauteruccio, Todd Wasson, Timothy Heath, Chenyan Xiong, Mounia Lalmas, Paul Bennett.
>
> **Аффилиации:** Spotify; Carnegie Mellon University.
>
> **Источник:** [arXiv 2604.07739](https://arxiv.org/abs/2604.07739) · дата metadata: 2026-04-09.
>
> **Категория/теги:** generative recommendation, efficiency, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://github.com/goodfeli/dlbook_notation](https://github.com/goodfeli/dlbook_notation).

## 1. Коротко

- Главная идея: исследует data selection для continual adaptation generative recommenders под temporal drift.
- Алгоритм: Сравниваются representation choices и sampling strategies; gradient-based representations + distribution matching выбирают small informative subsets вместо full retraining.
- Evidence: Авторы показывают training-efficiency gains при сохранении robustness to drift.
- Ограничение: Работа data-centric; эффект зависит от drift detector/selection budget и может не заменить tokenizer updates.
- Итог: Полезна для production loops: иногда дешевле выбрать правильные данные для adaptation, чем менять модель.

**Как читать статью:** это прежде всего работа типа *generative recommendation/retrieval*; поэтому основной audit должен смотреть на candidate validity, item-level metrics, baseline parity, serving cost и update path.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Recommendation systems must continuously adapt to evolving user behavior, yet the volume of data generated in large-scale streaming environments makes frequent full retraining impractical.</td></tr>
<tr><th>Предложение авторов</th><td>Recommendation systems must continuously adapt to evolving user behavior, yet the volume of data generated in large-scale streaming environments makes frequent full retraining impractical.</td></tr>
<tr><th>Главный evidence claim</th><td>Our results demonstrate that gradient-based representations, coupled with distribution-matching, improve downstream model performance, achieving training efficiency gains while preserving robustness to drift.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>generative recommendation/retrieval</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>cross-entropy, softmax</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source</td></tr>
<tr><th>Metrics</th><td>NDCG, HR, latency, MAP</td></tr>
<tr><th>Baselines</th><td>CoST, SASRec, TALLRec, HSTU, GENRE, BM25</td></tr>
<tr><th>Ключевое предположение</th><td>Generated object должен надежно связываться с конкретным item/document/action в каталоге.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Datasets and Models: In this section, we describe the data and modeling choices in our experiments: a decade-scale music and podcast streaming corpus with evolving users and item sets (Section ), and a sequential generative recommender model suited to non-stationary user historical sequences (Section ).
1. **Ключевой модуль.** Models: Following recent works in sequential generative recommender models, we use Meta's HSTU zhai2024actions. This model takes a user’s listening history as a sequence and is trained autoregressively to predict the next interaction, which may correspond to either an item or an action. We use separate output heads for item and action prediction, both trained with...
1. **Learning signal.** Models: HSTU zhai2024actions introduces a modified attention mechanism optimized for non-stationary streaming data (see Figure ). The architecture consists of a cascade of blocks, each with three components: projection, spatial aggregation, and pointwise transformation. These components operate as follows: align & U, V, Q, K = Split ( 1(f 1(E)))...
1. **Inference / deployment path.** Models: HSTU introduces three main innovations that distinguish it from previous transformer-based architectures for recommendation. Rather than relying solely on absolute positional information, HSTU incorporates the time difference between pairs of sequence elements to compute attention weights raffel. Another distinction is its adoption of the SiLU activation...
1. **Проверяемая деталь.** Data Selection Methods: In this section, we describe the data selection methods used for our experiments. In particular, given a set of training samples as D train = x i i=1 n, our objective is to select a small high-quality subset D select D train where | D select | | D train | for training. Following recent works pruthi2020estimating, xia2024less, we conduct data selection in...
1. **Проверяемая деталь.** Data Selection Methods: User Data: In our setting, a training sample x i is a user interaction history represented as a sequence of (item, action) pairs: x i = (o i1, a i1 ),..., (o ir, a ir ), where o ij is the item id and a ij is the associated action (reason end and interaction type).
1. **Проверяемая деталь.** Data Selection Methods: Gathering Representations: In this stage, we investigate three representation types: token-based, model-based, and gradient-based representations. Below, we describe the representations in detail.

## 5. Objectives, formulas и training details

**Detected objective keywords:** cross-entropy, softmax.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `& U, V, Q, K = Split(_1(f_1(E))) \\ & Attention(Q, K, V) = AV = _2 (Q K^T + rab^p,t)V \\ & S = f_2(Norm(A V) U)`
- `F_ select^ rep = (n + r)\,F_ fwd_ representations + n r d_ rep_ similarity + n n_ top-k sort`
- `F_ select^ grad = (n + r)(F_ fwd + F_ bwd)_ gradient representations + n r d_ grad_ similarity + n n_ top-k sort`
- `s(x) = 1| D_ ref| _x' D_ ref sim\! (rep(x), rep(x')).`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Performance in monthly continuous learning setting (no training)
- Performance in monthly continuous learning setting (full training)
- Performance in monthly continuous learning setting (Random 0.5)
- Performance in monthly continuous learning setting (Random 0.2)
- Performance in monthly continuous learning setting (Random 0.1)
- Performance in monthly continuous learning setting (RepSim 0.5 with TopK sampling)
- Performance in monthly continuous learning setting (RepSim 0.2 with TopK sampling)
- Performance in monthly continuous learning setting (RepSim 0.2 with BotomK sampling)

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source</td></tr>
<tr><th>Metrics</th><td>NDCG, HR, latency, MAP</td></tr>
<tr><th>Baselines</th><td>CoST, SASRec, TALLRec, HSTU, GENRE, BM25</td></tr>
</tbody></table>
</div>

- Evidence: Авторы показывают training-efficiency gains при сохранении robustness to drift.
- Experimental Setup: We evaluate data selection for continual adaptation using the rolling protocol shown in Figure learning setup, which provides a natural framework for assessing the effectiveness of data selection methods over time. We first train the HSTU model (Section 3.2) on the proprietary streaming dataset (Section 3.1) from 2015 up to 01/01/2022. Although...
- Experimental Setup: Continual adaptation proceeds in 6-month intervals starting at 01/01/2022, which we adopt instead of single-month intervals due to the larger performance degradations observed at this timescale (see Figure in Section ). For each interval I t, we define D train t as the set of newly observed user interaction chunks of length 100...
- Results: In this continual adaptation setting, we first compare representation and sampling strategies for data selection, with gradient-based representations and diversity-aware sampling emerging as the strongest combination (Figures 4-7). We then evaluate how and when this approach improves over random data selection (Figure 8 and Table reduction )....
- Results: Representation Results: Figure representation compares model-based (RepSim) and gradient-based (GradSim) representations for scoring user histories during data selection. Across all evaluation metrics (NDCG and HR), GradSim consistently yields stronger downstream performance, indicating that gradient-informed signals better identify training...
- Our results demonstrate that gradient-based representations, coupled with distribution-matching, improve downstream model performance, achieving training efficiency gains while preserving robustness to drift.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: исследует data selection для continual adaptation generative recommenders под temporal drift.
- **Algorithmic/system:** Алгоритм: Сравниваются representation choices и sampling strategies; gradient-based representations + distribution matching выбирают small informative subsets вместо full retraining.
- **Empirical:** Evidence: Авторы показывают training-efficiency gains при сохранении robustness to drift.
- **Practical:** Ограничение: Работа data-centric; эффект зависит от drift detector/selection budget и может не заменить tokenizer updates.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.
- Generated object должен надежно связываться с конкретным item/document/action в каталоге.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source.
- Метрики: NDCG, HR, latency, MAP.
- Baselines и parity settings: CoST, SASRec, TALLRec, HSTU, GENRE, BM25.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна для production loops: иногда дешевле выбрать правильные данные для adaptation, чем менять модель.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: исследует data selection для continual adaptation generative recommenders под temporal drift.
- Алгоритм: Сравниваются representation choices и sampling strategies; gradient-based representations + distribution matching выбирают small informative subsets вместо full retraining.
- Evidence: Авторы показывают training-efficiency gains при сохранении robustness to drift.
- Ограничение: Работа data-centric; эффект зависит от drift detector/selection budget и может не заменить tokenizer updates.
- Итог: Полезна для production loops: иногда дешевле выбрать правильные данные для adaptation, чем менять модель.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Sequential generative recommendation models take sequences of user interaction histories and output a distribution over likely next interactions. Motivated by recent advances in sequence modeling with transformer architectures, a growing line of work has...</td></tr>
<tr><td>Datasets and Models</td><td>In this section, we describe the data and modeling choices in our experiments: a decade-scale music and podcast streaming corpus with evolving users and item sets (Section ), and a sequential generative recommender model suited to non-stationary...</td></tr>
<tr><td>Dataset</td><td>We use a longitudinal sample from a proprietary music and podcast streaming dataset spanning 2015–2025. The dataset contains on the order of 10K users and a vocabulary on the order of 10M items (tracks and podcasts).</td></tr>
<tr><td>Models</td><td>Following recent works in sequential generative recommender models, we use Meta's HSTU zhai2024actions. This model takes a user’s listening history as a sequence and is trained autoregressively to predict the next interaction, which may correspond to either...</td></tr>
<tr><td>Data Selection Methods</td><td>In this section, we describe the data selection methods used for our experiments. In particular, given a set of training samples as D train = x i i=1 n, our objective is to select a small high-quality subset D select D train where | D select | | D train |...</td></tr>
<tr><td>Experimental Setup</td><td>We evaluate data selection for continual adaptation using the rolling protocol shown in Figure learning setup, which provides a natural framework for assessing the effectiveness of data selection methods over time. We first train the HSTU model...</td></tr>
<tr><td>Results</td><td>In this continual adaptation setting, we first compare representation and sampling strategies for data selection, with gradient-based representations and diversity-aware sampling emerging as the strongest combination (Figures 4-7). We then evaluate how and...</td></tr>
<tr><td>Conclusion and Future Work</td><td>In this work, we study data selection for continual adaptation by examining both representation choices and sampling strategies. We find that gradient-based representations outperform model-based embeddings, accounting for a large fraction of the gains from...</td></tr>
<tr><td>Appendix</td><td>текст не извлечен; смотреть PDF/source</td></tr>
</tbody></table>
</div>
