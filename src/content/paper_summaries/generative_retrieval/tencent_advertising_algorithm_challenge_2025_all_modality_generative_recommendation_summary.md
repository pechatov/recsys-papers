---
title: "Tencent Advertising Algorithm Challenge 2025: All-Modality Generative Recommendation"
category: "generative_retrieval"
slug: "tencent_advertising_algorithm_challenge_2025_all_modality_generative_recommendation_summary"
catalogId: "paper-tencent_advertising_algorithm_challenge_2025_all_modality_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/tencent_advertising_algorithm_challenge_2025_all_modality_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.04976"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Junwei Pan, Wei Xue, Chao Zhou, Xing Zhou, Lunan Fan, Yanbo Wang, Haoran Xin, Zhiyu Hu, Yaozheng Wang, Fengye Xu, Yurong Yang, Xiaotian Li, Junbang Huo, Wentao Ning, Yuliang Sun, Chengguo Yin, Jun Zhang, Shudong Huang, Lei Xiao, Huan Yu, Irwin King, Haijie Gu, Jie Jiang.
>
> **Аффилиации:** Tencent Inc.; Chinese University of Hong Kong.
>
> **Источник:** [arXiv 2604.04976](https://arxiv.org/abs/2604.04976) · дата metadata: 2026-04-04.
>
> **Категория/теги:** generative recommendation, industrial, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://github.com/pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch) [https://huggingface.co/datasets/TAAC2025/TencentGR-1M](https://huggingface.co/datasets/TAAC2025/TencentGR-1M) [https://huggingface.co/datasets/TAAC2025/TencentGR-10M](https://huggingface.co/datasets/TAAC2025/TencentGR-10M) [https://github.com/TencentAdvertisingAlgorithmCompetition/baseline](https://github.com/TencentAdvertisingAlgorithmCompetition/baseline).

## 1. Коротко

- Главная идея: описывает Tencent Advertising Algorithm Challenge 2025 и два all-modality GR datasets для industrial ads.
- Алгоритм: TencentGR-1M и TencentGR-10M содержат user sequences, collaborative IDs, multimodal embeddings, exposure/click/conversion labels и weighted evaluation для high-value conversions.
- Evidence: Бумага дает task definition, schema, baseline, evaluation protocol и findings top solutions; данные и baseline заявлены как released.
- Ограничение: Challenge datasets де-идентифицированы и ads-specific; нужно смотреть license, leakage и whether offline metric соответствует online ads value.
- Итог: Важна как benchmark entry point для multimodal ads GR: наконец появляется крупный public-ish industrial-style dataset.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>To foster research in this direction, we organised the Tencent Advertising Algorithm Challenge 2025, a global competition built on top of two all-modality datasets for GR: TencentGR-1M and TencentGR-10M.</td></tr>
<tr><th>Предложение авторов</th><td>Generative recommender systems are rapidly emerging as a new paradigm for recommendation, where collaborative identifiers and/or multi-modal content are mapped into discrete token spaces and user behavior is modelled with autoregressive sequence models.</td></tr>
<tr><th>Главный evidence claim</th><td>Despite progress on multi-modal recommendation datasets, there is still a lack of public benchmarks that jointly offer large-scale, realistic and fully all-modality data designed specifically for generative recommendation (GR) in industrial advertising.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>contrastive, masking, causal</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Tencent, TencentGR-1M, TencentGR-10M</td></tr>
<tr><th>Metrics</th><td>NDCG, latency, throughput, MAP</td></tr>
<tr><th>Baselines</th><td>TIGER, LETTER, OneRec</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Baseline Model: To provide a strong reference implementation and lower the barrier to entry, we release a baseline generative recommendation pipeline that participants can directly build upon. The baseline adopts a next-token prediction formulation with a causal Transformer backbone and approximate-nearest-neighbour (ANN) based retrieval.
1. **Ключевой модуль.** Training: Sequence construction. For each user, we construct the input sequence as introduced in Equation, where the first token aggregates all user-level features and each subsequent token represents one ad interaction. Each token consists of multiple feature fields drawn from the shared feature schema, together with multi-modal embeddings.
1. **Learning signal.** Training: Feature encoding. We adopt a multi-field feature fusion design based on the token schema in Equation. Each categorical/ID feature field corresponds to its own embedding table, while multi-modal features directly use the provided continuous embeddings. For the user-profile token and each item-interaction token, we first perform field-wise embedding...
1. **Inference / deployment path.** Training: Backbone architecture. The sequence encoder is a causal Transformer. Given the token embeddings from Equation, we first prepend the user-profile token and add positional encodings to preserve temporal order: equation H 0 =, equation where p t denotes the learnable positional embedding at position t. We then apply L...

## 5. Objectives, formulas и training details

**Detected objective keywords:** contrastive, masking, causal.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `NDCG@K(u) = _k=1^K I\ y_u,k = G_u\ _2(k+1).`
- `w -HitRate@K(u) = _k=1^K w(y_u,k) \, I\ y_u,k G_u\ _i G_u w(i),`
- `w -DCG@K(u) = _k=1^K w(y_u,k) \, I\ y_u,k G_u\ _2(k+1).`
- `w -IDCG@K(u) = _k=1^ (K, |G_u|) w(i_k^) _2(k+1),`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Effect of fingerprint-based expansion on offline metrics.
- Overview statistics of the TencentGR datasets.
- Feature schema and statistics in both TencentGR-1M and TencentGR-10M. "S" and "M" denote single-value and multi-value categorical features, respectively.
- Multi-modal embedding models used to construct TencentGR. "T" denotes text and "I" denotes image.
- Coverage of the multi-modal embeddings on two datasets.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Tencent, TencentGR-1M, TencentGR-10M</td></tr>
<tr><th>Metrics</th><td>NDCG, latency, throughput, MAP</td></tr>
<tr><th>Baselines</th><td>TIGER, LETTER, OneRec</td></tr>
</tbody></table>
</div>

- Главная идея: описывает Tencent Advertising Algorithm Challenge 2025 и два all-modality GR datasets для industrial ads.
- Алгоритм: TencentGR-1M и TencentGR-10M содержат user sequences, collaborative IDs, multimodal embeddings, exposure/click/conversion labels и weighted evaluation для high-value conversions.
- Evidence: Бумага дает task definition, schema, baseline, evaluation protocol и findings top solutions; данные и baseline заявлены как released.
- Despite progress on multi-modal recommendation datasets, there is still a lack of public benchmarks that jointly offer large-scale, realistic and fully all-modality data designed specifically for generative recommendation (GR) in industrial advertising.
- To foster research in this direction, we organised the Tencent Advertising Algorithm Challenge 2025, a global competition built on top of two all-modality datasets for GR: TencentGR-1M and TencentGR-10M.
- Both datasets are constructed from real de-identified Tencent Ads logs and contain rich collaborative IDs and multi-modal representations extracted with state-of-the-art embedding models.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: описывает Tencent Advertising Algorithm Challenge 2025 и два all-modality GR datasets для industrial ads.
- **Algorithmic/system:** Алгоритм: TencentGR-1M и TencentGR-10M содержат user sequences, collaborative IDs, multimodal embeddings, exposure/click/conversion labels и weighted evaluation для high-value conversions.
- **Empirical:** Evidence: Бумага дает task definition, schema, baseline, evaluation protocol и findings top solutions; данные и baseline заявлены как released.
- **Practical:** Ограничение: Challenge datasets де-идентифицированы и ads-specific; нужно смотреть license, leakage и whether offline metric соответствует online ads value.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.
- Reward/utility signal достаточно стабилен и не подменяет user relevance узкой бизнес-метрикой.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Tencent, TencentGR-1M, TencentGR-10M.
- Метрики: NDCG, latency, throughput, MAP.
- Baselines и parity settings: TIGER, LETTER, OneRec.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Важна как benchmark entry point для multimodal ads GR: наконец появляется крупный public-ish industrial-style dataset.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: описывает Tencent Advertising Algorithm Challenge 2025 и два all-modality GR datasets для industrial ads.
- Алгоритм: TencentGR-1M и TencentGR-10M содержат user sequences, collaborative IDs, multimodal embeddings, exposure/click/conversion labels и weighted evaluation для high-value conversions.
- Evidence: Бумага дает task definition, schema, baseline, evaluation protocol и findings top solutions; данные и baseline заявлены как released.
- Ограничение: Challenge datasets де-идентифицированы и ads-specific; нужно смотреть license, leakage и whether offline metric соответствует online ads value.
- Итог: Важна как benchmark entry point для multimodal ads GR: наконец появляется крупный public-ish industrial-style dataset.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Discriminative recommendation models have long been the dominant paradigm in industrial recommender systems. Their evolution has been marked by two major lines of progress: increasingly expressive feature interaction modeling rendle2010fm,pan2018fwfm,...</td></tr>
<tr><td>Challenge Setting</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Problem</td><td>The core task of the challenge is a next-item recommendation problem on multi-modal ad interaction sequences. For each user u we observe chronological sequence of ad-related behaviors (, impressions, clicks, conversions):</td></tr>
<tr><td>Challenge Rounds</td><td>Preliminary round. Participants receive the TencentGR-1M dataset, containing approximately one million user sequences of impression and click behaviors. The goal is to predict the next clicked ad. They must submit training and inference code that runs inside...</td></tr>
<tr><td>Awards and Talent Programs</td><td>The challenge offers substantial incentives to encourage broad and high-quality participation. Specifically, the champion team receives a prize of 2, 000, 000 RMB, while the second and third place teams are awarded 600, 000 RMB and 300, 000 RMB,...</td></tr>
<tr><td>Participation Rules and Target Audience</td><td>The competition is open to full-time students worldwide, including undergraduate, master's, PhD and qualified postdoctoral researchers. Each participant may join at most one team, and teams may consist of one to three members. Team formation and real-name...</td></tr>
<tr><td>Data</td><td>The TencentGR datasets are constructed from de-identified logs of Tencent Ads. We first set an answer time window [t begin, t end ]. Then we sample N users who have a positive behavior,, click in the preliminary round and click or conversion in the second...</td></tr>
<tr><td>Multi-modal Features</td><td>Raw ad creatives include text (titles, descriptions), images and sometimes videos. To protect advertiser privacy and reduce storage and bandwidth, we do not release raw creatives. Instead, we extract multi-modal embeddings using a suite of production models....</td></tr>
<tr><td>Baseline Model</td><td>To provide a strong reference implementation and lower the barrier to entry, we release a baseline generative recommendation pipeline that participants can directly build upon. The baseline adopts a next-token prediction formulation with a causal Transformer...</td></tr>
<tr><td>Training</td><td>Sequence construction. For each user, we construct the input sequence as introduced in Equation, where the first token aggregates all user-level features and each subsequent token represents one ad interaction. Each token consists of multiple feature...</td></tr>
<tr><td>Inference</td><td>itemize User embedding. We feed a user's behavior history into the Transformer and take the last-layer hidden state at the final position as the user embedding, which summarizes recent behavior and context.</td></tr>
<tr><td>Competition Platform</td><td>The overall process and date for the challenge are shown as Figure fig. All competition workflows are executed on the Tencent Angel machine learning platform zhao2024efficiently, nie2023angel. Angel provides distributed training and evaluation...</td></tr>
</tbody></table>
</div>
