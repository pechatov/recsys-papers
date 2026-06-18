---
title: "RCLRec: Reverse Curriculum Learning for Modeling Sparse Conversions in Generative Recommendation"
category: "generative_retrieval"
slug: "rclrec_reverse_curriculum_learning_for_modeling_sparse_conversions_in_generative_recommendation_summary"
catalogId: "paper-rclrec_reverse_curriculum_learning_for_modeling_sparse_conversions_in_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/rclrec_reverse_curriculum_learning_for_modeling_sparse_conversions_in_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2603.28124"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Yulei Huang, Hao Deng, Haibo Xing, Jinxin Hu, Chuanfei Xu, Zulong Chen, Yu Zhang, Xiaoyi Zeng.
>
> **Аффилиации:** Alibaba International Digital Commerce Group; Guangdong Laboratory of Artificial Intelligence and Digital Economy; Alibaba Group.
>
> **Источник:** [arXiv 2603.28124](https://arxiv.org/abs/2603.28124) · дата metadata: 2026-03-30.
>
> **Категория/теги:** generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: RCLRec добавляет reverse curriculum learning для sparse conversion supervision в GR.
- Алгоритм: Для каждого conversion target выбирается reverse subsequence conversion-related items; их semantic tokens идут decoder prefix alongside target conversion tokens; curriculum quality-aware loss фильтрует шум.
- Evidence: Offline и online A/B: +2.09% advertising revenue и +1.86% orders.
- Ограничение: Нужен надежный selector conversion-related history; неверный curriculum может зафиксировать spurious paths.
- Итог: Полезна для conversion-heavy domains: GR должен получать intermediate supervision по критическому decision process, иначе conversion signal слишком sparse.

**Как читать статью:** это прежде всего работа типа *benchmark/reproducibility/theory*; поэтому основной audit должен смотреть на protocol validity, leakage, dataset schema, assumptions и baseline fairness.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Conversion objectives in large-scale recommender systems are sparse, making them difficult to optimize.</td></tr>
<tr><th>Предложение авторов</th><td>To address these challenges, we propose RCLRec, a reverse curriculum learning-based GR framework for sparse conversion supervision.</td></tr>
<tr><th>Главный evidence claim</th><td>Experiments on offline datasets and an online A/B test show that RCLRec achieves superior performance, with +2.09% advertising revenue and +1.86% orders in online deployment.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Теоретический результат полезен как sanity check, но assumptions нужно явно сопоставлять с реальными tokenizer/decoding constraints.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>benchmark/reproducibility/theory</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>SFT</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Tmall</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, revenue, orders, MAP</td></tr>
<tr><th>Baselines</th><td>TIGER, SASRec, BERT4Rec</td></tr>
<tr><th>Ключевое предположение</th><td>Assumptions теоремы или benchmark protocol должны совпадать с реальным tokenizer/decoding setup.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Overview and Two-Stage Training: Overview of RCLRec. It uses a Reverse Curriculum Prefix Module (RCPM) to select conversion-relevant interactions from history as a decoder-side prefix, and is trained with a curriculum quality-aware loss. figure*

## 5. Objectives, formulas и training details

**Detected objective keywords:** SFT.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `L_ GR = - _ =1^L p_ \! (z_ T+1^ S_u,\ e_b,\ z_ T+1^<),`
- `$ q = MLP ([e_u; e_ pay]) R^d. $`
- `p_t = (s_t/) _j=1^T (s_j/), t=1,,T,`
- `K = argmax(p, k) = \t_1,,t_k\, | K|=k,`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- (a) Multi-behavior history–target semantic relevance. (b) Standard GR method. (c) Behavior-wise relevance statistics showing that conversions are preceded by more coherent clusters of related items. (d) Our proposed RCLRec.
- Overall framework of RCLRec.
- Overview of RCLRec. It uses a Reverse Curriculum Prefix Module (RCPM) to select conversion-relevant interactions from history as a decoder-side prefix, and is trained with a curriculum quality-aware loss.
- Performance comparison on Tmall and Industry datasets. Best results are in bold and second-best are underlined. " Improv." shows the relative improvement (\
- Experimental results on Industry and Tmall. The best results are boldfaced and the second-best results are underlined.
- Scalability and sensitivity of curriculum length $k$.
- Ablation results on the industrial dataset.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Tmall</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, revenue, orders, MAP</td></tr>
<tr><th>Baselines</th><td>TIGER, SASRec, BERT4Rec</td></tr>
</tbody></table>
</div>

- Evidence: Offline и online A/B: +2.09% advertising revenue и +1.86% orders.
- Experimental Setup: Datasets. We evaluate RCLRec on two datasets: (i) an in-house industrial advertising dataset from an Asian e-commerce platform with three behaviors and over 1B interactions, where conversions account for 1.23
- Experimental Setup: and (ii) the public Tmall dataset zhong2015stock with four behaviors using common preprocessing and user sampling settings, where conversions account for 0.95
- Experimental Results: Overall performance. Table reports the overall performance on both datasets. We observe that: (1) RCLRec achieves the best Recall@5/10 and NDCG@5/10 on the conversion target set, outperforming all baselines. (2) Traditional sequential models (e.g., SASRec, BERT4Rec) lag behind, indicating limited ability to exploit conversion-relevant signals under...
- Experimental Results: Scalability and sensitivity analysis. We vary the curriculum length k 1,,6 on the industrial dataset and report Recall@5/10, as shown in A moderate curriculum size (around k=4 ) provides a significant improvement over shorter prefixes, while further increasing k brings only marginal gains, suggesting that a small number of carefully...
- Experiments on offline datasets and an online A/B test show that RCLRec achieves superior performance, with +2.09

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: RCLRec добавляет reverse curriculum learning для sparse conversion supervision в GR.
- **Algorithmic/system:** Алгоритм: Для каждого conversion target выбирается reverse subsequence conversion-related items; их semantic tokens идут decoder prefix alongside target conversion tokens; curriculum quality-aware loss фильтрует шум.
- **Empirical:** Evidence: Offline и online A/B: +2.09% advertising revenue и +1.86% orders.
- **Practical:** Ограничение: Нужен надежный selector conversion-related history; неверный curriculum может зафиксировать spurious paths.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Теоретический результат полезен как sanity check, но assumptions нужно явно сопоставлять с реальными tokenizer/decoding constraints.
- Assumptions теоремы или benchmark protocol должны совпадать с реальным tokenizer/decoding setup.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Tmall.
- Метрики: Recall, NDCG, revenue, orders, MAP.
- Baselines и parity settings: TIGER, SASRec, BERT4Rec.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна для conversion-heavy domains: GR должен получать intermediate supervision по критическому decision process, иначе conversion signal слишком sparse.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: RCLRec добавляет reverse curriculum learning для sparse conversion supervision в GR.
- Алгоритм: Для каждого conversion target выбирается reverse subsequence conversion-related items; их semantic tokens идут decoder prefix alongside target conversion tokens; curriculum quality-aware loss фильтрует шум.
- Evidence: Offline и online A/B: +2.09% advertising revenue и +1.86% orders.
- Ограничение: Нужен надежный selector conversion-related history; неверный curriculum может зафиксировать spurious paths.
- Итог: Полезна для conversion-heavy domains: GR должен получать intermediate supervision по критическому decision process, иначе conversion signal слишком sparse.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Recently, generative recommendation (GR) based on semantic tokens has attracted extensive attention in both academia and industry reg4rec,hou2025survey,yang2024unifying. Unlike traditional sequential recommenders sasrec, csmf, GR organizes a user's...</td></tr>
<tr><td>Problem Formulation</td><td>Let U and I denote the user and item sets. Let B denote the set of behavior types, such as click (clk), add-to-cart (atc), and conversion (pay), i.e., B = clk, atc,, pay. For each user u U, the interaction history is a chronological sequence S u= (b 1,i...</td></tr>
<tr><td>Methodology</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Overview and Two-Stage Training</td><td>Overview of RCLRec. It uses a Reverse Curriculum Prefix Module (RCPM) to select conversion-relevant interactions from history as a decoder-side prefix, and is trained with a curriculum quality-aware loss. figure*</td></tr>
<tr><td>Reverse Curriculum Prefix Module (RCPM)</td><td> RCPM performs conversion-aware subsequence selection on the encoder side and exposes the selected items as decoder prefixes under teacher forcing. As shown in Figure (c), conversions are often preceded by a short sequence of...</td></tr>
<tr><td>Curriculum Quality-Aware Loss</td><td>Although RCPM injects selected intermediate curricula into the decoder, they may not always improve conversion modeling. Training the model with a curriculum prefix may increase its predictive confidence niculescu2005predicting, but it does not guarantee...</td></tr>
<tr><td>Experiments</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Experimental Setup</td><td>Datasets. We evaluate RCLRec on two datasets: (i) an in-house industrial advertising dataset from an Asian e-commerce platform with three behaviors and over 1B interactions, where conversions account for 1.23</td></tr>
<tr><td>Experimental Results</td><td>Overall performance. Table reports the overall performance on both datasets. We observe that: (1) RCLRec achieves the best Recall@5/10 and NDCG@5/10 on the conversion target set, outperforming all baselines. (2) Traditional sequential models (e.g.,...</td></tr>
<tr><td>Conclusion</td><td>This paper presents RCLRec, a reverse curriculum learning–based generative recommendation (GR) framework for sparse conversion supervision. RCLRec introduces a Reverse Curriculum Prefix Module (RCPM), which traces backward from each conversion target, selects...</td></tr>
</tbody></table>
</div>
