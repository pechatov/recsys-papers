---
title: "Asymmetric Generative Recommendation via Multi-Expert Projection and Multi-Faceted Hierarchical Quantization"
category: "semantic_ids_tokenization_indexing"
slug: "asymmetric_generative_recommendation_via_multi_expert_projection_and_multi_faceted_hierarchical_summary"
catalogId: "paper-asymmetric_generative_recommendation_via_multi_expert_projection_and_multi_faceted_hierarchical_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/asymmetric_generative_recommendation_via_multi_expert_projection_and_multi_faceted_hierarchical_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.14512"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Bin Huang, Xin Wang, Junwei Pan, Yongqi Zhou, Yifeng Zhou, Zhixiang Feng, Shudong Huang, Haijie Gu, Wenwu Zhu.
>
> **Аффилиации:** Tsinghua University; BNRist; Tencent.
>
> **Источник:** [arXiv 2605.14512](https://arxiv.org/abs/2605.14512) · дата metadata: 2026-05-14.
>
> **Категория/теги:** semantic IDs, quantization, generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: критикует симметричное использование дискретных SID как inputs и targets: input side теряет семантику, output side дает грубую supervision.
- Алгоритм: AsymRec разделяет continuous input и discrete output: MSP проецирует continuous embeddings через multi-expert projections, MHQ строит multi-view/multi-level targets с semantic regularization.
- Evidence: Авторы заявляют среднее превосходство над SOTA GR на 15.8%.
- Ограничение: Асимметричный pipeline увеличивает число компонентов и может сложнее обслуживаться, чем классический SID-only Transformer.
- Итог: Сильная идея для дискуссии о том, должен ли один SID одновременно быть компактным input representation и точной decoding target.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>We identify a critical dual-stage information bottleneck in this design: (1) the Input Bottleneck, where lossy quantization degrades fine-grained semantics, while popularity bias skews the learned representations toward frequent items, and (2) the Output Bottleneck, where imprecise discrete targets limit supervision quality.</td></tr>
<tr><th>Предложение авторов</th><td>To address these issues, we propose AsymRec, an asymmetric continuous-discrete framework that decouples input and output representations.</td></tr>
<tr><th>Главный evidence claim</th><td>We identify a critical dual-stage information bottleneck in this design: (1) the Input Bottleneck, where lossy quantization degrades fine-grained semantics, while popularity bias skews the learned representations toward frequent items, and (2) the Output Bottleneck, where imprecise discrete targets limit supervision quality.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>quantization loss</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Amazon, Beauty, Sports, Toys</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, MAP, Success</td></tr>
<tr><th>Baselines</th><td>SASRec, BERT4Rec, HSTU, RQ-VAE</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Method: In this section, we introduce our model, AsymRec, as illustrated in Fig.. We first formally define the generative recommendation problem and describe the representation of items in both continuous and discrete spaces. Then, we present the two core components of our framework: Multi-expert Semantic Projection (MSP), which preserves rich...
1. **Ключевой модуль.** Multi-faceted Hierarchical Quantization (MHQ): By mapping input embeddings into a continuous feature space via MSP, we avoid input-side quantization loss. While predicting continuous vectors at the output could seem natural, this often causes dimension collapse (Sec. output ), resulting in suboptimal performance. Instead, predicting discrete semantic identifiers preserves the structure...
1. **Learning signal.** Multi-faceted Hierarchical Quantization (MHQ): Nevertheless, as pointed out in Sec. related, existing quantization methods are insufficient to capture both multi-faceted and hierarchical semantic information simultaneously. To address this, we propose the Multi-faceted Hierarchical Quantization (MHQ) module. MHQ combines the strengths of residual quantization (RQ) and product quantization (PQ)...
1. **Inference / deployment path.** Multi-faceted Hierarchical Quantization (MHQ): Given a semantic embedding vector x R d, the MHQ module first projects it into a latent space R D via a learnable linear transformation x = W P x, where W P R D d. To extract diverse semantic facets, the projected vector x is partitioned into M disjoint subspaces: equation x = [z (1), z (2),, z (M) ], z (m) R d m equation where d m = D/M denotes the...
1. **Проверяемая деталь.** AsymRec Architecture: In this section, we present the overall architecture of AsymRec, as illustrated in Fig.. The framework adopts an asymmetric continuous-discrete design: continuous embeddings are mapped into the model’s feature space via MSP, while high-fidelity, multi-faceted discrete targets are produced by MHQ for supervision. These two complementary...
1. **Проверяемая деталь.** AsymRec Architecture: Given a sequence of item embeddings corresponding to a user's interactions [x 1, x 2,, x T] each item embedding x i is first mapped into the recommendation feature space via the Multi-expert Semantic Projection (MSP) module as h i = MSP (x i).
1. **Проверяемая деталь.** AsymRec Architecture: Positional encodings are then added: equation H 0 = [h 1 + p 1, h 2 + p 2,, h T + p T]. equation The resulting sequence H 0 is then fed into L T Transformer decoder layers. Each layer utilizes multi-head self-attention and feed-forward networks to model the complex transitions between user interests:

## 5. Objectives, formulas и training details

**Detected objective keywords:** quantization loss.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `i_m,l = _k \1,, K\ \| r_l^(m) - c_k^(m,l) \|_2^2`
- `L_rec = \| x - concat(z^(1),, z^(M)) \|_2^2,`
- `E = 1M _m=1^M E,`
- `L_bal = 1M _m=1^M | E - E |.`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Existing generative recommenders rely on symmetric quantization of item embeddings, causing semantic distortion and popularity bias at the input, and imprecise supervision at the output. Our method decouples input and output representations via multi-expert semantic projection and multi-faceted hierarchical quantization, enabling high-fidelity generative...
- Statistics of the processed datasets. "Avg. $t$" denotes the average number of interactions per input sequence.
- Performance comparison among baselines and AsymRec. The best performance score is denoted in bold. The second-best performance score is denoted in underline.
- Ablation study on the Beauty dataset.
- Retrieval performance at the input stage using Mean Pooling. Results are based on a 1-of-100 sampled ranking (1 positive target vs. 99 random negatives. 40\
- Normalized Singular Spectrum of Transformer Output. We observe that Continuous Embedding Token leads to collapsed singular values, while the Discrete Token leads more dimensionally robust representations.
- NDCG@10 under different quantization configurations on the Beauty dataset. The horizontal axis shows the number of subspaces $M$, and the vertical axis shows the number of residual layers per subspace $L$.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Amazon, Beauty, Sports, Toys</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, MAP, Success</td></tr>
<tr><th>Baselines</th><td>SASRec, BERT4Rec, HSTU, RQ-VAE</td></tr>
</tbody></table>
</div>

- Evidence: Авторы заявляют среднее превосходство над SOTA GR на 15.8%.
- Experiment: To evaluate the effectiveness of AsymRec and validate our hypotheses regarding the dual-stage bottleneck, we aim to answer the following research questions:
- Experiment: itemize RQ1: Overall Performance. How does AsymRec perform compared to state-of-the-art generative and sequential recommendation baselines across various benchmarks?
- Experimental Setup: Dataset. We conduct experiments on four widely-used categories from the Amazon Review benchmark amazon mcauley2015image: Sports and Outdoors (Sports), Beauty, Toys and Games (Toys), and CDs and Vinyl (CDs). Following previous studies rajput2023recommender,sasrec kang2018self,zhou2020s3, rpg hou2025generating, we treat user reviews as interactions and...
- Experimental Setup: Evaluation Protocol. We adopt the widely used leave-last-out evaluation protocol, reserving the last item in each sequence for testing, the second-to-last item for validation, and the remaining prefix for training. To measure recommendation performance, we employ two widely-adopted ranking metrics: Recall@ K and Normalized Discounted Cumulative Gain (NDCG@...
- We identify a critical dual-stage information bottleneck in this design: (1) the Input Bottleneck, where lossy quantization degrades fine-grained semantics, while popularity bias skews the learned representations toward frequent items, and (2) the Output Bottleneck, where imprecise discrete targets limit supervision quality.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: критикует симметричное использование дискретных SID как inputs и targets: input side теряет семантику, output side дает грубую supervision.
- **Algorithmic/system:** Алгоритм: AsymRec разделяет continuous input и discrete output: MSP проецирует continuous embeddings через multi-expert projections, MHQ строит multi-view/multi-level targets с semantic regularization.
- **Empirical:** Evidence: Авторы заявляют среднее превосходство над SOTA GR на 15.8%.
- **Practical:** Ограничение: Асимметричный pipeline увеличивает число компонентов и может сложнее обслуживаться, чем классический SID-only Transformer.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Система ускоряет inference, но не улучшает модельное качество сама по себе; важно проверять stale-cache и quality-latency frontier.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Amazon, Beauty, Sports, Toys.
- Метрики: Recall, NDCG, MAP, Success.
- Baselines и parity settings: SASRec, BERT4Rec, HSTU, RQ-VAE.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Сильная идея для дискуссии о том, должен ли один SID одновременно быть компактным input representation и точной decoding target.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: критикует симметричное использование дискретных SID как inputs и targets: input side теряет семантику, output side дает грубую supervision.
- Алгоритм: AsymRec разделяет continuous input и discrete output: MSP проецирует continuous embeddings через multi-expert projections, MHQ строит multi-view/multi-level targets с semantic regularization.
- Evidence: Авторы заявляют среднее превосходство над SOTA GR на 15.8%.
- Ограничение: Асимметричный pipeline увеличивает число компонентов и может сложнее обслуживаться, чем классический SID-only Transformer.
- Итог: Сильная идея для дискуссии о том, должен ли один SID одновременно быть компактным input representation и точной decoding target.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Recent advancements have given rise to Generative Recommendation (GenRec) models, which reformulate recommendation as a sequence-to-sequence generation task hstu zhai2024actions,deng2025onerec,zhou2025onerecv2. Drawing inspiration from the success of large...</td></tr>
<tr><td>Related Works</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Generative Recommendation</td><td>Sequential recommendation (SR) aims to capture the dynamic evolution of user preferences by modeling historical interaction sequences. Traditional discriminative approaches, from early Markov Chains rendle2010factorizing to modern Transformer-based models...</td></tr>
<tr><td>Semantic ID Generation</td><td> related A key component of Generative Recommendation is the discretization of continuous semantic embeddings into semantic IDs (SIDs), which enables item representation and generation in a discrete token space. Existing methods for SID generation can...</td></tr>
<tr><td>Method</td><td>In this section, we introduce our model, AsymRec, as illustrated in Fig.. We first formally define the generative recommendation problem and describe the representation of items in both continuous and discrete spaces. Then, we present the two...</td></tr>
<tr><td>Problem Definition</td><td>Sequential recommendation aims to model user preferences based on historical interactions and predict the next item a user is likely to interact with. Formally, given a user interaction sequence [ S u = [I 1, I 2,, I T ], ] where (I i ) denotes the (i )-th...</td></tr>
<tr><td>Multi-expert Semantic Projection (MSP)</td><td>For a user interaction sequence S u, each item is represented by a continuous semantic embedding x i as well as multiple quantized tokens ID(x i). In prior generative recommendation methods, only the discrete token embeddings are used as input: the tokens...</td></tr>
<tr><td>Multi-faceted Hierarchical Quantization (MHQ)</td><td>By mapping input embeddings into a continuous feature space via MSP, we avoid input-side quantization loss. While predicting continuous vectors at the output could seem natural, this often causes dimension collapse (Sec. output ), resulting in...</td></tr>
<tr><td>AsymRec Architecture</td><td>In this section, we present the overall architecture of AsymRec, as illustrated in Fig.. The framework adopts an asymmetric continuous-discrete design: continuous embeddings are mapped into the model’s feature space via MSP, while...</td></tr>
<tr><td>Experiment</td><td>To evaluate the effectiveness of AsymRec and validate our hypotheses regarding the dual-stage bottleneck, we aim to answer the following research questions:</td></tr>
<tr><td>Experimental Setup</td><td>Dataset. We conduct experiments on four widely-used categories from the Amazon Review benchmark amazon mcauley2015image: Sports and Outdoors (Sports), Beauty, Toys and Games (Toys), and CDs and Vinyl (CDs). Following previous studies...</td></tr>
<tr><td>Overall Performance (RQ1)</td><td>We compare AsymRec with item ID-based and semantic ID-based baselines across four datasets. The results are shown in </td></tr>
</tbody></table>
</div>
