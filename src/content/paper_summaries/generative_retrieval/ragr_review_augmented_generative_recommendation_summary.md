---
title: "RAGR: Review-Augmented Generative Recommendation"
category: "generative_retrieval"
slug: "ragr_review_augmented_generative_recommendation_summary"
catalogId: "paper-ragr_review_augmented_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/ragr_review_augmented_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.17267"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Yingyi Zhang, Junyi Li, Yejing Wang, Wenlin Zhang, Xiaowei Qian, Sheng Zhang, Yue Feng, Yichao Wang, Yong Liu, Xiangyu Zhao, Xianneng Li.
>
> **Аффилиации:** Dalian University of Technology; City University of Hong Kong; Huawei Technologies.
>
> **Источник:** [arXiv 2605.17267](https://arxiv.org/abs/2605.17267) · дата metadata: 2026-05-17.
>
> **Категория/теги:** generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://github.com/Zhang-Yingyi/TKDE_RAGR](https://github.com/Zhang-Yingyi/TKDE_RAGR) [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) [https://huggingface.co/sentence-transformers/sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base).

## 1. Коротко

- Главная идея: RAGR добавляет review feedback прямо в generative user sequence, потому что interactions показывают what, а reviews объясняют why.
- Алгоритм: Review-Augmented User Sequence interleaves item semantic IDs и review semantic IDs; Item-Centric Task Generation Alignment через DPO удерживает модель на item prediction.
- Evidence: На трех real-world datasets RAGR дает consistent significant gains over strong GR backbones.
- Ограничение: Reviews доступны не всегда и могут быть noisy/biased; есть риск, что model начнет предсказывать review tokens вместо recommendation objective.
- Итог: Полезна для domains с rich reviews: textual feedback можно сделать частью sequence language, а не side feature.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Although recent generative recommendation (GR) methods introduce new machinery, such as semantic IDs, autoregressive decoding, and unified token spaces, they largely inherit the same item-only modeling assumption.</td></tr>
<tr><th>Предложение авторов</th><td>Motivated by this observation, we propose Review-Augmented Generative Recommendation (RAGR), a novel GR framework that incorporates review feedback directly into the generative user sequence rather than treating reviews as auxiliary side information.</td></tr>
<tr><th>Главный evidence claim</th><td>Experiments on three real-world datasets show that RAGR yields consistent and significant gains over strong GR backbones across all metrics.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>DPO</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Amazon, Beauty, Sports, Toys</td></tr>
<tr><th>Metrics</th><td>NDCG, Hit, MAP</td></tr>
<tr><th>Baselines</th><td>TIGER, LETTER, SASRec, BERT4Rec, GRU4Rec, RQ-VAE</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** The proposed RAGR method: The overall framework of the proposed RAGR, which consists of three main stages: Tokenizer Training, Review-Augmented User Sequence Modeling, and Item-Centric Task Generation Alignment. -0.2in figure*
1. **Ключевой модуль.** Tokenizer Training: To address the need for a unified discrete token space for both item interactions and review feedback, we first train a semantic tokenizer based on item text. As shown in Fig., we first encode item text into dense semantic representations with an LLM-based text encoder, and then train an RQ-VAE tiger tokenizer to quantize such representations...
1. **Learning signal.** Tokenizer Training: Formally, let x i denote the textual content associated with item i I, such as its title, category, or description. We first employ a pretrained LLM-based text encoder to obtain a dense embedding equation e i = E(x i), equation where E( ) denotes the text encoder and e i R d is the semantic embedding of item i. Compared with raw item IDs, such...
1. **Inference / deployment path.** Tokenizer Training: Based on these item embeddings, we train a residual quantization variational autoencoder (RQ-VAE) to discretize the continuous semantic space into a sequence of code indices. Specifically, given an item embedding e i, the encoder of RQ-VAE first maps it into a latent representation equation h i = g enc (e i), equation where g enc ( ) denotes the encoder...
1. **Проверяемая деталь.** Review-Augmented User Sequence Modeling: To address the limitation that existing generative recommendation models primarily capture what users selected but not why they selected it, we augment the conventional item-only interaction sequence with review feedback, so that each historical interaction is represented by both its item semantic ID and its corresponding review semantic ID. As illustrated...
1. **Проверяемая деталь.** Review-Augmented User Sequence Modeling: Let z(i t) and z(r t) denote the semantic IDs of item i t and review r t, respectively, obtained from the tokenizer trained in the first stage (Section train ). For each user u, we construct the review-augmented sequence as equation S u =. equation Compared with conventional item-only sequences, this formulation allows the model to observe not...
1. **Проверяемая деталь.** Review-Augmented User Sequence Modeling: Based on the review-augmented sequence, we construct a unified textual generation task by serializing item SIDs and review SIDs into autoregressive training instances. Specifically, we generate two types of sequence-to-sequence samples from the same review-augmented interaction history.

## 5. Objectives, formulas и training details

**Detected objective keywords:** DPO.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `S_u^<t =,`
- `f_: S_u^<t i_t,`
- `S_u^<t =.`
- `p_ (z(i_t) S_u^<t).`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- The overall framework of the proposed RAGR, which consists of three main stages: Tokenizer Training, Review-Augmented User Sequence Modeling, and Item-Centric Task Generation Alignment.
- Examples of review-augmented textual training instances. From the same review-augmented user sequence, we construct two types of autoregressive samples: next-item SID prediction and next-review SID prediction.
- An example of the recommendation task alignment instance used for DPO training.
- Comparison Between Existing and Review-Augmented Generative Recommendation.
- Comparison of ID tokenization methods in LLM-based recommendations. Unlike the existing methods, our approach can tokenize users and items with LLM-compatible tokens by leveraging high-order collaborative knowledge.
- Statistics of the datasets after preprocessing.
- Performance comparison on Beauty, Toys, and Sport. Imp. denotes the relative improvement of RAGR over its corresponding backbone model in each metric.
- Illustration of the progressively enhanced training paradigms in the ablation study. Starting from the original item-only training of TIGER, we gradually introduce review-augmented input ( +Input), review-augmented task modeling ( +Task), and finally the full RAGR framework with item-centric DPO alignment ( +RAGR).

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Amazon, Beauty, Sports, Toys</td></tr>
<tr><th>Metrics</th><td>NDCG, Hit, MAP</td></tr>
<tr><th>Baselines</th><td>TIGER, LETTER, SASRec, BERT4Rec, GRU4Rec, RQ-VAE</td></tr>
</tbody></table>
</div>

- Evidence: На трех real-world datasets RAGR дает consistent significant gains over strong GR backbones.
- Experiment: In this section, we conduct comprehensive experiments to evaluate the proposed RAGR framework. The experiments are designed to answer the following five research questions:
- Experiment: itemize [leftmargin=*] RQ1: To what extent can RAGR improve the performance of existing generative recommendation backbones on next-item recommendation?
- Experimental Settings: в source здесь находится большая LaTeX-таблица с dataset/config statistics; в HTML оставлено текстовое описание setup, а численные значения нужно смотреть в PDF.
- Experimental Settings: To evaluate the effectiveness of RAGR, we conduct experiments on three benchmark datasets from the Amazon review datasets he2016ups: Amazon-Beauty (Beauty for short), Amazon-Sports and Outdoors (Sport for short), and Amazon-Toys and Games (Toys for short). https://cseweb.ucsd.edu/ jmcauley/datasets/amazon/links.html These datasets contain both user--item...
- Ablation Study (RQ2): To answer RQ2 and understand the contribution of each component in RAGR, we conduct an ablation study on two GR backbones. As illustrated in Fig., we progressively extend the training paradigm from the original item-only generative recommendation to the full RAGR framework. Specifically, the compared variants are defined as follows: itemize...

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: RAGR добавляет review feedback прямо в generative user sequence, потому что interactions показывают what, а reviews объясняют why.
- **Algorithmic/system:** Алгоритм: Review-Augmented User Sequence interleaves item semantic IDs и review semantic IDs; Item-Centric Task Generation Alignment через DPO удерживает модель на item prediction.
- **Empirical:** Evidence: На трех real-world datasets RAGR дает consistent significant gains over strong GR backbones.
- **Practical:** Ограничение: Reviews доступны не всегда и могут быть noisy/biased; есть риск, что model начнет предсказывать review tokens вместо recommendation objective.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.
- Reward/utility signal достаточно стабилен и не подменяет user relevance узкой бизнес-метрикой.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Amazon, Beauty, Sports, Toys.
- Метрики: NDCG, Hit, MAP.
- Baselines и parity settings: TIGER, LETTER, SASRec, BERT4Rec, GRU4Rec, RQ-VAE.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна для domains с rich reviews: textual feedback можно сделать частью sequence language, а не side feature.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: RAGR добавляет review feedback прямо в generative user sequence, потому что interactions показывают what, а reviews объясняют why.
- Алгоритм: Review-Augmented User Sequence interleaves item semantic IDs и review semantic IDs; Item-Centric Task Generation Alignment через DPO удерживает модель на item prediction.
- Evidence: На трех real-world datasets RAGR дает consistent significant gains over strong GR backbones.
- Ограничение: Reviews доступны не всегда и могут быть noisy/biased; есть риск, что model начнет предсказывать review tokens вместо recommendation objective.
- Итог: Полезна для domains с rich reviews: textual feedback можно сделать частью sequence language, а не side feature.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Acknowledgments</td><td>This research was supported by the Science Challenge Project, No.TZ2025005 and the National Natural Science Foundation of China (NSFC) under Grants 72071029, 72231010, 62502404. This research was partially supported by Hong Kong Research Grants Council...</td></tr>
<tr><td>Introduction</td><td>R ecommender systems are indispensable to modern e-commerce platforms zhao2024recommender, such as Amazon he2016ups and Alibaba wang2018billion, where they surface relevant items from catalogs of millions. Among various paradigms, sequential recommendation...</td></tr>
<tr><td>Problem Formulation</td><td>Let U and I denote the user set and item set, respectively. For each user u U, we observe a chronological interaction history equation S u =, equation where i t I denotes the interacted item at step t, and r t denotes the associated textual feedback, such...</td></tr>
<tr><td>The proposed RAGR method</td><td>The overall framework of the proposed RAGR, which consists of three main stages: Tokenizer Training, Review-Augmented User Sequence Modeling, and Item-Centric Task Generation Alignment. -0.2in figure*</td></tr>
<tr><td>Overview of RAGR</td><td>As illustrated in Fig., RAGR consists of three stages: Tokenizer Training, Review-Augmented User Sequence Modeling, and Item-Centric Task Generation Alignment.</td></tr>
<tr><td>Tokenizer Training</td><td>To address the need for a unified discrete token space for both item interactions and review feedback, we first train a semantic tokenizer based on item text. As shown in Fig., we first encode item text into dense semantic representations with an...</td></tr>
<tr><td>Review-Augmented User Sequence Modeling</td><td>To address the limitation that existing generative recommendation models primarily capture what users selected but not why they selected it, we augment the conventional item-only interaction sequence with review feedback, so that each historical interaction...</td></tr>
<tr><td>Item-Centric Task Generation Alignment</td><td>To address the risk that review-augmented sequence modeling may blur the task boundary of next-item recommendation and inadvertently drive the model to generate review semantic IDs, we further introduce an item-centric task alignment stage that explicitly...</td></tr>
<tr><td>Experiment</td><td>In this section, we conduct comprehensive experiments to evaluate the proposed RAGR framework. The experiments are designed to answer the following five research questions:</td></tr>
<tr><td>Experimental Settings</td><td>Experimental Settings: в source здесь находится большая LaTeX-таблица с dataset/config statistics; в HTML оставлено текстовое описание setup, а численные значения нужно смотреть в PDF.</td></tr>
<tr><td>Overall Performance (RQ1)</td><td>To answer RQ1, we conducted a comprehensive performance comparison against SR baselines. Table results compact reports the overall performance comparison on Beauty, Toys, and Sport. Several observations can be drawn.</td></tr>
<tr><td>Ablation Study (RQ2)</td><td>To answer RQ2 and understand the contribution of each component in RAGR, we conduct an ablation study on two GR backbones. As illustrated in Fig., we progressively extend the training paradigm from the original item-only generative...</td></tr>
</tbody></table>
</div>
