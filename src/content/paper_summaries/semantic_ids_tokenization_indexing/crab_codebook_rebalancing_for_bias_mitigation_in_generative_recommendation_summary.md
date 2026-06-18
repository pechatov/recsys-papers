---
title: "CRAB: Codebook Rebalancing for Bias Mitigation in Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "crab_codebook_rebalancing_for_bias_mitigation_in_generative_recommendation_summary"
catalogId: "paper-crab_codebook_rebalancing_for_bias_mitigation_in_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/crab_codebook_rebalancing_for_bias_mitigation_in_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.05113"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Zezhong Fan, Ziheng Chen, Luyi Ma, Jin Huang, Lalitesh Morishetti, Kaushiki Nag, Sushant Kumar, Kannan Achan.
>
> **Аффилиации:** Stony Brook University.
>
> **Источник:** [arXiv 2604.05113](https://arxiv.org/abs/2604.05113) · дата metadata: 2026-04-06.
>
> **Категория/теги:** semantic IDs, quantization, generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: CRAB диагностирует popularity bias в GeneRec как следствие imbalanced tokenization и training, которое переучивает popular tokens.
- Алгоритм: Post-hoc rebalance split-ит over-popular codebook tokens, сохраняя hierarchy, и добавляет tree-structured regularizer для semantic consistency.
- Evidence: На real-world datasets CRAB улучшает recommendation performance и снижает popularity bias.
- Ограничение: Post-hoc split может изменить mapping и потребовать переобучения/миграции generator; важно оценить head-quality trade-off.
- Итог: Статья важна для codebook auditing: collapse и popularity bias появляются на уровне tokens, не только на уровне ranker loss.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Despite its strong performance across multiple recommendation tasks, existing GeneRec approaches still suffer from severe popularity bias and may even exacerbate it.</td></tr>
<tr><th>Предложение авторов</th><td>Building on these insights, we propose CRAB, a post-hoc debiasing strategy for GeneRec that alleviates popularity bias by mitigating frequency imbalance among semantic tokens.</td></tr>
<tr><th>Главный evidence claim</th><td>Despite its strong performance across multiple recommendation tasks, existing GeneRec approaches still suffer from severe popularity bias and may even exacerbate it.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>objective не выражен стандартным ключевым словом; смотреть method/training sections</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Office</td></tr>
<tr><th>Metrics</th><td>NDCG, HR</td></tr>
<tr><th>Baselines</th><td>TIGER, CoST, OneRec, SASRec, RQ-VAE</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Residual Quantization: For each item i I, its textual description is first passed through a frozen encoder to obtain a continuous embedding z i. We aim to quantize it using L hierarchical tokens with L codebooks in a coarse-to-fine generation manner. At the l -th level, the codebook C l = c k l | k=1, K consists of K codeword embeddings c k l. When l=1, the initial residual...
1. **Ключевой модуль.** Model Optimization: To mitigate popularity bias in GeneRec, we jointly optimize the recommendation loss L Rec and the hierarchical regularizer L T: equation aligned L = L Rec + L T aligned equation where is a hyperparameter controlling the strength of regularization. During optimization, we update the embedding layers for both existing and newly introduced tokens. To...

## 5. Objectives, formulas и training details

**Detected objective keywords:** objective не выражен стандартным ключевым словом; смотреть method/training sections.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `Ch(c_k^l) = \ c_j^\,l+1 C^l+1 \; |\; c_k^l c_j^\,l+1 \.`
- `P(c^l_k(m)) = ^| Ch(c_k^l)| _j=1 z_j[m]P(c^l+1_j),\: L_bal= _m=1^M (P(c^l_k(m))- P)^2`
- `aligned _ z \;\; & _m=1^M ^| Ch(c_k^l)|_j=1 z_j[m]n_j \| r^l_j - _m \|^2+ L_bal, n_j=| I_c_j^l+1|\\ s.t. \;\; & z_j[m] \0,1\, _m=1^M z_j[m] = 1, \, c_j^\,l+1 Ch(c_k^l), aligned`
- `_i=1^n_j \| r^l_i - _m\|^2 = _i=1^n_j \| r^l_i- r^l_j\|^2 + n_j \| r^l_j\ - _m\|^2,\: r^l_j= _i=1^n_j r^l_in_j`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Illustration of CRAB with a three-level codebook in MOR. Over-popular tokens are split by redistributing their child tokens via regularized Kmeans. For clarity, we denote $c_i^1$, $c_j^2$, and $c_k^3$ as $ A_i$, $ B_j$, and $ C_k$, respectively.
- Performance and Efficiency Comparison on Industrial and Office Datasets.
- Left: Popularity bias of GeneRec on the industrial dataset, with the x-axis representing item groups by popularity. Right: The GU of item groups grouped by token popularity.
- PS($ $), PN($ $), and EC($ $) of factual and counterfactual explanations generated by GREASE, PGExplainer, PersonalRank, and Random for top-$10$, top-$15$, and top-$20$ recommended items according to LightGCN, NGCF, and GCMC GNN-based recommender systems trained on LastFM, Yelp, and Gowalla datasets.
- $HR@10 100$ for different attacks with $80\
- Unlearning effectivenss and model performance on Movie and Goodreads.
- Left: MOR performance under different splitting ratios ($5\
- Left: Effect of $ $ Right: Effect of LoRA

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Office</td></tr>
<tr><th>Metrics</th><td>NDCG, HR</td></tr>
<tr><th>Baselines</th><td>TIGER, CoST, OneRec, SASRec, RQ-VAE</td></tr>
</tbody></table>
</div>

- Evidence: На real-world datasets CRAB улучшает recommendation performance и снижает popularity bias.
- Theoretical Analysis: We provide a theoretical analysis of CURE to indicate that a single gradient update on the model parameters of CURE achieves lower loss than normal gradient descent. Let =,, denote the parameters of components in selected circuits. After applying CURE, the model parameters are =,,. An one step gradient update of is: equation...
- Theoretical Analysis:. We iteratively update each parameter group while fixing the others. The loss difference between the conventional update and CURE is analyzed via two components, A and B: equation aligned & ( ) - ( ) ( - ) ( + ) & + ( - ) + ( - ) &= - A & + ( - ) + ( - ) B aligned equation
- Experimental Settings: Datasets Experiments are carried out on two real-world datasets Office and Industrial. Following kong2025minionerec, we first filter out users and items with fewer than five interactions. For each dataset, interactions are split chronologically into training, validation, and test sets with an 8:1:1 ratio. Evaluation Metrics subsec:metric Following...
- Experimental Settings: Implementation We implement CRAB based on the MOR framework using Qwen2-0.5B as the backbone. The model is trained on 4 NVIDIA A100 GPUs for 10 epochs. We employ the AdamW optimizer with a global batch size of 128. The learning rate is set to 1 10 -4 with a weight decay of 0.01. To efficiently train the model, we utilize LoRA hu2022lora with the rank r=8,...
- Despite its strong performance across multiple recommendation tasks, existing GeneRec approaches still suffer from severe popularity bias and may even exacerbate it.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: CRAB диагностирует popularity bias в GeneRec как следствие imbalanced tokenization и training, которое переучивает popular tokens.
- **Algorithmic/system:** Алгоритм: Post-hoc rebalance split-ит over-popular codebook tokens, сохраняя hierarchy, и добавляет tree-structured regularizer для semantic consistency.
- **Empirical:** Evidence: На real-world datasets CRAB улучшает recommendation performance и снижает popularity bias.
- **Practical:** Ограничение: Post-hoc split может изменить mapping и потребовать переобучения/миграции generator; важно оценить head-quality trade-off.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Система ускоряет inference, но не улучшает модельное качество сама по себе; важно проверять stale-cache и quality-latency frontier.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Office.
- Метрики: NDCG, HR.
- Baselines и parity settings: TIGER, CoST, OneRec, SASRec, RQ-VAE.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Статья важна для codebook auditing: collapse и popularity bias появляются на уровне tokens, не только на уровне ranker loss.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: CRAB диагностирует popularity bias в GeneRec как следствие imbalanced tokenization и training, которое переучивает popular tokens.
- Алгоритм: Post-hoc rebalance split-ит over-popular codebook tokens, сохраняя hierarchy, и добавляет tree-structured regularizer для semantic consistency.
- Evidence: На real-world datasets CRAB улучшает recommendation performance и снижает popularity bias.
- Ограничение: Post-hoc split может изменить mapping и потребовать переобучения/миграции generator; важно оценить head-quality trade-off.
- Итог: Статья важна для codebook auditing: collapse и popularity bias появляются на уровне tokens, не только на уровне ranker loss.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td> Generative recommendation (GeneRec) wang2024learnable, han2025mtgr, liu2025onerec has emerged as a promising paradigm for sequential recommendation, demonstrating strong empirical performance across diverse domains...</td></tr>
<tr><td>Background and Preliminaries</td><td> We consider the sequential recommendation task. Let U denote the set of users and I the set of items. Given a user u U with a chronologically ordered interaction history H u = i 1,, i T, the goal is to predict the next item i T+1. Following...</td></tr>
<tr><td>Residual Quantization</td><td>For each item i I, its textual description is first passed through a frozen encoder to obtain a continuous embedding z i. We aim to quantize it using L hierarchical tokens with L codebooks in a coarse-to-fine generation manner. At the l -th level, the...</td></tr>
<tr><td>Autoregressive Generation</td><td>By applying the residual quantization to each item in H u, the input is transformered into a flattened token sequence X. Accordingly, the target next item i T+1 is also encoded as Y: [ X =, Y =[ s T+1 1,s 2 T+1, s L T+1 i T+1 ] ] where s l i denotes the...</td></tr>
<tr><td>Motivation</td><td>Mechanistically, this issue stems from the codebook construction process. Semantically similar items are mapped to the same token; when such items are popular, their interactions accumulate on that token, further increasing its frequency. After training on...</td></tr>
<tr><td>Method</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Rebalancing the Codebook</td><td>The hierarchical token assignment in Equation induces a parent–child relation between tokens at consecutive levels, denoted by c k l c j l+1 if there exists at least one item whose l -th and (l+1) -th tokens are c k l and c j l+1, respectively....</td></tr>
<tr><td>Hierarchical Semantic Alignment</td><td>Splitting the token c k l introduces M new tokens. To adapt the LLM to the rebalanced codebook and mitigate bias, we introduce a tree-structure-aware regularizer L T that promotes representation consistency among tokens sharing the same parent. equation...</td></tr>
<tr><td>Model Optimization</td><td>To mitigate popularity bias in GeneRec, we jointly optimize the recommendation loss L Rec and the hierarchical regularizer L T: equation aligned L = L Rec + L T aligned equation where is a hyperparameter controlling the strength of regularization....</td></tr>
<tr><td>Theoretical Analysis</td><td> We provide a theoretical analysis of CURE to indicate that a single gradient update on the model parameters of CURE achieves lower loss than normal gradient descent. Let =,, denote the parameters of components in selected circuits. After applying...</td></tr>
<tr><td>Experiments</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Experimental Settings</td><td>Datasets Experiments are carried out on two real-world datasets Office and Industrial. Following kong2025minionerec, we first filter out users and items with fewer than five interactions. For each dataset, interactions are split chronologically into...</td></tr>
</tbody></table>
</div>
