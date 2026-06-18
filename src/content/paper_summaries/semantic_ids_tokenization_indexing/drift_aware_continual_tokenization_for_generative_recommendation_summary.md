---
title: "Drift-Aware Continual Tokenization for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "drift_aware_continual_tokenization_for_generative_recommendation_summary"
catalogId: "paper-drift_aware_continual_tokenization_for_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/drift_aware_continual_tokenization_for_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2603.29705"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Yuebo Feng, Jiahao Liu, Mingzhe Han, Dongsheng Li, Hansu Gu, Peng Zhang, Tun Lu, Ning Gu.
>
> **Аффилиации:** Fudan University; Microsoft Research Asia.
>
> **Источник:** [arXiv 2603.29705](https://arxiv.org/abs/2603.29705) · дата metadata: 2026-03-31.
>
> **Категория/теги:** semantic IDs, tokenization, generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://github.com/HomesAmaranta/DACT](https://github.com/HomesAmaranta/DACT).

## 1. Коротко

- Главная идея: DACT решает проблему continual tokenization: новые items и interactions меняют collaborative structure, но полный retrain tokenizer+GRM слишком дорог.
- Алгоритм: Фаза fine-tuning использует Collaborative Drift Identification Module для разных режимов drifting/stationary items; затем relaxed-to-strict hierarchical reassignment обновляет коды с ограничением churn.
- Evidence: На трех datasets и двух GRM авторы показывают устойчивое улучшение continual adaptation.
- Ограничение: Главный риск - identifier churn: даже полезное обновление tokenizer может нарушить learned alignment generator.
- Итог: Это live-system paper: SID tokenizer должен уметь стареть и обновляться, иначе semantic IDs быстро становятся историческим артефактом.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>However, real-world environments evolve continuously: new items cause identifier collision and shifts, while new interactions induce collaborative drift in existing items (e.g., changing co-occurrence patterns and popularity).</td></tr>
<tr><th>Предложение авторов</th><td>To balance plasticity and stability for collaborative tokenizers, we propose DACT, a Drift-Aware Continual Tokenization framework with two stages: (i) tokenizer fine-tuning, augmented with a jointly trained Collaborative Drift Identification Module (CDIM) that outputs item-level drift confidence and enables differentiated...</td></tr>
<tr><th>Главный evidence claim</th><td>Авторы проверяют DACT на Amazon Beauty, Tools и Toys с двумя GRM-backbones, TIGER и LC-Rec, и заявляют устойчивое улучшение HR@k/NDCG@k across continual periods при меньшем разрушении старых code assignments.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>reconstruction</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Amazon Beauty, Tools, Toys; period-wise continual split P0-P4.</td></tr>
<tr><th>Metrics</th><td>HR@5/HR@10, NDCG@5/NDCG@10; дополнительно code-change rate, quantized-vs-CF embedding similarity, warm/cold item slices и training speed.</td></tr>
<tr><th>Baselines</th><td>TIGER, LC-Rec, frozen tokenizer/GRM variants, tokenizer FT, GRM FT, joint FT, Reformer, retraining GRM, LSAT, PESO.</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Collaboration-aware Tokenizer: -VAE Tokenizer. We implement the tokenizer as an L -level residual-quantization VAE (RQ-VAE). For item i, we extract static semantic information such as title and description, and obtain a semantic embedding z i R d using a pre-trained content encoder rajput2023recommender. We then feed z i into a learnable item encoder E( ), which produces...
1. **Ключевой модуль.** Collaboration-aware Tokenizer: RQ-VAE assigns tokens at each level by looking up a codebook C l = e l m m=1 M, where M is the codebook size and e l is the code embedding at level l. Let v i,1 =r i be the initial residual. At level l, we define the probability of choosing token m as equation p(m v i,l )= ! (- |v i,l -e l m | 2/T ) j=1 M ! (- |v i,l -e l j | 2/T ), prob equation where...
1. **Learning signal.** Collaboration-aware Tokenizer: A decoder reconstructs the input representation as z i=Decoder( r i), and we optimize the reconstruction loss equation L recon = |z i- z i | 2. equation
1. **Inference / deployment path.** Generative Recommender Model: Given a user u with interaction history S u=[i 1,,i T], the tokenizer maps each item i t to a discrete token sequence and we concatenate them as the input token sequence x u=[x u,1,,x u,|x u| ]. A generative recommender models next-token prediction via autoregressive factorization: equation p(x u)= t=1 |x u| p(x u,t x u,<t ), equation and the next...
1. **Проверяемая деталь.** Methods: In our framework, we decompose the item token update procedure into two stages. In the first stage, to obtain latent representations aligned with evolving collaborative signals, we introduce a CDIM trained end-to-end with the tokenizer, as illustrated in Fig.. In the second stage, we perform hierarchical code reassignment based on the...
1. **Проверяемая деталь.** Drift-Aware Tokenizer Adaptation: To capture item-level collaborative drift patterns, we introduce the CDIM, which is jointly trained with the tokenizer during fine-tuning. CDIM learns a reusable set of drift patterns and update-policy prototypes. For each item, CDIM predicts a drift confidence score d i. It takes three inputs: (i) the latent representation of item i from the previous...
1. **Проверяемая деталь.** Drift-Aware Tokenizer Adaptation: memory. We maintain two learnable slots: key slots K R S D representing drift patterns, and value slots V R S D representing the corresponding update policies, where S is the number of slots and D is the hidden dimension. At the first continual update period ( p=1 ), we randomly initialize the pattern memory K,V. For subsequent periods ( p>1 ), we...

## 5. Objectives, formulas и training details

**Detected objective keywords:** reconstruction.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `_i= Softmax\! (q_i K^),`
- `d_i= (MLP(_i V)),`
- `L_ reg= 1| I| _i I d_i^2,`
- `I_d = arg\,topK_i I \; d_i, I_s = I I_d,`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- The framework of drift-aware tokenizer adaptation in DACT. DACT introduces CDIM to learn drift patterns and update-policy prototypes, which predicts a drift confidence score to guide the differentiated training strategy. Additionally, a global code-assignment stability constraint is applied to all items.
- An example of item popularity and co-occurrence drift over time.
- Statistics of the datasets.
- Overall performance of baselines and DACT on TIGER. The best results are highlighted in bold, and the second-best results are underlined. "Tok." is the abbreviation for the tokenizer, while "$ $", "FT", "RT" refer to "Frozen", "Fine-tuning" and "Retraining" respectively.
- Overall performance of baselines and DACT on LC-Rec.
- Ablation study on the Tools dataset (TIGER).
- Average cosine similarity between quantized embeddings and CF embeddings on Toys and Tools.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Amazon Beauty, Tools, Toys; chronological continual-learning split: P0 for pretraining, P1-P4 for incremental updates.</td></tr>
<tr><th>Metrics</th><td>HR@5/HR@10, NDCG@5/NDCG@10; code-change rate; quantized-vs-CF embedding similarity; warm/cold item performance; training speed.</td></tr>
<tr><th>Baselines</th><td>TIGER and LC-Rec backbones; frozen tokenizer/GRM, tokenizer-only FT, GRM-only FT, joint FT, Reformer, TIGER retraining, LSAT and PESO for LC-Rec.</td></tr>
</tbody></table>
</div>

- Evidence: На трех datasets и двух GRM авторы показывают устойчивое улучшение continual adaptation.
- Experiment: In this section, we conduct extensive experiments on three real-world datasets with two representative GRMs to answer the following research questions: RQ1: How does DACT perform compared with different baselines? RQ2: How do the key components of DACT affect performance? RQ3: How sensitive is DACT to different hyperparameter settings? RQ4: Compared with...
- Experimental Settings: используются три Amazon-domain datasets: Beauty, Tools и Toys. P0 содержит первые 60% interactions для pretraining, P1-P4 делят оставшиеся 40% для continual updates; evaluation идет по leave-one-out protocol.
- Overall Performance: на TIGER DACT обычно дает лучшие HR/NDCG across periods, потому что обновляет identifiers для drifting items и одновременно ограничивает изменения stationary IDs, сохраняя token-embedding alignment GRM.
- LC-Rec results: на Qwen2.5-1.5B-Instruct based LC-Rec авторы показывают тот же qualitative pattern: frozen identifiers постепенно устаревают, а DACT лучше балансирует plasticity и stability.
- Stability evidence: full tokenizer fine-tuning почти полностью меняет assignments в глубоких codebook layers, тогда как DACT удерживает code-change rate существенно ниже через global stability и layer-wise reassignment.
- Ablation Study(RQ2): To thoroughly examine the role of each component in DACT, we conduct an ablation study on the Tools dataset with the TIGER backbone. Specifically, we take the variant that fine-tunes both the tokenizer and the GRM as our reference setting, and then progressively remove key designs in DACT: (1) removing CDIM and randomly selecting items for differentiated...
- Further analysis: DACT maintains higher alignment between quantized embeddings and current-period CF embeddings on Toys/Tools, which supports the claim that updated IDs track collaborative drift rather than only memorizing old content semantics.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: DACT решает проблему continual tokenization: новые items и interactions меняют collaborative structure, но полный retrain tokenizer+GRM слишком дорог.
- **Algorithmic/system:** Алгоритм: Фаза fine-tuning использует Collaborative Drift Identification Module для разных режимов drifting/stationary items; затем relaxed-to-strict hierarchical reassignment обновляет коды с ограничением churn.
- **Empirical:** Evidence: На трех datasets и двух GRM авторы показывают устойчивое улучшение continual adaptation.
- **Practical:** Ограничение: Главный риск - identifier churn: даже полезное обновление tokenizer может нарушить learned alignment generator.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Amazon Beauty, Tools, Toys; P0 pretraining plus P1-P4 continual updates with leave-one-out evaluation.
- Метрики: HR@5/HR@10, NDCG@5/NDCG@10, code-change rate, quantized-vs-CF similarity, warm/cold item slices, training speed.
- Baselines и parity settings: TIGER, LC-Rec, frozen/FT tokenizer and GRM variants, Reformer, retraining, LSAT, PESO.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Это live-system paper: SID tokenizer должен уметь стареть и обновляться, иначе semantic IDs быстро становятся историческим артефактом.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: DACT решает проблему continual tokenization: новые items и interactions меняют collaborative structure, но полный retrain tokenizer+GRM слишком дорог.
- Алгоритм: Фаза fine-tuning использует Collaborative Drift Identification Module для разных режимов drifting/stationary items; затем relaxed-to-strict hierarchical reassignment обновляет коды с ограничением churn.
- Evidence: На трех datasets и двух GRM авторы показывают устойчивое улучшение continual adaptation.
- Ограничение: Главный риск - identifier churn: даже полезное обновление tokenizer может нарушить learned alignment generator.
- Итог: Это live-system paper: SID tokenizer должен уметь стареть и обновляться, иначе semantic IDs быстро становятся историческим артефактом.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>In recent years, generative recommendation deng2025onerec,wang2023generative has emerged as a new paradigm that reformulates recommendation as a sequence generation task, different from traditional discriminative methods...</td></tr>
<tr><td>Preliminaries</td><td>We follow recent generative recommendation frameworks with a two-stage pipeline wang2024learnable,shi2025incremental: a collaborative tokenizer maps items into discrete tokens, and a generative recommender performs autoregressive prediction over these token...</td></tr>
<tr><td>Collaboration-aware Tokenizer</td><td> -VAE Tokenizer. We implement the tokenizer as an L -level residual-quantization VAE (RQ-VAE). For item i, we extract static semantic information such as title and description, and obtain a semantic embedding z i R d using a pre-trained content...</td></tr>
<tr><td>Generative Recommender Model</td><td>Given a user u with interaction history S u=[i 1,,i T], the tokenizer maps each item i t to a discrete token sequence and we concatenate them as the input token sequence x u=[x u,1,,x u,|x u| ]. A generative recommender models next-token prediction via...</td></tr>
<tr><td>Continuous Learning for GRM</td><td>With new items and new interactions arriving over time, continuous learning is essential for GRM. We denote the data stream as D 0, D 1,, D p,, where p denotes the p -th time period. We denote the model parameters at period p as M p, where M 0 represents...</td></tr>
<tr><td>Methods</td><td>In our framework, we decompose the item token update procedure into two stages. In the first stage, to obtain latent representations aligned with evolving collaborative signals, we introduce a CDIM trained end-to-end with the tokenizer, as illustrated in Fig....</td></tr>
<tr><td>Drift-Aware Tokenizer Adaptation</td><td>To capture item-level collaborative drift patterns, we introduce the CDIM, which is jointly trained with the tokenizer during fine-tuning. CDIM learns a reusable set of drift patterns and update-policy prototypes. For each item, CDIM predicts a drift...</td></tr>
<tr><td>Code Reassignment Strategy</td><td>Due to the residual-quantization nature of RQ-VAE, later codebook layers quantize progressively smaller residuals and therefore tend to encode finer-grained information. Meanwhile, to maximally preserve the token–embedding alignment learned by the GRM at...</td></tr>
<tr><td>Experiment</td><td>In this section, we conduct extensive experiments on three real-world datasets with two representative GRMs to answer the following research questions: RQ1: How does DACT perform compared with different baselines? RQ2: How do the key components of DACT affect...</td></tr>
<tr><td>Experimental Settings</td><td>Three Amazon datasets are used: Beauty, Tools, and Toys. The chronological split uses P0 as the first 60% of interactions for pretraining, then P1-P4 as four continual-update periods over the remaining 40%. The paper evaluates HR@5/10 and NDCG@5/10 on TIGER and LC-Rec.</td></tr>
<tr><td>Overall Performance(RQ1)</td><td>Performance on TIGER The comparison between our method and the baselines on the TIGER backbone is reported in Table, from which we observe that: itemize [leftmargin=*]</td></tr>
<tr><td>Ablation Study(RQ2)</td><td>To thoroughly examine the role of each component in DACT, we conduct an ablation study on the Tools dataset with the TIGER backbone. Specifically, we take the variant that fine-tunes both the tokenizer and the GRM as our reference setting, and then...</td></tr>
</tbody></table>
</div>
