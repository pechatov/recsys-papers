---
title: "Echoes in Filter Bubble: Diagnosing and Curing Popularity Bias in Generative Recommenders"
category: "generative_retrieval"
slug: "echoes_in_filter_bubble_diagnosing_and_curing_popularity_bias_in_generative_recommenders_summary"
catalogId: "paper-echoes_in_filter_bubble_diagnosing_and_curing_popularity_bias_in_generative_recommenders_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/echoes_in_filter_bubble_diagnosing_and_curing_popularity_bias_in_generative_recommenders_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.16825"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Jun Yin, Bangguo Zhu, Peng Huo, Ruochen Liu, Hao Chen, Senzhang Wang, Shirui Pan, Chengqi Zhang.
>
> **Аффилиации:** Hong Kong Polytechnic University; Central South University; National Super Computing Center, Tianjin; City University of Macau; Griffith University.
>
> **Источник:** [arXiv 2605.16825](https://arxiv.org/abs/2605.16825) · дата metadata: 2026-05-16.
>
> **Категория/теги:** generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: Ghost диагностирует popularity bias в GR как совместный эффект MLE token optimization и undifferentiated SID tokenization.
- Алгоритм: Asymmetric Unlikelihood Optimization дает negative feedback tail tokens against head counterparts; Skeleton-founded Tokenization строит tail SIDs вокруг head skeleton, уменьшая хаотичную competition.
- Evidence: На трех datasets Ghost улучшает fairness/tail performance при небольшой потере overall utility.
- Ограничение: Debiasing может снижать head relevance; важно смотреть Pareto frontier, а не одну aggregate metric.
- Итог: Это обязательная работа для SID audit: scaling backbone с 0.6B до 8B не исправляет popularity bias автоматически.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Despite their effectiveness, we recognize that GRs are still susceptible to the long-standing issue of popularity bias that has pervaded the recommendation community.</td></tr>
<tr><th>Предложение авторов</th><td>Recently, Generative Recommenders (GRs), characterized by a unified end-to-end framework, have exhibited astonishing potential in transforming the recommendation paradigm.</td></tr>
<tr><th>Главный evidence claim</th><td>Extensive empirical evaluations across three datasets, alongside multiple SOTA baselines, reveal that Ghost substantially alleviates popularity bias and promotes fairer recommendations, while incurring slight degradation to the overall recommendation utility.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>MLE, maximum likelihood, softmax</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Amazon</td></tr>
<tr><th>Metrics</th><td>NDCG, Hit, HR</td></tr>
<tr><th>Baselines</th><td>TIGER, LETTER, SASRec, LightGCN, RQ-VAE, VQ-VAE, LC-Rec</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** SID Branching Points under Undifferentiated Tokenization: Subsequently, we further investigate how this token-level optimization flaw propagates into item-level bias during the recommendation process. Let c head (i) and c tail (i) be candidate tokens competing at the i -th item generation step, and step i is denoted as a branching point. Due to the asymptotically repulsive gradient updates stemming from gradient...
1. **Ключевой модуль.** SID Branching Points under Undifferentiated Tokenization: COROLLARY 1 (Head Token Dominance at Branching Point). At step i, where head and tail paths compete, the token generation probability ratio diverges from the true data distribution P d by a local amplification factor i > 1: equation P (c head (i) |h u, c <i ) P (c tail (i) |h u, c <i ) = i P d(c head (i) |h u, c <i ) P d(c tail (i) |h u, c <i )....
1. **Learning signal.** SID Branching Points under Undifferentiated Tokenization: Eq. implies that the generation process becomes pathologically overconfident in head tokens, regardless of the current context (i.e., h u and c (<i) ). Furthermore, current GRs mostly tokenize items into SIDs in an undifferentiated style, which treats items identically without accounting for their inherent popularity disparities. The...
1. **Inference / deployment path.** Methodology: With a focused effort to the two diagnosed fundamental factors, the model is developed by designing the skeleton-founded item tokenization (SKT) and the asymmetric unlikelihood optimization (AUO). Procedurally, the overview of is illustrated in In particular, SKT subtly reduces unpredictable branching points by formulating a mechanism...
1. **Проверяемая деталь.** Skeleton-Founded Item Tokenization: Current GRs tiger,lcrec,ed2 mostly adopt standard vector quantization techniques, such as RQ-VAE rqvae and RQ-KMeans rqkmeans, to allocate item SIDs. These standard approaches are agnostic to item popularity disparities, where head and tail items are indiscriminately processed. Therefore, the generated SIDs contain unstructured branching points where head...
1. **Проверяемая деталь.** Skeleton-Founded Item Tokenization: To be more specific, SKT assigns the SIDs for head and tail items asynchronously. Firstly, given the head item set V head and the corresponding representations, SKT adopts the RQ-KMeans algorithm to generate the head item SIDs. For head item v with representation X v, the SID generation follows
1. **Проверяемая деталь.** Skeleton-Founded Item Tokenization: equation item sid split c v (i) &= n 1, 2,,N | r v (i) - n (i) | 2 2, r v (i+1) &= r v (i) - c v (i) (i), for i = 1, 2,, L h. split equation L h is the SID length for head items. Initialized with r v (1) =X v, the head item v is assigned SID v = (c v (1),c v (2),,c v (L h) ). Afterwards, for the tail item v' with representation X v', SKT...

## 5. Objectives, formulas и training details

**Detected objective keywords:** MLE, maximum likelihood, softmax.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `split c_v^(i) &= _n \1, 2,,N\ \|r_v^(i)- _n^(i)\|_2^2, \\ r_v^(i+1) &= r_v^(i) - _c_v^(i)^(i), for i=1,2,,L. split`
- `L_ NLL=- _i P_ (c_v^(i)|h_u,c_v^<i),`
- `split E_ D[e_c] & E_ D. split`
- `split & E_ D[e_c_ tail,X_h_u] \\ & - E_ D 0. split`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- a). Comparison of Hit-Rate@10 (i.e., HR@10) between head and tail items. b). Comparison between the number of head and tail items in the recommendation list provided by three GRs. c). Tendency of HR@10 as the backbone parameters of LC-Rec scaling up.
- Limitations of current popularity debiasing methods on GRs. a). Trade-off between overall recommendation performance and that of tail items. b). Dilemma between performance and fairness of recommendation results.
- Overview of the Ghost model. First, textual representations are encoded based on item features. After categorizing items into head and tail sets, assigns SIDs via RQ-KMeans, allowing tail items to inherit prefixes from their closest head items. At last, optimizes the GR model by penalizing an undesired collection customized for each tail item, alongside...
- Analysis of SID lengths, including head length $L^h$ and additional length $L^t$ for tail items.
- Numbers of head and tail items in the recommendation results provided by (left) Ghost and baseline models, and (right) Ghost with different backbones.
- Tendency of performance on Ins dataset, under different AUO weights $ $. The $x$-axis denotes the values of $ $, and the $y$-axis is the metric values.
- Tendency of performance on Ins dataset, under different undesired collection sizes. The $x$-axis denotes the size of undesired collection, and the $y$-axis is the metric values.
- Statistics of the evaluated datasets. Avg.L is the average length of the user interaction sequences.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Amazon</td></tr>
<tr><th>Metrics</th><td>NDCG, Hit, HR</td></tr>
<tr><th>Baselines</th><td>TIGER, LETTER, SASRec, LightGCN, RQ-VAE, VQ-VAE, LC-Rec</td></tr>
</tbody></table>
</div>

- Evidence: На трех datasets Ghost улучшает fairness/tail performance при небольшой потере overall utility.
- Итог: Это обязательная работа для SID audit: scaling backbone с 0.6B до 8B не исправляет popularity bias автоматически.
- Gradient Analysis of MLE Optimization: opt Under the MLE objective L NLL defined in Eq., the optimization conditioned on user historical behavior h u is governed by the Softmax derivative. Let D denote the training distribution of user-item interaction pairs (h u,v), where the history h u is encoded into a representation X h u and the target item v is tokenized as a SID v=(c v...
- Gradient Analysis of MLE Optimization: LEMMA 1 (Gradient Starvation in MLE). For an arbitrary SID token c, whose embedding is e c, the expected MLE gradient update E D[ e c] over the training distribution D conforms to
- Theoretical Analysis: Within SKT, since a tail item v' inherits the prefix skeleton of its closest head item v *, the branching point of item recommendation is uniformly deferred to step (L h+1). By establishing a singular, predictable branching point, SKT exponentially restricts the local amplification factor in LEMMA 2.
- Theoretical Analysis: LEMMA 3 (Mitigation of Bias Amplification). The generation probability of the tail item v' is insulated from the multi-step geometric suppression, and its deviation from the true data distribution P d is governed exclusively by a localized factor

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: Ghost диагностирует popularity bias в GR как совместный эффект MLE token optimization и undifferentiated SID tokenization.
- **Algorithmic/system:** Алгоритм: Asymmetric Unlikelihood Optimization дает negative feedback tail tokens against head counterparts; Skeleton-founded Tokenization строит tail SIDs вокруг head skeleton, уменьшая хаотичную competition.
- **Empirical:** Evidence: На трех datasets Ghost улучшает fairness/tail performance при небольшой потере overall utility.
- **Practical:** Ограничение: Debiasing может снижать head relevance; важно смотреть Pareto frontier, а не одну aggregate metric.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Amazon.
- Метрики: NDCG, Hit, HR.
- Baselines и parity settings: TIGER, LETTER, SASRec, LightGCN, RQ-VAE, VQ-VAE, LC-Rec.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Это обязательная работа для SID audit: scaling backbone с 0.6B до 8B не исправляет popularity bias автоматически.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: Ghost диагностирует popularity bias в GR как совместный эффект MLE token optimization и undifferentiated SID tokenization.
- Алгоритм: Asymmetric Unlikelihood Optimization дает negative feedback tail tokens against head counterparts; Skeleton-founded Tokenization строит tail SIDs вокруг head skeleton, уменьшая хаотичную competition.
- Evidence: На трех datasets Ghost улучшает fairness/tail performance при небольшой потере overall utility.
- Ограничение: Debiasing может снижать head relevance; важно смотреть Pareto frontier, а не одну aggregate metric.
- Итог: Это обязательная работа для SID audit: scaling backbone с 0.6B до 8B не исправляет popularity bias автоматически.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>R ecommender systems, due to the ability to capture user preferences by analyzing historical interactions, are essential for enhancing user experiences across platforms such as e-commerce alibaba, video streaming youtube, and social networks deepinf....</td></tr>
<tr><td>Preliminary</td><td> Problem Formulation. This study focuses on the sequential recommendation task sasrec,lcrec,ed2, which aims to predict the next most suitable item based on the user historical behavior. Considering a system of K items v k|k=1,2,,K and J users...</td></tr>
<tr><td>Diagnose Popularity Bias in GRs</td><td> To investigate the popularity bias in current GRs, this study starts with an analysis of (i) the gradient of MLE optimization and (ii) the SID structure under undifferentiated tokenization.</td></tr>
<tr><td>Gradient Analysis of MLE Optimization</td><td> opt Under the MLE objective L NLL defined in Eq., the optimization conditioned on user historical behavior h u is governed by the Softmax derivative. Let D denote the training distribution of user-item interaction pairs (h u,v), where the...</td></tr>
<tr><td>SID Branching Points under Undifferentiated Tokenization</td><td>Subsequently, we further investigate how this token-level optimization flaw propagates into item-level bias during the recommendation process. Let c head (i) and c tail (i) be candidate tokens competing at the i -th item generation step, and step i is denoted...</td></tr>
<tr><td>Methodology</td><td>With a focused effort to the two diagnosed fundamental factors, the model is developed by designing the skeleton-founded item tokenization (SKT) and the asymmetric unlikelihood optimization (AUO). Procedurally, the overview of is illustrated in Figure...</td></tr>
<tr><td>Skeleton-Founded Item Tokenization</td><td>Current GRs tiger,lcrec,ed2 mostly adopt standard vector quantization techniques, such as RQ-VAE rqvae and RQ-KMeans rqkmeans, to allocate item SIDs. These standard approaches are agnostic to item popularity disparities, where head and tail items are...</td></tr>
<tr><td>Asymmetric Unlikelihood Optimization</td><td>Based on the novel SIDs generated by SKT, further designs the asymmetric unlikelihood optimization (AUO) to rectify the gradient starvation issue of tail item tokens. As analyzed in Section opt, the optimization of current GRs relies on MLE, where...</td></tr>
<tr><td>Theoretical Analysis</td><td>Within SKT, since a tail item v' inherits the prefix skeleton of its closest head item v *, the branching point of item recommendation is uniformly deferred to step (L h+1). By establishing a singular, predictable branching point, SKT exponentially...</td></tr>
<tr><td>Experiment</td><td>and baselines on three public benchmarks extracted from the Amazon platform, i.e., "Musical Instruments", "Arts, Crafts and Sewing", and "Video Games".</td></tr>
<tr><td>Main Result</td><td>To demonstrate the effectiveness of in mitigating popularity bias, we compare its performance against SOTA baselines across three public datasets. The results are summarized in </td></tr>
<tr><td>Ablation Study</td><td>To investigate the contributions of the core components in Ghost, an ablation study is conducted by removing specific modules. Based on Table, the following insights can be drawn. First, bias amplification stemming from undifferentiated...</td></tr>
</tbody></table>
</div>
