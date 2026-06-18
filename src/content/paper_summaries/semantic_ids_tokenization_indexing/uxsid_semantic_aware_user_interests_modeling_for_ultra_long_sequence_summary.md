---
title: "UxSID: Semantic-Aware User Interests Modeling for Ultra-Long Sequence"
category: "semantic_ids_tokenization_indexing"
slug: "uxsid_semantic_aware_user_interests_modeling_for_ultra_long_sequence_summary"
catalogId: "paper-uxsid_semantic_aware_user_interests_modeling_for_ultra_long_sequence_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/uxsid_semantic_aware_user_interests_modeling_for_ultra_long_sequence_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.09040"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Hongwei Zhang, Qiqiang Zhong, Jiangxia Cao, Yiyang Lv, Huanjie Wang, Liwei Guan, Jing Yao, Yiyu Wang, Junfeng Shu, Zhaojie Liu, Han Li.
>
> **Аффилиации:** Kuaishou Technology.
>
> **Источник:** [arXiv 2605.09040](https://arxiv.org/abs/2605.09040) · дата metadata: 2026-05-09.
>
> **Категория/теги:** semantic IDs, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://tianchi.aliyun.com/dataset/22482](https://tianchi.aliyun.com/dataset/22482) [https://github.com/HonghuiBao2000/LETTER/tree/master](https://github.com/HonghuiBao2000/LETTER/tree/master).

## 1. Коротко

- Главная идея: предлагает UxSID для ultra-long user sequences: вместо item-specific search или item-agnostic compression используется semantic-group shared interest memory.
- Алгоритм: Метод использует SID-группы и dual-level attention, чтобы хранить интересы на уровне семантических групп и активировать их target-aware способом.
- Evidence: В production advertising A/B авторы сообщают +0.337% revenue lift и SOTA offline performance.
- Ограничение: Нужно проверять, насколько semantic groups устойчивы во времени и не теряют редкие user intents при агрегации.
- Итог: Это user-side продолжение SID-линии: semantic IDs помогают не только индексировать items, но и сжимать длинную историю пользователя.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Modeling ultra-long user sequences involves a difficult trade-off between efficiency and effectiveness.</td></tr>
<tr><th>Предложение авторов</th><td>While current paradigms rely on either item-specific search or item-agnostic compression, we propose UxSID, a framework exploring a third path: semantic-group shared interest memory.</td></tr>
<tr><th>Главный evidence claim</th><td>This end-to-end architecture balances computational parsimony with semantic awareness, achieving state-of-the-art performance and a 0.337% revenue lift in large-scale advertising A/B test.</td></tr>
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
<tr><th>Datasets/domains</th><td>KuaiRec</td></tr>
<tr><th>Metrics</th><td>Recall, AUC, CTR, revenue, latency, throughput, Success, accuracy</td></tr>
<tr><th>Baselines</th><td>TIGER, CoST, OneRec, SASRec, HSTU, DIN</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Methodology: In this section, we present the details of UxSID, a semantic-aware compression framework for ULSM. UxSID utilizes a hierarchical attention architecture to perform end-to-end interest compression and recommendation, ensuring high-fidelity interest perception.
1. **Ключевой модуль.** Methodology: Architecture Overview. The UxSID architecture comprises three core modules: (1) SIDs Generation, which leverages a Multimodal Large Language Model (MLLM) to transform heterogeneous item content into discrete semantic codes; (2) Item-Agnostic Interest Compression, which filters raw trajectories into structured interest anchors; and (3) Hierarchical Semantic...
1. **Learning signal.** Methodology: Semantic IDs Generation. To empower the model with a semantic probe that transcends literal ID matching, we generate SIDs via a reasoning-based alignment mechanism qarmv2. Given an item i with multimodal attributes (e.g., video frames and textual descriptions), we first employ a MLLM encoder to project it into a continuous business-aligned semantic space:...

## 5. Objectives, formulas и training details

**Detected objective keywords:** objective не выражен стандартным ключевым словом; смотреть method/training sections.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `z_i = Enc_ MLLM(Attributes_i)`
- `z_i _m=1^M C_m(k_m), k_m = _j \| r_m-1 - c_m,j \|_2`
- `H = Softmax ((Q_anc W^Q)(E W^K)^ d) (E W^V)`
- `p_k = LayerNorm (h_k + (h_k W_1^(k) + b_1^(k)) W_2^(k) + b_2^(k))`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Comparison of different paradigms for ULSM. (a) Item-specific Search: Online filtering for each candidate, incurring high computational cost. (b) Item-agnostic Compression: Offline distillation into static memories, lacking target-specificity. (c) UxSID: A semantic-specific path that shares compressed interest memories among items with the same SIDs.
- The architecture of UxSID primarily comprises three components: a target SIDs Generator that quantizes comprehensive attributes into semantically rich discrete IDs; a semantic-aware compression network featuring interest compression and hierarchical attention for sequence modeling; and an end-to-end multi-task supervision framework to ensure online-offline...
- AUC Performance comparison with baselines. The best results are in bold. ‘-’ indicates category unavailability.
- Performance comparison between UxSID and other baseline models.
- UxSID ablation results. ‘-’ indicates ablation-induced or feature unavailability.
- Inference time complexity comparison. $B$: batch size, $L$: original sequence length, $R$: retrieved sequence length, $d$: hidden size, $A$: attribute index size, $m$: hash functions, $c$: compressed interest length, $f$: feature numbers.
- AUC improvements (percentage points) across various sequence lengths on all datasets.
- Hyper-parameters analysis of UxSID on all three datasets.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>KuaiRec</td></tr>
<tr><th>Metrics</th><td>Recall, AUC, CTR, revenue, latency, throughput, Success, accuracy</td></tr>
<tr><th>Baselines</th><td>TIGER, CoST, OneRec, SASRec, HSTU, DIN</td></tr>
</tbody></table>
</div>

- Evidence: В production advertising A/B авторы сообщают +0.337% revenue lift и SOTA offline performance.
- Experiments: Datasets. The effectiveness of UxSID is evaluated on two public benchmarks, XLong XLong and KuaiRec-Big kuairec, alongside a large-scale industrial dataset. These public benchmarks were selected because they provide paired content features, which are indispensable for the training and semantic alignment of SIDs. Detailed statistics and preprocessing...
- Experiments: Baselines. To evaluate the effectiveness of UxSID, we compare it against several competitive baselines (which are detailed in Appendix app:baselines ): DIN din, SIM SIM, ETA eta, SDIM sdim, MIRRN MIRRN, TWIN twin, C-Former C-former. All baselines share the bottom-level layers, varying only in ULSM.
- Ablation Study: в source здесь идет ablation table; в summary оставлен qualitative вывод, а полные числа нужно смотреть в PDF.
- Ablation Study: We perform a comprehensive ablation study on UxSID to assess its components and target semantic-specific perception (Table ).
- Efficiency and Scaling Law Analysis: Theoretical complexities are summarized in Search-based paradigms (SIM, TWIN) suffer from online matching overhead. Conversely, UxSID offloads ULSM offline, achieving O(1) online complexity for target-level compression and maintaining constant latency even when scaling to 10k.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: предлагает UxSID для ultra-long user sequences: вместо item-specific search или item-agnostic compression используется semantic-group shared interest memory.
- **Algorithmic/system:** Алгоритм: Метод использует SID-группы и dual-level attention, чтобы хранить интересы на уровне семантических групп и активировать их target-aware способом.
- **Empirical:** Evidence: В production advertising A/B авторы сообщают +0.337% revenue lift и SOTA offline performance.
- **Practical:** Ограничение: Нужно проверять, насколько semantic groups устойчивы во времени и не теряют редкие user intents при агрегации.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: KuaiRec.
- Метрики: Recall, AUC, CTR, revenue, latency, throughput, Success, accuracy.
- Baselines и parity settings: TIGER, CoST, OneRec, SASRec, HSTU, DIN.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Это user-side продолжение SID-линии: semantic IDs помогают не только индексировать items, но и сжимать длинную историю пользователя.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: предлагает UxSID для ultra-long user sequences: вместо item-specific search или item-agnostic compression используется semantic-group shared interest memory.
- Алгоритм: Метод использует SID-группы и dual-level attention, чтобы хранить интересы на уровне семантических групп и активировать их target-aware способом.
- Evidence: В production advertising A/B авторы сообщают +0.337% revenue lift и SOTA offline performance.
- Ограничение: Нужно проверять, насколько semantic groups устойчивы во времени и не теряют редкие user intents при агрегации.
- Итог: Это user-side продолжение SID-линии: semantic IDs помогают не только индексировать items, но и сжимать длинную историю пользователя.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Massive industrial platforms like TikTok, Instagram Reels, and Kuaishou attract vast user bases and host millions of creators producing images, short videos, and live streams.</td></tr>
<tr><td>Related Work</td><td>Foundations of Sequence Modeling and the Scalability Bottleneck. Historically, industrial recommendation systems have adhered to a paradigm that combines sequence modeling with feature interaction. A defining milestone in this trajectory is DIN din, which...</td></tr>
<tr><td>Methodology</td><td> In this section, we present the details of UxSID, a semantic-aware compression framework for ULSM. UxSID utilizes a hierarchical attention architecture to perform end-to-end interest compression and recommendation, ensuring high-fidelity interest...</td></tr>
<tr><td>Experiments</td><td>Datasets. The effectiveness of UxSID is evaluated on two public benchmarks, XLong XLong and KuaiRec-Big kuairec, alongside a large-scale industrial dataset. These public benchmarks were selected because they provide paired content features, which are...</td></tr>
<tr><td>Overall Performance</td><td>Overall Performance: в source здесь идет широкая таблица с численными HR/NDCG/Recall results. Сырая таблица не вставлена в summary; качественный вывод и headline evidence приведены в пунктах выше.</td></tr>
<tr><td>Ablation Study</td><td>Ablation Study: в source здесь идет ablation table; в summary оставлен qualitative вывод, а полные числа нужно смотреть в PDF.</td></tr>
<tr><td>Efficiency and Scaling Law Analysis</td><td>Theoretical complexities are summarized in Search-based paradigms (SIM, TWIN) suffer from online matching overhead. Conversely, UxSID offloads ULSM offline, achieving O(1) online complexity for target-level compression and maintaining...</td></tr>
<tr><td>Parameter Sensitivity Study</td><td>Impact of the Number of IAIC anchors ( K ). K dictates the capacity of the IAIC block to capture diverse intents. At K=4, performance drops, suggesting that compressing ultra-long behavior into a highly bottlenecked embedding forces interest entanglement and...</td></tr>
<tr><td>Visualization and Case Study</td><td>We investigated the mechanisms underlying UxSID’s efficacy, focusing on candidate SID activation within sequences and compressed anchors on KuaiRec-Big. As shown in Figure (a), distinct attention routing patterns for different candidates demonstrate...</td></tr>
<tr><td>Conclusion</td><td>Conclusion In this paper, we propose UxSID, an end-to-end framework that leverages target SIDs for compressed ultra-long sequence modeling in modern recommender systems. UxSID introduces the Item-Agnostic Interest Compression mechanism and Hierarchical...</td></tr>
<tr><td>Online Implementation</td><td>Feasibility of Online Deployment. The primary challenge of online deployment for UxSID lies in the scale of user-target interactions. Unlike traditional sequence compression methods that assign a single fixed embedding to each user, UxSID performs user-target...</td></tr>
<tr><td>Datasets</td><td>Experimental Settings: в source здесь находится большая LaTeX-таблица с dataset/config statistics; в HTML оставлено текстовое описание setup, а численные значения нужно смотреть в PDF.</td></tr>
</tbody></table>
</div>
