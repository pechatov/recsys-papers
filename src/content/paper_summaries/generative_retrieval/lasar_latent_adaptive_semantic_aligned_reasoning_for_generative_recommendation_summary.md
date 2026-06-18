---
title: "LASAR: Latent Adaptive Semantic Aligned Reasoning for Generative Recommendation"
category: "generative_retrieval"
slug: "lasar_latent_adaptive_semantic_aligned_reasoning_for_generative_recommendation_summary"
catalogId: "paper-lasar_latent_adaptive_semantic_aligned_reasoning_for_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/lasar_latent_adaptive_semantic_aligned_reasoning_for_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.10207"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Yiwen Chen, Fuwei Zhang, Zehao Chen, Deqing Wang, Hehan Li, Peizhi Xu, Hanmeng Liu, Shuanglong Li, Xin Pei, Fuzhen Zhuang, Zhao Zhang.
>
> **Аффилиации:** Beihang University; Baidu.
>
> **Источник:** [arXiv 2605.10207](https://arxiv.org/abs/2605.10207) · дата metadata: 2026-05-11.
>
> **Категория/теги:** generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: LASAR переносит latent reasoning в decoder-only GR, чтобы избежать дорогого explicit CoT token generation.
- Алгоритм: Двухстадийная SFT grounding SID semantics + latent reasoning, bidirectional KL aligns latent states with CoT anchors, policy head выбирает reasoning depth, RL использует terminal KL и REINFORCE.
- Evidence: На Sports/Instruments/Beauty LASAR превосходит direct generation и explicit CoT, особенно на sparse dataset.
- Ограничение: Latent reasoning сложнее отлаживать, чем текстовый CoT; benefits зависят от качества CoT anchors.
- Итог: Важна как альтернатива CoT для recommendation: reasoning может происходить в hidden states, сохраняя latency.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Latent reasoning has emerged as an effective paradigm in LLMs, performing multi-step inference in a continuous hidden-state space to achieve stronger reasoning at lower cost.</td></tr>
<tr><th>Предложение авторов</th><td>To address these, we propose LASAR (Latent Adaptive Semantic Aligned Reasoning), an SFT-then-RL framework.</td></tr>
<tr><th>Главный evidence claim</th><td>Adapting it reveals three unique challenges: (1) the gap between prior-less Semantic ID (SID) symbols and continuous latent reasoning - SIDs lack pre-trained semantics, hindering joint optimization; (2) representation drift due to a lack of reasoning chain supervision; and (3) the suboptimality of applying a globally fixed...</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>GRPO, SFT, MSE, KL</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Amazon, Beauty, Sports</td></tr>
<tr><th>Metrics</th><td>NDCG, Hit, HR, latency</td></tr>
<tr><th>Baselines</th><td>TIGER, CoST, SASRec, BERT4Rec, P5, TALLRec, GRU4Rec, LC-Rec</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Methodology: The Introduction identified three challenges in adapting latent reasoning to generative recommendation. LASAR addresses them through three corresponding designs: a latent reasoning mechanism with sample-level adaptive step prediction (Section mechanism ), an SFT phase that bridges the semantic grounding gap and representation drift via two-stage...
1. **Ключевой модуль.** Inference Efficiency and Model Scaling (RQ4): в source здесь идет таблица latency/scaling; в summary оставлен qualitative вывод, а полные числа нужно смотреть в PDF.
1. **Learning signal.** Inference Efficiency and Model Scaling (RQ4): Inference Efficiency. Table summarizes inference latency across all three datasets under beam width 50 on 8 L40. LASAR adds only negligible overhead over MiniOneRec ( 7--16
1. **Inference / deployment path.** Inference Efficiency and Model Scaling (RQ4): в source здесь идет таблица latency/scaling; в summary оставлен qualitative вывод, а полные числа нужно смотреть в PDF.

## 5. Objectives, formulas и training details

**Detected objective keywords:** GRPO, SFT, MSE, KL.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `SID(i) = Q(e_i) = (s_1, s_2,, s_M), s_j C^(j),`
- `L_ align = 1N _t=1^N D_ KL^ bidir(h_t, h_t^ cot)`
- `L_ GRPO = - E + \, D_ KL (_ \| _ ref)`
- `L_ REINFORCE = - E_N _ - H(_)`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- LASAR framework overview. A hidden-state feedback loop iteratively refines latent tokens in continuous space, while a Policy Head predicts per-sample reasoning depth $N$ for adaptive reasoning. SFT: two-stage decoupling + step-wise bidirectional KL alignment with CoT anchors. RL: GRPO (generation quality) + REINFORCE (adaptive reasoning efficiency) +...
- Batch layout for variable $N$.
- Two-stage decoupling vs.\ mixed training convergence.
- Main results across three Amazon datasets. Best in bold, second-best underlined.
- SFT-phase ablation on Beauty: alignment methods.
- RL-phase ablation on Beauty. $ $: relative to previous row.
- Step distribution: SFT vs.\ RL.
- Adaptive step allocation on Sports: adaptive $N$ outperforms all fixed configurations, and RL redistributes steps toward fewer but better-allocated depths.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Amazon, Beauty, Sports</td></tr>
<tr><th>Metrics</th><td>NDCG, Hit, HR, latency</td></tr>
<tr><th>Baselines</th><td>TIGER, CoST, SASRec, BERT4Rec, P5, TALLRec, GRU4Rec, LC-Rec</td></tr>
</tbody></table>
</div>

- Evidence: На Sports/Instruments/Beauty LASAR превосходит direct generation и explicit CoT, особенно на sparse dataset.
- Experiments: We evaluate LASAR on three Amazon product review datasets to answer four questions: (RQ1) Does LASAR outperform traditional, generative, latent-reasoning, and explicit-CoT methods (Section results )? (RQ2) What are the individual contributions of latent reasoning, alignment, and REINFORCE (Section )? (RQ3) Does...
- Experimental Setup: Datasets. We evaluate on three Amazon product review datasets 63 AmazonReviews2018, Beauty, Instruments, and Sports, which are widely recognized benchmarks in sequential recommendation research. We apply the 5-core filtering protocol and adopt the leave-one-out evaluation, following prior works 28 SASRec,29 BERT4Rec,30 GRU4Rec,34 TIGER,36 LCRec,37...
- Experimental Setup: Baselines. We compare against six baselines spanning four categories: traditional sequential models (SASRec 28 SASRec, GRU4Rec 30 GRU4Rec ), LLM-based generative methods (LC-Rec 36 LCRec, MiniOneRec 37 MiniOneRec ), latent reasoning (ReaRec 46 ReaRec ), and explicit CoT reasoning (GREAM 41 GREAM ). All generative baselines share the same base model,...
- Main Results (RQ1): в source здесь идет широкая таблица с численными HR/NDCG/Recall results. Сырая таблица не вставлена в summary; качественный вывод и headline evidence приведены в пунктах выше.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: LASAR переносит latent reasoning в decoder-only GR, чтобы избежать дорогого explicit CoT token generation.
- **Algorithmic/system:** Алгоритм: Двухстадийная SFT grounding SID semantics + latent reasoning, bidirectional KL aligns latent states with CoT anchors, policy head выбирает reasoning depth, RL использует terminal KL и REINFORCE.
- **Empirical:** Evidence: На Sports/Instruments/Beauty LASAR превосходит direct generation и explicit CoT, особенно на sparse dataset.
- **Practical:** Ограничение: Latent reasoning сложнее отлаживать, чем текстовый CoT; benefits зависят от качества CoT anchors.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Система ускоряет inference, но не улучшает модельное качество сама по себе; важно проверять stale-cache и quality-latency frontier.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Amazon, Beauty, Sports.
- Метрики: NDCG, Hit, HR, latency.
- Baselines и parity settings: TIGER, CoST, SASRec, BERT4Rec, P5, TALLRec, GRU4Rec, LC-Rec.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Важна как альтернатива CoT для recommendation: reasoning может происходить в hidden states, сохраняя latency.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: LASAR переносит latent reasoning в decoder-only GR, чтобы избежать дорогого explicit CoT token generation.
- Алгоритм: Двухстадийная SFT grounding SID semantics + latent reasoning, bidirectional KL aligns latent states with CoT anchors, policy head выбирает reasoning depth, RL использует terminal KL и REINFORCE.
- Evidence: На Sports/Instruments/Beauty LASAR превосходит direct generation и explicit CoT, особенно на sparse dataset.
- Ограничение: Latent reasoning сложнее отлаживать, чем текстовый CoT; benefits зависят от качества CoT anchors.
- Итог: Важна как альтернатива CoT для recommendation: reasoning может происходить в hidden states, сохраняя latency.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Large Language Models (LLMs) in recommender systems are advancing rapidly along two paths. One is generative recommendation: P5 31 P5 and M6-Rec 32 M6Rec pioneered the unified recommendation pretraining paradigm, TIGER 34 TIGER introduced Semantic ID-based...</td></tr>
<tr><td>Methodology</td><td>The Introduction identified three challenges in adapting latent reasoning to generative recommendation. LASAR addresses them through three corresponding designs: a latent reasoning mechanism with sample-level adaptive step prediction (..</td></tr>
<tr><td>Problem Definition</td><td>Let I be the set of items, where each item i I is associated with text features (e.g., title, description). Given a user's chronological interaction history S = i 1, i 2,, i t, the objective of sequential recommendation is to predict the next item i t+1 the...</td></tr>
<tr><td>Latent Reasoning Mechanism</td><td> LASAR framework overview. A hidden-state feedback loop iteratively refines latent tokens in continuous space, while a Policy Head predicts per-sample reasoning depth N for adaptive reasoning. SFT: two-stage decoupling + step-wise bidirectional KL...</td></tr>
<tr><td>SFT Phase: Building Semantically Anchored Latent Reasoning (Challenge 1 &amp; 2)</td><td>The SFT phase addresses the first two challenges identified in the Introduction: (1) the semantic grounding gap between prior-less SIDs and latent reasoning, and (2) representation drift. With the latent reasoning mechanism defined above, we now describe how...</td></tr>
<tr><td>RL Phase: Joint Quality and Efficiency Optimization (Challenge 3)</td><td>The SFT phase provides the Policy Head with initial step prediction via CE loss, but fixed CoT segment labels do not directly optimize recommendation quality or reasoning efficiency. The RL phase addresses this through three coordinated objectives: GRPO...</td></tr>
<tr><td>Experiments</td><td> We evaluate LASAR on three Amazon product review datasets to answer four questions: (RQ1) Does LASAR outperform traditional, generative, latent-reasoning, and explicit-CoT methods (Section results )? (RQ2) What are the individual...</td></tr>
<tr><td>Experimental Setup</td><td> Datasets. We evaluate on three Amazon product review datasets 63 AmazonReviews2018, Beauty, Instruments, and Sports, which are widely recognized benchmarks in sequential recommendation research. We apply the 5-core filtering protocol and adopt the...</td></tr>
<tr><td>Main Results (RQ1)</td><td>Main Results (RQ1): в source здесь идет широкая таблица с численными HR/NDCG/Recall results. Сырая таблица не вставлена в summary; качественный вывод и headline evidence приведены в пунктах выше.</td></tr>
<tr><td>Ablation Studies (RQ2)</td><td>We next decompose LASAR to understand which components drive its gains. Table ablation and Table present ablations across the SFT and RL phases, examining the latent reasoning mechanism, alignment method, two-stage decoupling, and RL...</td></tr>
<tr><td>Step Optimization Analysis (RQ3)</td><td>The ablation in the previous section showed that REINFORCE effectively compresses reasoning steps while improving quality. We now investigate this mechanism in detail: whether adaptive step allocation outperforms fixed configurations, and how REINFORCE...</td></tr>
<tr><td>Inference Efficiency and Model Scaling (RQ4)</td><td>Inference Efficiency and Model Scaling (RQ4): в source здесь идет таблица latency/scaling; в summary оставлен qualitative вывод, а полные числа нужно смотреть в PDF.</td></tr>
</tbody></table>
</div>
