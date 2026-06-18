---
title: "Lost in Decoding? Reproducing and Stress-Testing the Look-Ahead Prior in Generative Retrieval"
category: "generative_retrieval"
slug: "lost_in_decoding_reproducing_and_stress_testing_the_look_ahead_prior_in_generative_retrieval_summary"
catalogId: "paper-lost_in_decoding_reproducing_and_stress_testing_the_look_ahead_prior_in_generative_retrieval_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/lost_in_decoding_reproducing_and_stress_testing_the_look_ahead_prior_in_generative_retrieval_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.23396"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Kidist Amde Mekonnen, Yongkang Li, Yubao Tang, Simon Lupart, Maarten de Rijke.
>
> **Аффилиации:** University of Amsterdam.
>
> **Источник:** [arXiv 2604.23396](https://arxiv.org/abs/2604.23396) · дата metadata: 2026-04-25.
>
> **Категория/теги:** generative retrieval, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://github.com/kidist-amde/lost-in-decoding](https://github.com/kidist-amde/lost-in-decoding) [https://github.com/Ziems/llm-url](https://github.com/Ziems/llm-url) [https://github.com/varshakishore/IncDSI](https://github.com/varshakishore/IncDSI) [https://github.com/google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) [https://github.com/microsoft/SEED-Encoder/](https://github.com/microsoft/SEED-Encoder/) [https://github.com/huggingface/trl](https://github.com/huggingface/trl) [https://github.com/huggingface/peft](https://github.com/huggingface/peft) [https://github.com/simengggg/MGR-CSC](https://github.com/simengggg/MGR-CSC).

## 1. Коротко

- Главная идея: репродуцирует и stress-test-ит Planning Ahead in Generative Retrieval / look-ahead prior.
- Алгоритм: Авторы воспроизводят checkpoint/artifacts PAG, добавляют plan-drift diagnostics и тестируют lexical variation, typos, cross-lingual robustness и query mitigation.
- Evidence: Основные results на MS MARCO/TREC-DL воспроизводятся, но planning signal оказывается brittle к intent-preserving surface variations.
- Ограничение: Работа не предлагает новый retriever; это reproducibility/stress-test, выводы зависят от конкретного PAG setup.
- Итог: Полезна как warning для decoding tricks: look-ahead prior может выглядеть сильным offline, но ломаться на query noise.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Generative retrieval (GR) ranks documents by autoregressively generating document identifiers.</td></tr>
<tr><th>Предложение авторов</th><td>Beyond reproduction, we introduce plan drift diagnostics that quantify how intent-preserving query variations alter the planner's top-n candidate set and highest-weight planner tokens, and how these changes affect guided decoding.</td></tr>
<tr><th>Главный evidence claim</th><td>Using the authors' released checkpoint and identifier/trie artifacts under the reported decoding setup, we reproduce the main effectiveness results on MS MARCO Dev and TREC-DL 2019/2020, and corroborate the reported beam-size-latency trade-off in our hardware setting.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>reinforcement</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>MS MARCO</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, MRR, latency</td></tr>
<tr><th>Baselines</th><td>NCI, PAG</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Reproducibility Methodology: This section formalizes the reproduced PAG inference pipeline and introduces the diagnostics used in our robustness analyses. Table summarizes the main notation used throughout the section.
1. **Ключевой модуль.** Model and identifier configuration: Experiments use the released T5-base checkpoint and fixed corpus-side artifacts. Each document d has (i) a sequential docid c d=[c d,1,,c d,L ] constructed via residual quantization (RQ) with L=8 and docid-token vocabulary size V=2048, and (ii) a set-based identifier t d= t d,1,,t d,m with m=64 planning tokens.

## 5. Objectives, formulas и training details

**Detected objective keywords:** reinforcement.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `L_ align = ^2\, KL\! (softmax\! (z_ teach)\ \|\ softmax\! (z_ stud)),`
- `z[v]= _t ((1+ ReLU(lexical\_logit_t[v])) m_t),`
- `s(c_d;q)= _i=1^L E_i[c_d,i] h_i(c_d,<i,q)`
- `s_ simul(q,d)= _j=1^m h_q[t_d,j],`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- RQ2: Retrieval performance under query variations.
- RQ2: Retrieval under query variations. Stage 1 ranks by $s_ simul$; Stage 2 is full PAG. Values are $ $ across five seeds, with degradation $ =M(q)-M( q)$ in parentheses. Primary metrics: MRR@10 (Dev) and NDCG@10 (DL19/20).
- RQ2: Planner stability and plan swapping.
- Planner stability and plan swapping (defined in ). Mean$ $std over five seeds; PlanSwapDrop $<0$ implies the clean plan helps.
- Effectiveness--efficiency trade-offs for PAG on MS MARCO Dev (cf.\ Table 3 in the original paper) across planner
- RQ1: MS MARCO Dev effectiveness--efficiency trade-offs (cf.\ original Table 3) across planner set size $m$ (tokens/doc) and beam size $k$. Latency values are hardware-specific (paper: A100 80GB; ours: H100 96GB). \ denotes settings requiring unreleased artifacts; $^ $ marks reproduced effectiveness below the paper.
- Ablation on MS MARCO Dev (cf.\ Table 2 in the original paper). \ denotes variants requiring retraining or unreleased checkpoints.
- PAG effectiveness reproduction (cf.\ Table 1 in the original paper) with additional recent dense baselines for context.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>MS MARCO</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, MRR, latency</td></tr>
<tr><th>Baselines</th><td>NCI, PAG</td></tr>
</tbody></table>
</div>

- Evidence: Основные results на MS MARCO/TREC-DL воспроизводятся, но planning signal оказывается brittle к intent-preserving surface variations.
- Experimental Setup: Reproducibility scope. Our experiments rely on the authors’ released checkpoint and corpus-side artifacts (docids and trie). When intermediate artifacts needed to reconstruct a training stage are unavailable, we treat the corresponding components as fixed and explicitly mark results that would require retraining to reproduce. Unless stated otherwise,...
- Released artifacts and experimental scopes: Inference-time reproducibility (RQ1, RQ2) We evaluate PAG using the released T5-base checkpoint, the trie built over 8.8M sequential passage identifiers ( L=8, V=2048 ), and the stored set-based identifiers used for simultaneous planning scores s simul (q,d). We follow the reported decoding procedure without modifying document identifiers.
- Released artifacts and experimental scopes: Cross-lingual query shift (RQ3) We issue non-English mMARCO queries against the fixed English MS MARCO passage collection using the same released docids and trie. This setting evaluates query--corpus language mismatch without re-indexing. As query-side mitigations, we evaluate translation into English before planning and decoding, and a learned...
- Using the authors' released checkpoint and identifier/trie artifacts under the reported decoding setup, we reproduce the main effectiveness results on MS MARCO Dev and TREC-DL 2019/2020, and corroborate the reported beam-size-latency trade-off in our hardware setting.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: репродуцирует и stress-test-ит Planning Ahead in Generative Retrieval / look-ahead prior.
- **Algorithmic/system:** Алгоритм: Авторы воспроизводят checkpoint/artifacts PAG, добавляют plan-drift diagnostics и тестируют lexical variation, typos, cross-lingual robustness и query mitigation.
- **Empirical:** Evidence: Основные results на MS MARCO/TREC-DL воспроизводятся, но planning signal оказывается brittle к intent-preserving surface variations.
- **Practical:** Ограничение: Работа не предлагает новый retriever; это reproducibility/stress-test, выводы зависят от конкретного PAG setup.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Система ускоряет inference, но не улучшает модельное качество сама по себе; важно проверять stale-cache и quality-latency frontier.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: MS MARCO.
- Метрики: Recall, NDCG, MRR, latency.
- Baselines и parity settings: NCI, PAG.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна как warning для decoding tricks: look-ahead prior может выглядеть сильным offline, но ломаться на query noise.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: репродуцирует и stress-test-ит Planning Ahead in Generative Retrieval / look-ahead prior.
- Алгоритм: Авторы воспроизводят checkpoint/artifacts PAG, добавляют plan-drift diagnostics и тестируют lexical variation, typos, cross-lingual robustness и query mitigation.
- Evidence: Основные results на MS MARCO/TREC-DL воспроизводятся, но planning signal оказывается brittle к intent-preserving surface variations.
- Ограничение: Работа не предлагает новый retriever; это reproducibility/stress-test, выводы зависят от конкретного PAG setup.
- Итог: Полезна как warning для decoding tricks: look-ahead prior может выглядеть сильным offline, но ломаться на query noise.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td> Generative retrieval and prefix pruning. Generative retrieval (GR) reframes search as sequence generation: given a query, a model retrieves by autoregressively generating a document identifier (docid) Tay2022DSI. At inference time, decoding is...</td></tr>
<tr><td>Related Work</td><td>Generative retrieval, constrained decoding, and guidance Generative retrieval (GR) ranks documents by generating corpus-specific identifiers (docids) rather than scoring documents directly in an embedding space...</td></tr>
<tr><td>Reproducibility Methodology</td><td>This section formalizes the reproduced PAG inference pipeline and introduces the diagnostics used in our robustness analyses. Table summarizes the main notation used throughout the section.</td></tr>
<tr><td>Problem Formulation</td><td>A GR model ranks documents by autoregressively decoding length- L docids c d=[c d,1,,c d,L ] conditioned on a query q. Inference uses trie-constrained beam search over valid prefixes. Decoding is scored additively using decoder states h i(c &lt;i,q) and...</td></tr>
<tr><td>Planning Ahead in Generative Retrieval</td><td>PAG Zeng2024PlanningAI reduces prefix pruning by pairing each document's sequential docid c d with a set-based docid t d= t d,1,,t d,m, an unordered set of m planning tokens drawn from a planning vocabulary. At inference time, PAG combines a fast planning...</td></tr>
<tr><td>PAG Optimization Pipeline</td><td>PAG trains a single backbone to support (i) set-based planning scores and (ii) sequential docid decoding, via three stages (see Zeng2024PlanningAI for full objectives and sampling).</td></tr>
<tr><td>Plan Stability, Plan Sensitivity, and Plan Collapse</td><td>For each query q, PAG’s planning stage produces a top- n candidate set D n(q) (top- n documents under s simul (q,d) ) and a planner token set P (q), the top- planning-vocabulary tokens under query weights h q[ ]. Given an original query q and an...</td></tr>
<tr><td>Experimental Setup</td><td>Reproducibility scope. Our experiments rely on the authors’ released checkpoint and corpus-side artifacts (docids and trie). When intermediate artifacts needed to reconstruct a training stage are unavailable, we treat the corresponding components as fixed and...</td></tr>
<tr><td>Released artifacts and experimental scopes</td><td>Inference-time reproducibility (RQ1, RQ2) We evaluate PAG using the released T5-base checkpoint, the trie built over 8.8M sequential passage identifiers ( L=8, V=2048 ), and the stored set-based identifiers used for simultaneous planning scores s simul (q,d)...</td></tr>
<tr><td>Datasets and metrics</td><td>subsec:data We use the MS MARCO passage retrieval benchmark (8.8M passages) and report results on MS MARCO Dev (6,980 queries) and TREC-DL 2019/2020. Following the original paper, we report MRR@10 on MS MARCO Dev, NDCG@10 on TREC-DL, and Recall@10 on all...</td></tr>
<tr><td>Model and identifier configuration</td><td>Experiments use the released T5-base checkpoint and fixed corpus-side artifacts. Each document d has (i) a sequential docid c d=[c d,1,,c d,L ] constructed via residual quantization (RQ) with L=8 and docid-token vocabulary size V=2048, and (ii) a set-based...</td></tr>
<tr><td>Inference and decoding settings</td><td>subsec:decode Retrieval uses trie-constrained beam search with the planning-guided prefix score in Eq. score. Unless otherwise stated, inference follows the paper’s default configuration: beam size k=100 and a planning set formed by the top n=1000...</td></tr>
</tbody></table>
</div>
