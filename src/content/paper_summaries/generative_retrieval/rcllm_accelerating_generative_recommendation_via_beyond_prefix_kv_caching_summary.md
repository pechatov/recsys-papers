---
title: "RcLLM: Accelerating Generative Recommendation via Beyond-Prefix KV Caching"
category: "generative_retrieval"
slug: "rcllm_accelerating_generative_recommendation_via_beyond_prefix_kv_caching_summary"
catalogId: "paper-rcllm_accelerating_generative_recommendation_via_beyond_prefix_kv_caching_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/rcllm_accelerating_generative_recommendation_via_beyond_prefix_kv_caching_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.07443"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Zhan Zhao, Yuxin Wang, Amelie Chi Zhou.
>
> **Аффилиации:** Hong Kong Baptist University.
>
> **Источник:** [arXiv 2605.07443](https://arxiv.org/abs/2605.07443) · дата metadata: 2026-05-08.
>
> **Категория/теги:** generative recommendation, efficiency, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://business.yelp.com/data/resources/open-dataset/](https://business.yelp.com/data/resources/open-dataset/).

## 1. Коротко

- Главная идея: RcLLM ускоряет LLM-based GR inference через Beyond-Prefix KV Caching, потому что reusable fragments в recommendation часто non-contiguous.
- Алгоритм: Промпты разбиваются на reusable blocks; user-history caches реплицируются, item caches shard-ятся similarity-aware, scheduler улучшает locality, selective attention чинит approximation errors.
- Evidence: На real-world datasets TTFT уменьшается в 1.31x-9.51x relative to prefix caching systems при negligible accuracy loss.
- Ограничение: Система требует distributed cache infrastructure и стабильных reusable prompt blocks.
- Итог: Полезна для serving LLM-GR: классический prefix caching недостаточен для user/item prompt composition.

**Как читать статью:** это прежде всего работа типа *serving/decoding/system efficiency*; поэтому основной audit должен смотреть на latency, throughput, memory footprint, cache hit ratio, training cost и отсутствие деградации item-level quality.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Large Language Models (LLMs) are transforming recommendation from ranking into a generative task, but industrial deployment remains limited by the high latency of processing long, personalized prompts.</td></tr>
<tr><th>Предложение авторов</th><td>We present RcLLM, a distributed inference system for generative recommendation with Beyond-Prefix KV Caching.</td></tr>
<tr><th>Главный evidence claim</th><td>Standard prefix caching provides limited benefit because reuse in recommendation workloads is often non-contiguous across user histories and item contexts.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>serving/decoding/system efficiency</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>softmax</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Amazon, Yelp, Goodreads</td></tr>
<tr><th>Metrics</th><td>Hit, latency, hit ratio, accuracy</td></tr>
<tr><th>Baselines</th><td>CoST, OneRec</td></tr>
<tr><th>Ключевое предположение</th><td>Workload/decoding setup должен иметь достаточную locality или parallelism, а ускорение не должно менять candidate distribution.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** LLM-Based Recommendation Systems: The landscape of Recommendation Systems (RecSys) is undergoing a fundamental paradigm shift. While traditional approaches like the Deep Learning Recommendation Model (DLRM) yu2026nearzero rely on sparse ID-based features to model user preferences, recent advancements have moved toward prompt-based Large Language Models (LLMs). Emerging frameworks such as...
1. **Ключевой модуль.** LLM-Based Recommendation Systems: Prompt analysis for generative recommendation. (a) An example prompt combining user interaction history, candidate items, and task instructions. (b) An example item with typical product fields (title, category, description) forming mostly static prompt content, and a review example with semantically redundant text. (c) Inference latency exceeding industrial...
1. **Learning signal.** LLM-Based Recommendation Systems: The core of this workflow is the prompt construction, which serves as the input workload for the model. As illustrated in Figure, the system does not ingest simple vectors, but rather constructs a composite textual prompt comprising three primary elements: detailed user history, a set of candidate items, and specific task instructions....
1. **Inference / deployment path.** System Overview: In this paper, we propose, a distributed inference system designed to bridge the gap between the computational demands of generative recommendation and the strict latency constraints of industrial serving. As illustrated in Figure, adopts a split-phase architecture that fundamentally decouples the management of KV cache states from...
1. **Проверяемая деталь.** System Overview: Offline Phase ( ) The offline pipeline (bottom of Figure ) is designed to align storage strategies with the distinct scale and access patterns of recommendation data. Instead of a monolithic cache, stratifies the prompt context into two distinct KV storage pools as shown in Table: itemize...
1. **Проверяемая деталь.** System Overview: в source здесь находится широкая LaTeX-таблица; сырая таблица удалена из HTML summary, чтобы не ломать чтение.
1. **Проверяемая деталь.** System Implementation: To validate RcLLM’s design across both performance and accuracy dimensions, we developed a hybrid implementation framework comprising two distinct artifacts: itemize [leftmargin=*] Distributed Serving Engine: We implemented RcLLM’s distributed logic on top of Vidur MLSYS2024 b74a8de4, a high-fidelity LLM inference simulator. We extended Vidur’s kernel to...

## 5. Objectives, formulas и training details

**Detected objective keywords:** softmax.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `Attention(Q, K, V) = softmax (QK^T d_k)V`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Prompt analysis for generative recommendation.
- Latency vs. Industrial SLO
- Prompt analysis for generative recommendation. (a) An example prompt combining user interaction history, candidate items, and task instructions. (b) An example item with typical product fields (title, category, description) forming mostly static prompt content, and a review example with semantically redundant text. (c) Inference latency exceeding industrial...
- Standard Autoregressive Inference Process
- Analysis of token characteristics and attention mechanisms.
- Semantic Clustering of Tokens
- Attention Sparsity in Qwen3-8B
- Analysis of token characteristics and attention mechanisms. (a) Visualizing token embeddings from 1-star and 5-star reviews in 2D space reveals distinct semantic clusters indicating a limited reusable vocabulary. (b) The CDF of cosine similarity shows over 93\ (c) The average attention score heatmap across heads and layers exhibits strong diagonal locality...

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Amazon, Yelp, Goodreads</td></tr>
<tr><th>Metrics</th><td>Hit, latency, hit ratio, accuracy</td></tr>
<tr><th>Baselines</th><td>CoST, OneRec</td></tr>
</tbody></table>
</div>

- Evidence: На real-world datasets TTFT уменьшается в 1.31x-9.51x relative to prefix caching systems при negligible accuracy loss.
- Evaluation: subfigure [b] 0.32 Figures/amazon ttft cdf k40 Qwen Qwen3-8B.pdf -1ex Amazon (Qwen3-8B) subfigure subfigure [b] 0.32 Figures/yelp ttft cdf k40 Qwen Qwen3-8B.pdf -1ex Yelp (Qwen3-8B) subfigure subfigure [b] 0.32 Figures/goodreads ttft cdf k40 Qwen Qwen3-8B.pdf -1ex Goodreads (Qwen3-8B)...
- Evaluation: subfigure [b] 0.32 Figures/amazon ttft cdf k40 Qwen Qwen-72B.pdf -1ex Amazon (Qwen-72B) subfigure subfigure [b] 0.32 Figures/yelp ttft cdf k40 Qwen Qwen-72B.pdf -1ex Yelp (Qwen-72B) subfigure subfigure [b] 0.32 Figures/goodreads ttft cdf k40 Qwen Qwen-72B.pdf -1ex Goodreads (Qwen-72B)...
- Experimental Setup: Testbed Environment. We simulate a local cluster with K=40 instances by default and scale to K 100 in scalability ablations. For experiments with Qwen3-8B yang2025qwen3technicalreport, each instance is equipped with a single NVIDIA A100 GPU (80GB). For the larger Qwen-72B yang2024qwen2technicalreport, each instance utilizes 4 NVIDIA A100 GPUs...
- Experimental Setup: We employ Qwen3-8B as the primary model for evaluating recommendation accuracy, while Qwen-72B is additionally used to assess system performance and scalability. This selection reflects a balance between reasoning capability and industrial serving costs. Models with even larger parameters are not considered due to the prohibitive gap between their baseline...
- Standard prefix caching provides limited benefit because reuse in recommendation workloads is often non-contiguous across user histories and item contexts.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: RcLLM ускоряет LLM-based GR inference через Beyond-Prefix KV Caching, потому что reusable fragments в recommendation часто non-contiguous.
- **Algorithmic/system:** Алгоритм: Промпты разбиваются на reusable blocks; user-history caches реплицируются, item caches shard-ятся similarity-aware, scheduler улучшает locality, selective attention чинит approximation errors.
- **Empirical:** Evidence: На real-world datasets TTFT уменьшается в 1.31x-9.51x relative to prefix caching systems при negligible accuracy loss.
- **Practical:** Ограничение: Система требует distributed cache infrastructure и стабильных reusable prompt blocks.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Система ускоряет inference, но не улучшает модельное качество сама по себе; важно проверять stale-cache и quality-latency frontier.
- Workload/decoding setup должен иметь достаточную locality или parallelism, а ускорение не должно менять candidate distribution.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Amazon, Yelp, Goodreads.
- Метрики: Hit, latency, hit ratio, accuracy.
- Baselines и parity settings: CoST, OneRec.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна для serving LLM-GR: классический prefix caching недостаточен для user/item prompt composition.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: RcLLM ускоряет LLM-based GR inference через Beyond-Prefix KV Caching, потому что reusable fragments в recommendation часто non-contiguous.
- Алгоритм: Промпты разбиваются на reusable blocks; user-history caches реплицируются, item caches shard-ятся similarity-aware, scheduler улучшает locality, selective attention чинит approximation errors.
- Evidence: На real-world datasets TTFT уменьшается в 1.31x-9.51x relative to prefix caching systems при negligible accuracy loss.
- Ограничение: Система требует distributed cache infrastructure и стабильных reusable prompt blocks.
- Итог: Полезна для serving LLM-GR: классический prefix caching недостаточен для user/item prompt composition.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Large Language Models (LLMs) are transforming recommendation systems from static retrieval engines updlrm2024,yu2026nearzero into generative reasoners. Unlike classical ID-based pipelines, LLM-based recommenders...</td></tr>
<tr><td>Background and Motivation</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>LLM-Based Recommendation Systems</td><td>The landscape of Recommendation Systems (RecSys) is undergoing a fundamental paradigm shift. While traditional approaches like the Deep Learning Recommendation Model (DLRM) yu2026nearzero rely on sparse ID-based features to model user preferences, recent...</td></tr>
<tr><td>The Autoregressive LLM Inference</td><td>Modern LLM-based recommendation systems typically rely on Transformer-based architectures (e.g., Llama grattafiori2024llama3herdmodels, Qwen yang2025qwen3technicalreport ) to generate rankings autoregressively. As illustrated in Figure...</td></tr>
<tr><td>Insights and Motivation</td><td>The severe latency bottleneck imposed by prefill stems from the inefficiency of treating every recommendation prompt as a unique computational event. In reality, the prompt context is dominated by two highly repetitive components: 1) candidate items, which...</td></tr>
<tr><td>System Design</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>System Overview</td><td>In this paper, we propose, a distributed inference system designed to bridge the gap between the computational demands of generative recommendation and the strict latency constraints of industrial serving. As illustrated in Figure,...</td></tr>
<tr><td>Offline Phase: Distributed KV Construction and Placement</td><td>The offline phase is responsible for materializing reusable KV states and preparing them for efficient access during online inference. As introduced above, builds two fundamentally different KV cache pools, including the KV cache pool for semantic review...</td></tr>
<tr><td>Online Phase: Cache-Aware Distributed Inference</td><td>The online phase executes the stateful inference pipeline introduced in the overview ( ). It orchestrates three pipelined stages to minimize TTFT while preserving model fidelity, including affinity scheduling, zero-copy retrieval, and...</td></tr>
<tr><td>System Implementation</td><td>To validate RcLLM’s design across both performance and accuracy dimensions, we developed a hybrid implementation framework comprising two distinct artifacts: itemize [leftmargin=*] Distributed Serving Engine: We implemented RcLLM’s distributed logic on top...</td></tr>
<tr><td>Evaluation</td><td>subfigure [b] 0.32 Figures/amazon ttft cdf k40 Qwen Qwen3-8B.pdf -1ex Amazon (Qwen3-8B) subfigure subfigure [b] 0.32 Figures/yelp ttft cdf k40 Qwen Qwen3-8B.pdf -1ex Yelp (Qwen3-8B) subfigure subfigure [b] 0.32...</td></tr>
<tr><td>Experimental Setup</td><td>Testbed Environment. We simulate a local cluster with K=40 instances by default and scale to K 100 in scalability ablations. For experiments with Qwen3-8B yang2025qwen3technicalreport, each instance is equipped with a single NVIDIA A100 GPU (80GB). For the...</td></tr>
</tbody></table>
</div>
