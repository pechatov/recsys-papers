---
title: "On Synthetic Data Strategies for Domain-Specific Generative Retrieval"
category: "generative_retrieval"
slug: "synthetic_data_strategies_domain_specific_generative_retrieval_summary"
catalogId: "paper-synthetic_data_strategies_domain_specific_generative_retrieval_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/synthetic_data_strategies_domain_specific_generative_retrieval_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2502.17957"
---
> **Авторы:**
>
> Haoyang Wen, Jiang Guo, Yi Zhang, Jiarong Jiang, Zhiguo Wang.
>
>   
>
>
> **Аффилиации:**
>
> Language Technologies Institute, Carnegie Mellon University; AWS AI / Amazon.
>
>   
>
>
> **Источник:**
>
> arXiv:2502.17957, 2025.

## 1. Коротко

Статья не предлагает новый backbone для generative retrieval; она систематически изучает data recipe для domain-specific corpora, где мало human-labeled queries. Pipeline двухстадийный: SFT учит LLM декодировать document identifiers из synthetic queries и context, затем preference learning улучшает ranking через hard negatives из собственных ошибок модели.

## 2. Контекст

Generative retrieval хранит корпус параметрически: по query модель генерирует identifier документа. Это убирает внешний dense index, но создает проблему обучения: модель должна и запомнить корпус, и научиться query-to-document relevance. Для domain-specific corpus ручная разметка запросов дорогая, поэтому авторы исследуют synthetic data.

В качестве identifiers в основных экспериментах используются semantic document identifiers: LLM генерирует keyword list, описывающий документ. Также проверяется atomic identifier setup, чтобы показать, что data strategy не привязана только к semantic IDs. На инференсе применяется constrained beam search with Trie, чтобы генерировать только валидные identifiers.

## 3. Метод / pipeline

- **Context2ID:** chunk или document content сопоставляется с document identifier; цель - заставить GR модель запомнить корпус.
- **Query2ID:** LLM генерирует synthetic queries для chunk/document, а модель учится выдавать relevant identifier.
- **Multi-granular queries:** chunk-level queries покрывают общий смысл, sentence-level queries вытаскивают мелкие факты.
- **Constraints-based queries:** prompt включает domain metadata: MultiHop-RAG использует author, publish time, source, category, title; AllSides - political polarity; AGNews - location и topic.
- **Preference learning:** используется Regularized Preference Optimization. После SFT модель retrieves top candidates для новых synthetic queries; negatives берутся из top-k результатов, которые ранжированы выше positive candidate. Random negatives проверяются как baseline и оказываются хуже.

Base model: Mistral-7B-Instruct-v0.3 для semantic identifiers и Mistral-7B-v0.3 для atomic identifiers. Synthetic queries генерируются Mixtral 8x7B, keywords - Claude 3 Sonnet. Training: 8x NVIDIA A100-SXM4-40GB, bfloat16; каждый training/inference procedure укладывается в 1 день.

## 4. Результаты и evidence

Датасеты: MultiHop-RAG, AllSides, AGNews и Natural Questions. Статистика synthetic data: MultiHop-RAG 7,724 contexts, 72,090 chunk queries, 472,193 sentence queries, 51,212 constraints queries; AllSides 645 contexts, 6,313 / 173,898 / 6,091; AGNews 1,050 contexts, 10,355 / 80,524 / 20,875; NQ 98,748 contexts и 1,459,031 chunk-level queries.

<div class="table-scroll">
<table>
<thead><tr><th>Experiment</th><th>Key result</th><th>Вывод</th></tr></thead>
<tbody>
<tr><td>MultiHop-RAG chunk vs +sentence</td><td>HIT@4 43.64 -&gt; 61.64; HIT@10 66.65 -&gt; 81.69; MRR@10 31.14 -&gt; 47.20</td><td>Sentence-level queries резко улучшают покрытие мелких фактов.</td></tr>
<tr><td>Constraints-based queries</td><td>MultiHop-RAG HIT@4 61.64 -&gt; 69.98; AllSides HIT@1 10.19 -&gt; 14.20; AGNews HIT@1 59.91 -&gt; 62.19</td><td>Доменная metadata помогает запросам быть ближе к реальным retrieval constraints.</td></tr>
<tr><td>Context2ID</td><td>MultiHop-RAG HIT@4 41.33 -&gt; 69.98; NQ HIT@1 69.72 -&gt; 70.71</td><td>Memorization objective особенно важен для domain-specific corpus.</td></tr>
<tr><td>Preference learning</td><td>SFT MultiHop-RAG HIT@4 69.98; Top-10 negatives 71.88. NQ HIT@1 70.71; Top-10 negatives 71.22</td><td>Hard negatives из собственных retrieval results полезнее random negatives.</td></tr>
</tbody>
</table>
</div>

LLM query generation превосходит docT5query: на MultiHop-RAG Mixtral 8x7B дает HIT@4 61.64 против 50.86, MRR@10 47.20 против 37.73; на NQ HIT@1 70.71 против 63.30. В сравнении с off-the-shelf retrievers итоговая generative model competitive или лучше: MultiHop-RAG HIT@4 71.88 против BM25 64.35 и GTE-Qwen2 63.24; NQ HIT@1 71.22 против GTE-Qwen2 60.45; AllSides HIT@1 14.20 против 9.11; AGNews HIT@1 62.19, но E5/GTE лучше на HIT@5/10.

## 5. Ограничения

- Авторы прямо ограничивают вклад SFT и preference-learning data strategy; incremental learning и unseen document generalization остаются future work.
- Synthetic queries в основном строятся из одного документа; real-world multi-document, multi-hop и comparative queries остаются открытой проблемой.
- Все experiment results single run; statistical variance не раскрыта.
- LLM-generated data требует Mixtral/Claude и отдельного prompt engineering под каждый домен.

## 6. Связь с GR/SID

Это одна из самых полезных работ для практического SID/GR обучения: она показывает, что качество semantic identifiers само по себе недостаточно. Нужны Context2ID для memorization, Query2ID на нескольких granularity, domain constraints и hard-negative preference learning. Для recommender-side GR аналогия прямая: user/item SIDs должны сопровождаться данными, которые учат модель не только декодировать валидный ID, но и ранжировать близкие альтернативы.
