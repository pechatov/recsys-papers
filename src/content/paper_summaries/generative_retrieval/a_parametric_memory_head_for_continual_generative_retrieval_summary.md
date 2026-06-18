---
title: "A Parametric Memory Head for Continual Generative Retrieval"
category: "generative_retrieval"
slug: "a_parametric_memory_head_for_continual_generative_retrieval_summary"
catalogId: "paper-a_parametric_memory_head_for_continual_generative_retrieval_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/a_parametric_memory_head_for_continual_generative_retrieval_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.23388"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Kidist Amde Mekonnen, Yubao Tang, Maarten de Rijke.
>
> **Аффилиации:** University of Amsterdam.
>
> **Источник:** [arXiv 2604.23388](https://arxiv.org/abs/2604.23388) · дата metadata: 2026-04-25.
>
> **Категория/теги:** generative retrieval, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://github.com/Ziems/llm-url](https://github.com/Ziems/llm-url) [https://github.com/varshakishore/IncDSI](https://github.com/varshakishore/IncDSI) [https://github.com/google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) [https://github.com/microsoft/SEED-Encoder/](https://github.com/microsoft/SEED-Encoder/) [https://github.com/huggingface/trl](https://github.com/huggingface/trl) [https://github.com/huggingface/peft](https://github.com/huggingface/peft) [https://huggingface.co/google-t5/t5-base](https://huggingface.co/google-t5/t5-base) [https://huggingface.co/intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct).

## 1. Коротко

- Главная идея: PAMT добавляет parametric memory head для continual GenIR, чтобы адаптироваться к новым documents без catastrophic forgetting.
- Алгоритм: Backbone frozen after adaptation; product-key memory head sparse-queries decoder hidden states и корректирует trie-valid token scores. Memory values обновляются по decoding-time access statistics с fixed budget.
- Evidence: На MS MARCO/NQ sequential adaptation memory tuning улучшает stability-plasticity trade-off.
- Ограничение: Memory growth, access policy и long-run maintenance требуют отдельного system analysis.
- Итог: Полезна для dynamic index: GenIR model-as-index нуждается в modular update path, иначе каждый update угрожает старому corpus knowledge.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Generative information retrieval (GenIR) consolidates retrieval into a single neural model that decodes document identifiers (docids) directly from queries.</td></tr>
<tr><th>Предложение авторов</th><td>To address this, we propose post-adaptation memory tuning (PAMT), a memory-only stabilization stage that augments an adapted model with a modular parametric memory head (PMH).</td></tr>
<tr><th>Главный evidence claim</th><td>We show that sequential adaptation improves retrieval on newly added documents but substantially degrades performance on earlier slices, exposing a pronounced stability-plasticity trade-off.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>cross-entropy, distillation</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>MS MARCO, NQ, Natural Questions</td></tr>
<tr><th>Metrics</th><td>метрики нужно сверить в таблицах experiments</td></tr>
<tr><th>Baselines</th><td>DSI, NCI</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Methodology: We propose post-adaptation memory tuning (PAMT) for continual GenIR. As shown in Figure, PAMT follows an adapt--then--stabilize procedure that separates learning new query--docid mappings from post-adaptation decoder-side calibration. For each session t 1, Stage 1 adapts the GenIR backbone on the arriving slice D t to ensure plasticity. Stage...
1. **Ключевой модуль.** Model architecture and document identifiers: backbone. We use T5-base. https://huggingface.co/google-t5/t5-base For SPQ docids, the decoder vocabulary is expanded from 32, 128 to 40, 320 tokens by adding 8, 192 dedicated docid tokens organized into M=32 disjoint blocks.
1. **Learning signal.** Model architecture and document identifiers: Document identifier schemes subsec:docid We use two docid schemes. SPQ IDs embed each document using SentenceTransformer E5-Mistral-7B-Instruct https://huggingface.co/intfloat/e5-mistral-7b-instruct over the title plus the first 50 sentences, apply 2 normalization, and PQ-code it with M=32 and K=256; the codebook is learned on D 0 and reused for all...
1. **Inference / deployment path.** Model architecture and document identifiers: Initial training on D 0 Models are initialized from t5-base and trained on D 0 using document-to-docid pairs, pseudo-query-to-docid pairs generated via doc2query https://huggingface.co/castorini/doc2query-t5-base-msmarco with 10 pseudo-queries per document, and real query-to-docid pairs. We optimize with AdamW (lr 1 10 -3 ) for 40 epochs.

## 5. Objectives, formulas и training details

**Detected objective keywords:** cross-entropy, distillation.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `'_k[] = _k[] + b^ hid_k,\,E[], \, A_k(),`
- `AF^ hist_t(n) \; \; AF^ hist_t-1(n) \;+\; _x X_t-1 1\!,`
- `AF_t(n) \;=\; _x X_t 1\!.`
- `AF_t(n) \;=\; AF_t(n) _n'=1^S^2 AF_t(n').`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- RQ4 (Expanded): Comparison to prior approaches. Diagonal entries report per-slice Hit@10 (\ AP, $ FWT_ diag$, and signed $ BWT^ $ are computed from the lower-triangular MRR@10 matrix (Section ). Best values in each column are underlined.
- RQ3 controls using the frozen $ _0$ checkpoint trained on $ D_0$. D0 Hit@10 is performance on $ Q_ test,0$ with the $ D_0$ trie. Frozen drop is the absolute Hit@10 decrease on $ Q_ test,0$ when decoding expands from $ I( D_0)$ to $ I( D_0:5)$ with no parameter updates. ZS Avg is average strict identifier-transfer Hit@10 on $ Q_ test,t$ for $t=1,,5$ under...
- Sensitivity of PAMT to protected-row ratio $p$ and value-update budget $m$ on MS MARCO with SPQ identifiers under the Expanded protocol. AP, $ FWT_ diag$, and signed $ BWT^ $ are computed from MRR@10; higher is better. $ $ denotes the default setting.
- Stage 1 vs.\ Stage 2 across temporal slices. Hit@10 (\ Dashed lines denote Stage 1 adaptation before PAMT; solid lines denote Stage 2 after PAMT. The $D_0$ point corresponds to the initially trained model. Colors indicate method/docid combinations.
- Stage 1 vs.\ Stage 2 aggregate continual-learning metrics. AP, $ FWT_ diag$, and signed $ BWT^ $ are computed from MRR@10 and reported in percentage points for NQ and MS MARCO under Expanded and Fixed protocols. Hatched bars denote Stage 1 adaptation before PAMT; dotted bars denote Stage 2 after PAMT.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>MS MARCO, NQ, Natural Questions</td></tr>
<tr><th>Metrics</th><td>метрики нужно сверить в таблицах experiments</td></tr>
<tr><th>Baselines</th><td>DSI, NCI</td></tr>
</tbody></table>
</div>

- Evidence: На MS MARCO/NQ sequential adaptation memory tuning улучшает stability-plasticity trade-off.
- Experimental Setup: We describe benchmarks, continual splits, model configurations, adaptation methods, and evaluation. We vary four factors, docid scheme (SPQ, TU), adaptation method (Full FT, LoRA), search-space protocol (Expanded, Fixed), and benchmark (MS MARCO, NQ), to analyze the plasticity--stability trade-off in
- We show that sequential adaptation improves retrieval on newly added documents but substantially degrades performance on earlier slices, exposing a pronounced stability-plasticity trade-off.
- Experiments on MS MARCO and Natural Questions under sequential, disjoint corpus increments show that PAMT substantially improves retention on earlier slices with minimal impact on retrieval performance for newly added documents, while modifying only a sparse subset of memory values per session.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: PAMT добавляет parametric memory head для continual GenIR, чтобы адаптироваться к новым documents без catastrophic forgetting.
- **Algorithmic/system:** Алгоритм: Backbone frozen after adaptation; product-key memory head sparse-queries decoder hidden states и корректирует trie-valid token scores. Memory values обновляются по decoding-time access statistics с fixed budget.
- **Empirical:** Evidence: На MS MARCO/NQ sequential adaptation memory tuning улучшает stability-plasticity trade-off.
- **Practical:** Ограничение: Memory growth, access policy и long-run maintenance требуют отдельного system analysis.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.
- Reward/utility signal достаточно стабилен и не подменяет user relevance узкой бизнес-метрикой.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: MS MARCO, NQ, Natural Questions.
- Метрики: метрики нужно сверить в таблицах experiments.
- Baselines и parity settings: DSI, NCI.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна для dynamic index: GenIR model-as-index нуждается в modular update path, иначе каждый update угрожает старому corpus knowledge.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: PAMT добавляет parametric memory head для continual GenIR, чтобы адаптироваться к новым documents без catastrophic forgetting.
- Алгоритм: Backbone frozen after adaptation; product-key memory head sparse-queries decoder hidden states и корректирует trie-valid token scores. Memory values обновляются по decoding-time access statistics с fixed budget.
- Evidence: На MS MARCO/NQ sequential adaptation memory tuning улучшает stability-plasticity trade-off.
- Ограничение: Memory growth, access policy и long-run maintenance требуют отдельного system analysis.
- Итог: Полезна для dynamic index: GenIR model-as-index нуждается в modular update path, иначе каждый update угрожает старому corpus knowledge.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Generative information retrieval (GenIR) reformulates document retrieval as an autoregressive generation task, mapping natural language queries directly to unique document identifiers (docids) Metzler2021RethinkingS, Tay2022TransformerMA, recent. By...</td></tr>
<tr><td>Related Work</td><td>Continual learning. Catastrophic forgetting mccloskey1989catastrophic, french1999catastrophic remains a central challenge when training neural models on non-stationary data. Common mitigation strategies include experience replay robins1995catastrophic,...</td></tr>
<tr><td>Preliminaries</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Generative information retrieval</td><td>Let D = d 1,,d | D | be a document corpus and Q = q 1,,q | Q | a set of queries. Each document d D is assigned an identifier sequence I d=(t 1,,t |I d| ) whose tokens lie in a vocabulary V. For semantic schemes (e.g., PQ codes), V is augmented with...</td></tr>
<tr><td>Continual adaptation task</td><td>itemize [leftmargin=*,nosep] Session 0 ( t=0 ): The base model 0 is optimized on the initial corpus D 0 and query set Q 0. A prefix trie T 0 is constructed. Subsequent sessions ( t 1 ): A new document set D t and corresponding query set Q t are introduced....</td></tr>
<tr><td>Document identifier schemes</td><td>Semantic product-quantized (SPQ) docids Each document is embedded and quantized using product quantization jegou2010product. The embedding is partitioned into M sub-vectors, each quantized via a codebook of K centroids learned on D 0. The identifier I d is...</td></tr>
<tr><td>Methodology</td><td>We propose post-adaptation memory tuning (PAMT) for continual GenIR. As shown in Figure, PAMT follows an adapt--then--stabilize procedure that separates learning new query--docid mappings from post-adaptation decoder-side calibration. For each...</td></tr>
<tr><td>Parametric memory head (PMH)</td><td>PMH is an external decoding-time component, co-trained with the backbone on D 0 and later used for post-adaptation calibration of docid generation. At each decoding step, it reads the current decoder hidden state, retrieves a small set of relevant memory...</td></tr>
<tr><td>Post-adaptation memory tuning (PAMT)</td><td>PAMT adds a memory-only interface calibration stage after session-level backbone adaptation. The goal is to incorporate new document mappings while preserving index stability, without replaying legacy relevance supervision and without further modifying the...</td></tr>
<tr><td>Experimental Setup</td><td>We describe benchmarks, continual splits, model configurations, adaptation methods, and evaluation. We vary four factors, docid scheme (SPQ, TU), adaptation method (Full FT, LoRA), search-space protocol (Expanded, Fixed), and benchmark (MS MARCO, NQ), to...</td></tr>
<tr><td>Datasets and continual splits</td><td>. We evaluate on MS MARCO Document Ranking Campos2016MSMA and Natural Questions (NQ) kwiatkowski-etal-2019-natural. Following prior GenIR work zhou2022ultron,CLEVER,mdgr,tang2023semantic,tang2024listwise,tang2024generative,tang2024bootstrapped, we use the...</td></tr>
<tr><td>Model architecture and document identifiers</td><td>backbone. We use T5-base. https://huggingface.co/google-t5/t5-base For SPQ docids, the decoder vocabulary is expanded from 32, 128 to 40, 320 tokens by adding 8, 192 dedicated docid tokens organized into M=32 disjoint blocks.</td></tr>
</tbody></table>
</div>
