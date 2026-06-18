---
title: "Understanding Generative Recommendation with Semantic IDs from a Model-scaling View"
category: "semantic_ids_tokenization_indexing"
slug: "understanding_generative_recommendation_with_semantic_ids_from_a_summary"
catalogId: "paper-understanding_generative_recommendation_with_semantic_ids_from_a_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/understanding_generative_recommendation_with_semantic_ids_from_a_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2509.25522"
---
> **Авторы:** Jingzhe Liu, Liam Collins, Jiliang Tang, Tong Zhao, Neil Shah, Clark Mingxuan Ju.
>
> **Аффилиации:** Michigan State University; Snap Inc..
>
> **Первичный источник:** arXiv:2509.25522.

## коротко

This paper studies SID-based GR through scaling laws and finds fast saturation when scaling modality encoder, tokenizer and RS backbone.

Ниже разбор сфокусирован на механике метода, objectives, экспериментальном setup, ablation conclusions и практических рисках внедрения.

- Identifies limited SID capacity as bottleneck.
- Compares SID-based GR with LLM-as-RS.
- LLM-as-RS shows better scaling and up to 20% improvement over best SID-based scaled performance.
- Challenges belief that LLMs cannot learn CF signals.
- Studies models from 44M to 14B parameters.

## контекст

Работа находится в линии generative recommendation / semantic identifiers, где item представляется не только atomic ID, а дискретным, текстовым, многокодовым или генерируемым представлением.

Общий контекст: чем сильнее сжатие item semantics в короткий код, тем больше риск потерять информацию, которую downstream recommender уже не восстановит.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>Paradigms</td><td>SID-based GR versus LLM-as-RS</td></tr>
<tr><td>Datasets</td><td>Amazon Review datasets</td></tr>
<tr><td>Metric model</td><td>Miss Rate / Recall@k scaling</td></tr>
</table>
</div>

## проблема

SIDs compress rich item semantics before the recommender sees them. If quantization discards semantic information, larger downstream models cannot recover it.

Для оценки важно отделять три уровня: качество item representation, качество tokenizer/identifier и качество sequence/generative model. Многие papers выигрывают именно за счет улучшения одного уровня, а не всей системы сразу.

- Tokenizer/recommender mismatch: код удобен одному модулю и неудобен другому.
- Collisions или near-collisions ухудшают exact item retrieval.
- Long-tail items получают слабый сигнал и нестабильные IDs.
- Beam search может генерировать invalid или duplicate identifiers.
- Offline lift может не перенестись в production из-за drift, catalog churn и constraints.

## метод/архитектура

Метод вводит специальные компоненты поверх базового recommender/tokenizer pipeline. Важно, что авторы обычно стараются сохранить inference совместимым с существующей GR схемой или явно обсуждают trade-off inference cost.

Архитектурная идея раскрывается через следующие элементы.

- Scaling equation decomposes semantic information and collaborative filtering components.
- SID-based GR with modality encoder, quantization tokenizer and RS module.
- LLM-as-RS directly consumes item text/title histories.
- LoRA tuning for LLM-as-RS.
- Embedding injection experiments for CF and LLM embeddings.

## objective/алгоритм

Objective связывает representation learning, identifier learning и downstream recommendation/retrieval signal. В некоторых работах это explicit loss, в других — training schedule или iterative relabeling.

Для практической реализации важно логировать каждый компонент loss отдельно: общий Recall/NDCG не объясняет, какой механизм сработал.

- MR@k = 1 - Recall@k as loss proxy.
- Power-law terms for semantic and CF capacity.
- Huber loss plus L-BFGS fitting of scaling parameters.
- Cross-entropy for LLM-as-RS LoRA training.
- Ablations over encoder size, codebook count/size, RS size and LLM size.

### подробная схема алгоритма исследования

Это не новый tokenizer, а экспериментальный protocol для доказательства bottleneck: авторы масштабируют каждый компонент SID-based GR отдельно и сравнивают с LLM-as-RS.

1. **Собрать SID-based pipeline.** Item text/title кодируется modality encoder'ом, затем tokenizer строит SID, после чего recommendation backbone учится генерировать/предсказывать item через SID.
1. **Масштабировать modality encoder.** Увеличивается размер encoder'а, но downstream SID quality проверяет, проходит ли дополнительная semantics через quantization.
1. **Масштабировать tokenizer capacity.** Меняются число codebooks и codebook size; измеряется, где MR@k перестает улучшаться.
1. **Масштабировать RS backbone.** Увеличивается recommender model size; если SID bottleneck уже насыщен, больший backbone не восстанавливает потерянную информацию.
1. **Построить LLM-as-RS baseline.** LLM получает item text/title histories напрямую и fine-tune'ится LoRA cross-entropy objective, без дискретного SID bottleneck.
1. **Подогнать scaling laws.** MR@k curves аппроксимируются power-law terms через Huber loss и L-BFGS, чтобы сравнить saturation rates.
1. **Проверить cold-start и efficiency.** LLM-as-RS сильнее масштабируется, но SID остается дешевле на inference; вывод зависит от compute/latency budget.

```
for encoder_size in encoder_grid:
    embeddings = modality_encoder(encoder_size, item_text)
    sid = tokenizer(embeddings)
    train_sid_gr(backbone_fixed, sid)
    log MR@k

for tokenizer_capacity in codebook_grid:
    sid = tokenizer(embeddings, tokenizer_capacity)
    train_sid_gr(backbone_fixed, sid)
    log MR@k and saturation

for backbone_size in rs_grid:
    train_sid_gr(backbone_size, sid)
    log MR@k

train LLM_as_RS with LoRA on item text histories
fit scaling curves and compare SID bottleneck vs direct text modeling
```

## эксперименты

Эксперименты в статье построены вокруг сравнения с классическими sequential recommenders, TIGER-like GR baselines, tokenizer variants или scaling baselines. Ниже перечислена конкретика setup.

При чтении результатов полезно проверять, совпадает ли inference format у baseline и нового метода: разные beam constraints, token lengths и collision handling могут сильно менять сравнение.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>Datasets</td><td>Amazon Beauty, Sports, Toys-style datasets in appendix</td></tr>
<tr><td>SID tokenizer optimum</td><td>analysis notes 3 codebooks and size 256 as strong setting</td></tr>
<tr><td>Model sizes</td><td>44M to 14B parameters</td></tr>
<tr><td>Cold-start</td><td>LLM-as-RS consistently better in appendix experiment</td></tr>
<tr><td>Efficiency</td><td>SID cheaper at inference; LLM-as-RS can surpass with enough budget</td></tr>
</table>
</div>

## рисунки/таблицы

Рисунки и таблицы в статье полезны как operational checklist: они показывают, какие компоненты надо воспроизводить, а какие являются ablation-only.

Если статья недоступна как production code, именно captions и ablation tables часто дают лучшие подсказки для повторной реализации.

- Two-paradigm figure: SID-based GR versus LLM-as-RS.
- LLM encoder scaling plots: little benefit for SID-based GR.
- Tokenizer codebook scaling plots: quick saturation.
- LLM-as-RS scaling plots with strong fit.
- Efficiency appendix: training/inference trade-off.

## сильные стороны

- **Правильный scaling question.** Работа проверяет не "какой метод лучше в одной точке", а где насыщается encoder, tokenizer и recommender backbone.
- **Изолирует SID capacity bottleneck.** Масштабирование modality encoder'а и RS backbone мало помогает, если короткий SID уже потерял семантику.
- **Сравнивает с сильной альтернативой.** LLM-as-RS напрямую потребляет item text/title histories и показывает, что LLM может учить CF signal при достаточном budget.
- **Есть cold-start и efficiency angle.** Вывод не сводится к quality: SID дешевле на inference, LLM-as-RS лучше масштабируется.
- **Дает практическую диагностику для новых SID papers.** Нужно строить scaling curves и semantic-loss checks, а не только reported NDCG@10.

## ограничения

- **LLM-as-RS дорогой на inference.** Даже если scaling лучше, serving cost может быть неприемлем для high-QPS recommender.
- **Amazon datasets не равны production feed.** Полные промышленные каталоги имеют freshness, policy constraints, multimodal features и feedback loops.
- **Scaling laws зависят от выбранных architectures.** Другие SID designs, multi-identifier systems или adaptive tokenizers могут сдвинуть saturation point.
- **Text/title quality критична для LLM-as-RS.** В доменах с бедным или noisy text преимущество direct text modeling может исчезнуть.
- **Constrained matching для generated titles остается отдельной проблемой.** Direct text generation должна быть надежно заземлена в catalog items.

## как реализовать/проверять

Практический путь — начинать с сильного baseline и добавлять новый механизм как isolated intervention. Нельзя менять tokenizer, backbone, beam search и preprocessing одновременно, иначе lift невозможно интерпретировать.

Ниже — минимальный набор проверок перед доверием к результату.

- Build scaling curves across model sizes.
- Do not infer scalability from one checkpoint.
- Measure semantic loss through quantization.
- Compare SID and text-generation under equal compute and latency budgets.
- Run cold-start split.
- Track inference cost per recommendation.

## связь

Эта работа связана с соседними подходами тем, что пытается уменьшить разрыв между rich item semantics и компактным recommender-friendly представлением.

This paper provides the critique that motivates MoC, LAMIA, MTGRec and richer identifier systems.

<div class="table-scroll">
<table>
<tr><th>Аспект</th><th>Что важно</th></tr>
<tr><td>CoST</td><td>retrieval-aware tokenizer loss</td></tr>
<tr><td>ETEGRec</td><td>end-to-end alignment tokenizer/recommender</td></tr>
<tr><td>MTGRec</td><td>multiple identifiers as pretraining augmentation</td></tr>
<tr><td>MoC/LAMIA</td><td>parallel/multi-aspect semantic representation</td></tr>
<tr><td>Scaling-view</td><td>diagnostics of SID capacity bottleneck</td></tr>
</table>
</div>

## итог

The central lesson is strategic: semantic IDs are efficient, but their capacity limit must be measured before scaling the recommender around them.

Ключевая рекомендация: воспроизводить не только top-line metric, но и diagnostic metrics по кодам, collision, distribution, head/tail и latency.

- Хороший кандидат для controlled offline reproduction.
- Требует versioned SID/token maps при production использовании.
- Нужны ablations по каждому заявленному компоненту.
- Нужно проверять head/tail и cold-start отдельно.
- Нужно явно считать training и inference cost.
