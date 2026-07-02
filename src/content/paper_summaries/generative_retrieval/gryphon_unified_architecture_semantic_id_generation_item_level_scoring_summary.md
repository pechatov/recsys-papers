---
title: "Gryphon: A Unified Architecture for Semantic-ID Generation and Item-Level Scoring in Industrial Recommendations"
category: "generative_retrieval"
slug: "gryphon_unified_architecture_semantic_id_generation_item_level_scoring_summary"
catalogId: "paper-gryphon_unified_architecture_semantic_id_generation_item_level_scoring_summary"
paperUrl: "https://arxiv.org/abs/2606.08604"
---
> **Авторы:** Daria Tikhonovich, Oleg Sorokin, Vladislav Dodonov, Mariia Ulianova, Ilya Murzin.
>
> **Аффилиации:** Yandex.
>
> **Источник:** arXiv:2606.08604v1 от 2026-06-09.

## 1. Коротко: о чем статья

Gryphon - industrial generative retrieval architecture, которая явно разделяет два шага: SID-level candidate generation и item-level scoring. Vanilla GR ранжирует generated SID sequences по beam likelihood. Но sequence likelihood не является хорошим item relevance score, а при SID collision несколько concrete items имеют один и тот же SID. Gryphon оставляет beam search только как способ предложить candidate SIDs, затем раскрывает SIDs в items и переоценивает их через Item-Level Scoring Module (ILSM).

Важность статьи в production evidence. Gryphon тестируется на крупной music recommendation platform Yandex и в 7-дневном A/B заменяет 15+ candidate generators и preranking stage одним Gryphon source. Primary metric total listening time меняется на +0.25% без статистической значимости, но active users ratio растет на +0.43%, unfinished tracks падают на -1.3%, а pipeline становится намного проще.

<figure class="paper-figure">
  <img src="../../assets/gryphon/inference.png" alt="Gryphon inference with SID beam search and item-level scoring module">
  <figcaption>Рисунок 1. Gryphon использует decoder beam search для генерации SID candidates, раскрывает их в concrete items и ранжирует items через ILSM, отбрасывая beam likelihood как final relevance score.</figcaption>
</figure>

## 2. Проблема vanilla SID generation

SID-based GR обычно обучает encoder-decoder модель генерировать semantic identifier next item'а. При inference beam search возвращает top-K SID sequences, отсортированные по произведению token probabilities. Это создает две проблемы.

Первая - **SID collisions**. Если несколько items sharing same SID, beam likelihood одинаков для всей collision group и не может различить конкретные items.

Вторая - **likelihood-relevance mismatch**. Даже без collision sequence probability не обязана быть лучшим score для item relevance. Autoregressive model оптимизирует next-token prediction, а production recommendation обычно требует item-level objectives: engagement, freshness, long-term value, multi-objective trade-offs.

Gryphon предлагает не пытаться сделать SID likelihood универсальным scorer-ом. Beam search должен искать reachable semantic neighborhoods, а item-level scorer должен выбрать items.

## 3. Architecture: SID generator and ILSM

### 3.1. Shared encoder

Для user history Gryphon считает encoder states `E_u`. Эти states шарятся между decoder и ILSM. Decoder использует их для autoregressive SID generation, ILSM - для item scoring. Это сохраняет один user representation backbone и не превращает architecture в две независимые модели.

### 3.2. Generative retrieval branch

Decoder генерирует top-K SID candidates. Важный implementation detail: beam likelihood используется только для membership в candidate pool, но не как final ranking score.

Generated SID `sigma` раскрывается в collision group `C_sigma`, то есть множество items с таким identifier. Candidate item set `I_u` - объединение всех items, достижимых через generated SID beam.

### 3.3. Item-Level Scoring Module

ILSM строит item-query embedding через item tower с item-level features. В experiments авторы используют только item-id feature, чтобы не дать Gryphon unfair feature advantage над vanilla GR. Затем lightweight item-to-user cross-attention block и MLP выдают scalar relevance score.

Objective в статье - sampled softmax next-item prediction с LogQ correction. Полный loss: `L = L_gen + lambda L_NIP`, где generation и item-level scoring обучаются совместно.

## 4. Offline evaluation

Данные: неделя реальных interactions music platform с tens of millions active users and items. Main metric - Recall@1000, потому что retrieval candidates дальше идут в production ranker.

<div class="table-scroll">
<table>
<thead><tr><th>Method</th><th>SIDs</th><th>Recall@10</th><th>Recall@1000</th></tr></thead>
<tbody>
<tr><td>ARGUS production baseline</td><td>-</td><td>0.0996</td><td>0.6582</td></tr>
<tr><td>Vanilla GR</td><td>3 x 32000</td><td>0.1961</td><td>0.8245</td></tr>
<tr><td>Vanilla GR Resolved</td><td>2 x 1024</td><td>0.2077</td><td>0.8343</td></tr>
<tr><td>Gryphon</td><td>3 x 32000</td><td>0.2178</td><td>0.8552</td></tr>
</tbody>
</table>
</div>

Gryphon выигрывает у vanilla GR и resolved variant при примерно сопоставимом parameter count и inference budget. Разница превышает наблюдаемую stochastic variability около +/-0.003, хотя формального multi-seed significance testing авторы не делают.

## 5. ILSM ablation

Самая содержательная ablation фиксирует один и тот же beam-generated SID pool (`K=2048`) и меняет scoring:

- SID-level beam scores: Recall@1000 0.8404;
- item-level beam scores после resolution: 0.8209;
- item-level ILSM: 0.8552.

Это показывает два эффекта. Во-первых, collision resolution снижает item-level recall, если использовать только beam likelihood. Во-вторых, ILSM может поднять items из SID paths, которые beam rank поставил ниже top-1000, и тем самым превзойти SID-level ceiling. Это прямое evidence, что beam likelihood miscalibrated как item scorer.

## 6. Online A/B

Контроль: production candidate-generation stack из 15+ heterogeneous generators, 10,000 initial candidates, preranking, затем 3,000 candidates в final ranker.

Treatment: Gryphon как единственный candidate source, 1,000 candidates напрямую в тот же final ranker, без preranking stage.

Результат 7-дневного A/B:

- Total listening time: +0.25%, статистически незначимо;
- Active users ratio: +0.43%, `p < 0.001`;
- Unfinished tracks: -1.3%, `p < 0.001`;
- число candidates passed to ranker снижено с 3,000 до 1,000.

Интерпретация аккуратная: Gryphon не доказал значимый рост primary engagement metric, но показал, что может заменить сложный candidate stack без деградации primary metric и с улучшением secondary quality metrics.

## 7. Сильные стороны

- Paper прямо признает, что SID likelihood не равен item relevance.
- ILSM решает collision problem без unbounded resolving token vocabulary.
- Offline ablation хорошо изолирует benefit item-level scorer-а.
- Online A/B особенно ценно: большинство SID papers остаются offline.

## 8. Ограничения и вопросы

Данные и код закрыты. Воспроизвести production conclusions нельзя, можно только использовать architecture pattern.

ILSM в статье обучен next-item objective. Авторы сами отмечают, что richer objectives - multi-objective engagement, distillation from production ranker, long-term value - остаются future work. Поэтому Gryphon не отменяет final ranker.

Online result по primary metric незначимый. Это не "Gryphon улучшил TLT", а "Gryphon упростил candidate generation без statistically significant TLT loss и улучшил secondary metrics".

## 9. Вывод

Gryphon - сильный industrial аргумент в пользу hybrid view: **генератор должен предлагать semantic candidate space, а item-level scorer должен отвечать за конкретный ranking**. Для semantic-ID production это практичнее, чем пытаться заставить autoregressive SID likelihood одновременно быть retrieval probability, collision resolver и business relevance score.
