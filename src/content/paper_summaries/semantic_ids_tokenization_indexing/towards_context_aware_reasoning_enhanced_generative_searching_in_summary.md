---
title: "Towards Context-aware Reasoning-enhanced Generative Searching in E-commerce"
category: "semantic_ids_tokenization_indexing"
slug: "towards_context_aware_reasoning_enhanced_generative_searching_in_summary"
catalogId: "paper-towards_context_aware_reasoning_enhanced_generative_searching_in_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/towards_context_aware_reasoning_enhanced_generative_searching_in_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2510.16925"
---
**Авторы:** Zhiding Liu, Ben Chen, Mingyue Cheng, Enhong Chen, Li Li, Chenyi Lei, Wenwu Ou, Han Li, Kun Gai.

**Аффилиации:** University of Science and Technology of China; Kuaishou Technology.

**Индустрия:** e-commerce search-based recommendation.

**Первичный источник:** arXiv source 2510.16925.

## Коротко

- CRS adds explicit reasoning to generative e-commerce search.
- Context is serialized as JSON/text and aligned with SID prediction.
- Self-evolving post-training alternates SFT and RL; R-GRPO fixes ranking bias.

## Контекст

- Queries are noisy/incomplete; context includes history, clicks, non-clicks, geography and time.
- LLMs can reason over structured context if domain aligned.
- Search output is ranked list, not one answer.

## Проблема

- Reasoning trajectories are scarce.
- Distilling large LLM can introduce data-prior and model-capacity bias.
- Standard GRPO ignores list quality when top-1 wrong.

## Метод/архитектура

- Context representation includes user profile, historical queries, clicked/non-clicked items and current query metadata.
- Item context includes title, price, brand, category and GMV.
- Alignment includes context reconstruction and Context2SID prediction.
- Self-evolving loop trains on hard wrong cases with RL then consolidates solved trajectories with SFT.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Towards Context-aware Reasoning-enhanced Generative Searching in E-commerce качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Objective/алгоритм

- Rewards cover reasoning structure, reasoning length, prediction accuracy and validity.
- R-GRPO decouples reasoning rollout and beam search over items.
- Rank-aware weighted reward summarizes full beam list.
- Constrained prefix trie ensures valid SID generation.

### Пошаговая схема CRS

CRS строит generative search как reasoning-over-context задачу: модель сначала учится читать JSON/text context и SID, затем в self-evolving loop улучшает reasoning на hard cases, а R-GRPO дает reward по полному ranked list, а не только top-1.

1. **Сериализовать context.** Для каждого request собираются user profile, recent queries, clicks/non-clicks, geography/time и current query metadata; item context содержит title, price, brand, category и GMV.
1. **Alignment SFT.** Модель обучается context reconstruction и Context2SID prediction, чтобы связать structured context с valid item identifiers.
1. **Initial reasoning generation.** LLM генерирует reasoning trajectory и SID candidates; constrained prefix trie не дает выйти за catalog vocabulary.
1. **Выделить hard wrong cases.** Self-evolving loop фокусируется на примерах, где initial SFT ошибается или ranked list слабый.
1. **R-GRPO update.** Reasoning rollout отделяется от beam search по items; reward агрегирует structure, length, validity и rank-aware weighted quality всего beam list.
1. **Consolidation SFT.** Успешные trajectories возвращаются в supervised set, чтобы модель не забывала найденные решения после RL iterations.

```
prepare_example:
    context = serialize_json(user_profile, history, clicks, non_clicks, query, geo_time)
    item_context = serialize_items(title, price, brand, category, gmv)
    target_sid = item_to_sid[target_item]

alignment_sft:
    loss = context_reconstruction_loss(context)
         + context_to_sid_loss(context, target_sid)
    update(model)

self_evolving_loop:
    hard_cases = collect_wrong_or_low_rank_cases(model, validation_data)
    for case in hard_cases:
        reasoning_rollout = model.generate_reasoning(case.context)
        beam_list = constrained_beam_search(model, reasoning_rollout, sid_prefix_trie)
        reward = structure_reward(reasoning_rollout)
               + length_reward(reasoning_rollout)
               + validity_reward(beam_list)
               + rank_aware_reward(beam_list, case.target_item)
        R_GRPO_update(model, reasoning_rollout, reward)
    solved = collect_successful_trajectories(model, hard_cases)
    SFT_update(model, solved)
```

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Towards Context-aware Reasoning-enhanced Generative Searching in E-commerce качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Эксперименты

- Datasets: All-100K, All-50K, Fashion-27K from Kuaishou logs.
- 10 most recent interactions retained per user.
- Metrics: HR@1/5/10 and NDCG@1/5/10.
- CRS improves 3-11% on large datasets and up to 47% on Fashion-27K.
- Self-evolving improves NDCG@10 up to 39.02% and 21.21% over initial SFT.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Towards Context-aware Reasoning-enhanced Generative Searching in E-commerce качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Рисунки/таблицы

- Framework figure: representation/alignment, self-evolving, R-GRPO, inference.
- Tables: dataset stats, main results, R-GRPO vs GRPO, scaling.
- Case study shows reasoning trajectory and notes possible repetition/inconsistency.

## Ablation conclusions

- R-GRPO outperforms GRPO across RL iterations.
- Larger backbones 1.7B/4B improve Fashion-27K.
- Cold-start group gains show value of reasoning with limited context.

## Сильные стороны

- **Ranking objective mismatch решается явно.** R-GRPO учитывает ranked list, поэтому top-1 miss не превращает весь rollout в одинаково плохой сигнал.
- **Context format прагматичен.** JSON/text serialization позволяет подключить user/query/item signals без построения отдельной сложной feature architecture.
- **Self-evolving loop работает на hard cases.** Модель не просто дообучается на всем датасете, а целенаправленно исправляет ошибки initial SFT.
- **Constrained prefix trie встроен в алгоритм.** Valid SID generation контролируется во время beam search, а не маскируется post-filtering.
- **Работа честно обсуждает distillation bias.** Это важно, потому что reasoning trajectories от larger LLM могут быть не только полезными, но и систематически смещенными.

## Ограничения

- **Reasoning добавляет latency.** Даже если SID beam constrained, генерация reasoning trajectory дороже обычного generative search.
- **Generated reasoning не всегда надежен.** Case study уже отмечает repetition/inconsistency; plausible explanation не доказывает causal decision process.
- **Private Kuaishou logs ограничивают воспроизводимость.** All-100K/All-50K/Fashion-27K отражают конкретный traffic mix и feature availability.
- **Reward design может быть хрупким.** Structure, length, validity и rank-aware weights задают разные incentives; неправильные веса приведут к reward hacking.
- **JSON context может не масштабироваться без отбора.** Слишком длинный user/item context ухудшит latency и может разбавить ключевые сигналы.

## Как реализовать/проверять

1. Зафиксировать версию каталога, train/eval split и mapping item/document -> identifier; без этого невозможно понять, улучшает ли метод саму модель или только меняет пространство кандидатов.
1. Считать отдельно качество tokenization и качество generator/ranker: collision rate, utilization, Gini, valid-path rate, Recall/HR/NDCG/MRR и latency должны лежать в одном отчете.
1. Делать ablation не только по среднему качеству, но и по head/torso/tail, cold-start, new users/new items, long-history и категориям с похожими объектами.
1. Проверять деградацию при обновлении каталога: semantic IDs могут устаревать, а генератор может помнить старые пути.
1. Сохранять обратный индекс identifier -> item/document и явно логировать случаи many-to-one collisions.
1. Для генеративного вывода использовать constrained decoding или post-validation, иначе invalid identifiers будут маскироваться в offline метриках.
1. В production считать стоимость не только модели, но и перестроения codebook, trie/index, feature pipeline и мониторинга drift.
1. В отчетах отделять retrieval-stage gains от downstream ranking/business metrics, потому что рост HR@K не всегда дает CTR/CVR uplift.

## Failure modes и мониторинг

- Identifier collapse: малая часть кодов получает большую долю объектов, а генератор начинает переиспользовать популярные пути.
- Semantic-collaborative mismatch: похожие по тексту объекты имеют разные user intents, или наоборот, совместно потребляемые объекты текстово далеки.
- Exposure bias autoregressive generation: ошибка раннего токена полностью меняет candidate path.
- Popularity bias: модель учится генерировать frequent IDs и выглядит хорошо на aggregate, но теряет novelty и tail coverage.
- Evaluation leakage: если ID/tokenizer обучался на будущем каталоге или post-training видел target-like сигналы, offline gain завышен.
- Serving mismatch: offline beam шире и медленнее production beam, поэтому качество не переносится напрямую.
- Специфично для этой статьи: Reasoning adds latency.
- Специфично для этой статьи: Generated reasoning may repeat or contradict itself.
- Специфично для этой статьи: Private logs limit reproducibility.

## Связь

- Related to UniSearch and GenSAR.
- Adds reasoning/RL layer over semantic ID generation.

## Итог

- CRS shows that context-aware search needs rank-aware reasoning.
- For e-commerce, top-1-only RL is the wrong abstraction.

## Минимальный план воспроизведения

<div class="table-scroll">
<table>
<thead>
<tr><th>Шаг</th><th>Что сделать</th><th>Что считать успехом</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>Собрать исходные item/document features и interaction/query logs в той же временной схеме, что у статьи.</td><td>Нет leakage, воспроизводимы splits и отрицательные примеры.</td></tr>
<tr><td>2</td><td>Построить identifiers/token representations и сохранить mapping plus collision report.</td><td>Utilization и collision metrics понятны до обучения generator.</td></tr>
<tr><td>3</td><td>Обучить baseline без нового компонента и полный метод с тем же budget.</td><td>Сравнение честное по compute, beam, candidates и preprocessing.</td></tr>
<tr><td>4</td><td>Запустить ablations по каждому заявленному компоненту.</td><td>Каждый компонент дает объяснимый вклад или честно признается избыточным.</td></tr>
<tr><td>5</td><td>Проверить production-like constraints: latency, invalid IDs, refresh, monitoring.</td><td>Offline gain не исчезает при реальных ограничениях serving.</td></tr>
</tbody>
</table>
</div>

Примечание: если в источнике не раскрыты приватные production details, summary явно помечает такие ограничения и не выдумывает закрытые числа.
