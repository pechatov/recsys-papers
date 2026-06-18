---
title: "GenSAR: Unifying Balanced Search and Recommendation with Generative Retrieval"
category: "semantic_ids_tokenization_indexing"
slug: "gensar_unifying_balanced_search_and_recommendation_with_generative_summary"
catalogId: "paper-gensar_unifying_balanced_search_and_recommendation_with_generative_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/gensar_unifying_balanced_search_and_recommendation_with_generative_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2504.05730"
---
**Авторы:** Teng Shi, Jun Xu, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Yang Song, Enyun Yu.

**Аффилиации:** Renmin University of China; Kuaishou Technology; Independent.

**Индустрия:** joint search and recommendation на коммерческой платформе.

**Первичный источник:** arXiv source 2504.05730.

## Коротко

- GenSAR строит balanced unified generative model для search и recommendation.
- Ключевая идея - item identifier должен иметь shared semantic/collaborative часть и task-specific часть.
- Модель использует task prompts, чтобы один generator понимал разные цели поиска и рекомендации.

## Контекст

- Search требует query-item semantic relevance; recommendation требует user-item collaborative preference.
- Unified SAR полезен для платформ, где один каталог обслуживает search box и feed.
- Без специальных identifiers multi-task learning может улучшить одну задачу и ухудшить другую.

## Проблема

- Semantic-only SID плохо покрывает collaborative preference.
- Collaborative-only SID хуже для explicit query matching.
- Нужно уменьшить trade-off между задачами, сохранив shared knowledge.

## Метод/архитектура

- RQ-VAE tokenizer обучает shared codebooks и separate codebooks.
- Каждый item получает semantic identifier и collaborative identifier с общей prefix-частью.
- Training stage смешивает search и recommendation examples в generative format.
- Inference использует task prompt и генерирует SID подходящего item.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для GenSAR: Unifying Balanced Search and Recommendation with Generative Retrieval качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Objective/алгоритм

- Tokenizer loss включает reconstruction для semantic/collaborative embeddings и RQ losses для shared/specific residuals.
- Total RQ-VAE loss суммирует reconstruction и quantization/codebook terms.
- Generator loss - autoregressive cross-entropy over target SID.
- Task prompts играют роль routing mechanism между search и recommendation.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для GenSAR: Unifying Balanced Search and Recommendation with Generative Retrieval качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

### Подробная схема алгоритма

GenSAR строит identifier как двухслойный компромисс: часть кода общая для search и recommendation, часть остается task-specific, чтобы одна задача не перезаписала другую.

1. **Собрать два embedding space.** Для item получают semantic embedding из query/content relevance и collaborative embedding из user-item interactions.
1. **Обучить shared RQ-VAE levels.** Первые codebooks кодируют общую структуру item, которая должна быть полезна обеим задачам.
1. **Обучить task-specific residual levels.** Separate codebooks сохраняют search-specific и recommendation-specific остатки, чтобы explicit query relevance и collaborative preference не смешивались насильно.
1. **Сформировать SID.** Item получает identifier вида shared prefix + task-specific suffix; это дает sharing на coarse уровне и specialization на fine уровне.
1. **Смешать training examples.** Search prompt содержит query, recommendation prompt содержит user history; target - SID релевантного item для соответствующей задачи.
1. **Обучить generative model.** Decoder оптимизирует autoregressive cross-entropy по target SID, а task prompt указывает, какую ветку identifier semantics использовать.
1. **На inference включить task prompt.** Для search и recommendation используется один generator, но разные prompts и соответствующие валидные SID paths.

```
semantic_emb = search_encoder(item)
collab_emb = recommendation_encoder(item)

shared_codes = rqvae_shared(semantic_emb, collab_emb)
search_codes = rqvae_search_residual(semantic_emb, shared_codes)
rec_codes = rqvae_rec_residual(collab_emb, shared_codes)

SID_search = shared_codes + search_codes
SID_rec = shared_codes + rec_codes

train generator on:
    "[search] query" -> SID_search(item)
    "[rec] user_history" -> SID_rec(item)
```

## Эксперименты

- Datasets: public Amazon semi-synthetic SAR и коммерческий dataset.
- Метрики: HR@1/5/10 и NDCG@5/10.
- Amazon search: GenSAR HR@1 0.5262, HR@5 0.7529, HR@10 0.8217.
- Recommendation tables показывают сильные HR@1/NDCG@5 gains и меньший trade-off.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для GenSAR: Unifying Balanced Search and Recommendation with Generative Retrieval качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Рисунки/таблицы

- Figure method identifier показывает shared/specific codebooks.
- Performance tables разделены на recommendation и search.
- Analysis figures сравнивают semantic-only, collaborative-only и joint identifiers.
- Identifier length analysis показывает деградацию слишком коротких и слишком длинных codes.

## Ablation conclusions

- Удаление shared information ухудшает перенос между задачами.
- Удаление task-specific частей повышает conflict между search и rec.
- Слишком короткий identifier увеличивает collisions; слишком длинный увеличивает decoding errors.

## Сильные стороны

- **Identifier-level балансировка задач.** Shared prefix переносит общую item semantics, а task-specific suffix защищает search и recommendation от взаимного interference.
- **Проверяемые ablations.** Удаление shared или task-specific частей напрямую проверяет, где находится выигрыш.
- **Один generator для двух режимов.** Task prompts дают понятный routing mechanism без двух полностью независимых моделей.
- **Метод явно работает с trade-off.** Важен не single-task максимум, а одновременное качество search и recommendation.

## Ограничения

- **Зависимость от двух upstream embedding spaces.** Если semantic encoder слаб для query relevance или collaborative encoder шумный, shared/specific decomposition будет наследовать эти ошибки.
- **Нужно тюнить длину identifier.** Слишком короткий SID повышает collisions, слишком длинный увеличивает autoregressive errors и latency.
- **Коммерческий dataset частично закрыт.** Абсолютные gains и свойства overlap трудно воспроизвести вне платформы.
- **Offline баланс может не совпасть с business балансом.** В production search и feed могут иметь разные KPI, где "balanced HR/NDCG" не является правильной целевой функцией.

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
- Специфично для этой статьи: Коммерческий dataset частично закрыт.
- Специфично для этой статьи: Метод требует качественных semantic и collaborative embeddings.
- Специфично для этой статьи: Production KPI search и rec могут конфликтовать сильнее, чем offline HR/NDCG.

## Связь

- Связан с Bridging Search and Recommendation: GenSAR дает architecture, Bridging объясняет условия пользы.
- Связан с FORGE через shared concern о collisions и codebook design.

## Итог

- Главный вывод: unified SAR требует identifier-level балансировки.
- Один генератор работает лучше, когда структура SID явно разделяет общее и task-specific.

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
