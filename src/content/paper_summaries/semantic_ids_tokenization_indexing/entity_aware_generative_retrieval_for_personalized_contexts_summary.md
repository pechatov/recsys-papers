---
title: "Entity-Aware Generative Retrieval for Personalized Contexts"
category: "semantic_ids_tokenization_indexing"
slug: "entity_aware_generative_retrieval_for_personalized_contexts_summary"
catalogId: "paper-entity_aware_generative_retrieval_for_personalized_contexts_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/entity_aware_generative_retrieval_for_personalized_contexts_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1145/3746252.3761211"
---
**Авторы:** Jihyeong Jeon, Jiwon Lee, Cheol Ryu, U. Kang.

**Аффилиации:** Samsung Electronics; Seoul National University.

**Индустрия:** персональный поиск, цифровая память и ассистенты, где запросы содержат приватные имена, места и временные ссылки.

**Первичный источник:** DOI/ACM metadata и страница Seoul National University Pure; полный ACM PDF напрямую не был доступен.

## Коротко

- PEARL переносит generative retrieval в personalized information retrieval, где query содержит неоднозначные персональные сущности.
- Метод усиливает генератор entity annotation, span-level regularization, prefix-based contrastive learning и context diversification.
- Цель не просто найти похожий текст, а разрешить персональные references вроде nickname, family relation, private location или temporal cue.

## Контекст

- PIR отличается от web search тем, что правильный ответ зависит от частного пользовательского контекста.
- Dense retriever может сопоставить общую семантику, но не понять, что конкретный nickname в одном профиле означает человека, а в другом место.
- Generative retrieval полезен, потому что может напрямую генерировать passage identifier, но без entity bias он тоже страдает от lexical sensitivity.

## Проблема

- Слова в query и passage могут лексически расходиться, хотя referent один и тот же.
- Похожие entity spans создают hard negatives: два документа имеют похожую структуру, но различаются конкретной персональной сущностью.
- Zero-shot evaluation особенно трудна, потому что тестовые имена и private references не встречались в train.

## Метод/архитектура

- Entity-aware annotation добавляет теги сущностей к query и passage: person, location, event, time и другие роли.
- Span-level regularization снижает зависимость от конкретной surface form и заставляет учитывать surrounding context.
- Prefix-based contrastive learning использует структуру identifiers: похожие prefix paths дают естественные hard negatives.
- Context diversification делает synthetic variations через transposition/замены однотипных entities, чтобы модель видела разные формулировки одного намерения.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Entity-Aware Generative Retrieval for Personalized Contexts качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Objective/алгоритм

- Базовый generative retriever обучается генерировать identifier правильного passage.
- Дополнительный contrastive objective притягивает query к позитивному passage identifier representation и отталкивает prefix-close negatives.
- Регуляризация entity span ограничивает слишком большую массу внимания/градиента на одном mention.
- Diversification расширяет train distribution без ручной разметки каждого персонального варианта.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Entity-Aware Generative Retrieval for Personalized Contexts качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

### Подробная схема алгоритма

PEARL можно понимать как generative passage retriever, у которого все трудные места персонального поиска вынесены в entity-aware preprocessing и prefix-aware training.

1. **Построить passage identifiers.** Каждый персональный документ получает генерируемый ID/path, по которому seq2seq retriever должен найти passage.
1. **Разметить entity spans.** Query и passages аннотируются ролями сущностей: person, location, event, time и другими приватными reference types.
1. **Собрать diversified contexts.** Для train examples создаются варианты с перестановками или заменами однотипных сущностей, чтобы модель не запоминала одну surface form.
1. **Обучить базовую генерацию ID.** Encoder получает entity-annotated query/context, decoder генерирует identifier позитивного passage.
1. **Добавить span-level regularization.** Loss не дает модели свести решение к одной строке имени: surrounding context и role должны влиять на retrieval.
1. **Выбрать prefix-close negatives.** Похожие paths в identifier trie используются как hard negatives: они структурно близки, но относятся к другой персональной сущности.
1. **Оптимизировать contrastive term.** Query representation притягивается к позитивному passage/path и отталкивается от prefix-close negatives.
1. **На inference выполнять constrained generation.** Decoder генерирует passage identifier только по валидным prefix paths, затем ID маппится обратно в документ.

```
annotate query and passages with entity roles
for each training query:
    positive = target passage identifier
    negatives = passages with close identifier prefixes
    augmented = diversify_entities(query, passages)
    optimize generation_loss(query -> positive)
    optimize span_regularization(entity spans)
    optimize contrastive_loss(query, positive, negatives)

at inference:
    generate valid passage identifier under prefix constraints
    return mapped personalized passage
```

## Эксперименты

- Оценка включает существующий PIR dataset и новый synthetic benchmark PAIR.
- Метрики из источника: Hits@1 и MRR@10; основной режим заявлен как zero-shot evaluation.
- PEARL достигает state-of-the-art на этих метриках и стабильно лучше strong sparse/dense/generative baselines.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Entity-Aware Generative Retrieval for Personalized Contexts качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Рисунки/таблицы

- Пример entity annotation показывает, как обычный запрос превращается в structured query с entity roles.
- Prefix contrastive figure иллюстрирует позитивные и hard-negative documents в trie/identifier space.
- Ablation table должна подтверждать вклад annotation, prefix contrastive learning и diversification.

## Ablation conclusions

- Без entity annotation модель должна сильнее путать homonymous/private mentions.
- Без prefix contrastive learning уменьшается способность различать структурно похожие, но разные passages.
- Без diversification ухудшается перенос на paraphrase и unseen personal references.

## Сильные стороны

- **Точная постановка PIR.** Метод не притворяется обычным semantic matching: он моделирует private nicknames, family relations, locations и temporal cues как сущности с ролями.
- **Hard negatives берутся из identifier structure.** Prefix-close negatives проверяют именно ту ошибку, которая вредна generative retrieval: почти правильный путь к другому passage.
- **Diversification решает zero-shot surface forms.** Перестановки и замены однотипных entities учат модель не переобучаться на конкретные имена.
- **Механизмы диагностируемы.** Можно отдельно смотреть ошибки annotation, span attention, prefix collisions и вклад diversification.

## Ограничения

- **Детали paper частично недоступны в текущем summary.** Полный ACM PDF не был доступен, поэтому численные гиперпараметры и точная формула loss могут быть неполными.
- **Entity annotation становится критическим dependency.** Ошибка в роли сущности может направить contrastive learning против правильного passage.
- **Synthetic PAIR ограничивает внешнюю валидность.** Реальные персональные данные имеют privacy constraints, sparse histories и memory drift, которые синтетика покрывает не полностью.
- **Identifier prefix сам может создавать bias.** Если passage IDs плохо организованы, prefix negatives будут отражать артефакт индекса, а не реальные трудные случаи.

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
- Специфично для этой статьи: Полный текст DOI не удалось получить напрямую, поэтому часть деталей недоступна.
- Специфично для этой статьи: Синтетический PAIR может не покрывать реальные privacy, sparsity и user-memory drift.
- Специфично для этой статьи: Метод зависит от качества entity annotation и от того, насколько корректно entity roles отражают персональный контекст.

## Связь

- Связан с generative document retrieval и с semantic ID recommendation через идею prefix-aware hard negatives.
- В отличие от item-tokenization работ, PEARL работает с passage identifiers и персональными entities.
- Идея span regularization может быть перенесена на товары с брендами, локациями и temporal campaigns.

## Итог

- Главный вывод: personalized generative retrieval требует entity-aware структуры.
- Semantic/generative identifier сам по себе не решает ambiguity; нужны роли сущностей и hard negatives на уровне prefix.

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
