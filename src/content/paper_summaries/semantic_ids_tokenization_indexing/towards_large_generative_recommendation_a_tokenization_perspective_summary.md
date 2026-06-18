---
title: "Towards Large Generative Recommendation: A Tokenization Perspective"
category: "semantic_ids_tokenization_indexing"
slug: "towards_large_generative_recommendation_a_tokenization_perspective_summary"
catalogId: "paper-towards_large_generative_recommendation_a_tokenization_perspective_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/towards_large_generative_recommendation_a_tokenization_perspective_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1145/3746252.3761451"
---
> **Авторы:** Yupeng Hou, An Zhang, Leheng Sheng, Jiancan Wu, Xiang Wang, Tat-Seng Chua, Julian McAuley.
>
> **Аффилиации:** University of California San Diego; University of Science and Technology of China; National University of Singapore.
>
> **Индустрия:** CIKM tutorial / large generative recommendation
>
> **Первичные источники:** ACM DOI landing возвращал HTTP 403; использован официальный сайт CIKM 2025 tutorial large-genrec.github.io/cikm2025.html.

## Коротко

This is a CIKM 2025 tutorial paper/site, not a new benchmark method; it frames action tokenization as the foundation of large generative recommendation.

Разбор ниже фокусируется на том, что именно добавляет paper к семейству Semantic ID/tokenization методов.

Отдельно отмечены данные, метрики, ablations и production notes, если они раскрыты в источнике.

## Контекст

The tutorial program covers background, LLM-based GenRec, Semantic IDs, SID-based GenRec, and open challenges.

В этой линии работ tokenization становится частью модели: она определяет, какие items делят параметры, какие ошибки возможны при генерации и как каталог обновляется.

Поэтому оценивать надо не только final NDCG/Recall, но и code utilization, invalid generation, cold-start behavior и serving cost.

## Проблема

Главная проблема paper формулируется как несовпадение удобного identifier space и реального downstream objective.

- Raw IDs сильны для memorization, но плохо переносят знания на новые или редкие items.
- Pure semantic IDs помогают sharing, но могут смешивать items, которые различны для бизнес-метрики.
- Tokenization без domain constraints часто создает коды, которые трудно интерпретировать и сопровождать.
- Generative decoding должен возвращать валидные catalog items, а не произвольные token combinations.
- Offline aggregate metrics скрывают провалы на cold-start, tail, geography/category или task-specific slices.
- Serving требует отдельного контроля latency, индексов, codebook refresh и feature-store compatibility.

## Метод/архитектура

Компоненты, заявленные в paper:

- Key objective: convert human-readable data/user-item interactions into machine-readable token sequences for generative models.
- Open issues include invalid generation, decoding constraints, catalog freshness, alignment, efficiency and scaling.
- Use as checklist for evaluating all other papers in the batch.
- Identifier строится как learned discrete representation, а не как произвольная строка.
- Метод меняет, какие items sharing embedding/code parameters.
- Важное инженерное следствие: tokenizer становится artifact, который надо версионировать и мониторить.
- Для production или large-scale setup нужна стратегия обновления кодов при изменении каталога.

## Objective/алгоритм

Objective и алгоритм связывают construction of IDs с downstream task.

- Loss/constraint design задает inductive bias tokenizer-а.
- Если есть alignment loss, он должен улучшать downstream signal, не разрушая semantic grouping.
- Если есть hierarchy/category/geography constraint, его надо проверять на collapse и over-constraining.
- Если используется autoregressive decoding, нужно контролировать ошибки ранних токенов.
- Если используется parallel/set decoding, нужно контролировать invalid combinations.
- Tokenization categories: item IDs, textual descriptions, semantic IDs.

### Практический алгоритм выбора tokenization strategy

1. **Определить action space.** Сначала решается, что генерирует модель: raw item IDs, text descriptions, semantic IDs, multi-token actions или hybrid representation.
1. **Оценить constraints каталога.** Размер item pool, churn, cold-start rate, content quality и допустимая latency определяют, нужен ли компактный SID, textual ID или feature-only adoption.
1. **Выбрать tokenizer family.** Atomic IDs дают memorization; textual tokens дают language prior; SID/RQ/PQ/hierarchical tokens дают parameter sharing и controlled vocabulary.
1. **Задать decoding constraints.** Для autoregressive IDs нужен trie/code-tree; для unordered/parallel IDs нужен graph или post-validation; для text IDs нужна entity resolution.
1. **Обучить generator под выбранный action language.** User histories переводятся в тот же token space, target item становится token sequence, loss обычно autoregressive NLL или its variants.
1. **Оценивать не только Recall/NDCG.** Обязательны invalid generation, collision rate, code utilization, head/tail/cold-start slices, latency и update cost.
1. **Итерировать tokenization как artifact.** При drift каталога tokenizer пересобирается или дообучается, а old/new mappings должны поддерживать совместимый rollout и rollback.

## Эксперименты

Экспериментальная конкретика из paper:

- Official page states tutorial time: Monday, Nov 10 2025, 13:45-18:00, Room 203; proceedings pages 6821-6824.
- No experimental dataset; the artifact is a taxonomy and lecture program.
- Its limitation is absence of new empirical results.
- Важно читать ablation не как список чисел, а как проверку: какой constraint действительно нужен, а какой сам по себе ухудшает модель.
- Для внутренних datasets переносимость ограничена; зато такие эксперименты лучше показывают serving и business constraints.

## Рисунки/таблицы

Что смотреть в рисунках и таблицах paper.

- Overview figure показывает, где именно SID/tokenizer входит в recommender pipeline.
- Dataset/statistics table задает масштаб и sparsity; без него нельзя судить о применимости.
- Overall performance table показывает aggregate gain, но его надо дополнять slices.
- Ablation table показывает, какие components несут основной вклад.
- Visualization/case-study figures полезны для диагностики semantic/category/geographic consistency.
- Online/production table, если есть, важнее небольших offline gains.

## Сильные стороны

Сильные стороны с практической точки зрения.

- **Хорошая карта пространства решений.** Tutorial четко разделяет item IDs, textual descriptions и semantic IDs как разные action languages для GenRec.
- **Фокус на tokenization как фундаменте.** Работа полезна тем, что ставит tokenizer до выбора backbone: плохой action space не спасается большим LLM.
- **Собирает open challenges в один checklist.** Invalid generation, constraints, freshness, alignment, efficiency и scaling перечислены как обязательные engineering questions.
- **Помогает сравнивать соседние papers.** Через эту рамку проще понять, чем отличаются TIGER, LETTER, STORE, RPG, CoFiRec и Snapchat-style feature use.
- **Подходит для проектирования production roadmap.** Можно начать с SID-as-feature, затем перейти к constrained retrieval, не смешивая эти этапы.

## Слабые стороны и ограничения

Ограничения и риски.

- **Это tutorial, а не новый метод.** Нет нового tokenizer objective, benchmark table или ablation, поэтому выводы являются рамкой анализа, а не эмпирическим доказательством.
- **Нет единого рецепта выбора SID.** Работа перечисляет alternatives, но practitioner все равно должен сам подобрать codebook shape, decoding policy и refresh strategy.
- **Open challenges остаются открытыми.** Invalid generation, freshness и efficiency названы, но не решены алгоритмически.
- **Риск слишком общей taxonomy.** Для конкретного домена решают детали: text quality, churn, business constraints, inventory policy и ranker integration.
- **Нельзя использовать как performance evidence.** Отсутствие новых экспериментов означает, что любой production decision требует отдельного benchmark.
- Главный общий failure mode для SID-подходов: semantic collision выглядит осмысленно offline, но смешивает конкурирующие items в конкретном business objective.
- Второй общий failure mode: tokenizer обучен на старом каталоге и перестает отражать новые категории, бренды, форматы или geography.
- Третий общий failure mode: offline Recall/NDCG улучшается, но serving latency, graph refresh, feature joins или beam constraints съедают online gain.

## Как реализовать/проверять

Практический план внедрения/проверки.

- Сначала воспроизвести baseline ID-only и SID-only.
- Добавить предложенный tokenizer/alignment/constraint как отдельный artifact.
- Сравнивать aggregate metrics и slices: head/tail, cold/warm, geography/category/task.
- Проверять code utilization: perplexity, entropy, top-code concentration, unused codes.
- Проверять examples of collisions вручную: похожи ли items с одинаковым prefix/code.
- Для generative setup логировать invalid ID rate и generated-but-filtered candidates.
- Для online rollout начинать с candidate augmentation или shadow scoring.
- Сделать rollback plan на прежний tokenizer/version.
- Зафиксировать версию tokenizer-а, vocabulary/codebook sizes, seed, дату обучения и покрытие каталога.
- Логировать invalid/generated-not-in-catalog rate отдельно от Recall/NDCG, потому что генеративная модель может улучшать ranking среди валидных, но терять кандидатов на этапе декодирования.
- Делать slice-анализ по popularity bucket, item age, cold-start cohort и длине пользовательской истории.
- Сравнивать не только с SID-only baseline, но и с ID-only/hybrid baseline: в production ID memorization часто остается сильным сигналом.
- Проверять распределение кодов: entropy/perplexity, долю неиспользуемых кодов, top-code concentration и collision examples.
- В online rollout начинать с shadow features или candidate sidecar, чтобы отделить эффект tokenizer-а от эффекта retriever/ranker.

## Связь

Связь с соседними работами.

- Работа находится в общей оси Semantic ID как learned interface между item catalog и model.
- С Meta stability paper ее связывает вопрос representation stability.
- С RPG/SETRec ее связывает вопрос формы identifier и decoding efficiency.
- С CoFiRec/CAT-ID2 ее связывает использование структуры: hierarchy, category, geography или task-specific decomposition.
- С Unified/Harmonizing papers ее связывает компромисс semantic sharing vs exact ID memorization.

## Итог

Итоговый вывод.

- Главный урок: качество Semantic ID определяется не только quantizer-ом, но и тем, с каким downstream signal он выровнен.
- Хороший SID должен быть валидным, компактным, обновляемым, диагностируемым и полезным на нужных slices.
- Без проверки code utilization, cold-start/tail и serving constraints offline gain легко переоценить.
