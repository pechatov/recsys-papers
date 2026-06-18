---
title: "CoFiRec: Coarse-to-Fine Tokenization for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "cofirec_coarse_to_fine_tokenization_for_generative_recommendation_summary"
catalogId: "paper-cofirec_coarse_to_fine_tokenization_for_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/cofirec_coarse_to_fine_tokenization_for_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2511.22707"
---
> **Авторы:** Tianxin Wei, Xuying Ning, Xuxing Chen, Ruizhong Qiu, Yupeng Hou, Yan Xie, Shuang Yang, Zhigang Hua, Jingrui He.
>
> **Аффилиации:** University of Illinois Urbana-Champaign; Meta; University of California San Diego.
>
> **Индустрия:** Meta / generative recommendation research
>
> **Первичные источники:** arXiv:2511.22707 HTML/PDF.

## Коротко

CoFiRec imposes a coarse-to-fine structure on tokenization and autoregressive generation, matching how users refine preferences.

Разбор ниже фокусируется на том, что именно добавляет paper к семейству Semantic ID/tokenization методов.

Отдельно отмечены данные, метрики, ablations и production notes, если они раскрыты в источнике.

## Контекст

Existing SID sequences can be arbitrary; CoFiRec makes positions semantically meaningful: broad group first, specific item later.

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

- Inference uses earlier coarse tokens to constrain later fine choices.
- Automatic Hierarchy Construction appendix covers cases without a clean taxonomy.
- Runtime/scalability analysis is included in appendix.
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
- Components: theoretical reformulation, coarse-to-fine tokenizer, coarse-to-fine autoregressive generation.

### Пошаговая схема CoFiRec

1. **Задать coarse-to-fine пространство.** Item identifier разбивается на уровни: ранние токены должны описывать broad semantic group, поздние - уточнять конкретный item внутри этой группы.
1. **Построить или восстановить hierarchy.** Если в домене есть чистая taxonomy, она используется как prior; если нет, Automatic Hierarchy Construction строит coarse groups из item representations так, чтобы lower-level choices были условно зависимы от prefix.
1. **Обучить tokenizer с positional meaning.** Tokenizer назначает item'у последовательность, где позиция не произвольна: первый токен ограничивает область поиска, следующие токены уточняют fine-grained identity.
1. **Подготовить training targets.** User history переводится в sequences coarse-to-fine IDs; target item также представлен как hierarchical token path.
1. **Обучить autoregressive generator.** Модель предсказывает target path слева направо. Ошибка верхнего уровня штрафуется особенно сильно practically, потому что она отсекает весь correct branch.
1. **Использовать prefix constraints на inference.** После выбора coarse token decoder ограничивает допустимые fine tokens только descendants выбранного prefix. Это снижает invalid combinations и ускоряет поиск.
1. **Проверить hierarchy diagnostics.** Нужно считать code distribution по уровням, cold-start slices и cases top-level mistakes, потому что хороший aggregate NDCG может скрывать плохое coarse partitioning.

## Эксперименты

Экспериментальная конкретика из paper:

- Experiments include overall performance, ablation, codebook size, ID distribution, cold-start case study, automatic hierarchy construction and backbone adaptability.
- Metrics are sequential/generative recommendation metrics such as Recall/NDCG/Hit depending on setup.
- Failure mode: top-level error can remove the correct item before fine generation.
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

- **Позиции ID получают смысл.** В отличие от произвольного SID tuple, CoFiRec делает ранние токены coarse decisions, а поздние - refinement, что проще диагностировать.
- **Inference использует структуру tokenizer-а.** Coarse prefix сужает fine choices, поэтому tokenization и decoding работают как один hierarchical retrieval process.
- **Есть путь для доменов без taxonomy.** Automatic Hierarchy Construction снижает зависимость от ручного дерева категорий.
- **Хорошо подходит для user preference refinement.** Модель естественно учится сначала выбирать broad intent, потом конкретизировать item.
- **Ablations по codebook size / ID distribution / cold-start полезны для переноса.** Они показывают, где иерархия помогает, а где over-constraining может навредить.

## Слабые стороны и ограничения

Ограничения и риски.

- **Top-level ошибка дорогая.** Если generator выбирает неверный coarse token, правильный item становится недоступен для fine decoding.
- **Hierarchy может закрепить старый bias.** Category/geography/tree prior полезен, только если он соответствует реальному user intent; иначе sharing будет неправильным.
- **Automatic hierarchy тоже надо мониторить.** Кластеры могут быть красивыми semantic groups, но смешивать товары с разной substitutability или business value.
- **Сложнее обновлять каталог.** Новый item надо вставить не только в codebook, но и в правильный branch; массовый refresh меняет prefix distribution.
- **Coarse-to-fine assumption не универсальна.** Для domains, где пользовательский intent пересекает категории, жесткая hierarchy может быть хуже flat или multi-aspect SID.
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
