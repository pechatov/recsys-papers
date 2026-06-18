---
title: "Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices"
category: "semantic_ids_tokenization_indexing"
slug: "semantic_ids_for_recommender_systems_at_snapchat_use_summary"
catalogId: "paper-semantic_ids_for_recommender_systems_at_snapchat_use_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/semantic_ids_for_recommender_systems_at_snapchat_use_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.03949"
---
> **Авторы:** Clark Mingxuan Ju, Tong Zhao, Leonardo Neves, Liam Collins, Bhuvesh Kumar, Jiwen Ren, Lili Zhang, Wenfeng Zhuo, Vincent Zhang, Xiao Bai, Jinchao Li, Karthik Iyer, Zihao Fan, Yilun Xu, Yiwen Chen, Peicheng Yu, Manish Malik, Neil Shah.
>
> **Аффилиации:** Snap Inc.

## Коротко

Это не одна новая модель, а practitioner's paper про то, как Snapchat использует semantic IDs: как auxiliary features в rankers, как retrieval source в generative retrieval и как объект инженерного дизайна. Центральная мысль: uniqueness SID важна, но не является "золотым стандартом" сама по себе.

## Контекст

В коротком видео и social content каталог быстро меняется, а raw IDs плохо переносят семантическую близость. SIDs позволяют передать ranker'у/генератору coarse semantic grouping, но их надо оценивать в реальной системе: offline AUC, online metrics, latency, sequence length и collision handling.

## Проблема

Academic SID papers часто фокусируются на генеративном retrieval. Snapchat показывает более широкий набор use cases: SID как feature для существующего ranker'а может дать быстрый production gain, даже если full generative retrieval еще не готов. При этом высокая uniqueness может ухудшить sharing и не гарантировать лучший GR performance.

## Метод / архитектура

Рассматриваются разные SID generation treatments, включая GraphHash-like SIDs. В ranker SIDs добавляются как auxiliary categorical/sequence features. В retrieval SIDs используются как direct retrieval sources с длинной user sequence. Для GR экспериментов на Amazon Beauty сравниваются codebook shapes и добавляется deduplicate token для честного сравнения.

## Objective / алгоритм

Для SID-as-feature objective остается ranking loss; SID служит дополнительным input, расширяющим item/user context. Для SID-as-retrieval objective ближе к generative recommendation: модель генерирует SID candidates и оценивается Recall/NDCG. Design choice заключается в подборе codebook shape, обработки collisions и балансе semantic sharing vs uniqueness.

### Пошаговая схема Snapchat SID adoption

1. **Сгенерировать SID treatment.** Для videos/items строится GraphHash-like или другой SID mapping; вместе с ним считаются uniqueness, collision groups и coverage новых объектов.
1. **Запустить low-risk path: SID as feature.** SID добавляется в existing ranker как categorical feature для target item и как sequence feature для user history. Objective ranker'а не меняется.
1. **Проверить feature value.** Offline считаются AUC/UAUC, calibration и feature importance; online A/B показывает, дает ли semantic grouping lift без изменения retrieval stack.
1. **Отдельно тестировать direct retrieval.** Для GR setup user history длиной до 480 переводится в SID sequence, модель генерирует SID candidates, а mapping SID -> item возвращает videos.
1. **Настроить collision policy.** Deduplicate token или other collision handling добавляется только после анализа: максимальная uniqueness не является целью, потому что tail может выигрывать от shared coarse codes.
1. **Подобрать codebook shape.** Сравниваются ширина и глубина codebook по Recall/NDCG, sequence length, uniqueness и latency. Побеждает shape с лучшим downstream metric, а не с максимальной уникальностью.
1. **Мониторить freshness.** Для short-form content SID backfill, drift, hot new videos и feature-store availability становятся такими же важными, как offline SID metrics.

## Эксперименты и метрики

Feature-use case оценивается offline AUC gains, где даже 0.01% считается значимым, и online A/B для разных rankers. Retrieval-use case оценивается offline ranking-based metrics Recall/NDCG и online A/B для short-form video recommendation с sequence length 480. Отдельные таблицы показывают uniqueness и Recall@10 на Amazon Beauty и uniqueness на internal data при разных treatments.

## Рисунки / таблицы

Таблицы auxiliary feature показывают offline/online improvements после запуска GraphHash SID. Таблицы retrieval показывают metrics direct SID sources. Таблица codebook shape vs uniqueness vs Recall@10 важна концептуально: она демонстрирует, что максимальная уникальность не обязательно дает лучший generative retrieval, потому что некоторая совместность кодов полезна.

## Сильные стороны

- Показывает несколько production use cases, а не один benchmark.
- Разделяет SID как feature и SID как retrieval target.
- Критически обсуждает uniqueness как неполный критерий качества.

## Слабые стороны и ограничения

Много внутренних данных и агрегированных gains; воспроизвести результаты полностью нельзя. Paper больше про design choices, чем про строго определенный новый algorithm. Для открытых датасетов scope уже, чем для Snapchat production.

## Как реализовать / проверять

Сначала стоит внедрить SID как auxiliary feature в ranker: это меньше меняет serving. Проверять AUC/UAUC, calibration, feature importance и online engagement. Для retrieval нужно отдельно валидировать sequence length, candidate coverage, invalid SID, uniqueness/collision, head-tail quality и latency. Не использовать uniqueness как единственный gate для запуска.

## Связь с соседними работами

Snapchat paper хорошо объясняет, почему AdaSID не должен просто уничтожать все overlap'ы, а SID-Coord координирует HID/SID в ranking. Он также служит практическим контекстом для GLASS/RecoChain: generative retrieval должен уживаться с production rankers.

## SID as auxiliary feature

Первый production use case - добавить SIDs в уже существующие rankers.

Это низкорисковый путь, потому что candidate generation и ranking stack сохраняются.

SID работает как compact semantic feature, который дополняет sparse IDs и dense features.

Offline improvements измеряются AUC gains.

В статье прямо указано, что gain 0.01% уже считается значимым для large-scale ranker.

Online tables показывают improvements после запуска GraphHash SID как auxiliary feature.

## SID as direct retrieval source

Второй use case - использовать SIDs как источник retrieval.

Здесь user sequence может иметь длину 480, что отражает long-history video consumption.

Модель работает с SID sequences, а не только с atomic item IDs.

Offline metrics обозначаются R/N, то есть Recall и NDCG.

Online A/B проверяет short-form video recommendation, где retrieval source должен давать достаточно свежих и релевантных candidates.

## Uniqueness is not the golden standard

Самый важный методологический раздел - критика uniqueness.

Высокая uniqueness означает, что меньше items делят один SID.

Но если uniqueness максимальна, semantic sharing исчезает.

Для tail items полезно делить coarse codes с похожими head или mid-tail items.

Amazon Beauty table показывает codebook shape, uniqueness и Recall@10.

Вывод: лучший Recall@10 не обязан соответствовать максимальной uniqueness.

## Codebook shape choices

Codebook shape задает количество уровней и размер каждого уровня.

Более широкий первый уровень повышает coarse discrimination.

Более глубокий code увеличивает fine-grained capacity, но удлиняет generation.

Deduplicate token добавляется, чтобы честно сравнивать codebook shapes при collision handling.

Для production shape надо выбирать по downstream metric, latency и валидности generated sequences.

## Technical challenges

Первый challenge - fresh content и rapid catalog churn.

Второй - collision handling между visually/semantically similar videos.

Третий - serving integration: SIDs должны быть доступны feature store и retrieval stack.

Четвертый - monitoring: offline SID metrics не всегда коррелируют с online outcomes.

Пятый - последовательности пользователей длинные, поэтому token budget становится constraint.

## Implementation notes

1. Запускать SID first как ranker feature, если full GR слишком рискован.
1. Сравнивать GraphHash/RQ/KMeans-like treatments по AUC и online metrics.
1. Считать uniqueness вместе с Recall/NDCG, а не вместо них.
1. Отдельно измерять gains для head, mid-tail и long-tail content.
1. Следить за feature freshness и backfill для новых videos.
1. Проверять, что SID feature не дублирует уже существующие category/topic features.

## Итог

Статья ценна тем, что переводит SID из "алгоритма tokenization" в компонент recommender platform. Хороший SID - это тот, который улучшает конкретный ranker/retriever при допустимой стоимости, а не тот, у которого красивая standalone uniqueness.

## Источники

- [arXiv:2604.03949](https://arxiv.org/abs/2604.03949) , PDF/source.
