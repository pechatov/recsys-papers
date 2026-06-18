---
title: "SID-Coord: Coordinating Semantic IDs for ID-based Ranking in Short-Video Search"
category: "semantic_ids_tokenization_indexing"
slug: "sid_coord_coordinating_semantic_ids_for_id_based_summary"
catalogId: "paper-sid_coord_coordinating_semantic_ids_for_id_based_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/sid_coord_coordinating_semantic_ids_for_id_based_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.10471"
---
> **Авторы:** Guowen Li, Yuepeng Zhang, Shunyu Zhang, Yi Zhang, Xiaoze Jiang, Yi Wang, Jingwei Zhuo.
>
> **Аффилиации:** Kuaishou Technology.

## Коротко

SID-Coord добавляет semantic IDs в ID-based ranking для short-video search. Вместо простой конкатенации HID и SID сигналов он координирует их через multi-resolution SID modeling, target-aware HID-SID gating и user-item semantic alignment.

## Контекст

ID-based rankers хорошо запоминают head items, но плохо обобщают на long-tail. SIDs дают semantic sharing, однако могут размывать точность head. Нужен механизм, который понимает, когда доверять hard ID memorization, а когда SID generalization.

## Проблема

Простое добавление SID features не решает heterogeneous interaction density. Head video с большим количеством экспозиций лучше описывается HID, tail video выигрывает от semantic neighborhood. Если смешивать их статически, либо теряется head accuracy, либо long-tail остается слабым.

## Метод / архитектура

Multi-resolution SID Modeling использует несколько уровней SID granularity и attention fusion. Target-aware HID-SID Gating вычисляет gate `g`, зависящий от target item и popularity bucket, чтобы балансировать HID и SID. User-Item Semantic Alignment добавляет distribution-level matching между target semantics и историей пользователя.

## Objective / алгоритм

Ranking objective остается click/search relevance loss, но representation для item строится как координированная смесь HID и SID. Gate фактически выполняет popularity-aware routing: для sparse long-tail повышается роль SID, для head items сохраняется HID memorization. Alignment loss/feature усиливает соответствие между user semantic interests и target SID distribution.

## Детальный алгоритм SID-Coord

SID-Coord не строит новый tokenizer; он учит ranker правильно использовать уже имеющиеся semantic IDs вместе с hard IDs. Алгоритм можно понимать как target-dependent routing между memorization channel и semantic generalization channel.

1. **Подготовить HID и SID признаки item-а.** HID - обычный hard item ID embedding; SID - multi-level semantic identifier, где уровни соответствуют разной granularity.
1. **Смоделировать SID на нескольких разрешениях.** Для каждого SID level берется embedding; multi-level attention fusion собирает semantic representation target video.
1. **Построить HID representation.** Hard ID branch сохраняет точное запоминание head videos и historical interactions.
1. **Вычислить target-aware gate.** Gate `g` зависит от target item и popularity bucket, заданного 7-day exposure deciles. Для tail ожидается больший вес SID, для head - больший вес HID.
1. **Смешать HID/SID representations.** Итоговый item vector подается в ID-based ranker вместе с остальными search/query/user features.
1. **Добавить user-item semantic alignment.** User history агрегируется в semantic distribution; target SID distribution сравнивается с ней, чтобы SID работал как язык user interests.
1. **Обучить ranking objective.** Основной loss - click/search relevance, а не generative likelihood. SID-Coord оптимизирует AUC/UAUC ranker-а.
1. **Проверить gate sanity.** Средний gate по exposure deciles должен меняться осмысленно; иначе компонент не решает head-tail conflict.

```
for impression in training_logs:
    hid_vec = HID_embedding[target_item]
    sid_level_vecs = [SID_embedding[level][code] for level, code in SID[target_item]]
    sid_vec = multi_level_attention(sid_level_vecs)

    popularity_bucket = exposure_decile(target_item, window="7d")
    gate = target_aware_gate(target_item_features, popularity_bucket)
    item_vec = gate * hid_vec + (1 - gate) * sid_vec

    user_semantic_dist = aggregate_history_SID_distribution(user_history)
    align_feature = semantic_alignment(user_semantic_dist, SID[target_item])
    score = ranker(query_user_features, item_vec, align_feature)
    update(click_or_relevance_loss(score, label))

monitor AUC/UAUC overall, long-tail, and gate by exposure decile
```

## Эксперименты и метрики

Метрики: AUC и UAUC overall и для long-tail subset. Все SID-based методы превосходят production baseline, а SID-Coord дает лучший результат: +0.33% AUC overall, +0.42% AUC на long-tail и +0.93% long-tail UAUC relative to baseline. Ablation показывает вклад компонентов: без multi-resolution SID AUC падает на -0.22% overall и -0.26% long-tail; без gating -0.13% overall; без semantic alignment -0.16% overall.

## Рисунки / таблицы

Overview figure показывает три компонента SID-Coord. Multi-level attention figure раскрывает fusion уровней SID. Popularity-aware coordination plot показывает средний gate по exposure deciles: это ключевой sanity check, что модель действительно меняет HID/SID balance по popularity. Таблицы overall, ablation и online A/B связывают offline AUC gains с production эффектом.

## Сильные стороны

- **Фокус на ranking-side использовании SID.** Работа показывает, что semantic IDs полезны не только generative retrieval, но и ID-based short-video search ranker-у.
- **Явная head-tail логика.** Target-aware gate учитывает exposure deciles и решает, где нужен HID memorization, а где SID sharing.
- **Multi-resolution SID не выбрасывает уровни.** Attention fusion позволяет ranker-у выбирать coarse или fine semantic signal по ситуации.
- **Alignment использует SID как язык user interests.** Семантика применяется не только к target item, но и к истории пользователя.
- **Ablation численно подтверждает компоненты.** Удаление multi-resolution, gating и alignment дает отдельные падения AUC/UAUC.

## Ограничения

- **Зависимость от Kuaishou short-video search.** Query mix, video freshness и internal SID construction могут не переноситься в e-commerce или long-form content.
- **SID tokenizer остается внешним предположением.** Если SID плохо отражает search intent, coordination только аккуратно смешает плохой semantic signal.
- **Popularity buckets устаревают.** Exposure deciles надо обновлять часто; иначе gate будет маршрутизировать head/tail неверно.
- **AUC gains дробные.** +0.33% overall AUC требует строгой статистики, guardrail metrics и online confirmation.
- **Возможен gate collapse.** Без мониторинга gate может уйти почти всегда в HID или SID, скрывая проблему за средним AUC.

## Как реализовать / проверять

Добавить SID embeddings на нескольких уровнях, attention fusion и gate между HID/SID ветками. Проверять AUC/UAUC overall/head/tail, распределение gate по popularity buckets, feature drift, latency ranker'а и online search satisfaction. Обязателен ablation "SID concat без gate", чтобы доказать координацию, а не просто добавление признаков.

## Связь с соседними работами

SID-Coord является ranking-side парой к Snapchat SID-as-feature use case. В отличие от AdaSID, он не учит новые IDs, а учит как использовать их совместно с HID. С RecoChain пересекается в цели harmonize retrieval/ranking signals.

## Multi-resolution SID modeling

Semantic ID содержит несколько уровней granularity.

Coarse levels дают статистическое sharing.

Fine levels дают различение похожих videos.

SID-Coord не выбирает один уровень вручную.

Он применяет multi-level attention fusion, чтобы ranker сам взвешивал уровни.

Это особенно полезно для tail videos, где fine ID sparse, а coarse semantic group содержит полезный signal.

## Target-aware HID-SID gating

Hard ID хорошо работает для head items.

Semantic ID лучше обобщает для sparse long-tail.

Gate `g` смешивает HID и SID representations в зависимости от target.

Popularity buckets задаются по 7-day exposure deciles.

Figure с confidence intervals показывает, как средний gate меняется по popularity.

Это важный sanity check: если gate не зависит от popularity, компонент не выполняет свою функцию.

## User-item semantic alignment

Alignment component сравнивает target item semantics с semantic distribution user history.

Это не просто item-side feature.

Он добавляет matching между тем, что пользователь смотрел/искал, и semantic region target video.

В ablation removal alignment дает -0.16% AUC overall и -0.19% на long-tail.

Значит SID полезен не только как item descriptor, но и как язык user-interest aggregation.

## Baselines and ablation interpretation

Все SID-based методы лучше production baseline.

Это подтверждает, что semantic signal полезен для ID-based ranking.

Однако SID-Coord лучше простой интеграции SID, потому что решает conflict HID vs SID.

Самый большой ablation drop дает removal multi-resolution SID module.

Это показывает, что granularity является главным источником long-tail gain.

Gating removal показывает, что статическая смесь HID/SID недостаточна.

## Online A/B considerations

В short-video search даже малые AUC/UAUC gains могут быть product-relevant.

Но online эффект зависит от query mix, freshness и downstream re-ranking.

SID-Coord должен проверяться на head preservation: long-tail gain не должен ухудшать high-traffic videos.

Нужно также мониторить creator/content diversity, потому что semantic sharing может перераспределять exposure.

## Failure modes

Первый failure mode - gate collapse к HID или SID для всех items.

Второй - popularity buckets устаревают, если exposure distribution резко меняется.

Третий - SID tokenizer не отражает search intent, а только visual/content similarity.

Четвертый - multi-resolution attention переобучается на head categories.

Пятый - long-tail AUC растет, но online satisfaction не улучшается из-за freshness или policy filters.

## Implementation notes

1. Добавить embeddings для каждого SID уровня.
1. Реализовать attention fusion поверх уровней.
1. Добавить HID ветку и target-aware gate.
1. Обновлять popularity buckets регулярно, лучше daily.
1. Считать AUC/UAUC отдельно overall и long-tail.
1. Строить dashboard gate value по exposure deciles.

## Итог

SID-Coord показывает, что semantic IDs полезны не только генератору. В зрелом ranker'е они должны быть скоординированы с hard IDs, особенно по оси head-tail popularity.

## Источники

- [arXiv:2604.10471](https://arxiv.org/abs/2604.10471) , PDF/source.
