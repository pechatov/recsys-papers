---
title: "DOS: Dual-Flow Orthogonal Semantic IDs for Recommendation in Meituan"
category: "semantic_ids_tokenization_indexing"
slug: "dos_dual_flow_orthogonal_semantic_ids_for_recommendation_summary"
catalogId: "paper-dos_dual_flow_orthogonal_semantic_ids_for_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/dos_dual_flow_orthogonal_semantic_ids_for_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2602.04460"
---
Подробное саммари статьи:

> **Авторы:** Junwei Yin, Senjie Kou, Changhao Li, Shuli Wang, Xue Wei, Yinqiu Huang, Yinhua Zhu, Haitao Wang, Xingxing Wang.
>
> **Аффилиации:** Meituan.

## 1. Коротко

DOS - промышленная работа Meituan про построение semantic IDs, которые лучше согласованы с downstream generative recommendation. Авторы выделяют две причины потерь: кодбук semantic tokenizer'а живет в пространстве item content, а генератор работает в user-item interaction context; кроме того, обычная residual quantization может терять семантику из-за неудачной ориентации пространства. DOS объединяет Dual-Flow Integration и Orthogonal Residual Quantization.

## 2. Контекст

Semantic IDs в промышленной системе должны одновременно быть компактными, сохранять знания LLM/content encoder'а и быть удобными для генерации в конкретном продукте. В Meituan это не академическая toy-задача: статья прямо сообщает об offline экспериментах, online A/B test и deployment в мобильном приложении с сотнями миллионов пользователей.

## 3. Проблема

- Context gap: item codebook строится без достаточного учета user sequence и target-item generation context.
- Quantization loss: близость в исходном semantic space не гарантирует, что residual уровни оптимально сохраняют информацию.
- Прямое добавление SID как фичи должно улучшать live model без тяжелой переделки всей архитектуры.
- Нужно сохранить latency и совместимость с production ranking/retrieval stack.

## 4. Метод и архитектура

Dual-Flow Integration строит две взаимосвязанные ветки: user-flow обрабатывает последовательность кликов пользователя, item-flow обрабатывает target item. Collaborative signal из взаимодействий используется для выравнивания codebook space и generation space. ORQ добавляет обучаемое ортогональное преобразование: перед residual quantization embedding вращается в такую ориентацию, где главные и остаточные компоненты лучше разделяются и меньше теряют информацию при последовательном кодировании.

## 5. Objective и алгоритм

Целевая функция DOS объединяет recommendation loss и quantization/alignment terms. DFI отвечает за то, чтобы SID не были только content-derived labels, а отражали реальные user-item transitions. ORQ можно понимать как поиск базиса, в котором residual quantizer менее разрушителен: после ортогонального поворота модель квантует компоненты, возвращая компактный SID, но с меньшим semantic loss. Важное свойство ортогональности - сохранение расстояний/норм, что делает преобразование менее произвольным, чем свободный MLP.

### 5.1. Пошаговый алгоритм DOS

1. **Собрать dual-flow batch.** Для каждого train example берется user click sequence и target item; user-flow кодирует историю, item-flow кодирует target representation.
1. **Получить collaborative alignment signal.** DFI учит user-flow и item-flow жить в согласованном generation space, чтобы SID target item был предсказуем из user context.
1. **Повернуть item embedding.** Перед residual quantization применяется обучаемая ортогональная матрица $R$, так что $\tilde{\mathbf{z}} = R\mathbf{z}$. Ортогональность сохраняет нормы и расстояния, но меняет basis.
1. **Выполнить residual quantization.** Первый codebook выбирает главный code, затем следующие codebooks квантуют residual information. Цель ORQ - сделать ранние уровни SID более информативными, а поздние - не шумовыми.
1. **Оптимизировать общий loss.** Recommendation objective давит на полезность SID для генерации, quantization loss удерживает reconstructability, alignment terms уменьшают gap между codebook и user-item context.
1. **Экспортировать SID как production feature.** После обучения SID добавляется в downstream Meituan stack как incremental feature, а не требует полной замены ranker/retriever.
1. **Проверить ablation.** Отключение DFI должно показать потерю contextual alignment; отключение ORQ - потерю quantization geometry.

```
for batch of (user_sequence, target_item):
    h_user = user_flow(user_sequence)
    z_item = item_flow(target_item)

    aligned = align(h_user, z_item)
    z_rot = orthogonal_matrix @ z_item

    residual = z_rot
    sid = []
    for codebook in residual_codebooks:
        code = nearest_code(residual, codebook)
        sid.append(code.index)
        residual = residual - code.vector

    loss = recommendation_loss(h_user, sid) + quantization_loss + alignment_loss
    update user_flow, item_flow, orthogonal_matrix and codebooks
```

## 6. Эксперименты и метрики

Offline сравнение включает downstream AUC и F1. В таблице performance DOS превосходит несколько baseline-вариантов: например AUC поднимается до 0.8763 против 0.8363-0.8539 у альтернатив, F1 также улучшается. Отдельно показаны Hit@10 по бизнес-категориям и ablation: удаление DFI или ORQ снижает качество. Online A/B test проводился неделю на 30% production traffic Meituan и подтверждает инкрементальный выигрыш от SID без прочих архитектурных изменений.

## 7. Что важно в рисунках и таблицах

Figure 1 важен, потому что показывает не просто tokenizer, а dual-flow learning: user sequence и target item обрабатываются совместно, поэтому SID становится contextualized representation для генератора. Правая часть с ORQ показывает, что авторы борются не с размером codebook, а с геометрией пространства перед quantization. Таблицы AUC/F1 и Hit@10 важны как evidence, что gain переносится с общей метрики на категории бизнеса.

## 8. Сильные стороны

- Редкий пример SID-работы с production deployment и online A/B test.
- Хорошо отделены две причины качества: alignment с recommendation task и качество quantization geometry.
- ORQ аккуратно ограничивает преобразование ортогональностью, снижая риск произвольного искажения embedding space.
- Ablation показывает вклад обоих компонентов.

## 9. Ограничения

- Статья раскрывает ограниченный набор production-деталей: latency budget, catalog dynamics и refresh policy описаны кратко.
- Данные Meituan закрыты, поэтому абсолютные цифры сложно воспроизвести.
- Dual-flow обучение может быть чувствительно к distribution shift между offline logs и online traffic.
- ORQ улучшает геометрию, но не решает полностью проблему collision/long-tail coverage.

## 10. Как реализовать и проверять

- Начать с замера baseline SID как отдельной feature: AUC/F1 offline и category-level Hit@K.
- Ввести dual-flow training только если есть надежные user sequence и target item pairs; иначе DFI будет учить шум.
- Для ORQ контролировать orthogonality error и reconstruction/semantic preservation после каждого уровня quantization.
- В online rollout отделять эффект SID от остальных фичей через frozen architecture experiment.

## 11. Связь с соседними работами

DOS соседствует с ReSID и CoST как работа про better quantization objective, но сильнее ориентирована на промышленный contextual alignment. С MERGE ее роднит production focus; с PIT - желание учитывать collaborative signals при построении IDs. В отличие от DIGER/UniGRec, DOS не делает акцент на полностью end-to-end soft identifiers, а строит deployable SID feature.

## 12. Итог

Главный урок DOS: semantic ID для рекомендаций должен быть не просто сжатым LLM/content embedding'ом. Он должен жить в координатах задачи генерации и взаимодействий. Dual-flow alignment и ортогональное residual quantization дают прагматичный путь к такому SID без радикальной смены production stack.

## 13. Детальный разбор механизмов статьи

### 13.1. Dual-Flow Integration

DFI в DOS нужен для того, чтобы semantic ID перестал быть только функцией target item. В генеративной рекомендации item code должен быть удобен для предсказания из user history, поэтому user-flow и item-flow обучаются совместно. Это особенно важно в Meituan, где одна и та же семантика товара или услуги может иметь разный recommendation meaning в разных пользовательских контекстах.

- User-flow кодирует последовательность кликов и формирует contextual collaborative signal.
- Item-flow кодирует target item и его semantic representation.
- Alignment между потоками уменьшает gap между codebook space и generation space.
- SID после DFI можно использовать как incremental feature в downstream ranker/retriever.
- DFI ablation показывает, что без контекстного alignment ORQ alone недостаточен.

### 13.2. Orthogonal Residual Quantization

ORQ исходит из того, что residual quantization чувствителен к ориентации embedding space. Если главные информационные направления плохо согласованы с residual decomposition, последовательная квантизация теряет смысл уже на ранних уровнях. Ортогональный поворот сохраняет геометрию, но меняет basis так, чтобы residual levels лучше разделяли primary и residual semantics.

- Ортогональность ограничивает transformation и снижает риск произвольного distortion.
- Первый уровень SID должен нести наиболее важную семантическую массу.
- Последующие уровни должны кодировать residual information, а не шум после плохого первого split.
- В таблицах ORQ дает вклад как в AUC, так и в F1.
- Метод применим к LLM-derived embeddings, где исходное пространство не оптимизировано под quantization.

### 13.3. Offline и online evidence

- Основная offline таблица показывает AUC 0.8763 у DOS против 0.8363-0.8539 у сравниваемых вариантов.
- Вторая таблица подтверждает gain при другом наборе baseline settings: DOS снова достигает AUC 0.8763.
- Hit@10 по business category нужен, чтобы исключить ситуацию, когда средний gain приходит из одной крупной категории.
- Online A/B test длился одну неделю и использовал 30% production traffic Meituan.
- Авторы подчеркивают deployment в Meituan mobile application, то есть SID прошел не только offline benchmark.

### 13.4. Production notes

- DOS полезен как additive SID feature: его можно внедрять без полной замены модели.
- Нужно мониторить drift между LLM semantic space и user interaction space после обновления каталога.
- ORQ matrix должна версионироваться вместе с codebook; иначе старые SIDs станут несовместимыми.
- Category-level метрики должны входить в rollout gate, потому что food/local services имеют сильную heterogeneity.
- Если online traffic меняется, DFI может начать переобучаться на recent popularity, поэтому нужен temporal validation.

## 14. Первичные источники

- arXiv abstract/source/PDF: [2602.04460](https://arxiv.org/abs/2602.04460) .
