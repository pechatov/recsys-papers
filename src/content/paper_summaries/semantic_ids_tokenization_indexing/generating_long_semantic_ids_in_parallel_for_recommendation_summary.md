---
title: "Generating Long Semantic IDs in Parallel for Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "generating_long_semantic_ids_in_parallel_for_recommendation_summary"
catalogId: "paper-generating_long_semantic_ids_in_parallel_for_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/generating_long_semantic_ids_in_parallel_for_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1145/3711896.3736979"
---
> **Авторы:** Yupeng Hou, Jiacheng Li, Ashley Shin, Jinsung Jeon, Abhishek Santhanam, Wei Shao, Kaveh Hassani, Ning Yao, Julian McAuley.
>
> **Аффилиации:** University of California, San Diego; Meta AI.
>
> **Индустрия:** Meta AI / large-scale generative recommendation
>
> **Первичные источники:** ACM DOI landing возвращал HTTP 403; использована открытая первичная версия arXiv:2506.05781 и arXiv HTML/PDF.

## Коротко

RPG предлагает длинные unordered Semantic IDs и параллельную генерацию всех токенов ID.

Это снимает главный bottleneck autoregressive SID models: длина ID больше не означает много forward passes.

Paper показывает, что SID длиной 64 может быть выразительнее короткого 4-token SID и при этом быстрее в inference.

## Контекст

TIGER-like системы обычно генерируют SID слева направо и используют beam search.

Длинный SID увеличивает expressiveness, но autoregressive decoding становится дорогим.

RPG меняет постановку: токены SID рассматриваются как unordered product-quantization code, а score item считается суммой token logits.

## Проблема

Ниже - конкретные failure modes, которые paper пытается закрыть.

- Короткий SID дает collisions и ограниченную емкость каталога.
- Autoregressive generation делает latency пропорциональной длине SID и beam size.
- Независимая генерация токенов может создавать invalid combinations, которых нет в catalog.
- Нужен decoding, независимый от полного item pool size.
- Нужно доказать, что unordered long SID не хуже retrieval-based alternatives.

## Метод/архитектура

Архитектура важна тем, как она меняет форму item identifier, а не только backbone.

<figure class="paper-figure">
  <img src="../../assets/rpg/framework.png" alt="RPG framework for generating long semantic IDs in parallel">
  <figcaption>Рисунок 1. RPG заменяет autoregressive SID decoding на parallel multi-token prediction и graph-constrained decoding. Поэтому длинный SID увеличивает capacity без линейного роста числа forward passes.</figcaption>
</figure>

- Long SID construction использует OPQ/PQ-style токены; paper также сравнивает random и RQ variants.
- Semantic ID embedding aggregation превращает несколько токенов item в representation.
- Learning block использует multi-token prediction objective: модель предсказывает токены next SID параллельно.
- Efficient logit calculation считает item score как сумму logits токенов соответствующего SID.
- Graph-constrained decoding строит граф похожих Semantic IDs и запускает iterative graph propagation.
- Граф нужен, чтобы избегать invalid IDs и быстро находить top candidates без полного перебора items.

## Objective/алгоритм

Objective/decoding design определяет, какие ошибки модель может делать на inference.

- Multi-token prediction decomposes next-item ID into independent token classification tasks.
- Candidate score получается агрегацией токеновых logits; благодаря unordered code нет необходимости генерировать prefix.
- Graph construction соединяет SID с малым Hamming/semantic difference.
- Iterative propagation стартует с sampled/random candidate set и распространяет scores к соседям.
- Complexity analysis подчеркивает независимость runtime memory/time от размера item pool.
- В ablation beam search для unordered long SID дает нули, потому что генерирует невалидные token sequences.

### Пошаговый алгоритм RPG

1. **Построить длинный unordered SID.** Item embedding разбивается OPQ/PQ-style на несколько subspaces; каждый subspace получает token. SID длиной 16/32/64 трактуется как set-like code, а не как ordered path.
1. **Собрать token embedding representation.** Item representation для обучения получается агрегированием embeddings всех его semantic tokens, плюс projection heads, которые не должны быть shared слишком грубо.
1. **Обучить multi-token prediction.** По user history модель одним forward pass предсказывает logits для всех token positions target SID. Нет autoregressive dependency между target tokens.
1. **Посчитать item score из token logits.** Для каждого item score равен сумме или агрегации logits тех tokens, которые входят в его SID. Это превращает independent token classification в ranking over catalog items.
1. **Построить graph of valid SIDs.** Узлы - реальные catalog items/SIDs; ребра соединяют близкие коды по Hamming/semantic distance. Граф нужен, потому что независимые token logits сами по себе могут составить несуществующий SID.
1. **Запустить graph-constrained decoding.** Стартовый candidate set расширяется итеративной propagation по соседям; score пересчитывается только для reachable valid items, а не для всех возможных token combinations.
1. **Вернуть top-K valid items.** Финальная выдача сортируется по aggregated score, а invalid sequences никогда не попадают в результат, в отличие от naive beam search по unordered tokens.

## Эксперименты

Экспериментальный блок полезен тем, что проверяет не только общий Recall/NDCG, но и efficiency/cold-start/ablation slices.

- Датасеты: Amazon Reviews Sports and Outdoors, Beauty, Toys and Games, CDs and Vinyl.
- Статистика: Sports 18,357 users / 35,598 items / 260,739 interactions; Beauty 22,363 / 12,101 / 176,139.
- Toys: 19,412 users / 11,924 items / 148,185 interactions; CDs: 75,258 users / 64,443 items / 1,022,334 interactions.
- Metrics: Recall@5/10 и NDCG@5/10, leave-last-out protocol.
- Baselines включают Caser, GRU4Rec, HGN, BERT4Rec, SASRec и semantic-ID generative baselines из TIGER lineage.
- Paper reports average 12.6% NDCG@10 gain over generative baselines when scaling SID length to 64.
- Ablation Table 3: OPQ Random и OPQ RQ хуже OPQ; no projection head/shared projection head хуже full RPG.
- Without graph constraints NDCG@10 резко падает; beam search variant генерирует invalid IDs и дает 0.0000 в ablation.
- Implementation mentions HuggingFace Transformers and FAISS.

<figure class="paper-figure">
  <img src="../../assets/rpg/sid_length_scaling.png" alt="RPG NDCG@10 scaling by semantic ID length">
  <figcaption>Рисунок 2. Scaling analysis показывает, зачем RPG нужен long SID: при увеличении длины до 64 NDCG@10 растет, а parallel decoding удерживает inference cost.</figcaption>
</figure>

## Сильные стороны

Сильные стороны ниже выделены с точки зрения внедрения и воспроизводимости.

- Сильная декомпозиция latency bottleneck.
- Длинные IDs повышают capacity без beam-search penalty.
- Graph constraints решают invalid-ID failure mode, который неизбежен при независимых токенах.
- Есть public code facebookresearch/RPG_KDD2025.
- Ablations показывают, что выигрыш не только от backbone, но и от graph decoding/tokenizer design.

## Слабые стороны и ограничения

Ограничения важны для production transfer.

- Independence assumption между токенами может терять полезную структуру.
- Граф валидных SID становится отдельным индексом, который надо строить и обновлять.
- Для динамичного каталога graph refresh latency может стать production issue.
- OPQ/PQ unordered codes хуже подходят, если позиции SID несут явную hierarchy.
- Генерация token logits не заменяет бизнес-фильтры, inventory constraints и policy constraints.
- Главный общий failure mode для SID-подходов: semantic collision выглядит осмысленно offline, но смешивает конкурирующие items в конкретном business objective.
- Второй общий failure mode: tokenizer обучен на старом каталоге и перестает отражать новые категории, бренды, форматы или geography.
- Третий общий failure mode: offline Recall/NDCG улучшается, но serving latency, graph refresh, feature joins или beam constraints съедают online gain.

## Как реализовать/проверять

Практический план проверки.

- Начать с Amazon-like offline reproduction и проверить invalid-ID rate.
- Сравнить SID length 4/8/16/32/64 при фиксированном backbone.
- Построить graph index через FAISS/nearest neighbors и отдельно измерять build time.
- Проверять memory/inference time против item pool size.
- Считать cold-start bucket отдельно: длинный SID должен помогать там сильнее.
- В production хранить graph snapshot и версию tokenizer-а как атомарный artifact.
- Зафиксировать версию tokenizer-а, vocabulary/codebook sizes, seed, дату обучения и покрытие каталога.
- Логировать invalid/generated-not-in-catalog rate отдельно от Recall/NDCG, потому что генеративная модель может улучшать ranking среди валидных, но терять кандидатов на этапе декодирования.
- Делать slice-анализ по popularity bucket, item age, cold-start cohort и длине пользовательской истории.
- Сравнивать не только с SID-only baseline, но и с ID-only/hybrid baseline: в production ID memorization часто остается сильным сигналом.
- Проверять распределение кодов: entropy/perplexity, долю неиспользуемых кодов, top-code concentration и collision examples.
- В online rollout начинать с shadow features или candidate sidecar, чтобы отделить эффект tokenizer-а от эффекта retriever/ranker.

## Связь

Связь с остальной подборкой Semantic ID/tokenization papers.

- RPG близок SETRec по отказу от token order, но SETRec делает это внутри LLM set identifier.
- RPG дополняет variable-length/CoFiRec: вместо осмысленного порядка он выбирает unordered емкость.
- Для больших каталогов это один из самых практичных ответов на вопрос, как увеличить SID length.

## Итог

Короткий вывод.

- RPG показывает, что длинный Semantic ID не обязан быть медленным.
- Ключевой компромисс: отказаться от autoregressive order и компенсировать invalid combinations graph constraints.
- Если catalog index можно обновлять надежно, подход выглядит production-friendly для large-scale GenRec.
