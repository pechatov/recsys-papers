---
title: "Harmonizing Generative Retrieval and Ranking in Chain-of-Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "harmonizing_generative_retrieval_and_ranking_in_chain_of_summary"
catalogId: "paper-harmonizing_generative_retrieval_and_ranking_in_chain_of_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/harmonizing_generative_retrieval_and_ranking_in_chain_of_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.25787"
---
> **Авторы:** Yu Liu, Jiangxia Cao.
>
> **Аффилиации:** Kuaishou Technology; NJUST.

## Коротко

RecoChain объединяет generative retrieval и ranking в одной chain-of-recommendation последовательности. Модель сначала генерирует next-item semantic ID, затем получает candidate-aware subsequence и оценивает ranking score, вместо того чтобы держать retriever и ranker полностью раздельно.

## Контекст

Классический промышленный stack retrieve-then-rank эффективен, но обучается фрагментированно: retrieval оптимизирует candidate recall, ranking - click/utility. Generative retrieval обещает end-to-end item generation, но часто не имеет сильного ranking signal для кандидатов внутри beam.

## Проблема

Если генерировать только SIDs, модель хорошо достает кандидатов, но порядок внутри top-k может быть слабым. Если отдельно ранжировать, возникает mismatch между representations и training objectives. RecoChain хочет, чтобы один sequence model видел и генерацию, и candidate-aware scoring.

## Метод / архитектура

Item tokenizer строит semantic IDs. Sequence model получает history, генерирует SID кандидата, а затем формирует candidate-aware subsequence, где candidate tokens становятся входом для reranking head. Inference: beam search достает candidates, затем тот же model оценивает click probability/ranking score и переупорядочивает выдачу.

## Objective / алгоритм

Training token organization состоит из двух шагов: next-item SID generation и последующий ranking-score prediction на candidate-aware tokens. Это chain: output retrieval stage становится условием ranking stage. Objective сочетает generative likelihood и ranking loss, что должно уменьшить разрыв между recall и order quality.

## Детальный алгоритм RecoChain

RecoChain делает retrieve-then-rank не двумя независимыми моделями, а одной цепочкой токенов. Сначала decoder генерирует SID кандидата, затем этот кандидат добавляется в candidate-aware segment, на котором та же модель оценивает ranking score.

1. **Построить SID tokenizer и trie.** Каждый item имеет semantic ID sequence; trie нужен, чтобы beam search не уходил в invalid paths.
1. **Закодировать user behavior sequence.** История пользователя подается как SID tokens с нужными разделителями и positional structure.
1. **Обучить retrieval segment.** Первая часть objective - next-item SID generation: модель должна предсказать token sequence target item.
1. **Сформировать candidate-aware continuation.** После generated/target SID добавляется специальный subsequence, в котором candidate tokens становятся условием ranking head.
1. **Обучить ranking segment.** Hidden states candidate-aware continuation передаются в head, который предсказывает click/relevance score.
1. **На inference сначала сгенерировать candidates.** Beam search возвращает top-B valid SIDs; каждый мапится в item или candidate group.
1. **Затем rerank тем же model checkpoint.** Для каждого candidate строится candidate-aware input, считается ranking score, candidates сортируются заново.
1. **Измерять paired metrics.** Нужно сравнивать Recall/NDCG до и после reranking при разных beam sizes, потому что reranker не может восстановить item, которого нет в candidate pool.

```
for training_example in logs:
    history_tokens = encode_history_as_SIDs(user_history)
    target_sid = SID[target_item]

    loss_gen = autoregressive_loss(model, history_tokens, target_sid)
    candidate_segment = build_candidate_aware_segment(history_tokens, target_sid)
    rank_score = ranking_head(model(candidate_segment))
    loss_rank = ranking_loss(rank_score, click_or_relevance_label)
    update(loss_gen + lambda_rank * loss_rank)

at serving:
    candidates = constrained_beam_search(model, history_tokens, beam_size=B)
    scores = [ranking_head(model(build_candidate_aware_segment(history, c))) for c in candidates]
    return sort_by_score(candidates, scores)
```

## Эксперименты и метрики

Авторы варьируют beam size, input sequence length и retrieval length. Метрики: Recall@5/10 и NDCG@5/10 до и после reranking. При beam size 10 reranking дает небольшой Recall@5 gain +0.27%, при beam 40 gain растет до +1.08%, а NDCG@10 gain до +0.65%. Длинная входная история уменьшает добавочную пользу reranking: Recall@5 gain падает с +3.14% при sequence length 32 до +0.92% при 128.

## Рисунки / таблицы

Figure 1 показывает классический two-stage retrieval-then-ranking, Figure 2 - организацию RecoChain tokens. Таблицы beam size, sequence length и retrieval length важны как sensitivity analysis: они показывают, что reranking полезнее, когда candidate pool шире и retrieval stage оставляет пространство для переупорядочивания.

## Сильные стороны

- **Связывает retrieval и ranking в одном sequence pipeline.** Generated candidate становится условием ranking stage, а не передается в полностью отдельную модель.
- **Хорошая sensitivity diagnostics.** Beam size, input sequence length и retrieval length показывают, когда reranking действительно добавляет качество.
- **Естественно встраивается поверх SID generation.** RecoChain не требует отказаться от trie/beam semantic retrieval.
- **Уменьшает objective mismatch.** Ranking loss видит representations того же decoder-а, который породил candidates.
- **Показывает ceiling candidate pool.** Результаты при разных beams помогают отделить проблемы retrieval recall от ordering.

## Ограничения

- **Reranker не лечит candidate starvation.** Если beam не достал релевантный item, ranking head не сможет его добавить.
- **Gain зависит от ширины beam.** При beam 10 эффект небольшой, при beam 40 больше; значит latency-quality trade-off остается.
- **Стоимость inference растет.** Нужно не только сгенерировать candidates, но и прогнать candidate-aware scoring для каждого.
- **Есть риск attention leakage.** Training sequence должен быть организован так, чтобы ranking segment не видел target information некорректно.
- **Нет полной online validation.** Offline Recall/NDCG gains не доказывают, что chain выдержит production latency и traffic shift.

## Как реализовать / проверять

Нужен валидный SID tokenizer, beam generator и reranking head над candidate-aware tokens. Проверять следует paired metrics до/после rerank, invalid SID rate, latency увеличения chain, calibration ranking score и зависимость gains от beam size. В production важно сохранить fallback на обычный ranker.

## Связь с соседними работами

RecoChain близок к GenRec по стремлению выйти за пределы next-token SFT. В отличие от GLASS, где semantic search помогает retrieval, RecoChain добавляет ranking stage внутри той же модели. С SID-Coord пересекается по идее согласовать semantic signal с ranking quality.

## Training token organization

Главная техническая идея статьи - организация sequence так, чтобы retrieval и ranking стали соседними задачами.

Сначала модель получает user behavior sequence и генерирует semantic ID следующего item.

Затем candidate-aware subsequence добавляется как продолжение input.

На этом continuation модель оценивает ranking score или click probability.

Так generated candidate становится условием для reranking head.

Это отличается от схемы, где ranker является отдельной моделью с отдельными features.

## Item tokenizer и architecture

Tokenizer должен обеспечить валидный mapping SID→item.

Если несколько items делят SID, reranking stage может частично разрешить ambiguity.

Но слишком высокая collision rate ухудшит candidate generation еще до reranking.

Ranking head использует hidden states после candidate segment.

Это позволяет передать retrieval uncertainty в ranking representation.

Нужно аккуратно маскировать attention, чтобы training не дал leakage target information.

## Inference procedure

Inference состоит из двух шагов.

Первый: beam search генерирует candidate SIDs.

Второй: для каждого candidate строится candidate-aware subsequence, и reranking head оценивает click probability.

Затем candidates сортируются по ranking score.

Это повышает latency относительно чистого generation, но может улучшить ordering при широком beam.

## Sensitivity conclusions

Beam size управляет шириной candidate pool.

При beam 10 reranking почти нечего исправлять, поэтому Recall@5 gain всего +0.27%.

При beam 40 pool шире, и Recall@5 gain растет до +1.08%.

NDCG@10 gain также растет до +0.65%.

При sequence length 32 reranking gain по Recall@5 составляет +3.14%.

При sequence length 128 gain падает до +0.92%.

Интерпретация: длинная история уже дает retriever'у больше контекста, поэтому ranking head добавляет меньше.

## Failure modes и implementation notes

Первый failure mode - candidate starvation: если beam не достал релевантный item, reranker не поможет.

Второй - reranker overfits на generated candidates и плохо калибруется на distribution shift.

Третий - latency multiplication: reranking каждого candidate дорог.

Четвертый - exposure bias от старой системы в ranking labels.

Пятый - attention leakage при неправильной организации training sequence.

1. Сначала валидировать standalone generative retrieval.
1. Добавить candidate-aware segment и ranking head.
1. Сравнить metrics до и после reranking при разных beam sizes.
1. Измерять calibration click probability.
1. Отдельно логировать Recall ceiling candidate pool.
1. Проверять latency на p50/p95 и возможность batch scoring candidates.

## Итог

RecoChain - прагматичный мост между generative retrieval и ranking. Он не отменяет two-stage логику, а делает ее последовательной и обучаемой внутри SID-aware model.

## Источники

- [arXiv:2604.25787](https://arxiv.org/abs/2604.25787) , PDF/source.
