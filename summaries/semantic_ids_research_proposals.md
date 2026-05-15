# Research proposals: Semantic IDs and item tokenization for recommender systems

Этот файл собирает направления, которые выглядят реалистичными для проекта на 3-6 месяцев и могут вырасти в short paper или full paper для конференции по recommender systems / IR, например ECIR, KDD workshop, RecSys workshop, SIGIR short, CIKM short. Идеи специально сформулированы так, чтобы их можно было проверить на публичных датасетах и не требовать закрытого production-трафика.

## 1. Stability-aware Semantic ID: tokenizer, который оптимизирует не только качество, но и drift

**Вопрос.** Можно ли строить semantic IDs, которые дают хорошее downstream recommendation quality и при этом меньше меняются при периодическом переобучении tokenizer'а?

**Мотивация.** Большинство работ оптимизирует offline Recall/NDCG после tokenization. Но в реальной системе tokenizer переобучается, каталог обновляется, и item-to-code mapping drift может ломать совместимость с downstream model, cache, serving и historical features.

**Идея метода.** Добавить в tokenizer objective stability regularization: новые коды должны сохранять prefix или edit-distance proximity к предыдущей версии, если item embedding / content / behavior изменились незначительно. Для новых items можно разрешать свободное назначение, а для старых вводить мягкий penalty на code movement.

**Эксперимент.** Взять Amazon Reviews / Yelp / MIND, разбить историю по времени, обучать tokenizer на rolling windows. Измерять Recall/NDCG, code drift rate, prefix preservation, collision churn и degradation при использовании recommender, обученного на предыдущей версии кодов.

**Потенциальный вклад.** Новый evaluation protocol для lifecycle semantic IDs плюс простой stability-aware tokenizer, который лучше подходит для production retraining.

## 2. Collision taxonomy: не все коллизии semantic IDs одинаково вредны

**Вопрос.** Какие коллизии в semantic ID действительно вредят recommendation, а какие работают как полезное sharing между похожими items?

**Мотивация.** В literature коллизии часто считаются generic problem. Но production papers показывают, что семантически осмысленные collisions могут помогать tail items.

**Идея метода.** Построить taxonomy collisions: substitute collision, complement collision, popularity-dominated collision, category-only collision, false semantic collision. Затем ввести collision-aware loss, который штрафует только вредные пары и допускает полезные shared prefixes.

**Эксперимент.** Использовать item co-occurrence graph, category metadata и embedding similarity. Сравнить random hash, RQ-VAE SID, balanced SID и collision-aware SID. Метрики: Recall/NDCG, tail Recall, collision purity, intra-code behavioral consistency.

**Потенциальный вклад.** Более тонкая постановка collision handling и practically useful diagnostics для tokenizer evaluation.

## 3. Variable-length IDs under fixed inference budget

**Вопрос.** Как выбирать длину semantic ID, если у системы есть жесткий token budget на decoding и latency?

**Мотивация.** Variable-length IDs экономят токены, но нужно понять, когда короткий код ухудшает disambiguation и как распределить budget между head/tail items.

**Идея метода.** Обучать policy длины кода с constraint на average generated tokens. Length decision зависит от popularity, embedding density, collision risk и uncertainty. Можно сравнить heuristic length assignment против learned policy.

**Эксперимент.** Amazon/MIND/Yelp с fixed average token budget. Сравнить fixed-length, popularity-based variable length, uncertainty-based variable length. Метрики: Recall/NDCG, average decoding steps, item ambiguity, tail performance.

**Потенциальный вклад.** Практический protocol для semantic ID design под latency constraints.

## 4. Semantic ID staleness in collaborative spaces

**Вопрос.** Как быстро устаревают semantic IDs, если они построены по collaborative signals, и можно ли обновлять их локально без полной переиндексации?

**Мотивация.** Content-based SIDs стабильнее, но хуже отражают behavior; collaborative SIDs релевантнее, но стареют при изменении user-item interactions.

**Идея метода.** Разделить code на stable content prefix и adaptive collaborative suffix. Prefix обновляется редко, suffix - часто и локально. Добавить staleness detector, который переassign'ит только items с большим behavioral shift.

**Эксперимент.** Rolling temporal splits. Метрики: downstream quality, number of reassigned items, prefix/suffix drift, retraining cost. Baselines: static SID, full retrain SID, content-only SID.

**Потенциальный вклад.** Hybrid lifecycle design для semantic IDs, который балансирует freshness и serving stability.

## 5. Behavior-aware contrastive tokenization beyond instance discrimination

**Вопрос.** Можно ли улучшить CoST-like contrastive tokenization, если positives/negatives выбирать по user behavior, а не только по instance identity?

**Мотивация.** Instance discrimination заставляет каждый item отличаться от других batch items, но в recommendation похожие substitutes или complements не всегда должны разъезжаться.

**Идея метода.** Построить contrastive objective с несколькими типами positives: same category, co-click substitutes, co-purchase complements, same session intent. Negatives выбирать hard, но избегать false negatives.

**Эксперимент.** Сравнить MSE RQ-VAE, CoST, behavior-aware CoST. Метрики: Recall/NDCG, neighborhood preservation, category/substitute/complement retrieval, false-negative sensitivity.

**Потенциальный вклад.** Retrieval-aware tokenizer objective, который лучше соответствует recommendation semantics.

## 6. Multi-identifier training without multi-identifier serving

**Вопрос.** Насколько можно улучшить generative recommender, используя несколько identifiers на item только во время pretraining, но оставляя один canonical ID на inference?

**Мотивация.** MTGRec показывает ценность multi-identifier augmentation. Остается пространство для более простых и более контролируемых variants.

**Идея метода.** Сравнить разные источники identifier variants: adjacent tokenizer checkpoints, stochastic quantization, dropout over code levels, semantic paraphrases, content-vs-collaborative tokenizers. На inference выбрать canonical tokenizer.

**Эксперимент.** Pretraining/fine-tuning setup на Amazon datasets. Измерять scale behavior при росте transformer layers, long-tail gains, robustness к tokenizer noise.

**Потенциальный вклад.** Четкий ablation study multi-identifier augmentation с practical recipe.

## 7. Order-robust semantic IDs for autoregressive decoding

**Вопрос.** Должен ли порядок токенов semantic ID всегда идти coarse-to-fine, или можно сделать generation order adaptive/order-agnostic?

**Мотивация.** Coarse-to-fine естественен для hierarchy, но autoregressive model может ошибиться на раннем coarse token и заблокировать correct item. Другой порядок может быть устойчивее.

**Идея метода.** Обучить несколько generation orders: coarse-to-fine, fine-to-coarse, popularity-first, uncertainty-first, random-order with permutation objective. Добавить constrained decoding, который поддерживает множество валидных permutations.

**Эксперимент.** Сравнить error propagation, prefix accuracy, final item Recall, beam efficiency. Отдельно оценить head/tail и dense/sparse catalog regions.

**Потенциальный вклад.** Новое понимание autoregressive factorization для semantic IDs.

## 8. Semantic ID diagnostics benchmark

**Вопрос.** Какие tokenizer diagnostics предсказывают downstream recommendation quality до обучения дорогого recommender?

**Мотивация.** Сейчас tokenizer часто оценивают reconstruction loss и downstream NDCG. Между ними нужен дешевый набор proxy metrics.

**Идея метода.** Собрать benchmark метрик: code utilization, entropy, collision purity, prefix mutual information with categories, behavioral neighborhood preservation, tail sharing, temporal drift. Проверить correlation с downstream Recall/NDCG на разных tokenizer'ах.

**Эксперимент.** Обучить family tokenizer variants на нескольких datasets. Для каждого посчитать diagnostics и downstream quality. Построить predictive model или ranking of diagnostics.

**Потенциальный вклад.** Практический evaluation suite для semantic ID research.

## 9. Hybrid semantic-hash identifiers for cold-start and tail items

**Вопрос.** Можно ли объединить semantic IDs и hash IDs так, чтобы semantic sharing помогал tail/cold-start, а hash component сохранял individuality popular items?

**Мотивация.** Чистые semantic IDs могут коллидировать там, где нужна точная item identity. Чистый hashing не переносит knowledge между похожими items.

**Идея метода.** Identifier состоит из semantic prefix и learned/hash suffix. Для head items suffix длиннее или точнее, для tail items больше reliance на semantic prefix. Length/suffix allocation зависит от popularity и ambiguity.

**Эксперимент.** Cold-start split и long-tail split. Baselines: random hash, semantic-only, ID-only. Метрики: tail Recall, head precision, collision harmfulness, embedding stability.

**Потенциальный вклад.** Простая production-friendly схема с понятным trade-off между sharing и identity.

## 10. Tokenization for multi-objective recommendation: relevance, diversity, novelty

**Вопрос.** Можно ли сделать semantic ID tokenizer, который помогает не только accuracy, но и diversity/novelty/calibration?

**Мотивация.** Generative recommenders часто оптимизируют next-item accuracy. Но структура token space может усиливать или ослаблять diversity выдачи.

**Идея метода.** Добавить tokenizer regularization, которая контролирует distribution generated prefixes: не схлопывать все в популярные clusters, сохранять category coverage, novelty buckets или creator diversity.

**Эксперимент.** Sequence recommendation с reranking-free generation. Метрики: Recall/NDCG плюс coverage, intra-list diversity, novelty, popularity bias. Проверить, можно ли получить diversity без отдельного reranker.

**Потенциальный вклад.** Связь semantic ID design с responsible/diverse recommendation.

## 11. LLM-native textual identifiers vs discrete semantic IDs

**Вопрос.** Когда structured textual identifiers лучше discrete SIDs, а когда компактные SIDs выигрывают?

**Мотивация.** GRLM/TID-like подходи используют native language vocabulary LLM, а TIGER-like подходи вводят discrete codes. Сравнения часто не изолируют факторы: length, hallucination, grounding, cold-start, domain transfer.

**Идея метода.** Построить unified benchmark: один и тот же backbone, одинаковые data splits, несколько identifier types: title, generated TID, RQ-VAE SID, behavior-aware SID, hybrid TID+SID.

**Эксперимент.** In-domain, cross-domain, cold-start. Метрики: recommendation quality, valid generation rate, grounding accuracy, sequence length, decoding latency.

**Потенциальный вклад.** Практическая карта выбора identifier type для LLM-based generative recommendation.

## 12. Local re-indexing for streaming catalogs

**Вопрос.** Можно ли поддерживать semantic ID index в streaming catalog без полного retraining и без сильного degradation?

**Мотивация.** Production catalogs постоянно меняются. MERGE-like systems поднимают вопрос dynamic indexing, но нужен воспроизводимый public protocol.

**Идея метода.** Разработать local re-indexing algorithm: новые items назначаются существующим clusters, создают local split/merge только при threshold violation, а affected recommender components дообучаются локально.

**Эксперимент.** Simulated streaming на temporal splits: items приходят батчами, часть исчезает. Сравнить full retrain, frozen index, local re-index. Метрики: quality, number of changed codes, compute cost, latency proxy.

**Потенциальный вклад.** Public benchmark и baseline для streaming semantic ID lifecycle.
