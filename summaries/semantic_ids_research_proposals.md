# Research proposals: Semantic IDs, tokenization and indexing for recommender systems

Обновлено: 2026-05-26. Список собран после просмотра 93 работ из раздела `Semantic IDs, tokenization и indexing` в `summaries/papers.html` и дополнительной проверки свежих arXiv/ACM источников. Фокус: идеи, которые реально проверить за 3-6 месяцев на публичных датасетах или аккуратном industrial-like replay без закрытого production traffic.

## Короткая карта домена

Главный сдвиг за 2025-2026: semantic IDs перестали быть "preprocessing detail". Работы вроде TIGER/LETTER/CoST задали RQ-VAE/contrastive tokenizer baseline, но последние статьи двигают тему в сторону lifecycle, evaluation reliability, variable-length codes, end-to-end alignment, expressiveness of decoding trees, long-tail transfer, LLM vocabulary grounding и production-grade indexing.

Практический research gap: большинство top-line улучшений все еще трудно сравнивать, потому что меняются одновременно tokenizer, generator, metrics, decoding, collision handling, datasets и serving assumptions. Поэтому наиболее перспективны темы, где можно изолировать один bottleneck и дать воспроизводимый protocol.

## 1. Collision-aware evaluation benchmark for SID recommenders

**Вопрос.** Насколько результаты SID-based generative recommendation завышены из-за того, что SID-level hit засчитывает collision group, а не конкретный item?

**Мотивация.** Свежая работа "How Reliable Are Semantic-ID Tokenizer Comparisons in Generative Recommendation?" показывает, что SID-level метрики могут сильно расходиться с item-level метриками при коллизиях. Это ставит под вопрос честность сравнений tokenizer'ов.

**Идея.** Собрать evaluation suite, который для каждого tokenizer'а считает: SID-level Recall/NDCG, item-level Recall/NDCG, collision group size, harmful collision rate, head/tail inflation и ranking flips между tokenizer'ами.

**Эксперимент за 3-6 месяцев.** Amazon Reviews, Yelp, MIND. Tokenizers: random hash, RQ-VAE, CoST/SimCIT-style contrastive, balanced tree, variable-length. Для каждого сделать одинаковый generator и одинаковый constrained decoding.

**Потенциальный вклад.** Не новый tokenizer, а надежный evaluation protocol. Это хороший short paper/workshop paper, потому что он может переинтерпретировать существующие результаты.

**Ключевые источники.** How Reliable Are Semantic-ID Tokenizer Comparisons; FORGE; GRID handbook.

## 2. Stability-aware continual tokenization for live catalogs

**Вопрос.** Можно ли обновлять semantic IDs при drift items/interactions без полной переиндексации и без разрушения compatibility с обученным generator'ом?

**Мотивация.** DACT, collaborative SID staleness и MERGE показывают, что lifecycle важнее single-shot tokenization: catalog grows, item content changes, interactions drift, old codes become stale.

**Идея.** Tokenizer objective с двумя ограничениями: recommendation alignment и stability penalty. Старые items получают мягкий штраф за изменение prefix/edit distance, новые items вставляются через local split/merge, drifting items переassign'ятся selectively.

**Эксперимент.** Rolling temporal splits. Метрики: Recall/NDCG, drift rate, changed-prefix percentage, warm-start degradation, amount of retraining, collision churn.

**Потенциальный вклад.** Production-realistic protocol для continual SID lifecycle, где качество измеряется вместе со стоимостью миграции index.

**Ключевые источники.** DACT; Mitigating Collaborative Semantic ID Staleness; MERGE; Meta semantic ID stability case study.

## 3. Variable-length SID under a fixed latency budget

**Вопрос.** Как распределять длину semantic ID между head, torso и tail items, если inference budget ограничен средним числом generated tokens?

**Мотивация.** Variable-Length Semantic IDs, VarLenRec, CapsID и IntRR сходятся в одном: fixed-length SID плохо использует capacity. Popular items часто можно кодировать короче, tail items требуют более детальной семантики.

**Идея.** Сравнить три стратегии length allocation: popularity prior, uncertainty/density prior, learned soft length controller. Ввести constraint на среднюю длину и измерять accuracy-efficiency Pareto.

**Эксперимент.** Amazon Beauty/Sports/Toys, Yelp. Baselines: fixed RQ-VAE, VarLen heuristic, CapsID-like soft routing approximation, SemanticBPE/post-merge.

**Потенциальный вклад.** Практическая recipe: какой length policy выбирать при заданном latency budget и long-tail profile.

**Ключевые источники.** Learning Variable-Length Tokenization; CapsID; Variable-Length Semantic IDs; IntRR.

## 4. Latent-conditioned decoding trees for SID expressiveness

**Вопрос.** Ограничивает ли autoregressive SID tree способность модели различать user-specific preferences между близкими leaves?

**Мотивация.** Latte показывает expressiveness limit: близкие leaves в decoding tree получают скоррелированные probabilities. Это может быть фундаментальным bottleneck не tokenizer'а, а factorization.

**Идея.** Сравнить способы ослабить tree-induced coupling: latent token before SID, multiple valid SID paths, order-agnostic generation, parallel long SID generation, stochastic path augmentation.

**Эксперимент.** Синтетический benchmark с контролируемыми substitute/complement pairs плюс real datasets. Метрики: pairwise preference separability, tree-distance/probability correlation, Recall/NDCG, beam efficiency.

**Потенциальный вклад.** Теоретически и эмпирически чистая работа про то, когда SID tree помогает, а когда связывает руки generator'у.

**Ключевые источники.** Expressiveness Limits/Latte; SETRec; RPG; TrieRec.

## 5. Discriminative-to-generative tokenizer distillation

**Вопрос.** Можно ли построить SID tokenizer, который наследует decision boundary сильного ranker'а, а не только content/collaborative embedding geometry?

**Мотивация.** "Discrimination Is Generation" и UniRec двигают мысль, что retrieval и ranking можно связать через tokenizer. Это особенно ценно, если в компании уже есть сильный ranker, но нужен generative retrieval interface.

**Идея.** Distill ranker logits/cross features into tokenizer assignment: items, которые ranker различает в данном user/context, не должны схлопываться в один SID; items с похожей ranker response могут share prefix.

**Эксперимент.** Обучить SASRec/DeepFM-like ranker, затем SID tokenizer с ranker-informed contrastive/assignment loss. Сравнить с RQ-VAE, CoST, DIG-style end-to-end.

**Потенциальный вклад.** Мост между production discriminative stack и generative retrieval без полного отказа от существующего ranker'а.

**Ключевые источники.** Discrimination Is Generation; UniRec; RecoChain; SID-Coord.

## 6. Grounded initialization for new SID vocabulary tokens

**Вопрос.** Насколько initialization новых SID tokens в pretrained LM влияет на sample efficiency, valid generation и semantic grounding?

**Мотивация.** GTI показывает, что mean initialization новых vocabulary tokens может загонять SID tokens в degenerate subspace. SIDReasoner и GRLM отдельно показывают, что SID-language alignment становится bottleneck для LLM-based recommenders.

**Идея.** Сравнить initialization schemes: random, mean vocabulary, content-title grounding, category grounding, neighbor-contrastive grounding, prefix-aware grounding.

**Эксперимент.** T5/Qwen-style backbone, SID vocabulary от RQ-VAE/CoST. Low-data regimes: 1%, 5%, 10%, 100%. Метрики: Recall/NDCG, valid SID rate, convergence speed, SID-to-text probing accuracy.

**Потенциальный вклад.** Небольшая, хорошо изолированная работа про bottleneck, который легко упускают в LLM-GR papers.

**Ключевые источники.** Grounded Token Initialization; SIDReasoner; GRLM/TID; STORE.

## 7. Domain-aware semantic codebooks

**Вопрос.** Какие domain constraints должны быть встроены в SID/codebook, а какие лучше оставить как обычные features?

**Мотивация.** Pro-GEO показывает пользу geographic proximity для local services, CARD -- non-uniform visual quantization, GRACE/UniRec -- attribute/CoT chains. Значит, "universal" SID может быть слабее domain-aware identifier.

**Идея.** Создать framework для domain constraints в tokenizer: geo distance, visual similarity, taxonomy/category, seller/brand, temporal freshness. Сравнить feature-as-input vs feature-as-identifier-token.

**Эксперимент.** POI/local dataset для geography, Amazon для visual/category/brand, MIND для news freshness. Метрики: Recall/NDCG, cold-start, domain-specific validity, code utilization.

**Потенциальный вклад.** Практическая карта: когда domain semantics нужно кодировать в identifier path.

**Ключевые источники.** Pro-GEO; CARD; GRACE; UniRec; HiD-VAE.

## 8. Qualification-aware collision taxonomy

**Вопрос.** Какие collision pairs вредны, а какие дают полезное sharing между похожими items?

**Мотивация.** QuaSID и AdaSID показывают, что treating collisions equally слишком грубо. Но нужен reproducible public taxonomy, иначе collision handling остается industrial black box.

**Идея.** Классифицировать collisions: substitutes, complements, category-only, popularity-dominated, false semantic, temporal/freshness collision. Добавить loss, который штрафует только harmful collisions.

**Эксперимент.** Использовать co-click/co-purchase graph, taxonomy и embedding similarity. Отдельно оценить head/tail, substitute/complement retrieval и diversity.

**Потенциальный вклад.** Более точная диагностика tokenizer quality, чем raw collision rate.

**Ключевые источники.** QuaSID; AdaSID; CRAB; FORGE.

## 9. Canonical-plus-personalized semantic IDs

**Вопрос.** Можно ли получить выгоду personalized/context-aware tokenization, сохранив canonical item index для serving/cache/trie?

**Мотивация.** Pctx и PIT показывают, что fixed SID mapping игнорирует user intent: один item может быть близок к разным clusters для разных пользователей. Но полностью personalized IDs плохо deployable.

**Идея.** Разделить identifier на canonical semantic prefix и personalized/context suffix. Prefix нужен для trie/constrained decoding/cache, suffix используется для local reranking или candidate disambiguation.

**Эксперимент.** Сравнить static SID, fully personalized SID, canonical+personalized hybrid. Метрики: Recall/NDCG, number of codes per item, cache-hit proxy, valid generation, latency proxy.

**Потенциальный вклад.** Deployable compromise между personalization и stable indexing.

**Ключевые источники.** Pctx; PIT; SETRec; Latte.

## 10. Tokenizer diagnostics that predict downstream quality

**Вопрос.** Какие дешевые tokenizer metrics предсказывают downstream GR quality до дорогого обучения generator'а?

**Мотивация.** GRID/FORGE/R3-VAE/CRAB показывают, что reconstruction loss недостаточен. Нужны proxy metrics, которые позволяют быстро отсеивать плохие tokenizers.

**Идея.** Собрать diagnostics: code utilization, entropy, collision purity, item-level collision inflation, prefix mutual information, behavioral neighborhood preservation, tail sharing, tree balance, drift.

**Эксперимент.** Обучить family tokenizer variants и проверить correlation с downstream Recall/NDCG/valid rate на нескольких datasets.

**Потенциальный вклад.** Benchmark suite и practical checklist для SID tokenizer papers.

**Ключевые источники.** GRID handbook; FORGE; R3-VAE; CRAB; How Reliable Are SID Tokenizer Comparisons.

## 11. Hybrid semantic/hash/attribute identifiers

**Вопрос.** Можно ли совместить semantic sharing, exact identity memorization и interpretable attributes в одном deployable identifier?

**Мотивация.** Semantic-only IDs помогают cold-start/tail, но могут терять individuality head items. Hash IDs дают identity, но не дают transfer. UniRec добавляет Chain-of-Attribute, а Best-of-Two-Worlds harmonizes semantic and hash IDs.

**Идея.** Identifier path: attribute/category prefix, semantic middle levels, hash or learned suffix. Длина suffix зависит от popularity и ambiguity.

**Эксперимент.** Head/tail/cold split. Baselines: random hash, semantic-only, hash+semantic dual branch, attribute+SID. Метрики: head precision, tail Recall, collision harmfulness, interpretability.

**Потенциальный вклад.** Production-friendly SID design для систем, где нельзя жертвовать exact identity популярных объектов.

**Ключевые источники.** Best of Two Worlds; UniRec; HiD-VAE; Better Generalization with Semantic IDs.

## 12. Search-recommendation shared identifiers

**Вопрос.** Как должен выглядеть SID, если один generative model обслуживает и query search, и recommendation?

**Мотивация.** Spotify SID for joint search/recommendation, GenSAR, UniSearch и bridging search+recommendation показывают, что shared model возможна, но task objectives конфликтуют: search требует query grounding, rec требует behavior prior.

**Идея.** Identifier с shared semantic/collaborative prefix и task-specific suffix. Для search suffix оптимизируется на query intent, для rec -- на sequential behavior.

**Эксперимент.** Dataset с queries и interactions или synthetic query generation поверх Amazon/MIND. Метрики: search Recall/NDCG, rec Recall/NDCG, task interference, generated invalid IDs.

**Потенциальный вклад.** Clear ablation для unified generative retrieval, где task sharing проверяется через identifier design.

**Ключевые источники.** Semantic IDs for Joint Generative Search and Recommendation; GenSAR; Bridging Search and Recommendation; UniSearch.

## 13. SID-aware long-history memory

**Вопрос.** Как использовать semantic ID hierarchy для ultra-long user histories без превращения sequence в тысячи tokens?

**Мотивация.** GLASS, UxSID, ACERec и STAMP атакуют overhead длинных SID/history sequences. Пока неясно, что лучше: compress input, trim tokens, group by SID prefix или use semantic memory.

**Идея.** Memory на уровне SID groups: долгосрочная история агрегируется по semantic prefixes, а short-term history остается item-level. Target-aware gate выбирает, какие groups раскрывать в full tokens.

**Эксперимент.** Long-sequence datasets или synthetic long histories. Baselines: truncation, SID-Tier, token trimming, long SID merger. Метрики: Recall/NDCG vs tokens processed, tail performance, latency proxy.

**Потенциальный вклад.** Efficiency-focused работа для realistic user histories.

**Ключевые источники.** GLASS; UxSID; ACERec; STAMP.

## 14. End-to-end tokenizer optimization with anti-collapse and stability

**Вопрос.** Можно ли совместить end-to-end recommendation gradients с codebook utilization и stable item-to-code mapping?

**Мотивация.** ETEGRec, BLOGER, DIGER и UniGRec решают objective mismatch, но end-to-end updates часто рискуют code collapse, instability и train/inference gap.

**Идея.** Bi-level или differentiable tokenizer с тремя regularizers: exploration/anti-collapse, recommendation alignment, prefix stability относительно previous checkpoint.

**Эксперимент.** Static split плюс rolling split. Метрики: Recall/NDCG, code utilization, changed-prefix rate, train/inference mismatch, generator warm-start degradation.

**Потенциальный вклад.** Практически безопасный вариант end-to-end SID learning.

**Ключевые источники.** BLOGER; DIGER; UniGRec; ETEGRec; DACT.

## 15. SID subword compression: SemanticBPE, trimming and token mergers

**Вопрос.** Какие SID tokens действительно нужны на input/output side, а какие можно merge/trim без потери item identity?

**Мотивация.** CapsID+SemanticBPE, STAMP, IntRR и ACERec показывают, что длинные или high-granularity SIDs дают capacity, но создают overhead и semantic dilution.

**Идея.** Изучить compression operators: adjacent SID BPE, attention-based token merger, semantic adaptive pruning, last-level trimming. Важно отдельно оценивать input compression и output decoding compression.

**Эксперимент.** Fixed tokenizer, разные compression policies. Метрики: Recall/NDCG, valid generation, collision inflation, average generated tokens, latency proxy.

**Потенциальный вклад.** Простая engineering paper с сильной практической ценностью: меньше tokens при том же item-level quality.

**Ключевые источники.** CapsID; STAMP; IntRR; ACERec; RPG.

## 16. Codebook rebalancing for popularity bias and fairness

**Вопрос.** Можно ли снижать popularity bias на уровне SID codebook, а не только reranker loss?

**Мотивация.** CRAB показывает, что tokenization imbalance сам усиливает popularity bias. Это открывает отдельную линию: debiasing через restructuring identifier space.

**Идея.** Post-hoc или online codebook rebalance: split over-popular tokens, merge underused semantically close subtrees, regularize subtree exposure. Сохранять semantic consistency и минимизировать mapping churn.

**Эксперимент.** Long-tail splits. Метрики: Recall/NDCG, tail coverage, popularity exposure, head degradation, code churn.

**Потенциальный вклад.** Fairness/bias angle внутри SID literature, где сейчас много accuracy-only работ.

**Ключевые источники.** CRAB; AKT-Rec; variable-length SID papers; FORGE.

## 17. Hierarchy supervision: taxonomy, behavior graph or LLM tags?

**Вопрос.** Что лучше задает hierarchy для semantic IDs: catalog taxonomy, behavior co-occurrence graph или LLM-generated attributes?

**Мотивация.** HiD-VAE, CAT-ID2, CoFiRec и UniRec используют разные источники hierarchy. Но taxonomy часто отражает merchant organization, а behavior graph -- реальные intents.

**Идея.** Единый tokenizer, где supervision source меняется: taxonomy path, LLM tags, graph coarsening, hybrid taxonomy-prefix plus behavior-suffix.

**Эксперимент.** Amazon/MIND/POI datasets с metadata. Метрики: Recall/NDCG, interpretability, subtree behavioral purity, collision harmfulness, diversity.

**Потенциальный вклад.** Ответ на базовый design question: какую иерархию semantic ID должен кодировать.

**Ключевые источники.** HiD-VAE; CAT-ID2; CoFiRec; UniRec.

## 18. Continuous generative recommenders as a control arm for SID papers

**Вопрос.** Когда discrete SID generation действительно лучше continuous diffusion/embedding generation с ANN retrieval?

**Мотивация.** DreamRec, DimeRec и Prompt-to-Slate обходят discrete identifiers через continuous generation. Для SID papers это важный control: если дискретизация дает overhead и collisions, нужен честный baseline.

**Идея.** Unified benchmark: same encoder/history model, two output heads -- SID generation and continuous next-interest vector. Сравнить quality, latency, update cost, cold-start и validity.

**Эксперимент.** Sequential recommendation datasets. Добавить controlled catalog churn: для SID нужна reindexing, для continuous нужен ANN update.

**Потенциальный вклад.** Честная граница применимости SID-based GR.

**Ключевые источники.** DreamRec; DimeRec; Prompt-to-Slate; HSTU/action-native generative recommenders.

## 19. Multi-identifier training with single-identifier serving

**Вопрос.** Можно ли получить robustness от нескольких identifiers на item при обучении, но оставить один canonical SID на inference?

**Мотивация.** MTGRec показывает ценность multi-identifier pretraining, а personalized/variable tokenizer papers показывают, что один SID не всегда достаточен. Но serving требует canonical mapping.

**Идея.** Training augmentation: adjacent tokenizer checkpoints, stochastic quantization, multiple hierarchy sources, content-vs-collaborative SIDs. Serving: distill into one canonical tokenizer or choose canonical path by validation stability.

**Эксперимент.** Compare no augmentation, multi-ID training, multi-ID serving, distilled single-ID serving. Метрики: Recall/NDCG, robustness to tokenizer noise, drift, inference cost.

**Потенциальный вклад.** Практичный compromise между richer supervision и простым serving.

**Ключевые источники.** MTGRec; Pctx; PIT; SETRec.

## 20. Open SID lifecycle benchmark

**Вопрос.** Как сравнивать SID methods не только по static offline quality, а по полному lifecycle: build, update, decode, migrate, rollback?

**Мотивация.** Industrial papers постоянно упоминают latency, catalog churn, code collisions, index rebuild и online A/B, но public benchmarks редко это моделируют.

**Идея.** Benchmark spec: temporal catalog snapshots, item additions/deletions, content edits, interaction drift, fixed training budget, fixed decoding budget. Метрики включают quality, build time, reindex ratio, changed-prefix ratio, collision inflation, serving memory, valid generation.

**Эксперимент.** Начать с Amazon Reviews 2023 temporal snapshots и MIND news freshness, затем добавить Yelp/POI.

**Потенциальный вклад.** Инфраструктурный paper/toolkit. Хорошо сочетается с GRID/FORGE и может стать базой для нескольких method papers.

**Ключевые источники.** GRID handbook; FORGE; MERGE; DACT; How Reliable Are SID Tokenizer Comparisons.

## Приоритет на ближайшие 3 месяца

Если нужно выбрать не 20, а 5 самых сильных и реалистичных направлений:

1. **Collision-aware evaluation benchmark** -- высокая новизна, легко воспроизводимо, актуально после майской работы 2026.
2. **Stability-aware continual tokenization** -- production-relevant, есть много свежих anchors, можно делать на temporal splits.
3. **Variable-length SID under latency budget** -- активная тема, понятный accuracy-efficiency протокол.
4. **Grounded initialization for SID vocabulary** -- изолированный bottleneck для LLM-GR, дешевые эксперименты.
5. **Tokenizer diagnostics that predict downstream quality** -- полезная инфраструктура, хорошо расширяет GRID/FORGE.
