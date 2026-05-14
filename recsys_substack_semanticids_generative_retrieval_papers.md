# Semantic IDs and Generative Retrieval Papers from RecSys Substack

Источник: `recsys_substack_papers_last_2_years.md`.

Критерии отбора: статьи и Substack-items, где в названии явно встречаются semantic ID(s), semantic token(s), semantic tokenization, SID, structured/hierarchical identifiers, item tokenization for generative recommendation, generative retrieval или generative information retrieval. Если paper source в исходном файле не найден, ссылка ведет на Substack item.

Найдено: 31.

Порядок внутри групп: сначала обзоры и вводные материалы, затем базовые методы токенизации/идентификаторов или generative retrieval, после этого масштабирование, оптимизация, продакшен и более специализированные работы. Статьи с semantic ID внутри generative retrieval отнесены в группу semantic IDs, если центральная тема — идентификаторы.

## Semantic IDs, Semantic Tokens, and Item Tokenization (21)

1. **[generative recommendation with semantic ids a practitioners handbook](https://recsys.substack.com/i/169809000/generative-recommendation-with-semantic-ids-a-practitioners-handbook)**
   - Почему здесь: лучший вводный материал перед чтением отдельных методов semantic IDs.
   - Теги: semantic IDs, generative recommendation, practitioner guide
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.115 for Jul 28 - Aug 03, 2025](https://recsys.substack.com/p/a-practitioners-guide-to-generative)
   - Аффилиации: н/д
   - Цитирования: н/д

2. **[end to end learnable item tokenization for generative recommendation](https://recsys.substack.com/i/148837728/end-to-end-learnable-item-tokenization-for-generative-recommendation)**
   - Почему здесь: базовая отправная точка для learnable item tokenization в generative recommendation.
   - Теги: item tokenization, semantic IDs, generative recommendation
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.69 for Sep 09 - Sep 15, 2024](https://recsys.substack.com/p/a-practical-guide-to-hnsw-vs-flat)
   - Аффилиации: н/д
   - Цитирования: н/д

3. **[store streamlining semantic tokenization and generative recommendation with a single llm](https://recsys.substack.com/i/148837728/store-streamlining-semantic-tokenization-and-generative-recommendation-with-a-single-llm)**
   - Почему здесь: ранний пример semantic tokenization через единый LLM-пайплайн.
   - Теги: semantic tokenization, generative recommendation, LLM
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.69 for Sep 09 - Sep 15, 2024](https://recsys.substack.com/p/a-practical-guide-to-hnsw-vs-flat)
   - Аффилиации: н/д
   - Цитирования: н/д

4. **[a simple contrastive framework of item tokenization for generative recommendation](https://recsys.substack.com/i/166951518/a-simple-contrastive-framework-of-item-tokenization-for-generative-recommendation)**
   - Почему здесь: следующий шаг после базовой токенизации — contrastive objective для качества semantic IDs.
   - Теги: item tokenization, semantic IDs, generative recommendation
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.110 for Jun 23 - Jun 29, 2025](https://recsys.substack.com/p/towards-ai-search-paradigm-zero-shot)
   - Аффилиации: Alibaba
   - Цитирования: н/д

5. **[generating long semantic ids in parallel for recommendation](https://recsys.substack.com/i/165843923/generating-long-semantic-ids-in-parallel-for-recommendation)**
   - Почему здесь: развивает тему длины и генерации идентификаторов, но еще до сложных промышленных оптимизаций.
   - Теги: semantic IDs, parallel generation, recommendation
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.108 for Jun 09 - Jun 15, 2025](https://recsys.substack.com/p/when-to-choose-graphrag-over-traditional)
   - Аффилиации: Meta
   - Цитирования: н/д

6. **[understanding generative recommendation with semantic ids from a model scaling view](https://recsys.substack.com/i/175168377/understanding-generative-recommendation-with-semantic-ids-from-a-model-scaling-view)**
   - Почему здесь: помогает понять, как semantic IDs ведут себя при масштабировании моделей.
   - Теги: semantic IDs, scaling laws, generative recommendation
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.124 for Sep 29 - Oct 05, 2025](https://recsys.substack.com/p/generative-embeddings-with-test-time)
   - Аффилиации: н/д
   - Цитирования: н/д

7. **[store semantic tokenization orthogonal rotation and efficient attention for scaling up ranking models](https://recsys.substack.com/i/180151580/store-semantic-tokenization-orthogonal-rotation-and-efficient-attention-for-scaling-up-ranking-models)**
   - Почему здесь: связывает semantic tokenization с масштабированием ranking-моделей.
   - Теги: semantic tokenization, ranking models, efficient attention
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.132 for Nov 24 - Nov 30, 2025](https://recsys.substack.com/p/unifying-retrieval-and-generation)
   - Аффилиации: Alibaba
   - Цитирования: н/д

8. **[cofirec coarse to fine tokenization for generative recommendation](https://recsys.substack.com/i/180773595/cofirec-coarse-to-fine-tokenization-for-generative-recommendation)**
   - Почему здесь: вводит более структурированную coarse-to-fine схему токенизации.
   - Теги: tokenization, semantic IDs, generative recommendation
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.133 for Dec 01 - Dec 07, 2025](https://recsys.substack.com/p/gpu-accelerated-feature-interaction)
   - Аффилиации: Meta
   - Цитирования: н/д

9. **[efficient optimization of hierarchical identifiers for generative recommendation](https://recsys.substack.com/i/182604769/efficient-optimization-of-hierarchical-identifiers-for-generative-recommendation)**
   - Почему здесь: продолжает тему структуры идентификаторов, но с фокусом на оптимизацию.
   - Теги: hierarchical identifiers, semantic IDs, generative recommendation
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.136 for Dec 22 - Dec 28, 2025](https://recsys.substack.com/p/why-multimodal-llms-fail-at-retrieval)
   - Аффилиации: University of Amsterdam
   - Цитирования: н/д

10. **[the best of the two worlds harmonizing semantic and hash ids for sequential recommendation](https://recsys.substack.com/i/181395964/the-best-of-the-two-worlds-harmonizing-semantic-and-hash-ids-for-sequential-recommendation)**
    - Почему здесь: стоит читать после базовых semantic IDs, чтобы понять гибрид semantic/hash подход.
    - Теги: semantic IDs, hash IDs, sequential recommendation
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.134 for Dec 08 - Dec 14, 2025](https://recsys.substack.com/p/a-geometric-analysis-of-bias-in-collaborative)
    - Аффилиации: н/д
    - Цитирования: н/д

11. **[unleashing the native recommendation potential llm based generative recommendation via structured term identifiers](https://recsys.substack.com/i/184736998/3-unleashing-the-native-recommendation-potential-llm-based-generative-recommendation-via-structured-term-identifiers)**
    - Почему здесь: переносит идею semantic IDs в structured term identifiers для LLM-based recommendation.
    - Теги: structured identifiers, semantic IDs, LLM-based generative recommendation
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.139 for Jan 12 - Jan 18, 2026](https://recsys.substack.com/p/the-journey-to-embedding-based-retrieval)
    - Аффилиации: Kuaishou
    - Цитирования: н/д

12. **[differentiable semantic id for generative recommendation](https://recsys.substack.com/i/186275831/5-differentiable-semantic-id-for-generative-recommendation)**
    - Почему здесь: более сложная тема end-to-end/differentiable обучения идентификаторов.
    - Теги: semantic IDs, differentiable identifiers, generative recommendation
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.141 for Jan 26 - Feb 01, 2026](https://recsys.substack.com/p/empirical-patterns-in-real-world)
    - Аффилиации: н/д
    - Цитирования: н/д

13. **[rethinking generative recommender tokenizer recsys native encoding and semantic quantization beyond llms](https://recsys.substack.com/i/187053456/5-rethinking-generative-recommender-tokenizer-recsys-native-encoding-and-semantic-quantization-beyond-llms)**
    - Почему здесь: переосмысливает tokenizer как recsys-native компонент, а не просто LLM-заимствование.
    - Теги: tokenizer, semantic quantization, generative recommendation
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.142 for Feb 02 - Feb 08, 2026](https://recsys.substack.com/p/the-case-for-semantic-tokens-in-modern)
    - Аффилиации: н/д
    - Цитирования: н/д

14. **[farewell to item ids unlocking the scaling potential of large ranking models via semantic tokens](https://recsys.substack.com/i/187053456/2-farewell-to-item-ids-unlocking-the-scaling-potential-of-large-ranking-models-via-semantic-tokens)**
    - Почему здесь: прикладное расширение semantic tokens на large ranking models.
    - Теги: semantic tokens, item IDs, ranking models
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.142 for Feb 02 - Feb 08, 2026](https://recsys.substack.com/p/the-case-for-semantic-tokens-in-modern)
    - Аффилиации: ByteDance
    - Цитирования: н/д

15. **[variable length semantic ids for recommender systems](https://recsys.substack.com/i/188579497/5-variable-length-semantic-ids-for-recommender-systems)**
    - Почему здесь: продвинутая вариация, где сама длина semantic ID становится моделируемым параметром.
    - Теги: semantic IDs, variable length identifiers
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.144 for Feb 16 - Feb 22, 2026](https://recsys.substack.com/p/the-case-for-multi-vector-contrastive)
    - Аффилиации: н/д
    - Цитирования: н/д

16. **[intrr a framework for integrating sid redistribution and length reduction](https://recsys.substack.com/i/189335924/5-intrr-a-framework-for-integrating-sid-redistribution-and-length-reduction)**
    - Почему здесь: специализированная работа про перераспределение и сокращение SID.
    - Теги: SID, semantic IDs, length reduction
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.145 for Feb 23 - Mar 01, 2026](https://recsys.substack.com/p/benchmarking-retrieval-and-re-ranking)
    - Аффилиации: Alibaba
    - Цитирования: н/д

17. **[stop treating collisions equally qualification aware semantic id learning for recommendation at industrial scale](https://recsys.substack.com/i/190073760/4-stop-treating-collisions-equally-qualification-aware-semantic-id-learning-for-recommendation-at-industrial-scale)**
    - Почему здесь: после базовых схем полезно перейти к collision handling в промышленном масштабе.
    - Теги: semantic IDs, collision handling, industrial scale
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.146 for Mar 02 - Mar 08, 2026](https://recsys.substack.com/p/bringing-the-muon-optimizer-to-large)
    - Аффилиации: Kuaishou
    - Цитирования: н/д

18. **[Beyond Static Collision Handling: Adaptive Semantic ID Learning for Multimodal Recommendation at Industrial Scale](https://arxiv.org/abs/2604.23522)**
    - Почему здесь: более новая и сложная версия темы collision handling, уже для multimodal/industrial setup.
    - Теги: semantic IDs, recommendation, industrial scale
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.154 for Apr 27 - May 03, 2026](https://recsys.substack.com/p/rethinking-semanticcollaborative)
    - Аффилиации: Kuaishou
    - Цитирования: н/д

19. **[mitigating collaborative semantic id staleness in generative retrieval](https://recsys.substack.com/i/194482549/6-mitigating-collaborative-semantic-id-staleness-in-generative-retrieval)**
    - Почему здесь: специализированная проблема устаревания collaborative semantic IDs в generative retrieval.
    - Теги: semantic IDs, generative retrieval, staleness
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.152 for Apr 13 - Apr 19, 2026](https://recsys.substack.com/p/a-late-chunking-approach-for-visual)
    - Аффилиации: ITMO University
    - Цитирования: н/д

20. **[deploying semantic id based generative retrieval for large scale podcast discovery at spotify](https://recsys.substack.com/i/191550289/1-deploying-semantic-id-based-generative-retrieval-for-large-scale-podcast-discovery-at-spotify)**
    - Почему здесь: практический production case после методов и проблем semantic IDs.
    - Теги: semantic IDs, generative retrieval, production system
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.148 for Mar 16 - Mar 22, 2026](https://recsys.substack.com/p/a-production-system-for-podcast-discovery)
    - Аффилиации: Spotify
    - Цитирования: н/д

21. **[reasoning over semantic ids enhances generative recommendation](https://recsys.substack.com/i/192277502/4-reasoning-over-semantic-ids-enhances-generative-recommendation)**
    - Почему здесь: самая продвинутая тема в группе — reasoning поверх semantic IDs.
    - Теги: semantic IDs, generative recommendation, reasoning
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.149 for Mar 23 - Mar 29, 2026](https://recsys.substack.com/p/disentangling-the-strengths-of-semantic)
    - Аффилиации: н/д
    - Цитирования: н/д

## Generative Retrieval (10)

1. **[A Survey of Generative Information Retrieval](http://arxiv.org/abs/2406.01197)**
   - Почему здесь: обзорная отправная точка для всей generative retrieval парадигмы.
   - Теги: generative information retrieval, survey
   - Раздел исходника: Other Information Retrieval Topics
   - Substack: [Vol.55 for Jun 03 - Jun 09, 2024](https://recsys.substack.com/p/a-comprehensive-review-of-state-of)
   - Аффилиации: н/д
   - Цитирования: 1

2. **[Bridging Search and Recommendation in Generative Retrieval: Does One Task Help the Other?](https://doi.org/10.1145/3640457.3688123)**
   - Почему здесь: ранняя работа, которая задает связь search и recommendation в generative retrieval.
   - Теги: generative retrieval, search, recommendation
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.75 for Oct 21 - Oct 27, 2024](https://recsys.substack.com/p/efficient-synthetic-data-generation)
   - Аффилиации: Signify; Spotify
   - Цитирования: 10

3. **[Unifying Generative and Dense Retrieval for Sequential Recommendation](http://arxiv.org/abs/2411.18814)**
   - Почему здесь: после общей постановки стоит посмотреть, как generative retrieval сочетается с dense retrieval.
   - Теги: generative retrieval, dense retrieval, sequential recommendation
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.81 for Dec 02 - Dec 08, 2024](https://recsys.substack.com/p/a-vision-for-information-retrieval)
   - Аффилиации: Meta AI
   - Цитирования: 0

4. **[Preference Discerning with LLM-Enhanced Generative Retrieval](http://arxiv.org/abs/2412.08604)**
   - Почему здесь: добавляет пользовательские preferences и LLM-enhancement поверх базовой retrieval-постановки.
   - Теги: generative retrieval, preferences, LLM-enhanced recommendation
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.82 for Dec 09 - Dec 15, 2024](https://recsys.substack.com/p/semantic-retrieval-at-walmart-text)
   - Аффилиации: Meta
   - Цитирования: 0

5. **[generative retrieval and alignment model a new paradigm for e commerce retrieval](https://recsys.substack.com/i/160560138/generative-retrieval-and-alignment-model-a-new-paradigm-for-e-commerce-retrieval)**
   - Почему здесь: переход от общей постановки к e-commerce retrieval и alignment.
   - Теги: generative retrieval, alignment, e-commerce retrieval
   - Раздел исходника: Other Information Retrieval Topics
   - Substack: [Vol.98 for Mar 31 - Apr 06, 2025](https://recsys.substack.com/p/storage-efficient-multi-vector-retrieval)
   - Аффилиации: н/д
   - Цитирования: н/д

6. **[on synthetic data strategies for domain specific generative retrieval](https://recsys.substack.com/i/158082872/on-synthetic-data-strategies-for-domain-specific-generative-retrieval)**
   - Почему здесь: полезно читать после e-commerce/domain постановки, чтобы понять роль synthetic data.
   - Теги: generative retrieval, synthetic data, domain adaptation
   - Раздел исходника: Other Information Retrieval Topics
   - Substack: [Vol.93 for Feb 24 - Mar 02, 2025](https://recsys.substack.com/p/agent-centric-information-access)
   - Аффилиации: Amazon
   - Цитирования: н/д

7. **[alleviating llm based generative retrieval hallucination in alipay search](https://recsys.substack.com/i/160047917/alleviating-llm-based-generative-retrieval-hallucination-in-alipay-search)**
   - Почему здесь: отдельная практическая проблема generative retrieval — hallucination в search.
   - Теги: generative retrieval, hallucination, search
   - Раздел исходника: RAG and Knowledge-Intensive QA
   - Substack: [Vol.97 for Mar 24 - Mar 30, 2025](https://recsys.substack.com/p/a-task-specific-approach-to-recommender)
   - Аффилиации: н/д
   - Цитирования: н/д

8. **[killing two birds with one stone unifying retrieval and ranking with a single generative recommendation model](https://recsys.substack.com/i/162099372/killing-two-birds-with-one-stone-unifying-retrieval-and-ranking-with-a-single-generative-recommendation-model)**
   - Почему здесь: расширяет generative retrieval до объединения retrieval и ranking в одной модели.
   - Теги: generative recommendation, retrieval, ranking
   - Раздел исходника: Recommender Systems, CTR and Ads
   - Substack: [Vol.101 for Apr 21 - Apr 27, 2025](https://recsys.substack.com/p/secure-hybrid-knowledge-retrieval)
   - Аффилиации: н/д
   - Цитирования: н/д

9. **[generative retrieval overcomes limitations of dense retrieval but struggles with identifier ambiguity](https://recsys.substack.com/i/193763322/8-generative-retrieval-overcomes-limitations-of-dense-retrieval-but-struggles-with-identifier-ambiguity)**
   - Почему здесь: более поздний анализ trade-off между dense retrieval и generative retrieval.
   - Теги: generative retrieval, identifier ambiguity, dense retrieval
   - Раздел исходника: Dense Retrieval, Embeddings and Representations
   - Substack: [Vol.151 for Apr 06 - Apr 12, 2026](https://recsys.substack.com/p/rethinking-negative-sampling-for)
   - Аффилиации: Vienna University of Economics and Business
   - Цитирования: н/д

10. **[vectorizing the trie efficient constrained decoding for llm based generative retrieval on accelerators](https://recsys.substack.com/i/189335924/3-vectorizing-the-trie-efficient-constrained-decoding-for-llm-based-generative-retrieval-on-accelerators)**
    - Почему здесь: инфраструктурная и наиболее техническая работа про constrained decoding на accelerators.
    - Теги: generative retrieval, constrained decoding, infrastructure
    - Раздел исходника: Recommender Systems, CTR and Ads
    - Substack: [Vol.145 for Feb 23 - Mar 01, 2026](https://recsys.substack.com/p/benchmarking-retrieval-and-re-ranking)
    - Аффилиации: н/д
    - Цитирования: н/д
