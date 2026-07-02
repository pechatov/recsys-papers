---
title: "How Reliable Are Semantic-ID Tokenizer Comparisons in Generative Recommendation?"
category: "semantic_ids_tokenization_indexing"
slug: "how_reliable_are_semantic_id_tokenizer_comparisons_in_generative_recommendation_summary"
catalogId: "paper-how_reliable_are_semantic_id_tokenizer_comparisons_in_generative_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2605.25330"
---
> **Авторы:** Qian Zhang, Lech Szymanski, Haibo Zhang, Jeremiah D. Deng.
>
> **Аффилиации:** School of Computing, University of Otago; University of New South Wales.
>
> **Источник:** arXiv:2605.25330v1 от 2026-05-25.

## 1. Коротко: о чем статья

Статья проверяет базовое предположение большинства работ по Semantic IDs: если генеративная модель сгенерировала SID target item'а, значит она нашла сам item. Авторы показывают, что это не всегда верно. Tokenizer сжимает item features в ограниченное пространство кодов, поэтому несколько разных items могут получить один и тот же SID. Тогда SID-level Hit@K засчитывает попадание в collision group как попадание в конкретный item.

Главный результат: SID-level метрики могут сильно завышать item-level качество и менять ранжирование tokenizers. На четырех датасетах и пяти tokenizers доля items в collision groups доходит до 30.5%, а inflation Hit@10 доходит примерно до 103-107%. Поэтому сравнение tokenizer-ов без учета collision может поощрять модели, которые хуже различают конкретные items, но чаще попадают в общий semantic bucket.

Работа предлагает два инструмента: **Collision-Corrected Evaluation (CCE)** для честной item-level оценки уже сгенерированных SID списков и **Zero-Collision Reassignment (ZCR)** как post-tokenizer процедуру, которая минимально меняет последний уровень кода, чтобы получить collision-free assignment.

<figure class="paper-figure">
  <img src="../../assets/how_reliable_sid_tokenizer_comparisons/collision_overview.png" alt="SID collision inflates SID-level generative recommendation metrics">
  <figcaption>Рисунок 1. Центральная проблема: при SID collision одна generated sequence соответствует группе items, поэтому стандартный SID-level hit переоценивает item-level recommendation quality.</figcaption>
</figure>

## 2. Контекст: почему это важно для всех SID papers

Generative recommendation с semantic IDs обычно обучает autoregressive model генерировать последовательность discrete codes следующего item'а. Evaluation часто делается на уровне SID: target считается найденным, если target SID есть в beam output. Это удобно, но корректно только при one-to-one mapping `item -> SID`.

В реальных tokenizers mapping many-to-one возникает естественно. RQ-VAE, RK-Means, LETTER, QuaSID и похожие методы оптимизируют reconstruction, semantic grouping или collaborative alignment, но не обязаны гарантировать уникальность каждого item'а. Если два похожих товара, видео или заведения получают один SID, beam search уже не может различить их через sequence likelihood.

Для research это особенно опасно, потому что tokenizer с большей collision rate может казаться лучше: он чаще генерирует общий SID cluster, и стандартный metric засчитывает это как успех. На item-level такой успех должен быть частичным или нулевым, если конкретный target не выбран.

## 3. Метод: Collision-Corrected Evaluation

CCE вводит ItemHit@K и ItemNDCG@K. Если generated SID уникально соответствует target item, credit остается обычным. Если generated SID соответствует collision group размера `|C(s)|`, то item-level credit делится на размер группы. Интуитивно модель указала не item, а корзину, поэтому ей нельзя начислять полный балл.

Это важное практическое свойство: CCE можно применить к уже обученным GR models и уже существующим tokenizer outputs. Не нужно переобучать tokenizer или generator, достаточно знать SID-to-items mapping и generated ranked SID list.

Авторы показывают, что inflation зависит от collision rate. Например, на Scientific RK-Means имеет Coll.% 30.52, SID-level H@10 0.1330, но corrected IH@10 только 0.0654, то есть inflation 103.36%. RQ-VAE на том же датасете имеет Coll.% 20.42 и inflation 72.48%. LETTER и QuaSID с меньшими collision rates дают намного меньшую inflation.

## 4. Zero-Collision Reassignment

ZCR решает другую задачу: не только оценить collisions честно, но и исправить assignment. Метод сохраняет первые `L-1` уровней SID и переassign-ит только последний уровень внутри каждого prefix group. Цель - сделать last-level codes уникальными внутри prefix, минимизируя суммарный reassignment cost.

Практически это похоже на constrained assignment problem. Если в одном prefix group два item'а получили один последний code, ZCR выбирает, какой item оставить на текущем code, а какой перевести на свободный code, исходя из минимальной общей distortion. Это лучше greedy-правила, которое принимает локальное решение.

В таблице reassignment cost ZCR снижает cost относительно greedy на 8.21-12.19% для RK-Means и на 18.17-24.00% для RQ-VAE. Это значит, что collision-free correction можно сделать без грубого разрушения исходной semantic structure tokenizer-а.

## 5. Эксперименты

Авторы проверяют четыре датасета: Scientific, Cell, Beauty и Yelp. Tokenizers: RK-Means, RQ-VAE, LETTER, QuaSID и MQL4GRec как zero-collision reference. Generator обучается одинаково, beam width равен 20, результаты усредняются по трем seeds.

Ключевой результат по native vs +ZCR: item-level метрики почти всегда растут после collision removal. Например, RK-Means+ZCR на Scientific повышает ItemHit@10 с 0.0654 до 0.0812, а ItemNDCG@10 с 0.0357 до 0.0504. Для RQ-VAE на Scientific ItemHit@10 растет с 0.0665 до 0.0800, а ItemNDCG@10 с 0.0379 до 0.0514.

Самый важный вывод не в том, что ZCR улучшает конкретный tokenizer, а в том, что ranking tokenizer-ов меняется. На Scientific, Cell и Beauty RK-Means лидирует по SID-level H@10, но становится последним по corrected IH@10. LETTER и QuaSID выглядят сильнее после item-level correction.

## 6. Что это меняет в чтении прошлых работ

Результаты TIGER-like работ, где метрика считается только по SID string match, нужно читать осторожно. Если paper не показывает collision rate, unique item rate, collision group size и item-level metrics, то reported Recall/NDCG могут быть завышены.

Особенно подозрительны сравнения tokenizers с разной емкостью codebook, разной длиной SID и разной collision policy. Tokenizer с короткими кодами и высоким semantic grouping может дать красивый SID Recall, но плохую способность отличать близкие items. Для production retrieval это критично: downstream ranker может получить слишком широкий или неправильный collision group.

## 7. Сильные стороны

- Статья формулирует evaluation bug, который легко пропустить при работе только с SID sequences.
- CCE применим без переобучения и подходит как audit layer для существующих results.
- ZCR дает практичный способ сделать tokenizers collision-free, меняя только last-level code.
- Эксперименты показывают не только metric inflation, но и реальные reversals в ranking tokenizers.

## 8. Ограничения и вопросы

CCE с fractional credit предполагает равномерную неопределенность внутри collision group. В production downstream scorer может различать items внутри группы по дополнительным features, и тогда фактический item-level utility зависит от всей retrieval-ranking цепочки.

ZCR сохраняет первые уровни SID, поэтому не исправляет случаи, где prefix grouping уже плохой с точки зрения preference. Это не новый tokenizer objective, а post-processing для идентифицируемости.

Работа не доказывает, что любой collision вреден. Для retrieval pipeline collision group может быть приемлемым, если она далее корректно раскрывается и ранжируется. Но тогда paper обязан измерять item-level post-resolution quality, а не только SID-level hit.

## 9. Вывод

Это обязательная audit-paper для semantic-ID recommendation. Главный takeaway: **SID-level Recall/NDCG нельзя считать item-level recommendation metric без проверки collision-free mapping**. Для честного сравнения tokenizers нужно публиковать collision rate, max/mean collision group size, corrected ItemHit/ItemNDCG и, желательно, zero-collision ablation.
