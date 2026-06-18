---
title: "CAT-ID$^2$: Category-Tree Integrated Document Identifier Learning for Generative Retrieval In E-commerce"
category: "semantic_ids_tokenization_indexing"
slug: "cat_id2_category_tree_integrated_document_identifier_learning_summary"
catalogId: "paper-cat_id2_category_tree_integrated_document_identifier_learning_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/cat_id2_category_tree_integrated_document_identifier_learning_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2511.01461"
---
> **Авторы:** Xiaoyu Liu, Fuwei Zhang, Yiqing Wu, Xinyu Jia, Zenghua Xia, Fuzhen Zhuang, Zhao Zhang, Fei Jiang, Wei Lin.
>
> **Аффилиации:** Beihang University; Chinese Academy of Sciences; Meituan.
>
> **Индустрия:** Meituan e-commerce search / generative retrieval
>
> **Первичные источники:** arXiv:2511.01461 HTML/PDF.

## Коротко

CAT-ID2 learns generative retrieval DocIDs constrained by e-commerce category trees.

Разбор ниже фокусируется на том, что именно добавляет paper к семейству Semantic ID/tokenization методов.

Отдельно отмечены данные, метрики, ablations и production notes, если они раскрыты в источнике.

## Контекст

Good DocIDs should be semantic, category-consistent, balanced in cluster size and dispersed enough to distinguish products.

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

- Baselines: BM25, DPR, Sen-T5, MPNet, DSI-naive, DSI-semantic, Hi-Gen, NCI, TIGER.
- Losses: Hierarchical Class Constraint Loss, Cluster Scale Constraint Loss, Dispersion Loss.
- Ablation Table 3 on ESCI-us: TIGER R@10 4.93/R@100 18.60; CAT-ID2 R@10 5.86/R@100 21.00; HCCL or CSCL alone hurts, combination helps.
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
- Metric: Recall@5/10/20/50/100; paper explicitly does not use NDCG because recall stage is primary.

### Подробная схема алгоритма CAT-ID2

CAT-ID2 строит DocID не как свободный RQ-VAE код, а как learned identifier, ограниченный e-commerce category tree. Алгоритм можно разделить на построение category-aware token space, обучение DocID generator и serving-time recall.

1. **Нормализовать category path.** Для каждого товара берется category tree depth до 3: слишком мелкие ветки удаляются, слишком глубокие обрезаются. Это задает внешний hierarchical prior.
1. **Получить document representation.** Product title/description и category metadata кодируются T5-base/RQ-VAE pipeline'ом в latent vector.
1. **Назначить discrete identifier.** RQ-VAE строит multi-token DocID, но assignment штрафуется не только reconstruction, а также category-aware constraints.
1. **Применить Hierarchical Class Constraint Loss.** Товары из одной category path должны иметь согласованные prefixes/regions identifier space. Это делает DocID интерпретируемым для e-commerce search.
1. **Применить Cluster Scale Constraint Loss.** Category-consistent clusters не должны становиться слишком маленькими или перегруженными; иначе recall stage либо теряет tail, либо возвращает слишком много candidates.
1. **Применить Dispersion Loss.** Похожие категории удерживаются близко, но соседние products внутри одной broad category должны оставаться различимыми.
1. **Обучить generative retriever.** Query-document pairs с Exact/Substitute labels используются для обучения decoder генерировать DocID relevant products.
1. **Serving.** Query декодируется в beam of DocIDs; intermediate-node truncation и DR scores используются как latency mitigation перед финальным candidate recall.

```
for product in catalog:
    path = normalize_category_tree(product.category, max_depth=3)
    z = document_encoder(product.text, path)
    doc_id = rq_vae_quantize(z)
    store(product, path, doc_id)

for batch in query_document_pairs:
    pred_ids = generative_retriever(batch.queries, target_prefix=batch.doc_ids)
    loss_recall = cross_entropy(pred_ids, batch.doc_ids)
    loss_hccl = hierarchical_class_constraint(batch.doc_ids, batch.category_paths)
    loss_cscl = cluster_scale_constraint(batch.doc_ids)
    loss_disp = dispersion_constraint(batch.doc_ids, batch.document_vectors)
    loss = loss_recall + a * loss_hccl + b * loss_cscl + c * loss_disp
    update(document_encoder, rq_vae, generative_retriever)

for query in serving:
    beams = constrained_decode(query)
    beams = truncate_intermediate_nodes(beams)
    candidates = map_docids_to_products(beams)
```

## Эксперименты

Экспериментальная конкретика из paper:

- Dataset: Amazon ESCI multilingual e-commerce search; processed ESCI-us 386,392 docs, train 20,001 queries/341,460 pairs, test 5,800 queries/29,351 pairs; ESCI-es 131,935 docs; ESCI-jp 152,845 docs.
- Training uses query-document pairs with Exact/Substitute labels; category depth less than 3 removed, greater than 3 truncated to 3.
- Implementation: T5-base backbone; RQ-VAE trained 300 epochs; batch size 4096; AdamW; 8 A100 80GB GPUs.
- Online A/B: 10-day Meituan recall-stage test, with latency mitigation via intermediate-node truncation and DR scores.
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

Сильные стороны CAT-ID2 связаны с использованием e-commerce taxonomy как production prior, а не просто как дополнительной feature.

- **Category tree встроен в learning objective.** Identifier space наследует реальную структуру e-commerce каталога, поэтому DocID меньше похож на произвольный hash.
- **Проверка идет именно на recall stage.** Отказ от NDCG в пользу Recall@5-100 согласован с задачей генеративного retriever-а, который должен дать кандидатов downstream ranker'у.
- **HCCL и CSCL показывают важный trade-off.** В ablation каждый constraint по отдельности может вредить, а вместе они удерживают category consistency и scale balance.
- **Есть production-oriented online test.** Meituan A/B с latency mitigation полезнее, чем только ESCI offline numbers.
- **Метод хорошо подходит к e-commerce semantics.** Категории, бренды и Substitute/Exact labels ближе к пользовательскому intent, чем чистая text similarity.

## Ограничения

Ограничения и риски.

- **Метод зависит от качества category tree.** Если taxonomy устарела, слишком broad или плохо отражает search intent, constraints будут закреплять неправильную структуру.
- **Category constraint может переограничить retrieval.** Substitute products иногда лежат в соседних категориях; слишком жесткий HCCL может ухудшить cross-category recall.
- **Много весов losses.** HCCL, CSCL и dispersion требуют настройки; ablation показывает, что отдельные компоненты могут ухудшать результат.
- **Online details частично закрыты.** Meituan serving подтверждает практичность, но без полного pipeline сложно воспроизвести latency и truncation policy.
- **Category depth нормализация теряет информацию.** Truncation до depth 3 упрощает обучение, но может смешивать товары, различимые только на более глубоком уровне.
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
