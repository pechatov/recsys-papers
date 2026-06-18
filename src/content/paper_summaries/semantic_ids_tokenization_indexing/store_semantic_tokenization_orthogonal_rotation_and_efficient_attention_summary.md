---
title: "STORE: Semantic Tokenization, Orthogonal Rotation and Efficient Attention for Scaling Up Ranking Models"
category: "semantic_ids_tokenization_indexing"
slug: "store_semantic_tokenization_orthogonal_rotation_and_efficient_attention_summary"
catalogId: "paper-store_semantic_tokenization_orthogonal_rotation_and_efficient_attention_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/store_semantic_tokenization_orthogonal_rotation_and_efficient_attention_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2511.18805"
---
**Авторы:** Yi Xu, Chaofan Fan, Jinxin Hu, Yu Zhang, Xiaoyi Zeng, Jing Zhang.

**Аффилиации:** Alibaba Group; Wuhan University.

**Индустрия:** CTR/ranking models на e-commerce платформе.

**Первичный источник:** arXiv source 2511.18805.

## Коротко

- STORE применяет semantic tokenization к ranking models.
- Три компонента: Semantic Tokenization, Orthogonal Rotation Transformation, Efficient Attention.
- Цель - качество и throughput при high-cardinality heterogeneous sparse features.

## Контекст

- DLRM/ranking transformers страдают от огромных sparse embedding tables.
- Feature space динамический: новые товары, продавцы, категории, user IDs.
- Attention по всем tokens вызывает computation bottleneck и attention dispersion.

## Проблема

- Representation bottleneck: sparse embeddings low-rank и плохо масштабируются.
- Computational bottleneck: unified features дают взрыв token count.
- Interaction-Collapse ограничивает feature interaction.

## Метод/архитектура

- Semantic tokenizer decomposes high-cardinality IDs into compact stable tokens.
- Orthogonal rotation rotates low-cardinality static feature subspace.
- Efficient attention filters low-contributing tokens.
- OPMQ setup uses K=32 and codebook size 300 in online deployment.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для STORE: Semantic Tokenization, Orthogonal Rotation and Efficient Attention for Scaling Up Ranking Models качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Objective/алгоритм

- Ranking objective остается CTR/prediction loss.
- Semantic tokenization меняет feature representation до ranker.
- Rotation improves decomposition and interaction geometry.
- Attention sparsity 1/2 in online setup improves serving speed.

### Пошаговая схема STORE

STORE не является generative retrieval методом: semantic tokenization здесь используется как способ масштабировать ranking model на high-cardinality sparse features. Алгоритм меняет representation перед ranker'ом, а не output space.

1. **Разделить sparse features по типу.** High-cardinality динамические признаки (items, sellers, ads) идут в semantic tokenization; low-cardinality/static признаки обрабатываются отдельно.
1. **Semantic tokenization для high-cardinality IDs.** OPMQ decomposes each ID embedding into compact semantic tokens, уменьшая зависимость от огромных sparse tables и улучшая sharing между похожими entities.
1. **Orthogonal Rotation Transformation.** Static/low-cardinality feature subspace ортогонально поворачивается, чтобы разные компоненты признаков меньше коллапсировали в одни и те же interaction directions.
1. **Собрать unified token sequence.** Ranker получает semantic tokens, rotated static features и остальные dense/context features.
1. **Efficient Attention.** Attention module оценивает contribution tokens и фильтрует низкополезные; online setup использует sparsity 1/2 для throughput.
1. **Обучить CTR/ranking loss.** Основной objective остается prediction loss, поэтому tokenization/rotation полезны только если улучшают downstream AUC/CTR при приемлемой latency.

```
for feature_batch in training_logs:
    high_card_tokens = []
    for feature_id in high_cardinality_features(feature_batch):
        semantic_tokens = OPMQ_tokenize(feature_id, K=32, codebook_size=300)
        high_card_tokens.extend(semantic_tokens)

    rotated_static = orthogonal_rotation(low_cardinality_features(feature_batch))
    tokens = concat(high_card_tokens, rotated_static, dense_context_features)

    token_scores = attention_importance(tokens)
    kept_tokens = keep_top_fraction(tokens, token_scores, fraction=0.5)
    prediction = ranking_transformer(kept_tokens)
    loss = ctr_or_cvr_loss(prediction, labels)
    update(tokenizer, rotation, attention, ranker)
```

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для STORE: Semantic Tokenization, Orthogonal Rotation and Efficient Attention for Scaling Up Ranking Models качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Эксперименты

- Offline experiments and online A/B tests.
- Online 15-day A/B on large-scale e-commerce platform.
- Reported online CTR +2.71%, AUC +1.195%, training throughput 1.84x.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для STORE: Semantic Tokenization, Orthogonal Rotation and Efficient Attention for Scaling Up Ranking Models качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Рисунки/таблицы

- Architecture diagrams should show three modules.
- Bottleneck figures explain One-Epoch/Interaction-Collapse.
- Online table reports CTR/AUC/throughput.

## Ablation conclusions

- Without semantic tokenization high-cardinality sparsity remains.
- Without rotation static features interact worse.
- Without efficient attention throughput/latency degrades.

## Сильные стороны

- **Расширяет semantic tokenization на ranking.** STORE показывает, что SID-like decomposition полезна не только как target identifier, но и как feature representation для DLRM/ranking transformers.
- **Качество и throughput оптимизируются вместе.** Efficient attention не является косметикой: online setup сообщает training throughput 1.84x при росте CTR/AUC.
- **OPMQ параметры раскрыты для deployment.** K=32 и codebook size 300 дают более конкретный ориентир, чем абстрактное "semantic tokenizer".
- **Rotation адресует interaction geometry.** Работа не сводится к сжатию embedding table; она также пытается уменьшить Interaction-Collapse между heterogeneous features.
- **Есть production A/B.** 15-дневный online test с CTR +2.71% делает выводы сильнее чистого offline ranking benchmark.

## Ограничения

- **Production details limited.** Без полного описания feature schema и serving stack трудно понять, какие gains идут от STORE, а какие от platform-specific engineering.
- **Token filtering может выкинуть редкие важные сигналы.** Low-contribution по attention не всегда означает низкую бизнес-ценность, особенно для fraud, niche interest и cold-start features.
- **Rotation/tokenization требуют schema-specific tuning.** Другой набор sparse fields может потребовать других codebook sizes, rotation dimensions и sparsity threshold.
- **Semantic tokens могут устаревать.** Dynamic e-commerce features меняются быстрее, чем static codebook; нужен refresh и drift monitoring.
- **Ranking objective не объясняет token semantics.** Улучшение AUC не гарантирует, что learned tokens интерпретируемы или переносимы в retrieval/generation.

## Как реализовать/проверять

1. Зафиксировать версию каталога, train/eval split и mapping item/document -> identifier; без этого невозможно понять, улучшает ли метод саму модель или только меняет пространство кандидатов.
1. Считать отдельно качество tokenization и качество generator/ranker: collision rate, utilization, Gini, valid-path rate, Recall/HR/NDCG/MRR и latency должны лежать в одном отчете.
1. Делать ablation не только по среднему качеству, но и по head/torso/tail, cold-start, new users/new items, long-history и категориям с похожими объектами.
1. Проверять деградацию при обновлении каталога: semantic IDs могут устаревать, а генератор может помнить старые пути.
1. Сохранять обратный индекс identifier -> item/document и явно логировать случаи many-to-one collisions.
1. Для генеративного вывода использовать constrained decoding или post-validation, иначе invalid identifiers будут маскироваться в offline метриках.
1. В production считать стоимость не только модели, но и перестроения codebook, trie/index, feature pipeline и мониторинга drift.
1. В отчетах отделять retrieval-stage gains от downstream ranking/business metrics, потому что рост HR@K не всегда дает CTR/CVR uplift.

## Failure modes и мониторинг

- Identifier collapse: малая часть кодов получает большую долю объектов, а генератор начинает переиспользовать популярные пути.
- Semantic-collaborative mismatch: похожие по тексту объекты имеют разные user intents, или наоборот, совместно потребляемые объекты текстово далеки.
- Exposure bias autoregressive generation: ошибка раннего токена полностью меняет candidate path.
- Popularity bias: модель учится генерировать frequent IDs и выглядит хорошо на aggregate, но теряет novelty и tail coverage.
- Evaluation leakage: если ID/tokenizer обучался на будущем каталоге или post-training видел target-like сигналы, offline gain завышен.
- Serving mismatch: offline beam шире и медленнее production beam, поэтому качество не переносится напрямую.
- Специфично для этой статьи: Production details limited.
- Специфично для этой статьи: Token filtering may hurt rare but important features.
- Специфично для этой статьи: Rotation/tokenization require feature-schema-specific tuning.

## Связь

- Related to Better Generalization with Semantic IDs and PLUM.
- Complements FORGE by focusing ranking stage.

## Итог

- Semantic IDs are useful beyond generative decoding.
- Ranking models can use tokenization to scale feature interactions.

## Минимальный план воспроизведения

<div class="table-scroll">
<table>
<thead>
<tr><th>Шаг</th><th>Что сделать</th><th>Что считать успехом</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>Собрать исходные item/document features и interaction/query logs в той же временной схеме, что у статьи.</td><td>Нет leakage, воспроизводимы splits и отрицательные примеры.</td></tr>
<tr><td>2</td><td>Построить identifiers/token representations и сохранить mapping plus collision report.</td><td>Utilization и collision metrics понятны до обучения generator.</td></tr>
<tr><td>3</td><td>Обучить baseline без нового компонента и полный метод с тем же budget.</td><td>Сравнение честное по compute, beam, candidates и preprocessing.</td></tr>
<tr><td>4</td><td>Запустить ablations по каждому заявленному компоненту.</td><td>Каждый компонент дает объяснимый вклад или честно признается избыточным.</td></tr>
<tr><td>5</td><td>Проверить production-like constraints: latency, invalid IDs, refresh, monitoring.</td><td>Offline gain не исчезает при реальных ограничениях serving.</td></tr>
</tbody>
</table>
</div>

Примечание: если в источнике не раскрыты приватные production details, summary явно помечает такие ограничения и не выдумывает закрытые числа.
