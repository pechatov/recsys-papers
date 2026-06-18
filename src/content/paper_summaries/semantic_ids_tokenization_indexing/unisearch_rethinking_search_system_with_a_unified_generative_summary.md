---
title: "UniSearch: Rethinking Search System with a Unified Generative Architecture"
category: "semantic_ids_tokenization_indexing"
slug: "unisearch_rethinking_search_system_with_a_unified_generative_summary"
catalogId: "paper-unisearch_rethinking_search_system_with_a_unified_generative_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/unisearch_rethinking_search_system_with_a_unified_generative_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2509.06887"
---
**Авторы:** Jiahui Chen, Xiaoze Jiang, Zhibo Wang, Quanzhi Zhu, Feng Hu, Zhiheng Qin, Enyun Yu, Zhixin Zhai, Xiaobo Guo, Kefeng Wang, Jingshan Lv, Yupeng Huang, Han Li и др.

**Аффилиации:** Kuaishou Technology; Independent.

**Индустрия:** Kuaishou live and short-video search.

**Первичный источник:** arXiv source 2509.06887.

## Коротко

- UniSearch replaces cascaded search with unified generative architecture.
- Search Generator and Video Encoder are trained jointly.
- Search Preference Optimization aligns generation with online feedback.

## Контекст

- Traditional search has recall/pre-rank/rank cascade.
- Existing generative systems often tokenize items separately from generator training.
- Kuaishou search needs low-latency serving over live and billion-scale video pools.

## Проблема

- Two-stage tokenizer/generator creates objective mismatch.
- Invalid semantic ID paths are common without constrained decoding.
- Online user preference is not fully captured by offline CE.

## Метод/архитектура

- Search Generator uses BART-like encoder-decoder.
- Video Encoder uses BERT-like model over multimodal/text/visual/statistical features.
- VQ-VAE discretizes item embeddings into k-level SID.
- Trie service constrains decoding to valid paths; deployment uses TensorRT and KV-cache.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для UniSearch: Rethinking Search System with a Unified Generative Architecture качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Objective/алгоритм

- Residual contrastive learning aligns query and item residual components.
- Coarse-to-fine strategy structures learning difficulty.
- SPO uses reward system plus clicks/likes/downloads/watch time.
- KL-like stability prevents catastrophic deviation.

## Детальный алгоритм UniSearch

UniSearch заменяет cascaded search не одним decoder-ом, а совместным обучением Search Generator, Video Encoder, SID quantizer и preference optimization. Алгоритм важен именно как связка training и serving: valid-path trie, TensorRT/KV-cache и SPO входят в production viability.

1. **Закодировать videos.** BERT-like Video Encoder читает multimodal/text/visual/statistical features и строит item embeddings для live/video pools.
1. **Дискретизировать items через VQ-VAE.** Item embeddings переводятся в k-level semantic IDs. SID должен быть пригоден для trie decoding и обратного mapping в item.
1. **Обучить Search Generator.** BART-like encoder-decoder получает query/user context и генерирует SID tokens релевантных items.
1. **Добавить coarse-to-fine learning.** Модель учится сначала более грубым semantic levels, затем fine levels; это облегчает задачу, но требует контроля path collapse.
1. **Добавить residual contrastive learning.** RCL выравнивает query residual components и item residual components, чтобы SID levels были complementary, а не дублировали друг друга.
1. **Запустить constrained decoding.** Trie service ограничивает beam только валидными SID paths; в статье valid path rate растет с 51.3% до 99.8%.
1. **Оптимизировать Search Preference Optimization.** SPO использует reward system и feedback clicks/likes/downloads/watch time, но KL-like stability удерживает model от catastrophic deviation.
1. **Deploy с serving оптимизациями.** TensorRT, KV-cache и pool-specific serving нужны, чтобы generative architecture выдержала live и billion-scale short-video search.

```
video_embeddings = VideoEncoder(multimodal_video_features)
sid_map = VQVAE_quantize(video_embeddings, levels=k)

for batch in search_logs:
    query_context = encode_query_user_context(batch)
    target_sid = sid_map[clicked_or_relevant_item]
    loss_gen = autoregressive_sid_loss(SearchGenerator, query_context, target_sid)
    loss_cf = coarse_to_fine_loss(target_sid)
    loss_rcl = residual_contrastive_loss(query_context, video_embeddings, sid_map)
    update(loss_gen + lambda_cf * loss_cf + lambda_rcl * loss_rcl)

for spo_batch in online_feedback:
    generated = constrained_generate(SearchGenerator, spo_batch.query, trie=sid_trie)
    reward = reward_system(generated, clicks_likes_downloads_watchtime)
    update(SPO_loss(generated, reward) + KL_stability_penalty)

serve with trie + TensorRT + KV_cache
```

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для UniSearch: Rethinking Search System with a Unified Generative Architecture качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Эксперименты

- Live dataset: ~80M user search sessions from July 9-23 2025.
- Train July 9-22, test July 23.
- Live pool ~500K, short-video pool ~1B.
- Online 7-day A/B on 10% traffic.
- Live: +3.31% TPC, +0.202% CTR, CQR/PFC down.
- Video: +0.213% VPD, +0.993% PVD, -0.602% CQR, +0.830% LPC.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для UniSearch: Rethinking Search System with a Unified Generative Architecture качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Рисунки/таблицы

- Architecture comparison figure: cascaded, existing generative, UniSearch.
- Model figure: Search Generator + Video Encoder + online SPO.
- Trie figure shows valid path rate from 51.3% to 99.8%.
- Online analysis decomposes TPC gains: 65.06% long-tail queries, 58.73% new users.

## Ablation conclusions

- Plain model weakest.
- Coarse-to-fine improves recall/ranking but can cause path collapse.
- RCL improves token complementarity and MRR.
- Full UniSearch combines CF and RCL for best results.

## Сильные стороны

- **Сильная production evidence.** 7-day A/B on 10% traffic и отдельные live/video results делают работу ближе к реальному search replacement.
- **Joint tokenization/generation атакует objective mismatch.** Video Encoder, SID quantizer и Search Generator обучаются как связанная система.
- **Constrained serving раскрыт явно.** Trie, TensorRT и KV-cache показывают, как generative search доводится до production latency.
- **RCL улучшает complementarity SID levels.** Это важнее простой VQ-VAE квантизации, где levels могут дублировать signal.
- **SPO связывает offline generation с online feedback.** Clicks/likes/downloads/watch time используются после supervised phase.

## Ограничения

- **Point-wise beam generation ограничивает diversity.** Модель генерирует candidates, но не оптимизирует listwise slate как GFN-style подходы.
- **Система сложная.** Нужны Video Encoder, VQ-VAE, generator, trie service, SPO loop, TensorRT/KV-cache и online reward plumbing.
- **Private reward definitions ограничивают воспроизводимость.** TPC/CTR/CQR/PFC/VPD/PVD/LPC завязаны на Kuaishou product semantics.
- **Coarse-to-fine может вызвать path collapse.** Ablation прямо указывает, что CF полезен, но требует RCL/контроля.
- **Каталог и traffic нестабильны.** Live pool и 1B short-video pool требуют постоянного refresh SID/trie, иначе valid generation не гарантирует freshness.

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
- Специфично для этой статьи: Point-wise beam generation limits list diversity.
- Специфично для этой статьи: Complex online system required.
- Специфично для этой статьи: Private data and reward definitions limit reproducibility.

## Связь

- Related to FORGE and CRS in industrial SID search.
- Differs by end-to-end co-training of tokenizer and generator.

## Итог

- UniSearch shows how generative search can replace cascade in production.
- The crucial piece is joint optimization plus valid-path serving.

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
