---
title: "FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets"
category: "semantic_ids_tokenization_indexing"
slug: "forge_forming_semantic_identifiers_for_generative_retrieval_in_summary"
catalogId: "paper-forge_forming_semantic_identifiers_for_generative_retrieval_in_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/forge_forming_semantic_identifiers_for_generative_retrieval_in_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2509.20904"
---
**Авторы:** Kairui Fu, Tao Zhang, Shuwen Xiao, Ziyang Wang, Xinming Zhang, Chenchi Zhang, Yuliang Yan, Junjun Zheng, Yu Li, Zhihong Chen, Jian Wu, Xiangheng Kong, Shengyu Zhang, Kun Kuang, Yuning Jiang, Bo Zheng.

**Аффилиации:** Zhejiang University; Alibaba Group / Taobao.

**Индустрия:** industrial e-commerce retrieval/search с multimodal item features и real user behavior logs.

**Первичный источник:** arXiv source 2509.20904.

## Коротко

- FORGE является industrial benchmark и набором практик для semantic identifiers в generative retrieval.
- Работа изучает SID quality, codebook utilization, collisions, multimodal features, collaborative i2i relations и online adaptation.
- Главная ценность - масштаб и production-oriented анализ, а не только новый loss.

## Контекст

- Generative retrieval декодирует SID целевого товара вместо ANN search по dense vector.
- На маленьких Amazon/MovieLens datasets многие проблемы не видны: industrial catalog меняется ежедневно, а user sequences длиннее.
- FORGE строит данные вокруг recommendation и search задач с реальной платформы и 40 млн sampled items в день.

## Проблема

- RQ-VAE может давать semantic IDs с плохой фактической utilization.
- Collision handling критичен: один SID может соответствовать многим товарам, что разрушает точность генерации.
- Feature fidelity не всегда коррелирует с retrieval effectiveness, поэтому нужны отдельные SID-quality метрики.

## Метод/архитектура

- Базовый tokenizer использует multimodal item features, включая image/text fusion через CN-CLIP-like features.
- RQ-VAE строит multi-level residual codewords; основной practical setup исследует трехуровневые SID.
- Collaborative signals добавляются через i2i relationships и item-side information: seller, price, category и richer co-occurrence.
- Post-processing контролирует collisions через KNN-based ограничения, например меньше 25, 10 или 5 items per SID.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Objective/алгоритм

- SID generation оптимизируется reconstruction/quantization loss и alignment с collaborative relations.
- Generative retrieval objective - autoregressive prediction target SID по sequence of historical behavior SIDs и, для search, query keywords.
- Offline continued pretraining на последних 7 днях interactions помогает LLM адаптироваться к обновленным SID.
- Оценка SID включает HitRate, Embedding HitRate, Gini coefficient, utilization, style/homogeneous consistency и production metrics вроде transaction count.

### Пошаговая схема FORGE pipeline

1. **Собрать multimodal item features.** Для товаров строятся image/text fusion embeddings, плюс item-side signals: seller, price, category и другие attributes.
1. **Добавить collaborative relations.** Из user behavior logs извлекаются i2i/co-occurrence связи, чтобы tokenizer различал не только визуально похожие, но и совместно потребляемые товары.
1. **Обучить SID tokenizer.** RQ-VAE/RQ-style tokenizer строит multi-level residual codes; FORGE сравнивает layouts 2/3/4 levels и разные capacities.
1. **Проверить raw SID distribution.** До обучения generator считаются utilization, Gini, collision groups, style/homogeneous consistency и Embedding HitRate; плохой SID не стоит чинить только LLM'ом.
1. **Применить collision post-processing.** KNN-based cap ограничивает число items на SID, например KNN-25/10/5; no-collision и intentional merge используются как ablation extremes.
1. **Обучить generative retriever.** T5/Qwen получает sequence of historical behavior SIDs, а для search еще query keywords, и autoregressively генерирует target SID.
1. **Адаптироваться к свежему каталогу.** Continued pretraining на последних 7 днях interactions синхронизирует LLM с обновленным SID mapping и меняющейся платформенной distribution.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Эксперименты

- Данные разделены на S1/S2/S3: S1 четыре дня warm-up, S2/S3 по три дня, после каждой стадии 100K samples для evaluation.
- Generative models: T5-Base и Qwen2.5-0.5B; history length до 100, max source length 1024, target length 256.
- T5-Base обучается 4 epochs, total batch 1280 на 16 PPU-ZW810E GPUs; learning rate 2e-4.
- Ablations показывают связь collision avoidance и HR@20/100/500/1000.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Рисунки/таблицы

- Dataset tables описывают sequence data, item info и task formats.
- Ablation tables сравнивают no-collision, base, KNN-10, KNN-5 и merge.
- Codebook structure tables проверяют 2-level, 3-level, 4-level и неодинаковые sizes вроде 1024_4096_32768.
- Quantization comparison показывает Random, Multiple-VQ, RQ-Kmeans, RQ-VAE base и optimized variants.

## Ablation conclusions

- No post-processing collision strategy снижает retrieval.
- Intentional merge, ухудшающий fairness, резко снижает performance и подтверждает роль utilization.
- 2-level SIDs слишком грубые, 4-level дороже и дают малый выигрыш; 3-level - лучший compromise.
- Random SID дает сильное падение HR@1000, Multiple-VQ страдает от decoupled codewords.

## Сильные стороны

- Показывает реальные operational knobs для SID: fairness, utilization, collision cap, layout, adaptation.
- Не полагается на одну offline метрику и явно показывает, какие метрики коррелируют с retrieval.
- Описывает training/inference formats, что делает работу полезной для воспроизведения pipeline.

## Слабые стороны и ограничения

- Часть данных анонимизирована и защищена privacy constraints; полная воспроизводимость ограничена.
- Выводы завязаны на e-commerce с частыми homogeneous/style duplicates.
- Инфраструктурная сложность высока: tokenizer, LLM, online updates и SID mapping должны обновляться согласованно.

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
- Специфично для этой статьи: Часть данных анонимизирована и защищена privacy constraints; полная воспроизводимость ограничена.
- Специфично для этой статьи: Выводы завязаны на e-commerce с частыми homogeneous/style duplicates.
- Специфично для этой статьи: Инфраструктурная сложность высока: tokenizer, LLM, online updates и SID mapping должны обновляться согласованно.

## Связь

- FORGE дополняет LETTER и CoST: те предлагают objectives, а FORGE показывает industrial evaluation.
- Связан с UniSearch/PLUM через practical SID serving.
- Работа полезна как чеклист для любой промышленной системы semantic ID retrieval.

## Итог

- FORGE говорит: SID качество - это distribution engineering, а не только reconstruction.
- Лучший practical recipe включает collision control, collaborative signals, utilization monitoring и continued pretraining при drift.

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
