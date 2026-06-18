---
title: "Multimodal Generative Retrieval Model with Staged Pretraining for Food Delivery on Meituan"
category: "semantic_ids_tokenization_indexing"
slug: "multimodal_generative_retrieval_model_with_staged_pretraining_for_summary"
catalogId: "paper-multimodal_generative_retrieval_model_with_staged_pretraining_for_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/multimodal_generative_retrieval_model_with_staged_pretraining_for_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2602.06654"
---
> **Авторы:** Boyu Chen, Tai Guo, Weiyu Cui, Yuqing Li, Xingxing Wang, Chuan Shi, Cheng Yang.
>
> **Аффилиации:** Meituan; BUPT.

## Коротко

SMGR - multimodal generative retrieval для food delivery search/recommendation. Статья утверждает, что совместное обучение всех модальностей нестабильно, поэтому предлагает staged pretraining: сначала получить качественные multimodal embeddings, затем превратить их в semantic IDs и дообучить модель под downstream retrieval.

## Контекст

В food delivery item содержит название блюда/ресторана, изображение, категорию, локальный geo context и query-click logs. Для online serving большие dense embeddings дороги, а semantic IDs позволяют сжать multimodal информацию в дискретные коды, пригодные для generative retrieval.

## Проблема

Наивная joint optimization разных objectives может конфликтовать: image/text/query losses тянут representation в разные стороны. Авторы показывают loss trends и сравнение с random image embeddings, чтобы обосновать staged training: порядок обучения модальностей важен.

## Метод / архитектура

Pipeline состоит из трех компонентов. Staged Pretraining строит multimodal item embeddings в заданном порядке objectives. Semantic IDs Generation применяет RQ-VAE к high-dimensional embeddings, получая дискретные SIDs. Semantic IDs Utilization fine-tune'ит модель на downstream tasks, используя modality-specific или fused SIDs.

## Objective / алгоритм

Pretraining оптимизирует несколько задач для item/query/image/text alignment. RQ-VAE переводит continuous embedding в codebook indices. Downstream adaptation использует prompt/task templates: модель учится генерировать/использовать SIDs для query-to-item retrieval в локальном geo space. Важно, что SIDs не просто compress, а становятся интерфейсом между multimodal representation и retrieval.

### Подробная схема алгоритма SMGR

1. **Собрать multimodal food item data.** Для блюда/ресторана используются title/text, image, category, query-click logs и локальный `geo_hash` .
1. **Обучить staged encoder.** Objectives для image-text, query-item и downstream retrieval включаются в выбранном порядке, а не одновременно; порядок проверяется через Order variants.
1. **Проверить, что модальности не шумовые.** Random image embedding baseline нужен, чтобы доказать, что gain от image идет из visual signal, а не из дополнительной емкости.
1. **Сгенерировать semantic IDs.** High-dimensional multimodal embeddings квантуются RQ-VAE: каждый residual level добавляет следующий code token.
1. **Выбрать modality-specific SID variant.** Сравниваются item-only, image-text, item-image и item-text SIDs, чтобы понять, какие сочетания реально помогают food delivery retrieval.
1. **Fine-tune downstream model.** Prompt/task templates учат модель использовать SID для query-to-item retrieval внутри локального geo space.
1. **Ограничить candidates по geography.** Retrieval считается только среди item'ов с тем же `geo_hash` , потому что глобальный каталог физически недоступен пользователю.
1. **Оценить offline и online.** Offline FAISS Recall/NDCG по query type и geo bucket дополняются week-long online A/B и case study.

```
raw_modalities = {text, image, category, query_clicks, geo_hash}

encoder = staged_pretrain(raw_modalities, ordered_objectives)
embeddings = encoder(items)
sid = RQ_VAE(embeddings)

for variant in {item_only, image_text, item_image, item_text, fused}:
    train downstream_retriever(query_prompt, sid_variant)
    candidates = restrict_by_geo_hash(catalog, query.geo_hash)
    evaluate Recall/NDCG with FAISS over candidates

deploy best staged+SID version to online A/B
```

## Эксперименты и метрики

Оценка строится на click logs food delivery. FAISS используется для ускорения recall computation, а matching ограничивается тем же `geo_hash`, что отражает реальное поведение пользователей. Метрики: Recall@5/10/20 и NDCG@5/10/20, 10 повторов, paired t-test. Main results сравнивают query types/datasets; отдельные RQ проверяют training order, modality utilization, downstream task adaptation и week-long online A/B.

## Рисунки / таблицы

Figure с loss trends показывает конфликт joint objectives. Главная схема фиксирует три этапа SMGR. Таблица main results дает retrieval performance по query types. Ablation tables сравнивают Joint, Random и разные Order variants; отдельные таблицы проверяют item-only, image-text, item-image, item-text SIDs и fine-tuning tasks. Case study с query "Peking Duck" показывает качественную разницу top-2 выдачи.

## Сильные стороны

- **Промышленный сценарий с реальными constraints.** `geo_hash` ограничение и week-long online A/B делают постановку ближе к food delivery serving, чем обычный global retrieval benchmark.
- **Training order является проверяемым механизмом.** Статья не просто добавляет image/text, а показывает, что joint training конфликтует и staged order влияет на Recall/NDCG.
- **Мультимодальность подтверждается ablations.** Item-only, image-text, item-image, item-text и random image baseline отделяют полезный visual/text signal от шума.
- **SID используется как serving interface.** RQ-VAE делает dense multimodal embeddings компактными codes, пригодными для generative retrieval и хранения.

## Ограничения

- **Внутренние данные и production details.** Food delivery logs, query mix и online A/B детали Meituan нельзя полноценно воспроизвести на public datasets.
- **Staged pipeline дороже сопровождать.** Нужно версионировать порядок stages, checkpoints encoder'а, RQ-VAE tokenizer и downstream fine-tuning.
- **Geo constraints доменно-специфичны.** Успех внутри `geo_hash` не гарантирует перенос на глобальные каталоги или domains без жесткой локальности.
- **RQ-VAE collisions между похожими блюдами остаются риском.** В food delivery похожие фото/названия из разных ресторанов могут получить близкие SID, но иметь разную цену, доступность и качество.
- **Модальности быстро устаревают.** Фото, меню, акции и сезонность требуют refresh strategy; иначе staged embeddings станут stale.

## Как реализовать / проверять

Практический план: зафиксировать query/item logs, обучить unimodal baselines, затем staged multimodal encoder, затем RQ-VAE tokenizer. Проверять нужно Recall/NDCG по query type и geo buckets, SID collision/uniqueness, влияние каждой модальности, latency ANN/retrieval и online order/GMV metrics. Для контроля обязательно сравнить с random image embedding и joint training.

## Связь с соседними работами

SMGR близок к OCR-text paper по вопросу "какую модальность квантизовать", но вместо text-as-vision изучает staged multimodal pretraining. С AdaSID и Snapchat его связывает industrial SID deployment, а с GLASS - использование semantic search/serving constraints.

## Pretraining objectives и конфликт joint optimization

Авторы показывают, что разные objectives в multimodal retrieval не всегда совместимы.

Image-text alignment, query-item matching и downstream retrieval могут давать разные gradients.

При joint training один objective может улучшаться за счет другого.

Figure с loss trends показывает такую нестабильность.

Staged pretraining задает порядок, в котором representation сначала получает базовую multimodal структуру, а затем адаптируется к retrieval.

В таблицах Order variants показывают, что порядок стадий влияет на Recall/NDCG.

## RQ-VAE details

High-dimensional multimodal embeddings переводятся в SIDs через RQ-VAE.

Residual quantization последовательно квантует остаток после предыдущего codebook.

Так item получает compositional code, который компактнее dense vector.

Appendix содержит детали RQ-VAE и hyperparameters.

Главный production мотив: дискретные SIDs легче хранить, передавать и использовать в generative retrieval, чем большие embeddings.

## Semantic IDs utilization

После tokenization модель fine-tune'ится под downstream task.

Авторы сравнивают item-only, image-text, item-image и item-text variants.

Это отвечает на вопрос, какие modality-specific SIDs реально нужны.

Если image modality дает gain только после staged pretraining, значит простой image feature не является гарантией улучшения.

Fine-tuning task variants показывают, что downstream adaptation не менее важна, чем pretraining.

## Evaluation protocol details

Кандидаты и queries embed'ятся отдельно.

FAISS используется для ускорения recall computation.

Для данного `geo_hash` query matching ограничивается candidates из того же `geo_hash`.

Это принципиально для food delivery: пользователь физически не может заказать любой item из глобального каталога.

Каждый эксперимент повторяется 10 раз.

Significance проверяется paired t-test с `p < 0.05`.

## Online A/B и case study

Week-long online A/B проверяет, переходит ли offline retrieval gain в product metrics.

Case study с query "Peking Duck" иллюстрирует semantic retrieval: baseline Joint-Que2search и SMGR возвращают разные top-2 items.

Такие examples важны для food domain, где визуально похожие блюда могут отличаться по региону, ресторану и цене.

Но case study нельзя считать доказательством; оно служит sanity check к quantitative tables.

## Failure modes

Первый failure mode - stale images или плохое качество food photos.

Второй - geo constraint слишком жесткий и скрывает релевантные substitutes.

Третий - staged order подобран под Meituan logs и не переносится напрямую.

Четвертый - RQ-VAE collisions между похожими dishes из разных ресторанов.

Пятый - online serving drift из-за сезонности и локальных акций.

## Implementation notes

1. Разделить pipelines для embedding pretraining, SID generation и downstream fine-tuning.
1. Версионировать staged order и checkpoints каждого этапа.
1. Проверить random image embedding baseline, чтобы доказать пользу визуальной модальности.
1. Считать metrics отдельно по query type и geo buckets.
1. Добавить offline FAISS recall и online business metrics.
1. Контролировать SID collisions внутри популярных food categories.

## Итог

Главный урок Meituan: качество SID начинается до quantization. Если multimodal embedding обучен в неправильном порядке, дискретные коды будут сжимать конфликтное пространство; staged pretraining делает SIDs пригодными для реального food delivery retrieval.

## Источники

- [arXiv:2602.06654](https://arxiv.org/abs/2602.06654) , PDF/source.
