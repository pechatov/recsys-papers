---
title: "When Text-as-Vision Meets Semantic IDs in Generative Recommendation: An Empirical Study"
category: "semantic_ids_tokenization_indexing"
slug: "when_text_as_vision_meets_semantic_ids_in_summary"
catalogId: "paper-when_text_as_vision_meets_semantic_ids_in_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/when_text_as_vision_meets_semantic_ids_in_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2601.14697"
---
> **Авторы:** Shutong Qiao, Wei Yuan, Tong Chen, Xiangyu Zhao, Quoc Viet Hung Nguyen, Hongzhi Yin.
>
> **Аффилиации:** The University of Queensland; City University of Hong Kong; Griffith University.

## Коротко

Статья проверяет необычную гипотезу для generative recommendation: если item text сначала отрендерить как изображение и пропустить через OCR/vision encoder, то получившееся *OCR-text* representation может лучше подходить для построения semantic IDs, чем стандартный text embedding. Это не новый retriever, а эмпирическое исследование входных модальностей для TIGER/LETTER-подобного pipeline.

## Контекст

Semantic ID-based GR обычно строит item-коды из текстовых описаний, изображений или их fusion. Текстовые encoder'ы хорошо читают свободный язык, но товарные описания часто являются списками атрибутов, размеров, брендов и технических признаков. Авторы показывают, что визуальное представление текста иногда образует более гладкую геометрию и лучше согласуется с image embeddings.

## Проблема

Вопрос статьи: какая модальность должна быть источником semantic IDs, если item имеет и текст, и изображение? Наивное предположение "текст надо кодировать text encoder'ом" оказывается не всегда оптимальным. Особенно это заметно на категориях, где описание похоже на табличный набор атрибутов, а не на естественный абзац.

## Метод / архитектура

Pipeline сохраняет стандартную схему: item content encoding, multimodal fusion, token generation через RQ-VAE/RQ-KMeans и autoregressive generation semantic IDs. Отличие в том, что текстовая часть заменяется OCR-text: описание рендерится в картинку, затем кодируется OCR/vision-language encoder'ом. Авторы сравнивают unimodal `T` vs `O`, early fusion `T+I` vs `O+I`, а также несколько late-fusion стратегий.

## Objective / алгоритм

Цель не меняет loss генеративного recommender'а: модель по истории semantic IDs генерирует next-item SID, а качество измеряется Recall/NDCG. Алгоритмический вклад находится до обучения GR: подобрать representation space, в котором tokenization дает более полезную дискретизацию. OCR-text выступает как drop-in replacement для text embedding.

## Детальный алгоритм Text-as-Vision SID

Алгоритм полезно читать как controlled replacement experiment: весь TIGER/LETTER-like pipeline фиксируется, а меняется только то, как item text превращается в embedding перед semantic ID learning.

1. **Подготовить стандартный text baseline.** Для каждого item получить обычный text embedding `T` из title/description/category fields.
1. **Отрендерить item text как изображение.** Описание превращается в картинку с фиксированным template: ширина, padding, шрифт, перенос строк и resolution должны быть одинаковыми.
1. **Получить OCR-text embedding.** Rendered image проходит через OCR/vision-language encoder; результат обозначается `O` .
1. **Собрать modality variants.** Unimodal: `T` vs `O` . Early fusion: `T+I` vs `O+I` , где `I` - image embedding. Late fusion: несколько стратегий объединения embeddings/SIDs/model inputs.
1. **Построить Semantic IDs одинаковым tokenizer-ом.** Для каждого representation variant применяется тот же RQ-VAE/RQ-KMeans pipeline, чтобы сравнение не смешивало encoder и tokenizer changes.
1. **Обучить downstream GR.** TIGER или LETTER обучается с одинаковым leave-one-out split и beam search 20.
1. **Сравнить не только Recall/NDCG.** Нужно смотреть geometry embeddings, codebook utilization, collision rate, prefix distribution и preprocessing cost.
1. **Повторить по категориям.** Scientific/Instruments выигрывают больше, Arts слабее: эффект зависит от того, является ли text attribute-centric.

```
for item in catalog:
    T[item] = text_encoder(item.text_fields)
    rendered = render_text_as_image(item.text_fields, template=fixed_template)
    O[item] = OCR_vision_encoder(rendered)
    I[item] = image_encoder(item.image)

variants = {
    "T": T,
    "O": O,
    "T+I": fuse(T, I),
    "O+I": fuse(O, I),
}

for name, representation in variants.items():
    sid_map = train_same_tokenizer(representation)
    model = train_TIGER_or_LETTER(logs_as_sids(logs, sid_map))
    evaluate(model, beam_size=20, metrics=["Recall", "NDCG", "collision", "utilization"])
```

## Эксперименты и метрики

Используются Amazon-подобные домены Scientific, Luxury, Instruments и Arts. Метрики: Recall@5/10/20 и NDCG@5/10/20, leave-one-out split, beam search 20. В unimodal TIGER OCR-text превосходит обычный text на большинстве наборов. В multimodal early fusion на Scientific замена `T+I` на `O+I` дает рост TIGER Recall@5 с 0.0531 до 0.0600 и NDCG@5 с 0.0378 до 0.0426. В late fusion улучшения наиболее стабильны для стратегий B/C; на Arts эффект слабее или почти нейтрален.

## Рисунки / таблицы

Figure 1 визуализирует геометрию embeddings: OCR-text ближе к image manifold, обычный text более дисперсен и анизотропен. Figure 3 фиксирует общий Semantic ID GR pipeline. Таблицы и графики RQ1-RQ3 показывают, что эффект зависит от датасета, fusion-стратегии и OCR backbone; robustness section показывает, что разные OCR encoders дают близкие Recall@10 на Instruments.

## Сильные стороны

- **Хорошо изолирует источник улучшения.** Меняется representation для tokenization, а не весь recommender, поэтому вывод про OCR-text чище.
- **Проверяет несколько fusion regimes.** Unimodal, early fusion и late fusion показывают, что эффект не привязан к одной архитектуре.
- **Дает dataset-specific интерпретацию.** OCR-text особенно полезен для attribute-centric descriptions, а на Arts эффект слабее.
- **Проверяет robustness OCR backbone/resolution.** Это снижает риск, что результат является артефактом одного encoder-а.
- **Практически применимо как offline preprocessing.** OCR-text embeddings можно кэшировать, не меняя online GR architecture.

## Ограничения

- **Preprocessing cost выше.** Нужно рендерить text images и запускать OCR/vision encoder для всего каталога.
- **Эффект зависит от структуры текста.** Для богатого естественного описания обычный text encoder может быть лучше.
- **Rendering template становится гиперпараметром.** Font, width, line wrap, resolution и cropping могут менять embedding quality.
- **Language/domain mismatch OCR encoder-а опасен.** Неподдержанные языки, шрифты или специальные symbols делают OCR-text noisy.
- **Работа эмпирическая.** Она показывает условия выигрыша, но не доказывает универсальное превосходство OCR-text.

## Как реализовать / проверять

Нужно зафиксировать baseline tokenizer и downstream TIGER/LETTER, затем заменить только text encoder на OCR-text encoder. Проверять стоит: качество semantic ID, collision rate, Recall/NDCG, latency offline preprocessing и устойчивость к разрешению рендера. Для production важен cache OCR embeddings и fallback на обычный text embedding.

## Связь с соседними работами

Работа дополняет CoST, AdaSID и SID-Coord: те улучшают objective/использование SID, а здесь исследуется, какие исходные embeddings стоит квантизовать. Для Meituan/Snapchat-подобных multimodal систем вывод практичен: modality choice может быть не менее важен, чем форма codebook.

## Детали постановки Semantic ID learning

Авторы описывают общий pipeline как три независимых выбора.

Первый выбор - item content encoder.

Здесь сравниваются обычный text encoder, image encoder и OCR-text encoder.

Второй выбор - способ fusion между модальностями.

В статье рассматриваются early fusion и несколько late-fusion вариантов.

Третий выбор - token generation, то есть перевод continuous item representation в semantic IDs.

Эта декомпозиция полезна практически: если качество упало, можно отдельно проверить encoder, fusion и tokenizer.

В unimodal setup OCR-text заменяет standard text embeddings без изменения остального retrieval pipeline.

В multimodal setup OCR-text используется в парах `O+I`, где `O` - OCR-text representation, а `I` - item image embedding.

Сравнение с `T+I` показывает, дает ли визуальное кодирование текста дополнительное выравнивание с изображениями.

## Почему OCR-text может работать лучше

Интуиция статьи не в том, что OCR понимает текст лучше языковой модели.

Скорее, OCR/vision encoder видит текст как визуальный layout с повторяющимися атрибутами, короткими фразами, числами и категориальными признаками.

Для e-commerce descriptions это часто ближе к реальному item signal, чем к свободному языку.

Обычный text encoder может создавать anisotropic space, где похожие attribute lists оказываются разнесены сильнее, чем нужно для quantization.

OCR-text embeddings, по наблюдению авторов, образуют более гладкие manifolds и лучше согласуются с image embeddings.

Это важно именно для semantic ID: quantizer дискретизирует геометрию embedding space, поэтому ошибки геометрии превращаются в ошибки code assignment.

## Экспериментальные варианты fusion

Early fusion объединяет модальности до token generation.

Это простой сценарий: один fused embedding квантуется в один SID.

Late fusion сохраняет отдельные representations и объединяет их на уровне semantic IDs или model input.

Авторы проверяют несколько late-fusion стратегий, потому что разные домены требуют разного баланса image/text.

На Scientific и Instruments OCR-text дает наиболее заметные gains.

На Luxury gains устойчивые, но меньше.

На Arts эффект минимальный: descriptions более free-form, поэтому text encoder уже достаточно хорош.

## Robustness: OCR backbone и resolution

В robustness section проверяются разные OCR encoders.

На Instruments результаты близки: Recall@10 у разных OCR encoders находится в узком диапазоне около 0.0961-0.0970.

Это важный практический вывод: эффект не выглядит артефактом одного конкретного OCR backbone.

Также проверяется rendering resolution 256 vs 1024.

Resolution влияет на качество распознавания мелких атрибутов, но слишком высокий resolution увеличивает preprocessing cost.

Для production это означает, что OCR-text можно кэшировать offline, а resolution выбирать по latency/storage budget.

## Failure modes

Первый failure mode - описание не attribute-centric.

Если текст похож на нормальный абзац, text encoder может быть сильнее OCR-text.

Второй failure mode - плохой rendering template.

Если текст обрезается, переносится хаотично или содержит слишком мелкий шрифт, OCR-text embedding будет шумным.

Третий failure mode - language/domain mismatch OCR encoder'а.

Четвертый - late fusion может переусилить image-like signals и потерять редкие textual attributes.

## Практический чеклист воспроизведения

1. Зафиксировать downstream backbone: TIGER или LETTER.
1. Сгенерировать стандартные text embeddings для всех items.
1. Сгенерировать изображения текста одинаковым template: font, width, padding, line wrap.
1. Прогнать OCR/vision encoder и сохранить OCR-text embeddings.
1. Построить SIDs отдельно для `T` , `O` , `T+I` и `O+I` .
1. Обучить downstream GR с одинаковым split и beam size.
1. Сравнить Recall/NDCG, collision rate и distribution of SID prefixes.
1. Повторить на разных categories, потому что эффект зависит от структуры descriptions.

## Что проверять сверх метрик Recall/NDCG

Нужно смотреть intra-cluster semantic consistency: похожи ли items внутри одного prefix bucket.

Нужно смотреть cross-modal alignment: насколько OCR-text embedding ближе к image embedding для того же item, чем к соседним items.

Нужно смотреть utilization codebook'ов: OCR-text может улучшить geometry, но вызвать collapse в отдельных tokens.

Нужно смотреть invalid generated SID rate, особенно если late fusion увеличивает длину или сложность token sequence.

Нужно считать preprocessing cost на миллион items, потому что OCR-text - offline-heavy подход.

## Итог

Главный вывод: "text as text" не является очевидным оптимумом для semantic IDs. Если описание товара структурное и визуально-атрибутивное, OCR-text может дать более полезное пространство для дискретизации и генеративного retrieval.

## Источники

- [arXiv:2601.14697](https://arxiv.org/abs/2601.14697) , PDF/source.
