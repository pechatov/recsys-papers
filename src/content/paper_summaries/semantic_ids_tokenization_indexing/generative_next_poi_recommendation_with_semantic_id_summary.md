---
title: "Generative Next POI Recommendation with Semantic ID"
category: "semantic_ids_tokenization_indexing"
slug: "generative_next_poi_recommendation_with_semantic_id_summary"
catalogId: "paper-generative_next_poi_recommendation_with_semantic_id_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/generative_next_poi_recommendation_with_semantic_id_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1145/3711896.3736981"
---
> **Авторы:** Dongsheng Wang, Yuxi Huang, Shen Gao, Yifan Wang, Chengrui Huang, Shuo Shang.
>
> **Аффилиации:** University of Electronic Science and Technology of China.

## Коротко

GNPR-SID применяет semantic IDs к next POI recommendation.

POI получает не случайный numeric ID, а semantic POI ID, построенный из semantic и collaborative features.

LLM затем fine-tune'ится генерировать следующий POI SID по истории посещений.

Авторы сообщают до 16% улучшения recommendation accuracy на benchmark datasets.

## Контекст

Next POI recommendation отличается от обычного item recommendation.

Важны последовательность посещений, время, spatial context и semantics мест.

Random POI IDs не показывают LLM, что два ресторана, станции или парка могут быть похожи.

Semantic ID должен сделать POI space более понятным для generative model.

## Проблема

Существующие generative POI methods часто используют random numeric IDs.

Такие IDs не передают semantic relation между похожими POIs.

LLM вынужден учить mapping ID→meaning почти с нуля.

Кроме того, POI recommendation должен учитывать both collaborative check-in patterns и semantic POI attributes.

## Метод / архитектура

GNPR-SID состоит из двух модулей.

Первый - Semantic POI ID Construction.

Он отображает POI в SID через codebook quantization.

Второй - Generative POI Recommendation.

Он fine-tune'ит LLM: input - historical POI SID sequence, output - next POI SID.

Overview figure показывает эти две части как основной pipeline.

## Semantic ID construction

POI representation строится из semantic features и collaborative features.

Semantic features описывают место: category, textual attributes и другие POI properties.

Collaborative features идут из user check-in patterns.

Residual quantized VAE переводит POI embeddings в discrete semantic space.

RQ-VAE последовательно выбирает codewords из codebooks и формирует compositional SID.

Так похожие POIs получают связанные SIDs, а не случайные номера.

## Diversity loss

Авторы добавляют diversity loss.

Его цель - равномернее распределять SIDs по semantic space.

Без такого loss возможен collapse: многие POIs получают одинаковые или близкие коды.

В experiments есть таблица с разными weights `lambda` diversity loss.

Она показывает Unique, Collision и Acc@1.

Это полезно, потому что слишком сильная уникальность может вредить semantic sharing, а слишком слабая - повышать collisions.

## Generative POI recommendation

LLM получает последовательность прошлых POI SIDs.

Prompt format включает SID tokens и, в некоторых вариантах, timestamps.

Модель генерирует SID следующего POI.

Appendix показывает prompt formats для variants without SID и without time.

Авторы также используют data augmentation: next POI task может заменяться fill-in-the-blank task.

## Пошаговый алгоритм GNPR-SID

GNPR-SID состоит из offline tokenization POI catalog и supervised fine-tuning LLM на последовательностях посещений. В отличие от обычного item SID, здесь после генерации нужно дополнительно проверять spatial/time feasibility.

1. **Собрать POI features.** Для каждого места объединяются category/text attributes, coordinates и collaborative check-in patterns.
1. **Обучить POI encoder.** Semantic и collaborative signals переводятся в общий embedding, пригодный для residual quantization.
1. **Построить SID через RQ-VAE.** Несколько codebooks последовательно кодируют residual; итоговый SID является compositional representation POI.
1. **Добавить diversity loss.** Weight `lambda` подбирается по Unique, Collision и Acc@1: задача не просто максимизировать уникальность, а избежать collapse без разрушения semantic grouping.
1. **Сериализовать user trajectory.** История POI SIDs и timestamp tokens превращается в prompt для LLM; augmentation добавляет fill-in-the-blank variants.
1. **Fine-tune LLM.** Target - SID следующего POI. Inference генерирует SID, который мапится на catalog POI и фильтруется по доступности/географии.

```
for poi in catalog:
    semantic = encode_text_category(poi)
    collaborative = encode_checkin_patterns(poi)
    embedding = fuse(semantic, collaborative, coordinates=poi.location)
    sid[poi] = rq_vae_quantize(embedding)

train_quantizer:
    loss = reconstruction_loss(poi_embedding, reconstructed_embedding)
         + lambda_div * diversity_loss(sid_assignments)
    update(encoder, rq_vae_codebooks)

for trajectory in user_checkins:
    prompt = format_history([sid[poi] for poi in trajectory.history], timestamps)
    target = sid[trajectory.next_poi]
    loss = llm_generation_loss(prompt, target)
    update(LLM)

serving:
    generated_sid = LLM.generate(history_prompt)
    poi = sid_to_poi[generated_sid]
    return spatial_time_filter(poi, user_context)
```

## Эксперименты и метрики

Датасеты: NYC, TKY и CA.

Основная метрика - Acc@1.

Accuracy считается как доля случаев, где predicted POI совпал с ground truth.

Main result table сравнивает models по input representation и использованию timestamps.

Ablation table проверяет вклад SID, time и других компонентов.

Out-of-domain table сравнивает GNPR-SID с LLM4POI на переносе.

Efficiency table показывает стоимость на NYC dataset.

## Рисунки / таблицы

Figure 1 сравнивает random ID и semantic ID: SID захватывает semantic associations между похожими POIs.

Overview figure показывает Semantic POI ID Construction и Generative POI Recommendation.

Table datasets фиксирует статистику NYC, TKY и CA.

Table main results показывает Acc@1 по трем datasets.

Table diversity loss связывает Unique, Collision и Acc@1.

Semantic relevance analysis показывает, что SIDs действительно группируют семантически близкие POIs.

## Сильные стороны

- Работа переносит SID idea в POI domain, где spatial/temporal context особенно важен.
- SID строится из semantic и collaborative features, а не из одного источника.
- Diversity loss явно контролирует collision/unique trade-off.
- Есть out-of-domain и efficiency analysis.

## Ограничения

Acc@1 - жесткая метрика, но она не показывает качество top-K alternatives.

POI domains зависят от города: NYC, TKY и CA имеют разные spatial patterns.

LLM fine-tuning может быть дорогим относительно классических POI models.

SID construction зависит от качества semantic/collaborative features.

Cold-start POIs без check-ins требуют content-only fallback.

## Как реализовать / проверять

1. Собрать POI features: category, text, координаты, check-in interactions.
1. Обучить POI encoder и RQ-VAE quantizer.
1. Подобрать diversity loss weight `lambda` .
1. Проверить Unique, Collision и semantic relevance SID groups.
1. Fine-tune LLM на historical SID sequences.
1. Оценить Acc@1, top-K accuracy, out-of-domain transfer и inference latency.

## Production notes

В POI serving нужен spatial filter: нельзя рекомендовать географически невозможные места.

SID generation должен обновляться при открытии/закрытии мест.

Для путешественников нужен cross-city fallback, потому что user history может быть из другого города.

Time features нельзя игнорировать: next POI утром и ночью имеют разные distributions.

## Failure modes

Первый failure mode - semantic IDs группируют похожие POIs, но не учитывают distance.

Второй - diversity loss слишком сильный и разрушает useful semantic grouping.

Третий - model memorizes city-specific IDs и плохо переносится.

Четвертый - timestamps отсутствуют или noisy.

Пятый - generated SID валиден, но POI закрыт или недоступен.

## Связь с соседними работами

GNPR-SID ближе всего к TIGER-style generative retrieval, но переносит идею в spatial mobility.

С TokenRec его объединяет LLM-based generation по discrete IDs.

С AdaSID/Snapchat его объединяет collision/uniqueness trade-off, здесь выраженный через diversity loss.

## Итог

GNPR-SID показывает, что semantic IDs полезны не только для товаров и видео, но и для mobility recommendation.

Ключевой вклад - связать POI semantic/collaborative representation с LLM-friendly SID и контролировать collision через diversity objective.

## Источники

- DOI/ACM: [10.1145/3711896.3736981](https://doi.org/10.1145/3711896.3736981) .
- Доступный полный препринт: [arXiv:2506.01375](https://arxiv.org/abs/2506.01375) , source/PDF.
