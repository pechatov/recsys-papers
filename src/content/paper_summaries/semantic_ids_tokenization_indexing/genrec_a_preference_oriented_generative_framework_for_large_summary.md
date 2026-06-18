---
title: "GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "genrec_a_preference_oriented_generative_framework_for_large_summary"
catalogId: "paper-genrec_a_preference_oriented_generative_framework_for_large_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/genrec_a_preference_oriented_generative_framework_for_large_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.14878"
---
> **Авторы:** Yanyan Zou, Junbo Qi, Lunsong Huang, Yu Li, Kewei Xu, Jiahao Gao, Binglei Zhao, Xuanhua Yang, Sulong Xu, Shengjie Li.
>
> **Аффилиации:** JD.com; Waseda University.

## Коротко

GenRec строит large-scale генеративный recommender вокруг semantic IDs, page-wise supervised fine-tuning и RL alignment. Главное отличие от TIGER-style next-item generation: модель оптимизируют не только на попадание в следующий item, но и на preference-oriented reward, который лучше связан с бизнес-целью.

## Контекст

LLM/decoder-only модели привлекательны для recommendation, но длинные SID-последовательности увеличивают prefilling cost, а teacher-forcing next-token objective может не совпадать с click/conversion preference. В больших каталогах также критичны invalid SID hallucinations.

## Проблема

Стандартное SFT учит воспроизводить один target item или sequence, но не умеет выбирать из множества приемлемых рекомендаций те, которые максимизируют preference reward. RL без ограничений может reward-hack'ить: повышать SIM reward, ухудшая HR@50 или генерируя невалидные коды.

## Метод / архитектура

Items квантуются в semantic IDs. Чтобы снизить стоимость длинных кодов, GenRec вводит Linear Token Merger: embeddings токенов item SID конкатенируются и проецируются в один latent vector на prefilling side, а служебные токены вроде `<sep>` остаются отдельно. Обучение включает user-centric page-wise NTP SFT и RL phase на базе GRPO-SR с gating mechanism.

## Objective / алгоритм

SFT objective сравнивает next-item и next-sequence formulations; page-wise NTP ускоряет сходимость, потому что обучает модель на странице рекомендаций, а не только на одиночном target. RL использует reward metrics на similarity/preference score `r^SIM` и gate, который защищает от reward hacking: модель должна улучшать reward, не разрушая валидность и hit-rate.

## Детальный алгоритм GenRec

GenRec можно разложить на четыре последовательных этапа: построение SID, supervised page-wise learning, сжатие prefilling tokens и preference RL с защитным gate. Это не просто TIGER с большим decoder-ом: статья меняет и input packing, и training target, и alignment phase.

1. **Построить Semantic IDs.** Сравниваются RQ-VAE и RQ-KMeans; качество tokenizer-а проверяется по entropy слоев и downstream HR/NDCG/HaR.
1. **Собрать user-centric page examples.** Вместо одного next item training target становится page-wise sequence рекомендаций, ближе к реальному интерфейсу.
1. **Применить Linear Token Merger.** SID tokens одного item конкатенируются на embedding level и проецируются в один latent vector для prefilling. Служебные tokens вроде `<sep>` не сжимаются.
1. **Обучить SFT через page-wise NTP.** Decoder учится генерировать SID sequence страницы; одновременно контролируются HR@K, NDCG@K и Hallucination Rate.
1. **Запустить GRPO-SR RL phase.** Candidate pages сэмплируются, reward считается через preference/similarity scorer `r^SIM` .
1. **Применить gate `G`.** Gate разрешает RL update только если reward improvement не сопровождается деградацией validity/hit-rate, снижая reward hacking.
1. **На inference декодировать валидные SID paths.** Beam outputs проверяются по SID vocabulary/trie; invalid SID входит в HaR и не должен маскироваться.

```
sid_map = train_or_select_tokenizer(items, method="RQ-KMeans/RQ-VAE")
train_examples = build_pagewise_sequences(user_logs, sid_map)

for batch in train_examples:
    merged_history = linear_token_merger(batch.history_sid_tokens)
    loss_sft = pagewise_next_token_loss(model, merged_history, batch.target_page_sids)
    update(loss_sft)

for rl_step in range(T):
    candidates = sample_pages(model, user_context)
    reward = r_SIM(candidates)
    gate = G(reward, HR50, hallucination_rate, validity)
    if gate:
        update(GRPO_SR_loss(model, candidates, reward))

serve with constrained SID decoding and monitor HR/NDCG/HaR/reward together
```

## Эксперименты и метрики

Для SFT используются HR@1/10/50, NDCG@10/50 и Hallucination Rate. Авторы сравнивают RQ-VAE и RQ-KMeans tokenization, task formulation, model scaling и loss curves. В RL часть добавляется Reward@K; полный GRPO-SR показывает крупнейший относительный gain на Reward@1 (+18.01%), но ablation без gate демонстрирует, почему reward-only оптимизация опасна: могут падать HR@50 и HaR.

## Рисунки / таблицы

Figure architecture показывает SID quantization и Linear Token Merger. Таблица tokenization quality сравнивает entropy слоев RQ-VAE/RQ-KMeans. Таблицы next-item vs next-sequence, task formulation и model scaling показывают компромисс качества, hallucination и размера модели. Online A/B table связывает RL alignment с click-conversion misalignment.

## Сильные стороны

- **Соединяет SID-based GR с preference alignment.** GenRec явно показывает, что next-token likelihood не равен бизнес-предпочтению.
- **Linear Token Merger бьет по реальному bottleneck.** Длинные SID histories увеличивают prefilling cost; merger уменьшает input positions без отказа от compositional SID.
- **Page-wise SFT ближе к интерфейсу.** Модель учится генерировать страницу, а не одиночный target item.
- **HaR включен как обязательная метрика.** Invalid SID hallucinations не прячутся за Recall/NDCG.
- **Gate в RL phase демонстрирует зрелость постановки.** Авторы прямо показывают риск reward hacking без ограничителя.

## Ограничения

- **Reward зависит от внутреннего scorer-а.** Если `r^SIM` слабо связан с click/conversion, RL улучшит proxy, а не recommendation quality.
- **RL stage трудно воспроизвести.** GRPO-SR, gate thresholds и reward data завязаны на JD.com production signals.
- **Token Merger может over-compress SID.** Если projection слишком мала, fine-grained information последних SID tokens теряется.
- **Page-wise training наследует exposure bias.** Модель может учить порядок страниц старой системы, а не оптимальную recommendation policy.
- **Serving требует строгий SID trie.** Без constrained decoding HaR может уничтожить практическую пользу высокого reward.

## Как реализовать / проверять

Нужны валидный SID trie, page-level training examples, scorer для reward и отдельный мониторинг HaR. Запускать следует по этапам: tokenizer quality, SFT без merger, SFT с merger, затем RL с gate. Проверять не только HR/NDCG, но и invalid SID, reward hacking signs, diversity, latency и conversion-correlated offline metrics.

## Связь с соседними работами

GenRec близок к Spotify GLIDE и TokenRec по использованию LLM-like generation, но сильнее фокусируется на preference alignment. С RecoChain его связывает идея совместить retrieval/ranking, а с Snapchat - практическая проблема валидности и полезности SID в production.

## Semantic ID tokenization construction

Статья сравнивает RQ-VAE и RQ-KMeans как способы построения semantic IDs.

Для production важны не только HR/NDCG downstream, но и entropy слоев codebook.

Низкая entropy первого слоя означает, что слишком много items попадают в один coarse bucket.

Низкая entropy последнего слоя означает, что fine-grained discrimination не используется.

GenRec поэтому рассматривает tokenization quality как отдельный этап перед SFT.

Это похоже на Snapchat design choice: uniqueness и entropy являются диагностикой, но итоговый критерий - recommendation performance.

## Linear Token Merger

SID item обычно состоит из нескольких tokens.

Если история пользователя содержит много items, prefilling sequence быстро растет.

Linear Token Merger берет embeddings SID-токенов одного item, конкатенирует их и проецирует в единый latent vector.

Тем самым item занимает меньше позиций в prefilling part.

Служебные tokens не сжимаются, чтобы не смешивать control structure с item representation.

Практический смысл: merger снижает memory/latency pressure без отказа от compositional SID.

## User-centric page-wise NTP SFT

Обычный next-item NTP учит модель предсказывать один следующий объект.

Page-wise formulation ближе к recommender interface: пользователь видит страницу рекомендаций.

Модель учится генерировать sequence items page-wise, учитывая user-centric preference context.

В loss curves авторы показывают, что page-wise NTP сходится быстрее, чем vanilla NTP.

Это также уменьшает mismatch между training target и serving output.

## RL alignment: GRPO-SR and gating

После SFT GenRec применяет RL alignment.

Reward связан с similarity/preference metric `r^SIM`.

GRPO-SR оптимизирует generated recommendations относительно reward, а не только likelihood historical item.

Gating mechanism `G` нужен как ограничитель.

Без gate модель может exploit reward: выбирать items с высоким SIM, но ухудшать HR@50 или повышать hallucination.

В таблице RL авторы прямо интерпретируют drop без `G` как reward hacking.

## Hallucination Rate и scaling

HaR - доля invalid SIDs в generated results.

Это обязательная метрика для SID-based LLM recommender.

Высокий HR@K бесполезен, если модель часто генерирует коды, которых нет в item catalog.

Более крупные модели дают меньший SFT loss, но gains имеют diminishing returns после 3B.

Это означает, что сначала нужно оптимизировать tokenizer, merger и decoding, а не сразу масштабировать backbone.

## Failure modes и проверка

Первый failure mode - invalid SID hallucination.

Второй - reward hacking в RL phase.

Третий - over-compression Token Merger: если проекция слишком мала, теряется fine-grained SID information.

Четвертый - page-wise exposure bias: модель может учиться порядку страницы из логов старой системы.

Пятый - scorer mismatch: `r^SIM` может не совпасть с clicks/conversions.

1. Сначала обучить tokenizer и зафиксировать SID vocabulary.
1. Проверить entropy L1/L3 и invalid mapping.
1. Обучить SFT с обычными SID tokens и с Token Merger.
1. Сравнить HR/NDCG/HaR при одинаковом decoding budget.
1. Запускать RL только после стабильного SFT baseline.
1. В RL dashboard держать reward, HR@50, HaR и diversity одновременно.

## Итог

GenRec показывает, что следующий шаг SID-based recommendation - это не только лучшие semantic tokens, но и выравнивание генератора с preference metrics. Без такого alignment модель может быть хорошим next-token predictor, но слабым recommender.

## Источники

- [arXiv:2604.14878](https://arxiv.org/abs/2604.14878) , PDF/source.
