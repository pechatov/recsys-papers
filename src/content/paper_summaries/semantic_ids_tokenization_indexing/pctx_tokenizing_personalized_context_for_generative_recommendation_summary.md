---
title: "Pctx: Tokenizing Personalized Context for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "pctx_tokenizing_personalized_context_for_generative_recommendation_summary"
catalogId: "paper-pctx_tokenizing_personalized_context_for_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/pctx_tokenizing_personalized_context_for_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2510.21276"
---
Подробное саммари статьи:

> **Авторы:** Qiyong Zhong, Jiajie Su, Yunshan Ma, Julian McAuley, Yupeng Hou.
>
> **Аффилиации:** Zhejiang University; University of California San Diego; Singapore Management University.

## 1. Коротко: о чем статья

Pctx ставит под вопрос базовое предположение static semantic IDs: один item должен иметь один фиксированный код для всех пользователей. В autoregressive GR это означает, что items с одинаковыми prefixes получают похожие вероятности для всех users, то есть система навязывает универсальную item similarity.

Авторы предлагают personalized context-aware tokenizer. Один и тот же item может быть представлен разными semantic IDs в зависимости от user history: дорогие часы могут быть подарком, инвестицией или fashion accessory, и в каждом контексте item должен соседствовать с разными объектами.

## 2. Метод

Tokenizer получает не только item features, но и user context representation, построенную по истории взаимодействий. На выходе формируются personalized semantic IDs. Generative recommender затем обучается на sequences, где actions представлены context-conditioned tokenization, а не глобальным item-to-code mapping.

Главная идея: semantic ID описывает не только объект каталога, но и интерпретацию action пользователем. Это сближает Pctx с ActionPiece, но Pctx явно работает в терминологии semantic IDs и personalized tokenization.

## 3. Пошаговый алгоритм

1. **Закодировать user history.** Auxiliary sequential model строит context embedding, отражающий пользовательские intent'ы.
1. **Закодировать item features.** Используются признаки item, такие как title/description или content embedding.
1. **Слить item и context.** Tokenizer получает оба сигнала и генерирует semantic IDs conditioned on user context.
1. **Построить training sequences.** История пользователя превращается в sequence personalized action IDs.
1. **Обучить GR model.** Autoregressive model предсказывает следующий personalized semantic ID.
1. **Сопоставить генерацию с item candidates.** На inference требуется аккуратный mapping generated personalized codes к валидным items.

## 4. Сильные стороны

- **Новая ось персонализации.** Personalization переносится в tokenizer, а не только в sequence model.
- **Решает проблему universal similarity.** Fixed prefixes больше не обязаны означать одно и то же для всех users.
- **Особенно полезно для многозначных items.** В fashion, retail, media и POI один объект часто имеет разные роли.
- **Хорошая исследовательская провокация.** Работа заставляет явно оценивать, когда canonical item ID полезен, а когда мешает.

## 5. Ограничения

- Personalized IDs усложняют serving: cache и trie больше не могут быть полностью item-centric.
- Grounding generated code обратно в item становится сложнее, особенно при beam search.
- Стабильность кодов зависит от user context encoder; шумная история может менять tokenization слишком резко.
- Offline evaluation должна различать ошибку item selection и ошибку context-conditioned code assignment.

## 6. Вывод

Pctx важен как мост между semantic IDs и personalized action tokenization. Это один из самых перспективных источников идей для работ на 3-6 месяцев: можно изучать, где динамические SIDs дают gain, а где canonical IDs остаются незаменимыми для стабильного serving.
