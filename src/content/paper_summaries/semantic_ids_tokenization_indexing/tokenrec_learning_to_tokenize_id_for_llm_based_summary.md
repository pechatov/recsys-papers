---
title: "TokenRec: Learning to Tokenize ID for LLM-Based Generative Recommendations"
category: "semantic_ids_tokenization_indexing"
slug: "tokenrec_learning_to_tokenize_id_for_llm_based_summary"
catalogId: "paper-tokenrec_learning_to_tokenize_id_for_llm_based_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/tokenrec_learning_to_tokenize_id_for_llm_based_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1109/tkde.2025.3599265"
---
> **Авторы:** Haohao Qu, Wenqi Fan, Zihuai Zhao, Qing Li.
>
> **Аффилиации:** Hong Kong Polytechnic University.

## Коротко

TokenRec предлагает токенизировать user IDs и item IDs в LLM-compatible discrete tokens.

Вместо текстового описания item или случайного ID модель использует collaborative filtering representations.

Эти representations квантуются Masked Vector-Quantized Tokenizer, чтобы перенести high-order collaborative knowledge в токены, понятные generative recommender'у.

Второй вклад - efficient generative retrieval без дорогого autoregressive beam search для каждого пользователя.

## Контекст

LLM-based recommender systems часто пытаются представить users/items текстом.

Но текст не всегда содержит collaborative сигнал: кто вместе слушает, покупает или смотрит объекты.

Atomic IDs несут collaborative knowledge через embedding tables, но не совместимы с LLM vocabulary и плохо обобщают unseen users/items.

TokenRec занимает промежуточную позицию: сначала учит CF representations, затем переводит их в discrete tokens.

## Проблема

Есть две ключевые проблемы.

Первая - как закодировать collaborative knowledge в дискретные токены, не потеряв high-order user-item structure.

Вторая - как делать top-K recommendation эффективно.

Обычные LLM generation methods используют autoregressive decoding и beam search, что дорого при массовом serving.

## Метод / архитектура

TokenRec состоит из ID tokenization и generative retrieval paradigm.

Masked Vector-Quantized Tokenizers строятся отдельно для users и items.

Сначала CF encoder учит latent representations из user-item graph/interactions.

Затем часть latent dimensions маскируется, что заставляет tokenizer восстанавливать и дискретизировать robust collaborative signal.

Vector quantization разбивает representation на sub-codebooks.

Каждый user/item получает несколько discrete tokens.

## Objective / алгоритм

Tokenizer objective сочетает reconstruction collaborative representation и vector quantization regularization.

Masking ratio `rho` управляет тем, сколько информации скрывается во время tokenization.

Количество sub-codebooks `K` и число tokens/codewords `J` задают емкость discrete space.

После tokenization recommendation формулируется как matching user tokens и item tokens.

Efficient retrieval stage избегает полного autoregressive item-by-item decoding.

Это важное отличие от TIGER-style beam search.

### Подробная схема алгоритма TokenRec

1. **Обучить CF encoder.** Из user-item interaction graph строятся continuous collaborative representations для users и items.
1. **Разделить user/item tokenizers.** Masked Vector-Quantized Tokenizer обучается отдельно для user IDs и item IDs, потому что распределения и cold-start свойства разные.
1. **Применить masking.** Доля latent dimensions скрывается с ratio `rho` , чтобы tokenizer не копировал representation напрямую, а восстанавливал robust collaborative signal.
1. **Квантизовать по sub-codebooks.** Representation разбивается на `K` частей, каждая выбирает codeword из `J` вариантов; итог - несколько discrete tokens на user/item.
1. **Обучить reconstruction/VQ objective.** Tokenizer должен восстановить CF representation и одновременно использовать codebooks без collapse.
1. **Сформировать retrieval representation.** User tokens и item tokens становятся LLM-compatible discrete interface для matching.
1. **Выполнить efficient top-K retrieval.** Вместо autoregressive beam search по всем item identifiers модель использует matching stage, который быстрее оценивает candidates.
1. **Проверить generalization.** Seen/unseen users/items оцениваются отдельно, потому что collaborative tokens легко переобучаются на известных ID.

```
CF_user, CF_item = train_cf_encoder(interactions)

for entity_type in {user, item}:
    Z = CF_representations(entity_type)
    masked_Z = mask_dimensions(Z, rho)
    tokens = VQ_tokenizer(masked_Z, K_subcodebooks, J_codewords)
    loss = reconstruction_loss(tokens, Z) + vq_commitment_loss
    update tokenizer

user_tokens = tokenize(users)
item_tokens = tokenize(items)
train generative/matching recommender(user_tokens, item_tokens)
retrieve top-K without full autoregressive beam over catalog
```

## Эксперименты и метрики

Датасеты: LastFM, ML1M, Amazon Beauty и Amazon Clothing.

Метрики: HR@10/20/30 и NDCG@10/20/30.

TokenRec сравнивается с traditional recommenders и LLM-based methods.

В LastFM TokenRec превосходит strongest baselines в среднем на 19.08% по HR@20 и 9.09% по NDCG@20.

Generalizability evaluation проверяет seen/unseen users.

Для P5 и POD наблюдается падение более 40% HR@20/NDCG@20 на unseen users.

TIGER падает меньше, около 6.10% HR@20, благодаря item-side semantic IDs.

TokenRec нацелен на generalizable user/item tokenization за счет collaborative tokens.

## Рисунки / таблицы

Intro figure сравнивает ID tokenization methods для LLM-based recommendation.

Main framework figure показывает MQ-Tokenizer и retrieval module.

Inference figure подчеркивает efficiency и поддержку new users/items.

Таблицы результатов разделены на LastFM/ML1M и Beauty/Clothing.

Ablation table показывает вклад tokenization components.

Hyperparameter figures проверяют masking ratio `rho`, число sub-codebooks `K` и codewords `J`.

## Сильные стороны

- **Токенизируются и users, и items.** Это важное отличие от item-only SID: user side тоже получает compact collaborative language.
- **Источник токенов - CF representations.** TokenRec переносит high-order user-item structure, а не только text/content semantics.
- **Efficiency является частью метода.** Работа явно уходит от дорогого TIGER-style beam search для каждого пользователя.
- **Проверяется generalizability.** Seen/unseen users analysis показывает, где LLM/text-based методы проседают и зачем нужны collaborative tokens.
- **Есть hyperparameter analysis.** Masking ratio `rho` , число sub-codebooks `K` и codewords `J` напрямую контролируют capacity и robustness.

## Ограничения

- **Зависимость от CF encoder.** Если collaborative representation шумная или popularity-biased, tokenizer аккуратно дискретизирует уже плохой сигнал.
- **Cold-start items без interactions остаются слабыми.** Без side information content-based SID может быть лучше для новых объектов.
- **Temporal drift ломает collaborative tokens.** User/item co-consumption меняется, поэтому mappings требуют refresh и alignment strategy.
- **LLM-compatible не значит semantically understood.** Pretrained language model не знает смысл новых discrete CF tokens без grounding/fine-tuning.
- **Masking ratio чувствителен.** Слишком высокий `rho` уничтожает collaborative signal, слишком низкий дает memorization и слабую generalization.

## Как реализовать / проверять

1. Обучить CF encoder на user-item interactions.
1. Сохранить user/item latent representations.
1. Обучить Masked VQ tokenizer с разными `rho` , `K` и `J` .
1. Проверить utilization codebooks и reconstruction quality.
1. Оценить HR/NDCG на seen и unseen users/items.
1. Измерить inference time per user для Top-20/Top-K recommendation.

## Production notes

Для serving нужно версионировать tokenizers и mappings users/items→tokens.

Новые users/items требуют online или periodic token assignment.

Если токены используются в LLM prompt, нужен guard против invalid token combinations.

Если используется efficient retrieval без beam search, нужно проверить recall ceiling относительно exact top-K.

## Failure modes

Первый failure mode - codebook collapse.

Второй - masking ratio слишком высокий, tokenizer теряет collaborative signal.

Третий - masking ratio слишком низкий, tokens переобучаются и хуже generalize.

Четвертый - unseen users без достаточного interaction context.

Пятый - temporal drift collaborative tokens.

## Связь с соседними работами

TokenRec отличается от TIGER/RQ-VAE тем, что строит ID tokens из collaborative knowledge.

С Spotify GLIDE и GenRec его связывает LLM-based recommendation, но TokenRec больше занят tokenizer/retrieval efficiency.

С VK staleness paper связь в lifecycle collaborative tokenization: collaborative SIDs сильны, но требуют refresh strategy.

## Итог

TokenRec отвечает на вопрос, как дать LLM recommender'у не только язык и item text, но и collaborative structure.

Его практическая ценность в совместной токенизации users/items и снижении inference cost относительно autoregressive beam generation.

## Источники

- DOI/IEEE: [10.1109/TKDE.2025.3599265](https://doi.org/10.1109/tkde.2025.3599265) .
- Доступный полный препринт: [arXiv:2406.10450](https://arxiv.org/abs/2406.10450) , source/PDF.
