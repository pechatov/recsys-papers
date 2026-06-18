---
title: "HiD-VAE: Interpretable Generative Recommendation via Hierarchical and Disentangled Semantic IDs"
category: "semantic_ids_tokenization_indexing"
slug: "hid_vae_interpretable_generative_recommendation_via_hierarchical_summary"
catalogId: "paper-hid_vae_interpretable_generative_recommendation_via_hierarchical_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/hid_vae_interpretable_generative_recommendation_via_hierarchical_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2508.04618"
---
Подробное саммари статьи:

> **Авторы:** Dengzhao Fang, Jingtong Gao, Chengcheng Zhu, Yu Li, Xiangyu Zhao, Yi Chang.
>
> **Аффилиации:** Jilin University; City University of Hong Kong; Nanjing University.

## 1. Коротко: о чем статья

HiD-VAE утверждает, что стандартные semantic IDs в generative recommendation страдают двумя связанными проблемами: они flat/uninterpretable и склонны к entanglement, из-за которого возникают вредные ID collisions. Даже если RQ-VAE хорошо реконструирует content embedding, он не гарантирует, что уровни кода соответствуют человечески понятной иерархии.

Работа предлагает hierarchical and disentangled semantic IDs: каждый уровень quantization должен соответствовать уровню item taxonomy, а collision control должен быть встроен в обучение tokenizer'а.

## 2. Метод

Основа HiD-VAE -- VAE/RQ-VAE tokenizer с hierarchical supervision. Для items используются multi-level tags: например category, subcategory, fine-grained type. Если готовых tags нет, авторы предлагают LLM-assisted retrieval-then-classification pipeline, где LLM выбирает label из ограниченного candidate set, снижая риск hallucination.

Tokenizer получает tag alignment loss и tag prediction loss, чтобы разные quantization layers кодировали разные уровни иерархии. Для борьбы с коллизиями вводится uniqueness loss: если разные items получают одинаковый SID, их continuous latent representations дополнительно разводятся margin-based penalty.

## 3. Пошаговый алгоритм

1. **Получить hierarchical tags.** Использовать существующую taxonomy или сгенерировать labels через retrieval-then-classification.
1. **Обучить HiD-VAE tokenizer.** Совместить reconstruction/VQ losses с tag alignment, tag prediction и uniqueness loss.
1. **Назначить semantic IDs.** Каждый item получает hierarchical code path, который должен соответствовать category path.
1. **Заморозить tokenizer.** User histories переводятся в sequences of hierarchical IDs.
1. **Обучить autoregressive recommender.** Transformer генерирует следующий SID с hierarchy-aware embeddings.
1. **Использовать constrained decoding.** На inference разрешаются только valid prefixes, соответствующие реальным items.

## 4. Сильные стороны

- **Интерпретируемость не post-hoc.** Hierarchical path становится частью semantic ID design.
- **Collision mitigation на уровне latent space.** Uniqueness loss решает проблему раньше, чем post-processing conflict token.
- **Подходит для аудита.** Category path можно анализировать как объяснение, почему item попал в область candidate generation.
- **Совместим с constrained decoding.** Иерархическая структура естественно ложится в prefix tree.

## 5. Ограничения

- Нужна хорошая taxonomy; плохая taxonomy превращается в жесткое неверное inductive bias.
- LLM-generated tags могут быть стабильными только при аккуратном candidate restriction.
- Слишком сильная category supervision может ухудшить discovery, если поведенческая близость не совпадает с taxonomy.
- Метод остается двухстадийным: recommendation loss напрямую не оптимизирует tokenizer.

## 6. Вывод

HiD-VAE важен для линии исследований, где semantic IDs должны быть не только эффективными, но и объяснимыми. Это сильный кандидат для сравнения в работах про collision-aware, taxonomy-aware и interpretable item tokenization.
