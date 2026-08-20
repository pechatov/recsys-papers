---
title: "Attribute-Conditioned Multimodal Slot Factorization for Controllable Fashion Retrieval"
category: "generative_retrieval"
slug: "attribute_conditioned_multimodal_slot_factorization_for_controllable_fashion_retrieval_summary"
catalogId: "paper-attribute_conditioned_multimodal_slot_factorization_for_controllable_fashion_retrieval_summary"
paperUrl: "https://arxiv.org/abs/2608.12570"
---
> **Авторы:** Najmeh Forouzandehmehr, Topojoy Biswas, Evren Korpeoglu, Kannan Achan.
>
> **Аффилиации:** Walmart Global Tech.
>
> **Источник:** arXiv:2608.12570v1 от 2026-08-12. Публичные код, данные и модель не заявлены.

## Коротко: о чем статья

MM-slotgate раскладывает text/image-представления fashion items на именованные attribute slots с отдельными text-image gates, сохраняя управляемость отдельных атрибутов.

## Abstract

Fashion retrieval often requires satisfying multiple attributes at once, such as category, color, pattern, and demographic. Monolithic embeddings mix these signals into a single vector, making attribute-specific control difficult at retrieval time. Many existing semantic-ID methods provide discrete item codes, but these codes are typically optimized as item-level or residual addresses and do not expose named, independently controllable attribute slots. We introduce MM-slotgate, a multimodal slot encoder that factorizes Fashion-CLIP text and image embeddings into four named attribute slots. Each slot learns its own text-image gate, so visually grounded attributes such as color and pattern can rely more on image evidence, while taxonomy-oriented attributes such as category and demographic can remain more text-driven. On H&M, using a combined slot-similarity and slot-logit retrieval score, MM-slotgate achieves 0.7566 macro ConstraintSatisfied@10, outperforming equal-weight multimodal fusion (0.7142) and fCLIP text-only retrieval (0.4755). The largest gain is on color, which improves from 0.321 to 0.889 (+0.568 absolute), as the learned color gate assigns 57.4% weight to image evidence. The learned gates are interpretable without modality supervision: color is image-leaning, category is text-leaning, and pattern and demographic lie near the middle. The resulting slots also remain controllable: linear probes show no measured excess leakage beyond the label-correlation baseline, and quantized slot codes support targeted intervention, including a 15.3x lift for color. These results suggest that controllable fashion retrieval benefits from typed, attribute-conditioned multimodal slots rather than either a single global embedding or opaque item-level semantic IDs.

## Ограничения

Это свежий препринт: заявленные результаты и сравнения следует интерпретировать в границах раскрытых авторами данных, протоколов и baseline-ов.
