---
title: "SIDScope: A Diagnostic Resource for Semantic-ID Interfaces in Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "sidscope_a_diagnostic_resource_for_semantic_id_interfaces_in_generative_recommendation_summary"
catalogId: "paper-sidscope_a_diagnostic_resource_for_semantic_id_interfaces_in_generative_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2608.18779"
---
> **Авторы:** Jiandong Ding, Huijie Qin, Tiandeng Wu, Yi Cao.
>
> **Аффилиации:** Huawei Technologies Co., Ltd., Shanghai, China. Explicitly listed in arXiv HTML..
>
> **Источник:** arXiv:2608.18779 от 2026-08-19.

## Коротко: о чем статья

SIDScope — диагностический ресурс для оценки происхождения, структуры, revision drift и path-to-item поведения интерфейсов semantic ID.

## Abstract

Semantic-ID mappings are reusable interfaces between item tokenizers and generative recommenders, yet released mappings rarely state whether they are coherent, what structure they expose, how generated paths resolve, or what must be revalidated after a refresh. SIDScope is a source-traced diagnostic resource for these decisions. It normalizes item-to-code artifacts, verifies provenance and joins, profiles mapping structure, compares paired revisions, and accounts for path-to-item outcomes in generated traces. Across nine source-traced tokenizer exports from seven families on Amazon and Yelp data - eight executable routes plus one auditable snapshot - SIDScope reveals that interface health is multi-signal rather than scalar. Its central finding is mechanism-conditional: prefix alignment strongly tracks held-out candidate exposure when retrieval consumes SID prefixes, then weakens as scoring becomes prefix-independent. Trained trace accounting exposes a second hidden gap: a valid target path can survive without uniquely retrieving the target item by 1.2-3.0 percentage points. A refresh case establishes a third: repairing the mapping does not by itself restore an inherited generator; model reuse requires a separate handoff check. The package provides frozen evidence summaries, conformance reports, trace labels, table builders, and CPU-only verifiers. It supports decisions about artifact readiness, interface risks, and revalidation before model reuse.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
