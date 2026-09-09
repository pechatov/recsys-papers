---
title: "Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval"
category: "generative_retrieval"
slug: "conversational_recommendation_over_live_e_commerce_catalogues_with_self_refreshing_retrieval_summary"
catalogId: "paper-conversational_recommendation_over_live_e_commerce_catalogues_with_self_refreshing_retrieval_summary"
paperUrl: "https://arxiv.org/abs/2608.27006"
---
> **Авторы:** Ante Kapetanovic, Tomislav Duricic, Dionizije Fa, Andro Mercep, Emanuel Lacic.
>
> **Аффилиации:** Infobip..
>
> **Источник:** arXiv:2608.27006 от 2026-08-27.

## Коротко: о чем статья

Практический conversational CRS поддерживает live e-commerce catalog через delta-sync self-refreshing retrieval, оставляя LLM роль intent и preference layer.

## Abstract

Conversational recommender systems based on large language models (LLMs) are usually evaluated on static, pre-indexed item collections, yet e-commerce catalogues change continuously as products are added or removed, repriced, and restocked. We present a merchant-agnostic, multi-turn conversational shopping assistant that operates over such live catalogues. Its central component is a self-refreshing retriever that ingests a merchant product feed, enriches the records, and synchronizes them into a vector index. On each run, per-item hashes identify which products are new, changed, deleted, or unchanged, so only the delta is processed rather than rebuilding the whole catalogue. A controller-based dialogue layer consumes this index, using an LLM only for intent classification and preference elicitation while retrieval, reranking, and diversity selection run as dedicated functions. Our demonstration is a WhatsApp shopping assistant in which catalogue changes reach the recommendations after the next successful sync. A live chatbot, documentation, and a recorded walkthrough are available at https://github.com/infobip/infobip-agentic-crs.

## Ограничения

Это свежий препринт; результаты необходимо интерпретировать в рамках раскрытых авторами протоколов, данных и baseline-ов.
