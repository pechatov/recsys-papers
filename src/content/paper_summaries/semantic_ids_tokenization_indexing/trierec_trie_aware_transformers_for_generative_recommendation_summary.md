---
title: "Trie-Aware Transformers for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "trierec_trie_aware_transformers_for_generative_recommendation_summary"
catalogId: "paper-trierec_trie_aware_transformers_for_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/trierec_trie_aware_transformers_for_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2602.21677"
---
Подробное саммари статьи:

> **Авторы:** Zhenxiang Xu, Jiawei Chen, Sirui Chen, Yong He, Jieyu Yang, Chuan Yuan, Ke Ding, Can Wang.
>
> **Аффилиации:** Zhejiang University; Ant Group.

## 1. Коротко: о чем статья

TrieRec не предлагает новый semantic-ID tokenizer, но решает важную проблему использования уже построенных hierarchical identifiers. Item tokenization задает prefix tree над каталогом: items с общими prefixes находятся рядом в trie. Обычный Transformer при этом видит плоскую sequence of tokens и не получает явного сигнала о depth, ancestors, descendants и topology relations.

Авторы добавляют trie-aware structural bias в Transformer, чтобы generative recommender лучше использовал иерархию, уже заложенную в semantic IDs.

## 2. Метод

TrieRec вводит два вида positional encoding. Trie-aware absolute positional encoding добавляет в node representation локальный структурный контекст: глубину, ancestors и descendants. Topology-aware relative positional encoding добавляет в self-attention информацию о pairwise structural relation между tokens/nodes.

Метод model-agnostic: его можно встроить в разные GR backbones без изменения самого tokenizer'а. Поэтому работа особенно полезна как "serving/modeling layer" поверх TIGER-like, LETTER-like или других hierarchical SID pipelines.

## 3. Пошаговый алгоритм

1. **Построить semantic-ID trie.** Каждый item path добавляется в prefix tree.
1. **Посчитать node features.** Для каждого token node вычисляются depth, ancestor/descendant context и другие structural features.
1. **Добавить absolute structural encoding.** Node representation получает trie-aware information до Transformer layers.
1. **Добавить relative topology encoding.** Self-attention учитывает отношения между token nodes в trie.
1. **Обучить generative recommender.** Target остается next item identifier, но model уже видит tree topology.
1. **Использовать обычный constrained decoding.** Generated path ограничивается валидными prefixes/items.

## 4. Сильные стороны

- **Использует скрытый ресурс SID-пайплайна.** Hierarchical IDs создают trie, но многие модели этот trie игнорируют.
- **Не требует переобучать tokenizer.** Это снижает adoption cost для уже построенных semantic IDs.
- **Model-agnostic improvement.** Подход можно проверять поверх нескольких GR backbones.
- **Хороший baseline для topology-aware decoding/modeling.** Особенно релевантен для работ про long IDs и variable-length IDs.

## 5. Ограничения

- Метод усиливает структуру, которую дал tokenizer; если hierarchy плохая, bias может вредить.
- Не решает assignment-level проблемы: collisions, staleness, code utilization и tokenizer objective mismatch остаются внешними.
- Для personalized/dynamic IDs trie может стать нестабильным или слишком большим.
- Нужны отдельные latency measurements для production decoding на больших каталогах.

## 6. Вывод

TrieRec закрывает важную дыру между semantic-ID construction и Transformer modeling. Для исследований semantic IDs это напоминание: качество зависит не только от tokenizer'а, но и от того, умеет ли generator использовать induced structure.
