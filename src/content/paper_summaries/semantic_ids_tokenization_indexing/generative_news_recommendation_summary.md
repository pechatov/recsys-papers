---
title: "Generative News Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "generative_news_recommendation_summary"
catalogId: "paper-generative_news_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/generative_news_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2403.03424"
---
**Авторы:** Shen Gao, Jiabao Fang, Quan Tu, Zhitao Yao, Zhumin Chen, Pengjie Ren, Zhaochun Ren.

**Аффилиации:** University of Electronic Science and Technology of China; Shandong University; Renmin University of China; Leiden University.

**Индустрия:** academic news recommendation; no direct industry affiliation.

**Первичный источник:** arXiv source 2403.03424.

## Коротко

- GNR recommends by generating personalized multi-news narratives.
- It uses dual-level semantic/theme representations from LLM.
- UIFT aligns narrative generator with user interests.

## Контекст

- Traditional news recommendation ranks existing articles.
- Users often need event-level understanding across multiple related articles.
- LLMs can summarize themes and fuse narratives.

## Проблема

- Need improve ranking and generate factual personalized narrative.
- General LLM summaries may ignore user interest.
- Related news must be coherent and not random top-ranked items.

## Метод/архитектура

- Dual-level representation combines semantic-level news/user embeddings and LLM-generated theme-level representations.
- Personalized related news exploration ranks focal news, finds related news and filters by user preference.
- Interest-aware fusion generates narrative around focal news.
- UIFT distills ChatGPT then uses ranking loss.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Generative News Recommendation качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Objective/алгоритм

- Ranking loss uses negative sampling log-likelihood.
- Relation classifier uses Siamese network and contrastive margin loss.
- UIFT scores generated narratives with recommender-derived interest ranking.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Generative News Recommendation качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

### Подробная схема алгоритма

GNR - это не только ранжирование новостей: pipeline сначала находит focal/related articles, затем превращает их в персонализированный multi-news narrative.

1. **Построить dual-level representations.** Для новости берутся semantic-level embeddings и LLM-generated theme-level descriptions; для пользователя агрегируются истории кликов в тех же пространствах.
1. **Обучить ranking model.** На positive clicked news и negative samples оптимизируется log-likelihood, чтобы выбрать focal news под пользователя.
1. **Найти related news.** Siamese relation classifier проверяет, какие статьи связаны с focal news, используя contrastive margin loss.
1. **Отфильтровать по user interest.** Related candidates не просто добавляются в narrative: они проходят preference filter, чтобы тематическая связность не вытеснила персональную релевантность.
1. **Сгенерировать narrative.** LLM получает focal news, отобранные related news и user-interest signal и строит связный personalized story.
1. **Сделать UIFT.** Сначала генератор distill'ится от ChatGPT-style outputs, затем донастраивается через interest-aware ranking signal, чтобы narrative был не только fluent, но и полезен пользователю.
1. **Оценить два слоя.** Ranking метрики проверяют выбор новостей, generation win-rate/factuality checks - качество narrative.

```
encode each news article into semantic and theme features
encode user history in the same dual-level space
rank focal news with negative-sampling likelihood
for each focal news:
    retrieve related news with Siamese relation classifier
    filter related news by user preference
    generate narrative conditioned on focal, related set and user interest
    tune generator with UIFT interest-aware feedback
```

## Эксперименты

- Experiments evaluate recommendation accuracy and narrative fusion.
- Metrics: NDCG@5, AUC, MRR for ranking.
- PLM4NR title improves NDCG@5 62.38 -> 63.46 and AUC 69.44 -> 70.31 with dual-level representation.
- Additional RQs study alpha threshold, Tmax reference news and personalized vs non-personalized narratives.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Generative News Recommendation качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Рисунки/таблицы

- Figures: limitation of recommending only existing news, GNR paradigm, UIFT framework.
- Tables: prompt templates, recommendation performance, generation win-rate, alpha/Tmax effects.

## Ablation conclusions

- Dual-level representations improve multiple backbones.
- Personalized filtering matters for narrative relevance.
- UIFT improves personalized/factual narrative compared with generic generation.

## Сильные стороны

- **Выходом является продуктовая форма, а не только список ID.** Для новостей event-level narrative часто полезнее, чем набор похожих заголовков.
- **Dual-level representation соответствует домену.** Semantic embeddings ловят текстовую близость, theme-level LLM summaries ловят событие и угол подачи.
- **Related-news exploration отделена от personalization.** Это снижает риск собрать тематически связную, но неинтересную конкретному пользователю историю.
- **UIFT выравнивает генерацию с recommender signal.** Генератор не остается generic summarizer'ом, а получает preference-aware feedback.

## Ограничения

- **Hallucination/factuality risk выше, чем у обычного ranker.** Модель не только выбирает статьи, но и формулирует narrative; ошибки могут добавить несуществующие связи между событиями.
- **Оценка generation субъективна и дорога.** NDCG/AUC для ranking не доказывают, что итоговый narrative фактически корректен и редакционно приемлем.
- **Source attribution обязателен для production.** Без ссылок на исходные статьи пользователь не сможет проверить, откуда взяты утверждения.
- **Freshness/drift критичны.** News тематика быстро устаревает; theme representations и related-news graph требуют частого обновления.

## Как реализовать/проверять

1. Зафиксировать версию каталога, train/eval split и mapping item/document -> identifier; без этого невозможно понять, улучшает ли метод саму модель или только меняет пространство кандидатов.
1. Считать отдельно качество tokenization и качество generator/ranker: collision rate, utilization, Gini, valid-path rate, Recall/HR/NDCG/MRR и latency должны лежать в одном отчете.
1. Делать ablation не только по среднему качеству, но и по head/torso/tail, cold-start, new users/new items, long-history и категориям с похожими объектами.
1. Проверять деградацию при обновлении каталога: semantic IDs могут устаревать, а генератор может помнить старые пути.
1. Сохранять обратный индекс identifier -> item/document и явно логировать случаи many-to-one collisions.
1. Для генеративного вывода использовать constrained decoding или post-validation, иначе invalid identifiers будут маскироваться в offline метриках.
1. В production считать стоимость не только модели, но и перестроения codebook, trie/index, feature pipeline и мониторинга drift.
1. В отчетах отделять retrieval-stage gains от downstream ranking/business metrics, потому что рост HR@K не всегда дает CTR/CVR uplift.

## Failure modes и мониторинг

- Identifier collapse: малая часть кодов получает большую долю объектов, а генератор начинает переиспользовать популярные пути.
- Semantic-collaborative mismatch: похожие по тексту объекты имеют разные user intents, или наоборот, совместно потребляемые объекты текстово далеки.
- Exposure bias autoregressive generation: ошибка раннего токена полностью меняет candidate path.
- Popularity bias: модель учится генерировать frequent IDs и выглядит хорошо на aggregate, но теряет novelty и tail coverage.
- Evaluation leakage: если ID/tokenizer обучался на будущем каталоге или post-training видел target-like сигналы, offline gain завышен.
- Serving mismatch: offline beam шире и медленнее production beam, поэтому качество не переносится напрямую.
- Специфично для этой статьи: Hallucination/factuality risk.
- Специфично для этой статьи: Evaluation of narratives is expensive and partly subjective.
- Специфично для этой статьи: Needs editorial safeguards and source attribution in production.

## Связь

- Related to Prompt-to-Slate as output generation beyond item ID.
- Different from SID papers, but part of broader generative recommendation.

## Итог

- GNR expands recommender output from item list to personalized narrative.
- LLM themes can improve both matching and presentation.

## Минимальный план воспроизведения

<div class="table-scroll">
<table>
<thead>
<tr><th>Шаг</th><th>Что сделать</th><th>Что считать успехом</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>Собрать исходные item/document features и interaction/query logs в той же временной схеме, что у статьи.</td><td>Нет leakage, воспроизводимы splits и отрицательные примеры.</td></tr>
<tr><td>2</td><td>Построить identifiers/token representations и сохранить mapping plus collision report.</td><td>Utilization и collision metrics понятны до обучения generator.</td></tr>
<tr><td>3</td><td>Обучить baseline без нового компонента и полный метод с тем же budget.</td><td>Сравнение честное по compute, beam, candidates и preprocessing.</td></tr>
<tr><td>4</td><td>Запустить ablations по каждому заявленному компоненту.</td><td>Каждый компонент дает объяснимый вклад или честно признается избыточным.</td></tr>
<tr><td>5</td><td>Проверить production-like constraints: latency, invalid IDs, refresh, monitoring.</td><td>Offline gain не исчезает при реальных ограничениях serving.</td></tr>
</tbody>
</table>
</div>

Примечание: если в источнике не раскрыты приватные production details, summary явно помечает такие ограничения и не выдумывает закрытые числа.
