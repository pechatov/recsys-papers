---
title: "GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation"
category: "generative_retrieval"
slug: "gpt4rec_generative_framework_personalized_recommendation_summary"
catalogId: "paper-gpt4rec_generative_framework_personalized_recommendation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/gpt4rec_generative_framework_personalized_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2304.03879"
---
Подробное саммари статьи:

> **Авторы:** Jinming Li, Wentao Zhang, Tian Wang, Guanglei Xiong, Alan Lu, Gerard Medioni.
>
> **Аффилиации:** University of Michigan, Ann Arbor; Amazon, United States.
>
> **Публикация:** arXiv:2304.03879, 2023.

## 1. Коротко

GPT4Rec предлагает гибридный генеративно-поисковый фреймворк: GPT-2 не генерирует item ID напрямую, а генерирует несколько гипотетических поисковых запросов, которые описывают интересы пользователя. Затем обычный search engine, в экспериментах BM25, ищет item'ы по этим запросам в каталоге.

Это важный контраст к TIGER: вместо short semantic IDs здесь используется натуральный язык как промежуточный representation пользователя. Сгенерированные запросы одновременно служат retrieval bridge и интерпретацией интересов: их можно читать человеку, анализировать на coverage/diversity и применять к новым item'ам, если у них есть title/metadata.

## 2. Контекст

Авторы критикуют embedding-based candidate matching за три проблемы: item'ы часто представлены как ID без текстового содержания; discriminative models плохо интерпретируют, какие интересы пользователя активны; growing inventory и cold-start требуют дообучения или пересчета embeddings.

GPT4Rec вдохновлен search engines: пользователь в e-commerce явно вводит запрос, а поисковик сопоставляет его с текстом item'ов. Для рекомендаций такого явного запроса нет, поэтому модель должна вывести “возможные запросы” из истории пользователя. Это превращает recommendation в задачу `query generation + searching`.

## 3. Метод

Модель состоит из двух независимых компонентов:

- **Generative language model.** GPT-2 117M дообучается на последовательностях item titles. Input - item titles из истории пользователя плюс prompt; target - title последнего item'а. Идея: самый точный fine-grained query для target item - его собственное название.
- **Search engine.** Каталог индексируется по item titles. На inference сгенерированные queries подаются в BM25, который возвращает top candidates.

Ключевая техника - **multi-query beam search**. Вместо одного запроса модель генерирует несколько beam outputs. Для top-K рекомендации авторы часто используют стратегию “K queries, по одному item на query”: это повышает diversity и покрывает несколько интересов пользователя, а не только один dominant intent.

## 4. Эксперименты/результаты

Эксперименты проведены на Amazon Review 5-core в категориях Beauty и Electronics. Истории пользователей дедуплицируются и обрезаются до длины 15; split 0.8/0.1/0.1; последний item в test sequence - target. Baselines: FM-BPR, ContentRec, YouTubeDNN, BERT4Rec.

<div class="table-scroll">
<table>
<thead><tr><th>Датасет</th><th>Лучший baseline</th><th>GPT4Rec</th><th>Что важно</th></tr></thead>
<tbody>
<tr><td>Beauty, Recall@40</td><td>BERT4Rec 0.1161</td><td>0.2040 с 40 queries</td><td>+75.7% relative improvement по заявлению авторов.</td></tr>
<tr><td>Electronics, Recall@40</td><td>BERT4Rec 0.0751</td><td>0.0918 с 40 queries</td><td>+22.2% relative improvement на большем каталоге.</td></tr>
</tbody>
</table>
</div>

Анализ multi-query показывает монотонный рост Recall при увеличении числа generated queries. В таблице diversity/coverage лучшие значения также достигаются при большем числе queries, что поддерживает тезис о multi-interest representation.

## 5. Ограничения

- **Retrieval зависит от BM25.** Если title не содержит нужных терминов или несколько item'ов имеют похожие названия, генерация хорошего query не гарантирует хороший candidate recall.
- **Двухстадийная оптимизация.** GPT-2 и BM25 оптимизируются отдельно; авторы прямо оставляют joint supervision from search results как future work.
- **Нет проверки в truly large-scale serving.** BM25 удобен и масштабируем, но качество и latency в production зависят от search инфраструктуры и качества item metadata.
- **Natural-language grounding неоднозначен.** Запросы интерпретируемы, но это не валидные identifiers. В отличие от SID, они не дают строгий one-to-one mapping на item.

## 6. Как читать для GR/SID

GPT4Rec полезен как альтернатива “генерировать ID”: модель может генерировать *запрос*, а не item. Для SID-направления это хороший reminder, что semantic ID решает только часть проблемы: он ускоряет и структурирует target space, но может терять человеческую интерпретируемость, которую GPT4Rec получает почти бесплатно.

В практическом GR-пайплайне эту работу стоит читать как вариант hybrid retrieval: LLM/LM строит explainable user intent, а внешний retriever отвечает за catalog grounding. Это особенно релевантно для cold-start item'ов с хорошими titles/metadata.
