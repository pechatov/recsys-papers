---
title: "MoMoREC: A Multi-agent Motivation Generation Framework for Residual Semantic ID-Aware Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "momorec_a_multi_agent_motivation_generation_framework_for_summary"
catalogId: "paper-momorec_a_multi_agent_motivation_generation_framework_for_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/momorec_a_multi_agent_motivation_generation_framework_for_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1609/aaai.v40i19.38623"
---
Подробное саммари статьи:

> **Авторы:** Yige Wang, Mingming Li, Li Wang, Kaichen Zhao, Wangming Li, Weipeng Jiang, Xueying Li.
>
> **Аффилиации:** Taobao & Tmall Group of Alibaba; Xi'an Jiaotong University.

## 1. Коротко

MoMoREC использует LLM не как медленный генератор item lists, а как offline/side pipeline для извлечения latent shopping motivations. Multi-agent LLM framework анализирует частые purchase sequences, получает мотивационные тексты/эмбеддинги, распространяет мотивации на long-tail items и затем превращает dense motivation embeddings в residual semantic IDs, совместимые с обычными recommendation models.

## 2. Контекст

Авторы начинают с наблюдения: традиционные ID-based models быстры, но мало интерпретируемы; прямые LLM recommenders имеют слишком много параметров, высокую latency и требуют adaptation training. Motivation - скрытая причина покупки - позволяет использовать comprehension capabilities LLM без online LLM inference на каждый запрос.

## 3. Проблема

- LLM embeddings high-dimensional и плохо совместимы с low-dimensional ID embeddings в классических моделях.
- Прямое LLM generation дает latency на порядок выше и огромную разницу в parameter count.
- Частые items легче анализировать, но long-tail остается без мотиваций.
- Нужно сохранить real-time inference, поэтому LLM output надо дискретизировать/сжать.

## 4. Метод и архитектура

Framework состоит из трех блоков. Motivation Generation использует Analysis Agent, Summary Module и Judgement/Arena/Scoring components, чтобы из user purchase history получить representative motivation. Motivation Spreading переносит мотивационные признаки с frequent items на long-tail через embedding model, contrastive learning, clustering и similarity top-K. Residual Semantic ID Retrieval строит codebook в motivation space и последовательно извлекает residual IDs.

## 5. Objective и алгоритм

Objective распределен по pipeline. LLM agents генерируют и выбирают мотивации по prompt/arena scoring. Spreading module учится сближать items с релевантными мотивациями и переносить их по semantic similarity. Residual SID строится как итеративный nearest-center lookup: выбирается ближайший code, residual обновляется, процесс повторяется M раз. Полученный multi-dimensional ID vector подается как categorical feature в sequential/recommendation model.

### 5.1. Пошаговая схема MoMoREC

1. **Выделить frequent purchase patterns.** Из логов берутся purchase sequences, встречающиеся два и более раза; это снижает шум для LLM motivation inference.
1. **Сгенерировать candidate motivations.** Analysis Agent описывает возможные причины покупки, Summary Module агрегирует themes, Judgement/Arena/Scoring выбирает наиболее representative motivation.
1. **Назначить мотивации head/frequent items.** Items из устойчивых purchase lists получают LLM-derived motivation text как latent behavioral explanation.
1. **Обучить пространство spreading.** Motivation texts и item features переводятся в embeddings; contrastive learning делает items с релевантными мотивациями ближе.
1. **Распространить мотивации на tail.** Для long-tail items берутся top-K похожие frequent items/clusters, после чего motivation labels/embeddings переносятся по semantic similarity.
1. **Построить residual semantic IDs.** Dense motivation embedding квантуется итеративно: nearest center -> subtract residual -> repeat $M$ раз, получая multi-code categorical vector.
1. **Подать IDs в обычный recommender.** DeepFM/DCN/DIN-like или sequential backbone использует residual motivation SIDs как sparse features без online LLM inference.

## 6. Эксперименты и метрики

Эксперименты AAAI проведены на трех benchmark datasets и нескольких traditional recommendation backbones. Метрики включают AUC/accuracy-style recommendation metrics в таблицах; авторы показывают, что MoMoREC значительно улучшает традиционные модели. В тексте и Figure 2 подчеркивается, что LLM generation имеет 10x или более latency increase и на сотни раз больше параметров относительно DeepFM/DCNmix/DIN-like models, тогда как MoMoREC переносит LLM knowledge в offline IDs.

## 7. Что важно в рисунках и таблицах

Figure 1 важен как постановка: мотивация покупки скрыта, но может быть inferred из observable behavior. Figure 2 сравнивает parameters/runtime traditional vs generation models и оправдывает offline LLM usage. Figure 3 показывает diminishing returns от простого увеличения language embedding size: recommendation model не умеет линейно использовать огромный dense vector. Figure 4 раскрывает весь pipeline MoMoREC и показывает место residual semantic IDs.

## 8. Сильные стороны

- Прагматично использует LLM: дорогая reasoning часть вынесена из online inference.
- Motivation spreading явно закрывает long-tail gap.
- Residual semantic IDs делают LLM-derived signals совместимыми с классическими categorical pipelines.
- Multi-agent схема повышает качество мотиваций по сравнению с single prompt generation.

## 9. Слабые стороны и ограничения

- Качество мотиваций зависит от prompts, LLM и языка/домена marketplace.
- Pipeline сложен: generation, selection, spreading, clustering и SID retrieval нужно версионировать отдельно.
- Long-tail spreading может переносить неверные мотивации через поверхностную similarity.
- AAAI PDF не полностью раскрывает все production параметры и стоимость offline generation.

## 10. Как реализовать и проверять

- Начинать с frequent co-purchase sequences и вручную проверить sample motivations на бизнес-смысл.
- Измерять coverage: доля head/mid/tail items с мотивациями до и после spreading.
- Сравнить dense motivation embedding feature vs residual SID feature при одинаковом backbone.
- В residual SID проверять collision rate, codebook usage и incremental AUC/NDCG по bucket'ам item frequency.

## 11. Связь с соседними работами

MoMoREC близок к Structured Term Identifiers тем, что использует language-native/LLM-derived semantics, но TID/GRLM генерирует textual identifiers для LLM generation, а MoMoREC превращает мотивации в residual IDs для traditional recommendation. С FusID его роднит идея: сначала получить richer semantic embedding, затем дискретизировать.

## 12. Итог

MoMoREC полезен как production-minded компромисс между LLM understanding и low-latency recommenders. Главная мысль: LLM может объяснять и кодировать мотивации offline, а online модель должна получать компактные residual semantic IDs, а не огромный dense vector или сам LLM в serving path.

## 13. Детальный разбор механизмов статьи

### 13.1. Multi-agent motivation generation

MoMoREC использует несколько LLM agents, потому что одна генерация мотивации по purchase history может быть шумной. Analysis Agent предлагает candidate motivations и объяснения, Summary Module собирает coherent themes, а Judgement/Arena/Scoring выбирает наиболее representative motivation. Это похоже на offline deliberation pipeline, а не на online recommender.

- Input - user purchase sequences, причем paper сохраняет sequences, которые встречаются два и более раза.
- Motivation назначается item'ам из purchase list как latent explanation поведения.
- Arena/scoring снижает риск случайной single-prompt hallucination.
- LLM используется для understanding, а не для прямой генерации рекомендаций.
- Это снижает online latency по сравнению с Qwen2.5-style generation models.

### 13.2. Motivation spreading

- Frequent items получают более надежные мотивации, потому что чаще встречаются в purchase patterns.
- Long-tail items получают мотивации через semantic similarity propagation.
- Embedding model переводит motivation text в dense space.
- Contrastive learning помогает выстроить пространство распространения.
- Top-K spreading и clustering контролируют, какие motivations переносятся на tail items.

### 13.3. Residual Semantic ID Retrieval

Dense motivation embeddings напрямую плохо входят в классические recommendation models: Figure 3 показывает diminishing returns при росте embedding size. Поэтому MoMoREC превращает motivation embeddings в residual semantic IDs: выбирает ближайший code center, вычитает его вклад и повторяет несколько шагов. Получается multi-dimensional categorical vector.

- Codebook строится по motivation embeddings.
- Каждый residual step дает один semantic ID component.
- Discrete IDs можно подать как обычные sparse/categorical features.
- Это снижает OOD gap между LLM embeddings и traditional low-dimensional ID embeddings.
- Residual formulation лучше сохраняет информацию, чем single cluster id.

### 13.4. Failure modes

- Motivation hallucination: LLM придумывает правдоподобную, но неверную причину покупки.
- Head bias: frequent item motivations могут доминировать и неверно распространяться на tail.
- Semantic over-smoothing: похожие items получают одинаковые мотивации, хотя purchase intent различается.
- Codebook collision: разные motivations попадают в один residual ID pattern.
- Lifecycle cost: offline LLM generation нужно повторять при смене каталога, сезона и пользовательских трендов.

## 14. Первичные источники

- Официальная AAAI страница/DOI: [AAAI 40(19), 15904-15914](https://ojs.aaai.org/index.php/AAAI/article/view/38623) .
- Официальный PDF AAAI: [article PDF](https://ojs.aaai.org/index.php/AAAI/article/view/38623/42585) .
