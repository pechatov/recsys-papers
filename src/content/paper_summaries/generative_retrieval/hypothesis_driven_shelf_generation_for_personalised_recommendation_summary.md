---
title: "Hypothesis-Driven Shelf Generation for Personalised Recommendation"
category: "generative_retrieval"
slug: "hypothesis_driven_shelf_generation_for_personalised_recommendation_summary"
catalogId: "paper-hypothesis_driven_shelf_generation_for_personalised_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2607.25823"
---
> **Авторы:** Aleksandr V. Petrov, Tarun Chillara, Matthew D. Moellman, Lucas de Haas, Yabai Song, Alina Susoykina, Melissa Crawford, Gabriel Negash, Erik Franco, Tasnim Rahman, Binal Jhaveri, Shubham Bansal, Hugues Bouchard, Roberto Mirizzi, Mounia Lalmas, Aloïs Gruson.
>
> **Аффилиации:** Spotify.
>
> **Источник:** ACM RecSys 2026 Industry Track, DOI 10.1145/3773078.3831914; arXiv:2607.25823v1 от 2026-07-28. Код, модели и production data не опубликованы.

## 1. Коротко

Работа Spotify переносит генерацию из уровня отдельных items на уровень интерфейса Home. Вместо конечного набора вручную заданных shelf templates система сначала формулирует для пользователя natural-language гипотезу о том, какой ряд контента ему нужен, а затем независимо наполняет эту гипотезу реальными объектами каталога. Пример гипотезы: не общий шаблон `ambient music`, а узкий концепт вроде `glacial ambient post-rock with orchestral textures` с заданными content type, familiarity и freshness constraints.

Главная идея - использовать shelf hypothesis как промежуточный planning representation и контракт между user modeling и retrieval. User profile видит только planner; generative retrieval получает уже сформулированную гипотезу и не имеет доступа к исходному профилю. После retrieval отдельный LLM выбирает финальный набор items и переписывает title/subtitle так, чтобы обещание в заголовке соответствовало содержимому ряда. Вся тяжёлая часть считается offline, а на Home поступает precomputed candidate shelf.

Это сильная systems paper, но не доказательство того, что generated shelves в среднем лучше всего существующего Spotify Home. Offline LLM judges показывают большие gains относительно BM25/MiniLM и до/после alignment, однако online часть сравнивает лучшие варианты внутри отдельных content-type pools и прямо названа авторами descriptive, а не pooled causal estimate. Результат неоднороден: albums +36% к лучшему classic shelf, episodes +2%, но playlists -14%, artists -8% и podcast shows -41%.

<figure class="paper-figure">
  <img src="../../assets/hypothesis_driven_shelf/pipeline.png" alt="Four-stage Spotify pipeline from hypothesis generation to offline serving">
  <figcaption>Рисунок 1. Четыре стадии: hypothesis generation, catalogue fulfilment, shelf alignment и precomputed serving.</figcaption>
</figure>

## 2. Зачем нужен отдельный shelf planner

Shelf - это не просто ranked list с подписью. Title и subtitle объясняют пользователю, почему весь ряд появился на странице, то есть задают semantic promise для набора. В template-based системе одна сущность неявно связывает три решения: user intent, допустимый catalog subset и retrieval logic. Такая схема хорошо работает для повторяющихся намерений вроде new releases, но плохо масштабируется на пересечения genre, era, mood, familiarity, market и media type.

Spotify разрывает эту связь. Planner решает, какие shelves должны существовать для пользователя; fulfilment решает, какие catalogue entities реализуют концепт; alignment проверяет coherence всего набора и согласует с ним user-facing text. Это позволяет независимо менять генератор гипотез, retriever и presentation layer.

Формально hypothesis представлена как

$$
h = (q, c, f, r, t_0, d_0),
$$

где $q$ - natural-language concept, $c$ - content type, $f$ - familiarity level, $r$ - market/freshness constraints, а $t_0,d_0$ - черновые title и subtitle. Финальный shelf имеет вид

$$
s = (t, d, q, \mathcal{I}),
$$

где $\mathcal{I}$ - упорядоченный список resolved Spotify entities.

## 3. Архитектура и метод

### 3.1. Stage 1: hypothesis generation

Planner строит compact taste profile из recent listening, long-term affinities, market context, familiar content и podcast engagement. По нему генерируется несколько structured hypotheses, чтобы один Home surface мог покрыть разные стороны taste profile.

На этапе разработки использовался frontier LLM. Для production его поведение distill'или в compact open-source LLM, запущенный как GPU batch job. Модель выводит constrained schema; отсутствующие или невалидные categorical fields заменяются safe defaults. На фиксированной выборке из 800 production profiles frontier и distilled models получили почти одинаковый overall judge score: 78.3% и 78.2%.

### 3.2. Stage 2: catalogue fulfilment

Fulfilment формулируется как generative retrieval по Semantic IDs. Vocabulary небольшого open-source LLM расширяется SemID tokens, выученными на Spotify catalogue. Decoder получает только hypothesis и structured constraints, но не user profile: персонализация уже должна быть сжата в $h$.

Valid generation обеспечивается content-type-specific tries для albums, artists, editorial playlists, podcast shows и episodes. Дополнительные indexes задают область поиска:

- editorial shelves используют market-specific catalog;
- fresh shelves - recency-filtered catalog;
- familiar shelves - динамический trie из знакомых пользователю entities;
- discovery shelves - более широкий catalog.

После decoding SemIDs преобразуются в Spotify URIs и фильтруются по type consistency, prior listening и familiarity constraints.

### 3.3. Stage 3: candidate selection and shelf alignment

Retrieval возвращает candidate set, а не готовый ряд. Отдельный LLM получает hypothesis, черновые title/subtitle и enriched metadata candidates, выбирает финальные $k$ items и переписывает текст. Это set-level optimization: даже individually relevant items могут образовать incoherent shelf или не выполнить слишком узкое обещание title.

### 3.4. Stage 4: serving

Pipeline выполняется daily offline batch'ем для eligible users. Готовые shelves сохраняются как Home candidates и конкурируют с существующими рядами в обычном production ranker; они не pin'ятся на фиксированные позиции. Поэтому online LLM inference и дополнительная latency при открытии приложения не нужны.

## 4. Как оценивали систему

Открытого gold target для новой shelf hypothesis нет, поэтому авторы используют два LLM judges со шкалой 0/1/2:

- User-to-Hypothesis Judge видит profile evidence и hypothesis, но не retrieved items; оценивает taste alignment, personalization depth, discovery potential, specificity и title quality;
- Hypothesis-to-Shelf Judge видит hypothesis, title/subtitle и enriched item list; оценивает style match, item relevance, set coherence, hypothesis coverage, completeness, diversity и title-promise fulfilment.

Основная offline cohort для fulfilment содержит 1,000 users и 10,000 hypotheses. Confidence intervals считаются 10,000 bootstrap resamples. Для retrieval comparisons используется paired $t$-test с Bonferroni correction; для pre/post alignment - Welch test, потому что cohorts разные и не содержат общих shelf IDs.

## 5. Результаты

### 5.1. Качество гипотез

<div class="table-scroll">
<table>
<thead><tr><th>Dimension, 0-2</th><th>Score</th></tr></thead>
<tbody>
<tr><td>Overall</td><td>1.59 ± 0.01</td></tr>
<tr><td>Taste Alignment</td><td>1.59 ± 0.01</td></tr>
<tr><td>Personalisation Depth</td><td>1.45 ± 0.01</td></tr>
<tr><td>Discovery Potential</td><td>1.31 ± 0.01</td></tr>
<tr><td>Hypothesis Specificity</td><td>1.99 ± 0.00</td></tr>
<tr><td>Title Quality</td><td>1.76 ± 0.01</td></tr>
</tbody>
</table>
</div>

Music hypotheses заметно сильнее spoken word. Content-type average равен 1.90 для Album, 1.98 для Artist и 1.86 для Playlist, но только 0.66 для Show и 0.51 для Episode. Это заранее объясняет неоднородность online results.

### 5.2. Generative retrieval против text retrieval

<div class="table-scroll">
<table>
<thead><tr><th>Method</th><th>Overall</th><th>Relevance</th><th>Coverage</th><th>Completeness</th><th>Diversity</th><th>Title</th></tr></thead>
<tbody>
<tr><td>BM25</td><td>0.56</td><td>0.84</td><td>0.92</td><td>0.75</td><td>1.06</td><td>0.54</td></tr>
<tr><td>Dense MiniLM</td><td>0.39</td><td>0.65</td><td>0.78</td><td>0.61</td><td>0.94</td><td>0.37</td></tr>
<tr><td>Hybrid</td><td>0.49</td><td>0.76</td><td>0.92</td><td>0.74</td><td>1.06</td><td>0.46</td></tr>
<tr><td>Generative Retrieval</td><td><strong>0.71</strong></td><td><strong>1.04</strong></td><td><strong>1.15</strong></td><td><strong>1.28</strong></td><td><strong>1.39</strong></td><td><strong>0.66</strong></td></tr>
</tbody>
</table>
</div>

По author-defined judge generative retrieval выигрывает все dimensions; особенно большие разрывы видны в completeness, diversity и coverage. BM25 остаётся сильным там, где curated descriptors почти дословно совпадают с hypothesis. Преимущество GR больше для indirect stylistic/cultural associations и комбинаций атрибутов, плохо отражённых в metadata.

### 5.3. Alignment - не косметический post-processing

<div class="table-scroll">
<table>
<thead><tr><th>Metric</th><th>Pre</th><th>Post</th><th>Relative change</th></tr></thead>
<tbody>
<tr><td>Overall</td><td>0.71</td><td>1.27</td><td>+78%</td></tr>
<tr><td>Item Relevance</td><td>1.04</td><td>1.58</td><td>+52%</td></tr>
<tr><td>Shelf Coherence</td><td>1.05</td><td>1.64</td><td>+56%</td></tr>
<tr><td>Diversity</td><td>1.39</td><td>1.74</td><td>+25%</td></tr>
<tr><td>Title Promise</td><td>0.66</td><td>1.31</td><td>+99%</td></tr>
</tbody>
</table>
</div>

<figure class="paper-figure">
  <img src="../../assets/hypothesis_driven_shelf/shelf_alignment_by_content_type.png" alt="LLM judge score before and after shelf alignment by content type">
  <figcaption>Рисунок 2. После candidate selection и text alignment judge score растёт для всех content types.</figcaption>
</figure>

Большой gain ожидаем: alignment stage напрямую оптимизирует тот же semantic contract, который измеряет Hypothesis-to-Shelf Judge. Но pre и post cohorts не matched: 10,000 pre-alignment shelves и более 16,000 production shelves не имеют общих IDs. Поэтому +78% нельзя читать как clean paired causal effect одной и той же операции.

### 5.4. Online random exposure

Метрика - 30-second stream rate. Shelf order рандомизируется на уровне request, чтобы снизить влияние Home ranker и позиции.

<div class="table-scroll">
<table>
<thead><tr><th>Content type</th><th>Hypothesis-driven</th><th>Best classic</th><th>Rank</th><th>Delta</th></tr></thead>
<tbody>
<tr><td>Album</td><td>1.20 ± 0.13%</td><td>0.88 ± 0.09%</td><td>1 / 10</td><td>+36%</td></tr>
<tr><td>Artist</td><td>0.82 ± 0.17%</td><td>0.89 ± 0.06%</td><td>2 / 9</td><td>-8%</td></tr>
<tr><td>Playlist</td><td>0.92 ± 0.09%</td><td>1.07 ± 0.07%</td><td>5 / 15</td><td>-14%</td></tr>
<tr><td>Show</td><td>0.92 ± 0.31%</td><td>1.57 ± 0.11%</td><td>2 / 6</td><td>-41%</td></tr>
<tr><td>Episode</td><td>0.63 ± 0.20%</td><td>0.62 ± 0.08%</td><td>1 / 7</td><td>+2%</td></tr>
</tbody>
</table>
</div>

Это подтверждает production viability и сильный album use case, но не общий superiority claim. Таблица выбирает strongest generated и strongest classic shelf в каждом pool и не агрегирует causal lift системы целиком.

## 6. Сильные стороны

- Хорошая decomposition между planning, retrieval, set alignment и serving; у каждой стадии есть явный interface и отдельный failure mode.
- Shelf рассматривается как coherent set с semantic promise, а не как набор независимых relevant items.
- Frontier-to-small-model distillation и fully offline serving делают подход реалистичным для production latency.
- Есть online evaluation с random exposure, а авторы не скрывают отрицательные результаты для playlists и shows.

## 7. Ограничения и вопросы

LLM judges одновременно являются главным offline metric и близко соответствуют optimization target alignment stage. Без опубликованной human agreement study для конкретных rubrics остаётся риск judge bias и circular optimization.

Сравнение fulfilment ограничено BM25, MiniLM и их linear hybrid. Нет сильного production dense retriever, cross-encoder reranker или trained text-to-item retrieval baseline, поэтому paper надёжно показывает преимущество над простыми text baselines, но не над всем современным retrieval stack.

Pre/post alignment cohorts разные, поэтому reported +78% смешивает effect alignment с cohort composition. Нужен paired ablation на одинаковых hypotheses/candidates.

Online часть - ранняя descriptive evaluation. Не опубликованы pooled treatment effect, standard production-ranking experiment, traffic/sample size и long-term novelty/fatigue metrics. Особенно слабым остаётся podcast show generation.

Нет деталей о SemID tokenizer, model sizes, training compute, catalogue refresh, batch cost и частоте stale shelves. Это не позволяет воспроизвести или оценить economics перехода к near-real-time generation.

## 8. Связь с соседними работами

По отношению к PLUM эта работа использует Semantic IDs не для прямой next-item recommendation из user history, а как execution layer для уже сформулированного language plan. User reasoning и catalog grounding намеренно разделены.

По отношению к обычным carousel recommenders меняется объект генерации: система не только ранжирует фиксированный inventory rows, но создаёт новые personalised shelf concepts. При этом production ranker остаётся поверх и решает, какие готовые shelves реально показать.

## 9. Итог

Главный вклад статьи - не новый SID tokenizer, а ясная architecture for generated recommendation surfaces: `user profile -> language hypothesis -> constrained catalogue fulfilment -> set/text alignment -> offline serving`. Самый убедительный результат - то, что такая система вообще работает на Spotify Home и создаёт сильные album shelves. Самый важный caveat - offline gains сильно зависят от LLM judge, а online advantage пока content-type-dependent и не доказывает общий lift Home experience.

## Источники

- [arXiv:2607.25823](https://arxiv.org/abs/2607.25823)
- [ACM DOI: 10.1145/3773078.3831914](https://doi.org/10.1145/3773078.3831914)
