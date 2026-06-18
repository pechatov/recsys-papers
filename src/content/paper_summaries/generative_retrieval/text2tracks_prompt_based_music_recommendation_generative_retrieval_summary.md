---
title: "Text2Tracks: Prompt-based Music Recommendation via Generative Retrieval"
category: "generative_retrieval"
slug: "text2tracks_prompt_based_music_recommendation_generative_retrieval_summary"
catalogId: "paper-text2tracks_prompt_based_music_recommendation_generative_retrieval_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/text2tracks_prompt_based_music_recommendation_generative_retrieval_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2503.24193"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Enrico Palumbo, Gustavo Penha, Andreas Damianou, José Luis Redondo García, Timothy Christopher Heath, Alice Wang, Hugues Bouchard, Mounia Lalmas.
>
> **Аффилиации:** Spotify.
>
> **Источник:** [arXiv 2503.24193](https://arxiv.org/abs/2503.24193) · дата metadata: 2025-03-31.
>
> **Категория/теги:** generative retrieval, music recommendation, semantic track IDs, новое из ссылок.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://github.com/zetaalphavector/inpars](https://github.com/zetaalphavector/inpars) [https://github.com/borisveytsman/acmart/issues/440](https://github.com/borisveytsman/acmart/issues/440).

## 1. Коротко

- Главная идея: Text2Tracks формулирует prompt-based music recommendation как generative retrieval track IDs, а не generation of titles.
- Алгоритм: Модель учит mapping от natural-language prompt к track identifiers; ключевым фактором становится стратегия построения track IDs, включая semantic IDs.
- Evidence: Offline playlists-with-language evaluation показывает, что semantic/custom track IDs сильнее title-token generation.
- Ограничение: Нужен entity-resolution audit и проверка distribution shift между curated playlist prompts и живыми user requests.
- Итог: Полезна как domain case: semantic IDs работают не только для e-commerce, но и для music catalog с длинными названиями.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>While intuitive, this approach has several limitation.</td></tr>
<tr><th>Предложение авторов</th><td>In this paper, we propose to address the task of prompt-based music recommendation as a generative retrieval task.</td></tr>
<tr><th>Главный evidence claim</th><td>Within this setting, we introduce novel, effective, and efficient representations of track identifiers that significantly outperform commonly used strategies.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>objective не выражен стандартным ключевым словом; смотреть method/training sections</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source</td></tr>
<tr><th>Metrics</th><td>метрики нужно сверить в таблицах experiments</td></tr>
<tr><th>Baselines</th><td>TIGER, NCI</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Method: We start by formally defining prompt-based music recommendation as a generative track retrieval task, followed by the main components of the model. Figure diagram b displays a diagram of the model, showcasing how the training dataset D is first pre-processed into pairs of training instances using an ID strategy to represent the items. Then, the...
1. **Ключевой модуль.** Method: Prompt-based Music Recommendation The modeling assumption of this paper is that prompt-based music recommendation can be framed as a generative track retrieval task. The track retrieval task is defined as retrieving a set of tracks relevant to a given music recommendation query. Formally let D = (Q i, t 1, t 2,..., t k ) i=1 N be a dataset composed of...
1. **Learning signal.** Method: The ID strategy is responsible for generating a string identifier for each item in the collection T. We explore three types of IDs: based on the content of the item, based on integers, and learned ones. Figure strategies diagram describes the three classes of approaches at a high level.

## 5. Objectives, formulas и training details

**Detected objective keywords:** objective не выражен стандартным ключевым словом; смотреть method/training sections.

Явные equations не удалось надежно извлечь автоматически; при ручной проверке смотреть loss/objective в method/training subsections.

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Examples of the IDs generated for each strategy to represent a track for. "<" and ">" delimiters indicate that the string is added to the vocabulary of the model as a new independent token. UI indicates if the strategy uniquely identifies each track. tealUnderline indicates track name and purpledashed underline indicates artist name.
- Table of track ID strategies and results. Bold denotes the highest effectiveness for the use of semantic IDs leveraging collaborative filtering embeddings.
- Effectiveness of Text2Tracks and competing models. Bold denotes the highest effectiveness for hits@10.
- (a) Pre-trained LLMs deal with prompt-based music recommendation by generating the recommended artist name and song title, which are then resolved against an index to find the actual track identifiers. (b) is a generative track retrieval model composed of a component that represents tracks, i.e. the ID strategy $ $ that maps from a track to its ID, and a...
- The three categories of ID strategies using "\_" as a separator. Content-based strategies use textual metadata associated with the item. Integer-based approaches use random integer values for each metadata, potentially leveraging the hierarchy of metadata available. Learned approaches go from embeddings that represent the item to hierarchically structured...
- The effect on Hits@10 and on the diversity of the artists when increasing the homogeneity penalty hyperparameter, which applies a penalty for generating tokens that were selected in other beam search groups at prediction with

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source</td></tr>
<tr><th>Metrics</th><td>метрики нужно сверить в таблицах experiments</td></tr>
<tr><th>Baselines</th><td>TIGER, NCI</td></tr>
</tbody></table>
</div>

- Главная идея: Text2Tracks формулирует prompt-based music recommendation как generative retrieval track IDs, а не generation of titles.
- Evidence: Offline playlists-with-language evaluation показывает, что semantic/custom track IDs сильнее title-token generation.
- Experimental Setup: We here introduce the setup that we used to conduct our experiments. Dataset We use a dataset of playlist data. While the train and evaluation data are a subset of playlists with long descriptive titles that can be used as proxy for music recommendation prompts (e.g. chill lofi vibes to focus), the test data is a selected set of playlists created by a team...
- Experimental Setup: Baselines & Implementation Our first baseline,, retrieves the most popular tracks regardless of the query. For the dense (Bi-encoder) and sparse ( ) baselines, we represent each track by the titles of playlists they appear in the train set. So for example if t i appears in the playlists "rock", "metal" and "guitar solos" we represent t i by their...
- Results: What is the impact of different ID strategies on 's effectiveness? In Table id strategies we see the results for the different strategies to represent IDs. Our results show that the ID strategy is a deciding factor for the effectiveness of, achieving the best performance with, showing the power of using strong collaborative filtering-based track...
- Results: The worst-performing ID representation is which does not add any prior knowledge to the initial representation of the tracks, and requires the model to learn how queries match groups of tokens (on average 4 tokens) that represent the tracks. When using, the LM can leverage the "knowledge" stored in its weights regarding the items. Even though the number of...

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: Text2Tracks формулирует prompt-based music recommendation как generative retrieval track IDs, а не generation of titles.
- **Algorithmic/system:** Алгоритм: Модель учит mapping от natural-language prompt к track identifiers; ключевым фактором становится стратегия построения track IDs, включая semantic IDs.
- **Empirical:** Evidence: Offline playlists-with-language evaluation показывает, что semantic/custom track IDs сильнее title-token generation.
- **Practical:** Ограничение: Нужен entity-resolution audit и проверка distribution shift между curated playlist prompts и живыми user requests.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.
- Reward/utility signal достаточно стабилен и не подменяет user relevance узкой бизнес-метрикой.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source.
- Метрики: метрики нужно сверить в таблицах experiments.
- Baselines и parity settings: TIGER, NCI.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна как domain case: semantic IDs работают не только для e-commerce, но и для music catalog с длинными названиями.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: Text2Tracks формулирует prompt-based music recommendation как generative retrieval track IDs, а не generation of titles.
- Алгоритм: Модель учит mapping от natural-language prompt к track identifiers; ключевым фактором становится стратегия построения track IDs, включая semantic IDs.
- Evidence: Offline playlists-with-language evaluation показывает, что semantic/custom track IDs сильнее title-token generation.
- Ограничение: Нужен entity-resolution audit и проверка distribution shift между curated playlist prompts и живыми user requests.
- Итог: Полезна как domain case: semantic IDs работают не только для e-commerce, но и для music catalog с длинными названиями.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Conversational assistants such as ChatGPT ouyang2022training are getting increasingly more popular thanks to recent advances in Large Language Models (LLMs) achiam2023gpt,touvron2023llama,fitzgerald2022alexa. A prominent feature of modern LLMs is the...</td></tr>
<tr><td>Related Work</td><td>Prompt-based Recommendation Language-based preferences and textual inputs have always been a core part of recommender systems. Content-based recommender systems lops2011content have been around for several decades, matching user profiles with item textual...</td></tr>
<tr><td>Method</td><td>We start by formally defining prompt-based music recommendation as a generative track retrieval task, followed by the main components of the model. Figure diagram b displays a diagram of the model, showcasing how the training dataset D is first...</td></tr>
<tr><td>Experimental Setup</td><td>We here introduce the setup that we used to conduct our experiments. Dataset We use a dataset of playlist data. While the train and evaluation data are a subset of playlists with long descriptive titles that can be used as proxy for music recommendation...</td></tr>
<tr><td>Results</td><td>What is the impact of different ID strategies on 's effectiveness? In Table id strategies we see the results for the different strategies to represent IDs. Our results show that the ID strategy is a deciding factor for the effectiveness of,...</td></tr>
<tr><td>Conclusions</td><td>In this paper, we propose to address the problem of prompt-based music recommendation through the lenses of generative retrieval and we introduce a generative track retrieval model. Given that tracks are items with scarce text, learns to generate track IDs...</td></tr>
</tbody></table>
</div>
