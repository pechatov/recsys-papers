---
title: "On Generative Agents in Recommendation"
category: "generative_retrieval"
slug: "on_generative_agents_in_recommendation_summary"
catalogId: "paper-on_generative_agents_in_recommendation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/on_generative_agents_in_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1145/3626772.3657844"
---
Подробное саммари статьи:

> **Авторы:** An Zhang, Yuxin Chen, Leheng Sheng, Xiang Wang, Tat-Seng Chua.
>
> **Аффилиации:** National University of Singapore; Tsinghua University; University of Science and Technology of China.
>
> **Публикация:** SIGIR 2024 perspective paper. DOI: 10.1145/3626772.3657844; arXiv:2310.10108.

## 1. Коротко

Статья вводит Agent4Rec - симулятор пользователей для рекомендательных систем на базе LLM-powered generative agents. Это не generative retrieval в узком смысле “модель генерирует item ID”, а generative recommendation infrastructure: LLM-агенты имитируют поведение пользователей, взаимодействуют с recommender'ом page-by-page, ставят оценки, выходят из системы и дают объяснения.

Главный вопрос авторов: насколько LLM-агенты могут достоверно симулировать реальных автономных пользователей в recommendation loop? Ответ осторожный: агенты неплохо воспроизводят часть вкусов, рейтинговых распределений и различают алгоритмы, но имеют системные отклонения, включая склонность к false positives и слабую имитацию низких оценок.

## 2. Контекст

Обычная offline evaluation в recommendation часто плохо коррелирует с online utility. Метрики Recall/NDCG фиксируют попадание в исторический target, но не моделируют усталость пользователя, page-by-page feedback, удовлетворенность, filter bubbles и долгосрочные эффекты. Online A/B tests дороги и рискованны.

Agent4Rec предлагает промежуточный слой: симулировать пользователей с помощью LLM-агентов, и на этом sandbox'е сравнивать recommender strategies. Для GR/SID это важно не как retrieval architecture, а как способ проверять поведение генеративных/семантических retrieval систем в интерактивной среде, где один top-K список - не вся история.

## 3. Метод

Архитектура агента состоит из трех модулей:

- **Profile module.** Инициализируется из MovieLens-1M, Steam и Amazon-Book. Профиль содержит social traits: activity, conformity, diversity; и personalized tastes, извлеченные из 25 исторических item'ов пользователя через ChatGPT.
- **Memory module.** Хранит factual memories: какие item'ы были показаны, просмотрены и оценены; и emotional memories: satisfaction, fatigue, чувства после взаимодействий. Есть retrieval, writing и emotion-driven reflection.
- **Action module.** Поддерживает taste-driven actions: view/rate/feelings; и emotion-driven actions: exit, system rating, post-exit interview.

Recommendation environment реализует page-by-page сценарий: агент видит страницу рекомендаций, выбирает действия, обновляет memory, затем решает продолжать или выйти. Алгоритмы в sandbox: Random, Popularity, MF, LightGCN, MultVAE; интерфейс позволяет подключать внешние recommenders.

## 4. Эксперименты/результаты

В user taste alignment 1000 агентов получают по 20 item'ов с разным соотношением позитивных и distractor item'ов. При ratio 1:1 accuracy составляет примерно 0.69 на MovieLens, 0.72 на Amazon-Book и 0.69 на Steam. Recall остается около 0.7-0.78 при разных ratios, но precision и F1 резко падают при росте числа distractors.

В симуляции стратегий на MovieLens агенты ожидаемо выше оценивают algorithmic recommendations, чем random/popular. Например, средняя satisfaction score: Random 2.93, Popularity 3.42, MF 3.80, MultVAE 3.75, LightGCN 3.85. LightGCN также дает лучший viewing ratio и like ratio в таблице авторов.

Page-by-page augmentation показывает, что добавление просмотренных агентом фильмов как positive signal улучшает offline Recall/NDCG и simulated satisfaction для MF, MultVAE и LightGCN. Добавление unviewed item'ов, наоборот, часто ухудшает user experience. Это поддерживает идею, что действия агента несут сигнал предпочтений, но не доказывает полную человеческую реалистичность.

## 5. Ограничения

- **LLM bias и hallucination.** Агенты склонны выбирать слишком стабильное число понравившихся item'ов; precision падает при большом числе distractors.
- **Низкие рейтинги плохо моделируются.** Авторы отмечают, что агенты редко ставят 1-2, вероятно из-за prior knowledge LLM и предварительного избегания “плохих” фильмов.
- **Симуляция не заменяет online A/B.** Agent4Rec полезен для sandbox и диагностики, но не валидирует business metrics на реальных пользователях.
- **Зависимость от prompts и LLM версии.** В статье агенты powered by gpt-3.5-turbo; поведение может заметно меняться при другой модели или политике safety/alignment.

## 6. Как читать для GR/SID

Для generative retrieval эта работа полезна как evaluation-side paper. SID-based модели обычно оценивают Recall/NDCG на frozen split, но production recommendation живет в интерактивном цикле: пользователь может устать, выйти, дать негативный implicit feedback, изменить короткосрочный intent. Agent4Rec предлагает язык для таких сценариев.

Если вы проектируете GR/SID pipeline, читать стоит секции про page-by-page environment и social/emotional memory. Они помогают сформулировать проверки beyond top-K: satisfaction, exit page, feedback augmentation, filter bubble effects и объяснимость поведения пользователя.
