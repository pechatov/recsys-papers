---
title: "From Matching to Generation: A Survey on Generative Information Retrieval"
category: "generative_retrieval"
slug: "from_matching_to_generation_survey_summary"
catalogId: "paper-from_matching_to_generation_survey_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/from_matching_to_generation_survey_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2404.14851"
---
Подробное саммари статьи:

> **Авторы:** Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian Zhang, Yutao Zhu, Zhicheng Dou.
>
> **Аффилиации:** Renmin University of China (RUC); Tsinghua University.
>
> **Публикация:** ACM Transactions on Information Systems (TOIS), 2025.

## 1. Коротко: о чем статья

Лучший на сегодня систематический обзор всего направления **generative information retrieval (GenIR)**. Охватывает два главных направления:

- **Generative Document Retrieval (GDR)** — модель параметрически запоминает документы и генерирует их DocID в ответ на запрос. Сюда относятся DSI, GENRE, SEAL, NCI, Ultron, TIGER, LETTER и десятки других работ.
- **Reliable Response Generation** — модель генерирует ответ напрямую, опираясь на внутренние знания или внешние источники. Сюда относятся RAG, системы с цитированием источников, персональные ассистенты.

Обзор опубликован в TOIS 2025, охватывает более 150 работ и сопровождается [официальным GitHub-репозиторием](https://github.com/RUC-NLPIR/GenIR-Survey) с актуальным reading list.

## 2. Контекст

До 2022 года dominance в information retrieval принадлежала двум парадигмам: sparse retrieval (BM25, TF-IDF) и dense retrieval (dual encoders, ColBERT). Generative retrieval, начавшееся с DSI (2022) и GENRE (2021), предложило третью парадигму, быстро выросшую в самостоятельное направление с сотнями работ. К моменту публикации обзора стало очевидно, что GenIR делится на качественно разные ветки, требующие раздельного рассмотрения. Более ранние обзоры (Survey of GIR, 2024) охватывали только часть пространства; данный обзор — первый, систематически покрывающий все ветки включая рекомендательные системы.

## 3. Ключевые компоненты

### 3.1. Generative Document Retrieval (GDR)

Авторы структурируют GDR по нескольким ортогональным осям:

- **DocID Design:**
  - *Atomic IDs* — случайные числа (DSI baseline). Простейший вариант, плохо масштабируется.
  - *Hierarchical IDs* — иерархическая кластеризация (DSI semantically structured, NCI, Ultron). Модель может использовать префиксы как «грубый адрес».
  - *N-gram / Substring IDs* — фрагменты самих документов (SEAL, GENRE). Максимальная семантическая насыщенность.
  - *Semantic IDs (quantization-based)* — коды из RQ-VAE или аналогов (TIGER, LETTER, DSI++). Компромисс между компактностью и семантикой.
- **Training strategies:** indexing task vs retrieval task, совместное обучение, различные виды augmentation.
- **Constrained Decoding:** trie (GENRE, DSI), FM-Index (SEAL), vectorized trie для ускорения инференса.
- **Incremental Learning:** проблема добавления новых документов без полного переобучения — DSI++, IncDSI, подходы на основе continual learning.
- **Downstream Task Adaptation:** перенос GDR на knowledge-intensive NLP, open-domain QA, entity linking.
- **Multimodal GR:** документы с изображениями, аудио, видео — ранние работы по generative retrieval для мультимодальных корпусов.

### 3.2. Generative Recommendation

Обзор отдельно рассматривает применение GDR к рекомендательным системам: TIGER (первая промышленная система generative recommendation с semantic IDs), LETTER (learnable tokenization), семейство работ по дизайну semantic item IDs. Авторы показывают, что задача рекомендаций отличается от document retrieval: items не имеют текстового содержания в явном виде, history-conditioned generation сложнее, а corpus динамичен (новые items появляются постоянно).

### 3.3. Reliable Response Generation

Вторая половина обзора покрывает:

- **Internal Knowledge Memorization:** как LLM «запоминают» факты и когда это надёжно.
- **Retrieval-Augmented Generation (RAG):** таксономия RAG-систем, анализ trade-off между параметрической памятью и внешним retrieval.
- **Response with Citations:** системы, генерирующие ответы с явными ссылками на источники — верификация и grounding.
- **Personal Information Assistant:** персонализированные системы, использующие пользовательский контекст.

### 3.4. Evaluation

Авторы систематизируют evaluation benchmarks по каждому направлению:

<div class="table-scroll">
<table>
<thead><tr><th>Направление</th><th>Основные датасеты</th><th>Метрики</th></tr></thead>
<tbody>
<tr><td>Document Retrieval</td><td>NQ, MS MARCO, KILT, TriviaQA, FEVER</td><td>Recall@K, MRR, NDCG</td></tr>
<tr><td>Generative Recommendation</td><td>Amazon Reviews, MovieLens, Yelp</td><td>Recall@K, NDCG@K, Hit Rate</td></tr>
<tr><td>RAG / Response Generation</td><td>NQ, TriviaQA, WebQ, PopQA</td><td>EM, F1, Faithfulness</td></tr>
</tbody>
</table>
</div>

## 4. Эксперименты / Coverage

Обзор не содержит собственных экспериментов, но систематически сопоставляет результаты из более чем 150 работ. Ключевые выводы авторов:

- Semantic / hierarchical DocID стабильно превосходят atomic IDs на всех масштабах корпуса.
- Constrained decoding обязателен для практического использования GR — без него качество падает драматически из-за галлюцинаций.
- Проблема динамических корпусов (incremental learning) остаётся открытой: ни одно из текущих решений не обеспечивает zero-cost добавления новых документов.
- GR для рекомендаций находится в активной фазе развития: большинство работ датируются 2023–2024 годами.

## 5. Сильные стороны

- **Самый полный охват на 2025 год.** Единственный обзор, систематически покрывающий GDR, GenRec и Response Generation в единой таксономии.
- **Чёткая таксономия DocID Design.** Классификация типов идентификаторов — один из самых ценных вкладов: позволяет быстро ориентироваться в литературе.
- **GitHub с living reading list.** Репозиторий [RUC-NLPIR/GenIR-Survey](https://github.com/RUC-NLPIR/GenIR-Survey) обновляется авторами — редкое качество для академического обзора.
- **Inclusion рекомендательных систем.** Рассмотрение GenRec как части GenIR, а не как отдельного направления, помогает понять общность underlying principles.

## 6. Ограничения

- **Глубина vs. широта.** При охвате 150+ работ обзор неизбежно поверхностен в деталях: для глубокого понимания конкретного метода нужно читать оригинальную статью.
- **Быстро устаревает.** Поле генеративного IR развивается очень быстро; обзор 2024–2025 года уже может не включать новейшие работы (хотя GitHub частично компенсирует это).
- **Раздел по рекомендациям менее детален.** Глубина анализа GDR значительно превосходит глубину анализа GenRec — специалистам по рекомендательным системам стоит дополнительно читать профильные обзоры (например, обзоры по semantic ID design).

## 7. Почему важно прочитать

Данный обзор — лучшая точка входа для построения полной карты литературы по generative retrieval. В отличие от более ранних обзоров, он охватывает все ветки направления, включая recommender systems GR, и имеет официальный GitHub с актуальным reading list. Рекомендуется читать *после* DSI и TIGER — когда уже понятна базовая постановка задачи — чтобы увидеть полную картину: куда развивается направление, какие проблемы остаются открытыми (scalability, incremental learning, identifier design, hallucination) и какие подходы уже апробированы промышленностью. Особенно ценна таксономия DocID Design: она сразу объясняет, почему в литературе так много работ о «семантических идентификаторах» и как они все соотносятся друг с другом.
