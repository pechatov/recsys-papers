---
title: "DSI: Transformer Memory as a Differentiable Search Index"
category: "generative_retrieval"
slug: "dsi_transformer_memory_differentiable_search_index_summary"
catalogId: "paper-dsi_transformer_memory_differentiable_search_index_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/dsi_transformer_memory_differentiable_search_index_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2202.06991"
---
Подробное саммари статьи:

> **Авторы:** Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, Don Metzler.
>
> **Аффилиации:** Google Research; Google Brain.
>
> **Публикация:** NeurIPS 2022.

## 1. Коротко: о чем статья

DSI предлагает принципиально новую парадигму информационного поиска: вместо традиционной схемы «encode query → compare with document vectors → rank» используется единая модель-Transformer, которая по текстовому запросу напрямую *генерирует* идентификатор релевантного документа (DocID). Вся информация о корпусе закодирована в параметрах модели — нет отдельного индекса, нет векторного ANN-поиска.

Это первая работа, показавшая, что generative indexing может **превзойти** strong dense retrieval baselines (dual encoders) на хорошо изученной задаче поиска документов. Статья открыла целое направление generative retrieval (GR): большинство последующих работ — DSI++, NCI, SEAL, Ultron, TIGER, LETTER и другие — строятся поверх или в диалоге с этой концепцией.

## 2. Контекст и мотивация

Классический IR-пайплайн состоит из двух шагов: (1) offline indexing — построение индекса (BM25, dense vectors) и (2) online retrieval — поиск по индексу при поступлении запроса. Такая архитектура требует поддерживать отдельный поисковый движок, и encoder запроса не оптимизируется напрямую под конечное качество retrieval.

DSI предлагает заменить это всё одним seq2seq Transformer: модель одновременно «индексирует» (запоминает документы в параметрах) и «отвечает» на запросы, генерируя DocID. Авторы показывают, что с правильными design choices это работает лучше, чем dual encoder при сопоставимом масштабе.

## 3. Ключевые компоненты DSI

### 3.1. Дизайн DocID

Центральный вопрос DSI — как представить «адрес» документа, чтобы модель могла его генерировать. Статья исследует три подхода:

- **Unstructured atomic:** каждому документу назначается уникальное случайное целое число. Простейший вариант — но модель не имеет никаких подсказок о структуре пространства документов.
- **Naively structured string:** DocID — строка из нескольких числовых токенов (например, «123 456»). Модель генерирует их авторегрессионно.
- **Semantically structured (hierarchical):** DocID строится через иерархическую кластеризацию (k-means по embeddings): документ получает код вида «3 7 12», где первый токен — кластер верхнего уровня, следующий — подкластер и т.д. Это ключевое улучшение: похожие документы получают похожие DocID-префиксы, что облегчает обучение. Эта же идея лежит в основе Residual Quantization (RQ-VAE) в последующих системах TIGER и LETTER.

### 3.2. Стратегии индексирования

DSI обучается на двух типах задач одновременно:

- **Indexing task** — модель принимает текст документа (или его части) и генерирует соответствующий DocID. Авторы исследуют несколько вариантов: Inputs2Targets (документ → DocID), Targets2Inputs (DocID → документ), Bidirectional (оба направления), Span Corruption (BERT-style masking с последующей генерацией DocID).
- **Retrieval task** — модель принимает запрос и генерирует DocID целевого документа.

Всё обучается как единая seq2seq задача с cross-entropy loss. Модель одновременно «запоминает» корпус и учится отвечать на запросы.

### 3.3. Constrained Decoding

Чтобы модель генерировала только *валидные* DocID (из тех, что существуют в корпусе), при инференсе используется constrained beam search с prefix tree (trie). На каждом шаге генерации допустимы только токены, которые являются валидным продолжением какого-то существующего DocID. Это избавляет от галлюцинаций — генерации несуществующих документов. Тот же механизм используется в GENRE, TIGER и всех последующих системах generative retrieval.

## 4. Экспериментальные результаты

Оценка проводится на NQ (Natural Questions) в трёх режимах по размеру корпуса: NQ10K, NQ100K, NQ320K. Метрика — Hits@1 (попал ли правильный документ в топ-1 результат beam search).

<div class="table-scroll">
<table>
<thead><tr><th>Модель</th><th>NQ10K Hits@1</th><th>NQ100K Hits@1</th><th>NQ320K Hits@1</th></tr></thead>
<tbody>
<tr><td>BM25</td><td>12.4</td><td>11.2</td><td>10.4</td></tr>
<tr><td>Dual Encoder</td><td>12.4</td><td>26.7</td><td>35.6</td></tr>
<tr><td>DSI (best)</td><td><strong>33.9</strong></td><td><strong>35.6</strong></td><td><strong>38.2</strong> (11B)</td></tr>
</tbody>
</table>
</div>

Наилучшие результаты показывают semantically structured DocID в сочетании со Span Corruption индексированием. Выбор стратегии индексирования и типа DocID оказывает существенное влияние на качество.

## 5. Сильные стороны

- **Единая модель вместо сложного пайплайна.** Нет отдельного индекса, нет ANN-движка — только один Transformer. Это открывает возможность для end-to-end дифференцируемой оптимизации.
- **Semantic DocID работает.** Иерархическая кластеризация embeddings даёт DocID, структура которых помогает модели обобщаться: токены «смежных» документов похожи.
- **Масштабируемость по параметрам.** DSI выигрывает от большего числа параметров значительнее, чем BM25 или DE — перспективный сигнал для scaling.
- **Zero-shot превосходит BM25.** Даже без fine-tuning на labeled queries DSI, обученный только на корпусе, работает лучше BM25.

## 6. Ограничения

- **Масштаб корпуса.** Эксперименты ограничены 320K документами. На корпусах из миллионов документов как будет вести себя DSI — открытый вопрос (частично решённый в DSI++, IncDSI).
- **Добавление новых документов требует переобучения.** Если в корпусе появляется новый документ, нужно обновлять модель — критическая проблема для живых систем (решена в DSI++, IncDSI).
- **Стоимость инференса.** Autoregressive beam search значительно дороже ANN-поиска по dense vectors. Это основной practical bottleneck для high-QPS систем (частично решается vectorized trie decoding).
- **Memorization vs. generalization.** Часть качества DSI объясняется запоминанием тренировочных пар, а не обобщением на новые запросы.

## 7. Почему эту статью важно прочитать первой

DSI — отправная точка всего направления generative retrieval. Понять её — значит понять базовую постановку задачи: что такое DocID, зачем нужен constrained decoding, в чём принципиальное отличие от dense retrieval. Почти все последующие работы (SEAL, NCI, Ultron, TIGER, LETTER, MINDER и десятки других) либо улучшают отдельные компоненты DSI, либо применяют его идеи к новым доменам (рекомендательные системы, knowledge-intensive NLP). Без знания DSI эти работы читать значительно труднее.
