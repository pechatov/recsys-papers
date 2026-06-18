---
title: "Generative Retrieval Meets Multi-Graded Relevance"
category: "generative_retrieval"
slug: "generative_retrieval_meets_multi_graded_relevance_summary"
catalogId: "paper-generative_retrieval_meets_multi_graded_relevance_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/generative_retrieval_meets_multi_graded_relevance_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2409.18409"
---
Подробное саммари статьи:

> **Авторы:** Yubao Tang, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Xueqi Cheng.
>
> **Аффилиации:** CAS Key Lab of Network Data Science and Technology, ICT, CAS; University of Chinese Academy of Sciences; University of Amsterdam.
>
> **Публикация:** arXiv:2409.18409, 2024.

## 1. Коротко

Статья вводит GR2 - GRaded Generative Retrieval, расширение generative retrieval на multi-graded relevance. Большинство GR-моделей учится на бинарных парах “query -> relevant docid” и оптимизирует MLE по generated identifier. Но в real IR один query может иметь документы с разными relevance grades, и простое превращение всех positive docs в одинаковые targets теряет порядок важности.

GR2 решает две связанные проблемы: как сделать docids одновременно релевантными и различимыми, и как обучить seq2seq GR-модель учитывать отношения между relevance grades. Для этого авторы предлагают regularized fusion для построения docids и multi-graded constrained contrastive loss (MGCC) для обучения retrieval model.

## 2. Контекст

Generative retrieval обычно строит encoder-decoder модель, которая по query генерирует docid. В DSI/NCI/SEAL/TIGER-like подходах качество сильно зависит от identifier design: docid должен быть коротким, валидным, семантически полезным и достаточно уникальным. Для binary relevance это уже сложно, но multi-graded setup добавляет новый constraint: документы разных grades должны быть упорядочены.

Naive pairwise ranking по likelihood generated docids плохо работает, потому что docids имеют разную длину: вероятность длинной последовательности может быть ниже просто из-за произведения token probabilities. Поэтому авторы переводят query и docid в representation space и задают contrastive constraints там, а не сравнивают raw generation likelihood.

## 3. Метод

### 3.1. Relevant and distinct docids

Docid строится как pseudo-query, сгенерированный по документу query generation model. Чтобы pseudo-queries не слипались у похожих документов, вводится regularized fusion: совместно обучаются QG model и autoencoder с общим decoder'ом.

- **Relevance regularization.** Representation документа от QG encoder притягивается к representation соответствующего pseudo-query от AE encoder и отталкивается от чужих pseudo-queries.
- **Distinctness regularization.** Представления разных документов и разных docids раздвигаются, при этом document и его docid остаются близкими.
- **Generation.** На inference вокруг document representation сэмплируются latent vectors, затем beam search генерирует pseudo-queries; при duplicates берутся следующие beam candidates, пока docids не станут уникальными.

### 3.2. MGCC loss

После построения docids GR-модель обучается не только MLE по query-docid и document-docid парам, но и multi-graded constrained contrastive loss. Для каждой пары query-docid вычисляются representations из encoder/decoder hidden states. Positive pairs разных grades притягиваются с разной силой: более высокий grade получает больший penalty coefficient.

Дополнительно вводится grade constraint: loss для более релевантного grade не должен быть хуже, чем loss для менее релевантного. Для binary relevance MGCC вырождается в supervised contrastive loss, поэтому framework применим и к обычным бинарным датасетам.

## 4. Эксперименты/результаты

Multi-graded datasets: Gov2, ClueWeb09-B, Robust04. Binary datasets: MS MARCO Document Ranking и NQ 320K. Для больших корпусов авторы используют 500K subsets: Gov 500K, ClueWeb 500K, MS 500K. Backbone для GR2 и GR baselines - T5-base.

<div class="table-scroll">
<table>
<thead><tr><th>Сценарий</th><th>Результат</th><th>Вывод</th></tr></thead>
<tbody>
<tr><td>Multi-graded relevance</td><td>GR<sup>2P</sup> на Gov 500K: P@20 0.5506 против RIPOR 0.4831; nDCG@20 0.4912 против 0.4578.</td><td>Pretraining + MGCC дает заметный выигрыш над GR baseline.</td></tr>
<tr><td>ClueWeb 500K</td><td>GR<sup>2P</sup>: nDCG@20 0.2969; лучший среди таблицы.</td><td>Метод работает не только на одном корпусе.</td></tr>
<tr><td>Binary relevance</td><td>На MS 500K GR<sup>2P</sup>: Hits@1 0.3821 против RIPOR 0.3421; Hits@10 0.6405 против 0.5873.</td><td>Framework не ломается при переходе к binary setup.</td></tr>
</tbody>
</table>
</div>

Авторы также показывают, что GR2P обычно лучше GR2S: pre-training на сконструированных Wikipedia multi-grade pairs помогает, особенно когда downstream labels ограничены.

## 5. Ограничения

- **Docid design все еще отдельный этап.** Regularized fusion улучшает identifiers, но не делает docid learning полностью end-to-end с retrieval task.
- **Сложность pipeline.** Нужны QG model, AE model, GR model, pretraining/fine-tuning варианты и несколько loss components.
- **Multi-grade labels дороги.** Основное преимущество метода раскрывается там, где есть graded relevance judgments; в recommender datasets такие labels часто косвенные или шумные.
- **Длина и качество pseudo-query docids.** Natural-language docids интерпретируемы, но могут быть длиннее и менее стабильны, чем short semantic IDs.

## 6. Как читать для GR/SID

Эта работа особенно полезна для SID-исследований как reminder: identifier должен не просто “указывать” на item/document, а поддерживать ranking semantics. Если несколько item'ов релевантны одному user/query с разной силой, SID/tokenization должен помогать различать grades, а не только собрать positives в один semantic bucket.

Для recommendation можно адаптировать идею MGCC к implicit graded signals: purchase vs click vs view, dwell time buckets, rating levels. Важно, что авторы обходят проблему сравнения raw generation likelihood и переносят ranking constraints в representation space.
