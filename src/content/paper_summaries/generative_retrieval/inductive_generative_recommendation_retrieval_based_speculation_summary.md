---
title: "Inductive Generative Recommendation via Retrieval-based Speculation"
category: "generative_retrieval"
slug: "inductive_generative_recommendation_retrieval_based_speculation_summary"
catalogId: "paper-inductive_generative_recommendation_retrieval_based_speculation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/inductive_generative_recommendation_retrieval_based_speculation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2410.02939"
---
Подробное саммари статьи:

> **Авторы:** Yijie Ding, Jiacheng Li, Julian McAuley, Yupeng Hou.
>
> **Аффилиации:** University of California, San Diego.
>
> **Публикация:** arXiv:2410.02939, 2024; в LaTeX source используется AAAI 2026 template. Код: github.com/Jamesding000/SpecGR.

## 1. Коротко

SpecGR атакует важную слабость semantic-ID generative recommendation: несмотря на ожидание, что GR сможет генерировать unseen item'ы по семантике, на практике TIGER-like модели почти всегда генерируют semantic ID patterns, виденные при обучении. На timestamp split, где в validation/test появляются новые item'ы, unseen performance у GR может быть около нуля.

Идея SpecGR - не заставлять GR самому придумывать unseen semantic IDs, а использовать его как verifier. Индуктивная retrieval model сначала предлагает candidates, включая новые item'ы; затем GR-модель оценивает, принять или отвергнуть эти candidates по likelihood target semantic ID. Это похоже на speculative decoding, но цель не ускорение, а добавление inductive capability.

## 2. Контекст

Классические ID-based sequential recommenders вроде SASRec трансдуктивны: новый item без обученного ID embedding нельзя рекомендовать напрямую. Semantic-ID GR вроде TIGER выглядит более индуктивным, потому что новый item можно токенизировать в SID по текстовым признакам. Но autoregressive decoder обучался на распределении seen SID sequences и редко попадает в новые комбинации токенов.

До SpecGR часть работ добавляла unseen candidates через blending с non-GR retrievers. Авторы считают это suboptimal: если просто смешать списки, GR не использует свою сильную ranking ability. SpecGR сохраняет GR как финальный scorer, но расширяет candidate source через inductive drafter.

## 3. Метод

Framework состоит из четырех шагов:

- **Inductive drafting.** Drafter D(X) предлагает delta candidates по той же user history. Drafter не обязан быть GR; в SpecGR Aux это UniSRec, а в SpecGR++ - собственный encoder GR-модели.
- **Target-aware verifying.** GR получает candidate item и считает normalized log-likelihood его semantic ID как target. Для unseen item'ов последний identification token исключается, потому что он не является семантическим и находится вне training distribution.
- **Guided re-drafting.** Если принято меньше K candidates, GR генерирует beam prefixes, а drafter в следующем раунде предлагает candidates только с совпадающими SID prefixes.
- **Adaptive exiting.** Когда принято K item'ов, loop завершается; если после всех шагов кандидатов не хватает, список дополняется beam outputs.

SpecGR++ дополнительно обучает GR encoder как inductive retriever: item SID и user sequence кодируются тем же encoder'ом, representations получают через mean pooling, затем обучаются contrastive objective sequence-next-item и learning-to-rank fine-tuning.

## 4. Эксперименты/результаты

Датасеты: Amazon Reviews 2023 categories Video Games, Office Products, Cell Phones and Accessories. Split по timestamp cut-offs, поэтому validation/test естественно содержат unseen items. Метрики: Recall@10/50 и NDCG@10/50; отдельно анализируются In-Sample и Unseen subsets.

<div class="table-scroll">
<table>
<thead><tr><th>Датасет</th><th>Лучший SpecGR результат</th><th>Что показывает</th></tr></thead>
<tbody>
<tr><td>Games</td><td>SpecGR<sub>Aux</sub>: R@50 0.0778, N@50 0.0239; +5.13% R@50 и +9.72% N@50 к лучшему baseline.</td><td>GR-verification сохраняет сильный overall ranking.</td></tr>
<tr><td>Office</td><td>SpecGR<sub>Aux</sub>: R@50 0.0360, N@50 0.0119.</td><td>Выигрыш умеренный, но устойчивый.</td></tr>
<tr><td>Phones</td><td>SpecGR<sub>Aux</sub>: R@50 0.0285, N@50 0.0090; +20.64% R@50 и +14.80% N@50.</td><td>Сильнее всего помогает на sparse/inductive-heavy setting.</td></tr>
</tbody>
</table>
</div>

Subset analysis важен: TIGER имеет сильный in-sample score, но не генерирует unseen items; LIGER лучше на unseen через retrieval blending, но хуже балансирует overall качество. SpecGR принимает unseen candidates через drafter, но ранжирует их GR likelihood'ом, поэтому компромисс лучше.

Ablation показывает, что нужны все части: inductive drafting, likelihood score adjustment, guided re-drafting, verification-based re-ranking и adaptive exiting. Без contrastive pretraining или fine-tuning SpecGR++ заметно слабее.

## 5. Ограничения

- **Это framework поверх GR, а не исправление decoder'а.** Базовая неспособность decoder'а генерировать unseen SID patterns сохраняется; ее обходят drafting'ом.
- **Нужен хороший drafter.** Если inductive retriever плохо находит fresh candidates, verifier не спасет recall.
- **Hyperparameter trade-offs.** Draft size delta, threshold gamma и beam size beta меняют баланс между unseen coverage, in-sample quality и latency.
- **Semantic ID assumptions.** Score adjustment полагается на то, что последний token - identification token; для других SID schemes это придется адаптировать.

## 6. Как читать для GR/SID

SpecGR - одна из самых практичных работ для dynamic catalog. Она показывает, что “новый item можно токенизировать” не равно “decoder будет его генерировать”. Для SID-систем это обязательная проверка: измерять seen/unseen separately и смотреть likelihood bias на unseen token sequences.

Архитектурно это хороший шаблон hybrid GR: semantic-ID decoder остается сильным ranker/verifier, а candidate generation для fresh items делается retrieval-based. Такой дизайн ближе к production retrieve-then-rank, но сохраняет преимущества GR scoring.
