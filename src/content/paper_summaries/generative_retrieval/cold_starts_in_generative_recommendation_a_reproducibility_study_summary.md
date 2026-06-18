---
title: "Cold-Starts in Generative Recommendation: A Reproducibility Study"
category: "generative_retrieval"
slug: "cold_starts_in_generative_recommendation_a_reproducibility_study_summary"
catalogId: "paper-cold_starts_in_generative_recommendation_a_reproducibility_study_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/cold_starts_in_generative_recommendation_a_reproducibility_study_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2603.29845"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Zhen Zhang, Jujia Zhao, Xinyu Ma, Xin Xin, Maarten de Rijke, Zhaochun Ren.
>
> **Аффилиации:** Shandong University; Leiden University; Baidu Inc.; University of Amsterdam.
>
> **Источник:** [arXiv 2603.29845](https://arxiv.org/abs/2603.29845) · дата metadata: 2026-03-31.
>
> **Категория/теги:** generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://github.com/zhangzhen-research/ColdGenrec](https://github.com/zhangzhen-research/ColdGenrec) [https://huggingface.co/docs/transformers/en/model_doc/flan-t5](https://huggingface.co/docs/transformers/en/model_doc/flan-t5) [https://github.com/westlake-repl/MicroLens](https://github.com/westlake-repl/MicroLens) [https://github.com/kang205/SASRec](https://github.com/kang205/SASRec) [https://github.com/RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole).

## 1. Коротко

- Главная идея: репродуцирует cold-start claims generative recommendation under unified protocols.
- Алгоритм: Авторы отделяют model scale, identifier design и training strategy, рассматривая user и item cold-start как primary evaluation settings.
- Evidence: Работа показывает, какие cold-start gains действительно устойчивы, а какие зависят от protocol/metadata.
- Ограничение: Это reproducibility study: ценность в протоколе, не в новом model component.
- Итог: Важна как проверка популярного тезиса, что PLM/semantic GR автоматически решает cold-start.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Cold-start recommendation remains a central challenge in dynamic, open-world platforms, requiring models to recommend for newly registered users (user cold-start) and to recommend newly introduced items to existing users (item cold-start) under sparse or missing interaction signals.</td></tr>
<tr><th>Предложение авторов</th><td>In this work, we present a systematic reproducibility study of generative recommendation under a unified suite of cold-start protocols.</td></tr>
<tr><th>Главный evidence claim</th><td>However, cold-start is rarely treated as a primary evaluation setting in existing studies, and reported gains are difficult to interpret because key design choices, such as model scale, identifier design, and training strategy, are frequently changed together.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>reinforcement</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source</td></tr>
<tr><th>Metrics</th><td>метрики нужно сверить в таблицах experiments</td></tr>
<tr><th>Baselines</th><td>TIGER, P5, DPR, DSI, NCI, SEAL, MINDER, LC-Rec, BM25</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Abstract: Cold-start recommendation remains a central challenge in dynamic, open-world platforms, requiring models to recommend for newly registered users (user cold-start) and to recommend newly introduced items to existing users (item cold-start) under sparse or missing interaction signals.
1. **Ключевой модуль.** Abstract: Recent generative recommenders built on pre-trained language models (PLMs) are often expected to mitigate cold-start by using item semantic information (e.g., titles and descriptions) and test-time conditioning on limited user context.
1. **Learning signal.** Abstract: However, cold-start is rarely treated as a primary evaluation setting in existing studies, and reported gains are difficult to interpret because key design choices, such as model scale, identifier design, and training strategy, are frequently changed together.
1. **Inference / deployment path.** Abstract: In this work, we present a systematic reproducibility study of generative recommendation under a unified suite of cold-start protocols.

## 5. Objectives, formulas и training details

**Detected objective keywords:** reinforcement.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `L_ gen = - _t=1^|y_k+1| P(y_k+1^(t) y_k+1^(<t), H_u;),`
- `i = _j I P(y_j H_u).`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Comparison of generative recommendation methods. Enc--Dec denotes encoder--decoder architecture, and Dec denotes decoder-only architecture.
- User cold-start recommendation performance comparison. Performance is measured using Recall@10 (R@10) and NDCG@10 (N@10), with the best and second-best results in each category bolded and underlined, respectively. The best performing models achieve significant improvements over the other baselines (paired t-test, p<0.05).
- Item cold-start recommendation performance comparison. Performance is measured using Recall@10 (R@10) and NDCG@10 (N@10), with the best and second-best results in each category bolded and underlined, respectively. The best performing models achieve significant improvements over the other baselines (paired t-test, p<0.05).
- Main recommendation performance comparison of various methods under warm-start, user cold-start, and item cold-start settings on three datasets.
- Performance impact of RL on Amazon-Toys measured by Recall@10. The redred numbers in parentheses represent the percentage change ($ $) relative to the SFT baseline.
- Statistics of the datasets used in the experiments. "W-" and "C-" denote Warm and Cold, respectively. "Inter." stands for the number of interactions.
- Comparison of recall performance under warm-start and cold-start conditions across different model scales. The plot shows Recall@10 for both item and user cold-start settings as model size increases, using representative generative recommender methods (TIGER) with different variants of Flan-T5.
- Performance comparison across different identifier designs. This figure illustrates the Recall@10 performance under warm-start, item cold-start, and user cold-start conditions for various identifier types: Atomic IDs, Textual Titles, and Semantic Codes (RQ-VAE, Balanced k-means, and OPQ).

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source</td></tr>
<tr><th>Metrics</th><td>метрики нужно сверить в таблицах experiments</td></tr>
<tr><th>Baselines</th><td>TIGER, P5, DPR, DSI, NCI, SEAL, MINDER, LC-Rec, BM25</td></tr>
</tbody></table>
</div>

- Evidence: Работа показывает, какие cold-start gains действительно устойчивы, а какие зависят от protocol/metadata.
- However, cold-start is rarely treated as a primary evaluation setting in existing studies, and reported gains are difficult to interpret because key design choices, such as model scale, identifier design, and training strategy, are frequently changed together.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: репродуцирует cold-start claims generative recommendation under unified protocols.
- **Algorithmic/system:** Алгоритм: Авторы отделяют model scale, identifier design и training strategy, рассматривая user и item cold-start как primary evaluation settings.
- **Empirical:** Evidence: Работа показывает, какие cold-start gains действительно устойчивы, а какие зависят от protocol/metadata.
- **Practical:** Ограничение: Это reproducibility study: ценность в протоколе, не в новом model component.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.
- Reward/utility signal достаточно стабилен и не подменяет user relevance узкой бизнес-метрикой.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source.
- Метрики: метрики нужно сверить в таблицах experiments.
- Baselines и parity settings: TIGER, P5, DPR, DSI, NCI, SEAL, MINDER, LC-Rec, BM25.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Важна как проверка популярного тезиса, что PLM/semantic GR автоматически решает cold-start.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: репродуцирует cold-start claims generative recommendation under unified protocols.
- Алгоритм: Авторы отделяют model scale, identifier design и training strategy, рассматривая user и item cold-start как primary evaluation settings.
- Evidence: Работа показывает, какие cold-start gains действительно устойчивы, а какие зависят от protocol/metadata.
- Ограничение: Это reproducibility study: ценность в протоколе, не в новом model component.
- Итог: Важна как проверка популярного тезиса, что PLM/semantic GR автоматически решает cold-start.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Resources</td><td>We share the following resources to facilitate reproducibility of our work at https://github.com/zhangzhen-research/ColdGenrec: itemize Source code for the reproduced models and evaluation pipeline. Scripts for dataset preprocessing and cold-start split...</td></tr>
<tr><td>Introduction</td><td>Recommender systems are a core component of two-sided marketplaces, matching users with items in dynamic, open-world environments fu2026differentiable,lin2025order. A fundamental challenge in these settings is cold-start on both sides: recommending relevant...</td></tr>
<tr><td>Cold-starts in generative recommendation</td><td>Despite these hypotheses, cold-starts are seldom treated as a primary evaluation setting for generative recommendation, leaving open how these models behave when users or items have little to no interaction history. Moreover, existing generative...</td></tr>
<tr><td>Research questions</td><td>This gap motivates a systematic reproducibility study that isolates and evaluates the effects of key confounding factors under a unified set of cold-start settings. Accordingly, we conduct a reproducibility study guided by four research questions: enumerate*...</td></tr>
<tr><td>Findings</td><td>Our experiments provide the following key findings in response to the above research questions: enumerate* [label=( )] The cold-start problem exhibits highly asymmetric behavior: while user cold-starts lead to only moderate performance degradation, item...</td></tr>
<tr><td>Contributions</td><td>Our main contributions are as follows: enumerate* [label=( )] We conduct a comprehensive evaluation of representative generative recommenders under a unified suite of user and item cold-start settings, providing a controlled assessment of their cold-start...</td></tr>
<tr><td>Related Work</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Generative retrieval</td><td>With the rapid advancement of large language models (LLMs) vaswani2017attention, zhao2023survey, generative retrieval has emerged as a paradigm shift in information retrieval. Unlike the traditional "index-retrieve-rank" pipeline that relies on sparse or...</td></tr>
<tr><td>Generative recommendation</td><td>Recent advances have reformulated recommendation as a generative task, where models directly generate item identifiers conditioned on user interaction history and context deng2025onerec,zhao2025model,liu2024multi. We organize recent generative recommenders...</td></tr>
<tr><td>Cold-start recommendations</td><td>The cold-start problem, which refers to making recommendations for new items or users with limited interaction history, remains a core challenge for recommender systems yuan2023user,du2022metakg,xu2024cmclrec. For item cold-starts, common approaches include...</td></tr>
<tr><td>Problem Formulation</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Generative recommendation</td><td>In generative recommendation, we formalize recommendation as a generative task, where the model autoregressively predicts the tokenized identifier of the next item conditioned on a user's interaction history. Let U and I denote the sets of users and items,...</td></tr>
</tbody></table>
</div>
