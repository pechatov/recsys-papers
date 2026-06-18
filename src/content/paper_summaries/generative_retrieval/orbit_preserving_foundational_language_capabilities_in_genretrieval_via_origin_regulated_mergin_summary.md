---
title: "ORBIT: Preserving Foundational Language Capabilities in GenRetrieval via Origin-Regulated Merging"
category: "generative_retrieval"
slug: "orbit_preserving_foundational_language_capabilities_in_genretrieval_via_origin_regulated_mergin_summary"
catalogId: "paper-orbit_preserving_foundational_language_capabilities_in_genretrieval_via_origin_regulated_mergin_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/orbit_preserving_foundational_language_capabilities_in_genretrieval_via_origin_regulated_mergin_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.12419"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Neha Verma, Nikhil Mehta, Shao-Chuan Wang, Naijing Zhang, Alicia Tsai, Li Wei, Lukasz Heldt, Lichan Hong, Ed Chi, Xinyang Yi.
>
> **Аффилиации:** Johns Hopkins University; Google DeepMind; Google.
>
> **Источник:** [arXiv 2605.12419](https://arxiv.org/abs/2605.12419) · дата metadata: 2026-05-12.
>
> **Категория/теги:** efficiency, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: ORBIT борется с catastrophic forgetting LLM abilities при fine-tuning for GenRetrieval.
- Алгоритм: Метод отслеживает distance от исходных weights и при превышении threshold делает origin-regulated weight averaging, ограничивая drift.
- Evidence: ORBIT лучше continual-learning/regularization baselines сохраняет и language, и retrieval performance.
- Ограничение: Баланс между retrieval specialization и сохранением language capability зависит от threshold и base model.
- Итог: Полезна для LLM-GR: retrieval fine-tuning может разрушать foundation capabilities, особенно если модель нужна еще и для language reasoning.

**Как читать статью:** это прежде всего работа типа *benchmark/reproducibility/theory*; поэтому основной audit должен смотреть на protocol validity, leakage, dataset schema, assumptions и baseline fairness.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>This work investigates and addresses this challenge in the context of the Generative Retrieval (GenRetrieval) task.</td></tr>
<tr><th>Предложение авторов</th><td>Given these observations, we propose ORBIT, a novel approach that actively tracks the distance between fine-tuned and initial model weights, and uses a weight averaging strategy to constrain model drift during GenRetrieval fine-tuning when this inter-model distance exceeds a maximum threshold.</td></tr>
<tr><th>Главный evidence claim</th><td>Our results show that ORBIT retains substantial text and retrieval performance by outperforming both common continual learning baselines and related regularization methods that also employ weight averaging.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Теоретический результат полезен как sanity check, но assumptions нужно явно сопоставлять с реальными tokenizer/decoding constraints.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>benchmark/reproducibility/theory</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>objective не выражен стандартным ключевым словом; смотреть method/training sections</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Amazon, Beauty, Sports, Toys</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, Success, F1</td></tr>
<tr><th>Baselines</th><td>CoST, RQ-VAE</td></tr>
<tr><th>Ключевое предположение</th><td>Assumptions теоремы или benchmark protocol должны совпадать с реальным tokenizer/decoding setup.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Algorithm: Given the early and rapid degradation of text performance during fine-tuning, we are interested in a method that intervenes quickly during this process. While other methods that also employ merging during fine-tuning may be able to intervene quickly, these methods generally use a fixed cadence throughout training, which may not be the optimal schedule for...
1. **Ключевой модуль.** Algorithm: Our method is simple yet surprisingly effective: for a given metric, we fix a maximum distance allowable between the original model parameters, which serves as our origin, and the model parameters. When a training step causes the current model parameters to exceed this distance, we average the original parameters with the offending current model parameters...
1. **Learning signal.** Algorithm: algorithm: Origin-Regulated Back-merging of iterative trajectories algorithmic [1] Input: init, T,, d(, ) t in 1,...,T 0.5cm t+1 * = t - t L task Optimizer Update 0.5cm while d( t+1 *, init ) > then 0.5cm t+1 * = t+1 * + init 2 Back-merging 0.5cm t+1 = t+1 * return T algorithmic algorithm
1. **Inference / deployment path.** Why use inter-model distance?: While averaging as a regularization tool is inspired by prior work kleiman2025soup, marczak2024magmax, marouf2024weighted, the use of inter-model distance to schedule averaging steps is an important and novel distinction in our work. Our motivation to use distance to dictate averaging steps comes from 1) a longstanding notion of model knowledge...
1. **Проверяемая деталь.** Why use inter-model distance?: In testing the performance of Soup-to-Go in our setting, we observe that across checkpoints saved throughout training, there is a correlation between text performance and distance from the initial parameters. This observation suggests that fine-tuning that pushes the model starkly away from its starting point can induce severe forgetting. As a result, we...

## 5. Objectives, formulas и training details

**Detected objective keywords:** objective не выражен стандартным ключевым словом; смотреть method/training sections.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `T' = T - T_minT_max - T_min and R' = R - R_minR_max - R_min`
- `M() \;=\; 12 (+ _ init),`
- `_k \;=\; 12^k\, _0 \;+\; (1 - 12^k)\, _ init.`
- `r_i \;=\; | _0(i)|| _ init(i)|.`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- An overview of our method. During the fine-tuning of the downstream task, inter-model distance is tracked; when this distance exceeds a threshold $ $, weight averaging is used as a regularization step to reduce the forgetting of parametric knowledge from $ _ init$.
- Text datasets used for evaluating language and reasoning capabilities.
- Text benchmark evaluations before and after fine-tuning Gemma3-1B on the Amazon Review Sports and Outdoors dataset using GenRetrieval. After fine-tuning, performance drops significantly.
- Recall@10 on the Amazon Review Toys and Games dataset and average text performance across our 8 benchmarks during the first 10k steps of baseline GenRetrieval fine-tuning.
- Quantitative analysis measuring forgetting during GenRetrieval finetuning.
- Average text accuracy and Recall@5 performance across post-hoc, one-round weight interpolations.
- : Origin-Regulated Back-merging of iterative trajectories
- A scatter plot demonstrating the correlation between sign dissimilarity (SD) and average text performance. Points are collected from a Soup-to-Go experiment with a cadence of 1000 steps.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Amazon, Beauty, Sports, Toys</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, Success, F1</td></tr>
<tr><th>Baselines</th><td>CoST, RQ-VAE</td></tr>
</tbody></table>
</div>

- Evidence: ORBIT лучше continual-learning/regularization baselines сохраняет и language, и retrieval performance.
- Text evaluation: To evaluate the language capabilities of our GenRetrieval models, we use the benchmarks from Table config, as described with their evaluation settings. These benchmarks reflect a mix of scoring and sampling-based evaluations in order to cover a breadth of capabilities. We report the average.
- Results: We evaluate our baselines and for GenRetrieval fine-tuning across two primary objectives: text-based reasoning and sequential recommendation. We report average text performance, NDCG@10, and Recall@10. To identify techniques that perform well in both domains, we employ Pareto efficiency principles.
- Results: Because Pareto-optimal sets often contain multiple solutions, we introduce a Distance To Ideal Point (DTIP) metric to select a single, balanced model. DTIP measures the normalized Euclidean distance between a model's performance as represented by a tuple, and an "ideal" point representing the theoretical maximum in both domains. For a given model, let T...
- Our results show that ORBIT retains substantial text and retrieval performance by outperforming both common continual learning baselines and related regularization methods that also employ weight averaging.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: ORBIT борется с catastrophic forgetting LLM abilities при fine-tuning for GenRetrieval.
- **Algorithmic/system:** Алгоритм: Метод отслеживает distance от исходных weights и при превышении threshold делает origin-regulated weight averaging, ограничивая drift.
- **Empirical:** Evidence: ORBIT лучше continual-learning/regularization baselines сохраняет и language, и retrieval performance.
- **Practical:** Ограничение: Баланс между retrieval specialization и сохранением language capability зависит от threshold и base model.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Система ускоряет inference, но не улучшает модельное качество сама по себе; важно проверять stale-cache и quality-latency frontier.
- Assumptions теоремы или benchmark protocol должны совпадать с реальным tokenizer/decoding setup.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Amazon, Beauty, Sports, Toys.
- Метрики: Recall, NDCG, Success, F1.
- Baselines и parity settings: CoST, RQ-VAE.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна для LLM-GR: retrieval fine-tuning может разрушать foundation capabilities, особенно если модель нужна еще и для language reasoning.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: ORBIT борется с catastrophic forgetting LLM abilities при fine-tuning for GenRetrieval.
- Алгоритм: Метод отслеживает distance от исходных weights и при превышении threshold делает origin-regulated weight averaging, ограничивая drift.
- Evidence: ORBIT лучше continual-learning/regularization baselines сохраняет и language, и retrieval performance.
- Ограничение: Баланс между retrieval specialization и сохранением language capability зависит от threshold и base model.
- Итог: Полезна для LLM-GR: retrieval fine-tuning может разрушать foundation capabilities, особенно если модель нужна еще и для language reasoning.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>The Generative Retrieval (GenRetrieval) paradigm rajput2023genretrieval, tay2022transformer has demonstrated considerable efficacy in sequential recommendation tasks. GenRetrieval introduces items or queries as tokenized ID sequences enabling an...</td></tr>
<tr><td>Related Work</td><td>Model merging refers to a set of techniques that combine the capabilities of two or more models by combining their parameters directly in weight space. wortsman2022robust introduce a simple method to reduce the forgetting in a pre-trained model after...</td></tr>
<tr><td>Background and Motivation</td><td>GenRetrieval task In a GenRetrieval recommendation system, the sequential recommendation task is converted to an autoregressive generation problem via framing a user's item history as a context, and predicting the next item as its completion. Prior work has...</td></tr>
<tr><td></td><td> In this section, we first define preliminaries necessary to define our method, followed by a definition of our method and observations that motivate its design.</td></tr>
<tr><td>Preliminaries</td><td>Notation init R p refers to initial LLM parameters before GenRetrieval fine-tuning. current R p refers to parameters at the current fine-tuning step. d is a distance function between two sets of parameters, is a scalar denoting a maximum distance, and T is...</td></tr>
<tr><td>Algorithm</td><td>Given the early and rapid degradation of text performance during fine-tuning, we are interested in a method that intervenes quickly during this process. While other methods that also employ merging during fine-tuning may be able to intervene quickly, these...</td></tr>
<tr><td>Why use inter-model distance?</td><td>While averaging as a regularization tool is inspired by prior work kleiman2025soup, marczak2024magmax, marouf2024weighted, the use of inter-model distance to schedule averaging steps is an important and novel distinction in our work. Our motivation to use...</td></tr>
<tr><td>Experimental Setup</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>GenRetrieval Fine-tuning</td><td>For fine-tuning our GenRetrieval models, we use with Gemma3 team2025gemma as our base model. Gemma3 is released under the Gemma Terms of Use. We use the instruction-tuned version as it more closely matches the capabilities we are interested in preserving...</td></tr>
<tr><td>Baselines</td><td>Simple baselines We include no-intervention baselines, as well as L2-weight decay to reflect a baseline from traditional continual learning literature. Other techniques in this space, like Elastic Weight Consolidation and data replay methods, require data...</td></tr>
<tr><td>Text evaluation</td><td>To evaluate the language capabilities of our GenRetrieval models, we use the benchmarks from Table config, as described with their evaluation settings. These benchmarks reflect a mix of scoring and sampling-based evaluations in order to cover a...</td></tr>
<tr><td>Results</td><td>We evaluate our baselines and for GenRetrieval fine-tuning across two primary objectives: text-based reasoning and sequential recommendation. We report average text performance, NDCG@10, and Recall@10. To identify techniques that perform well in both domains,...</td></tr>
</tbody></table>
</div>
