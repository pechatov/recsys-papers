---
title: "TurboGR: An Accelerated Training System for Large-Scale Generative Recommendation"
category: "generative_retrieval"
slug: "turbogr_an_accelerated_training_system_for_large_scale_generative_recommendation_summary"
catalogId: "paper-turbogr_an_accelerated_training_system_for_large_scale_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/turbogr_an_accelerated_training_system_for_large_scale_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.13433"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Huichao Chai, Zhixin Wu, Xuemiao Li, Shiqing Fan, Hengfeng Wang, Maojun Peng, Lu Xu, Yaoyuan Wang, Yibo Jin, Wei Guo, Yongxiang Feng.
>
> **Аффилиации:** Huawei.
>
> **Источник:** [arXiv 2605.13433](https://arxiv.org/abs/2605.13433) · дата metadata: 2026-05-13.
>
> **Категория/теги:** generative recommendation, efficiency, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: TurboGR - system paper для ускоренного обучения large-scale GR на Ascend NPUs.
- Алгоритм: Система оптимизирует jagged operators, sparse distributed communication, semi-async training, pipeline orchestration и negative sampling через offloading/quantization/logit sharing.
- Evidence: На KuaiRand-27K и производственных сценариях авторы заявляют high NPU utilization и масштабируемое обучение.
- Ограничение: Это infrastructure-specific работа: выгоды зависят от Ascend NPU и irregular sparse workload.
- Итог: Полезна для понимания training bottlenecks GR: модельные идеи бесполезны, если jagged sequence/operators плохо ложатся на hardware.

**Как читать статью:** это прежде всего работа типа *serving/decoding/system efficiency*; поэтому основной audit должен смотреть на latency, throughput, memory footprint, cache hit ratio, training cost и отсутствие деградации item-level quality.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>However, deploying GR at scale on Ascend NPUs faces fundamental system-level challenges.</td></tr>
<tr><th>Предложение авторов</th><td>In this paper, we present \model, an Ascend-affinity training system for generative recommendation that systematically addresses these bottlenecks through three core innovations: (i) Ascend-affinity jagged acceleration, including fusion operators that eliminate padding redundancy and dynamic load balancing that reduces...</td></tr>
<tr><th>Главный evidence claim</th><td>Generative recommendation (GR) has emerged as a promising paradigm that replaces fragmented, scenario-specific architectures with unified Transformer-based models, exhibiting scaling-law behavior where recommendation quality improves systematically with increased model capacity and training data.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Ускорение не является улучшением качества модели само по себе; нужно проверять quality-latency frontier и stale-cache effects.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>serving/decoding/system efficiency</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>contrastive, masking, negative sampling</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>KuaiRand</td></tr>
<tr><th>Metrics</th><td>Recall, revenue, latency, throughput</td></tr>
<tr><th>Baselines</th><td>HSTU</td></tr>
<tr><th>Ключевое предположение</th><td>Workload/decoding setup должен иметь достаточную locality или parallelism, а ускорение не должно менять candidate distribution.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** System Overview and Architecture: The overall architecture of is sketched in Designed for large-scale recommendation workloads, the system is organized into modular layers that optimize performance from infrastructure perspectives. The architecture of adopts a highly customizable design capable of conducting universal recommendation tasks including retrieval and...
1. **Ключевой модуль.** System Overview and Architecture: Accelerated Operators: This layer provides a suite of custom, high-performance operators specifically designed for Ascend NPUs, bypassing the limitations of standard operator libraries. These include jagged fusion operators for attention and relative attention bias (RAB) computation that eliminate padding redundancy, jagged embedding lookup kernels that...
1. **Learning signal.** System Overview and Architecture: CANN: CANN (Compute Architecture for Neural Networks) provides the low-level, full-stack hardware and software structure specifically designed to interface directly with the Ascend platform, handling bottom-level scheduling and memory allocation. itemize
1. **Inference / deployment path.** System Deployment Workflow: is architected for seamless industrial deployment and extensive customization. The training workflow proceeds in three stages: (1) Data Pipeline Configuration, where raw interaction logs are preprocessed into sequential sparse inputs with distributed loading optimized to eliminate I/O bottlenecks; (2) Model and Topology Instantiation, where hardware...
1. **Проверяемая деталь.** End-to-End System Efficiency: We evaluated the training efficiency of the system under varying computational workloads. As presented in Table performance metrics, the cluster's MFU systematically increases as the model capacity scales up. The gap in MFU across different model sizes demonstrates that larger architectures better exploit NPU tensor core utilization and...
1. **Проверяемая деталь.** End-to-End System Efficiency: Furthermore, the evaluations of HSTU-long and FuXi-long indicate that longer sequences consistently yield higher MFU, directly attributable to the system's end-to-end redundancy reduction and length-adaptive acceleration capabilities. For both the large and long variants, the cluster linearity strictly exceeds 0.9, as their sustained computational workloads...
1. **Проверяемая деталь.** System Optimizations: optimization Compared to DLRMs, GR poses significant training challenges due to the tight coupling of large-scale sparse embeddings and deep dense architectures. We formally characterize these challenges as follows.

## 5. Objectives, formulas и training details

**Detected objective keywords:** contrastive, masking, negative sampling.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `Loss = - ((l_i^+) (l_i^+) + _j=1^R (o_i^ n_i,j) +)`
- `1T _t=0^T-1 E[\| f(w_t)\|^2] O (L T + LT + L T),`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Overall Architectural Design of
- Training performance of HSTU and FuXi on KuaiRand-27K (Ascend 910B1 64GB).
- Jagged fusion operators (a) and their efficiency comparison with baseline (b).
- Jagged embedding lookup acceleration.
- Jagged embedding lookup latency comparison.
- Training timeline of the baseline (a) and the dynamic jagged load balancing (b).
- Token-aware dynamic batch scaling and global token reallocation.
- Performance comparison with and without the load balancing optimization.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>KuaiRand</td></tr>
<tr><th>Metrics</th><td>Recall, revenue, latency, throughput</td></tr>
<tr><th>Baselines</th><td>HSTU</td></tr>
</tbody></table>
</div>

- Evidence: На KuaiRand-27K и производственных сценариях авторы заявляют high NPU utilization и масштабируемое обучение.
- Experimental Settings:.1 All experiments are conducted on an Ascend 910B1 64GB NPU cluster (scaling from 32 to 128 NPUs across 4 to 16 nodes) powered by Kunpeng-920 ARM CPUs. We utilize the KuaiRand-27K dataset, applying a 5-core filtering mechanism to eliminate noisy interactions and employing a chronological leave-one-out strategy for the train-test split. The models...
- Generative recommendation (GR) has emerged as a promising paradigm that replaces fragmented, scenario-specific architectures with unified Transformer-based models, exhibiting scaling-law behavior where recommendation quality improves systematically with increased model capacity and training data.
- In this paper, we present, an Ascend-affinity training system for generative recommendation that systematically addresses these bottlenecks through three core innovations: (i) Ascend-affinity jagged acceleration, including fusion operators that eliminate padding redundancy and dynamic load balancing that reduces inter-device...
- Evaluated on the KuaiRand-27K dataset, supports training at up to 0.2B parameters and achieves 54.71\

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: TurboGR - system paper для ускоренного обучения large-scale GR на Ascend NPUs.
- **Algorithmic/system:** Алгоритм: Система оптимизирует jagged operators, sparse distributed communication, semi-async training, pipeline orchestration и negative sampling через offloading/quantization/logit sharing.
- **Empirical:** Evidence: На KuaiRand-27K и производственных сценариях авторы заявляют high NPU utilization и масштабируемое обучение.
- **Practical:** Ограничение: Это infrastructure-specific работа: выгоды зависят от Ascend NPU и irregular sparse workload.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Система ускоряет inference, но не улучшает модельное качество сама по себе; важно проверять stale-cache и quality-latency frontier.
- Workload/decoding setup должен иметь достаточную locality или parallelism, а ускорение не должно менять candidate distribution.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: KuaiRand.
- Метрики: Recall, revenue, latency, throughput.
- Baselines и parity settings: HSTU.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна для понимания training bottlenecks GR: модельные идеи бесполезны, если jagged sequence/operators плохо ложатся на hardware.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: TurboGR - system paper для ускоренного обучения large-scale GR на Ascend NPUs.
- Алгоритм: Система оптимизирует jagged operators, sparse distributed communication, semi-async training, pipeline orchestration и negative sampling через offloading/quantization/logit sharing.
- Evidence: На KuaiRand-27K и производственных сценариях авторы заявляют high NPU utilization и масштабируемое обучение.
- Ограничение: Это infrastructure-specific работа: выгоды зависят от Ascend NPU и irregular sparse workload.
- Итог: Полезна для понимания training bottlenecks GR: модельные идеи бесполезны, если jagged sequence/operators плохо ложатся на hardware.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Recommendation systems now operate as a core computational module in large-scale digital platforms, delivering personalized experiences to billions of users across tens of millions of items wu2023personalized, sarwar2000analysis, zannettou2024analyzing....</td></tr>
<tr><td>System Overview and Architecture</td><td>The overall architecture of is sketched in Designed for large-scale recommendation workloads, the system is organized into modular layers that optimize performance from infrastructure perspectives. The architecture of adopts a highly...</td></tr>
<tr><td>System Evaluation</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>System Deployment Workflow</td><td>is architected for seamless industrial deployment and extensive customization. The training workflow proceeds in three stages: (1) Data Pipeline Configuration, where raw interaction logs are preprocessed into sequential sparse inputs with distributed loading...</td></tr>
<tr><td>Experimental Settings</td><td>.1 All experiments are conducted on an Ascend 910B1 64GB NPU cluster (scaling from 32 to 128 NPUs across 4 to 16 nodes) powered by Kunpeng-920 ARM CPUs. We utilize the KuaiRand-27K dataset, applying a 5-core filtering mechanism to eliminate noisy...</td></tr>
<tr><td>End-to-End System Efficiency</td><td>We evaluated the training efficiency of the system under varying computational workloads. As presented in Table performance metrics, the cluster's MFU systematically increases as the model capacity scales up. The gap in MFU across different...</td></tr>
<tr><td>System Optimizations</td><td> optimization Compared to DLRMs, GR poses significant training challenges due to the tight coupling of large-scale sparse embeddings and deep dense architectures. We formally characterize these challenges as follows.</td></tr>
<tr><td>Ascend-affinity Jagged Acceleration</td><td>During distributed training of GR models, significant variation in the number of tokens per instance and high sparsity in embeddings pose two fundamental challenges. As a consequence, distributed training suffers from redundant computation, suboptimal memory...</td></tr>
<tr><td>Distributed Communication Optimization</td><td>In distributed GR training, scaling to larger clusters improves computational efficiency but simultaneously incurs substantial communication overhead. As shown in Figure Communication time varies (a), distributed communication has emerged as the...</td></tr>
<tr><td>Negative Sampling Optimization</td><td>In GR training for recall tasks, each positive sample is paired with a fixed number of independently sampled negatives, forming a contrastive loss that discriminates positives from negatives. However, scaling up negative sampling creates severe system...</td></tr>
<tr><td>Conclusion</td><td>We have presented, an Ascend-affinity training system for generative recommendation that systematically addresses the three fundamental bottlenecks arising when deploying GR models on NPU clusters: jagged computation redundancy, sparse-dense communication...</td></tr>
<tr><td>Acknowledgments</td><td>текст не извлечен; смотреть PDF/source</td></tr>
</tbody></table>
</div>
