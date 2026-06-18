---
title: "MBGR: Multi-Business Prediction for Generative Recommendation at Meituan"
category: "generative_retrieval"
slug: "mbgr_multi_business_prediction_for_generative_recommendation_at_meituan_summary"
catalogId: "paper-mbgr_multi_business_prediction_for_generative_recommendation_at_meituan_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/mbgr_multi_business_prediction_for_generative_recommendation_at_meituan_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.02684"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Changhao Li, Junwei Yin, Zhilin Zeng, Senjie Kou, Shuli Wang, Wenshuai Chen, Yinhua Zhu, Haitao Wang, Xingxing Wang.
>
> **Аффилиации:** Meituan.
>
> **Источник:** [arXiv 2604.02684](https://arxiv.org/abs/2604.02684) · дата metadata: 2026-04-03.
>
> **Категория/теги:** generative recommendation, industrial, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: MBGR адаптирует GR к multi-business recommendation at Meituan, где NTP дает seesaw между бизнесами.
- Алгоритм: Business-aware semantic ID сохраняет domain semantics, Multi-Business Prediction дает business-specific heads/capabilities, Label Dynamic Routing превращает sparse labels в dense supervision.
- Evidence: Offline и online experiments на Meituan food delivery подтверждают gains; система deployed in production.
- Ограничение: Может быть сильная negative transfer между бизнесами; нужен контроль per-business metrics, а не только общий lift.
- Итог: Полезна для multi-scenario GR: один SID space для всех бизнесов может путать разные semantics.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>However, existing GR methods suffer from two critical issues: (1) a seesaw phenomenon in multi-business scenarios arises due to NTP's inability to capture complex cross-business behavioral patterns; and (2) a unified SID space causes representation confusion by failing to distinguish distinct semantic...</td></tr>
<tr><th>Предложение авторов</th><td>To address these issues, we propose Multi-Business Generative Recommendation (MBGR), the first GR framework tailored for multi-business scenarios.</td></tr>
<tr><th>Главный evidence claim</th><td>However, existing GR methods suffer from two critical issues: (1) a seesaw phenomenon in multi-business scenarios arises due to NTP's inability to capture complex cross-business behavioral patterns; and (2) a unified SID space causes representation confusion by failing to distinguish distinct semantic...</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>InfoNCE, contrastive, reconstruction, causal</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Meituan</td></tr>
<tr><th>Metrics</th><td>Recall</td></tr>
<tr><th>Baselines</th><td>TIGER, OneRec, HSTU, DIN, DeepFM, RQ-VAE</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Proposed Method: We propose an innovative multi-business generative recommendation system architecture that consists of three core components: 1) Business-aware semantic ID (BID) module, 2)Multi-Business Prediction (MBP) module, and 3) Label Dynamic Routing (LDR) module.
1. **Ключевой модуль.** Model Training: Our multi-business generative recommendation system is trained end-to-end using a carefully designed multi-objective loss function that combines contrastive learning with semantic reconstruction. The training process simultaneously optimizes for both accurate multi-business prediction and semantic consistency across all business domains.
1. **Learning signal.** Training Objective: The overall training objective combines InfoNCE loss for multi-business prediction with reconstruction loss for semantic preservation:

## 5. Objectives, formulas и training details

**Detected objective keywords:** InfoNCE, contrastive, reconstruction, causal.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `S_u = \s_u,1, s_u,2,..., s_u,L\`
- `P(T_u^(1:K)|S_u) = _k=1^K P(T_u^(k)|S_u, b_k)`
- `g^b = SiLu(FFN_gate(z^b)) R^K`
- `e^b = _k=1^K g_k^b FFN_k^exp(z^b) R^d_e`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- The overall architecture of MBGR.
- Hit@10 Performance Across Business Categories
- Downstream CTCVR GAUC Comparison
- Comparison of embedding distributions between sum pooling and BID encoder approaches. The left panel shows embeddings generated by simple sum pooling of token embeddings, while the right panel displays embeddings produced by our BID encoder. All embeddings are projected to 2D space using PCA for visualization. Different colors represent different business...
- Ablation Study Results on Hit@10 Performance
- Impact of Time Decay Coefficient $ $ on Hit@10
- Business Weight Configuration
- Impact of Business Weight Configuration on Hit@10

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Meituan</td></tr>
<tr><th>Metrics</th><td>Recall</td></tr>
<tr><th>Baselines</th><td>TIGER, OneRec, HSTU, DIN, DeepFM, RQ-VAE</td></tr>
</tbody></table>
</div>

- Evidence: Offline и online experiments на Meituan food delivery подтверждают gains; система deployed in production.
- However, existing GR methods suffer from two critical issues: (1) a seesaw phenomenon in multi-business scenarios arises due to NTP's inability to capture complex cross-business behavioral patterns; and (2) a unified SID space causes representation confusion by failing to distinguish distinct semantic information across...
- Extensive offline and online experiments on Meituan's food delivery platform validate MBGR's effectiveness, and we have successfully deployed it in production.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: MBGR адаптирует GR к multi-business recommendation at Meituan, где NTP дает seesaw между бизнесами.
- **Algorithmic/system:** Алгоритм: Business-aware semantic ID сохраняет domain semantics, Multi-Business Prediction дает business-specific heads/capabilities, Label Dynamic Routing превращает sparse labels в dense supervision.
- **Empirical:** Evidence: Offline и online experiments на Meituan food delivery подтверждают gains; система deployed in production.
- **Practical:** Ограничение: Может быть сильная negative transfer между бизнесами; нужен контроль per-business metrics, а не только общий lift.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Meituan.
- Метрики: Recall.
- Baselines и parity settings: TIGER, OneRec, HSTU, DIN, DeepFM, RQ-VAE.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна для multi-scenario GR: один SID space для всех бизнесов может путать разные semantics.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: MBGR адаптирует GR к multi-business recommendation at Meituan, где NTP дает seesaw между бизнесами.
- Алгоритм: Business-aware semantic ID сохраняет domain semantics, Multi-Business Prediction дает business-specific heads/capabilities, Label Dynamic Routing превращает sparse labels в dense supervision.
- Evidence: Offline и online experiments на Meituan food delivery подтверждают gains; система deployed in production.
- Ограничение: Может быть сильная negative transfer между бизнесами; нужен контроль per-business metrics, а не только общий lift.
- Итог: Полезна для multi-scenario GR: один SID space для всех бизнесов может путать разные semantics.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Internet technology has made online businesses like e-commerce platforms and social networks essential to daily life W&amp;D, deepfm, xdeepfm. Today's major platforms provide diverse businesses - Meituan, for instance, spans food delivery, entertainment and...</td></tr>
<tr><td>Related Work</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Generative Recommendations</td><td>Generative recommender systems aim to replace the traditional two-tower recall model by directly generating candidate items. They have garnered widespread attention in the industry in recent years. Early work primarily explored representing items as sequences...</td></tr>
<tr><td>Multi-business recommendations</td><td>Multi-business (scenario) recommendation systems aim to leverage data from multiple scenarios to train a unified model to improve recommendation effectiveness. Early research, starting from a causal perspective, first proposed a causal-inspired intervention...</td></tr>
<tr><td>Problem Formulation</td><td>In this section, we formally define the multi-business generative sequential recommendation problem. Given a user u 's historical interaction sequence across multiple businesses: equation equation:1 S u = s u,1, s u,2,..., s u,L equation</td></tr>
<tr><td>Proposed Method</td><td>We propose an innovative multi-business generative recommendation system architecture that consists of three core components: 1) Business-aware semantic ID (BID) module, 2)Multi-Business Prediction (MBP) module, and 3) Label Dynamic Routing (LDR) module.</td></tr>
<tr><td>Business-aware semantic ID (BID)</td><td>The Business-aware semantic ID (BID) module is a novel dual-path representation learning framework designed to address two critical challenges in multi-business recommendation: 1) incorporating business context into shared token representations, and 2)...</td></tr>
<tr><td>Multi-Business Prediction (MBP)</td><td>This architecture serves as the core generative engine that simultaneously predicts next items across multiple business domains. Built upon a Transformer-based autoregressive framework, the architecture processes user interaction sequences S u to generate...</td></tr>
<tr><td>Label Dynamic Routing</td><td>For each position t in the sequence and each business type b k, the prediction target is determined by finding the nearest future interaction of the same business type: equation i u,t+1 (k) = i u,t' where t' = t" &gt; t | b u,t" = b k equation</td></tr>
<tr><td>Model Training</td><td>Our multi-business generative recommendation system is trained end-to-end using a carefully designed multi-objective loss function that combines contrastive learning with semantic reconstruction. The training process simultaneously optimizes for both accurate...</td></tr>
<tr><td>Training Objective</td><td>The overall training objective combines InfoNCE loss for multi-business prediction with reconstruction loss for semantic preservation:</td></tr>
<tr><td>InfoNCE Loss for Multi-Business Prediction</td><td>The InfoNCE loss is computed across all business domains and token positions to optimize the model's ability to predict the next item's Semantic ID tokens for different business:</td></tr>
</tbody></table>
</div>
