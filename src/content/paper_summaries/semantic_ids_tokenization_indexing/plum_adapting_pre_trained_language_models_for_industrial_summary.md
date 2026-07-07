---
title: "PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations"
category: "semantic_ids_tokenization_indexing"
slug: "plum_adapting_pre_trained_language_models_for_industrial_summary"
catalogId: "paper-plum_adapting_pre_trained_language_models_for_industrial_summary"
paperUrl: "https://arxiv.org/abs/2510.07784"
---
> **Авторы:** Ruining He, Lukasz Heldt, Lichan Hong, Raghunandan Keshavan, Shifan Mao, Nikhil Mehta, Zhengyang Su, Alicia Tsai, Yueqi Wang, Shao-Chuan Wang, Xinyang Yi, Lexi Baugher, Baykal Cakici, Ed Chi, Cristos Goodrow, Ningren Han, He Ma, Romer Rosales, Abby Van Soest, Devansh Tandon, Su-Lin Wu, Weilong Yang, Yilin Zheng.
>
> **Аффилиации:** Google DeepMind; YouTube.
>
> **Источник:** arXiv:2510.07784, submitted 2025-10-09. Код и открытый production-датасет не опубликованы.

## 1. Коротко

PLUM - это YouTube/Google DeepMind framework для адаптации pre-trained LLM к industrial-scale recommendation. Центральная идея: уйти от large embedding tables как главного носителя памяти и представить videos через Semantic IDs, расширить LLM vocabulary новыми SID tokens, сделать Continued Pre-Training на смеси user behavior и video metadata, а потом fine-tune модель для generative retrieval.

Это важная production-работа, потому что она показывает не только "TIGER-like GR на Amazon", а запуск на YouTube surfaces с миллиардами videos/users. PLUM сравнивается с сильным LEM baseline, у которого основная емкость сидит в embedding tables с O(10M) input/output item ID vocabulary. В PLUM наоборот: около 90% параметров приходится на neural network, а не на embedding lookup.

Ключевой результат: 900M activated-parameter MoE PLUM дает гораздо больший effective vocab coverage, конкурентные offline/acceptance метрики и положительные online lifts при добавлении в candidate pool поверх усиленного LEM+ baseline. Важная методологическая часть - ablation: SIDv2, CPT и LLM initialization отдельно дают вклад, а scaling study показывает, что retrieval Recall@10 продолжает расти до диапазона MoE-900M в рассмотренных compute budgets.

<figure class="paper-figure">
  <img src="../../assets/plum/generative_retrieval.png" alt="PLUM generative retrieval prompt and SID decoding">
  <figcaption>Рисунок 1. PLUM fine-tuning превращает user context, watch history, text/custom features и SID tokens в autoregressive decoding следующего video SID.</figcaption>
</figure>

## 2. Контекст

Industrial recommenders часто масштабируются через massive sparse embedding tables: item IDs, user/item categorical features, hashing, output vocabularies. Это хорошо memorizes interactions, но плохо сочетается с LLM-style scaling, где основная емкость находится в Transformer layers и compact token vocabulary.

PLUM формулирует альтернативу: videos становятся последовательностями discrete SID tokens, а recommender становится language-model-like sequence model. Это не просто "LLM ranker": PLUM специально grounding'ит SID vocabulary через CPT, потому что обычный language checkpoint не знает ни YouTube video corpus, ни user behavior dynamics.

Для линии semantic IDs статья важна как industrial anchor: SID tokenizer - не вспомогательный preprocessing step, а address space для retrieval по billions-scale corpus. Поэтому в статье много внимания уделено uniqueness, collisions, invalid SID hallucinations, CPT sample efficiency и serving через beam search.

## 3. Метод: три стадии PLUM

### 3.1. SIDv2 tokenizer

PLUM строит Semantic ID как tuple discrete codewords из video content features. Авторы начинают от RQ-VAE/TIGER-like подхода, но вводят SIDv2:

- fused multi-modal representation: несколько video embeddings кодируются отдельными encoders, concat'ятся и проецируются в общий latent vector;
- multi-resolution codebooks: ранние levels имеют большую cardinality, последующие residual levels меньше, примерно `2048 / 2^(level - 1)`;
- progressive masking: при обучении случайно выбираются первые `r` codebook levels, чтобы hierarchy была устойчивой на разных prefix depths;
- co-occurrence contrastive regularization: behavior co-watch signal не добавляется как динамический CF embedding, а используется как contrastive objective, чтобы похожие по поведению videos получали близкие SID representations.

Итоговый tokenizer loss:

$$
\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{rq} + \mathcal{L}_{con}.
$$

Это важный design choice. Авторы избегают прямого fusion с CF item embeddings, потому что такие embeddings быстро drift'ят вместе с популярностью и потребовали бы частого retraining tokenizer/downstream model. Contrastive co-occurrence signal дает recommendation awareness, но меньше ломает стабильность content-based SID mapping.

### 3.2. Continued Pre-Training

После расширения vocabulary SID tokens модель проходит CPT на смеси:

- user behavior data: watch histories, SID tokens, channel names, watch ratio/time, time features;
- video metadata corpus: SID + title, description, ASR captions, channel name, topics и synthetic data.

Смесь 50/50 между behavior и metadata. CPT stage: 1M steps, batch size 16, около 260B tokens. Цель - научить LLM joint modeling of SIDs and text, а не просто добавить случайные новые embeddings в vocabulary.

Интересная appendix-диагностика: CPT checkpoint, initialized from LLM, сохраняет basic few-shot text ability поверх SID inputs, тогда как random-initialized CPT чаще генерирует incoherent strings. Это не главный retrieval metric, но полезный сигнал, что SID tokens действительно grounded в language space.

### 3.3. SFT для generative retrieval

Fine-tuning обучает модель предсказывать SID clicked video по user context/history. Objective - autoregressive maximum likelihood по SID tokens, с reward signal для кликов:

$$
\mathcal{L}_{SFT} = - \sum_{t=1}^{L} r(user, v_{click}) \log P(sid_t | Context_{user}, History_{user}, sid_{<t}).
$$

На практике авторы sample'ят training examples пропорционально reward и затем взвешивают одинаково, чтобы снизить cost. Inference использует beam search: generated SID sequences становятся retrieved candidates и мапятся обратно в videos. Заявленная hallucination rate после SFT низкая, меньше 5%, а SID uniqueness остается высокой.

## 4. Эксперименты и результаты

### 4.1. PLUM против LEM

В production comparison используется 900M activated-parameter Gemini-1.5 MoE PLUM, warm-started from Gemini и continuously finetuned на свежих engagement data. Baseline - top-performing Transformer-based LEM retrieval, где большинство параметров находится в embedding layer. PLUM's neural network - около 90% total parameters; у LEM neural network - около 0.4%.

<div class="table-scroll">
<table>
<thead><tr><th>Metric, ratio PLUM / LEM</th><th>LFV</th><th>Shorts</th><th>Комментарий</th></tr></thead>
<tbody>
<tr><td>Effective Vocab Size</td><td>2.60x</td><td>13.24x</td><td>PLUM покрывает гораздо более широкий set videos для 95% impressions.</td></tr>
<tr><td>CTR</td><td>1.42x</td><td>1.33x</td><td>Сильный offline/user-reaction сигнал, но это ratio, не online lift.</td></tr>
<tr><td>WT/View</td><td>0.72x</td><td>1.13x</td><td>Long-form watch time per view ниже, Shorts выше.</td></tr>
<tr><td>WF/View</td><td>1.32x</td><td>1.03x</td><td>Fraction watched лучше на обоих surfaces.</td></tr>
</tbody>
</table>
</div>

Самый сильный практический аргумент - не только CTR ratio, а effective vocab size. PLUM генерирует candidates из более широкой части corpus без large output embedding table. Это полезно для discovery и long-tail coverage, но не автоматически означает лучший final ranking.

### 4.2. Online добавление PLUM к candidate pool

Online test не заменяет весь retrieval stack. PLUM recommendations добавляются в candidate pool, а baseline - LEM+ с увеличенной квотой для лучшего production retrieval model на тот же объем.

<div class="table-scroll">
<table>
<thead><tr><th>Metric, PLUM vs LEM+</th><th>LFV</th><th>Shorts</th></tr></thead>
<tbody>
<tr><td>Engaged Users</td><td>+0.07%</td><td>+0.28%</td></tr>
<tr><td>Panel CTR</td><td>+0.76%</td><td>+4.96%</td></tr>
<tr><td>Views</td><td>+0.80%</td><td>+0.39%</td></tr>
<tr><td>Satisfaction</td><td>+0.06%</td><td>+0.39%</td></tr>
</tbody>
</table>
</div>

Авторы также подчеркивают sample efficiency: 900M MoE PLUM обучается примерно на 250M examples/day, тогда как traditional LEM - на нескольких billion examples/day. Из-за более быстрой convergence общий training FLOPs для retrieval task меньше 0.55x от LEM, хотя cost per example выше.

### 4.3. SIDv2 ablation

<div class="table-scroll">
<table>
<thead><tr><th>SID model</th><th>SID Uniqueness</th><th>VID Recall@10</th></tr></thead>
<tbody>
<tr><td>SIDv1 baseline</td><td>94.0%</td><td>12.3%</td></tr>
<tr><td>SIDv2</td><td>96.7%</td><td>14.4%</td></tr>
<tr><td>Ablate Multi-Resolution</td><td>94.8%</td><td>13.2%</td></tr>
<tr><td>Ablate Multi-Embedding</td><td>96.9%</td><td>12.8%</td></tr>
<tr><td>Ablate Co-occurrence</td><td>91.8%</td><td>12.6%</td></tr>
</tbody>
</table>
</div>

Co-occurrence ablation особенно важен: без него падает и uniqueness, и downstream Recall@10. Multi-embedding ablation показывает, что одна только высокая uniqueness не гарантирует retrieval quality: у ablated multi-embedding uniqueness даже 96.9%, но recall хуже SIDv2.

### 4.4. CPT и LLM initialization

<figure class="paper-figure">
  <img src="../../assets/plum/cpt_ablation_recall.png" alt="PLUM CPT ablation Recall@10 over SFT training steps">
  <figcaption>Рисунок 2. CPT ускоряет convergence SFT: модели с CPT быстрее достигают высокого 8th-day Recall@10.</figcaption>
</figure>

<div class="table-scroll">
<table>
<thead><tr><th>Model</th><th>Pre-trained LLM</th><th>CPT</th><th>Recall@10, Day 8</th></tr></thead>
<tbody>
<tr><td>R1</td><td>No</td><td>No</td><td>0.19</td></tr>
<tr><td>R2</td><td>Yes</td><td>No</td><td>0.23</td></tr>
<tr><td>CR1</td><td>No</td><td>Yes</td><td>0.27</td></tr>
<tr><td>CR2</td><td>Yes</td><td>Yes</td><td>0.28</td></tr>
</tbody>
</table>
</div>

Главный вклад здесь - CPT, а не только general LLM initialization. LLM init сам по себе улучшает 0.19 -> 0.23; CPT from random дает 0.27; full CPT + LLM init дает 0.28. Это поддерживает тезис, что domain-specific SID/text/behavior pretraining bridge важнее простого "возьмем Gemini и SFT".

### 4.5. Scaling

Scaling study использует MoE-110M, 370M, 900M и 3B activated parameters; total parameters от <1B до >10B. Dataset - YouTube production surface, 7 continuous days from July 2025 for training, Day 8 evaluation, input length 1536 tokens, примерно 100 recent watches plus features. Training запускали на 1024 Google v6e TPUs, 4 trainers по 256 TPUs.

<figure class="paper-figure">
  <img src="../../assets/plum/scaling_recall_budget.png" alt="PLUM evaluation Recall@10 for fixed compute budgets">
  <figcaption>Рисунок 3. Eval Recall@10 при fixed Iso-FLOPS budgets: optimal model size сдвигается к более крупным моделям, но MoE-3B не выигрывает в рассмотренном compute range.</figcaption>
</figure>

<div class="table-scroll">
<table>
<thead><tr><th>Model size</th><th>Learning rate</th><th>Global batch size</th></tr></thead>
<tbody>
<tr><td>MoE-110M</td><td>1e-4</td><td>25,600</td></tr>
<tr><td>MoE-370M</td><td>7e-5</td><td>15,360</td></tr>
<tr><td>MoE-900M</td><td>5e-5</td><td>7,680</td></tr>
<tr><td>MoE-3B</td><td>2e-5</td><td>3,584</td></tr>
</tbody>
</table>
</div>

Вывод не "больше всегда лучше". MoE-3B не обгоняет 900M на выделенных budgets, вероятно из-за batch size и data throughput: 3B обработал только 0.57 epochs, тогда как 900M - 1.22, 370M - 2.25, 110M - 4.24. Для recommendation scaling нужно синхронно масштабировать model size и number of training examples.

## 5. Сильные стороны

- Production evidence: online lifts на YouTube surfaces поверх сильного LEM+ baseline.
- Реальный ablation discipline: SIDv2, CPT, LLM initialization и model scale проверяются отдельно.
- Хорошее объяснение, зачем нужен CPT: grounding SID modality в LLM space и ускорение downstream SFT.
- Не прячет system constraints: sample efficiency, beam search, invalid SIDs, uniqueness, compute-optimal batch size.

## 6. Ограничения и риски

Почти все ключевые данные внутренние. Нельзя воспроизвести corpus scale, labels, reward sampling, production baselines, online routing и serving. Поэтому статью лучше читать как production evidence и design blueprint, а не как открытый benchmark.

SID quality все еще сильно зависит от proprietary multimodal embeddings и co-occurrence data. Неясно, насколько SIDv2 переносится на domains без богатого video metadata или с быстрым semantic drift.

Online gains представлены как добавление PLUM к существующему candidate pool, а не полная замена retrieval stack. Это честный production setting, но он не доказывает, что standalone PLUM всегда лучше LEM для всех retrieval budgets.

Beam search лучше random decoding по quality, но снижает diversity. Авторы отмечают это как trade-off, но не раскрывают подробную diversity/latency Pareto-диагностику.

## 7. Связь с соседними работами

PLUM - прямой predecessor для TokenMinds: сначала YouTube показывает SID+LLM framework для item retrieval, затем расширяет идею на user tokens and embeddings. По отношению к PinRec это другой industrial answer: Pinterest отказался от SID в пользу dense generated embeddings из-за collapse/capacity issues, а YouTube показывает, что SID path может работать при сильном tokenizer/CPT/system design.

Для semantic-ID исследований главный урок такой: tokenizer, LLM initialization и domain CPT нельзя рассматривать изолированно. SID mapping становится частью language-model vocabulary, и качество retrieval зависит от того, насколько хорошо discrete IDs grounded в content, behavior и sequence modeling.

## 8. Итог

PLUM - одна из самых важных industrial SID papers, потому что переводит semantic-ID generative recommendation из academic setup в масштаб YouTube. Основной takeaway: generative retrieval может конкурировать с large embedding retrieval, если сделать три вещи вместе - SIDv2 tokenizer с behavior-aware signal, large-scale CPT для grounding SID modality и production-minded SFT/serving. Без CPT или без сильного tokenizer это уже не тот метод.

## Источники

- [arXiv:2510.07784](https://arxiv.org/abs/2510.07784)
