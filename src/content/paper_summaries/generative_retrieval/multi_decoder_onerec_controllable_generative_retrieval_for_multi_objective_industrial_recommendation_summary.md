---
title: "Multi-Decoder OneRec: Controllable Generative Retrieval for Multi-Objective Industrial Recommendation"
category: "generative_retrieval"
slug: "multi_decoder_onerec_controllable_generative_retrieval_for_multi_objective_industrial_recommendation_summary"
catalogId: "paper-multi_decoder_onerec_controllable_generative_retrieval_for_multi_objective_industrial_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2607.26500"
---
> **Авторы:** You Wang, Zhao Liu, Guoping Tang, Yiqing Yang, Shuo Su, Jing Liu, Naifu Zhou, Xiaoyou Zhou, Wei Jiang, Jian Liang, Xiao Lv, Ruiming Tang, Liyin Hong, Wenwu Ou.
>
> **Аффилиации:** Kuaishou Technology.
>
> **Источник:** arXiv:2607.26500v1 от 2026-07-29; venue на arXiv не указан. Авторы открыли [Kwai26 Data Pipeline](https://github.com/liuzhao09/Kwai26-Data-Pipeline), но на момент проверки repository содержит data card и scaffold: executable pipeline, predefined splits и data artifacts ещё не опубликованы.

## 1. Коротко

Multi-Decoder OneRec решает production-проблему, которую single-decoder generative retrieval обычно прячет. Candidate pool должен одновременно покрывать exposure, long-view/watch-time, likes/shares и cold-start, причём product team хочет явно задавать quota каждого route. Один общий decoder унифицирует model stack, но связывает objectives общими weights и часто генерирует одинаковые high-probability items для разных целей. Независимые retrievers дают контроль, но возвращают fragmentation training и serving.

Решение Kuaishou - shared user-context encoder и General Decoder плюс lightweight objective-specific decoders. Каждый objective добавляет собственные LoRA adapters, BOS embedding и residual к shared SID embedding; gradients objective route не меняют shared base или другие experts. Discrete events обучаются filtered NTP/SFT, watch time - relative-reward policy optimization с KL anchor к General Decoder. На inference explicit quotas и Multi-Decoder Constrained Beam Search не дают поздним routes повторять SIDs, уже занятые ранними routes.

На новом Kwai26 benchmark с 25.03M retrievable items метод улучшает OneRec на 1.69-5.62% relative по четырём Recall@512 metrics. В 7-day A/B test на Kwai Brazil reported lifts составляют +0.37% usage time/device, +0.19% Day-7 retention и +2.09% Cold-Start. Цена: только +20% parameters, но 2.23x inference FLOPs для четырёх routes. Это не «один decoder вместо многих» в compute sense; это shared parameterization с несколькими decoder executions.

<figure class="paper-figure">
  <img src="../../assets/multi_decoder_onerec/framework.png" alt="Multi-Decoder OneRec shared architecture isolated training and coordinated inference">
  <figcaption>Рисунок 1. Shared encoder/base decoder, gradient-isolated LoRA experts и quota-aware Multi-Decoder Constrained Beam Search.</figcaption>
</figure>

## 2. Проблема: unified, но controllable retrieval

Traditional multi-route candidate generation независимо обучает retrievers и выделяет каждому quota. Это хорошо контролирует composition, но число models, data flows и serving paths растёт вместе с objectives.

Single-decoder GR использует общий SID vocabulary и backbone. Objective prompt/BOS token меняет conditioning, но core attention/FFN transformations остаются общими. Update одного objective может ухудшать другие, а separate prompted runs всё равно концентрируются в одних SID regions. После post-hoc dedup реальный pool оказывается меньше budget.

Авторы требуют одновременно:

- reuse user representation и general SID prior;
- изолировать objective updates без full decoder copy;
- поддержать heterogeneous supervision: events и continuous reward;
- получить complementary candidates с явными quotas под fixed budget $B$.

## 3. Архитектура и метод

### 3.1. Shared user context

Encoder получает две time-truncated history views: 20 последних interacted videos и 256 последних Long-View videos. Каждый position содержит Item-ID, author ID, tag, time difference и watch time. Четыре blocks чередуют feature-level cross-attention и sequence-level self-attention. Полученные states $H_u$ вычисляются один раз и переиспользуются всеми routes.

### 3.2. Objective-specific decoder state

General Decoder - полноразмерный Transformer decoder, обученный на exposure targets. Для objective $t$ к query/key/value projections masked self-attention и cross-attention добавляются LoRA updates:

$$
W^{(t)} = W_0 + B^{(t)}A^{(t)}.
$$

Кроме LoRA route имеет собственный BOS $b_t$ и residual $\Delta E_t$ к shared SID embedding. Полное task state:

$$
\phi_t = \{A^{(t)}, B^{(t)}, b_t, \Delta E_t\}.
$$

Это сильнее Multi-BOS conditioning: objective меняет internal transformations и SID token representations, но не копирует весь decoder.

## 4. Objective-specific training

General Decoder обучается exposure next-token prediction и единственный обновляет shared parameters $\theta_0$:

$$
\mathcal{L}_{base} = -\sum_{\ell=0}^{L-1}\log p_{gen}(y_\ell \mid c,y_{<\ell}).
$$

Long-View и Like experts используют ту же NTP loss только на samples с соответствующим behavior flag. Их gradient идёт исключительно в $\phi_t$.

Для watch time thresholded SFT теряет magnitude positive feedback. Поэтому reward стандартизируется относительно последних $K$ valid watch-time events пользователя:

$$
r_i = \frac{w_i - \mu(\mathcal{G}_i)}{\sigma(\mathcal{G}_i)+\epsilon}.
$$

Watch-time expert оптимизируется L-GBPO по relative reward. Stop-gradient forward General Decoder задаёт reference distribution, а observed-token KL surrogate не позволяет policy уйти из valid SID prior. Без KL SID legal rate и recall collapse; лучший tested weight равен 1.0.

Общая gradient-routed objective выглядит как base loss плюс SFT/RL losses, где $\theta_0$ передаётся через stop-gradient. Training concurrent, то есть isolation обеспечивается routing gradients, а не последовательным freezing checkpoints.

<figure class="paper-figure">
  <img src="../../assets/multi_decoder_onerec/kl_weight_sweep.png" alt="Recall and SID legal rate for different KL weights">
  <figcaption>Рисунок 2. KL anchor к General Decoder критичен: при нулевом весе падают и recall, и SID legal rate.</figcaption>
</figure>

## 5. Multi-Decoder Constrained Beam Search

Routes запускаются в predefined priority order. У route $r$ есть beam size $b_r$ и output quota $q_r$, причём $\sum q_r=B$. После каждого route accepted SID prefixes добавляются в mask для следующих routes. General Decoder идёт последним и backfill'ит оставшийся budget.

Constraint можно наложить на любой уровень hierarchical SID. У SIDs три levels. L2 CBS запрещает весь occupied second-level prefix и слишком рано отсекает другие items из той же semantic region. L3 CBS маскирует только complete SID и оказывается лучшим вариантом.

Default offline allocation для budget 512:

- quotas Long-View / Like / Watch-time / General: 86 / 85 / 85 / 256;
- beams: 86 / 171 / 256 / 512;
- CBS на третьем, финальном SID level.

## 6. Kwai26

Dataset построен из 60 дней short-video traffic с 2 мая по 30 июня 2026 года. Истории строго обрезаются до request time; последний valid session каждого из 50,000 users используется как test, все предыдущие - train.

<div class="table-scroll">
<table>
<thead><tr><th>Statistic</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Raw item-level records</td><td>1,311,923,604</td></tr>
<tr><td>Positive-play interactions</td><td>821,842,758</td></tr>
<tr><td>Training / test sessions</td><td>125,261,311 / 50,000</td></tr>
<tr><td>Item-ID vocabulary</td><td>31,854,181</td></tr>
<tr><td>Items with valid SIDs</td><td>25,028,687</td></tr>
<tr><td>SID levels / codes per level</td><td>3 / 8,192</td></tr>
<tr><td>Recent / Long-View history</td><td>20 / 256</td></tr>
</tbody>
</table>
</div>

SIDs построены RQ-KMeans. В catalog есть 27.14M valid Item-ID-SID pairs и 23.91M codes; 1.98M items имеют несколько SIDs. Evaluation micro-averages target items в 50,000 test sessions.

Важно для reproducibility: статья говорит «publicly release Kwai26», но public GitHub пока явно называет себя release preview. В нём нет raw/processed dataset, executable processing code, model implementation и лицензии; всё это заявлено как planned после anonymization, privacy review и reproducibility checks.

## 7. Offline results

<div class="table-scroll">
<table>
<thead><tr><th>Model</th><th>Exposure</th><th>Long-View</th><th>Like</th><th>WT Recall</th></tr></thead>
<tbody>
<tr><td>HSTU</td><td>0.0342</td><td>0.0469</td><td>0.0387</td><td>0.0506</td></tr>
<tr><td>TIGER</td><td>0.1380</td><td>0.1613</td><td>0.1161</td><td>0.1660</td></tr>
<tr><td>OneRec</td><td>0.1539</td><td>0.1833</td><td>0.1318</td><td>0.1923</td></tr>
<tr><td><strong>Multi-Decoder OneRec</strong></td><td><strong>0.1565</strong></td><td><strong>0.1907</strong></td><td><strong>0.1391</strong></td><td><strong>0.2031</strong></td></tr>
</tbody>
</table>
</div>

Relative gains над OneRec: +1.69% Exposure, +4.04% Long-View, +5.54% Like и +5.62% Watch-time Recall@512. Все методы используют один split/history budget; ANN baselines ищут по full catalog, GR models используют общий SID lexicon.

### 7.1. Specialization и aggregation

Каждый single expert максимизирует собственную метрику, но теряет по другим. Long-View Decoder достигает Long-View 0.1956, Watch-time Decoder - WT 0.2041, однако их Exposure падает до 0.1403 и 0.1219. Quota union даёт более balanced point и обгоняет Multi-BOS по всем metrics.

### 7.2. Deduplication действительно возвращает budget

<div class="table-scroll">
<table>
<thead><tr><th>Method</th><th>Exposure</th><th>Long-View</th><th>Like</th><th>WT</th><th># Candidates</th></tr></thead>
<tbody>
<tr><td>L3 CBS</td><td>0.1565</td><td>0.1907</td><td>0.1391</td><td>0.2031</td><td>512</td></tr>
<tr><td>L2 CBS</td><td>0.1522</td><td>0.1827</td><td>0.1350</td><td>0.1942</td><td>512</td></tr>
<tr><td>No CBS</td><td>0.1258</td><td>0.1544</td><td>0.1122</td><td>0.1647</td><td>207.67</td></tr>
</tbody>
</table>
</div>

No-CBS routes тратят больше половины nominal pool на overlap. L3 лучше L2 при одинаковых 512 candidates, значит выигрыш связан не только с count: shallow prefix mask преждевременно запрещает полезные соседние items.

### 7.3. Continuous reward

Watch-time SFT с 12-second threshold даёт WT Recall 0.1999, continuous-reward RL - 0.2031. Увеличение reward history с $K=8$ до $K=500$ даёт 0.2003 -> 0.2031. Default LoRA rank 32 лучший среди 4, 16, 32 и 64.

## 8. Cost и quota control

Три LoRA experts увеличивают parameters с 7.93M до 9.50M, то есть на 20%. Training FLOPs растут 14.93G -> 22.97G (1.54x), inference - 325.02G -> 725.38G (2.23x). Shared encoder экономит representation compute, но каждый route всё равно требует decoder search.

Quota shifts действительно управляют target mix, но из-за fixed budget создают trade-off. Больше WT quota даёт +0.31% WT Recall, иногда снижая другие metrics. Увеличение всех beams до 512 улучшает все metrics, но требует дополнительного compute.

<figure class="paper-figure">
  <img src="../../assets/multi_decoder_onerec/quota_beam_scaling.png" alt="Recall changes after route quota shifts and beam scaling">
  <figcaption>Рисунок 3. Quota перераспределяет utility между objectives, а увеличение beam улучшает все routes ценой compute.</figcaption>
</figure>

## 9. Production A/B test

7-day test проведён на Kwai Brazil. Randomization unit - device ID. Control: 14.64% devices с single-decoder OneRec; treatment: 7.32% с Watch-time, Share и Cold-Start decoders. Generative retrieval candidates дают 57% всех impressions, downstream ranking одинаков.

<div class="table-scroll">
<table>
<thead><tr><th>Metric</th><th>Relative change</th></tr></thead>
<tbody>
<tr><td>Usage time per device</td><td>+0.37%</td></tr>
<tr><td>Day-1 / Day-7 retained users</td><td>+0.12% / +0.19%</td></tr>
<tr><td>Share / Like devices</td><td>+0.19% / +0.35%</td></tr>
<tr><td>Comment / Follow devices</td><td>+0.52% / +0.51%</td></tr>
<tr><td>Cold-Start</td><td>+2.09%</td></tr>
</tbody>
</table>
</div>

Все reported outcomes имеют $p<0.05$. Source diagnostics также согласуются со specialization: Share Decoder имеет лучший FTR 0.0077, Cold-Start Decoder - 95.48% cold-start rate, Watch-time Decoder - лучший average watch time 26.36.

## 10. Сильные стороны

- Решает реальный control problem: explicit quotas остаются first-class serving primitive.
- Хорошо разделены parameter sharing, gradient isolation и candidate complementarity; ablations проверяют каждую часть.
- Один из редких industrial GR papers с детальным dataset construction и leakage-safe chronological protocol.
- Offline gains подтверждены production A/B test, а compute overhead опубликован явно.

## 11. Ограничения и вопросы

Kwai26 пока нельзя реально воспроизвести: public repository содержит спецификацию и placeholders, но не данные, pipeline или training implementation. Формулировка «publicly release» опережает фактический release state.

Strongest baseline OneRec разделяет с методом encoder и General Decoder, что делает ablation чистым. Но сравнение с modern multi-task discriminative/multi-route industrial retrievers ограничено: DSSM/SASRec/HSTU как single ANN routes не воспроизводят полный quota-aware production ensemble.

MD-CBS зависит от priority order. Paper показывает quota perturbations, но не systematic search по route ordering и fairness между low-priority objectives. General Decoder всегда идёт последним, поэтому его distribution получает остаточное пространство.

Parameter efficiency не равна serving efficiency: четыре routes стоят 2.23x FLOPs. Нет latency, accelerator utilization, tail latency и quality-per-cost Pareto для production serving.

A/B test длится семь дней и сообщает много outcomes с отдельным $p<0.05$, но не описывает correction for multiple online metrics, confidence intervals или long-term novelty effects.

## 12. Итог

Multi-Decoder OneRec - убедительная architecture paper о том, как сохранить route quotas внутри generative retrieval. Главный takeaway: objective-specific LoRA experts нужны для разных policies, gradient isolation - для защиты shared prior, а coordinated constrained decoding - чтобы specialization превратилась в дополнительные candidates, а не в дубликаты. Evidence сильное благодаря offline ablations и A/B test, но открытая reproducibility пока остаётся обещанием: Kwai26 release ещё не завершён.

## Источники

- [arXiv:2607.26500](https://arxiv.org/abs/2607.26500)
- [Kwai26 Data Pipeline](https://github.com/liuzhao09/Kwai26-Data-Pipeline)
