---
title: "Gryphon-v2: One Model in Place of a Cascade - Generate-and-Rank Recommender with Rollout Distillation"
category: "generative_retrieval"
slug: "gryphon_v2_one_model_in_place_of_a_cascade_generate_and_rank_recommender_with_rollout_distillation_summary"
catalogId: "paper-gryphon_v2_one_model_in_place_of_a_cascade_generate_and_rank_recommender_with_rollout_distillation_summary"
paperUrl: "https://arxiv.org/abs/2608.06213"
---
> **Авторы:** Anna Lipkina, Daria Tikhonovich, Viktor Yanush, Mariia Ulianova, Oleg Sorokin, Vladislav Dodonov, Ilya Murzin, Denis Burshtein, Nikolay Savushkin.
>
> **Аффилиации:** Yandex, Moscow, Russia.
>
> **Источник:** arXiv:2608.06213v1 от 2026-08-06. Публичные код, данные и модель не заявлены.

## 1. Коротко: о чем статья

Gryphon-v2 продолжает идею Gryphon: Semantic-ID decoder используется как high-recall proposal mechanism, а конкретные items после раскрытия SID collision groups переупорядочиваются легким item-level Ranking Module. Обе ветки используют один encoder user history. Главное изменение v2 - Ranking Module больше не обучается обычному next-item prediction, а дистиллирует multi-objective оценки большого Teacher Ranker, который достаточно силен для улучшения production ranker, но слишком дорог для полного online serving.

Авторы предлагают **Rollout Distillation**. Teacher оценивает два типа candidates: текущие beam-search rollouts самого decoder-а и logged impressions production-системы. Rollouts приближают train distribution к candidates, которые student увидит на serving; impressions расширяют покрытие объектами, реально показанными пользователям. Это supervised distillation: policy gradients не используются, через beam search градиент не проходит, а decoder по-прежнему обучается next-token prediction.

Главный результат - industrial online A/B на Yandex Music. Один Gryphon-v2 заменяет всю production cascade из 15+ candidate generators, отдельного pre-ranker-а и final ranker-а. При сопоставимой end-to-end latency treatment дает +1.41% active users, +1.62% total listening time, +7.12% likes, +15.25% команд Repeat и -9.65% unfinished-track ratio; для всех опубликованных дельт указано $p \lt 0.001$.

<figure class="paper-figure">
  <img src="../../assets/gryphon_v2/framework.png" alt="Gryphon-v2 serving path and rollout distillation training scheme">
  <figcaption>Рисунок 1. На serving history кодируется один раз, decoder генерирует SIDs, а Ranking Module сортирует раскрытые items. Teacher Ranker существует только в training path и размечает rollout candidates вместе с logged impressions.</figcaption>
</figure>

## 2. Что изменилось по сравнению с Gryphon

Исходный Gryphon уже отделял SID generation от item-level scoring, но его Ranking Module обучался next-item supervision и отдавал candidates в неизмененный production final ranker. Поэтому он заменял candidate generators и pre-ranking, но не всю каскадную систему.

Gryphon-v2 делает два дополнительных шага:

- переносит fine-grained multi-objective preferences большого Teacher Ranker в компактный Ranking Module;
- использует этот модуль для финального online ordering, полностью убирая downstream learned ranker из treatment path.

Таким образом, название «one model in place of a cascade» относится к served graph, а не ко всему жизненному циклу. Offline Teacher Ranker, Semantic-ID tokenizer, подготовка features и непрерывное обновление student-а остаются отдельными компонентами training/data pipeline.

## 3. Метод: архитектура generate-and-rank

### 3.1. Shared history encoder

Для request $u$ история $H_u$ из событий до request-time cutoff один раз преобразуется bidirectional Transformer-ом в contextual states:

$$
E_u = \operatorname{Encode}_{\mathrm{hist}}(H_u).
$$

Эти states одновременно получает autoregressive decoder и Ranking Module. Served model имеет 7 encoder layers, 2 decoder layers, hidden dimension 1024, 16 attention heads, максимум 2048 history events и около 0.5B parameters.

### 3.2. SID generation и collision resolution

Каждому item назначается SID из трех hierarchical codebooks размером 32,000. Multimodal item representation строится из text и audio через Qwen2.5-Omni, небольшой projection Transformer и collaborative alignment; затем применяется residual $K$-means. Independent Code Rate индекса равен 0.98, то есть collisions редки, но не исчезают.

Decoder cross-attends к $E_u$ и teacher-forced NTP обучается генерировать SID положительного item. На inference catalogue-trie-constrained beam search выдает 1024 valid SIDs. Каждый SID раскрывается во все соответствующие catalogue items; если общий pool превышает 1200 items, авторы удаляют SID hypotheses с минимальным beam score вместе с их collision groups.

Beam likelihood определяет попадание SID в candidate pool и редкое capacity truncation, но не финальный порядок. Это принципиально: items внутри одной collision group имеют одинаковый beam score, а SID sequence likelihood в целом плохо калиброван как item relevance.

### 3.3. Ranking Module

Candidate representation включает item ID и SID prefix $n$-grams. Они кодируются compositional multi-hash embeddings из общей embedding table, после чего candidate query делает cross-attention к shared history states:

$$
\tilde e_{u,i} = \operatorname{CrossAttn}(e_i, E_u).
$$

Отдельные heads предсказывают оценки для набора ranking tasks $t \in T$. Финальный score - фиксированная комбинация этих heads:

$$
\hat R(u,i) = \sum_{t \in T} w_t \hat r^t_{u,i}.
$$

В deployed configuration Ranking Module состоит из одного layer. После него нет дополнительного learned reranking.

## 4. Teacher Ranker и Rollout Distillation

Teacher Ranker - отдельная большая sequential model с cross-attention ranker-ом и task-specific heads. Она использует до 8000 history events против 2048 у student-а и обучается autoregressively по множеству позиций. По внутренним предшествующим тестам teacher превосходит production ranker, но эти тесты не являются контролируемым результатом данной статьи. Из-за стоимости teacher используется только во время training.

Для каждого request формируются два candidate set:

1. **Rollouts $G_u$.** Текущий decoder синхронно, без checkpoint lag, запускает constrained beam search. Полученные SIDs раскрываются в items и оцениваются teacher-ом.
2. **Logged impressions $M_u$.** Teacher также оценивает items, которые production system реально показала в соответствующем request.

Ranking Module повторяет task-wise teacher scores. В основной конфигурации используется element-wise MAE, отдельно усредненная внутри каждого source:

$$
L_{\mathrm{distill}} = L_{\mathrm{roll\text{-}distill}} + L_{\mathrm{impr\text{-}distill}}.
$$

Более 90% уникальных distillation candidates приходят из rollouts, но это не означает 90% веса loss: оба source term нормализуются по собственному числу candidates и входят с одинаковыми коэффициентами. Полная joint objective:

$$
L = L_{\mathrm{NTP}} + L_{\mathrm{distill}}.
$$

NTP обновляет decoder и shared encoder; distillation - Ranking Module и тот же shared encoder. Teacher scores являются единственным supervision ranking branch-а.

## 5. Training и serving

Offline comparison использует две недели логов infinite personalized music feed и temporal split: последний день идет в test. Request сохраняется, если есть хотя бы один positive item - like или long listen без dislike. Offline model делает один chronological pass; effective batch size равен 16,384 requests на 128 GPUs с FSDP2. Training rollout beam size - 32, хотя serving beam size - 1024.

Online experiment стартует с checkpoint, обученного на четырех неделях. Далее model обновляется каждые 10 минут; update занимает десятки минут, поэтому served checkpoint обычно видел события менее чем часовой давности. GPU inference работает через NVIDIA Triton, а CPU service собирает request features.

Практическая последовательность inference:

1. один раз закодировать историю пользователя;
2. с trie constraints сгенерировать beam из 1024 valid SIDs;
3. раскрыть collision groups и ограничить pool максимум 1200 items;
4. переоценить все items однослойной Ranking Module на shared encoder states;
5. отсортировать по фиксированной комбинации task scores и вернуть slate.

## 6. Offline evaluation

Авторы используют три разные метрики, каждая покрывает только часть системы:

- **R@1000** - recall generated candidate pool до reranking;
- **TeacherRecall@K** - совпадение top-K student-а и Teacher Ranker-а на одном generated pool;
- **WPA** (weighted pair accuracy) - правильность порядка пар logged impressions с engagement order `like > play > skip > dislike`, взвешенная расстоянием между target weights.

<div class="table-scroll">
<table>
<thead><tr><th>Model</th><th>R@1000</th><th>T-R@10</th><th>T-R@100</th><th>WPA</th></tr></thead>
<tbody>
<tr><td>Generative retrieval</td><td>0.8643</td><td>0.0382</td><td>0.1944</td><td>0.5429</td></tr>
<tr><td>Gryphon</td><td>0.8593</td><td>0.0392</td><td>0.1701</td><td>0.5528</td></tr>
<tr><td>Gryphon-v2, beam order</td><td>0.8615</td><td>0.0381</td><td>0.1936</td><td>0.5478</td></tr>
<tr><td><strong>Gryphon-v2</strong></td><td>0.8615</td><td><strong>0.5654</strong></td><td><strong>0.7344</strong></td><td><strong>0.5892</strong></td></tr>
<tr><td>Production ranker</td><td>-</td><td>-</td><td>-</td><td>0.6141</td></tr>
<tr><td>Teacher Ranker</td><td>-</td><td>-</td><td>-</td><td>0.6199</td></tr>
</tbody>
</table>
</div>

Candidate recall Gryphon-v2 находится внутри run-to-run standard deviation около 0.003 относительно baselines: distillation не улучшает retrieval branch, но и не разрушает его. Главный сдвиг происходит после reranking: T-R@10 растет с примерно 0.04 до 0.5654, T-R@100 - до 0.7344, а WPA - с 0.5528 у Gryphon до 0.5892. По расчету авторов, distillation закрывает 54% WPA gap от Gryphon до teacher и 59% gap до production ranker.

Reference rankers не являются matched baselines: они обучены на полном году feedback, тогда как generative models видят двухнедельное окно. Кроме того, TeacherRecall измеряет верность тому же teacher-у, который дает targets, а не независимую recommendation quality.

## 7. Ablations

<div class="table-scroll">
<table>
<thead><tr><th>Configuration</th><th>R@1000</th><th>T-R@10</th><th>T-R@100</th><th>WPA</th></tr></thead>
<tbody>
<tr><td>Rollout + impressions</td><td>0.8615</td><td>0.5654</td><td>0.7344</td><td>0.5892</td></tr>
<tr><td>Rollout only</td><td>0.8610</td><td>0.5618</td><td>0.7288</td><td>0.5730</td></tr>
<tr><td>Impressions only</td><td>0.8663</td><td>0.2983</td><td>0.5281</td><td>0.5872</td></tr>
<tr><td>Rollout beam 64</td><td>0.8565</td><td>0.5661</td><td>0.7359</td><td>0.5901</td></tr>
<tr><td>Rollout beam 128</td><td>0.8616</td><td>0.5762</td><td>0.7427</td><td>0.5905</td></tr>
<tr><td>MSE loss</td><td>0.8671</td><td>0.5748</td><td>0.7329</td><td>0.5915</td></tr>
<tr><td>Huber loss</td><td>0.8679</td><td>0.5568</td><td>0.7220</td><td>0.5894</td></tr>
<tr><td>KL loss</td><td>0.8662</td><td>0.5452</td><td>0.7151</td><td>0.5860</td></tr>
</tbody>
</table>
</div>

Rollout-only сохраняет высокую teacher fidelity, но хуже ранжирует logged impressions; impression-only почти сохраняет WPA, но T-R@10 падает до 0.2983. Смесь действительно покрывает разные distributions. Увеличение training beam с 32 до 128 дает небольшие descriptive gains ценой teacher scoring большего pool. MSE имеет лучший point estimate по WPA и T-R@10, но deployed MAE был выбран до post-hoc sweep, а statistical testing различий в ablation table не приведен.

## 8. Online A/B и serving efficiency

Control arm - полная production cascade: 15+ generators создают около 10,000 candidates, pre-ranker оставляет 3000, final ranker строит выдачу. Treatment arm - только Gryphon-v2 с 1024 generated SIDs и не более 1200 resolved items. В каждом arm находилось по 8% eligible users; active user определяется как пользователь с не менее чем 7 минутами listening в день.

<div class="table-scroll">
<table>
<thead><tr><th>Metric</th><th>Relative change</th></tr></thead>
<tbody>
<tr><td>Total listening time</td><td>+1.62%</td></tr>
<tr><td>Active users, primary metric</td><td>+1.41%</td></tr>
<tr><td>Likes</td><td>+7.12%</td></tr>
<tr><td>Repeat commands</td><td>+15.25%</td></tr>
<tr><td>Unfinished-track ratio</td><td>-9.65%</td></tr>
</tbody>
</table>
</div>

Все дельты статистически значимы при $p \lt 0.001$. End-to-end latency, включая CPU feature construction, network, generation и ranking, сопоставима с cascade. Gryphon-v2 также дает примерно 4x throughput относительно варианта, где тот же generative backbone обслуживается вместе с Teacher Ranker: student переиспользует 2048-event encoding, тогда как teacher отдельно обрабатывает до 8000 событий.

Это сильнее результата первого Gryphon: там generative model заменяла generators и pre-ranking, но candidates все еще проходили через production final ranker. Здесь treatment устраняет все три learned stages каскада.

## 9. Сильные стороны

- Очень сильное production evidence: заменена не отдельная retrieval route, а вся mature cascade.
- Matched offline baselines используют один tokenizer, backbone, candidate budget и decoding procedure; сравнение Gryphon с Gryphon-v2 хорошо изолирует источник supervision Ranking Module.
- Rollout candidates уменьшают train-serving mismatch, а logged impressions дают дополнительное покрытие; ablation подтверждает взаимодополняемость sources.
- Авторы аккуратно разделяют retrieval recall, teacher fidelity, logged-pair ranking и online quality, не объявляя одну offline metric полной оценкой системы.
- Teacher удален из serving graph, поэтому его capacity можно увеличивать без прямого роста online inference cost.

## 10. Ограничения и открытые вопросы

- Данные, код, duration эксперимента и абсолютные значения online metrics не раскрыты, поэтому независимо проверить результат нельзя.
- Online test измеряет aggregate effect архитектуры, distillation, online updates и замененной cascade. Он не изолирует causal contribution Rollout Distillation.
- Эксперимент проведен на одной music surface, по 8% eligible users в каждой группе. Нет evidence о long-term effects и полном rollout.
- Не измерены long-tail coverage, artist diversity, novelty и exposure concentration. Большие engagement gains могут сочетаться с ухудшением этих свойств.
- TeacherRecall@K по определению благоприятен для student-а, обученного на том же teacher-е. WPA наследует exposure bias production logs и не проверяет ranking новых items, которые генерирует Gryphon-v2.
- Нет контролируемого сравнения с RL-based alignment. Rollout Distillation не обновляет policy по reward и может быть complementary, а не альтернативой RL.
- Teacher и production ranker имеют год training data, student - две недели offline data; reference WPA полезен как ceiling, но не как честное matched comparison.

## 11. Вывод

Gryphon-v2 - одна из наиболее убедительных industrial работ о полном переходе от retrieve-pre-rank-rank cascade к единой generative architecture. Ее центральный урок не в том, что SID likelihood заменяет ranking, а в обратном: **SID decoder должен отвечать за broad proposal set, а дешевый item-level ranker на shared history states - за финальный multi-objective порядок**.

Rollout Distillation превращает большой offline ranker в training-time источник preference supervision и переносит большую часть его ranking advantage в served model. Evidence сильно для краткосрочной production viability на одной музыкальной поверхности, но пока недостаточно для выводов о долгосрочном качестве, diversity, переносимости между доменами и превосходстве над RL alignment.
