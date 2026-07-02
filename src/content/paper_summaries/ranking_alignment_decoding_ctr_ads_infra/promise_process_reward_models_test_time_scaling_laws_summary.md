---
title: "PROMISE: Process Reward Models Unlock Test-Time Scaling Laws in Generative Recommendations"
category: "ranking_alignment_decoding_ctr_ads_infra"
slug: "promise_process_reward_models_test_time_scaling_laws_summary"
catalogId: "paper-promise_process_reward_models_test_time_scaling_laws_summary"
paperUrl: "https://arxiv.org/abs/2601.04674"
---
> **Авторы:** Chengcheng Guo, Kuo Cai, Yu Zhou, Qiang Luo, Ruiming Tang, Han Li, Kun Gai, Guorui Zhou.
>
> **Аффилиации:** Kuaishou Inc.; Kun Gai указан как unaffiliated.
>
> **Источник:** arXiv:2601.04674v1 от 2026-01-08.

## 1. Коротко: о чем статья

PROMISE переносит идею Process Reward Models (PRM) из LLM reasoning в generative recommendation. В SID-based GR item генерируется как hierarchical sequence codes. Ошибка на раннем, coarse token может увести beam search в неправильную semantic subspace. Авторы называют это **Semantic Drift**.

PROMISE добавляет path-level PRM, который оценивает промежуточные SID prefixes во время generation. На inference используется PRM-guided beam search: decoder расширяет больший candidate set `K+`, PRM переоценивает partial paths, и дальше проходят top `K'` candidates. Это дает test-time scaling: можно увеличить inference compute через ширину поиска и PRM scoring, не увеличивая backbone model.

Главный claim: smaller model с PRM-guided search может матчить или превосходить larger model при меньшем serving cost. На Kuaishou industrial dataset PROMISE улучшает Recall@100 на +47.92% при `K+=4000` и +59.31% при `K+=6000` относительно strongest baseline; online A/B дает статистически значимый рост usage time.

<figure class="paper-figure">
  <img src="../../assets/promise/framework.png" alt="PROMISE framework with path-level process reward model for SID generation">
  <figcaption>Рисунок 1. PROMISE обучает path-level PRM на positive/negative SID prefixes и использует его в beam search, чтобы отсекать semantic drift на промежуточных шагах.</figcaption>
</figure>

## 2. Semantic Drift

В обычном teacher-forced GR decoder на training видит ground-truth previous tokens. На inference он conditioning on own predictions. Если первый или второй SID token неверный, subsequent tokens генерируются уже внутри неправильного semantic subtree. Standard beam search сортирует по decoder likelihood, но likelihood может оставлять высокий score у semantically wrong prefix.

Semantic Drift особенно вреден в hierarchical SIDs: ранние tokens отвечают за coarse semantics, поздние - за fine details. Ошибка на coarse level не просто немного меняет item, а переводит generation в другой раздел catalog.

## 3. Метод: path-level PRM

PRM получает user/context и candidate SID prefix `[s_1, ..., s_b]`, где `b <= d`, и возвращает relevance score prefix-а. Positive samples строятся из ground-truth SID prefixes. Negative samples - sampled invalid/wrong paths из valid path space. Loss - InfoNCE по prefix level.

Ключевое отличие от outcome reward: PRM дает dense feedback на каждом SID step. Модель может понять, что prefix уже ушел в неверную semantic branch, не дожидаясь complete generated item.

Architecture lightweight: PRM block использует encoder states и cross-attention для parallel scoring candidate paths. В deployment PRM имеет один block и меньше attention heads, чтобы ограничить latency.

## 4. PRM-guided beam search

На каждом generation step decoder производит expanded candidate set размера `K+`. PRM оценивает prefixes, выбирает top `K'`, и только они идут на следующий step. Это отличается от brute-force увеличения обычного beam size: decoder compute не растет так же сильно, потому что PRM дешевле main decoder scoring.

В production version используется `K+=4000`, target `K=K'=1000`. Для ускорения Top-K авторы применяют Radix Top-K optimization. Итоговый overhead: parameter size +15%, inference latency +10% относительно версии без PRM.

## 5. Public benchmarks

На Amazon Beauty и Sports PROMISE сравнивается с traditional sequential methods и generative baselines TIGER, HSTU, ActionPiece. PROMISE лучше по Recall@5/10 и NDCG@5/10.

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>Promise R@10</th><th>Promise N@10</th><th>Lift к лучшему baseline</th></tr></thead>
<tbody>
<tr><td>Sports</td><td>0.0689</td><td>0.0373</td><td>+37.80% R@10, +42.19% N@10</td></tr>
<tr><td>Beauty</td><td>0.0821</td><td>0.0437</td><td>+5.60% R@10, +3.07% N@10</td></tr>
</tbody>
</table>
</div>

Sports lift намного больше, чем Beauty. Это важно: PRM benefit зависит от severity semantic drift and search space, а не является универсальной константой.

## 6. Industrial-scale experiments

Industrial dataset основан на Kuaishou logs: больше 400M daily active users, около 50B interactions/day, больше 100M items optimized daily. Items квантуются residual K-means tokenizer depth 3, codebook size 8192.

PROMISE сравнивается с deployed industrial baselines GRank, MPFormer, MISS, GPRP, Kuaiformer, CRM. Результаты:

<div class="table-scroll">
<table>
<thead><tr><th>Method</th><th>Recall@100</th><th>NDCG@100</th><th>Recall@1000</th><th>NDCG@1000</th></tr></thead>
<tbody>
<tr><td>Best baseline GRank</td><td>0.1010</td><td>0.00472</td><td>0.3178</td><td>0.01422</td></tr>
<tr><td>Promise K+=4000</td><td>0.1494</td><td>0.00652</td><td>0.3358</td><td>0.01445</td></tr>
<tr><td>Promise K+=6000</td><td>0.1609</td><td>0.00663</td><td>0.3637</td><td>0.01504</td></tr>
</tbody>
</table>
</div>

Улучшение особенно большое на smaller retrieval sizes: NDCG@100 +38-40%, Recall@100 +48-59%. На Recall@1000 lift меньше, что логично: широкий retrieval set уже частично компенсирует drift.

## 7. Online A/B

A/B проводился 7 дней на Kuaishou и Kuaishou Lite. 5% users в control и 5% users в treatment, около 20M users на группу. Control - generative recommendation with traditional beam search; treatment - PRM-guided search.

Результаты:

- Kuaishou total app usage time +0.121%, CI `[+0.04%, +0.20%]`;
- Kuaishou app usage time per user +0.120%, CI `[+0.06%, +0.18%]`;
- Kuaishou Lite total app usage time +0.131%, CI `[+0.03%, +0.23%]`;
- Kuaishou Lite app usage time per user +0.160%, CI `[+0.07%, +0.25%]`;
- total video watch time также растет примерно на +0.398-0.431%.

Абсолютные lifts небольшие, но для платформы с таким масштабом и статистически значимыми CI это production-relevant.

## 8. Ablation and scaling laws

Ablation по SID steps показывает cumulative benefit PRM. Без PRM HRecall@3@1000 = 0.2296. PRM только на step 3 дает 0.2650. PRM на steps 2+3 дает 0.3199. PRM на всех трех steps дает 0.3358. Это поддерживает тезис, что dense stepwise verification важнее single final rerank.

Test-time scaling plot показывает рост HRecall при увеличении `K+` от 1000 до 6000. Brute-force beam scaling тоже помогает, но дороже. Сравнение с parameter scaling показывает, что PRM test-time scaling достигает лучшего HRecall@3@1000 при равных FLOPs.

## 9. Сильные стороны

- Хорошая адаптация PRM к SID generation: reward ставится на prefixes, а не только на final item.
- Есть public benchmark, industrial offline и online A/B.
- Deployment section содержит реальные overhead numbers: +15% params, +10% latency.
- Работа задает новую ось scaling для recommenders: inference-time search compute.

## 10. Ограничения и вопросы

PRM требует negative path sampling и valid SID path lookup. Если tokenizer нестабилен или имеет много invalid paths/collisions, training PRM может стать сложным.

Увеличение `K+` все равно повышает inference cost. Для latency-critical surfaces нужно строить Pareto curve quality-latency, а не смотреть только max Recall.

Online gains зависят от Kuaishou stack, traffic и user behavior. В других domains semantic drift может быть слабее или сильнее.

## 11. Вывод

PROMISE показывает, что generative recommendation можно улучшать не только tokenizer-ом или backbone scale, но и **stepwise inference-time verification**. Главный takeaway: для hierarchical SIDs beam search должен проверять semantic prefixes, иначе early token errors превращаются в irreversible drift. PRM-guided search - практичный способ обменять немного latency на существенный retrieval-quality gain.
