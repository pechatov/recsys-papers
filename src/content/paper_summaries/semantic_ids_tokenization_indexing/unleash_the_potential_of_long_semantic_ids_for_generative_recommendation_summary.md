---
title: "Unleash the Potential of Long Semantic IDs for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "unleash_the_potential_of_long_semantic_ids_for_generative_recommendation_summary"
catalogId: "paper-unleash_the_potential_of_long_semantic_ids_for_summary"
paperUrl: "https://arxiv.org/abs/2602.13573"
---
> **Авторы:** Ming Xia, Zhiqin Zhou, Guoxin Ma, Dongmin Huang.
>
> **Аффилиации:** Southern University of Science and Technology; Nanjing University; Xi'an Jiaotong University.
>
> **Источник:** arXiv:2602.13573v1 от 2026-02-14.

## 1. Коротко: о чем статья

Статья предлагает ACERec - framework, который пытается снять конфликт между длинными expressive Semantic IDs и эффективным sequential recommendation. RQ-based методы используют короткие SID sequences, потому что autoregressive generation длинных кодов дорогой. OPQ/PQ-based методы могут дать длинные параллельные codes, но часто сжимают их грубым pooling и теряют fine-grained semantics.

ACERec делает decoupling: item сначала получает длинный semantic ID, например `m=32` OPQ digits, а затем **Attentive Token Merger (ATM)** distills эти длинные tokens в небольшой набор latent tokens `k`. Recommender работает уже с compact latents, но они получены из high-resolution input. Дополнительно вводится **Intent Token** как dynamic prediction anchor и dual-granularity objective: token-level prediction плюс item-level semantic alignment.

Главный результат: ACERec превосходит ID-based и SID-based baselines на шести datasets, средний lift заявлен как +14.40% Recall@10 и +17.66% NDCG@10. Важный engineering claim: модель быстрее autoregressive baselines и сохраняет качество на cold-start/long-tail items.

<figure class="paper-figure">
  <img src="../../assets/acerec/framework.png" alt="ACERec framework with attentive token merger, intent token, and dual-granularity optimization">
  <figcaption>Рисунок 1. ACERec отделяет granularity tokenizer-а от длины sequence, которую видит recommender: длинные OPQ IDs сжимаются ATM в compact latents, затем Intent Token агрегирует user intent.</figcaption>
</figure>

## 2. Проблема: длинные IDs полезны, но неудобны

Короткий RQ-SID удобен для generation, но ограничивает capacity. Когда item описывается 3-4 tokens, разные attributes вроде brand, category, color, style и use case могут смешиваться в одном code path. Это усиливает collisions и ухудшает cold-start generalization.

Длинный PQ/OPQ identifier лучше сохраняет fine-grained attributes, потому что разные subspaces квантуются отдельно. Но если напрямую подать 32+ tokens в sequential model, attention cost и decoding cost растут. Если просто mean-pool длинные tokens, возникает semantic blurring: разные attributes усредняются и теряют различимость.

ACERec формулирует компромисс: tokenizer может быть high-resolution, а recommender input - compact, если compression learned и content-adaptive.

## 3. Метод

### 3.1. Long semantic IDs через OPQ

Items кодируются длинной последовательностью OPQ tokens. В отличие от RQ, OPQ разбивает embedding space на orthogonal subspaces и позволяет предсказывать/использовать tokens параллельно. Это повышает representational capacity и уменьшает pressure на каждый отдельный token.

### 3.2. Attentive Token Merger

ATM генерирует `k` content-adaptive query vectors из item summary и применяет cross-attention к token embeddings длинного SID. В результате получается compact set latent tokens `Z_i = {z_1, ..., z_k}`. Важное отличие от pooling: разные latent queries могут выбрать разные subspace signals.

В ablation ATM лучше MLP, mean pooling и convolution. На Instruments ATM дает R@10 0.1211 и N@10 0.0674, тогда как MLP дает 0.1085/0.0606, mean pooling 0.1106/0.0618, convolution 0.1190/0.0640.

### 3.3. Intent Token

Для user sequence ACERec добавляет специальный learnable Intent Token. Он не представляет item, а служит prediction anchor, который step-wise causally attends к historical latent tokens и агрегирует evolving user intent.

Intent Token лучше static anchors. На Instruments Recall@10 растет с 0.1001 у Last-Token/MLP до 0.1162, NDCG@10 с 0.0545 до 0.0653.

### 3.4. Dual-granularity alignment

ACERec обучается на двух уровнях:

- fine-grained token prediction для semantic ID tokens;
- item-level semantic alignment через intent-item contrastive objective.

Это важно, потому что token-level loss может оптимизировать локальную правильность codes, но не гарантировать, что generated/retrieved item семантически соответствует user intent. Item-level alignment стабилизирует global recommendation semantics.

## 4. Эксперименты

Датасеты: Sports, Beauty, Toys, Instruments, Office, Baby. Baselines: HGN, SASRec, S3-Rec, ICLRec, ELCRec, TIGER, ETEGRec, ActionPiece, RPG.

На четырех основных datasets ACERec дает:

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>ACERec R@10</th><th>ACERec N@10</th><th>Lift над лучшим baseline</th></tr></thead>
<tbody>
<tr><td>Sports</td><td>0.0493</td><td>0.0280</td><td>+11.79% R@10, +15.70% N@10</td></tr>
<tr><td>Beauty</td><td>0.0841</td><td>0.0501</td><td>+10.37% R@10, +14.91% N@10</td></tr>
<tr><td>Toys</td><td>0.0916</td><td>0.0579</td><td>+12.39% R@10, +22.15% N@10</td></tr>
<tr><td>Instruments</td><td>0.1211</td><td>0.0674</td><td>+4.85% R@10, +9.24% N@10</td></tr>
</tbody>
</table>
</div>

На Office ACERec дает R@10 0.1023, N@10 0.0579; на Baby R@10 0.0427, N@10 0.0249. В Baby lift по R@10 небольшой (+0.71%), но N@10 +10.67%.

## 5. Granularity analysis

Compression ratio `r=m/k` имеет inverted-V behavior. Слишком малый compression ratio оставляет слишком много деталей и перегружает sequence model; слишком большой создает bottleneck. Лучший режим в main experiments - `r=8`.

Сравнение с direct short OPQ особенно важно: ACERec и short-OPQ получают одинаковое число tokens на вход recommender-а, но ACERec distills их из long ID, а short-OPQ сразу строит короткий ID. ACERec выигрывает в среднем +44.48% Recall@10 и +56.91% NDCG@10. Это поддерживает claim, что high-resolution source matters даже при том же inference input length.

## 6. Cold-start и efficiency

Cold-start buckets показывают, что ACERec устойчив на sparse items. На Instruments и Baby модель остается выше baselines во всех frequency intervals. В appendix авторы показывают, что extreme-sparsity bucket `[0,5]` составляет значительную долю test items: 38.84% для Instruments, 32.19% для Toys, 27.71% для Beauty.

По inference trade-off ACERec примерно в 2.2 раза быстрее ActionPiece/TIGER/RPG в их setup и одновременно лучше по NDCG@10 на Toys. Причина: длинные semantic IDs используются в learned compression, но final recommendation не требует expensive autoregressive traversal длинной sequence.

## 7. Сильные стороны

- Хорошо формулирует granularity-efficiency trade-off в SID generation.
- ATM - разумная альтернатива naive pooling для long IDs.
- Intent Token дает явную dynamic aggregation point для user intent.
- Эксперименты включают tokenization granularity, compression ratio, cold-start buckets и throughput/quality analysis.

## 8. Ограничения и вопросы

ACERec полагается на OPQ и sentence embeddings; качество зависит от исходного embedding space и разбиения subspaces. Если item semantics сильно multimodal или interaction-driven, content OPQ может не уловить preference structure.

Метод усложняет pipeline: нужно строить long IDs, обучать ATM, поддерживать dual objectives и holistic scoring. Для production важно проверить stability under catalog updates и reproducibility of long IDs.

Сравнение с industrial dense retrieval или hybrid ranker отсутствует. ACERec сильна в offline sequential setup, но production retrieval потребует candidate deduplication, freshness, invalid code handling и latency guardrails.

## 9. Вывод

ACERec полезна как работа про **high-resolution semantic tokenization без quadratic/decoding bottleneck**. Главный переносимый takeaway: если короткий SID плохо различает items, не обязательно отдавать длинную sequence напрямую в recommender; можно сначала извлечь long semantic evidence, а затем learned attention compression превратить ее в компактные intent-aware latents.
