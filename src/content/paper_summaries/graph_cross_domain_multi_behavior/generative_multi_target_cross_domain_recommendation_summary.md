---
title: "Generative Multi-Target Cross-Domain Recommendation"
category: "graph_cross_domain_multi_behavior"
slug: "generative_multi_target_cross_domain_recommendation_summary"
catalogId: "paper-generative_multi_target_cross_domain_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2507.12871"
---
> **Авторы:** Jinqiu Jin, Yang Zhang, Fuli Feng, Xiangnan He.
>
> **Аффилиации:** University of Science and Technology of China; National University of Singapore.
>
> **Источник:** arXiv:2507.12871v3 от 2025-08-07.

## 1. Коротко: о чем статья

GMC предлагает generative approach для Multi-Target Cross-Domain Recommendation (MTCDR). В MTCDR нужно одновременно улучшить рекомендации в нескольких доменах, например Scientific, Pantry, Instruments, Arts и Office. Классические cross-domain методы часто требуют overlapped users/items или большой auxiliary pre-training corpus. GMC пытается обойти это через shared semantic identifiers и unified generative recommender.

Метод состоит из трех частей: domain-shared semantic identifier learning, unified recommender training и domain-specific lightweight fine-tuning. Items из разных domains квантуются в общий SID vocabulary; seq2seq recommender обучается на mixed sequences из всех domains; затем для каждого domain добавляется LoRA fine-tuning.

Главный результат: GMC стабильно улучшает weighted average NDCG@10 и NDCG@5 относительно сильных baselines, особенно на малых domains вроде Pantry и Scientific. В ablation видно, что sharable codebook и unified recommender нужны вместе: общий tokenizer без общей generative model не раскрывает cross-domain transfer.

<figure class="paper-figure">
  <img src="../../assets/gmc/framework.png" alt="GMC framework for generative multi-target cross-domain recommendation">
  <figcaption>Рисунок 1. GMC: shared semantic identifier learning across domains, unified generative recommender training и lightweight domain-specific LoRA adaptation.</figcaption>
</figure>

## 2. Контекст: чем MTCDR отличается от обычного transfer learning

В single-target transfer recommendation обычно есть source domains и один target domain. MTCDR сложнее: нужно улучшать несколько target domains одновременно. При этом domains могут не иметь общих users/items. Если нет overlap, graph-style transfer через shared nodes уже не работает.

Некоторые recent methods используют language/content representations как domain-shared bridge. Но такие методы часто требуют больших pre-training corpora или сильно зависят от textual side information. GMC предлагает другой bridge: items из разных domains получают discrete semantic identifiers из общей codebook structure, а recommendation обучается как generation task.

## 3. Метод

### 3.1. Semantic identifier learning

Для item text используется LLM/text encoder, который строит semantic embedding. Затем RQ-VAE-style quantization получает multi-level codes. В отличие от domain-specific tokenization, codebooks шарятся между domains. Это позволяет разным domains использовать один semantic token set.

Ключевая деталь - domain-aware contrastive loss. Авторы замечают, что первый уровень quantization часто ловит coarse domain information, а более глубокие residual levels могут стать domain-agnostic и шариться между domains. Contrastive objective подтягивает intra-domain items и отталкивает inter-domain items, чтобы code assignment отражал как domain-specific, так и shared semantics.

### 3.2. Unified generative recommender

После tokenization каждая user behavior sequence превращается в sequence semantic codes. Все domains объединяются в один training corpus, и Transformer encoder-decoder обучается next-token generation. Это позволяет модели видеть multi-domain transition patterns и переносить knowledge между domains через общий code vocabulary.

### 3.3. Domain-specific fine-tuning

Чтобы не потерять domain-specific quirks, GMC добавляет lightweight LoRA modules для каждого domain. Это компромисс между одним общим recommender и отдельной моделью для каждого domain. В ablation LoRA modules составляют около 3.7% параметров unified model, но дают заметный gain на малых domains.

## 4. Эксперименты

Датасеты - пять Amazon domains: Scientific, Pantry, Instruments, Arts и Office. Baselines включают ID-based sequential recommenders (SASRec, BERT4Rec, HGN, GRU4Rec), transferable/content recommenders (FDSA, S3-Rec, UniSRec, VQ-Rec, RecFormer, P5-CID) и generative methods (TIGER, IDGenRec).

GMC улучшает weighted average NDCG@10 до 0.0850 против 0.0797 у IDGenRec и 0.0782 у TIGER; weighted average NDCG@5 до 0.0780 против 0.0742 у IDGenRec и 0.0716 у TIGER.

На отдельных domains:

<div class="table-scroll">
<table>
<thead><tr><th>Domain</th><th>GMC R@5</th><th>GMC N@5</th><th>Что заметно</th></tr></thead>
<tbody>
<tr><td>Scientific</td><td>0.0791</td><td>0.0556</td><td>N@5 +5.34% к сильнейшему baseline</td></tr>
<tr><td>Pantry</td><td>0.0422</td><td>0.0262</td><td>N@5 +25.41%, один из самых больших gains</td></tr>
<tr><td>Instruments</td><td>0.0980</td><td>0.0828</td><td>GMC лучше IDGenRec/VQ-Rec/UniSRec</td></tr>
<tr><td>Office</td><td>0.1063</td><td>0.0912</td><td>малый gain, но лучший или near-best результат</td></tr>
</tbody>
</table>
</div>

## 5. Ablation

Самая важная ablation показывает, что cross-domain transfer в GMC не является просто следствием "общих codes":

- без sharable codebook Scientific R@5 падает с 0.0791 до 0.0632, Pantry R@5 с 0.0422 до 0.0316;
- без unified recommender Scientific R@5 падает до 0.0608, Pantry до 0.0283;
- без domain-aware contrastive loss качество тоже падает, но мягче;
- без LoRA fine-tuning малые domains теряют больше;
- full fine-tuning всех параметров может переобучиться и разрушить shared knowledge, особенно на Pantry.

Вывод: shared identifier и unified generative model должны обучаться совместно. Если учить shared identifiers, а потом тренировать отдельные generators, cross-domain benefit теряется.

## 6. Анализ code sharing

Авторы визуализируют residual vectors и code assignments. Наблюдение полезное: на первом уровне квантизации codes часто разделяются по domains, а на втором уровне residual vectors из разных domains смешиваются и codewords начинают шариться. Это соответствует интуиции RQ: первый code забирает coarse/domain-specific information, residual codes кодируют более общие semantic factors.

В code similarity analysis GMC дает более похожие item identifiers между relevant cross-domain items, чем separate tokenization. Это и есть желаемый bridge: не overlapped users/items, а shared semantic code structure.

## 7. Сильные стороны

- Хорошая постановка non-overlapped MTCDR, где нельзя полагаться на shared entities.
- Метод соединяет semantic IDs и cross-domain recommendation, а не просто добавляет text encoder.
- Ablation ясно показывает, что sharable codebook, unified model и LoRA adaptation решают разные части задачи.
- LoRA fine-tuning делает approach параметрически дешевле, чем отдельная full model на domain.

## 8. Ограничения и вопросы

GMC по-прежнему зависит от textual item descriptions. Если domains имеют бедную metadata или разные languages/modalities, shared semantic codes могут стать шумными.

Метод рассматривает item identifier learning и recommender training как два этапа. Авторы сами отмечают, что end-to-end совместное обучение tokenizer-а и generator-а остается открытым направлением.

В работе нет online evaluation и production constraints. Для реального MTCDR важны incremental catalog updates, domain imbalance, latency beam search и monitoring negative transfer.

## 9. Вывод

GMC стоит читать как работу про **semantic IDs как мост между non-overlapped domains**. Ее главный takeaway: в cross-domain recommendation общий discrete vocabulary может заменить overlapped entities, но только если он используется вместе с unified generative recommender и аккуратной domain-specific adaptation.
