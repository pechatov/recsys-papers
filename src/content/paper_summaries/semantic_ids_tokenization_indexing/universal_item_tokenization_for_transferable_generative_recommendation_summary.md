---
title: "Universal Item Tokenization for Transferable Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "universal_item_tokenization_for_transferable_generative_recommendation_summary"
catalogId: "paper-universal_item_tokenization_for_transferable_generative_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2504.04405"
---
> **Авторы:** Bowen Zheng, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ji-Rong Wen.
>
> **Аффилиации:** Gaoling School of Artificial Intelligence, Renmin University of China; WeChat, Tencent.
>
> **Код:** <https://github.com/RUCAIBox/UTGRec>.

## 1. Коротко: о чем статья

UTGRec решает проблему transferability в generative recommendation. Большинство SID tokenizer-ов обучаются внутри одного домена: tokenizer для Beauty, tokenizer для Instruments, tokenizer для Games и т.д. Это плохо переносится, потому что semantic codebooks и downstream generator завязаны на конкретный catalog и конкретное распределение interactions.

Авторы предлагают **Universal Item Tokenization**: один tokenizer обучается на нескольких доменах и должен выдавать codes, которые можно переносить на новые downstream domains. Для этого UTGRec использует MLLM item content encoder, tree-structured codebooks, reconstruction raw content и collaborative integration через co-occurring item alignment/reconstruction.

Главный claim: если tokenizer и generator pre-trained across domains, generative recommender начинает лучше масштабироваться и лучше работает на long-tail items. На четырех downstream Amazon domains UTGRec стабильно превосходит traditional, content-based и generative baselines.

<figure class="paper-figure">
  <img src="../../assets/utgrec/framework.png" alt="UTGRec universal item tokenization and item content reconstruction framework">
  <figcaption>Рисунок 1. UTGRec учит universal item tokenizer через MLLM, tree-structured codebooks и reconstruction/collaborative objectives, после чего pre-trained generator переносится на downstream domains.</figcaption>
</figure>

## 2. Почему domain-specific tokenization ограничивает GR

Semantic-ID GR обычно выглядит так: item text/image кодируется embedding model, затем RQ-VAE или похожий quantizer превращает item в sequence codes, и seq2seq recommender учится генерировать target item code. Но если tokenizer обучен только на одном домене, он не знает, какие code semantics могут быть reusable across domains.

Transferable sequential recommenders вроде UniSRec/MISSRec используют content representations и могут переноситься лучше, но они не всегда используют generative retrieval interface. UTGRec пытается совместить оба свойства: transfer learning и discrete generation.

Основная идея: codebook должен быть не просто "хорошим для этого каталога", а reusable semantic vocabulary, где разные domains могут разделять multi-level code structure.

## 3. Метод: Universal Item Tokenizer

### 3.1. MLLM item content encoding

Для каждого item используется textual и visual information. MLLM получает prompt вида "compress the image and text into L tokens" и формирует несколько item content representations. Это отличается от обычного "один text embedding -> RQ-VAE", потому что модель сразу учит несколько content views разной гранулярности.

### 3.2. Tree-structured codebooks

Вместо независимых multi-level codebooks UTGRec вводит tree-structured codebooks с root/leaf components. Мотивация такая: представления разных granularities могут быть слишком похожи, и обычный multi-level quantizer плохо переносит code semantics между domains. Tree structure заставляет codebooks разделять часть параметров и поддерживать связь между coarse и fine codes.

### 3.3. Content reconstruction and collaborative integration

Tokenizer обучается не только восстанавливать embedding. Он reconstructs raw item content через text decoder и vision decoder. Дополнительно используются co-occurring items: alignment loss сближает representation item'а с positive co-occurring item, а reconstruction loss заставляет codes сохранять collaborative neighbors.

Это важный design choice. Content-only tokenizer переносим, но может не отражать preference structure. Collaborative signal recommendation-aware, но плохо переносится без аккуратной интеграции. UTGRec добавляет collaborative knowledge как auxiliary signal при обучении universal codebooks.

## 4. Transferable generative recommender

После tokenizer pre-training авторы pre-train generative recommender на mixed code sequences из нескольких domains. Downstream adaptation состоит из tokenizer fine-tuning и recommender fine-tuning для целевого домена.

Важная ablation: просто применить pre-trained tokenizer без fine-tuning плохо работает. Но full fine-tuning всех codebook matrices тоже хуже, потому что разрушает learned cross-domain code associations. Лучший вариант - аккуратная adaptation, где сохраняется universal structure.

## 5. Эксперименты

Pre-training domains: Arts Crafts and Sewing, Baby Products, CDs and Vinyl, Cell Phones and Accessories, Clothing Shoes and Jewelry. Downstream domains: Musical Instruments, Industrial and Scientific, Video Games, Office Products.

Baselines включают GRU4Rec, BERT4Rec, SASRec, FMLP-Rec, FDSA, S3-Rec, UniSRec, VQ-Rec, MISSRec, TIGER, LETTER и TIGER с multimodal features.

UTGRec дает лучшие результаты во всех четырех downstream domains. Например:

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>UTGRec R@10</th><th>UTGRec N@10</th><th>Комментарий</th></tr></thead>
<tbody>
<tr><td>Instrument</td><td>0.0616</td><td>0.0334</td><td>выше MISSRec, VQ-Rec, LETTER и TIGER variants</td></tr>
<tr><td>Scientific</td><td>0.0481</td><td>0.0255</td><td>заметный выигрыш над content/generative baselines</td></tr>
<tr><td>Game</td><td>0.0909</td><td>0.0491</td><td>улучшение на плотном domain</td></tr>
<tr><td>Office</td><td>0.0462</td><td>0.0269</td><td>наиболее сильный lift над LETTER/TIGER</td></tr>
</tbody>
</table>
</div>

## 6. Ablation и анализ

В ablation на Instrument и Scientific удаление любого ключевого компонента снижает качество:

- без tree-structured codebooks: Instrument R@10 падает с 0.0616 до 0.0591;
- без co-occurring alignment: до 0.0599;
- без co-occurring reconstruction: до 0.0607;
- без tokenizer fine-tuning: до 0.0551;
- full fine-tuning codebooks: до 0.0559;
- без pre-training tokenizer и recommender: до 0.0575.

Самый сильный спад дает неправильный transfer regime, а не один auxiliary loss. Это поддерживает тезис авторов: cross-domain tokenization полезна только если сохранить universal code associations и аккуратно адаптировать их к target domain.

Scalability analysis показывает, что TIGER/LETTER начинают деградировать при росте числа layers, а UTGRec продолжает улучшаться. Интерпретация: multi-domain pre-training дает достаточно данных и устойчивую code structure, чтобы larger generator не переобучался так быстро.

Long-tail analysis также в пользу UTGRec: метод особенно сильнее на менее популярных item groups, где transferable content/collaborative semantics важнее local ID memorization.

## 7. Сильные стороны

- Работа переносит discussion о semantic IDs из single-domain tokenizer quality в multi-domain transfer.
- Tree-structured codebooks дают понятный механизм sharing между granularities.
- Есть публичный код.
- Ablations проверяют tokenizer pre-training, recommender pre-training, fine-tuning strategy и collaborative objectives отдельно.

## 8. Ограничения и вопросы

Все эксперименты построены на Amazon-style domains. Не ясно, насколько universal tokenizer переносится между совсем разными modalities и languages, например news, music, short video и local services.

MLLM item content encoding дорогой. Для production важно оценить cost обновления tokenizer-а при catalog churn, latency offline tokenization и стабильность codes при incremental retraining.

Collaborative integration через co-occurring items может переносить exposure/popularity bias. Для использования в retrieval pipeline нужны popularity-stratified, cold-start и collision diagnostics.

## 9. Вывод

UTGRec полезен как blueprint для **transferable semantic-ID tokenizer-а**. Главная идея: tokenizer должен учиться на multi-domain content и behavior signals, а не быть одноразовым preprocessing step для одного каталога. Если semantic IDs должны стать reusable vocabulary для generative recommendation, их нужно pre-train и адаптировать почти как language representations.
