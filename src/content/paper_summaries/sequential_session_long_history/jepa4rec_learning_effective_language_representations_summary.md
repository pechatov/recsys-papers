---
title: "JEPA4Rec: Learning Effective Language Representations for Sequential Recommendation via Joint Embedding Predictive Architecture"
category: "sequential_session_long_history"
slug: "jepa4rec_learning_effective_language_representations_summary"
catalogId: "paper-jepa4rec_learning_effective_language_representations_summary"
paperUrl: "https://arxiv.org/abs/2504.10512"
---
> **Авторы:** Minh-Anh Nguyen, Dung D. Le.
>
> **Аффилиации:** College of Engineering and Computer Science, VinUniversity, Vietnam.
>
> **Статус:** COLM 2025 conference paper.

## 1. Коротко: о чем статья

JEPA4Rec адаптирует Joint Embedding Predictive Architecture (JEPA) к sequential recommendation, где items представлены textual metadata sentences. Вместо того чтобы учить только token-level MLM или contrastive instance discrimination, модель учится восстанавливать full item representations в embedding space по частично замаскированной user history и next-item text.

Работа нацелена на cross-domain, cross-platform и low-resource recommendation. Главная мотивация: ID-based recommenders плохо переносятся, потому что item IDs не совпадают между domains. Text-only approaches вроде RecFormer переносятся лучше, но требуют large-scale pre-training. JEPA4Rec пытается получить сопоставимую или лучшую transferability с меньшим pre-training corpus.

<figure class="paper-figure">
  <img src="../../assets/jepa4rec/framework.png" alt="JEPA4Rec model structure and pre-training framework">
  <figcaption>Рисунок 1. JEPA4Rec кодирует item metadata как text sentences, применяет masking к history/next item и обучает predictor восстанавливать target representations в embedding space.</figcaption>
</figure>

## 2. Метод: input representation

Каждый item превращается в sentence из metadata: title, category, brand и другие textual fields. User history становится sequence таких item sentences. Модель использует Longformer-style bidirectional encoder для item text и sequence context.

Важно, что item representation строится из текста, а не из arbitrary ID embedding. Поэтому модель может делать zero-shot или low-resource inference для новых domains, если доступна metadata.

## 3. Pre-training objectives

JEPA4Rec использует три цели.

### 3.1. Masked Language Modeling

MLM нужен не как главный recommender objective, а как bridge между pretrained language knowledge и e-commerce/recommendation text. Удаление MLM снижает Scientific NDCG@10 с 0.1282 до 0.1170 и Online Retail NDCG@10 с 0.1266 до 0.1118.

### 3.2. L2 mapping loss

Это JEPA-style component: predictor должен восстановить complete target representation из partial observations. В отличие от token reconstruction, модель учится в embedding space и может восстанавливать missing item information на уровне representation.

### 3.3. Sequence-item contrastive loss

Sequence representation пользователя сближается с next-item representation, а negatives берутся из mixed-domain items. Это дает richer training signal, чем BPR с одним negative. В ablation fine-tuning with BPR особенно сильно проигрывает на Online Retail: NDCG@10 0.0934 против 0.1266 у full JEPA4Rec.

## 4. Fine-tuning

После pre-training в target domain используется Context Encoder. Fine-tuning сохраняет language/text representation path и адаптирует model к next-item prediction. Такой setup должен работать даже когда target domain не имеет overlapped item IDs с source domains.

## 5. Эксперименты

Pre-training/fine-tuning проводится на Amazon datasets и Online Retail. Сценарии: cross-domain и cross-platform. Baselines: SASRec, BERT4Rec, LlamaRec, S3-Rec, UniSRec, VQ-Rec, RecFormer.

Main table показывает, что JEPA4Rec лучший почти везде:

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>JEPA4Rec R@10</th><th>JEPA4Rec N@10</th><th>Lift к RecFormer</th></tr></thead>
<tbody>
<tr><td>Scientific</td><td>0.1761</td><td>0.1282</td><td>+4.57% R@10, +7.01% N@10</td></tr>
<tr><td>Pet</td><td>0.1471</td><td>0.1210</td><td>+7.92% R@10, +11.41% N@10</td></tr>
<tr><td>Instruments</td><td>0.1347</td><td>0.1057</td><td>+4.49% R@10, +5.59% N@10</td></tr>
<tr><td>Arts</td><td>0.1920</td><td>0.1442</td><td>+6.84% R@10, +15.45% N@10</td></tr>
<tr><td>Office</td><td>0.1676</td><td>0.1276</td><td>+7.51% R@10, +10.86% N@10</td></tr>
<tr><td>Online Retail</td><td>0.2429</td><td>0.1266</td><td>+3.14% R@10, +1.36% N@10; MRR не лучше RecFormer</td></tr>
</tbody>
</table>
</div>

В среднем авторы сообщают +6.22% Recall@10 и +10.06% NDCG@10. На Online Retail ID-based models остаются конкурентными из-за высокой плотности данных, что честно ограничивает claim.

## 6. Efficiency of pre-training

Авторы подчеркивают, что JEPA4Rec pre-trained на трех datasets может конкурировать с RecFormer checkpoint, pre-trained на семи Amazon datasets. В zero-shot comparison official RecFormer иногда лучше, но JEPA4Rec сопоставим и выигрывает на Pet/Office по Recall/NDCG.

Low-resource analysis показывает ожидаемый pattern: text-only models особенно сильны при 1% или 5% training data, потому что используют item descriptions и pre-training. ID-based SASRec/LlamaRec догоняют по мере роста данных, но JEPA4Rec остается сильным across data scales.

## 7. Ablation

Важные ablations:

- w/o MLM loss: Scientific R@10 0.1653 против 0.1761;
- w/o pre-training: Scientific R@10 0.1365, сильное падение;
- w/o token type embedding: Scientific почти не страдает, но Online Retail R@10 падает с 0.2429 до 0.1755;
- removing MLM + token position + token type progressively degrades both Scientific and Online Retail;
- dropping contrastive loss during pre-training снижает Scientific R@10 до 0.1634 и Online Retail R@10 до 0.1853.

Masking ratio analysis показывает peak около 50%; слишком слабое masking дает мало self-supervision, слишком сильное разрушает signal.

## 8. Сильные стороны

- Хорошо подходит для domains, где metadata богата, а interactions sparse.
- JEPA-style embedding prediction полезен для recommendation, потому что target - representation, а не exact text token.
- Работа явно проверяет cross-platform Online Retail, а не только Amazon-to-Amazon transfer.
- Ablations достаточно хорошо разделяют вклад MLM, pre-training, embeddings и contrastive loss.

## 9. Ограничения и вопросы

Метод зависит от качества item text. Если titles/descriptions бедные, noisy или adversarial, text-only representation может проиграть ID/collaborative models.

Online Retail показывает, что dense interaction data может снижать преимущество text transfer. Для production надо комбинировать JEPA-style text embeddings с collaborative features, а не обязательно заменять ID features.

Работа не является semantic-ID tokenizer paper: она не решает discrete decoding, invalid generation и catalog indexing. Ее лучше читать как representation learning baseline для sequential recommendation.

## 10. Вывод

JEPA4Rec показывает, что **embedding-level predictive learning** может быть сильной альтернативой pure MLM/contrastive text recommenders. Главный takeaway для recsys: если item IDs не переносятся между domains, полезно учить item-text representations через masked sequence-to-item prediction, особенно в low-resource и cold-start regimes.
