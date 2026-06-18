---
title: "ETEGRec: Generative Recommender with End-to-End Learnable Item Tokenization"
category: "semantic_ids_tokenization_indexing"
slug: "etegrec_end_to_end_learnable_item_tokenization_summary"
catalogId: "paper-etegrec_end_to_end_learnable_item_tokenization_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/etegrec_end_to_end_learnable_item_tokenization_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2409.05546"
---
> **Авторы:** Enze Liu, Bowen Zheng, Cheng Ling, Lantao Hu, Han Li, Wayne Xin Zhao, Ji-Rong Wen.
>
> **Аффилиации:** Renmin University of China; Kuaishou Technology.

## 1. Коротко: о чем статья

ETEGRec решает decoupling problem между item tokenizer-ом и generative recommender-ом. В TIGER/LETTER tokenizer обучается отдельно, фиксирует item-to-SID map, и только после этого recommender учится генерировать SIDs. ETEGRec связывает эти два модуля end-to-end через alignment objectives и alternating optimization.

Ключевые компоненты:

- **Item tokenizer:** RQ-VAE, обученный на collaborative item embeddings из SASRec.
- **Generative recommender:** T5-like encoder-decoder, генерирующий target SID.
- **SIA (Sequence-Item Alignment):** выравнивает encoder hidden state recommender-а с target item representation.
- **PSA (Preference-Semantic Alignment):** выравнивает decoder preference state с reconstructed semantic embedding target item-а.
- **Alternating training:** tokenizer и recommender обучаются по очереди, чтобы SID churn не дестабилизировал generator.

<figure class="paper-figure">
  <img src="../../assets/etegrec/framework.png" alt="ETEGRec dual encoder-decoder framework with SIA and PSA">
  <figcaption>Рисунок 1. ETEGRec связывает RQ-VAE tokenizer и T5-like recommender двумя alignment losses: SIA работает на encoder/sequence side, PSA - на decoder/preference side.</figcaption>
</figure>

## 2. Почему нужен end-to-end tokenizer

В two-stage pipeline tokenizer оптимизирует reconstruction loss, а recommender оптимизирует next-item generation. Хороший reconstruction не гарантирует хорошие recommendation tokens: model может смешать items, которые похожи в embedding space, но различаются в user behavior.

LETTER добавляет collaborative/diversity regularization в tokenizer, но recommender все равно не участвует в token assignment. ETEGRec делает следующий шаг: recommender hidden states и tokenizer latent distributions обучаются быть согласованными.

## 3. Метод

### 3.1. Item tokenizer

Input - collaborative embedding $\mathbf{z}$ из pretrained SASRec. RQ-VAE encoder переводит его в latent representation, residual quantization выбирает $L=3$ codebook tokens, decoder восстанавливает $\tilde{\mathbf{z}}$. Semantic quantization loss включает reconstruction, codebook и commitment terms:

$$
\mathcal{L}_{SQ} =
\|\mathbf{z}-\tilde{\mathbf{z}}\|^2
+ \sum_{l=1}^{L}
\left(
\|\operatorname{sg}[\mathbf{v}_l]-\mathbf{e}_{c_l}^{l}\|^2
+ \beta\|\mathbf{v}_l-\operatorname{sg}[\mathbf{e}_{c_l}^{l}]\|^2
\right).
$$

### 3.2. SIA: sequence-item alignment

Encoder output recommender-а mean-pooled и проецируется в tokenizer space. Затем target item embedding и sequence representation проходят через tokenizer, получая distributions по codebooks. SIA минимизирует symmetric KL divergence между этими distributions. Интуиция: sequence encoder должен предсказывать не только next token через decoder, но и находиться рядом с target item в tokenizer space.

### 3.3. PSA: preference-semantic alignment

Decoder hidden state интерпретируется как preference state. Reconstructed semantic embedding $\tilde{\mathbf{z}}$ представляет target item. PSA использует symmetric InfoNCE с in-batch negatives, чтобы preference state был близок к правильному target semantics и далек от других items.

### 3.4. Alternating optimization

Полностью совместное обучение нестабильно: если tokenizer постоянно меняет assignments, recommender учится на движущейся target vocabulary. Поэтому ETEGRec использует циклы. В первой эпохе цикла обучается tokenizer:

$$
\mathcal{L}_{IT} = \mathcal{L}_{SQ} + \mu\mathcal{L}_{SIA} + \lambda\mathcal{L}_{PSA}.
$$

В оставшихся эпохах обучается recommender:

$$
\mathcal{L}_{GR} = \mathcal{L}_{REC} + \mu\mathcal{L}_{SIA} + \lambda\mathcal{L}_{PSA}.
$$

После сходимости tokenizer замораживается, а recommender дообучается на fixed final token map. Inference cost остается таким же, как у TIGER, потому что item tokens кэшируются.

## 4. Эксперименты

Datasets: Amazon 2023 Instruments, Scientific и Game. Evaluation: leave-one-out, full ranking, beam size 20. Baselines включают SASRec, FDSA, TIGER и LETTER.

<div class="table-scroll">
<table>
<thead>
<tr><th>Model</th><th>Instrument R@10</th><th>Instrument N@10</th><th>Game R@10</th><th>Game N@10</th></tr>
</thead>
<tbody>
<tr><td>SASRec</td><td>0.0530</td><td>0.0277</td><td>0.0821</td><td>0.0426</td></tr>
<tr><td>FDSA</td><td>0.0557</td><td>0.0295</td><td>0.0857</td><td>0.0453</td></tr>
<tr><td>TIGER</td><td>0.0574</td><td>0.0308</td><td>0.0895</td><td>0.0471</td></tr>
<tr><td>LETTER</td><td>0.0581</td><td>0.0310</td><td>0.0901</td><td>0.0475</td></tr>
<tr><td><strong>ETEGRec</strong></td><td><strong>0.0624</strong></td><td><strong>0.0331</strong></td><td><strong>0.0947</strong></td><td><strong>0.0507</strong></td></tr>
</tbody>
</table>
</div>

ETEGRec превосходит LETTER примерно на 7-8% по Recall@10 на Instrument и около 5% на Game. Улучшения статистически значимы.

<figure class="paper-figure">
  <img src="../../assets/etegrec/seen_unseen_users.png" alt="ETEGRec performance on seen and unseen users">
  <figcaption>Рисунок 2. Анализ seen/unseen users показывает, что alignment помогает не только frequent users: ETEGRec устойчивее работает и на пользователях с короткой историей.</figcaption>
</figure>

## 5. Ablation

Главный ablation: без alternating training качество падает сильнее всего. Это подтверждает, что end-to-end tokenizer нельзя просто обновлять на каждом step без контроля SID churn. Удаление SIA и PSA тоже снижает качество; удаление обоих alignment losses приближает модель к TIGER/LETTER-style baselines.

Вариант "w/o End-to-End" - обучить tokenizer, затем отдельный recommender на финальных tokens - хуже полного ETEGRec. Это поддерживает claim: выигрыш не только от лучших identifiers, но и от совместного alignment во время обучения.

## 6. Сильные стороны

- Прямо атакует decoupling problem между tokenizer и generator.
- SIA и PSA работают на разных сторонах seq2seq architecture, поэтому дополняют друг друга.
- Alternating training сохраняет inference compatibility: serving использует fixed token map.
- Ablation design изолирует SIA, PSA, alternating schedule и end-to-end effect.
- Код открыт в `RUCAIBox/ETEGRec`, что упрощает воспроизведение.

## 7. Ограничения

Training pipeline сложнее two-stage TIGER. Нужно обновлять token maps, переключать frozen/unfrozen модули и следить, чтобы recommender не обучался на устаревших assignments.

SID churn остается production risk. Пока tokenizer меняется, prefix tree, item-to-token map и model checkpoint должны быть синхронизированы.

Эксперименты только offline и среднего масштаба. Нет online A/B и нет проверки на каталоге в миллионы items.

Hyperparameters $\mu$ и $\lambda$ чувствительны к датасету. Перенос на новый домен потребует tuning.

## 8. Вывод

ETEGRec показывает, что tokenizer и generative recommender должны быть aware друг друга. Если LETTER делает tokenizer collaborative-aware, то ETEGRec добавляет feedback loop между tokenizer space и seq2seq hidden states. Главный практический takeaway: end-to-end SID learning требует controlled alternating training, иначе token churn дестабилизирует generator.
