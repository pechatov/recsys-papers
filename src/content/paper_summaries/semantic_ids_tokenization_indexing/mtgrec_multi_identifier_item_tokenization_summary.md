---
title: "Pre-training Generative Recommender with Multi-Identifier Item Tokenization"
category: "semantic_ids_tokenization_indexing"
slug: "mtgrec_multi_identifier_item_tokenization_summary"
catalogId: "paper-mtgrec_multi_identifier_item_tokenization_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/mtgrec_multi_identifier_item_tokenization_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2504.04400"
---
> **Авторы:** Bowen Zheng, Enze Liu, Zhongfu Chen, Zhongrui Ma, Yue Wang, Wayne Xin Zhao, Ji-Rong Wen.
>
> **Аффилиации:** Renmin University of China; Huawei Poisson Lab.

## 1. Коротко: о чем статья

MTGRec предлагает использовать **несколько semantic identifiers для одного item'а** на этапе pretraining generative recommender. Стандартный TIGER/LETTER-style pipeline жестко связывает item с одним SID. Это удобно для serving, но бедно как training signal: одна user sequence превращается в одну token sequence, а rare items получают мало вариантов представления.

Идея MTGRec: взять несколько соседних checkpoints RQ-VAE tokenizer-а в конце обучения. Они уже близки друг к другу семантически, но дают немного разные item-to-SID assignments. Эти варианты используются как data augmentation: одна и та же history превращается в несколько tokenized sequence groups. Затем model pretraining выбирает группы через curriculum based on data influence, а перед serving модель fine-tune'ится на одном выбранном tokenizer-е.

Главный компромисс: во время обучения item может иметь несколько близких identifiers, но inference остается совместимым с обычным single-SID map.

<figure class="paper-figure">
  <img src="../../assets/mtgrec/framework.png" alt="MTGRec framework with adjacent tokenizer checkpoints and curriculum pretraining">
  <figcaption>Рисунок 1. MTGRec использует соседние RQ-VAE checkpoints как semantically related tokenizers. Multi-identifier variants расширяют pretraining data, а curriculum sampler выбирает наиболее полезные sequence groups.</figcaption>
</figure>

## 2. Контекст и проблема

Generative recommender учится генерировать SID следующего item'а. Если tokenizer назначил item неидеальный SID, downstream model вынуждена учиться на этом единственном представлении. Особенно страдают low-frequency items: мало interactions, мало градиентов, высокая чувствительность к случайному assignment.

Наивный способ - обучить несколько независимых tokenizers и давать item несколько IDs. Но независимые tokenizers могут создавать слишком разные mappings: один и тот же item будет выглядеть как разные объекты, а модель получит noisy labels.

MTGRec выбирает более мягкий источник diversity: adjacent final checkpoints одного tokenizer-а. Они отличаются из-за training trajectory, но остаются близкими по semantic structure.

## 3. Метод

### 3.1. Multi-identifier tokenization

Сначала обучается enhanced RQ-VAE tokenizer. В реализации используются 3 codebooks size 256 и дополнительный collision codebook; также упоминаются TIGER++-style улучшения вроде whitening, deeper MLP и EMA.

Дальше сохраняются последние $n$ checkpoints. Каждый checkpoint перекодирует весь catalog, получая свой SID map. Одна user history затем превращается в $n$ tokenized variants.

### 3.2. Curriculum через data influence

Не все tokenizer variants одинаково полезны. MTGRec оценивает влияние каждой group на validation objective через first-order gradient approximation: если gradient группы должен уменьшить validation loss, sampling probability этой группы растет.

Такой curriculum решает проблему "добавили много вариантов и зашумели training". Модель чаще видит те SID variants, которые реально помогают downstream recommendation.

### 3.3. Fine-tuning для serving

После pretraining на multi-identifier groups модель fine-tune'ится на одном selected tokenizer-е. Это критично: production system не должна одновременно обслуживать несколько противоречивых SID maps. Multi-identifier эффект остается в весах модели как pretraining benefit.

## 4. Пошаговый алгоритм

1. Обучить RQ-VAE tokenizer и сохранить последние $n$ checkpoints.
1. Для каждого checkpoint построить item-to-SID map.
1. Каждую user sequence перекодировать всеми SID maps, получив несколько sequence groups.
1. На каждом pretraining step оценивать полезность groups через gradient influence относительно validation batch.
1. Sampling probabilities обновлять так, чтобы чаще выбирать полезные groups.
1. Pretrain generative recommender на mixture groups.
1. Fine-tune recommender на одном final tokenizer-е.
1. На inference использовать обычный constrained decoding / SID lookup выбранного tokenizer-а.

## 5. Эксперименты

Datasets: Amazon 2023 Musical Instruments, Industrial and Scientific, Video Games. Baselines включают Caser, HGN, GRU4Rec, BERT4Rec, SASRec, FMLP-Rec, HSTU, TIGER, LETTER и TIGER++.

Метрики: Recall@5/10 и NDCG@5/10. Основной результат: MTGRec улучшает TIGER/LETTER-style baselines, а ablations показывают, что contribution дают оба компонента - multiple identifiers и curriculum sampling.

<figure class="paper-figure">
  <img src="../../assets/mtgrec/model_scale.png" alt="MTGRec performance comparison by model scale">
  <figcaption>Рисунок 2. Scale analysis показывает, что эффект MTGRec сохраняется при разных размерах encoder/decoder. Это важно: gain не сводится только к одному удачному model size.</figcaption>
</figure>

<figure class="paper-figure">
  <img src="../../assets/mtgrec/tokenizer_number.png" alt="MTGRec performance by number of tokenizers">
  <figcaption>Рисунок 3. Sensitivity к числу tokenizer checkpoints показывает bias-variance trade-off: слишком мало variants дает мало augmentation, слишком много может добавить шум.</figcaption>
</figure>

<figure class="paper-figure">
  <img src="../../assets/mtgrec/long_tail_items.png" alt="MTGRec long-tail item performance">
  <figcaption>Рисунок 4. Long-tail analysis важен для мотивации MTGRec: multi-identifier pretraining должен помогать именно sparse items, где один SID assignment особенно нестабилен.</figcaption>
</figure>

## 6. Сильные стороны

- Использует "бесплатную" diversity из training trajectory tokenizer-а, а не тренирует много независимых tokenizers.
- Сохраняет простое single-SID inference после fine-tuning.
- Curriculum sampling не считает все augmented groups одинаково полезными.
- Хорошо попадает в проблему rare items и нестабильных semantic assignments.

## 7. Ограничения и риски

Эффект зависит от trajectory RQ-VAE. Если соседние checkpoints почти идентичны, augmentation слабая; если сильно расходятся, labels становятся noisy.

Influence-based curriculum требует дополнительных gradient computations и усложняет distributed training. Для большого production setup это может быть заметной стоимостью.

После fine-tuning на одном tokenizer-е трудно диагностировать, какие именно alternative IDs помогли модели. Multi-identifier benefit становится скрытым pretraining effect.

Versioning остается критичным: model checkpoint, selected SID map и constrained decoding index должны обновляться атомарно.

## 8. Вывод

MTGRec - практичная работа про training-time diversity для semantic-ID generative recommendation. Ее главная идея: item может иметь несколько близких IDs во время обучения, но production может остаться в привычной single-ID схеме. Это хороший компромисс между richer supervision и serving simplicity.
