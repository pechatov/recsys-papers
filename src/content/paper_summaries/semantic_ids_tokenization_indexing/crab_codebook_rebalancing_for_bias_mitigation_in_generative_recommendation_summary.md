---
title: "CRAB: Codebook Rebalancing for Bias Mitigation in Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "crab_codebook_rebalancing_for_bias_mitigation_in_generative_recommendation_summary"
catalogId: "paper-crab_codebook_rebalancing_for_bias_mitigation_in_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/crab_codebook_rebalancing_for_bias_mitigation_in_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.05113"
---
> **Авторы:** Zezhong Fan, Ziheng Chen, Luyi Ma, Jin Huang, Lalitesh Morishetti, Kaushiki Nag, Sushant Kumar, Kannan Achan.
>
> **Источник:** arXiv:2604.05113.

## 1. Коротко: о чем статья

CRAB изучает popularity bias в generative recommendation с semantic IDs. Главный тезис: bias появляется не только в ranker loss или exposure logs, но и в самом **codebook**. Если popular items часто попадают в одни и те же semantic tokens, эти tokens получают больше training signal, становятся еще сильнее и начинают доминировать в generation.

Авторы выделяют два механизма. Первый - imbalanced tokenization: popular items и их похожие соседи концентрируются в over-popular tokens. Второй - training imbalance: autoregressive generator чаще видит popular token paths и хуже обучает редкие ветки code tree.

CRAB предлагает post-hoc debiasing: найти over-popular tokens, split-ить их child subtrees через regularized KMeans и дообучить модель с tree-structured regularizer, который сохраняет semantic consistency между parent и newly split tokens.

<figure class="paper-figure">
  <img src="../../assets/crab/popularity_bias.png" alt="CRAB popularity bias and generation utility by token popularity">
  <figcaption>Рисунок 1. CRAB показывает, что popularity bias в GeneRec связан с token popularity: популярные token groups получают большую generation utility, а tail groups системно недополучают качество.</figcaption>
</figure>

## 2. Контекст: почему bias на уровне codebook важен

В TIGER/CoST/LETTER-style системах item представляется sequence of semantic tokens. Если codebook сбалансирован плохо, generator видит неравномерный язык: часть token prefixes встречается очень часто, часть почти отсутствует. Это похоже на class imbalance, но на каждом уровне semantic hierarchy.

Обычные debiasing методы для recommendation работают на уровне sampling, loss weighting или reranking. CRAB переносит debiasing внутрь identifier space: если популярный token перегружен, нужно изменить структуру codebook, а не только штрафовать output probability.

## 3. Метод CRAB

CRAB состоит из двух этапов.

1. **Codebook rebalancing.** Для уже обученного semantic-ID codebook находятся over-popular tokens. Такой token split-ится на несколько новых child tokens. Redistribution делается не произвольно: regularized KMeans учитывает residual representations child tokens/items и балансировку частот.
1. **Hierarchical semantic alignment.** После split появляются новые token embeddings. Чтобы они не разрушили semantic hierarchy, CRAB добавляет tree-structured regularizer: siblings и parent-related tokens должны оставаться семантически согласованными.

<figure class="paper-figure">
  <img src="../../assets/crab/codebook_rebalancing.png" alt="CRAB three-level codebook rebalancing illustration">
  <figcaption>Рисунок 2. Основная механика CRAB: over-popular token не удаляется, а дробится через перераспределение child tokens. Это сохраняет hierarchy, но уменьшает frequency concentration.</figcaption>
</figure>

### 3.1. Почему post-hoc split

Post-hoc дизайн практичен: не нужно заново обучать tokenizer с нуля. Система берет уже существующий codebook и исправляет самые перегруженные области. Это особенно важно для industrial GeneRec, где tokenizer, trie, item-to-SID map и trained generator являются дорогими artifacts.

Но post-hoc split создает migration risk. Если SID mapping меняется, downstream generator должен увидеть новые tokens и научиться их использовать. Поэтому CRAB включает retraining/optimization step с recommendation loss и tree regularizer.

## 4. Objective и training

Финальная оптимизация объединяет recommendation objective и regularizer:

$$
\mathcal{L} = \mathcal{L}_{Rec} + \lambda \mathcal{L}_{T}.
$$

$\mathcal{L}_{Rec}$ отвечает за обычную next-item / next-token recommendation задачу. $\mathcal{L}_{T}$ сохраняет semantic consistency в tree after splitting. Гиперпараметр $\lambda$ управляет тем, насколько сильно модель должна удерживать hierarchy.

Важно: CRAB не пытается максимизировать uniqueness любой ценой. Его цель - уменьшить frequency imbalance, не уничтожив semantic grouping. Поэтому правильные diagnostics должны смотреть одновременно на head/tail performance, token distribution и item-level recommendation metrics.

<figure class="paper-figure">
  <img src="../../assets/crab/splitting_sensitivity.png" alt="CRAB sensitivity to splitting ratio and split level">
  <figcaption>Рисунок 3. Sensitivity analysis показывает, что результат зависит от splitting ratio и уровня hierarchy. Это важный production knob: слишком агрессивный split может разрушить sharing, слишком слабый не снимет bias.</figcaption>
</figure>

## 5. Эксперименты

CRAB проверяется на Industrial и Office datasets. Baselines включают SASRec, TIGER, CoST, RQ-VAE-based variants и OneRec/MOR-style GeneRec setup. Метрики: HR@K, NDCG@K и efficiency.

Главный результат: CRAB улучшает recommendation performance и одновременно снижает popularity bias. По тексту статьи, gain особенно важен для менее популярных item groups, потому что rebalancing дает им более различимые token paths и больше полезного training signal.

## 6. Что доказывает статья

Самый сильный contribution - диагностический. CRAB показывает, что popularity bias можно увидеть в token distribution и generation utility, а не только в final ranked list. Это полезно для любого SID pipeline: codebook auditing должен быть обязательной частью оценки tokenizer-а.

Алгоритмический contribution - split over-popular tokens с сохранением hierarchy. Это не новый tokenizer с нуля, а ремонт существующего codebook. Такой подход ближе к production reality, где mapping нельзя часто полностью перестраивать.

## 7. Как внедрять и проверять

Практический rollout CRAB лучше делать как migration experiment.

Сначала нужно построить audit текущего SID tree: token frequency по уровням, head/tail item distribution внутри каждого prefix, dead-token fraction, average generation probability по prefix и utility per token bucket. Затем выбираются over-popular tokens, но split применяется не сразу ко всему каталогу, а к ограниченному subset или shadow tree.

После split нужны две группы метрик. Первая - structural: насколько снизилась концентрация popularity, выросла ли utilization, не увеличились ли collisions и не вырос ли prefix churn. Вторая - recommendation: HR/NDCG по popularity buckets, head-item regression, tail uplift, invalid generation rate и latency constrained decoding.

Особенно важно хранить mapping old SID -> new SID и item-level audit trail. Без этого трудно объяснить online изменения: улучшение tail может быть связано с rebalancing, а может быть побочным эффектом retraining generator-а на другой token distribution.

## 8. Сильные стороны

- Четко связывает popularity bias с semantic-token imbalance.
- Предлагает post-hoc correction, а не полный rebuild tokenizer-а.
- Учитывает hierarchy: split работает с parent-child structure, а не просто создает новые flat clusters.
- Sensitivity analysis помогает понять, какие knobs надо мониторить.

## 9. Ограничения и риски

Главный риск - SID churn. После split часть item paths меняется, а это влияет на trie, caches, generator outputs и offline/online consistency.

Второй риск - head-tail trade-off. Если aggressively split-ить popular tokens, можно улучшить tail, но ухудшить head items, которые дают большую долю traffic. Нужны per-bucket и business-weighted metrics.

Третий риск - reproducibility. Industrial dataset закрыт, а production bias может сильно зависеть от exposure policy и traffic mix.

Еще один риск - неправильный target bias. Если exposure policy сама по себе biased, то split over-popular tokens может лечить симптом в codebook, но не причину в logging/retrieval/ranking loop. Поэтому CRAB стоит сочетать с debiasing diagnostics на уровне impressions, clicks и training sampler-а.

## 10. Вывод

CRAB стоит читать как paper про **codebook governance**. В semantic-ID recommendation мало построить коды один раз: их нужно регулярно аудировать на imbalance, over-popular prefixes, dead/rare tokens и head-tail effects. Главный takeaway: popularity bias может быть зашит в identifier space, и исправлять его иногда нужно именно на уровне codebook.
