---
title: "Learning Variable-Length Tokenization for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "learning_variable_length_tokenization_for_generative_recommendation_summary"
catalogId: "paper-learning_variable_length_tokenization_for_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/learning_variable_length_tokenization_for_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.17779"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Minhao Wang, Bowen Wu, Wei Zhang.
>
> **Аффилиации:** East China Normal University; Shanghai Innovation Institute.
>
> **Источник:** [arXiv 2605.17779](https://arxiv.org/abs/2605.17779) · дата metadata: 2026-05-18.
>
> **Категория/теги:** semantic IDs, tokenization, generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: ставит под сомнение фиксированную длину SID: популярным item часто хватает короткого кода, а tail item нужны более длинные и детальные identifiers.
- Алгоритм: VarLenRec использует Popularity-Weighted Information Budget Allocation, Hyperbolic Residual Quantization и Soft Length Controller для дифференцируемого выбора длины.
- Evidence: Авторы заявляют систематический Popularity-Length Paradox на четырех датасетах и улучшение качества за счет распределения capacity по популярности.
- Ограничение: Variable-length decoding усложняет trie, latency и batching; также нужно следить, не станет ли popularity prior новым источником bias.
- Итог: Это важная работа рядом с CapsID и long-SID papers: capacity semantic tokenizer лучше распределять адаптивно, а не одинаково для всего каталога.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Directly implementing variable-length allocation faces two technical challenges: standard Euclidean residual quantization lacks geometric capacity to support diverse code lengths without distortion, and discrete length decisions are non-differentiable.</td></tr>
<tr><th>Предложение авторов</th><td>To address this, we propose VarLenRec, a framework for learning variable-length tokenization.</td></tr>
<tr><th>Главный evidence claim</th><td>Through systematic experiments across four datasets, we discover the Popularity-Length Paradox: popular items achieve optimal performance with short IDs, while tail items require substantially longer codes to capture discriminative semantics.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>negative sampling</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Amazon, Beauty, Sports, Toys, Yelp</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, Success, accuracy</td></tr>
<tr><th>Baselines</th><td>TIGER, LETTER, ETEGRec, SASRec, BERT4Rec, GRU4Rec, HSTU, RQ-VAE</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Methodology: We present our approach in four parts: the empirical discovery motivating our work ( ), the theoretical foundation ( ), the Hyperbolic Adaptive Residual Quantization (HARQ) architecture ( ), and the downstream generative recommendation ( ). Figure illustrates the complete architecture.
1. **Ключевой модуль.** HARQ Architecture: We present HARQ, which performs adaptive-length tokenization in hyperbolic space through residual quantization. The architecture consists of a hyperbolic encoder, a hyperbolic residual quantizer, a soft length controller, and a hyperbolic decoder.
1. **Learning signal.** HARQ Architecture: Standard vector quantization methods operate in Euclidean space with polynomial volume growth V Eucl (r) r d. However, item semantics follow hierarchical structures where broad categories branch into subcategories. This creates a geometric mismatch: tree hierarchies contain exponentially many leaf nodes, while Euclidean balls provide only polynomial...
1. **Inference / deployment path.** HARQ Architecture: In contrast, hyperbolic geometry provides exponential volume growth. In the Poincar ' e ball model c d = x d: |x | < 1/ c with curvature -c (c>0), volume grows as V Hyp (r) d-1 ( c r) e (d-1) c r for large r ratcliffe2006foundations. This directly matches hierarchical branching, enabling distortion-free tree embeddings sarkar2011low.

## 5. Objectives, formulas и training details

**Detected objective keywords:** negative sampling.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `_ I(X; Z) s.t. |Z| = L.`
- `_L_i, I(X; Z_1:L_i) - E[L_i]\,,`
- `I_ collab(p_i) = (1 + p_i)\,,`
- `I_ semantic(L_i) = _l=1^L_i 1l = H_L_i L_i \,,`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- The Popularity-Length Paradox. NDCG@10 on different target groups (Head: Top 20\
- Overall architecture of VarLenRec.
- Overall performance comparison. The best and second-best results are marked in bold and underlined, respectively.
- Ablation study results on Beauty and Toys. We ablate three key components: training objectives ($ L_ cost$ and $ L_ len$), hyperbolic space, and inference integration strategies.
- Training and testing efficiency comparison (seconds per epoch) across the four datasets. VarLenRec consistently achieves faster training and inference.
- Hyperparameter sensitivity analysis.
- ID collision rates (\ Our full VarLenRec with HARQ consistently achieves the lowest collision rates.
- Distribution of learned semantic ID lengths by VarLenRec across the four datasets (maximum length = 10). The bars show the percentage of items in each length range, while the line plot indicates the average length for each dataset.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Amazon, Beauty, Sports, Toys, Yelp</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, Success, accuracy</td></tr>
<tr><th>Baselines</th><td>TIGER, LETTER, ETEGRec, SASRec, BERT4Rec, GRU4Rec, HSTU, RQ-VAE</td></tr>
</tbody></table>
</div>

- Evidence: Авторы заявляют систематический Popularity-Length Paradox на четырех датасетах и улучшение качества за счет распределения capacity по популярности.
- Experiments: In this section, we conduct comprehensive experiments to answer the following research questions: RQ1. How does integrating VarLenRec into state-of-the-art generative recommendation frameworks improve performance compared to both traditional sequential recommenders and existing fixed-length generative baselines? RQ2: How do different components of VarLenRec...
- Experimental Setup: Datasets We evaluate our method on four widely-used benchmark datasets: Amazon Beauty, Sports and Outdoors, Toys and Games ni2019justifying, and Yelp. Following previous work rajput2023tiger,wang2024letter, we filter out users and items with fewer than 5 interactions.
- Experimental Setup: Baselines We evaluate VarLenRec by integrating it with generative recommendation backbones and comparing against three categories of baselines. The first category includes traditional sequential recommendation methods: HGN ma2019hgn, GRU4Rec hidasi2016gru4rec, BERT4Rec sun2019bert4rec, SASRec kang2018sasrec, FMLP zhou2022fmlp, S 3 Rec zhou2020s3rec,...
- Through systematic experiments across four datasets, we discover the Popularity-Length Paradox: popular items achieve optimal performance with short IDs, while tail items require substantially longer codes to capture discriminative semantics.
- Extensive experiments demonstrate that VarLenRec achieves significant improvements over state-of-the-art methods in recommendation accuracy and training/inference efficiency, revealing the importance of adaptive encoding capacity in generative recommendation.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: ставит под сомнение фиксированную длину SID: популярным item часто хватает короткого кода, а tail item нужны более длинные и детальные identifiers.
- **Algorithmic/system:** Алгоритм: VarLenRec использует Popularity-Weighted Information Budget Allocation, Hyperbolic Residual Quantization и Soft Length Controller для дифференцируемого выбора длины.
- **Empirical:** Evidence: Авторы заявляют систематический Popularity-Length Paradox на четырех датасетах и улучшение качества за счет распределения capacity по популярности.
- **Practical:** Ограничение: Variable-length decoding усложняет trie, latency и batching; также нужно следить, не станет ли popularity prior новым источником bias.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.
- Reward/utility signal достаточно стабилен и не подменяет user relevance узкой бизнес-метрикой.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Amazon, Beauty, Sports, Toys, Yelp.
- Метрики: Recall, NDCG, Success, accuracy.
- Baselines и parity settings: TIGER, LETTER, ETEGRec, SASRec, BERT4Rec, GRU4Rec, HSTU, RQ-VAE.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Это важная работа рядом с CapsID и long-SID papers: capacity semantic tokenizer лучше распределять адаптивно, а не одинаково для всего каталога.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: ставит под сомнение фиксированную длину SID: популярным item часто хватает короткого кода, а tail item нужны более длинные и детальные identifiers.
- Алгоритм: VarLenRec использует Popularity-Weighted Information Budget Allocation, Hyperbolic Residual Quantization и Soft Length Controller для дифференцируемого выбора длины.
- Evidence: Авторы заявляют систематический Popularity-Length Paradox на четырех датасетах и улучшение качества за счет распределения capacity по популярности.
- Ограничение: Variable-length decoding усложняет trie, latency и batching; также нужно следить, не станет ли popularity prior новым источником bias.
- Итог: Это важная работа рядом с CapsID и long-SID papers: capacity semantic tokenizer лучше распределять адаптивно, а не одинаково для всего каталога.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td> figure [!t] Figures/ndcg 2x2 grid.pdf -.5em The Popularity-Length Paradox. NDCG@10 on different target groups (Head: Top 20</td></tr>
<tr><td>Related Work</td><td>In this section, we review the related work in two major aspects: sequential recommendation and generative recommendation.</td></tr>
<tr><td>Sequential Recommendation</td><td>Sequential recommendation aims to predict the next item a user may interact with based on the user's historical behavior sequences. Early studies rendle2010fpmc,he2016fusing primarily adhere to the Markov Chain assumption and focus on estimating the...</td></tr>
<tr><td>Generative Recommendation</td><td>Generative recommendation has emerged as a next-generation paradigm for recommendation systems li2024survey,deldjoo2024review. In such a generative paradigm, the item sequence is tokenized into a token sequence and then fed into generative models to predict...</td></tr>
<tr><td>Methodology</td><td>We present our approach in four parts: the empirical discovery motivating our work ( ), the theoretical foundation ( ), the Hyperbolic Adaptive Residual Quantization (HARQ) architecture ( ), and the downstream generative...</td></tr>
<tr><td>The Popularity-Length Paradox</td><td>Let I = i 1, i 2,, i N denote the item set. Each item i I has content features x i F and popularity p i, defined as the normalized interaction frequency in the training set. Standard RQ-VAE assigns each item a fixed-length semantic ID z i = (z i (1), z i...</td></tr>
<tr><td>Variable-Length Allocation</td><td>We formalize the length allocation problem from an information-theoretic perspective and derive a principled mapping from item popularity to optimal ID length.</td></tr>
<tr><td>HARQ Architecture</td><td>We present HARQ, which performs adaptive-length tokenization in hyperbolic space through residual quantization. The architecture consists of a hyperbolic encoder, a hyperbolic residual quantizer, a soft length controller, and a hyperbolic decoder.</td></tr>
<tr><td>Variable-Length ID Integration</td><td>After training HARQ, we integrate the variable-length semantic IDs into the downstream Transformer-based generative recommendation model. During training, given a user's interaction history as a sequence of item IDs, we train the model to predict the next...</td></tr>
<tr><td>Experiments</td><td>In this section, we conduct comprehensive experiments to answer the following research questions: RQ1. How does integrating VarLenRec into state-of-the-art generative recommendation frameworks improve performance compared to both traditional sequential...</td></tr>
<tr><td>Experimental Setup</td><td>Datasets We evaluate our method on four widely-used benchmark datasets: Amazon Beauty, Sports and Outdoors, Toys and Games ni2019justifying, and Yelp. Following previous work rajput2023tiger,wang2024letter, we filter out users and items with fewer than 5...</td></tr>
<tr><td>Overall Performance (RQ1)</td><td>When integrated with generative backbones, VarLenRec outperforms all existing semantic ID-based methods. VarLenRec-LCRec achieves the best results on three datasets (Beauty, Sports, Toys), while VarLenRec-TIGER performs best on Yelp. Against ETEGRec and...</td></tr>
</tbody></table>
</div>
