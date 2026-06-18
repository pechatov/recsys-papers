---
title: "CARD: Non-Uniform Quantization of Visual Semantic Unit for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "card_non_uniform_quantization_of_visual_semantic_unit_for_generative_recommendation_summary"
catalogId: "paper-card_non_uniform_quantization_of_visual_semantic_unit_for_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/card_non_uniform_quantization_of_visual_semantic_unit_for_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.26427"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Yibiao Wei, Jie Zou, Pengfei Zhang, Xiao Ao, Weikang Guo, Zeyu Ma, Yang Yang.
>
> **Аффилиации:** University of Electronic Science and Technology of China; Southwestern University of Finance and Economics.
>
> **Источник:** [arXiv 2604.26427](https://arxiv.org/abs/2604.26427) · дата metadata: 2026-04-29.
>
> **Категория/теги:** semantic IDs, quantization, generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://github.com/HAI-UESTC/CARD](https://github.com/HAI-UESTC/CARD).

## 1. Коротко

- Главная идея: CARD фокусируется на visual semantic units и non-uniform quantization для multimodal generative recommendation.
- Алгоритм: Метод сначала объединяет textual, visual и collaborative signals в visual semantic unit, затем NU-RQ-VAE применяет invertible non-uniform transform, чтобы сбалансировать skewed latent distribution.
- Evidence: На нескольких Amazon datasets авторы показывают стабильные улучшения и лучшую codebook utilization/quantization accuracy.
- Ограничение: Качество зависит от visual features; для товаров без хороших изображений или с noisy multimodal content эффект может упасть.
- Итог: Полезна для multimodal SID: проблема не только в fusion, но и в том, что item embeddings обычно распределены неравномерно.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>While existing studies have sought to enhance SID construction by incorporating multimodal content, collaborative signals, or more advanced quantization techniques, learning high-quality SIDs still faces two key challenges: (1) The two-stage generative recommendation paradigm (SID construction and autoregressive generation)...</td></tr>
<tr><th>Предложение авторов</th><td>To address these challenges, we propose a novel generative recommendation framework, called CARD.</td></tr>
<tr><th>Главный evidence claim</th><td>While existing studies have sought to enhance SID construction by incorporating multimodal content, collaborative signals, or more advanced quantization techniques, learning high-quality SIDs still faces two key challenges: (1) The two-stage generative recommendation paradigm (SID construction and autoregressive generation)...</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>contrastive, MSE, reconstruction</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Amazon, Cell</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, MAP, Success, accuracy</td></tr>
<tr><th>Baselines</th><td>TIGER, LETTER, OneRec, SASRec, GRU4Rec, HSTU, LightGCN, RQ-VAE, VQ-VAE, LC-Rec</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** METHODOLOGY: This section introduces the CARD framework, as illustrated in CARD first unifies heterogeneous semantic information into visual representations by constructing visual semantic units, then applies a Non-Uniform Residual Quantized Variational Autoencoder (NU-RQ-VAE) to quantize non-uniform semantic embeddings, and finally performs generative...
1. **Ключевой модуль.** Non-Uniform Quantization Module: The quantization module transforms embeddings z i into discrete SID sequences, whose quality is crucial for downstream generative models to capture fine-grained item semantics. RQ-VAE RQ-VAE optimizes a global reconstruction loss (e.g., MSE), implicitly allocating codebook capacity uniformly across the latent space. This assumption often breaks in...
1. **Learning signal.** Non-Uniform Quantization Module: RQ-VAE Quantization Process The RQ-VAE consists of an Encoder( ), a Decoder( ), and K codebooks C 1,, C K. Specifically, for each code level k 1,,K, we have a codebook C k= e i i=1 N, where e i R d is a learnable code embedding and N denotes the codebook size.
1. **Inference / deployment path.** Non-Uniform Quantization Module: For an item, we first project its embedding z into the latent space via the encoder h = Encoder( z ). This h is then quantized through K successive codebook levels via residual quantization. The residual quantization process can then be formally expressed as: equation aligned &c k= i | r k-1 - e i | 2, e i C k, & r k= r k-1 - e c k, aligned. equation...
1. **Проверяемая деталь.** Training and Recommendation: . The training of generative recommender models consists of two phases: quantization and model optimization. Each item is quantized into an SID i = [c 1, c 2,, c K] using the well-trained NU-RQ-VAE. Based on these SIDs, user interaction sequences are translated into code sequences. We construct a training dataset D = (x ( ), y ( ) ) =1 | D |, where x ( )...
1. **Проверяемая деталь.** Training and Recommendation: . During inference, the generative recommender produces the code sequence of the next item in an autoregressive fashion. At each decoding step t, the model estimates a conditional probability distribution P (v y <t ( ), x) over the code vocabulary V. Sequence decoding is carried out using beam search, which maintains the top- B most probable partial...

## 5. Objectives, formulas и training details

**Detected objective keywords:** contrastive, MSE, reconstruction.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `N(i) = N I \i\, | N|=K_ neighbors argmax _j N s(i, j),`
- `z = f(G_i) R^m.`
- `\ aligned &c_k= _i\| r_k-1- e_i\|^2, e_i C_k, \\ & r_k= r_k-1- e_c_k, aligned.`
- `\ aligned & L_ Recon = \| z - z\|^2, \\ & L_ RQ = _k=1^K (\| sg[r_k-1] - e_c_k\|^2 + \| r_k-1 - sg[e_c_k]\|^2), \\ & L_ Sem = L_ Recon + L_ RQ, aligned.`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- The left shows item embeddings in a 2D PCA space, with dense clusters and sparse regions (darker colors indicate higher density). The right shows skewed codeword usage, where a few frequent codewords dominate and are further amplified in generated SIDs, causing generation bias.
- Overview of the CARD framework. CARD constructs a visual semantic unit for each item by unifying visual, textual, and collaborative signals within the image modality. Each unit is encoded by SigLIP2 and processed by a Non-Uniform Residual Quantization Variational Autoencoder (NU-RQ-VAE), where a non-uniform transformation and multi-level residual...
- Examples of visual semantic units on the Clothing and Food datasets.
- PDF (left) and CDF (right) of the Kumaraswamy distribution. The monotonic and invertible CDF enables reparameterizing non-uniform distributions into an approximately uniform space, making it suitable for distribution-aware quantization.
- The left shows the embedding distributions of 10 samples from the T5 encoder on the Food dataset, and the right shows those from the SigLIP2 encoder. In both cases, the distributions exhibit an approximately bell-shaped form.
- Dataset statistics. “Avg. len” denotes the mean length of user interaction sequences.
- Top- K recommendation performance on three datasets. Best results are in bold, and best baselines are underlined. $^*$ indicates $p 0.01$ in paired t-tests against the best baseline. CARD$_K$ and CARD$_S$ denote variants with Kumaraswamy-based and scaled logistic/logit-based non-uniform quantization, respectively.
- Ablation study of CARD$_K$ on the Food and Phones datasets.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Amazon, Cell</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG, MAP, Success, accuracy</td></tr>
<tr><th>Baselines</th><td>TIGER, LETTER, OneRec, SASRec, GRU4Rec, HSTU, LightGCN, RQ-VAE, VQ-VAE, LC-Rec</td></tr>
</tbody></table>
</div>

- Evidence: На нескольких Amazon datasets авторы показывают стабильные улучшения и лучшую codebook utilization/quantization accuracy.
- Experimental Setup: Datasets. We conduct experiments on three categories Amazon amazon-2014 datasets: "Grocery and Gourmet Food" (Food), "Cell Phones and Accessories" (Phones), and "Clothing, Shoes and Jewelry" (Clothing). Following prior work SASRec,TIGER,LETTER, we apply the 5-core filtering strategy to remove users and items with fewer than five interactions. To...
- Experimental Setup: Evaluation Metrics. We adopt Top- K Recall (Recall@ K ) and Normalized Discounted Cumulative Gain (NDCG@ K ) as evaluation metrics, and assess recommendation performance with K = 5, 10, 20. We conduct a full-ranking TIGER,LETTER evaluation over the entire item set.
- While existing studies have sought to enhance SID construction by incorporating multimodal content, collaborative signals, or more advanced quantization techniques, learning high-quality SIDs still faces two key challenges: (1) The two-stage generative recommendation paradigm (SID construction and autoregressive generation)...
- Furthermore, to deal with the highly non-uniform distribution of item semantic embeddings in recommendation scenarios, we develop a non-uniform quantization framework (NU-RQ-VAE), which incorporates a learnable and invertible non-uniform transformation into the quantization process to map skewed semantic distributions into a...
- Experiments on multiple datasets show that CARD consistently outperforms baseline methods under various settings; meanwhile, the proposed non-uniform transformation module is plug-and-play and remains robust across different quantization schemes.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: CARD фокусируется на visual semantic units и non-uniform quantization для multimodal generative recommendation.
- **Algorithmic/system:** Алгоритм: Метод сначала объединяет textual, visual и collaborative signals в visual semantic unit, затем NU-RQ-VAE применяет invertible non-uniform transform, чтобы сбалансировать skewed latent distribution.
- **Empirical:** Evidence: На нескольких Amazon datasets авторы показывают стабильные улучшения и лучшую codebook utilization/quantization accuracy.
- **Practical:** Ограничение: Качество зависит от visual features; для товаров без хороших изображений или с noisy multimodal content эффект может упасть.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Amazon, Cell.
- Метрики: Recall, NDCG, MAP, Success, accuracy.
- Baselines и parity settings: TIGER, LETTER, OneRec, SASRec, GRU4Rec, HSTU, LightGCN, RQ-VAE, VQ-VAE, LC-Rec.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна для multimodal SID: проблема не только в fusion, но и в том, что item embeddings обычно распределены неравномерно.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: CARD фокусируется на visual semantic units и non-uniform quantization для multimodal generative recommendation.
- Алгоритм: Метод сначала объединяет textual, visual и collaborative signals в visual semantic unit, затем NU-RQ-VAE применяет invertible non-uniform transform, чтобы сбалансировать skewed latent distribution.
- Evidence: На нескольких Amazon datasets авторы показывают стабильные улучшения и лучшую codebook utilization/quantization accuracy.
- Ограничение: Качество зависит от visual features; для товаров без хороших изображений или с noisy multimodal content эффект может упасть.
- Итог: Полезна для multimodal SID: проблема не только в fusion, но и в том, что item embeddings обычно распределены неравномерно.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>INTRODUCTION</td><td>Recommender systems SASRec,GRU4Rec,Qrec,MCCRS,GeoCRS,TSCRKG,yaoyan,lijingzhi,DivReason play a crucial role in modern information services by connecting users with massive information resources. Traditional recommendation methods...</td></tr>
<tr><td>RELATED WORK</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Generative Recommendation</td><td>In recent years, inspired by the success of large language models (LLMs), generative recommendation GenCDR,GRAM,GRID,TIGER,RPG,ActionPiece,GNPR-SID has emerged as an important research direction in recommender systems. Such methods typically represent items...</td></tr>
<tr><td>Multi-modal Recommendation</td><td>Multi-modal recommendation m1,m2,m3,MSCRS,FUMMER,MCCL enhances performance by leveraging the multi-modal features of items. Earlier models VBPR,MMGCN primarily used such signals to complement ID-based collaborative filtering. More recent studies MMGCN,MMSSL,...</td></tr>
<tr><td>METHODOLOGY</td><td>This section introduces the CARD framework, as illustrated in CARD first unifies heterogeneous semantic information into visual representations by constructing visual semantic units, then applies a Non-Uniform Residual Quantized Variational...</td></tr>
<tr><td>Card-style Unified Item Representation</td><td>Existing SID construction methods typically use modality-specific encoders to separately model textual, visual, and collaborative signals. This separated modeling places heterogeneous information in disjoint representation spaces, so cross-modal relations are...</td></tr>
<tr><td>Non-Uniform Quantization Module</td><td>The quantization module transforms embeddings z i into discrete SID sequences, whose quality is crucial for downstream generative models to capture fine-grained item semantics. RQ-VAE RQ-VAE optimizes a global reconstruction loss (e.g., MSE), implicitly...</td></tr>
<tr><td>Training and Recommendation</td><td>. The training of generative recommender models consists of two phases: quantization and model optimization. Each item is quantized into an SID i = [c 1, c 2,, c K] using the well-trained NU-RQ-VAE. Based on these SIDs, user interaction sequences are...</td></tr>
<tr><td>EXPERIMENTS</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Experimental Setup</td><td>Datasets. We conduct experiments on three categories Amazon amazon-2014 datasets: "Grocery and Gourmet Food" (Food), "Cell Phones and Accessories" (Phones), and "Clothing, Shoes and Jewelry" (Clothing). Following prior work SASRec,TIGER,LETTER, we apply the...</td></tr>
<tr><td>Overall Performance</td><td>More specifically, the performance gains of CARD arise from two complementary designs. First, by unifying multimodal content and collaborative semantics at the encoding stage via visual semantic units, CARD mitigates the instability and semantic gap...</td></tr>
<tr><td>Ablation Study</td><td>текст не извлечен; смотреть PDF/source</td></tr>
</tbody></table>
</div>
