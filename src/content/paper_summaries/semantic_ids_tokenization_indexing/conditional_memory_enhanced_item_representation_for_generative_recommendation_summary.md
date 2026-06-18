---
title: "Conditional Memory Enhanced Item Representation for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "conditional_memory_enhanced_item_representation_for_generative_recommendation_summary"
catalogId: "paper-conditional_memory_enhanced_item_representation_for_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/conditional_memory_enhanced_item_representation_for_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.11447"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Ziwei Liu, Yejing Wang, Shengyu Zhou, Xinhang Li, Xiangyu Zhao.
>
> **Аффилиации:** City University of Hong Kong; Independent Researcher; Tsinghua University.
>
> **Источник:** [arXiv 2605.11447](https://arxiv.org/abs/2605.11447) · дата metadata: 2026-05-12.
>
> **Категория/теги:** semantic IDs, generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: сдвигает фокус с самого tokenizer на item-level input representation: простое склеивание SID-token embeddings теряет структуру и усиливает коллизии.
- Алгоритм: ComeIR использует MM-guided token scoring, dual-level Engram memory и token-granular decoding, чтобы сохранить intra-item composition и inter-item transitions.
- Evidence: Эксперименты показывают, что улучшение representation construction может давать gains без полной замены tokenizer.
- Ограничение: Memory-компоненты увеличивают состояние модели и требуют аккуратной оценки serving/memory overhead.
- Итог: Полезна как напоминание: после построения SID еще остается отдельная задача, как превратить SID tokens в хороший item representation для encoder.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>However, these item-level constructors still expose two practical challenges: direct merging may amplify the information loss caused by quantization and ID collision while obscuring SID code relations, whereas external-input-based methods can strengthen item semantics but cannot reliably preserve the SID-structured evidence...</td></tr>
<tr><th>Предложение авторов</th><td>To this end, we propose ComeIR, a Conditional Memory enhanced Item Representation framework that reconstructs SID-token embeddings into item-aware inputs and restores the token granularity during SID decoding.</td></tr>
<tr><th>Главный evidence claim</th><td>Extensive experiments demonstrate the effectiveness and flexibility of ComeIR, and further reveal scalable gains from enlarging conditional memory.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>cross-entropy</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Amazon, Yelp</td></tr>
<tr><th>Metrics</th><td>Hit, latency, MAP, Success</td></tr>
<tr><th>Baselines</th><td>RQ-VAE</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Method: figure* [!t] FigureMaking/ComeIR.pdf The overall framework of the proposed. The code layer L is set to 3 for illustration. figure*
1. **Ключевой модуль.** Framework Overview: The overview of our proposed framework is illustrated in Given the SID-level representations R n S defined in Section, our target is to model a fine-grained f( ) that maps each SID into an item-level representation while preserving the structured SID evidence needed for autoregressive decoding. Accordingly, instead of...
1. **Learning signal.** Framework Overview: Specifically, our representation construction contains three coupled components. First, MM-guided Token Scoring reuses the cached multimodal item embeddings M =[ m 1,, m N], from which the SIDs are produced, to estimate the contribution of each code-layer embedding within an SID @. This cached embedding acts as an identity query tied to SID construction,...
1. **Inference / deployment path.** MM-guided Token Scoring: As discussed in the Identity-Structure Preservation Conflict, item-level compression should retain item-specific identity without discarding SID-structured evidence. We first address the identity side through MM-guided Token Scoring. The guidance comes from the same multimodal embedding used by the quantizer to produce the item's SID, so it acts as an...
1. **Проверяемая деталь.** Memory-conditioned Token Merge: For a specific item v n, after obtaining the MM-guided item context s n 0 and dual-level memories, n, S and n, T at each code level, we need to transform multiple signals into one item-level input. However, a naive merge collapses all code embeddings at once and cannot recognize item identity or the SID structure. Unlike external-input-based constructors...
1. **Проверяемая деталь.** Training and Inference: infer Training. Given the ground-truth next-item SID c N+1, we train with teacher forcing and token-level cross-entropy over SID layers. At layer, the ground-truth prefix c N+1 < is provided to the prediction head, so the model learns to score the next code under valid historical and prefix-conditioned memory evidence. The training loss can then...
1. **Проверяемая деталь.** Training and Inference: Inference. During inference, first constructs R I and obtains h u from the LLM. It then performs beam search over SID layers, while each candidate is scored by restoring the same intra-item and inter-item memories used in training. Detailed settings for different architectures, normal GR and NEZHA, can be found in Appendix infer supp.

## 5. Objectives, formulas и training details

**Detected objective keywords:** cross-entropy.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `_n^ = (q_n^ e_c_n^ / d) _r=1^L (q_n^ e_c_n^r/ d), s_n^0 = _ =1^L _n^ e_c_n^`
- `R_ (q, p) = LN (_o O_ _,o W_V,,o a_,o(p))`
- `z_n, ^S = R_ (s_n^0, p_n, ^S), _n, ^S = W_C, ^S + b_C, ^S, =2,,L`
- `_n, = ((W_Q^ M s_n^ -1)^ (W_K, ^ M u_n,) + b_ ^ M), s_n^ = s_n^ -1 + _n, W_V, ^ M u_n,`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- The overall framework of the proposed. The code layer $L$ is set to 3 for illustration.
- Overview of GR pipeline. Quantization transforms items from features to SID. Representation organizes SID as input to the LLM. Generation uses the LLM to generate the next item's SID.
- Overall performance of under different settings. bold values are the best, and "*" marks significant gains (one-side t-test with p<0.05) over the matched architecture.
- Performance with Qwen3-0.6B as the LLM backbone.
- Performance with LLaMA3-1B as the LLM backbone.
- Ablation results on Yelp dataset. w/o MM-Scoring replaces original module with mean pooling, and w/o Mem. Merge replaces the original module with a linear layer. Other variants remove intra-item or inter-item memory from the encoding ( E) or decoding ( D).
- font=small,labelfont=bf,justification=centering
- Scaling analysis of dual-level Engram memory on the Yelp dataset.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Amazon, Yelp</td></tr>
<tr><th>Metrics</th><td>Hit, latency, MAP, Success</td></tr>
<tr><th>Baselines</th><td>RQ-VAE</td></tr>
</tbody></table>
</div>

- Evidence: Эксперименты показывают, что улучшение representation construction может давать gains без полной замены tokenizer.
- Experimental Settings: Dataset. There are three public datasets applied for evaluation, Yelp, Amazon Industrial, and Amazon Instrument. More details can be found in the Appendix.
- Experimental Settings: Baselines & Backbones. We construct our experiments leveraging different quantization mechanisms, RQ-VAE lee2022autoregressive and RQ-Kmeans jegou2010product, different LLM backbones, Qwen3-0.6B and LLaMA3-1B, and different architectures, normal GR and NEZHA wang2026nezha. The '+ TM' denotes token merging, following the settings of previous work...
- Extensive experiments demonstrate the effectiveness and flexibility of ComeIR, and further reveal scalable gains from enlarging conditional memory.

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: сдвигает фокус с самого tokenizer на item-level input representation: простое склеивание SID-token embeddings теряет структуру и усиливает коллизии.
- **Algorithmic/system:** Алгоритм: ComeIR использует MM-guided token scoring, dual-level Engram memory и token-granular decoding, чтобы сохранить intra-item composition и inter-item transitions.
- **Empirical:** Evidence: Эксперименты показывают, что улучшение representation construction может давать gains без полной замены tokenizer.
- **Practical:** Ограничение: Memory-компоненты увеличивают состояние модели и требуют аккуратной оценки serving/memory overhead.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Amazon, Yelp.
- Метрики: Hit, latency, MAP, Success.
- Baselines и parity settings: RQ-VAE.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Полезна как напоминание: после построения SID еще остается отдельная задача, как превратить SID tokens в хороший item representation для encoder.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: сдвигает фокус с самого tokenizer на item-level input representation: простое склеивание SID-token embeddings теряет структуру и усиливает коллизии.
- Алгоритм: ComeIR использует MM-guided token scoring, dual-level Engram memory и token-granular decoding, чтобы сохранить intra-item composition и inter-item transitions.
- Evidence: Эксперименты показывают, что улучшение representation construction может давать gains без полной замены tokenizer.
- Ограничение: Memory-компоненты увеличивают состояние модели и требуют аккуратной оценки serving/memory overhead.
- Итог: Полезна как напоминание: после построения SID еще остается отдельная задача, как превратить SID tokens в хороший item representation для encoder.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td> With recent advances in natural language processing, large language models (LLMs) have demonstrated powerful capabilities in both semantic understanding and sequence modeling zhao2023survey,chang2024survey, which has motivated generative...</td></tr>
<tr><td>Problem Definition</td><td> The goal of GR is to directly generate the next item that a user is likely to interact with based on historical interactions. Let U and I denote the user set and item set, respectively. For each user u U, the historical interactions are...</td></tr>
<tr><td>Method</td><td> figure* [!t] FigureMaking/ComeIR.pdf The overall framework of the proposed. The code layer L is set to 3 for illustration. figure*</td></tr>
<tr><td>Framework Overview</td><td>The overview of our proposed framework is illustrated in Given the SID-level representations R n S defined in Section, our target is to model a fine-grained f( ) that maps each SID into an item-level representation while...</td></tr>
<tr><td>MM-guided Token Scoring</td><td>As discussed in the Identity-Structure Preservation Conflict, item-level compression should retain item-specific identity without discarding SID-structured evidence. We first address the identity side through MM-guided Token Scoring. The guidance comes from...</td></tr>
<tr><td>Dual-level Engram Memory</td><td>To preserve the SID-structure side, we model two structures carried by SIDs. Within one item, an SID is an ordered code sequence (c n 1,,c n L), where later codes refine the prefix formed by previous codes. Across a user history, the level- prefixes form a...</td></tr>
<tr><td>Memory-conditioned Token Merge</td><td>For a specific item v n, after obtaining the MM-guided item context s n 0 and dual-level memories, n, S and n, T at each code level, we need to transform multiple signals into one item-level input. However, a naive merge collapses all code embeddings at...</td></tr>
<tr><td>Memory-restoring Prediction Head</td><td> head While item-level input improves efficiency, the target item still needs to be generated as a token-level SID during decoding for fine-grained item retrieval zou2026genrec. Consequently, we design the Memory-restoring Prediction Head,...</td></tr>
<tr><td>Training and Inference</td><td> infer Training. Given the ground-truth next-item SID c N+1, we train with teacher forcing and token-level cross-entropy over SID layers. At layer, the ground-truth prefix c N+1 &lt; is provided to the prediction head, so the model learns to score the...</td></tr>
<tr><td>Experiment</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Experimental Settings</td><td> Dataset. There are three public datasets applied for evaluation, Yelp, Amazon Industrial, and Amazon Instrument. More details can be found in the Appendix.</td></tr>
<tr><td>Overall Performance</td><td>To validate the effectiveness and flexibility of the proposed, we compare its performance across various quantization mechanisms, LLM backbones, and architectures, both with the general setting (flattening the sequence) and with token merging. As shown in...</td></tr>
</tbody></table>
</div>
