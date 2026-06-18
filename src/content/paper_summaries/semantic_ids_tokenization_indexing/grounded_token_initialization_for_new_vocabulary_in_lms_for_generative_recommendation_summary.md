---
title: "Grounded Token Initialization for New Vocabulary in LMs for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "grounded_token_initialization_for_new_vocabulary_in_lms_for_generative_recommendation_summary"
catalogId: "paper-grounded_token_initialization_for_new_vocabulary_in_lms_for_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/grounded_token_initialization_for_new_vocabulary_in_lms_for_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.02324"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Daiwei Chen, Zhoutong Fu, Chengming Jiang, Haichao Zhang, Ran Zhou, Tan Wang, Chunnan Yao, Guoyao Li, Rui Cai, Yihan Cao, Ruijie Jiang, Fedor Borisyuk, Jianqiang Shen, Jingwei Wu, Ramya Korlakai Vinayak.
>
> **Аффилиации:** University of Wisconsin-Madison; Northeastern University; University of California, Davis.
>
> **Источник:** [arXiv 2604.02324](https://arxiv.org/abs/2604.02324) · дата metadata: 2026-04-02.
>
> **Категория/теги:** semantic IDs, generative recommendation, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** [https://business.yelp.com/data/resources/open-dataset/](https://business.yelp.com/data/resources/open-dataset/) [https://huggingface.co/docs/trl/en/quickstart](https://huggingface.co/docs/trl/en/quickstart) [https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental-dataset](https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental-dataset).

## 1. Коротко

- Главная идея: GTI показывает, что mean initialization новых vocabulary tokens в LM-based GR создает degenerate subspace и мешает SID tokens различаться.
- Алгоритм: Grounded Token Initialization перед fine-tuning мапит новые tokens в разные meaningful позиции pretrained embedding space через paired linguistic supervision.
- Evidence: Метод выигрывает у mean initialization и auxiliary adaptation в большинстве benchmark settings, включая industrial-scale cases.
- Ограничение: Нужно контролировать качество linguistic grounding и отсутствие leakage из metadata в evaluation.
- Итог: Практически важно для всех LLM-GR систем с новыми SID/item tokens: initialization является отдельным bottleneck.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>These findings suggest that token initialization is a key bottleneck when extending LMs with new vocabularies.</td></tr>
<tr><th>Предложение авторов</th><td>We present a systematic analysis of this strategy: through spectral and geometric diagnostics, we show that mean initialization collapses all new tokens into a degenerate subspace, erasing inter-token distinctions that subsequent fine-tuning struggles to fully recover.</td></tr>
<tr><th>Главный evidence claim</th><td>Despite its simplicity, GTI outperforms both mean initialization and existing auxiliary-task adaptation methods in the majority of evaluation settings across multiple generative recommendation benchmarks, including industry-scale and public datasets.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>SFT, KL</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG</td></tr>
<tr><th>Baselines</th><td>TIGER, RQ-VAE, VQ-VAE, LC-Rec</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Token-Embedding Misalignment: We formalize the vocabulary extension problem in the context of generative retrieval, our primary application domain, and then use spectral and geometric diagnostics to characterize a systematic token-embedding misalignment that arises from standard initialization practices when new tokens are added to a pretrained language model.
1. **Ключевой модуль.** Token-Embedding Misalignment: Generative Retrieval. We adopt the framework of TIGER. Each item I i I has content features (title, description, etc.) that a pretrained text encoder maps to a semantic embedding z i R d. An RQ-VAE RQ-VAE with L codebook levels of K entries each discretizes z i into a Semantic ID (c 1,,c L), c l 1,,K, via recursive residual quantization: [ r 1:= z i;...
1. **Learning signal.** Token-Embedding Misalignment: Mean-of-Vocabulary Initialization. Standard practice initializes all novel token embeddings to the mean of the existing vocabulary embeddings hewitt2021initializing: equation e c:= 1 | V text | v V text e v,; c V SID, equation where e v denotes the input embedding of token v.
1. **Inference / deployment path.** : Grounded Token Initialization Stage: The diagnosis in Section subsect:misalign motivates a straightforward modification to the standard training pipeline: before downstream fine-tuning, insert a grounding stage that freezes the LM backbone and only learns new token embeddings via paired linguistic supervision. This design builds on the established principle of training new token...
1. **Проверяемая деталь.** : Grounded Token Initialization Stage: Algorithm. Let V = V text V new denote the extended vocabulary, where V new are the newly added domain tokens. Given a pretrained autoregressive LM with input-embedding matrix E R | V | d, we partition E into the pretrained rows E text and the new rows E new R | V new | d. Each domain entity is associated with a natural-language description x i (e.g.,...
1. **Проверяемая деталь.** : Grounded Token Initialization Stage: We construct a grounding corpus D ground = (x i,y i) i=1 n pairing each description with its new token sequence, along with reversed pairs (y i,x i) that require the model to generate descriptions from new tokens Bidirectional training encourages new token embeddings to encode semantics in both the input and output directions; see ablation in Section...

## 5. Objectives, formulas и training details

**Detected objective keywords:** SFT, KL.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `_E_ new _(x,y) D_ ground _t=1^|y| - P_ (y_t | y_<t, prompt(x))`
- `e_c:= 1| V_ text| _v V_ text e_v, \; c V_ SID,`
- `r_1:= z_i; c_l = _k \| r_l - q_k^(l) \|_2, r_l+1:= r_l - q_c_l^(l), l = 1,,L,`
- `P_ (c_1,,c_L x) = _t=1^L P_ (c_t c_<t, x).`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Token-embedding collapse under mean initialization and the effect of grounding. (a) Left: Mean initialization maps all SID tokens (white triangles) to a single point, collapsing inter-token distinctions. Top-right: \ grounds SID tokens (colored triangles) into distinct regions by training only the $| V_ SID|\! \!d$ embedding parameters while freezing the...
- Relative Precision@K gain (\
- Relative gain versus candidate pool size. Left/Middle: Relative Precision@K gain under Good Match and Good \& Maybe Match; Right: Relative NDCG@K gain (Composite). \ consistently outperforms both baselines across all pool sizes, with the largest gains at small $K$. Shaded areas denote variability across runs.
- Relative NDCG@K (Composite) gain (\
- Relative Recall@K and NDCG@K (\
- Relative gain versus candidate pool size. Left: Relative Recall@K gain; Right: Relative NDCG@K gain. Shaded areas denote variability across runs.
- Pairwise cosine-similarity matrices under three initialization strategies. Each matrix shows similarities between 50 pretrained tokens (upper-left block) and 50 SID tokens (bottom-right block). Random initialization (left) yields noninformative SID embeddings. Mean initialization (middle) collapses SID tokens into a near-uniform block. GTI (right) produces...
- Pairwise SID similarity after fine-tuning (public dataset). We visualize the pairwise cosine similarity matrix of SID embeddings at the fine-tuned checkpoint. \ is the only initialization strategy that preserves a clear blockwise hierarchical semantics among SID tokens, suggesting improved preservation of semantic geometry. By contrast, mean and random...

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source</td></tr>
<tr><th>Metrics</th><td>Recall, NDCG</td></tr>
<tr><th>Baselines</th><td>TIGER, RQ-VAE, VQ-VAE, LC-Rec</td></tr>
</tbody></table>
</div>

- Evidence: Метод выигрывает у mean initialization и auxiliary adaptation в большинстве benchmark settings, включая industrial-scale cases.
- Experiments: We evaluate within the highly demanding domain of generative recommendation. This domain serves as an ideal testbed for the initialization bottleneck, as it requires incorporating thousands of new Semantic-ID (SID) tokens into a pretrained language model. To empirically validate whether aligning these tokens with the model’s pre-existing...
- Overall Performance Analysis: The effectiveness of Initialization. Across all cutoffs, evaluation metrics, and relevance thresholds (Good Match and Good & Maybe Match), outperforms both baselines. Under the strict Good Match criterion, achieves +21.63
- Overall Performance Analysis: Evidence for the hypothesis. The comparison between LC-Rec and provides a controlled test of our hypothesis, as both methods introduce linguistic supervision for new tokens but differ in when it is applied: LC-Rec incorporates auxiliary language modeling objectives during fine-tuning while retaining mean initialization, whereas addresses the initialization...
- Further Analysis: The preceding results establish that grounded initialization improves downstream performance; we now investigate why. We use spectral and geometric diagnostics on the SID embedding subspace, both at initialization and after fine-tuning. These analyses provide direct evidence to the Grounded Token Initialization Hypothesis (Section subsect:misalign ).
- Further Analysis: Grounded initialization produces differentiated embedding geometry. Figure cossim visualizes pairwise cosine similarities among pretrained vocabulary tokens and SID tokens under three initialization strategies. Random initialization avoids uniformity but yields unstructured noise with no coherent affinity to the pretrained manifold. Mean...

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: GTI показывает, что mean initialization новых vocabulary tokens в LM-based GR создает degenerate subspace и мешает SID tokens различаться.
- **Algorithmic/system:** Алгоритм: Grounded Token Initialization перед fine-tuning мапит новые tokens в разные meaningful позиции pretrained embedding space через paired linguistic supervision.
- **Empirical:** Evidence: Метод выигрывает у mean initialization и auxiliary adaptation в большинстве benchmark settings, включая industrial-scale cases.
- **Practical:** Ограничение: Нужно контролировать качество linguistic grounding и отсутствие leakage из metadata в evaluation.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Gain может идти от capacity, metadata/features, negative sampling или candidate-space differences, а не от заявленного компонента.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: явные датасеты не извлечены автоматически; смотреть Experimental setup в PDF/source.
- Метрики: Recall, NDCG.
- Baselines и parity settings: TIGER, RQ-VAE, VQ-VAE, LC-Rec.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Практически важно для всех LLM-GR систем с новыми SID/item tokens: initialization является отдельным bottleneck.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: GTI показывает, что mean initialization новых vocabulary tokens в LM-based GR создает degenerate subspace и мешает SID tokens различаться.
- Алгоритм: Grounded Token Initialization перед fine-tuning мапит новые tokens в разные meaningful позиции pretrained embedding space через paired linguistic supervision.
- Evidence: Метод выигрывает у mean initialization и auxiliary adaptation в большинстве benchmark settings, включая industrial-scale cases.
- Ограничение: Нужно контролировать качество linguistic grounding и отсутствие leakage из metadata в evaluation.
- Итог: Практически важно для всех LLM-GR систем с новыми SID/item tokens: initialization является отдельным bottleneck.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Pretrained language models (LMs) are increasingly adapted to specialized domains by extending their vocabulary with new learnable tokens. A prominent example is generative retrieval, where items TIGER, GRreview or documents tay2022transformer are assigned...</td></tr>
<tr><td>Token-Embedding Misalignment</td><td>We formalize the vocabulary extension problem in the context of generative retrieval, our primary application domain, and then use spectral and geometric diagnostics to characterize a systematic token-embedding misalignment that arises from standard...</td></tr>
<tr><td>: Grounded Token Initialization Stage</td><td> The diagnosis in Section subsect:misalign motivates a straightforward modification to the standard training pipeline: before downstream fine-tuning, insert a grounding stage that freezes the LM backbone and only learns new token embeddings via...</td></tr>
<tr><td>Experiments</td><td> We evaluate within the highly demanding domain of generative recommendation. This domain serves as an ideal testbed for the initialization bottleneck, as it requires incorporating thousands of new Semantic-ID (SID) tokens into a pretrained...</td></tr>
<tr><td>Setup</td><td>(1) Industrial candidate retrieval. Data access and use complied with internal privacy and security frameworks. Member profile attributes were processed in accordance with applicable member controls and visibility settings, and analyses were conducted on...</td></tr>
<tr><td>Overall Performance Analysis</td><td>The effectiveness of Initialization. Across all cutoffs, evaluation metrics, and relevance thresholds (Good Match and Good &amp; Maybe Match), outperforms both baselines. Under the strict Good Match criterion, achieves +21.63</td></tr>
<tr><td>Further Analysis</td><td>The preceding results establish that grounded initialization improves downstream performance; we now investigate why. We use spectral and geometric diagnostics on the SID embedding subspace, both at initialization and after fine-tuning. These analyses provide...</td></tr>
<tr><td>Related Work</td><td>Vocabulary Extension in Language Models. Extending a pretrained LM's vocabulary with new tokens is a recurring challenge. Standard approaches initialize new embeddings at the vocabulary mean hewitt2021initializing or randomly, then rely on fine-tuning....</td></tr>
<tr><td>Conclusion</td><td>Through spectral and geometric diagnostics, we show that mean-of-vocabulary initialization collapses new tokens into a degenerate subspace that fine-tuning does not fully recover. Motivated by this diagnosis, we propose, a lightweight grounding stage that...</td></tr>
<tr><td>Appendix</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Datasets</td><td>Industrial Candidate Retrieval Dataset. The industrial-scale candidate retrieval dataset Data access and use complied with internal privacy and security frameworks. Member profile attributes were processed in accordance with applicable member controls and...</td></tr>
<tr><td>Prompt Templates</td><td>promptboxgray [label= lst:semid-prompt ] Item Title/Description Semantic IDs (Title New Vocabulary Tokens) &lt;system&gt; You are a helpful assistant. &lt;user&gt; Which item has the title: title ? &lt;assistant&gt; ITEM SEMANTIC ID promptboxgray</td></tr>
</tbody></table>
</div>
