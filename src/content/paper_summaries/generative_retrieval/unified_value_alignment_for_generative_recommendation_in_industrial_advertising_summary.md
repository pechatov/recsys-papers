---
title: "Unified Value Alignment for Generative Recommendation in Industrial Advertising"
category: "generative_retrieval"
slug: "unified_value_alignment_for_generative_recommendation_in_industrial_advertising_summary"
catalogId: "paper-unified_value_alignment_for_generative_recommendation_in_industrial_advertising_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/unified_value_alignment_for_generative_recommendation_in_industrial_advertising_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.05803"
---
Расширенное саммари по обновленному `cs-paper-reading`: metadata, technical spine, method walkthrough, experiments, limitations и practical readout.

> **Авторы:** Xinxun Zhang, Yuling Xiong, Jiale Zhou, Zhengkai Guo, Zhennan Pang, Junbang Huo, Jingwen Wang, Xuyang Sun, Enming Zhang, Jiaguang Jin, Changping Wang, Yi Li, Jun Zhang, Xiao Yan, Jiawei Jiang, Jie Jiang.
>
> **Аффилиации:** Wuhan University; Tencent Inc.; Peking University.
>
> **Источник:** [arXiv 2605.05803](https://arxiv.org/abs/2605.05803) · дата metadata: 2026-05-07.
>
> **Категория/теги:** generative recommendation, industrial, alignment, новое за 2 месяца.
>
> **Ссылки из source (код, данные, baseline или reference):** не найдено явных repository/dataset links в arXiv source.

## 1. Коротко

- Главная идея: UniVA выравнивает GR для industrial advertising с commercial value, а не только semantic relevance/user interest.
- Алгоритм: Commercial SID tokenizer кодирует value attributes; Generation-as-Ranking decoder соединяет supervised learning и eCPM-aware RL; personalized beam search использует value-guided logits и request-valid trie.
- Evidence: На Tencent WeChat Channels: offline HR@100 +37.04%, ValueHR/wNDCG gains, online GMV +1.5%.
- Ограничение: Value alignment может конфликтовать с user welfare; rewards основаны на production pCTR/pCVR и требуют governance.
- Итог: Важна для ads: SID и decoding должны быть value-aware на всех стадиях, иначе GR оптимизирует не тот objective.

**Как читать статью:** это прежде всего работа типа *semantic-ID/tokenizer*; поэтому основной audit должен смотреть на collision rate, codebook utilization, item-level Recall/NDCG, tail/cold-start slices и identifier churn.

## 2. Авторская постановка и claim

<div class="table-scroll">
<table><tbody>
<tr><th>Проблема</th><td>Generative Recommendation (GR) reformulates recommendation as a next-token generation problem and has shown promise in industrial applications.</td></tr>
<tr><th>Предложение авторов</th><td>To address this issue, we propose UniVA, a Unified Value Alignment framework for advertising recommendation.</td></tr>
<tr><th>Главный evidence claim</th><td>Existing GR pipelines remain largely semantics-centric, making it difficult to align value signals across tokenization, decoding, and online serving.</td></tr>
<tr><th>Моя проверочная рамка</th><td>Отделять авторский claim от того, что реально доказано experiments: для этой статьи ключевой риск - Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.</td></tr>
</tbody></table>
</div>

## 3. Technical Spine

<div class="table-scroll">
<table><tbody>
<tr><th>Тип вклада</th><td>semantic-ID/tokenizer</td></tr>
<tr><th>Input signal</th><td>user history / item metadata / collaborative signals / prompt или production logs; точный набор нужно сверять в setup</td></tr>
<tr><th>Representation</th><td>semantic IDs / discrete tokens / generated IDs / cache state / value-aware target в зависимости от задачи; см. method walkthrough ниже</td></tr>
<tr><th>Learning signal</th><td>PPO, reinforcement, causal</td></tr>
<tr><th>Inference path</th><td>constrained decoding, beam/trie/hash verification, diffusion/parallel decoding, cache reuse или business-rule routing - если применимо</td></tr>
<tr><th>Datasets/domains</th><td>Tencent</td></tr>
<tr><th>Metrics</th><td>Hit, HR, GMV, revenue, Success, accuracy</td></tr>
<tr><th>Baselines</th><td>CoST, HSTU</td></tr>
<tr><th>Ключевое предположение</th><td>Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.</td></tr>
</tbody></table>
</div>

## 4. Метод: walkthrough по source sections

Ниже не пересказ названий секций, а рабочая карта того, где в методе находится основной механизм. Короткие английские anchors оставлены как привязка к arXiv source/PDF.

1. **Постановка представления.** Methodology: The overall framework of UniVA is shown in Figure fig model. First, a commercial SID tokenizer injects commercial attributes and bid information into the final SID layer, making the token space value-discriminative. Second, the Generation-as-Ranking SID Decoder combines generation scores and token-level value scores within the same decoding process....
1. **Ключевой модуль.** Commercial SID Tokenization: Existing SID construction methods primarily encode semantic characteristics, which leaves the token space insufficiently discriminative for commercial value. To address this limitation, UniVA adopts a semantic-commercial hybrid SID structure with an explicit Commercial SID:
1. **Learning signal.** Commercial SID Tokenization: where sem reuses the RQ-Kmeans+ semantic tokenizer zhang2025gpr to preserve the semantic organization of the upper SID levels, and com maps structured commercial attributes into a discrete value-aware token at the final level. In this way, UniVA preserves the coarse-to-fine semantic hierarchy of GR while explicitly injecting commercial discriminability into...
1. **Inference / deployment path.** Commercial SID Tokenization: Attribute Space Compression. The raw commercial attribute space is too sparse to support reliable token construction directly. In particular, optimization goals and industry categories exhibit substantial long-tail behavior, and directly taking their Cartesian product would lead to severe vocabulary explosion and unstable fine-grained statistics. UniVA...

## 5. Objectives, formulas и training details

**Detected objective keywords:** PPO, reinforcement, causal.

Формульные anchors из TeX, которые стоит открыть рядом с method section:

- `(s_i^1,, s_i^L-1) = _ sem(x_i^s), s_i^L = _ com(x_i^c),`
- `x_i^o' = _ O(x_i^o), x_i^r' = _ R(x_i^r), x_i^ ind' = _ I(x_i^ ind),`
- `k_i = (x_i^o', x_i^r', x_i^ ind') K O R I.`
- `B_k = \ x_i^b k_i = k \.`

Практически важный вопрос: совпадает ли training objective с тем, что потом считается в item-level или business-level evaluation. Для SID/GenIR papers особенно опасен разрыв между token likelihood, SID-prefix match и реальным попаданием конкретного item/document.

## 6. Figures / Tables для ручной сверки

- Existing GR methods introduce commercial value only in isolated stages, whereas UniVA consistently aligns value signals throughout SID tokenization, autoregressive decoding, and online serving.
- Offline next interacted item prediction performance comparison of UniVA and its SID-level variants on the industrial advertising benchmark. Parameters and FLOPs report only the SID decoder cost, excluding the encoder. $ $HR@100 denotes the relative improvement over the base GPR + SID Decoder.
- Offline value analysis under four SID designs on the GMV-weighted next-conversion set.
- Path-level bid-dispersion statistics for 3-level SID and 2-level SID + Commercial SID. Each subfigure reports Mean, P75, and P99 over complete SID paths, and the y-axis is shown in log scale.
- HR@K comparison under different SID codebook sizes. All values are reported in percentage points (\
- Online A/B test results on Tencent WeChat Channels advertising traffic from March 7 to March 11, 2026, over 5\
- Commercial SID strategy comparison across three overall strategies and three in-bin strategies. Each cell corresponds to one strategy combination and reports weighted entropy $H$ and vocabulary size $V$.

## 7. Эксперименты и evidence

<div class="table-scroll">
<table><tbody>
<tr><th>Datasets/domains</th><td>Tencent</td></tr>
<tr><th>Metrics</th><td>Hit, HR, GMV, revenue, Success, accuracy</td></tr>
<tr><th>Baselines</th><td>CoST, HSTU</td></tr>
</tbody></table>
</div>

- Evidence: На Tencent WeChat Channels: offline HR@100 +37.04%, ValueHR/wNDCG gains, online GMV +1.5%.
- Experimental Setup: Datasets and Baselines. Following GPR zhang2025gpr, we build the offline dataset from a large-scale Tencent advertising corpus that mixes ads with organic media such as short videos, social feeds, and news, so that evaluation reflects realistic mixed-traffic contexts. Each sample contains session-level behaviors together with item-level multimodal...
- Experimental Setup: Implementation Details. UniVA uses a three-level SID structure with a codebook size of 2048. The SID decoder contains 4 layers with embedding dimension 256. For Commercial SID construction, the adaptive bid-binning hyperparameters are set to n =25 and n =3, and the final binning configuration is selected by grid search to maximize the weighted entropy H...
- Existing GR pipelines remain largely semantics-centric, making it difficult to align value signals across tokenization, decoding, and online serving.
- Experiments on the Tencent WeChat Channels advertising platform show that UniVA achieves a 37.04\

**Что проверять перед тем, как верить числам:** candidate universe, одинаковый доступ к content/collaborative features, negative sampling, beam size/decoding constraints, item-level vs SID-level metric, variance/significance и наличие ablations по главному компоненту.

## 8. Contributions и novelty

- **Conceptual:** Главная идея: UniVA выравнивает GR для industrial advertising с commercial value, а не только semantic relevance/user interest.
- **Algorithmic/system:** Алгоритм: Commercial SID tokenizer кодирует value attributes; Generation-as-Ranking decoder соединяет supervised learning и eCPM-aware RL; personalized beam search использует value-guided logits и request-valid trie.
- **Empirical:** Evidence: На Tencent WeChat Channels: offline HR@100 +37.04%, ValueHR/wNDCG gains, online GMV +1.5%.
- **Practical:** Ограничение: Value alignment может конфликтовать с user welfare; rewards основаны на production pCTR/pCVR и требуют governance.
- **Новизна, которую стоит отделять от инженерного контекста:** reusable idea находится в связке objective + representation + inference protocol; одно только использование LLM/RQ-VAE/SID/GRPO не делает contribution новым.

## 9. Слабые места и открытые вопросы

- Часть evidence приходит из закрытого production setup: практический сигнал сильный, но воспроизводимость и переносимость ограничены.
- Дискретный identifier должен сохранять полезную item semantics и не создавать неконтролируемые collisions/churn.
- Нужно проверить, не совпадает ли improvement с большим capacity, richer metadata, более легким candidate space или неравным decoding budget.
- Для production/industrial работ отдельно нужны latency, refresh cost, rollback path, monitoring of drift/collisions и per-slice metrics для tail/cold-start groups.

## 10. Reproduction Checklist

- Данные и split: Tencent.
- Метрики: Hit, HR, GMV, revenue, Success, accuracy.
- Baselines и parity settings: CoST, HSTU.
- Artifacts: tokenizer/codebook assignment, item-to-SID map, collision statistics, decoding constraints, train/valid/test split, negative sampling, reward/value construction или cache policy.
- Serving checks: latency, memory, batchability, update/churn cost, invalid generation rate и fallback behavior.

## 11. Практические последствия

Итог: Важна для ads: SID и decoding должны быть value-aware на всех стадиях, иначе GR оптимизирует не тот objective.

Для локального проекта я бы превращал статью в минимальный ablation: заменить только заявленный компонент, замерить item-level Recall/NDCG, collision/invalid rate, tail/cold-start slices и latency. Без такой изоляции легко перепутать эффект tokenizer, backbone, features, decoding constraints и production reward.

## 12. Выжимка для каталога

- Главная идея: UniVA выравнивает GR для industrial advertising с commercial value, а не только semantic relevance/user interest.
- Алгоритм: Commercial SID tokenizer кодирует value attributes; Generation-as-Ranking decoder соединяет supervised learning и eCPM-aware RL; personalized beam search использует value-guided logits и request-valid trie.
- Evidence: На Tencent WeChat Channels: offline HR@100 +37.04%, ValueHR/wNDCG gains, online GMV +1.5%.
- Ограничение: Value alignment может конфликтовать с user welfare; rewards основаны на production pCTR/pCVR и требуют governance.
- Итог: Важна для ads: SID и decoding должны быть value-aware на всех стадиях, иначе GR оптимизирует не тот objective.

## 13. Карта структуры статьи

<div class="table-scroll">
<table><thead><tr><th>Секция</th><th>Что искать</th></tr></thead><tbody>
<tr><td>Introduction</td><td>Driven by the generative modeling capability of large language models (LLMs) achiam2023gpt,touvron2023llama,zhou2024large, recommendation systems are shifting from multi-stage deep-learning pipelines to end-to-end generative architectures...</td></tr>
<tr><td>Preliminaries</td><td>Semantic ID. GR formulates recommendation as a sequence generation problem. Given a user u, context c, and historical item sequence i 1:T = (i 1, i 2,, i T), the model directly predicts the next target instead of ranking over a candidate set. Each item is...</td></tr>
<tr><td>Methodology</td><td>The overall framework of UniVA is shown in Figure fig model. First, a commercial SID tokenizer injects commercial attributes and bid information into the final SID layer, making the token space value-discriminative. Second, the Generation-as-Ranking SID...</td></tr>
<tr><td>Commercial SID Tokenization</td><td>Existing SID construction methods primarily encode semantic characteristics, which leaves the token space insufficiently discriminative for commercial value. To address this limitation, UniVA adopts a semantic-commercial hybrid SID structure with an explicit...</td></tr>
<tr><td>Generation-as-Ranking SID Decoder</td><td>Commercial SID makes the token space value-discriminative, while UniVA further injects commercial value into SID decoding through a Generation-as-Ranking SID Decoder. Following GPR, UniVA adopts the same unified input schema and HSTU encoder backbone...</td></tr>
<tr><td>eCPM-aware Reinforcement Learning</td><td>Supervised learning stabilizes SID generation but does not directly optimize commercial return. UniVA therefore introduces an eCPM-aware reinforcement learning stage. Concretely, the generation head learned in the SL stage is directly reused as the RL policy...</td></tr>
<tr><td>Joint Optimization</td><td>Supervised learning is used to establish stable SID generation, while reinforcement learning introduces commercial value supervision through simulator-generated reranking rewards. UniVA combines the two stages through collaborative iterative training so that...</td></tr>
<tr><td>Value-Guided Personalized Beam Search</td><td>UniVA keeps online serving consistent with the generation-as-ranking design under the unified value-alignment objective. The central principle of this stage is to let commercial value directly participate in beam expansion, so that online SID trajectory...</td></tr>
<tr><td>Experiments</td><td>текст не извлечен; смотреть PDF/source</td></tr>
<tr><td>Experimental Setup</td><td>Datasets and Baselines. Following GPR zhang2025gpr, we build the offline dataset from a large-scale Tencent advertising corpus that mixes ads with organic media such as short videos, social feeds, and news, so that evaluation reflects realistic mixed-traffic...</td></tr>
<tr><td>Overall Performance</td><td>As summarized in Table, UniVA consistently improves next interacted item prediction performance performance over the base GPR + SID Decoder, and the reported parameters and FLOPs include only the SID decoder. Commercial SID improves HR@100 by 5.78</td></tr>
<tr><td>Value Alignment Performance</td><td>To verify whether UniVA can capture commercial value, we conduct a value analysis experiment under different SID designs on the GMV-weighted next-conversion set. Figure value presents the value-oriented evaluation under different SID designs on...</td></tr>
</tbody></table>
</div>
