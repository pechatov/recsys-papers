---
title: "TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems"
category: "semantic_ids_tokenization_indexing"
slug: "tokenminds_pretrained_user_tokens_and_embeddings_for_user_understanding_in_large_recommender_systems_summary"
catalogId: "paper-tokenminds_pretrained_user_tokens_and_embeddings_for_user_understanding_in_large_recommender_systems_summary"
paperUrl: "https://arxiv.org/abs/2606.25147"
---
> **Авторы:** Qingyun Liu, Bo Yan, Yang Liu, Yuji Roh, Ekansh Sharma, Likang Yin, Emma Olowo, Min-hsuan Tsai, Yuxuan Li, Diego Uribe, Saksham Aggarwal, Siqi Wu, Yuan Hao, Vikas Kedigehalli, Lukasz Heldt, Lichan Hong, Li Wei, Xinyang Yi.
>
> **Аффилиации:** Google DeepMind; YouTube.
>
> **Источник:** arXiv:2606.25147, submitted 2026-06-23. Код и открытый production-датасет не опубликованы.

## 1. Коротко

TokenMinds расширяет PLUM с item retrieval на user modeling. Если PLUM учит LLM генерировать SID target items, то TokenMinds учит encoder-decoder model выдавать две user representations: dense user embedding из encoder и discrete SID-based user tokens из decoder.

Главная production-мотивация: dense user embeddings совместимы с существующими rankers, но сжимают весь спектр интересов пользователя в фиксированный vector. Textual user profiles от LLM более интерпретируемы, но плохо grounded в item attributes и плохо подходят для non-text downstream systems. TokenMinds занимает промежуточную позицию: user tokens - это coarse SID prefixes будущих интересов, grounded в том же item SID vocabulary, который уже используется PLUM.

Система запущена на YouTube full traffic через asynchronous serving: тяжелая генерация user representations происходит background refresh'ем и кладется в key-value cache; online rankers читают cached features. В live A/B tokens дают additive value, embeddings дают совместимость, а `Embed+Token` дает лучший результат на SFV: +0.11% Engaged Users и +0.62% Satisfied Engagement.

<figure class="paper-figure">
  <img src="../../assets/tokenminds/framework.png" alt="TokenMinds dual-output framework for user tokens and embeddings">
  <figcaption>Рисунок 1. TokenMinds обрабатывает watches, search queries и engagement features, затем одновременно выдает dense user embedding и discrete SID-based user tokens.</figcaption>
</figure>

## 2. Чем TokenMinds отличается от PLUM

PLUM делает candidate generation: по контексту пользователя генерирует SID следующего video и мапит SID в video candidate. TokenMinds делает reusable user representation: модель генерирует набор future-interest SID prefixes и dense embedding, а downstream rankers используют их как features.

Это смещает objective. Не нужно точно восстановить конкретный immediate next item. Нужно получить представление интересов пользователя, которое помогает разным production ranking systems. Поэтому TokenMinds использует look-ahead window, multiple future targets и coarse SID prefix truncation. Это делает user tokens менее похожими на "ID следующего видео" и больше похожими на compact intent vocabulary.

## 3. Архитектура и обучение

### 3.1. Encoder-decoder вместо decoder-only

TokenMinds использует Gemini V1.5-based architecture: 370M-parameter MoE encoder и 370M-parameter dense decoder. Оба initialized from CPT checkpoints, following PLUM. Encoder видит полную user history и дает dense embedding через pooling. Decoder cross-attend'ит encoder outputs и autoregressively генерирует SID tokens для нескольких future watches.

<figure class="paper-figure">
  <img src="../../assets/tokenminds/enc_dec.png" alt="TokenMinds encoder-decoder training with SID targets">
  <figcaption>Рисунок 2. На training decoder генерирует SID prefixes нескольких future watches, а encoder через cross-attention получает supervision для dense user representation.</figcaption>
</figure>

Авторы выбирают encoder-decoder по двум причинам:

- encoder лучше подходит для full-history contextualization и dense embedding extraction;
- encoder/decoder можно decouple'ить в serving: тяжелый history encoder refresh'ить реже, decoder использовать для scenario-specific tokens.

### 3.2. Inputs and targets

Training setup:

- most recent 1,200 watches;
- `S = 10` recent search queries interleaved chronologically;
- maximum input length 1,024 tokens;
- each watch: condition token, prefix-`L = 4` SID tokens, one soft token for non-SID features;
- full video SID has `L_full = 8`, but model uses prefix `L < L_full`;
- target selection samples up to `N = 15` watches from a 24-hour look-ahead window.

Loss считается только по prefix-`L` SID tokens:

$$
\mathcal{L} = - \sum_{i=1}^{N} r(W_i) \sum_{j=1}^{L}
\log P(SID_{i,j} | W_1, \dots, W_t, W_{<i}, SID_{i,<j}).
$$

Как и в PLUM, reward используется через sampling proportional to reward, а не через тяжелое per-example weighting.

### 3.3. Cross-scenario modeling

TokenMinds объединяет LFV и SFV. Мотивация: почти половина пользователей потребляет оба формата, а SID vocabulary overlap для первых двух prefixes около 40%. Вместо двух отдельных models вводятся condition tokens `<LFV>`/`<SFV>` и multi-context decoding.

<figure class="paper-figure">
  <img src="../../assets/tokenminds/unified_decoding.png" alt="TokenMinds multi-context decoding for LFV and SFV user tokens">
  <figcaption>Рисунок 3. Multi-context decoding переиспользует один encoder pass и запускает parallel decoder sub-batches для LFV и SFV tokens.</figcaption>
</figure>

Serving configuration: для каждого user извлекается 1152-dimensional dense embedding и декодируется `B = 40` SID sequences через beam search: 20 LFV и 20 SFV.

### 3.4. Downstream adaptation

SID-based user tokens надо превратить в features для rankers. Авторы сравнивают:

- Prefix Embedding Mapping: static mapping prefix SID -> mean content embedding videos с тем же prefix;
- N-gram Embedding: learnable embeddings для fixed-length SID subwords;
- SPM Embedding: SentencePiece-like variable subwords.

В production pivot study learnable embeddings лучше static prefix mapping. Это совпадает с идеей из YouTube semantic-ID ranking papers: downstream model должен учить свою task-specific embedding space для token subpieces.

## 4. Serving system

<figure class="paper-figure">
  <img src="../../assets/tokenminds/infra.png" alt="TokenMinds asynchronous serving infrastructure">
  <figcaption>Рисунок 4. TokenMinds генерирует user representations asynchronously и кладет их в cache; online scoring читает cached embeddings/tokens без запуска тяжелой модели в request path.</figcaption>
</figure>

Основные serving numbers:

- representation generation: около 339 ms per user, полностью в background processing;
- refresh cadence: 24 hours;
- discrete token representation: 1,280 bytes;
- dense embedding: 4,608 bytes;
- storage reduction for tokens vs embedding: 72%;
- cache hit rate: 96.4%;
- read load: 1.44M requests/sec from production surfaces.

Это важнее, чем сама latency model inference: TokenMinds фактически превращает LLM-style user understanding в offline feature generation layer, а не в online generative ranker.

## 5. Offline evidence

### 5.1. Training objectives

<div class="table-scroll">
<table>
<thead><tr><th>Model</th><th>Session Recall@10</th><th>Cold-Start Recall@10</th><th>Комментарий</th></tr></thead>
<tbody>
<tr><td>TokenMinds</td><td>0.291</td><td>0.210</td><td>Full setup.</td></tr>
<tr><td>w/o Multiple Targets</td><td>0.265 (-8.9%)</td><td>0.203 (-3.3%)</td><td>Predicting one target weaker.</td></tr>
<tr><td>w/o Look-ahead Window</td><td>0.278 (-4.5%)</td><td>0.189 (-10.0%)</td><td>Immediate next-watch target hurts future-interest generalization.</td></tr>
<tr><td>w/o SID Truncation</td><td>0.247 (-15.1%)</td><td>0.174 (-17.1%)</td><td>Full SIDs overfit/memorize more than coarse prefixes.</td></tr>
</tbody>
</table>
</div>

Самый важный вывод: для user modeling coarse-grained SID prefixes лучше full item IDs/SIDs. User token должен представлять interest region, а не конкретный video.

### 5.2. Initialization and search queries

<div class="table-scroll">
<table>
<thead><tr><th>Initialization</th><th>Search Queries</th><th>Session Recall@10 delta</th><th>Cold-Start Recall@10 delta</th></tr></thead>
<tbody>
<tr><td>Pre-trained</td><td>No</td><td>+3.3%</td><td>+5.7%</td></tr>
<tr><td>CPT</td><td>No</td><td>+5.3%</td><td>+8.7%</td></tr>
<tr><td>Random</td><td>Yes</td><td>+12.5%</td><td>+16.9%</td></tr>
<tr><td>Pre-trained</td><td>Yes</td><td>+18.5%</td><td>+25.1%</td></tr>
<tr><td>CPT</td><td>Yes</td><td>+23.5%</td><td>+31.5%</td></tr>
</tbody>
</table>
</div>

CPT лучше обычного pre-trained Gemini, а search queries дают огромный additive signal. Search benefit особенно растет при сильной initialization, что логично: LLM backbone лучше умеет совместить textual intent с SID-grounded behavior.

### 5.3. Diversity and embedding stability

Авторы проверяют, что beam search не схлопывает все user tokens в один и тот же prefix. На 5K users generated SID diversity сопоставима с ground-truth watches из 24h look-ahead window по duplication/collision CDFs.

Для dense embeddings они берут 2K users и сравнивают full-history embedding `E_A`, perturbed-history embedding `E_A*` и random user `E_B`: `Sim(E_A, E_A*) = 0.993`, `Sim(E_A, E_B) = 0.761`. Это сильный sanity check: embedding стабилен к небольшим perturbations, но различает пользователей.

## 6. Online evaluation

### 6.1. Token adaptation pivot

<div class="table-scroll">
<table>
<thead><tr><th>Strategy, 110M SFV</th><th>Engaged Users</th><th>Satisfied Engagement</th></tr></thead>
<tbody>
<tr><td>Prefix Embedding Mapping</td><td>+0.07%</td><td>-0.02%</td></tr>
<tr><td>Learnable Embedding, Unigram</td><td>+0.08%</td><td>+0.22%</td></tr>
</tbody>
</table>
</div>

LE wins, поэтому full-scale deployments используют learnable token embeddings.

### 6.2. Embedding vs token vs both

<div class="table-scroll">
<table>
<thead><tr><th>Surface</th><th>Representation</th><th>Engaged Users</th><th>Satisfied Engagement</th></tr></thead>
<tbody>
<tr><td>SFV</td><td>Embed-only</td><td>0.00%</td><td>+0.05%</td></tr>
<tr><td>SFV</td><td>Token-only</td><td>+0.04%</td><td>+0.40%</td></tr>
<tr><td>SFV</td><td>Embed+Token</td><td>+0.11%</td><td>+0.62%</td></tr>
<tr><td>LFV</td><td>Embed-only</td><td>+0.04%</td><td>+0.03%</td></tr>
<tr><td>LFV</td><td>Token-only</td><td>+0.01%</td><td>+0.04%</td></tr>
<tr><td>LFV</td><td>Embed+Token</td><td>+0.02%</td><td>+0.08%</td></tr>
</tbody>
</table>
</div>

На SFV tokens особенно сильны по Satisfied Engagement, а combo дает лучший result. На LFV signal слабее, но token-only additional LFV launches вне основной таблицы дали statistically significant gains +0.04%/+0.16% Engaged Users и +0.07%/+0.11% Satisfied Engagement.

### 6.3. Downstream cost

<div class="table-scroll">
<table>
<thead><tr><th>Metric</th><th>Token-only</th><th>Embed+Token</th></tr></thead>
<tbody>
<tr><td>Training Cost</td><td>+2.85%</td><td>+3.05%</td></tr>
<tr><td>Training Speed</td><td>-0.7%</td><td>-4.2%</td></tr>
<tr><td>Serving Throughput, Max QPS</td><td>-1.3%</td><td>-7.4%</td></tr>
</tbody>
</table>
</div>

Overhead невелик для token-only и заметнее для `Embed+Token`, но основная heavy computation уже вынесена из online path.

### 6.4. Cross-scenario model

<div class="table-scroll">
<table>
<thead><tr><th>Cross-scenario vs LFV-only</th><th>SFV</th><th>LFV</th></tr></thead>
<tbody>
<tr><td>Engaged Users</td><td>+0.02%</td><td>+0.00%</td></tr>
<tr><td>Satisfied Engagement</td><td>+0.03%</td><td>+0.03%</td></tr>
<tr><td>Fresh Engagement</td><td>+0.33%</td><td>+0.19%</td></tr>
</tbody>
</table>
</div>

Resource impact vs two separate models: upstream training compute -50%, upstream serving compute -31%. Это сильный production result: unified model почти не просаживает core metrics, улучшает freshness и экономит compute.

## 7. Сильные стороны

- Четко разделяет user tokens и user embeddings: tokens дают discrete multi-interest signal, embeddings сохраняют compatibility.
- Хорошая production-integrated serving story: asynchronous cache превращает heavy model в feature generator.
- Проверяет downstream adaptation, а не только token-generation Recall.
- Показывает complementary value: token-only и embed-only не заменяют друг друга.
- Cross-scenario modeling экономит compute без очевидного quality regression.

## 8. Ограничения и вопросы

Как и PLUM, работа почти полностью на закрытой YouTube инфраструктуре. Offline benchmarks, reward signals, production rankers и exact serving stack не воспроизводимы.

TokenMinds оценивается преимущественно как ranking feature source. Это не доказывает, что SID-based user tokens будут одинаково полезны для retrieval, ads, search или notifications без отдельной adaptation.

Refresh cadence 24h подходит для stable long-term interests, но может быть слабым для very fresh intent. Авторы обсуждают decoupling heavy encoder and lightweight decoder как future direction, но текущий production setup остается mostly asynchronous.

Coarse prefix truncation полезен для diversity/generalization, но может терять fine-grained intent. Нужны slice-анализы по head/tail users, new users, volatile interests и sensitive topical shifts.

## 9. Связь с соседними работами

TokenMinds - естественное продолжение PLUM. PLUM показывает, что SID vocabulary можно grounded в LLM и использовать для generative retrieval. TokenMinds использует тот же SID vocabulary как язык user interests. В связке эти две статьи задают YouTube architecture pattern: SID-first LLM adaptation, CPT, затем разные downstream heads/tasks.

По отношению к text user profiles TokenMinds прагматичнее: модель не генерирует natural-language explanation, а distill'ит preference state в compact discrete tokens, которые downstream rankers умеют потреблять.

## 10. Итог

TokenMinds показывает, что SID-based representation полезна не только для items, но и для users. Главный takeaway: discrete user tokens не должны заменять dense embeddings сразу. Лучший production path - dual-output design, где embeddings обеспечивают совместимость, а tokens добавляют multi-interest, semantically grounded signal. Именно комбинация дает самый сильный online lift.

## Источники

- [arXiv:2606.25147](https://arxiv.org/abs/2606.25147)
