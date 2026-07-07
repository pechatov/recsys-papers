---
title: "ChronoID: Infusing Explicit Temporal Signals into Semantic IDs for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "chronoid_infusing_explicit_temporal_signals_into_semantic_ids_for_generative_recommendation_summary"
catalogId: "paper-chronoid_infusing_explicit_temporal_signals_into_semantic_ids_for_generative_recommendation_summary"
paperUrl: "https://arxiv.org/abs/2606.14260"
---
> **Авторы:** Dongdong Nian, Dongqi Fu, Chenliang Xu, Yinglong Xia, Hong Li, Hong Yan, Jian Kang.
>
> **Аффилиации:** University of Rochester; Meta MRS; MBZUAI.
>
> **Источник:** arXiv:2606.14260, submitted 2026-06-12. В TeX source есть commented metadata для GitHub, но публичная code link в arXiv page не раскрыта.

## 1. Коротко

ChronoID критикует стандартную SID-парадигму за time-agnostic tokenization. В обычном generative recommendation время влияет через порядок sequence, session construction, positional encodings или training sampling, но сам item SID остается одинаковым для разных temporal contexts. Авторы считают это неправильной semantic abstraction: один и тот же item может иметь разную relevance и смысл при разных rhythms, seasons или user intervals.

Работа предлагает не один конкретный tokenizer, а design space для time-aware semantic IDs: как кодировать время, где fusion'ить time и item semantics, и какой quantizer использовать. Главный эмпирический вывод: explicit temporal signal действительно помогает, relative time intervals лучше absolute timestamps, а лучший общий вариант в таблицах - `Parallel Quantization + Relative Time`.

Важно: ChronoID также строит time-aware benchmark с temporal cutoff. Codebook training, SFT train и SFT test разделены по времени, чтобы tokenizer не учился на future interactions. Это делает задачу ближе к production setting, где future items/interactions нельзя leak'нуть в SID mapping.

<figure class="paper-figure">
  <img src="../../assets/chronoid/framework.png" alt="ChronoID framework with early fusion, late fusion, and parallel quantization">
  <figcaption>Рисунок 1. ChronoID исследует три варианта: early fusion, late fusion и parallel quantization для time-aware SID learning.</figcaption>
</figure>

## 2. Проблема

В SID-based GR item обычно получает content embedding, затем RQ-VAE/quantizer выдает discrete code sequence. Дальше LLM/seq2seq model генерирует SID target item. Даже если user sequence ordered, сам SID не знает, было ли взаимодействие вчера, год назад, в holiday season или после короткого/длинного interval.

ChronoID формулирует это как gap между dynamic interaction context и static semantic abstraction. Time-aware modeling на уровне sequence может помочь user dynamics, но не меняет discrete vocabulary. Если два interactions с одним item в разных temporal contexts получают один SID, generator вынужден объяснять time-dependent relevance только внешним positional signal.

## 3. Метод: три design dimensions

### 3.1. Time embedding

Время кодируется sinusoidal positional encoding:

$$
h_t[2i] = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad
h_t[2i+1] = \cos\left(\frac{t}{10000^{2i/d}}\right).
$$

Сравниваются два входа:

- absolute timestamp: UNIX time события, должен ловить global trends и seasonality;
- relative time: interval между соседними interactions пользователя, $\Delta t_{u,i} = t_{u,i} - t_{u,i-1}$.

Авторы ожидают, что relative time лучше для generative SID, потому что quantized token sequence не сохраняет continuous inner-product geometry sinusoidal embeddings. Relative intervals прямо кодируют user rhythm: immediate repeat, periodic replacement, long inactivity.

### 3.2. Fusion strategy

Early fusion: concatenate item text embedding и time embedding до quantization:

$$
h = [h_{item} || h_t], \quad SID = Q(h).
$$

Late fusion: quantize item and time independently, затем concat discrete IDs:

$$
SID = [ID_{item} || ID_{time}].
$$

Идея late fusion - не заставлять один codebook сжимать гетерогенные distributions item semantics и temporal signals.

### 3.3. Quantization mechanism

Residual quantization: RQ-VAE-like hierarchy, каждый следующий codebook моделирует residual от предыдущего.

Parallel quantization: несколько независимых codebooks/encoders quantize one fused representation и создают facets. В ChronoID parallel quantization получает early-fused input `[item || time]`, но codebooks независимы, поэтому меньше страдает от residual error propagation.

## 4. Benchmark без future leakage

ChronoID расширяет Amazon Industrial и Office, а также использует Mercari. Для Industrial/Office авторы вводят global temporal cutoff, например 2018-01-01:

- codebook training использует только interactions до cutoff;
- SFT train examples имеют target item до cutoff;
- SFT test targets находятся после cutoff.

Это важно: если SID tokenizer обучать на всех interactions сразу, то future item/co-occurrence information может leak'нуть в codebook. ChronoID делает более строгую проверку того, насколько time-aware SID помогает predict future interactions.

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>Split</th><th>Codebook items</th><th>SFT train</th><th>SFT valid</th><th>SFT test</th></tr></thead>
<tbody>
<tr><td>Industrial</td><td>Original</td><td>43,102</td><td>29,446</td><td>3,682</td><td>3,683</td></tr>
<tr><td>Industrial</td><td>Time-Aware</td><td>39,636</td><td>19,368</td><td>2,422</td><td>2,421</td></tr>
<tr><td>Office</td><td>Original</td><td>472,091</td><td>328,422</td><td>41,054</td><td>41,054</td></tr>
<tr><td>Office</td><td>Time-Aware</td><td>436,775</td><td>229,920</td><td>28,740</td><td>28,740</td></tr>
</tbody>
</table>
</div>

Item text embeddings: Qwen3-Embedding-4B, 2560 dimensions. Time embeddings: 768 dimensions. Quantizer input for fused setup: 3328 dimensions. Codebook training config: 3 codebooks, 256 codes per codebook, code dimension 42, 10,000 epochs. SFT uses AdamW, learning rate 3e-4, 10 epochs.

## 5. Main results

Метрики: HR@K и NDCG@K, leave-one-out with one ground-truth item per test instance.

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>Method</th><th>Strategy</th><th>HR@3</th><th>NDCG@3</th><th>HR@10</th><th>NDCG@10</th></tr></thead>
<tbody>
<tr><td>Industrial</td><td>MiniOneRec</td><td>Text + random abs time</td><td>9.26</td><td>8.44</td><td>13.53</td><td>10.01</td></tr>
<tr><td>Industrial</td><td>ChronoID</td><td>Parallel, Relative</td><td>12.60</td><td>11.15</td><td>16.22</td><td>12.41</td></tr>
<tr><td>Office</td><td>MiniOneRec</td><td>Text + random abs time</td><td>6.01</td><td>4.89</td><td>9.53</td><td>6.22</td></tr>
<tr><td>Office</td><td>ChronoID</td><td>Parallel, Relative</td><td>8.42</td><td>7.08</td><td>13.59</td><td>8.95</td></tr>
<tr><td>Mercari</td><td>MiniOneRec</td><td>Text + random abs time</td><td>1.61</td><td>1.08</td><td>2.98</td><td>1.82</td></tr>
<tr><td>Mercari</td><td>ChronoID</td><td>Parallel, Relative</td><td>3.28</td><td>2.59</td><td>5.78</td><td>3.50</td></tr>
</tbody>
</table>
</div>

Relative gains над MiniOneRec по HR@3:

- Industrial: 12.60 vs 9.26, +36.1%;
- Office: 8.42 vs 6.01, +40.1%;
- Mercari: 3.28 vs 1.61, примерно +103.7%.

Сравнение с TokenRec тоже в пользу ChronoID Parallel-Relative: Industrial HR@3 12.60 vs 8.20, Office 8.42 vs 7.54, Mercari 3.28 vs 1.34.

## 6. What drives the gain

### 6.1. Relative time beats absolute time

Во всех ключевых configurations relative time лучше absolute. Примеры:

- Industrial, RQ early: HR@3 10.62 relative vs 7.44 absolute;
- Industrial, parallel: 12.60 relative vs 11.22 absolute;
- Office, parallel: 8.42 relative vs 7.52 absolute;
- Mercari, parallel: 3.28 relative vs 2.07 absolute.

Интерпретация авторов: absolute timestamp ловит seasonality, но может страдать от monotonic distribution shift. Relative intervals лучше отражают repeat/replacement rhythm и обобщаются между users.

### 6.2. Parallel quantization is the robust winner

В main table лучший вариант в каждом dataset - `Parallel Quantization + Relative Time`. Это сильнее, чем спор early/late fusion внутри residual quantization.

Авторы утверждают, что late fusion лучше early fusion, потому что item/time feature spaces heterogeneous. Но таблица не абсолютно однозначна: на Industrial среди residual models early relative HR@3/HR@10 выше late relative, на Office и Mercari late часто лучше. Более надежный вывод: residual fusion choice dataset-dependent, а parallel + relative consistently dominates.

### 6.3. t-SNE diagnostics

<div class="table-scroll">
<table>
<tbody>
<tr>
<td><img src="../../assets/chronoid/parallel_tsne.png" alt="ChronoID t-SNE for parallel quantization"></td>
<td><img src="../../assets/chronoid/rq_absolute_tsne.png" alt="ChronoID t-SNE for residual quantization with absolute time"></td>
<td><img src="../../assets/chronoid/rq_relative_tsne.png" alt="ChronoID t-SNE for residual quantization with relative time"></td>
</tr>
<tr>
<td>Parallel quantization</td>
<td>Residual, absolute time</td>
<td>Residual, relative time</td>
</tr>
</tbody>
</table>
</div>

t-SNE по top-10 frequent SIDs показывает более compact clusters для parallel quantization и relative time, тогда как residual absolute time дает более dispersed/overlapping structure. Это diagnostic evidence, не causal proof, но хорошо согласуется с main metrics.

## 7. Sanity checks and ablations

### 7.1. Не просто больше ID capacity

<div class="table-scroll">
<table>
<thead><tr><th>Configuration, Industrial</th><th>HR@3</th><th>NDCG@3</th><th>HR@10</th><th>NDCG@10</th></tr></thead>
<tbody>
<tr><td>3-digit SID, text only</td><td>8.64</td><td>7.77</td><td>11.47</td><td>8.78</td></tr>
<tr><td>4-digit SID, text only</td><td>8.02</td><td>7.17</td><td>10.66</td><td>8.25</td></tr>
<tr><td>ChronoID, 3-digit, Parallel-Rel</td><td>12.60</td><td>11.15</td><td>16.22</td><td>12.41</td></tr>
</tbody>
</table>
</div>

Прирост не объясняется просто увеличением SID length/capacity. 4-digit text-only ухудшается относительно 3-digit text-only, а time-aware 3-digit резко лучше.

### 7.2. Remove vs zero-padding

<div class="table-scroll">
<table>
<thead><tr><th>Variant, Industrial relative time</th><th>HR@3</th><th>NDCG@3</th><th>HR@10</th><th>NDCG@10</th></tr></thead>
<tbody>
<tr><td>ChronoID standard</td><td>10.43</td><td>9.28</td><td>13.20</td><td>10.29</td></tr>
<tr><td>Remove temporal digit</td><td>9.47</td><td>8.70</td><td>12.58</td><td>9.76</td></tr>
<tr><td>Replace with zero-padding</td><td>9.04</td><td>7.72</td><td>12.85</td><td>9.47</td></tr>
</tbody>
</table>
</div>

Zero-padding хуже удаления: all-zero signal может быть OOD noise для LLM/SID space. Это полезный engineering warning: нельзя просто зарезервировать time slot и заполнить его константой.

### 7.3. High-level temporal tags

Weekend/season/holiday binary tags дают marginal или inconsistent gains. Для best parallel-relative Industrial HR@3 растет 12.60 -> 12.96, HR@10 16.22 -> 16.33, но NDCG@10 падает 12.41 -> 11.68. Авторы считают, что raw timestamp/relative signals уже позволяют internalize high-level calendar patterns, а hand-crafted tags часто redundant.

### 7.4. Hyperparameters

Лучший tested time embedding dimension - 768:

<div class="table-scroll">
<table>
<thead><tr><th>Time dimension</th><th>HR@3</th><th>NDCG@3</th><th>HR@10</th><th>NDCG@10</th></tr></thead>
<tbody>
<tr><td>512</td><td>10.08</td><td>9.16</td><td>14.15</td><td>10.48</td></tr>
<tr><td>768</td><td>10.62</td><td>9.56</td><td>14.29</td><td>10.86</td></tr>
<tr><td>1280</td><td>8.99</td><td>8.12</td><td>12.66</td><td>9.45</td></tr>
</tbody>
</table>
</div>

Лучшее число codebooks в tested residual setup - 3. При 5 codebooks качество падает, и авторы связывают это с discretizing noise in later residual layers.

## 8. Сильные стороны

- Явно атакует слепое место SID papers: tokenizer обычно time-agnostic.
- Benchmark protocol лучше защищен от future leakage, чем обычные random/leave-one-out splits.
- Хорошая decomposition: time encoding, fusion, quantizer type проверяются раздельно.
- Sanity checks отделяют temporal semantics от простого роста ID capacity.

## 9. Ограничения и вопросы

Результаты в основном на Amazon Industrial/Office и Mercari. Это полезные domains, но они не доказывают перенос на fast-changing video/news/music feeds, где temporal dynamics сильнее и шумнее.

Time-aware SID может увеличивать volatility mapping: один item потенциально получает разные semantic IDs в разных temporal contexts. Для production это усложняет caching, index update, collision policy и backward compatibility с уже обученными generators.

В тексте есть некоторая неоднозначность вокруг fusion conclusion. Paper claims late fusion consistently outperforms early fusion, но main table для Industrial residual setup показывает early relative лучше late relative по HR@3/HR@10. Практически стоит запомнить не "late always wins", а "parallel relative wins in this paper".

Нет online/industrial deployment evidence. В отличие от PLUM/TokenMinds, ChronoID пока остается benchmark/methodology paper.

## 10. Связь с соседними работами

ChronoID хорошо дополняет PLUM и TokenMinds. PLUM/TokenMinds показывают, что SID vocabulary можно запустить в YouTube-scale systems, но почти не раскрывают explicit temporal SID design. ChronoID говорит: если SID становится "языком" recommendation, в нем должны быть не только content/collaborative semantics, но и temporal semantics.

По отношению к DACT/continual tokenization ChronoID задает другую ось: не как обновлять tokenizer при drift, а как встроить time прямо в token semantics. Эти направления легко объединяются: time-aware SIDs все равно будут нуждаться в continual stability controls.

## 11. Итог

ChronoID полезен как framework для проверки time-aware SID design. Самый надежный takeaway: temporal signal на уровне semantic ID реально дает gain, relative intervals лучше absolute timestamps, а parallel quantization лучше residual hierarchy для смешивания item semantics и time. Для production нужно дополнительно решить stability и serving problems, потому что time-aware IDs могут стать менее статичными, чем привычные content SIDs.

## Источники

- [arXiv:2606.14260](https://arxiv.org/abs/2606.14260)
