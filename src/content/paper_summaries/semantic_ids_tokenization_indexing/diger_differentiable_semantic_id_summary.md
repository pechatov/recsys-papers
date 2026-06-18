---
title: "DIGER: Differentiable Semantic ID for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "diger_differentiable_semantic_id_summary"
catalogId: "paper-diger_differentiable_semantic_id_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/diger_differentiable_semantic_id_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2601.19711"
---
> **Авторы:** Junchen Fu, Xuri Ge, Alexandros Karatzoglou, Ioannis Arapakis, Suzan Verberne, Joemon M. Jose, Zhaochun Ren.
>
> **Аффилиации:** University of Glasgow; Shandong University; Amazon; Telefonica Scientific Research; Leiden University.

## 1. Коротко: о чем статья

DIGER делает semantic IDs дифференцируемой частью generative recommender. В TIGER-style pipeline tokenizer обучается отдельно, затем SID mapping замораживается, и recommendation loss уже не может изменить item-to-code assignments. DIGER утверждает, что это objective mismatch: tokenizer оптимизирует reconstruction/content loss, а generator оптимизирует ranking/recommendation.

Наивное решение через straight-through estimator (STE) оказывается хуже frozen tokenizer-а: codebook быстро схлопывается, несколько кодов забирают большую часть item'ов, а остальные почти не обучаются. DIGER предлагает **DRIL** - differentiable semantic IDs with exploratory learning: Gumbel noise дает controlled exploration, soft Gumbel-Softmax path передает градиенты нескольким codewords, а uncertainty decay постепенно переводит обучение в deterministic режим.

<figure class="paper-figure">
  <img src="../../assets/diger/framework.png" alt="DIGER conventional pipeline versus differentiable semantic ID framework">
  <figcaption>Рисунок 1. DIGER устраняет gradient blocking между recommender и tokenizer: recommendation loss начинает влиять на item-to-SID mapping, но inference остается hard/discrete.</figcaption>
</figure>

## 2. Проблема objective mismatch

В двухстадийной схеме RQ-VAE tokenizer выбирает SID по content/text embeddings. Затем T5-like generator учится генерировать эти SIDs по user history. Если tokenizer создал codes, неудобные для recommendation, generator вынужден адаптироваться к плохому языку item'ов.

DIGER формализует это как restricted optimization: two-stage training ищет решение только в подмножестве mappings, созданных frozen tokenizer-ом, тогда как end-to-end optimization могла бы улучшить mapping с учетом recommendation loss. Но просто включить градиенты нельзя: дискретный argmax ломает differentiability, а STE дает collapse.

## 3. Метод: DRIL, Gumbel exploration и soft updates

Для item $v$ на уровне $j$ tokenizer считает logits $\ell_{v,j,i}$ до codewords. DIGER добавляет Gumbel noise и строит soft distribution:

$$
\tilde{y}_{v,j,i} =
\frac{\exp((\ell_{v,j,i}+g_{v,j,i})/\tau)}
{\sum_{k=1}^{K}\exp((\ell_{v,j,k}+g_{v,j,k})/\tau)}.
$$

Forward pass использует hard assignment:

$$
c_{v,j} = \arg\max_i(\ell_{v,j,i}+g_{v,j,i}),
$$

а backward pass использует weighted sum:

$$
\bar{e}_{v,j} = \sum_{i=1}^{K}\tilde{y}_{v,j,i}e_i.
$$

Ключевое отличие от STE: градиент получает не только победивший codeword. Это снижает раннюю фиксацию на нескольких кодах.

### 3.1. Uncertainty decay

DIGER использует две стратегии:

- **SDUD:** снижает масштаб шума по мере падения recommendation loss.
- **FrqUD:** применяет noise преимущественно к high-frequency codes, чтобы бороться с early domination популярных codewords.

Итоговый loss:

$$
\mathcal{L} = \mathcal{L}_{gen} + \mathcal{L}_{vq} + \mathcal{L}_{recon}.
$$

$\mathcal{L}_{gen}$ - главный recommendation signal, а VQ/reconstruction terms удерживают tokenizer от разрушения content space.

<figure class="paper-figure">
  <img src="../../assets/diger/diger_vs_ste.png" alt="DIGER versus STE validation NDCG and code balance">
  <figcaption>Рисунок 2. Диагностика на B-Shop: STE нестабилен и имеет плохой code balance, а DIGER сохраняет более ровное использование codebook и улучшает validation NDCG.</figcaption>
</figure>

## 4. Пошаговый алгоритм

1. Обучить обычный RQ-VAE tokenizer и получить initial hard SID map.
1. Обучить базовый T5-like generator на frozen SIDs.
1. Включить DRIL: hard Gumbel assignment в forward, soft Gumbel-Softmax path в backward.
1. Передавать $\mathcal{L}_{gen}$ в tokenizer/codebook.
1. Управлять exploration через SDUD или FrqUD.
1. Мониторить code usage, dead codes, SID drift и train-inference agreement.
1. На inference выключить noise и использовать deterministic SID map через обычный trie/constrained decoding.

## 5. Эксперименты

Datasets: B-Shop, I-Shop и Yelp. Content features извлекаются через LLaMA-7B из titles/descriptions. Evaluation: leave-one-out и full item ranking. Tokenizer: RQ-VAE с codebook size $K=256$, SID length $m=3$ плюс conflict code; recommender - T5-style encoder-decoder.

<div class="table-scroll">
<table>
<thead>
<tr><th>Model</th><th>B-Shop R@10</th><th>B-Shop N@10</th><th>I-Shop R@10</th><th>I-Shop N@10</th><th>Yelp R@10</th><th>Yelp N@10</th></tr>
</thead>
<tbody>
<tr><td>Two-stage</td><td>0.0610</td><td>0.0331</td><td>0.1058</td><td>0.0797</td><td>0.0407</td><td>0.0213</td></tr>
<tr><td>STE</td><td>0.0134</td><td>0.0067</td><td>-</td><td>0.0077</td><td>-</td><td>-</td></tr>
<tr><td>DIGER FrqUD</td><td>0.0683</td><td>0.0372</td><td>0.1138</td><td>0.0844</td><td>-</td><td>-</td></tr>
<tr><td>DIGER SDUD</td><td>0.0657</td><td>0.0361</td><td>0.1124</td><td>0.0823</td><td>0.0439</td><td>0.0227</td></tr>
</tbody>
</table>
</div>

Главный вывод: STE действительно ломает обучение, а DIGER улучшает frozen two-stage baseline. На Yelp DIGER по NDCG немного уступает LETTER, что авторы связывают с тем, что LETTER использует collaborative signals, а DIGER в этом setup работает только с text features.

## 6. Ablations и diagnostics

Самая разрушительная ablation - убрать Gumbel noise: NDCG@10 на B-Shop падает почти до STE-like уровня. Gaussian noise хуже Gumbel, temperature annealing хуже adaptive uncertainty decay. Оптимальный setup около $K=256$, $m=3$: меньший codebook недостаточно выразителен, больший усложняет exploration.

Для production особенно важны diagnostics:

- usage entropy по каждому codebook level;
- dead-code fraction;
- incremental и cumulative SID drift;
- train-inference agreement;
- head/tail performance.

Еще одна полезная проверка - snapshot stability. Нужно сравнить SID map до DRIL, после DRIL и после финальной deterministic фиксации. Если большая доля popular items сменила prefixes, online migration будет рискованной даже при хорошем offline Recall. Если меняются в основном tail/cold items и при этом растет code utilization, это более здоровый сценарий.

## 7. Сильные стороны

- Четко показывает, что naively differentiable SID через STE может быть хуже frozen tokenizer-а.
- Сохраняет hard SID inference API, поэтому метод совместим с trie/beam search.
- Дает полезные diagnostics для любого joint SID learning.
- Ablations хорошо отделяют Gumbel exploration, uncertainty decay и soft update.

## 8. Ограничения

Joint training усложняет versioning: SID mapping может меняться между checkpoints, а значит prefix tree, cached maps и generator должны обновляться синхронно.

Метод добавляет hyperparameters: temperature, noise schedule, FrqUD threshold, VQ/reconstruction weights. Плохая настройка может вернуть collapse или train-serve mismatch.

Эксперименты относительно небольшие. На catalog масштаба миллионов items cost joint training и drift management могут выглядеть иначе.

## 9. Вывод

DIGER важен как предупреждение и как решение. Предупреждение: differentiable semantic IDs не являются бесплатным улучшением, STE может разрушить codebook. Решение: controlled exploration через Gumbel noise позволяет recommendation loss улучшать SID mapping, сохраняя discrete inference.
