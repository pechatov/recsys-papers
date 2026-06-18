---
title: "Mitigating Collaborative Semantic ID Staleness in Generative Retrieval"
category: "semantic_ids_tokenization_indexing"
slug: "mitigating_collaborative_semantic_id_staleness_in_generative_retrieval_summary"
catalogId: "paper-mitigating_collaborative_semantic_id_staleness_in_generative_retrieval_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/mitigating_collaborative_semantic_id_staleness_in_generative_retrieval_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.13273"
---
Подробное саммари статьи:

> **Авторы:** Vladimir Baikalov, Iskander Bagautdinov, Sergey Muravyov.
>
> **Аффилиации:** AI VK; ITMO University.
>
> **Конференция:** SIGIR 2026. **Код:** [github.com/iskbaga/semantic-id-alignment](https://github.com/iskbaga/semantic-id-alignment).

## 1. Коротко: о чем статья

Статья решает проблему устаревания (staleness) collaborative semantic IDs в generative retrieval. Collaborative SIDs, построенные из interaction logs, обычно дают лучший retrieval, чем content-only SIDs, но при temporal drift старые коды перестают отражать актуальную структуру взаимодействий. Полная перестройка SID vocabulary ломает совместимость с checkpoint'ом retriever'а, а наивный warm-start с новыми кодами не гарантирует улучшения. Авторы предлагают Semantic-ID Alignment Update: новые SIDs строятся по свежим данным, затем выравниваются к старому vocabulary через Greedy или Hungarian matching, после чего retriever warm-start fine-tune'ится. На VK-LSVD метод достигает Recall@500 0.4098, превосходя и stale SIDs (0.3512), и full retraining (0.3910), при 8-9x экономии compute.

## 2. Контекст: жизненный цикл SID в production

В production recommender'е данные обновляются постоянно: новые пользователи, новые item'ы, меняющиеся паттерны поведения. Content-based SIDs (из текстовых/визуальных эмбеддингов) стабильны, но не отражают collaborative structure. Collaborative SIDs (из interaction-informed embeddings, например SASRec) лучше улавливают паттерны co-click/co-purchase, но их коды зависят от конкретного snapshot логов. При обновлении логов RQ/KMeans-коды могут полностью переименоваться: кластер, который раньше назывался "42", теперь может стать "137", хотя семантически он тот же. Старая модель, обученная на старом token space, перестает понимать новые коды.

## 3. Проблема: плохой выбор между staleness и incompatibility

Без refresh: collaborative SIDs устаревают, interaction patterns смещаются, retrieval quality падает. С refresh без alignment: новые коды несовместимы с checkpoint'ом, warm-start может быть хуже, чем stale SIDs. С полным retraining: дорого (8-9x compute), неэффективно для регулярных обновлений. Авторы систематически количественно оценивают эту деградацию через rolling-origin evaluation.

## 4. Метод: Semantic-ID Alignment Update

### 4.1. SID representation

Каждый item $i \in \mathcal{I}$ получает SID длины $L$:

$$
\mathbf{z}_i = (z_{i,1}, \ldots, z_{i,L}), \quad z_{i,\ell} \in \mathcal{V}_\ell
$$

где $\mathcal{V}_\ell$ - vocabulary $\ell$-го codebook.

### 4.2. Alignment procedure

Для каждого item существуют два SID назначения: старое и новое (rebuilt из свежих логов):

$$
\mathbf{z}_i^{\text{old}} = (z_{i,1}^{\text{old}}, \ldots, z_{i,L}^{\text{old}}), \quad \mathbf{z}_i^{\text{new}} = (z_{i,1}^{\text{new}}, \ldots, z_{i,L}^{\text{new}})
$$

Цель: найти биективную функцию $\phi$, выравнивающую новые SIDs к старому token space:

$$
\tilde{\mathbf{z}}_i = (\tilde{z}_{i,1}, \ldots, \tilde{z}_{i,L}), \quad \tilde{z}_{i,\ell} = \phi_\ell(z_{i,\ell}^{\text{new}})
$$

Каждая $\phi_\ell$ - one-to-one mapping внутри одной codebook позиции.

### 4.3. Co-occurrence weight computation

Для overlap item'ов $\mathcal{I}_\cap$ (присутствующих в обоих наборах данных) вычисляется матрица co-occurrence:

$$
W_\ell(a, b) = \sum_{i \in \mathcal{I}_\cap} \mathbb{I}[z_{i,\ell}^{\text{new}} = a] \cdot \mathbb{I}[z_{i,\ell}^{\text{old}} = b]
$$

Это двудольное matching: новые active tokens сопоставляются со старыми, причем большее $W_\ell(a,b)$ означает более стабильное соответствие. Два solver'а: Greedy (дешевле, $O(M \log M)$ per position) и Hungarian (глобально оптимален, $O(V^3)$ per position).

### 4.4. Complexity

При $V=512$, $L=4$ оба метода lightweight относительно GPU training. Alignment выполняется на CPU. Unmapped новые tokens сопоставляются с оставшимися unassigned старыми tokens.

## 5. Evaluation protocol: rolling-origin

Работа моделирует production update через rolling-origin evaluation. Interactions сортируются по timestamp и делятся на 10 contiguous blocks с равным числом interactions. Base модель обучается на blocks 1-8, block 9 для model selection. Full обучается на blocks 1-9 с нуля. FT-* варианты warm-start'ятся из Base checkpoint. Evaluation на block 10 как future-item-prediction task. Inference: constrained beam search, beam width 500, restricted to valid existing SIDs.

## 6. Датасеты

<div class="table-scroll">
<table>
<tr><th>Датасет</th><th>#Users</th><th>#Items</th><th>#Interactions</th><th>Density</th></tr>
<tr><td>Amazon Beauty</td><td>22,363</td><td>12,101</td><td>198,502</td><td>0.07%</td></tr>
<tr><td>VK-LSVD</td><td>49,691</td><td>21,928</td><td>1,284,340</td><td>0.12%</td></tr>
<tr><td>Yambda</td><td>6,633</td><td>33,029</td><td>631,227</td><td>0.29%</td></tr>
</table>
</div>

VK-LSVD - short-video streaming benchmark (1% subsample). Yambda - music streaming (500M-like subsample). Amazon Beauty - стандартный e-commerce dataset.

## 7. Implementation details

Retriever: decoder-only Transformer, embedding dim 128, 4 layers, 6 heads, FFN dim 1024, ReLU, dropout 0.1. SID: RQ-VAE tokenizer, 4 codebooks по 512. Optimizer: AdamW, lr = $10^{-4}$, batch 256. Inference: constrained beam search, beam width 500. Interaction-informed tokenization использует SASRec collaborative item embeddings.

## 8. Baselines и update policies

Две парадигмы SID: TIGER-style (content-only) и LETTER-style (interaction-informed). Пять update policies: Base (fixed SIDs), FT-old (fine-tune с stale SIDs), FT-new (rebuild SIDs без alignment), FT-ours (rebuild + alignment), Full (rebuild + train from scratch).

## 9. Результаты

### 9.1. Основные результаты (block 10)

<div class="table-scroll">
<table>
<tr><th>Model / Policy</th><th>Beauty R@500</th><th>Yambda R@500</th><th>VK-LSVD R@500</th></tr>
<tr><td>TIGER (Base)</td><td>0.0885</td><td>0.0735</td><td>-</td></tr>
<tr><td>LETTER (Base)</td><td>0.1206</td><td>0.0964</td><td>0.2424</td></tr>
<tr><td>LETTER (FT-old)</td><td>0.1645</td><td>0.1116</td><td>0.3512</td></tr>
<tr><td>LETTER (FT-new)</td><td>0.1573</td><td>0.1100</td><td>0.3951</td></tr>
<tr><td>LETTER (FT-ours, Greedy)</td><td>0.1736</td><td>0.1136</td><td><strong>0.4098</strong></td></tr>
<tr><td>LETTER (FT-ours, Hungarian)</td><td><strong>0.1756</strong></td><td><strong>0.1174</strong></td><td>0.4044</td></tr>
<tr><td>LETTER (Full)</td><td>0.1992</td><td>0.1131</td><td>0.3910</td></tr>
</table>
</div>

Ключевые выводы: FT-new (rebuild без alignment) не всегда лучше FT-old (stale SIDs). На Beauty Recall@500 FT-new = 0.1573 vs FT-old = 0.1645, на Yambda 0.1100 vs 0.1116. Это доказывает, что просто обновить SIDs недостаточно: без alignment token semantics разрушаются. FT-ours consistently улучшает: Beauty 0.1756 vs 0.1645, Yambda 0.1174 vs 0.1116, VK-LSVD 0.4098 vs 0.3512. На VK-LSVD FT-ours даже превосходит Full retraining (0.4098 vs 0.3910).

### 9.2. Greedy vs Hungarian

Hungarian обычно сильнее (лучший на Beauty и Yambda), но Greedy побеждает на VK-LSVD (0.4098 vs 0.4044). Гипотеза авторов: при росте числа new block interactions Greedy может быть достаточно хорош или даже лучше, так как Hungarian оптимизирует глобальное matching, которое может быть less relevant при большом item churn.

### 9.3. Temporal adaptation (multi-step)

На VK-LSVD с инициализацией на $t=5$ и адаптацией для $t \in \{6,7,8\}$: FT-old показывает steep performance decline по шагам. FT-ours consistently поддерживает высокий Recall, на уровне Full. К финальному шагу FT-ours даже превосходит Full retraining.

### 9.4. Quality-compute trade-off

<div class="table-scroll">
<table>
<tr><th>Policy</th><th>Delta TFLOPs</th><th>Total TFLOPs</th><th>R@500</th></tr>
<tr><td>Base</td><td>-</td><td>22,975</td><td>0.2424</td></tr>
<tr><td>FT-old</td><td>2,815</td><td>25,790</td><td>0.3512</td></tr>
<tr><td>FT-new</td><td>3,003</td><td>25,978</td><td>0.3951</td></tr>
<tr><td>Full</td><td>25,283</td><td>48,258</td><td>0.3910</td></tr>
<tr><td>FT-ours (Greedy)</td><td>3,003</td><td>25,978</td><td><strong>0.4098</strong></td></tr>
</table>
</div>

FT-ours требует ту же стоимость, что FT-new (3,003 TFLOPs), но достигает лучшего Recall@500 (0.4098 vs 0.3951). По сравнению с Full экономия compute составляет 8.4x. Alignment почти не добавляет compute (CPU-only), но улучшает quality-compute frontier.

## 10. Рисунки

Figure 1 показывает pipeline: previous interaction history и new events, rebuilt SIDs, aligned SIDs и warm-start finetuning. Figure 2 - line plot Recall@500 по шагам temporal adaptation на VK-LSVD: FT-old деградирует, FT-ours держится на уровне Full.

## 10.1. Пошаговый алгоритм Semantic-ID Alignment Update

1. **Зафиксировать base state.** На старых blocks обучены old collaborative SIDs и generative retriever checkpoint.
1. **Построить fresh SIDs.** По обновленным interaction logs пересчитываются SASRec collaborative item embeddings и RQ-VAE tokenizer выдает new SID для каждого item.
1. **Найти overlap items.** Берутся item'ы, присутствующие и в old, и в new snapshots; только они дают надежный сигнал соответствия tokens.
1. **Посчитать co-occurrence matrices.** Для каждой позиции SID $\ell$ строится $W_\ell(a,b)$: сколько overlap items имели new token $a$ и old token $b$.
1. **Решить matching.** Greedy выбирает максимальные пары token-token дешево; Hungarian решает глобальную assignment задачу, но дороже.
1. **Переименовать new vocabulary.** New SID tokens заменяются через $\phi_\ell$, чтобы fresh semantics остались совместимы со старым checkpoint'ом.
1. **Warm-start fine-tune.** Retriever стартует из old checkpoint, но обучается на new events с aligned SIDs.
1. **Сравнить policies.** FT-ours должен превосходить FT-old и FT-new при той же GPU-стоимости, а Full служит дорогим upper/alternative baseline.

```
old_sid, old_model = train_on_blocks(1..8)
new_sid = rebuild_sid_on_blocks(1..9)
overlap = items_in_both_snapshots()

for position l in SID_positions:
    W = count_new_old_token_cooccurrence(new_sid, old_sid, overlap, l)
    phi_l = greedy_or_hungarian_matching(W)
    aligned_sid[:, l] = phi_l(new_sid[:, l])

model = load(old_model)
fine_tune(model, blocks=9, target_sids=aligned_sid)
evaluate on future block 10 with constrained beam search
```

## 11. Сильные стороны

- **Решает maintenance-проблему SID lifecycle.** Staleness collaborative IDs обычно игнорируется, хотя в production refresh неизбежен.
- **Alignment отделен от retriever architecture.** Метод переименовывает token space и может применяться к разным generative retrievers.
- **Quality-compute trade-off явно посчитан.** FT-ours на VK-LSVD дает Recall@500 0.4098 при той же стоимости, что FT-new, и с 8.4x экономией относительно Full.
- **Rolling-origin evaluation ближе к реальному времени.** Blocks 1-8/9/10 моделируют future drift лучше, чем случайный leave-one-out split.
- **Код опубликован.** Для SID maintenance это особенно важно, потому что детали matching и evaluation легко исказить.

## 12. Ограничения

- **Refresh все равно нужен.** Метод делает обновление совместимее, но не избавляет от периодического rebuild SIDs.
- **Высокий item churn ослабляет matching.** Если мало overlap items между old и new snapshots, co-occurrence matrix становится шумной.
- **Alignment может закрепить старые ошибки.** Если old SID space был плохим, переименование fresh tokens в старую систему частично наследует эту проблему.
- **Нужно выбирать refresh cadence.** Редкие cycles дают drift, слишком частые добавляют operational overhead и нестабильность mappings.
- **Full retraining иногда лучше.** На Beauty Full = 0.1992 Recall@500 против 0.1756 у FT-ours, значит alignment не всегда компенсирует полный retrain.

## 13. Failure modes

Пять основных: (1) высокий item churn, когда новым items нечему соответствовать в старом vocabulary; (2) cluster split/merge, когда один старый SID распадается на несколько новых semantic groups; (3) alignment закрепляет старые ошибки; (4) слишком редкие refresh cycles; (5) слишком частые refresh cycles с нестабильностью mappings.

## 14. Практические рекомендации

Хранить versioned SID vocabulary и mapping item-to-SID для каждого training block. Считать similarity matrix между old и new SID clusters. Выбирать Greedy или Hungarian matching по размеру vocabulary и latency budget. Fine-tune retriever с aligned SIDs и previous checkpoint. Мониторить Recall@500, nDCG, token drift, долю unmatched/new items и compute FLOPs. Планировать periodic full retraining как reset, если alignment gains падают.

## 15. Связь с другими работами

В отличие от CoST/AdaSID, которые улучшают качество свежих IDs, эта работа занимается жизненным циклом IDs. Она особенно связана с промышленными сценариями Spotify, Snapchat, Kuaishou, VK, где SID vocabulary живет месяцами и обязан обновляться без полного rebuild системы. С LETTER работа связана через использование interaction-informed tokenization как baseline.

## 16. Итоговая оценка

Collaborative SIDs полезны, но становятся operational liability без refresh protocol. Alignment update предлагает практичный компромисс: актуализировать семантику кодов и сохранить совместимость с уже обученным generative retriever. Особенно ценно, что на VK-LSVD alignment + warm-start превосходит full retraining при 8x меньшем compute, что делает метод привлекательным для production deployment с регулярными обновлениями.
