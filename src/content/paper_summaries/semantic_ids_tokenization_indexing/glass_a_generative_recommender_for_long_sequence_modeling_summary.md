---
title: "GLASS: A Generative Recommender for Long-sequence Modeling via SID-Tier and Semantic Search"
category: "semantic_ids_tokenization_indexing"
slug: "glass_a_generative_recommender_for_long_sequence_modeling_summary"
catalogId: "paper-glass_a_generative_recommender_for_long_sequence_modeling_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/glass_a_generative_recommender_for_long_sequence_modeling_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2602.05663"
---
Подробное саммари статьи:

> **Авторы:** Shiteng Cao, Junda She, Ji Liu, Bin Zeng, Chengcheng Guo, Kuo Cai, Qiang Luo, Ruiming Tang, Han Li, Kun Gai, Zhiheng Li, Cheng Yang.
>
> **Аффилиации:** Kuaishou Inc.; Tsinghua University; BUPT.

## 1. Коротко: о чем статья

GLASS решает проблему длинных пользовательских историй в генеративных рекомендательных системах на основе semantic ID. В таких системах каждый item кодируется несколькими токенами (обычно 3), и длинная история из сотен item'ов превращается в последовательность из тысяч токенов, что создает вычислительные и качественные проблемы.

GLASS предлагает два ключевых механизма. **SID-Tier** агрегирует долгосрочную историю в единый interest vector, используя иерархическую структуру semantic ID и компактность codebook'а для cross-feature extraction. **Semantic Hard Search** использует сгенерированный первый (coarse) SID-токен как ключ для поиска релевантных item'ов из длинной истории, дополняя autoregressive decoding персонализированным контекстом. Дополнительно предлагаются две стратегии борьбы с разреженностью: semantic neighbor augmentation и codebook resizing.

На TAOBAO-MM и KuaiRec GLASS показывает значительные улучшения: до +21.57% Hit@1 и +29.91% NDCG@3 на TAOBAO-MM относительно сильнейшего baseline.

## 2. Контекст: зачем нужна работа с длинными последовательностями

Промышленные рекомендательные системы используют двухстадийную парадигму: General Search Unit (GSU) фильтрует кандидатов из длинной истории, а Exact Search Unit (ESU) точно оценивает отфильтрованных. Критическая проблема retrieval-стадии -- **target-absent dilemma**: в отличие от ranking, где целевой item известен, retrieval не имеет явного target'а, и недавнее поведение пользователя служит лишь субоптимальным proxy.

Генеративные рекомендательные системы (TIGER, OneRec) переосмысливают retrieval как autoregressive generation hierarchических Semantic ID, но вычислительная стоимость растет квадратично с длиной последовательности. При истории в 1000 item'ов и 3 SID-токенах на item получается последовательность из 3000 токенов -- слишком дорого для полного self-attention.

Авторы также идентифицируют **эффект накопления ошибки**: после генерации первого SID-токена "фокус модели на ground-truth item'е часто деградирует, а не улучшается". Это означает, что ошибка на coarse уровне каскадно ухудшает precision на fine-grained уровнях.

## 3. Проблема

Три основных вызова, на которые отвечает GLASS.

**Недоиспользование длинной истории.** Стандартные GR-модели работают с короткими последовательностями (50 item'ов), игнорируя сотни и тысячи прошлых взаимодействий, которые содержат долгосрочные интересы пользователя.

**Деградация precision на глубоких SID-уровнях.** Вероятность правильного пути экспоненциально уменьшается: $P(\text{path}) = P(\text{sid}_1) \cdot P(\text{sid}_2|\text{sid}_1) \cdot P(\text{sid}_3|\text{sid}_1, \text{sid}_2)$. Если $P(\text{sid}_2|\text{sid}_1)$ падает после корректного $\text{sid}_1$, каскадная ошибка растет.

**Разреженность semantic space.** При 1000 item'ов и codebook размером 128, каждый SID-bucket содержит в среднем лишь ~7.8 item'ов. Для редких bucket'ов retrieved context может быть пустым или слишком шумным.

## 4. SID-Tier: агрегация долгосрочных интересов

SID-Tier превращает длинную историю в компактный interest vector, используя иерархическую природу semantic ID.

Для каждого first-level codeword $a \in \mathcal{C}_0$ вычисляется prototype embedding -- среднее по всем item'ам, попавшим в этот bucket:

$$
\tilde{\mathbf{h}}_a = \text{Pooling}(\{\mathbf{h}_v \mid v \in \mathcal{V},\; c_0(v) = a\})
$$

Затем для каждого prototype вычисляется cosine similarity с item'ами из длинной истории:

$$
\mathbf{S}_a = \{\cos(\tilde{\mathbf{h}}_a, \mathbf{h}_v) \mid v \in H^\text{id}_\text{long},\; a \in \mathcal{C}_0\}
$$

Диапазон similarity $[-1, 1]$ разбивается на $N$ дискретных tier'ов, и для каждого prototype строится tier-histogram:

$$
\mathbf{t}_a[i] = \sum_{s \in \mathbf{S}_a} \mathbb{I}(s \in \text{Tier}_i)
$$

Все histograms конкатенируются и проецируются через MLP:

$$
\mathbf{x} = \text{Concat}(\mathbf{t}_0, \mathbf{t}_1, \ldots, \mathbf{t}_{K_0 - 1}) \in \mathbb{R}^{K_0 \times N}
$$

$$
\mathbf{e}_\text{tier} = \text{MLP}(\mathbf{x}) \in \mathbb{R}^d
$$

Полученный interest vector добавляется к encoder input короткой последовательности:

$$
\mathbf{O}^\text{enc}_\text{short} = \text{Encoder}(\mathbf{E}_\text{short} \| \mathbf{e}_\text{tier})
$$

Ключевая идея: вместо прямой обработки тысяч токенов длинной истории (квадратичная стоимость), SID-Tier использует компактность codebook'а ($K_0$ prototype'ов) как "мост" для cross-feature extraction. Это позволяет агрегировать глобальные предпочтения с фиксированной стоимостью, не зависящей от длины истории.

## 5. Semantic Hard Search: декодирование с retrieval

После генерации первого SID-токена $\hat{c}_0(v_{n+1})$ выполняется hard search по длинной истории:

$$
H_\text{ret} = \{c(v_i) \mid v_i \in H^\text{id}_\text{long},\; c_0(v_i) = \hat{c}_0(v_{n+1})\}
$$

Это RAG-подобный подход: сгенерированный coarse SID-токен используется как поисковый ключ для извлечения контекста из длинной истории. Результат -- набор SID-последовательностей item'ов, чей первый код совпадает с предсказанным.

Два параллельных cross-attention модуля обрабатывают короткий контекст и retrieved контекст:

$$
\mathbf{Z}_\text{short} = \text{CrossAttn}_\theta(\mathbf{X}, \mathbf{O}^\text{enc}_\text{short}, \mathbf{O}^\text{enc}_\text{short})
$$

$$
\mathbf{Z}_\text{ret} = \text{CrossAttn}_\phi(\mathbf{X}, \mathbf{E}_\text{ret}, \mathbf{E}_\text{ret})
$$

Adaptive gate контролирует баланс:

$$
\mathbf{g} = \sigma([\mathbf{Z}_\text{short}; \mathbf{Z}_\text{ret}] \mathbf{W}_g)
$$

$$
\mathbf{Z}_\text{context} = (\mathbf{1} - \mathbf{g}) \cdot \mathbf{Z}_\text{short} + \mathbf{g} \cdot \mathbf{Z}_\text{ret}
$$

Модель учится автоматически определять, когда retrieved context полезен (большой gate value), а когда лучше полагаться на короткую историю (малый gate value). Анализ показывает устойчивый восходящий тренд: чем длиннее retrieved sequence, тем выше average gate value.

## 6. Борьба с разреженностью

### 6.1. Semantic Neighbor Augmentation

Для каждого first-level code вычисляется попарная cosine similarity с другими кодами, и сохраняются top-k ближайших соседей. Если количество retrieved item'ов $|R(\text{SID}_1)| < \tau$, активируется протокол augmentation: к retrieved set'у добавляются item'ы из соседних bucket'ов.

### 6.2. Codebook Resizing

Размер first-level codebook ограничивается при обучении для увеличения плотности item'ов в каждом bucket'е. Авторы описывают это как "assigning density to the primary layer and precision to subsequent layers". При $L = 1000$ item'ов и $|\mathcal{C}_1| = 128$ каждый SID-bucket содержит в среднем лишь ~7.8 item'ов; уменьшение codebook'а до 64 удваивает эту плотность.

## 7. Теоретическое обоснование: Conditional Rank Progression

Авторы вводят метрику Conditional Rank Progression (CRP):

$$
\text{CRP}_d = \frac{1}{|\mathcal{E}_d|} \sum_{n \in \mathcal{E}_d} R_d^{(n)}
$$

где $\mathcal{E}_d$ -- множество примеров с корректными prefix'ами до глубины $d-1$, $R_d$ -- ранг ground-truth в beam. CRP измеряет деградацию точности на каждом уровне SID.

При beam size 20: baseline GR-модель показывает рост CRP на 1.25 на уровне 2. Semantic Hard Search снижает рост до 0.98, **уменьшая деградацию на 22%**. На уровне 3 SHS уменьшает деградацию на **31%** (с 0.19 до 0.13).

Теоретическое объяснение: SHS добавляет auxiliary context к вероятностям последующих токенов:

$$
P(\text{path}_i|\text{context}) = P(\text{sid}_1|\text{context}) \cdot P(\text{sid}_2|\text{sid}_1, \text{context}, \text{context}') \cdot P(\text{sid}_3|\text{sid}_1, \text{sid}_2, \text{context}, \text{context}')
$$

Auxiliary context (retrieved по $\text{sid}_1$) инжектирует персонализированное знание, отсутствующее в исходном контексте, расширяя probability gap между ground-truth и distractor'ами.

### 7.1. Пошаговый алгоритм GLASS

В реализации GLASS важно разделять offline подготовку SID buckets и online decoding, где первый сгенерированный SID-токен становится ключом hard search.

```
offline:
    train or load SID tokenizer with codebooks C0, C1, C2
    for item in catalog:
        sid[item] = tokenizer(item)
        first_level_bucket[sid[item][0]].append(item)
    for first_token in C0:
        prototype[first_token] = mean_pool(embeddings in first_level_bucket[first_token])
        semantic_neighbors[first_token] = top_k_similar_prototypes(first_token)

training:
    short_history = last_50_items(user_history)
    long_history = older_items(user_history)
    tier_histograms = []
    for first_token in C0:
        sims = cosine(prototype[first_token], embeddings(long_history))
        tier_histograms.append(discretize_to_tiers(sims))
    e_tier = MLP(concat(tier_histograms))
    encoder_context = encoder(short_history, extra_vector=e_tier)

    first_sid = decoder.predict_first_token(encoder_context)
    retrieved = items_from_long_history_with_first_sid(long_history, first_sid)
    if len(retrieved) < tau:
        retrieved += items_from_neighbor_buckets(long_history, semantic_neighbors[first_sid])
    z_short = cross_attention(decoder_state, encoder_context)
    z_ret = cross_attention(decoder_state, encode(retrieved))
    gate = sigmoid(linear(concat(z_short, z_ret)))
    context = (1 - gate) * z_short + gate * z_ret
    predict_remaining_sid_tokens(context)
```

## 8. Эксперименты и результаты

### 8.1. Датасеты

<div class="table-scroll">
<table>
<tr><th>Статистика</th><th>TAOBAO-MM</th><th>KuaiRec</th></tr>
<tr><td>#Users</td><td>189,423</td><td>3,790</td></tr>
<tr><td>#Items</td><td>10,000</td><td>10,728</td></tr>
<tr><td>#Training samples</td><td>265,301</td><td>30,320</td></tr>
<tr><td>Avg seq length</td><td>987</td><td>884</td></tr>
</table>
</div>

TAOBAO-MM: multimodal e-commerce, 10K items отобраны с порогом >= 5 появлений. KuaiRec: short video, watch ratio > 100%. Последние 50 interaction'ов -- short-term, остальные -- long-term.

### 8.2. Основные результаты (TAOBAO-MM)

<div class="table-scroll">
<table>
<tr><th>Model</th><th>H@1</th><th>H@5</th><th>N@3</th><th>N@20</th></tr>
<tr><td>HSTU</td><td>0.0274</td><td>0.0512</td><td>0.0384</td><td>0.0592</td></tr>
<tr><td>Tiger</td><td>0.0280</td><td>0.0756</td><td>0.0448</td><td>0.0733</td></tr>
<tr><td><strong>GLASS</strong></td><td><strong>0.0372</strong></td><td><strong>0.0934</strong></td><td><strong>0.0669</strong></td><td><strong>0.0893</strong></td></tr>
<tr><td>Improvement</td><td>+21.57%</td><td>+19.59%</td><td>+29.91%</td><td>+21.00%</td></tr>
</table>
</div>

### 8.3. Результаты на KuaiRec

<div class="table-scroll">
<table>
<tr><th>Model</th><th>H@1</th><th>H@5</th><th>H@10</th><th>N@3</th></tr>
<tr><td>Tiger</td><td>0.0406</td><td>0.0859</td><td>0.1202</td><td>0.0574</td></tr>
<tr><td>DualGR</td><td>0.0445</td><td>0.0932</td><td>0.1292</td><td>0.0632</td></tr>
<tr><td><strong>GLASS</strong></td><td><strong>0.0467</strong></td><td><strong>0.0971</strong></td><td><strong>0.1345</strong></td><td><strong>0.0659</strong></td></tr>
<tr><td>Improvement</td><td>+4.94%</td><td>+4.18%</td><td>+4.10%</td><td>+4.27%</td></tr>
</table>
</div>

Значительно больший прирост на TAOBAO-MM объясняется более качественными multimodal features из supervised contrastive learning, тогда как KuaiRec использует простые text-based embeddings.

### 8.4. Ablation Study

SIDTier поднимает Recall@5 с ~0.0702 до 0.0842; $\text{acc}_1$ растет с 0.2410 до 0.2485; $P_1$ -- с 0.1470 до 0.1587. Joint model (Tiger+SIDTier+SHS) значительно улучшает $\text{acc}_2$ и $P_2$. Tiger+SIDTier+SHS+CR достигает peak Recall@10 ~0.13. Semantic Neighbor Augmentation при мелком codebook может вносить шум, но становится эффективнее в более fine-grained semantic space.

## 9. Анализ gating mechanism

Fixed weight (например, 0.5) работает для малых codebook'ов, но деградирует при большей гранулярности из-за высокой дисперсии retrieved sequences. Adaptive gate показывает устойчивый восходящий тренд: по мере увеличения длины retrieved sequence average gate value растет. Модель учится придавать больший вес retrieved sequences, когда они становятся более информативными.

Две конфигурации проанализированы: codebook [128,128,128] и [64,128,128]. При меньшем first-level codebook плотность bucket'ов выше, retrieved set'ы богаче, и gate value систематически выше.

## 10. Сильные стороны

**Явное использование иерархии SID.** GLASS -- одна из первых работ, которая не просто использует SID как плоскую последовательность, а эксплуатирует coarse-to-fine структуру: first-level код определяет semantic region, а subsequent коды уточняют item. Это inductive bias, который natural для hierarchical codebooks.

**RAG-подобный подход к generative recommendation.** Semantic Hard Search привносит идею retrieval-augmented generation в мир SID-based рекомендаций. Сгенерированный coarse код становится поисковым ключом, а retrieved items -- дополнительным контекстом для decoder'а.

**Теоретическое обоснование через CRP.** Метрика Conditional Rank Progression дает формальный инструмент для измерения деградации точности на каждом уровне SID. Это полезно не только для GLASS, но и как diagnostic metric для любого SID-based GR.

**Практичность для industrial длинных историй.** Средняя длина последовательности в экспериментах -- ~1000 item'ов, что ближе к реальным production сценариям, чем типичные 20-50 item'ов в академических benchmark'ах.

## 11. Ограничения

**Сложность системы.** GLASS добавляет SID-Tier module, Semantic Hard Search с двумя cross-attention'ами, adaptive gate, codebook resizing и neighbor augmentation. Это значительно сложнее baseline TIGER и требует тщательной настройки множества hyperparameter'ов.

**Зависимость от качества SID tokenizer.** Все механизмы GLASS строятся поверх существующих SID. Если tokenizer плохо разделяет item'ы (высокий collision rate, плохая semantic grouping), SID-Tier будет агрегировать шум, а SHS будет извлекать нерелевантные item'ы.

**Разреженность как системная проблема.** При 10K item'ов и codebook размером 128, средний bucket содержит ~78 item'ов. Но при росте каталога до миллионов (production scale) ситуация радикально меняется, и выводы о codebook resizing и neighbor augmentation могут не перенестись.

**Нет полноценного online A/B.** Статья не раскрывает детали online deployment, хотя авторы из Kuaishou имеют доступ к production среде.

**Латентность search component.** Semantic Hard Search требует runtime lookup'а в SID index, что добавляет latency к inference pipeline. Авторы не детализируют timing impact.

## 12. Практические выводы

**SID -- это не плоская строка.** GLASS убедительно демонстрирует, что если модель и decoding учитывают coarse-to-fine структуру semantic ID, long-sequence generative recommendation становится устойчивее. Первый SID-токен определяет semantic region, и его качество критично.

**Comбинация generation + search.** Вместо чистого autoregressive decoding полезно дополнять генерацию retrieval'ом из истории. Это особенно важно для mid-to-late стадий генерации, где model confidence падает.

**Диагностика per-level accuracy.** При разработке SID-based системы необходимо логировать accuracy по позициям SID ($\text{acc}_1, \text{acc}_2, \text{acc}_3$), prefix ceiling, gate value и latency search. Без этого невозможно понять, на каком уровне система теряет качество.

## 13. Связь с соседними работами

GLASS дополняет работы по variable-length SID и Spotify GLIDE: все три обращаются к проблеме token budget'а и длинных последовательностей, но GLASS фокусируется на tier-aware generation. С DualGR его роднит использование SID как hard search key, но DualGR "overlooks the inherent structural properties of SIDs". С RecoChain общая идея -- добавить post-generation search/ranking вместо чистой генерации.

## 14. Итоговая оценка

GLASS показывает, что semantic ID -- это не просто плоская строка токенов для autoregressive generation. Если модель и decoding учитывают coarse-to-fine иерархию, а генерация дополняется retrieval'ом из длинной истории, quality и robustness generative recommendation существенно возрастают. Метрика CRP и анализ rank degradation дают полезный инструментарий для диагностики любой SID-based системы. Основное ограничение -- сложность системы и зависимость от качества исходного tokenizer'а.
