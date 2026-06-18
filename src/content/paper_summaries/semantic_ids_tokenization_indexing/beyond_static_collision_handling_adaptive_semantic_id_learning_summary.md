---
title: "Beyond Static Collision Handling: Adaptive Semantic ID Learning for Multimodal Recommendation at Industrial Scale"
category: "semantic_ids_tokenization_indexing"
slug: "beyond_static_collision_handling_adaptive_semantic_id_learning_summary"
catalogId: "paper-beyond_static_collision_handling_adaptive_semantic_id_learning_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/beyond_static_collision_handling_adaptive_semantic_id_learning_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.23522"
---
Подробное саммари статьи:

> **Авторы:** Yongsen Pan, Yuxin Chen, Zheng Hu, Xu Yuan, Daoyuan Wang, Yuting Yin, Songhao Ni, Hongyang Wang, Jun Wang, Fuji Ren, Wenwu Ou.
>
> **Аффилиации:** University of Electronic Science and Technology of China (UESTC); Kuaishou Technology.
>
> **ID:** arXiv:2604.23522, апрель 2026.

## 1. Коротко: о чем статья

AdaSID (Adaptive Semantic ID Learning) решает проблему статической обработки коллизий в semantic IDs. Авторы предлагают двухстадийный адаптивный framework: первая стадия (SeAR, Semantic-Adaptive Overlap Relaxation) определяет, какие overlap'ы допустимы семантически и ослабляет для них repulsion, вторая стадия распределяет оставшееся давление неравномерно через load-adaptive collision strengthening (LAS) и progress-adaptive objective rebalancing (PAR). В отличие от QuaSID, который квалифицирует коллизии по Hamming severity, AdaSID адаптируется по трем осям: семантической совместимости, локальной загрузке bucket'ов и прогрессу обучения. Эксперименты на Amazon Toys и Beauty показывают средний прирост около 4.5% над сильными baseline'ами, а промышленный A/B-тест на Kuaishou подтверждает +0.98% GMV.

## 2. Контекст: от статической к адаптивной регуляции

Предшествующие collision handling методы используют фиксированные правила: uniform repulsion force, константные margin'ы, один и тот же penalty на протяжении всего обучения. Это создает конфликт с реальной динамикой token space: на ранних этапах обучения representation еще нестабильно и жесткие constraints могут вызвать collapse, на поздних этапах нужно более целенаправленное давление. Кроме того, в каталоге Kuaishou с мультимодальными item'ами часть совпадений SID отражает полезное semantic sharing (варианты одного товара, похожие видео), а часть создает критическую ambiguity.

## 3. Проблема: три измерения статичности

Авторы выделяют три типа статичности в collision handling: (1) семантическая статичность, когда все overlap'ы наказываются одинаково независимо от семантической близости item'ов; (2) пространственная статичность, когда не учитывается локальная загрузка discrete bucket'ов; (3) временная статичность, когда collision penalty фиксирован на протяжении всего training. Каждый тип статичности приводит к субоптимальному SID space.

## 4. Формальная постановка

Обучающие данные $\mathcal{D} = \{(\mathbf{x}_i^{tr}, \mathbf{x}_i^{ta})\}_{i=1}^N$ состоят из collaborative trigger-target пар, построенных через Swing алгоритм. Shared encoder $f_\theta$ отображает мультимодальный вход в $\mathbf{z}_i \in \mathbb{R}^d$. Residual quantizer с $L$ codebook'ами размера $K$ порождает SID. Overlap depth между item'ами $i$ и $j$ определяется как:

$$
o_{ij} = \sum_{l=1}^{L} \mathbb{I}[s_i^{(l)} = s_j^{(l)}]
$$

Базовый collision penalty использует hinge loss с margin, зависящим от типа overlap:

$$
\ell_{ij}^{col} = \max(0, m_{ij} - d_{ij}^c), \quad d_{ij}^c = 1 - \frac{\mathbf{z}_i^\top \mathbf{z}_j}{\|\mathbf{z}_i\|_2 \cdot \|\mathbf{z}_j\|_2}
$$

## 5. Stage One: Semantic-Adaptive Overlap Relaxation (SeAR)

Первая стадия отвечает на вопрос: нужно ли вообще отталкивать данную пару? Для каждой overlap пары вычисляется cosine similarity в continuous пространстве:

$$
\text{sim}_{ij} = \frac{\mathbf{z}_i^\top \mathbf{z}_j}{\|\mathbf{z}_i\|_2 \cdot \|\mathbf{z}_j\|_2}
$$

Depth-aware threshold vector $\boldsymbol{\eta} = [\eta_1, \eta_2, \ldots, \eta_L]$ с ограничением $\eta_1 \leq \eta_2 \leq \cdots \leq \eta_L$ задает пороги для разных глубин overlap: shallow overlap'ы ослабляются при мягких requirements, глубокие требуют сильного семантического обоснования. Relaxation gate:

$$
g_{ij} = \mathbb{I}[\text{sim}_{ij} \geq \eta_{o_{ij}}]
$$

Если $g_{ij} = 1$, overlap считается допустимым и repulsion для этой пары отключается. Инженерный смысл: модель не тратит емкость на искусственное разделение естественных семантических групп, что снижает риск over-fragmentation.

## 6. Stage Two: Adaptive Pressure Allocation

### 6.1. Load-Adaptive Collision Strengthening (LAS)

Collision signature каждой пары:

$$
\kappa_{ij} = [\mathbb{I}[s_i^{(1)} = s_j^{(1)}], \ldots, \mathbb{I}[s_i^{(L)} = s_j^{(L)}]]
$$

Локальная collision load считает, сколько других пар имеют тот же collision pattern:

$$
c_{ij} = \sum_{(u,v) \in \mathcal{P}} \mathbb{I}[\kappa_{uv} = \kappa_{ij}]
$$

Load-adaptive коэффициент $a_{ij} = g(c_{ij}; f_{\min}, f_{\max}, d_{\max}, \alpha)$ - bounded monotone scaling function, которая усиливает repulsion в перегруженных регионах. $f_{\max}$ контролирует максимальное усиление, $d_{\max}$ определяет effective load range, $\alpha$ управляет крутизной роста.

### 6.2. Progress-Adaptive Objective Rebalancing (PAR)

Нормализованный прогресс обучения:

$$
\tau = \text{clip}\left(\frac{t - T_{\text{start}}}{T_{\text{end}} - T_{\text{start}}}, 0, 1\right)
$$

Веса objectives меняются по ходу обучения:

$$
\lambda_{\text{col}}(\tau) = 1 - (1 - \lambda_{\text{col}}^{\min}) \cdot \tau, \quad \lambda_{\text{cf}}(\tau) = \lambda_{\text{cf}}^{\max} \cdot \tau
$$

Collision regulation подчеркивается рано и убывает к non-zero floor; collaborative alignment weight нарастает. Логика: на ранних этапах важнее развести перегруженные bucket'и, на поздних - усилить collaborative alignment, когда representation уже стабилизировался.

## 7. Общий training objective

Adaptive collision term:

$$
\mathcal{L}_{\text{col}}^{\text{ada}} = \sum_{(i,j) \in \mathcal{P}} a_{ij} \cdot (1 - g_{ij}) \cdot \ell_{ij}^{col}
$$

Полный objective:

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{rq}} + \lambda_{\text{col}}(\tau) \cdot \mathcal{L}_{\text{col}}^{\text{ada}} + \lambda_{\text{cf}}(\tau) \cdot \mathcal{L}_{\text{cf}}
$$

Все компоненты обучаются end-to-end через backpropagation с STE. При inference adaptive механизмы не добавляют extra complexity: используется стандартный SID pipeline (encoder + residual quantizer).

### 7.1. Псевдокод AdaSID

```
for batch of trigger-target item pairs:
    z = shared_multimodal_encoder(items)
    sid = residual_quantizer(z)  # L codebooks, STE

    collision_pairs = find_pairs_with_overlap(sid)
    adaptive_collision_loss = 0

    for i, j in collision_pairs:
        overlap_depth = count_equal_codes(sid[i], sid[j])
        sim_ij = cosine(z[i], z[j])

        # SeAR: semantically justified overlap is allowed
        relax = sim_ij >= eta[overlap_depth]
        if relax:
            continue

        # LAS: overloaded collision signatures get stronger pressure
        signature = equal_code_pattern(sid[i], sid[j])
        load = count_pairs_with_signature(signature)
        pressure = bounded_monotone_scale(load, f_min, f_max, d_max, alpha)

        collision_margin = margin_by_overlap(overlap_depth)
        adaptive_collision_loss += pressure * hinge(collision_margin - cosine_distance(z[i], z[j]))

    progress = clip((step - T_start) / (T_end - T_start), 0, 1)
    lambda_col = 1 - (1 - lambda_col_min) * progress
    lambda_cf = lambda_cf_max * progress

    loss = reconstruction_loss + rq_loss + lambda_col * adaptive_collision_loss + lambda_cf * cf_loss
    update(loss)
```

## 8. Эксперименты: setup и датасеты

<div class="table-scroll">
<table>
<tr><th>Датасет</th><th>#Users</th><th>#Items</th><th>#Interactions</th><th>Density</th></tr>
<tr><td>Amazon-Toys</td><td>19,412</td><td>11,924</td><td>905,253</td><td>0.3911%</td></tr>
<tr><td>Amazon-Beauty</td><td>22,363</td><td>12,101</td><td>1,048,296</td><td>0.3874%</td></tr>
</table>
</div>

5-core фильтрация, leave-one-out split. Item features: Title, Brand, Categories, Price. Эмбеддинги: Sentence-T5-XXL (768-dim). SID: $L=3$, $K=256$, code embedding dimension 32. Downstream backbone: TIGER.

## 9. Результаты экспериментов

### 9.1. Offline performance

<div class="table-scroll">
<table>
<tr><th>Tokenizer</th><th>Toys R@3</th><th>Toys N@3</th><th>Toys R@5</th><th>Toys N@5</th></tr>
<tr><td>RQ-VAE</td><td>0.0164</td><td>0.0142</td><td>0.0197</td><td>0.0161</td></tr>
<tr><td>Improved VQGAN</td><td>0.0191</td><td>0.0164</td><td>0.0224</td><td>0.0177</td></tr>
<tr><td>Rotation Trick</td><td>0.0182</td><td>0.0157</td><td>0.0221</td><td>0.0183</td></tr>
<tr><td>RQ-KMeans</td><td>0.0193</td><td>0.0160</td><td>0.0271</td><td>0.0187</td></tr>
<tr><td>QuaSID</td><td>0.0195</td><td>0.0157</td><td>0.0273</td><td>0.0191</td></tr>
<tr><td><strong>AdaSID</strong></td><td><strong>0.0214</strong></td><td><strong>0.0175</strong></td><td><strong>0.0281</strong></td><td><strong>0.0202</strong></td></tr>
</table>
</div>

<div class="table-scroll">
<table>
<tr><th>Tokenizer</th><th>Beauty R@3</th><th>Beauty N@3</th><th>Beauty R@5</th><th>Beauty N@5</th></tr>
<tr><td>RQ-VAE</td><td>0.0161</td><td>0.0131</td><td>0.0206</td><td>0.0149</td></tr>
<tr><td>RQ-KMeans</td><td>0.0199</td><td>0.0151</td><td>0.0271</td><td>0.0184</td></tr>
<tr><td>QuaSID</td><td>0.0201</td><td>0.0155</td><td>0.0268</td><td>0.0186</td></tr>
<tr><td><strong>AdaSID</strong></td><td><strong>0.0205</strong></td><td><strong>0.0164</strong></td><td><strong>0.0275</strong></td><td><strong>0.0190</strong></td></tr>
</table>
</div>

AdaSID улучшает Toys Recall@3 на 9.6% относительно QuaSID (0.0195 до 0.0214), NDCG@3 на 11.5% (0.0157 до 0.0175). На Beauty gains скромнее: около 2.9% в среднем. Общее среднее улучшение: около 4.5% над лучшим baseline'ом.

### 9.2. Industrial validation (Kuaishou)

<div class="table-scroll">
<table>
<tr><th>Метрика</th><th>Gain</th></tr>
<tr><td>GMV</td><td>+0.98%</td></tr>
<tr><td>Orders</td><td>+0.91%</td></tr>
<tr><td>GPM</td><td>+1.16%</td></tr>
</table>
</div>

Online retrieval A/B-тест (4 дня, short-video retrieval model, десятки миллионов пользователей). Offline ranking: AUC gain +0.05 pp overall CTCVR, +0.08 pp для cold-start CVR. Промышленная валидация использует мультимодальные features из images, text, keyframes и audio transcripts через foundation model.

## 10. Ablation study

<div class="table-scroll">
<table>
<tr><th>Вариант</th><th>Toys R@3</th><th>Toys N@3</th><th>Toys R@5</th><th>Toys N@5</th></tr>
<tr><td>AdaSID (full)</td><td>0.0214</td><td>0.0175</td><td>0.0281</td><td>0.0202</td></tr>
<tr><td>w/o SeAR</td><td>0.0192</td><td>0.0153</td><td>0.0246</td><td>0.0175</td></tr>
<tr><td>w/o PAR</td><td>0.0204</td><td>0.0169</td><td>0.0252</td><td>0.0189</td></tr>
<tr><td>w/o LAS</td><td>0.0205</td><td>0.0161</td><td>0.0271</td><td>0.0188</td></tr>
</table>
</div>

На Toys удаление SeAR дает самое большое падение по всем четырем метрикам: Recall@5 падает с 0.0281 до 0.0246 (-12.5%). Это подтверждает, что semantic relaxation критична: без нее модель переотталкивает допустимые overlap'ы. PAR наиболее важен на Beauty. LAS дает consistent, но более mild вклад. Практический вывод: collision handling должен быть и semantic-aware, и training-aware.

## 11. SID-space diagnostics

Статья явно отделяет recommendation metrics от статистик token space. Figure 3 показывает scatter plots, где каждая точка - tokenizer, по осям: inverse normalized Top-1 Load (x) и normalized Minimum Perplexity (y), размер точки = Mean Perplexity, цвет = SID Entropy. Лучший tokenizer должен быть в верхнем правом углу, крупным и темным. AdaSID находится близко к upper-right frontier и конкурентен по всем закодированным измерениям одновременно.

Ключевые метрики: SID Entropy показывает разнообразие полных SID sequences. Average Perplexity показывает среднюю utilization codebook layers. Minimum Perplexity важна как worst-layer metric: один мертвый слой ограничивает всю емкость SID. Mean Top-1 Load Ratio измеряет доминирование самого популярного code в каждом слое.

## 12. Рисунки и визуализации

Figure 1 контрастирует static overlap regulation (фиксированное суждение + фиксированная обработка) с подходом AdaSID: adaptive semantic qualification определяет, нужна ли repulsion, затем adaptive pressure allocation распределяет оставшееся давление. Figure 2 показывает полный pipeline: collaborative trigger-target пары проходят shared encoder, residual quantizer, затем двухстадийную adaptive overlap regulation. Figure 4 анализирует чувствительность к гиперпараметрам: PAR создает наибольшую вариацию performance (наиболее чувствителен), LAS и SeAR показывают более плавные и стабильные тренды.

## 13. Сильные стороны

- **Три механизма соответствуют трем failure modes.** SeAR решает semantic statics, LAS - spatial/load statics, PAR - temporal/training statics.
- **Разрешает полезные overlap'ы.** SeAR не заставляет модель искусственно разводить item'ы, если shared SID семантически оправдан.
- **Pressure распределяется локально.** LAS усиливает repulsion именно в перегруженных collision signatures, а не по всему code space.
- **Training schedule осмыслен.** PAR сначала наводит порядок в collision space, затем усиливает collaborative alignment.
- **Есть online industrial validation.** Kuaishou A/B с GMV/Orders/GPM важнее одной offline таблицы.
- **SID diagnostics сильные.** Entropy, perplexity, minimum perplexity и top-1 load дают операциональную картину здоровья tokenizer-а.

## 14. Ограничения

- **Много чувствительных гиперпараметров.** $\boldsymbol{\eta}$, $f_{\min}$, $f_{\max}$, $d_{\max}$, $\alpha$, PAR schedule и веса objectives нужно перенастраивать между доменами.
- **SeAR зависит от качества continuous embeddings.** Если cosine similarity плохо отражает recommendation-equivalent items, relaxation будет разрешать плохие overlap'ы или запрещать хорошие.
- **Collision load statistics могут устаревать.** В быстро меняющемся каталоге signature load между retraining cycles перестает отражать текущую структуру.
- **Промышленный контекст трудно воспроизвести.** Kuaishou использует rich multimodal features и collaborative Swing pairs, которые доступны не всем каталогам.
- **Online details закрыты частично.** GMV gain убедителен, но traffic split, latency и serving constraints раскрыты ограниченно.
- **Inference простой, training сложнее.** На serving остается обычный SID pipeline, но training требует pair mining, load accounting и schedule management.

## 15. Практические рекомендации

Начать с baseline RQ-VAE/RQ-KMeans tokenizer и полного логирования overlap groups. Добавить SeAR первым: для каждой collision пары вычислять cosine similarity и сравнивать с depth-aware thresholds. Затем добавить LAS: вести moving statistics collision load по codebook regions. Затем PAR: начать с collision-heavy начала и постепенно усиливать collaborative alignment. Обязательно мониторить entropy, perplexity, top-1 load и head-tail Recall отдельно.

## 16. Связь с другими работами

AdaSID является прямым продолжением QuaSID (та же группа авторов, Kuaishou + UESTC). QuaSID квалифицирует коллизии по Hamming severity с фиксированными margin'ами, AdaSID делает regulation адаптивным по трем осям. CoST атакует reconstruction objective contrastive loss'ом. GLASS и SID-Coord используют SID внутри ranking/retrieval, а AdaSID улучшает сами IDs. С Meta stability paper AdaSID связан через общую тему production-ready SID infrastructure.

## 17. Итоговая оценка

AdaSID делает collision handling recommendation-aware и training-aware: разрешает полезное semantic sharing, усиливает давление в перегруженных регионах и адаптирует веса objectives по прогрессу обучения. Это один из наиболее продуманных шагов от академической tokenization к промышленной SID-инфраструктуре, подкрепленный как offline метриками, так и промышленным A/B-тестом.
