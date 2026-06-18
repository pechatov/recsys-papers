---
title: "MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation"
category: "semantic_ids_tokenization_indexing"
slug: "mmq_multimodal_mixture_of_quantization_tokenization_for_semantic_summary"
catalogId: "paper-mmq_multimodal_mixture_of_quantization_tokenization_for_semantic_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/mmq_multimodal_mixture_of_quantization_tokenization_for_semantic_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2508.15281"
---
Подробное саммари статьи:

> **Авторы:** Yi Xu, Moyu Zhang, Chenxuan Li, Zhihao Liao, Haibo Xing, Hao Deng, Jinxin Hu, Yu Zhang, Xiaoyi Zeng, Jing Zhang.
>
> **Аффилиации:** Alibaba Group; Peking University; Beihang University; Wuhan University.
>
> **Индустрия:** Alibaba e-commerce multimodal recommendation

## 1. Коротко: о чем статья

MMQ (Multimodal Mixture-of-Quantization) предлагает двухстадийный фреймворк для построения мультимодальных semantic IDs, который одновременно улавливает межмодальную синергию и сохраняет уникальные характеристики каждой модальности. В первой стадии обучается мультимодальный токенайзер с архитектурой shared-specific экспертов и ортогональной регуляризацией кодбуков. Во второй стадии semantic IDs выравниваются с пользовательским поведением через механизм мягких индексов (soft indices), который решает проблему недифференцируемости дискретных кодов при обратном распространении градиента.

Главная идея статьи: существующие подходы к мультимодальной токенизации item'ов работают либо в режиме Modality-Aligned (MA), где модальности сливаются в единое представление до квантования, либо в режиме Modality-Separated (MS), где каждая модальность квантуется отдельно. MA теряет модально-специфичные детали, MS теряет межмодальную синергию. MMQ объединяет преимущества обоих подходов через мультиэкспертную архитектуру, а затем адаптирует полученные коды к реальным паттернам пользовательского поведения.

В онлайн A/B тесте на крупной e-commerce платформе в Юго-Восточной Азии (30 дней, 10% трафика) MMQ показал +0.90% рекламной выручки, +4.33% CVR и +3.52% заказов.

## 2. Контекст: зачем нужны мультимодальные semantic IDs

В крупных e-commerce каталогах каждый item обладает несколькими модальностями: текстовое описание (название, категория, бренд), изображение товара, иногда видео или структурированные атрибуты. Классический подход -- присвоение каждому item'у уникального ItemID и обучение embedding'а через collaborative filtering -- сталкивается с двумя фундаментальными проблемами. Во-первых, в динамичных каталогах с высокой оборачиваемостью товаров и сдвигающейся популярностью статические ID-based embedding'и быстро устаревают. Во-вторых, для long-tail item'ов с малым числом взаимодействий embedding'и получаются плохо обобщенными.

Semantic IDs, построенные из контента item'а, предлагают альтернативу: похожие по содержанию товары получают близкие дискретные коды, что позволяет модели обобщать знания между item'ами. Однако для мультимодальных item'ов возникает дилемма. Если слить все модальности в один вектор до квантования (MA-подход), теряются тонкие различия, которые видны только в одной модальности -- например, визуальный стиль одежды, который не описан в тексте. Если квантовать каждую модальность отдельно (MS-подход), теряется информация о том, как модальности дополняют друг друга -- например, текст "пляжные сандалии" и изображение с тропическим фоном совместно сигнализируют о vacation-стиле.

Вторая проблема semantic IDs -- разрыв между семантическим пространством контента и пространством пользовательского поведения. Два товара могут быть семантически похожи (одна категория, похожие описания), но вызывать совершенно разные паттерны взаимодействия. Без выравнивания с поведенческими данными semantic IDs остаются в лучшем случае хорошим content representation, но не оптимальным recommendation representation.

## 3. Проблема: что именно решает MMQ

MMQ формулирует три конкретные проблемы. Первая: существующие MA-подходы (такие как MA-RQ-VAE, MA-OPQ) вынуждены сжимать мультимодальный вектор в единое квантованное представление, что при большом масштабе промышленных каталогов усложняет задачу квантования и приводит к потере модально-специфичных нюансов. Вторая: существующие MS-подходы (MS-RQ-VAE, MS-OPQ) квантуют модальности независимо и упускают cross-modal synergy. Третья: даже если токенайзер построен хорошо, дискретные коды (semantic IDs) не обязательно оптимальны для downstream recommendation задачи, потому что обучаются на контентной реконструкции, а не на пользовательском поведении.

## 4. Метод: стадия 1 -- мультимодальный shared-specific токенайзер

### 4.1. Входные данные

Для каждого item из pretrained энкодеров извлекаются текстовый embedding $\mathbf{e}_t$ (через Qwen3-Embedding 7B, размерность 256) и визуальный embedding $\mathbf{e}_v$ (через Pailitao v8, размерность 256). Конкатенация $[\mathbf{e}_t, \mathbf{e}_v]$ подается на вход токенайзеру.

### 4.2. Мультиэкспертная архитектура

Архитектура MMQ содержит три типа экспертов. Modality-Shared эксперты ($N_s$ штук) принимают конкатенированный мультимодальный вход и порождают shared latent representations:

$$
\mathbf{z}_{s,i} = E_{s,i}([\mathbf{e}_t, \mathbf{e}_v]), \quad i \in [1, N_s]
$$

Эти эксперты улавливают кросс-модальную синергию: как текст и изображение совместно определяют пользовательский appeal товара.

Modality-Specific эксперты работают с одномодальным входом: $N_t$ текстовых экспертов $\mathbf{z}_{t,i} = E_{t,i}(\mathbf{e}_t)$ и $N_v$ визуальных экспертов $\mathbf{z}_{v,i} = E_{v,i}(\mathbf{e}_v)$. Они сохраняют уникальные атрибуты каждой модальности.

Shared эксперты получают детерминированное взвешивание, а specific эксперты -- динамическое через gating networks:

$$
\mathbf{g}_t = \mathrm{softmax}(\mathrm{MLP}_t(\mathbf{e}_t) + \mathbf{b}_t), \quad \mathbf{g}_v = \mathrm{softmax}(\mathrm{MLP}_v(\mathbf{e}_v) + \mathbf{b}_v)
$$

Финальное latent representation:

$$
\mathbf{z} = \sum_{i=1}^{N_s} \mathbf{z}_{s,i} + \sum_{i=1}^{N_v} g_{v,i} \cdot \mathbf{z}_{v,i} + \sum_{i=1}^{N_t} g_{t,i} \cdot \mathbf{z}_{t,i}
$$

В экспериментах используется $N_s = 2$ shared экспертов, $N_t = 2$ текстовых и $N_v = 2$ визуальных, что дает 6 экспертов в сумме. Каждый эксперт порождает один segment semantic ID, поэтому длина SID равна 6.

### 4.3. Cosine distance quantizer

Вместо стандартного L2-расстояния для поиска ближайшего code vector в codebook MMQ использует косинусное сходство:

$$
c_{s,i} = \arg\max_{j \in \{1,\ldots,K\}} \frac{\mathbf{z}_{s,i}^\top \mathbf{z}_{q,j}}{\|\mathbf{z}_{s,i}\| \cdot \|\mathbf{z}_{q,j}\|}
$$

Мотивация: embedding'и от разных модальных энкодеров могут иметь разные шкалы (scale), и L2-расстояние чувствительно к этим различиям. Косинусное сходство нормализует шкалу, что дает более равномерное использование кодбука.

### 4.4. Ортогональная регуляризация

Чтобы shared и specific эксперты не обучались одному и тому же, MMQ вводит ортогональную регуляризацию. Для весовых матриц экспертов $\mathbf{W}_i \in \mathbb{R}^{d \times d}$ каждая матрица разворачивается в вектор $\mathbf{v}_i \in \mathbb{R}^{d^2}$, из них формируется матрица $\mathbf{V} = [\mathbf{v}_1, \ldots, \mathbf{v}_N]$, строки нормализуются:

$$
\mathcal{L}_{\mathrm{ortho\_shared}} = \|\mathbf{V}_{\mathrm{norm}} \cdot \mathbf{V}_{\mathrm{norm}}^\top - \mathbf{I}\|_F^2
$$

Аналогичная loss считается для specific экспертов. Суммарная:

$$
\mathcal{L}_{\mathrm{ortho}} = \mathcal{L}_{\mathrm{ortho\_shared}} + \sum_{i}^{m} \mathcal{L}_{\mathrm{ortho\_specific}}^{(i)}
$$

### 4.5. Функции потерь стадии 1

Multimodal reconstruction loss измеряет верность восстановления исходных embedding'ов из квантованного представления:

$$
\mathcal{L}_{\mathrm{recon}} = \|\mathbf{e} - \mathrm{decoder}(\mathbf{z} + \mathrm{sg}(\mathbf{z}_q - \mathbf{z}))\|^2
$$

Auxiliary modality-specific reconstruction loss помогает specific энкодерам учить дискриминативные признаки каждой модальности:

$$
\mathcal{L}_{\mathrm{aux}} = \|\mathbf{e}_t - \mathrm{dec}_t(\cdots)\|^2 + \|\mathbf{e}_v - \mathrm{dec}_v(\cdots)\|^2
$$

Итоговая loss стадии 1:

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{\mathrm{recon}} + \beta \cdot \mathcal{L}_{\mathrm{aux}} + \gamma \cdot \mathcal{L}_{\mathrm{ortho}}
$$

Гиперпараметры по умолчанию: $\alpha = 12$, $\beta = 10$, $\gamma = 0.005$.

## 5. Метод: стадия 2 -- behavior-aware fine-tuning

Ключевая проблема второй стадии: дискретные индексы кодбука недифференцируемы, что разрывает поток градиентов между downstream recommendation моделью и токенайзером. MMQ решает это через механизм soft indices, вдохновленный Index Backpropagation Quantization (IBQ).

Для каждого latent representation $\mathbf{z}_i$ вычисляется косинусное сходство со всеми $K$ codeword'ами:

$$
\mathbf{p}_i = [\cos(\mathbf{z}_i, \mathbf{z}_{q,1}), \ldots, \cos(\mathbf{z}_i, \mathbf{z}_{q,K})] \in \mathbb{R}^K
$$

Из этого вектора формируются soft и hard индексы:

$$
\mathrm{soft\_ind} = \mathrm{softmax}(\mathbf{p}_i / \tau), \quad \mathrm{hard\_ind} = \arg\max_j \cos(\mathbf{z}_i, \mathbf{z}_{q,j})
$$

Через Straight-Through Estimator (STE):

$$
\mathrm{ind} = \mathrm{soft\_ind} + \mathrm{sg}(\mathrm{hard\_ind} - \mathrm{soft\_ind})
$$

В forward pass используется discrete hard_ind для lookup, а при backpropagation градиенты текут через дифференцируемый soft_ind. Это позволяет downstream recommendation loss обновлять параметры токенайзера.

Joint optimization objective стадии 2:

$$
\mathcal{L}_{\mathrm{finetune}} = \mathcal{L}_{\mathrm{downstream}} + \alpha' \cdot \mathcal{L}_{\mathrm{recon}} + \beta' \cdot \mathcal{L}_{\mathrm{aux}}
$$

Значения по умолчанию: $\alpha' = 0.5$, $\beta' = 0.5$.

### 5.1. Псевдокод MMQ

MMQ важно воспроизводить как двухстадийный pipeline: сначала tokenizer учит мультимодальную shared-specific структуру, затем downstream behavior loss донастраивает индексы через soft indices.

```
stage 1: multimodal tokenizer pretraining
for item in catalog:
    e_text = text_encoder(item.title_category_brand)
    e_vision = vision_encoder(item.image)
    z_shared = [shared_expert_i(concat(e_text, e_vision)) for i in 1..N_s]
    z_text = weighted_sum(text_experts(e_text), gate_text(e_text))
    z_vision = weighted_sum(vision_experts(e_vision), gate_vision(e_vision))
    z_parts = z_shared + [z_text, z_vision]

    sid = []
    for z_part in z_parts:
        code = argmax_cosine(z_part, codebook)
        sid.append(code)
    reconstructed = decoder(quantized(z_parts, sid))
    loss = alpha * recon_loss(reconstructed, [e_text, e_vision])
         + beta * auxiliary_modality_recon_loss()
         + gamma * orthogonal_regularization(experts)
    update(tokenizer, codebooks, decoders)

stage 2: behavior-aware fine-tuning
for interaction_batch in logs:
    soft_indices = softmax(cosine_to_all_codewords(z_parts) / tau)
    hard_indices = argmax(cosine_to_all_codewords(z_parts))
    indices = soft_indices + stopgrad(hard_indices - soft_indices)
    downstream_features = lookup_or_mix_code_embeddings(indices)
    loss = downstream_recommendation_loss(downstream_features, labels)
         + alpha_prime * recon_loss
         + beta_prime * auxiliary_modality_recon_loss
    update(tokenizer, codebooks, downstream_model)
```

## 6. Экспериментальная установка

### 6.1. Датасеты

Промышленный датасет: данные крупной e-commerce advertising платформы в Юго-Восточной Азии (октябрь 2024 -- май 2025). 30M пользователей, 40M рекламных объявлений. Пользовательские последовательности поведения (impressions, clicks, conversions) средней длиной 128.

Публичный датасет: Amazon Product Reviews "Beauty" (май 1996 -- сентябрь 2014). Для generative retrieval: 5-core фильтр, максимальная длина последовательности 20. Для discriminative ranking: ratings > 3 = positive, ≤ 3 = negative, хронологический split 90/10.

### 6.2. Baselines

Сравниваются три метода квантования в двух парадигмах (MA и MS): RQ-VAE (residual vector quantization с обучаемыми кодбуками), RQ-Kmeans (residual quantization через K-means кластеризацию), OPQ (Optimized Product Quantization). Дополнительно PPNet как baseline на чистых ItemID для discriminative ranking.

### 6.3. Реализация

Размер кодбука: 3,000 (промышленный датасет), 100 (публичный). Длина semantic ID: 6 (2 shared + 2 text + 2 vision). Текстовый энкодер: Qwen3-Embedding 7B (dim 256). Визуальный энкодер: Pailitao v8 (dim 256). Backbone для generative retrieval: HeterRec. Backbone для discriminative ranking: PPNet.

## 7. Основные результаты

### 7.1. Discriminative ranking -- промышленный датасет

<div class="table-scroll">
<table>
<tr><th>Метод</th><th>$\mathcal{L}_{\mathrm{recon}}\downarrow$</th><th>Entropy$\uparrow$</th><th>Util.$\uparrow$</th><th>AUC$\uparrow$</th><th>GAUC$\uparrow$</th></tr>
<tr><td>PPNet (ItemID)</td><td>--</td><td>--</td><td>--</td><td>0.7144</td><td>0.6034</td></tr>
<tr><td>MA-RQ-VAE</td><td>0.6165</td><td>7.1178</td><td>0.79</td><td>0.7169</td><td>0.6064</td></tr>
<tr><td>MA-RQ-kmeans</td><td>0.6759</td><td>7.7933</td><td>1.00</td><td>0.7170</td><td>0.6056</td></tr>
<tr><td>MA-OPQ</td><td>0.6730</td><td>5.3092</td><td>0.33</td><td>0.7149</td><td>0.6046</td></tr>
<tr><td>MS-RQ-VAE</td><td>0.5547</td><td>6.3389</td><td>0.99</td><td>0.7171</td><td>0.6067</td></tr>
<tr><td>MS-OPQ</td><td>0.5678</td><td>5.2789</td><td>0.30</td><td>0.7177</td><td>0.6071</td></tr>
<tr><td><strong>MMQ</strong></td><td><strong>0.5529</strong></td><td><strong>7.9560</strong></td><td><strong>1.00</strong></td><td><strong>0.7184</strong></td><td><strong>0.6081</strong></td></tr>
</table>
</div>

### 7.2. Generative retrieval -- промышленный датасет

<div class="table-scroll">
<table>
<tr><th>Метод</th><th>R@5</th><th>R@10</th><th>N@5</th><th>N@10</th></tr>
<tr><td>MA-RQ-VAE</td><td>0.0570</td><td>0.0754</td><td>0.1362</td><td>--</td></tr>
<tr><td>MA-RQ-kmeans</td><td>0.0488</td><td>0.0522</td><td>0.1339</td><td>0.1388</td></tr>
<tr><td>MS-RQ-VAE</td><td>0.0779</td><td>0.0988</td><td>0.1897</td><td>0.2191</td></tr>
<tr><td>MS-OPQ</td><td>0.0757</td><td>0.0927</td><td>0.1866</td><td>0.2105</td></tr>
<tr><td><strong>MMQ</strong></td><td><strong>0.1034</strong></td><td><strong>0.1192</strong></td><td><strong>0.2661</strong></td><td><strong>0.2883</strong></td></tr>
<tr><td>Улучшение vs лучший baseline</td><td>+32.73%</td><td>+20.64%</td><td>+40.27%</td><td>+31.58%</td></tr>
</table>
</div>

Наиболее впечатляющий результат: на generative retrieval MMQ превосходит лучший baseline (MS-RQ-VAE) на 32.73% по Recall@5 и на 40.27% по NDCG@5. Это показывает, что combined shared-specific архитектура особенно критична для генеративного подхода, где модель должна точно генерировать правильную последовательность токенов.

### 7.3. Generative retrieval -- Amazon Beauty

<div class="table-scroll">
<table>
<tr><th>Метод</th><th>R@5</th><th>R@10</th><th>N@5</th><th>N@10</th></tr>
<tr><td>MA-RQ-VAE</td><td>0.0343</td><td>0.0477</td><td>0.0236</td><td>0.0292</td></tr>
<tr><td>MS-RQ-kmeans</td><td>0.0413</td><td>0.0549</td><td>0.0293</td><td>0.0349</td></tr>
<tr><td><strong>MMQ</strong></td><td><strong>0.0455</strong></td><td><strong>0.0675</strong></td><td><strong>0.0296</strong></td><td><strong>0.0384</strong></td></tr>
<tr><td>Улучшение</td><td>+10.16%</td><td>+22.95%</td><td>+1.02%</td><td>+10.02%</td></tr>
</table>
</div>

## 8. Ablation study

<div class="table-scroll">
<table>
<tr><th>Вариант</th><th>$\mathcal{L}_{\mathrm{recon}}$</th><th>Entropy</th><th>Util.</th><th>R@5</th><th>N@5</th></tr>
<tr><td><strong>MMQ (full)</strong></td><td><strong>0.5529</strong></td><td><strong>7.9560</strong></td><td><strong>1.00</strong></td><td><strong>0.1034</strong></td><td><strong>0.2661</strong></td></tr>
<tr><td>w/o Cosine Quantizer</td><td>0.5817</td><td>4.9524</td><td>0.59</td><td>0.0786</td><td>0.1936</td></tr>
<tr><td>w/o Auxiliary Recon Loss</td><td>0.5879</td><td>4.3854</td><td>0.51</td><td>0.0684</td><td>0.1695</td></tr>
<tr><td>w/o Orthogonal Reg.</td><td>0.5816</td><td>7.5640</td><td>0.68</td><td>0.0583</td><td>0.1111</td></tr>
<tr><td>w/o Behavior Fine-tuning</td><td>0.5726</td><td>7.6927</td><td>0.99</td><td>--</td><td>--</td></tr>
</table>
</div>

Наиболее критичный компонент -- ортогональная регуляризация: без нее NDCG@5 падает с 0.2661 до 0.1111 (более чем двукратное снижение), а codebook utilization падает до 0.68. Это означает, что без ортогонального ограничения эксперты обучают избыточную, перекрывающуюся информацию, что приводит к коллапсу значительной части кодбука.

Удаление auxiliary modality-specific reconstruction loss снижает utilization до 0.51 и entropy до 4.3854, подтверждая, что без явного надзора за каждой модальностью specific эксперты не обучаются дискриминативным признакам.

Замена cosine quantizer на L2-based снижает utilization до 0.59 -- подтверждение, что нормализация шкалы важна для равномерного использования кодбука при работе с разными модальностями.

## 9. Анализ по популярности item'ов

Item'ы стратифицированы на Popular (верхние 25% по числу impressions) и Long-tail (нижние 25%). Результаты показывают, что все подходы на semantic IDs значительно улучшают AUC на long-tail item'ах по сравнению с традиционным ItemID baseline (PPNet). MMQ дает наибольший gain именно в long-tail сегменте -- как в MA, так и в MS парадигмах. Это подтверждает, что semantic IDs действительно решают проблему data sparsity для редких item'ов, а мультимодальная shared-specific архитектура MMQ особенно эффективна для item'ов с ограниченными взаимодействиями.

## 10. Масштабирование по длине SID

Авторы тестируют длину semantic ID в диапазоне [6, 9, 12, 15, 18]. NDCG@5 и NDCG@10 последовательно растут с увеличением длины. Reconstruction loss и entropy распределения токенов остаются стабильными во всех настройках. Это демонстрирует, что MMQ эффективно масштабируется: более длинные последовательности позволяют кодировать более богатую семантику без деградации стабильности обучения.

## 11. Вклад shared экспертов

Сравнение MMQ (2 shared + 4 specific экспертов) с вариантами Specific-Only при [4, 6, 8, 10] экспертах показывает, что даже Specific-Only с 10 экспертами не достигает качества MMQ с 2+4 экспертами. Без shared экспертов, отвечающих за общие кросс-модальные знания, specific эксперты обучают избыточную перекрывающуюся информацию, что приводит к неэффективному использованию параметров. Это ключевой аргумент в пользу архитектуры с явным разделением на shared и specific компоненты.

## 12. Онлайн A/B тест

MMQ был развернут в generative retrieval системе крупной e-commerce платформы. Тест длился 30 дней (6 июля -- 6 августа 2025), 10% случайного пользовательского трафика направлялось в экспериментальную группу. Control использовал существующую ItemID-based систему.

<div class="table-scroll">
<table>
<tr><th>Метрика</th><th>Изменение</th></tr>
<tr><td>Рекламная выручка</td><td><strong>+0.90%</strong></td></tr>
<tr><td>CVR (Conversion Rate)</td><td><strong>+4.33%</strong></td></tr>
<tr><td>Заказы</td><td><strong>+3.52%</strong></td></tr>
</table>
</div>

CVR +4.33% -- это значительный результат для промышленной рекомендательной системы. Важно отметить, что control был не weak baseline, а действующая production система на ItemID. Это подтверждает, что переход на мультимодальные semantic IDs с behavior-aware fine-tuning дает измеримый business value.

## 13. Сильные стороны

Архитектурное решение MMQ с shared-specific экспертами элегантно решает дилемму MA vs MS. Вместо выбора между слиянием и разделением модальностей, MMQ явно моделирует оба аспекта. Ортогональная регуляризация делает это разделение не только архитектурным, но и функциональным: shared и specific эксперты действительно обучают разную информацию.

Behavior-aware fine-tuning через soft indices -- практически важный вклад. Проблема недифференцируемости дискретных кодов давно известна в литературе по VQ, но MMQ показывает конкретный способ ее решения в контексте рекомендательных систем, где downstream task -- не реконструкция, а предсказание поведения.

Наличие онлайн A/B теста на 10% трафика крупной платформы в течение 30 дней -- редкая и ценная валидация для статей о semantic IDs. Большинство работ в этой области ограничиваются офлайн экспериментами.

Стратификация по популярности показывает, что MMQ особенно помогает long-tail item'ам -- именно тому сегменту, для которого semantic IDs наиболее теоретически обоснованы.

## 14. Ограничения и спорные моменты

Промышленный датасет внутренний и не воспроизводим. Публичный датасет (Amazon Beauty) значительно меньше по масштабу и может не отражать все свойства промышленной системы. Двухстадийное обучение добавляет pipeline complexity: нужно сначала обучить токенайзер, затем fine-tune'ить его с downstream loss. В production это означает дополнительные этапы в ML pipeline, дополнительные checkpoints для версионирования и потенциально более длинный цикл обновления.

Авторы сами отмечают, что метод может требовать значительного объема мультимодальных данных для полного раскрытия потенциала модально-специфичных экспертов. В доменах с бедной визуальной информацией (например, финансовые продукты) преимущество MMQ может быть ограниченным.

Behavior-aware fine-tuning зависит от конкретной downstream задачи, что может ограничивать переносимость semantic IDs между разными задачами (retrieval vs ranking vs CTR prediction). Если один набор SIDs используется в нескольких downstream моделях, непонятно, под какую задачу оптимизировать fine-tuning.

Абсолютные improvement'ы по AUC/GAUC на промышленном датасете скромные (AUC +0.04%, GAUC +0.07%). Основной выигрыш виден на generative retrieval метриках и в онлайн тесте. Для discriminative ranking разница между методами квантования менее выражена.

## 15. Практические выводы

Для мультимодальных каталогов с текстом и изображениями shared-specific архитектура предпочтительнее как чистого MA, так и чистого MS подхода. Ортогональная регуляризация -- не опциональное украшение, а критический компонент: без нее происходит collapse экспертов и резкое падение downstream метрик.

Cosine distance quantizer предпочтительнее L2 при работе с embedding'ами из разных pretrained энкодеров. Auxiliary modality-specific reconstruction loss нужен для направления specific экспертов -- без него they учат шум.

Behavior-aware fine-tuning через soft indices -- портативный компонент: авторы показывают, что его можно интегрировать и в обычный RQ-VAE baseline, получая consistent gains. Это делает его полезным building block для любого semantic ID pipeline.

## 16. Связь с другими работами

MMQ развивает линию работ о semantic ID tokenization (TIGER, CoST, LETTER, TokenRec) в мультимодальном направлении. С CoST его объединяет идея retrieval-aligned tokenization, но MMQ решает более широкую задачу: не только объектив квантования, но и архитектуру мультимодального энкодера и downstream alignment. С EAGER его связывает идея двухпотоковой архитектуры (semantic + behavioral), но MMQ реализует это через стадийное обучение, а не через параллельные потоки. С FusID его связывает мультимодальная fusion до квантования, но MMQ добавляет specific экспертов для сохранения модально-уникальной информации.

## 17. Итоговая оценка

MMQ -- сильная промышленная работа, которая предлагает комплексное решение для мультимодальной semantic tokenization. Ее главная ценность -- в демонстрации того, что простого квантования мультимодальных embedding'ов недостаточно: нужна архитектура, которая явно моделирует shared и specific информацию, контролирует redundancy через ортогональность и выравнивает коды с поведенческими данными. Онлайн результаты (+4.33% CVR) подтверждают практическую значимость подхода. Ограничения связаны с воспроизводимостью (внутренние данные) и сложностью pipeline, но для промышленных систем с мультимодальными каталогами MMQ предоставляет убедительный blueprint.

## 18. Что читать после MMQ

**[TIGER](https://arxiv.org/abs/2305.05065)** -- базовый framework semantic IDs для generative recommendation, на котором строится MMQ.

**[CoST](https://arxiv.org/abs/2404.14774)** -- contrastive quantization для semantic IDs, решающий другой аспект той же проблемы (объектив квантования).

**[EAGER](https://arxiv.org/abs/2406.14017)** -- двухпотоковая semantic-behavioral архитектура, предшественник идеи behavior-aware alignment.

**[FusID](https://arxiv.org/abs/2601.08764)** -- мультимодальная fusion для музыкальных semantic IDs, альтернативный подход к слиянию модальностей.
