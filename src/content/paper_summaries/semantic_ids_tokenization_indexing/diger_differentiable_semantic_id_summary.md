---
title: "DIGER: Differentiable Semantic ID for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "diger_differentiable_semantic_id_summary"
catalogId: "paper-diger_differentiable_semantic_id_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/diger_differentiable_semantic_id_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2601.19711"
---
Подробное саммари статьи:

> **Авторы:** Junchen Fu, Xuri Ge, Alexandros Karatzoglou, Ioannis Arapakis, Suzan Verberne, Joemon M. Jose, Zhaochun Ren.
>
> **Аффилиации:** University of Glasgow; Shandong University; Amazon; Telefonica Scientific Research; Leiden University.

## 1. Коротко: о чем статья

DIGER предлагает сделать semantic IDs (SID) дифференцируемой частью генеративного рекомендателя, а не замороженным артефактом отдельного tokenizer'а. В стандартном pipeline типа TIGER сначала обучается RQ-VAE, который назначает каждому item'у дискретную последовательность токенов, а затем Transformer учится генерировать эти токены по истории пользователя. Проблема в том, что recommendation loss не может повлиять на codebook и item-to-code mapping, потому что semantic IDs заморожены. DIGER показывает, что наивное решение через straight-through estimator (STE) приводит к catastrophic codebook collapse: несколько кодов захватывают подавляющее большинство item'ов, остальные коды остаются мертвыми. Решение DIGER называется DRIL (Differentiable Semantic ID with Exploratory Learning): в assignment logits добавляется Gumbel noise, а backward pass использует soft probability distribution, позволяя нескольким кодам получать градиентный сигнал одновременно. Две стратегии uncertainty decay (SDUD и FrqUD) постепенно снижают шум, переводя обучение из режима exploration в exploitation.

## 2. Контекст: generative recommendation и objective mismatch

В генеративных рекомендательных системах item'ы представляются дискретными semantic IDs, а задача next-item recommendation формулируется как autoregressive generation последовательности токенов. Это требует двух обученных компонентов: tokenizer (обычно RQ-VAE), который строит mapping item → token tuple, и generator (Transformer), который предсказывает следующий token tuple по истории пользователя.

Стандартная двухстадийная схема обучает эти компоненты раздельно. Tokenizer оптимизирует reconstruction/content objective, а generator оптимизирует recommendation objective. Между ними существует fundamental mismatch: semantic IDs могут быть хорошими с точки зрения восстановления item embedding'а, но неудобными для autoregressive generation и ranking. Поскольку SID заморожены после первой стадии, recommendation loss не распространяет градиенты обратно в codebook.

DIGER формализует это наблюдение теоретически. Теорема A.1 показывает, что двухстадийное обучение оптимизирует recommendation loss только в restricted subset пространства параметров tokenizer'а: $J_{\text{2st}}^* = \inf_{\phi \in A} g(\phi)$, тогда как joint optimization имеет доступ к полному пространству: $J_{\text{e2e}} = \inf_{\phi \in \Phi} g(\phi)$. Поскольку $A \subseteq \Phi$, всегда выполняется $J_{\text{e2e}} \le J_{\text{2st}}^*$, а теорема A.2 показывает, что разрыв может быть произвольно большим.

## 3. Проблема: почему наивная дифференцируемость ведет к collapse

Очевидное решение objective mismatch -- сделать SID дифференцируемыми через straight-through estimator. Forward pass остается дискретным (argmax выбирает ближайший code vector), а backward pass передает градиенты мимо недифференцируемого argmax. Однако эксперименты DIGER показывают, что STE приводит к catastrophic performance degradation.

На B-Shop STE дает Recall@10 = 0.0134 и NDCG@10 = 0.0067, что в 5 раз хуже, чем даже двухстадийный baseline (Recall@10 = 0.0610, NDCG@10 = 0.0331). Причина -- codebook collapse: ранние deterministic assignment'ы быстро концентрируют вероятность на нескольких codebook entries, entropy распределения кодов падает, большинство кодов перестают использоваться, и дальнейшие градиенты уже не могут восстановить баланс. Визуализации code usage (Figure 7) показывают, что STE имеет severe imbalance, особенно в глубоких quantization layers.

Дополнительная проблема STE -- нестабильность SID drift. Figure 6 показывает, что STE демонстрирует резкие, непредсказуемые изменения SID assignments между эпохами, тогда как DIGER эволюционирует более плавно.

## 4. Метод DIGER: DRIL и uncertainty decay

### 4.1. Preliminaries: semantic IDs и autoregressive generation

Каждый item $v \in V$ кодируется semantic ID длины $m$:

$$
z_v = (c_{v,1}, \ldots, c_{v,m}), \quad c_{v,j} \in \{1, \ldots, K\}
$$

История пользователя представляется конкатенацией SID'ов:

$$
x_u^{(t)} = z_{v_1} \oplus \ldots \oplus z_{v_t}
$$

Autoregressive generation моделирует:

$$
p_\theta(z_{v_{t+1}} \mid x_u^{(t)}) = \prod_{j=1}^{m} p_\theta(c_{v_{t+1},j} \mid x_u^{(t)}, c_{v_{t+1},

Основной training loss -- negative log-likelihood: \[\mathcal{L}_{\text{gen}} = -\sum_u \sum_t \log p_\theta(z_{v_{t+1}} \mid x_u^{(t)})
$$

 4.2. DRIL: Gumbel noise и soft update Для каждого item'а $v$ на quantization level $j$ считаются similarity logits к codebook entries: 

$$
\ell_{v,j,i} = \text{sim}(r_{v,j}, e_i), \quad i \in \{1, \ldots, K\}
$$

 Вместо детерминированного argmax, DIGER добавляет Gumbel noise и строит Gumbel-Softmax distribution: 

$$
\tilde{y}_{v,j,i} = \frac{\exp((\ell_{v,j,i} + g_{v,j,i}) / \tau)}{\sum_{k=1}^{K} \exp((\ell_{v,j,k} + g_{v,j,k}) / \tau)}
$$

 где $g_{v,j,i} \sim \text{Gumbel}(0,1)$ -- i.i.d. Gumbel noise, $\tau$ -- temperature. Для forward pass и indexing используется hard assignment: 

$$
c_{v,j} = \arg\max_i (\ell_{v,j,i} + g_{v,j,i})
$$

 Но для backward pass используется soft weighted sum: 

$$
\bar{e}_{v,j} = \sum_{i=1}^{K} \tilde{y}_{v,j,i} \cdot e_i
$$

 Это ключевое отличие от STE: несколько candidate codes получают gradient signal, а не только победивший codeword. Gumbel noise обеспечивает stochastic exploration, особенно на ранних эпохах обучения. 4.3. SDUD: Standard Deviation Uncertainty Decay SDUD связывает масштаб шума $\sigma$ с recommendation loss через auxiliary objective: 

$$
\mathcal{L}_\sigma = \frac{\mathcal{L}_{\text{gen}}}{2(\sigma + \lambda)^2} + \log(\sigma + \lambda)
$$

 где $\sigma \ge 0$ -- обучаемый параметр, $\lambda > 0$ -- гиперпараметр. Оптимальное значение в замкнутой форме: 

$$
\sigma^* = \max\{0, \sqrt{\mathcal{L}_{\text{gen}}} - \lambda\}
$$

 По мере уменьшения $\mathcal{L}_{\text{gen}}$ в процессе обучения $\sigma^*$ сокращается, автоматически переводя систему из exploration в exploitation. Когда $\sqrt{\mathcal{L}_{\text{gen}}} \approx \lambda$, шум исчезает и assignment становится детерминированным. 4.4. FrqUD: Frequency-based Uncertainty Decay FrqUD использует EMA-сглаженную частоту использования кодов: 

$$
f_i^{(e)} \leftarrow \beta \cdot f_i^{(e-1)} + (1 - \beta) \cdot \hat{f}_i^{(e)}, \quad \beta \in [0, 1)
$$

 Определяется threshold для hot codes: 

$$
\gamma = r \cdot \bar{f} = r / K
$$

 Коды делятся на high-frequency (hot) и low-frequency: 

$$
I_{\text{high}}^{(e)} = \{i \mid f_i^{(e)} > \gamma\}, \quad I_{\text{low}}^{(e)} = \{1, \ldots, K\} \setminus I_{\text{high}}^{(e)}
$$

 Gumbel noise применяется только к hot codes (которые уже достаточно часто выбираются), а cold codes получают deterministic assignment (без шума), чтобы не сбивать их нестабильным exploration'ом. Это селективный подход: exploration там, где codebook уже перенасыщен, и exploitation там, где коды еще набирают usage. 4.5. Joint training objective Итоговый loss объединяет три компонента: 

$$
\mathcal{L} = \mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{vq}} + \mathcal{L}_{\text{recon}}
$$

 где $\mathcal{L}_{\text{gen}}$ -- recommendation objective (основной driver), $\mathcal{L}_{\text{vq}}$ и $\mathcal{L}_{\text{recon}}$ -- стандартные RQ-VAE losses для предотвращения distribution drift. Модель инициализируется из pretrained RQ-VAE; VQ и reconstruction losses остаются малыми по масштабу. Авторы специально не используют code-diversity loss, чтобы изолировать эффект differentiable SID. 4.6. Пошаговый алгоритм DIGER **Pretrain tokenizer.** Сначала обучается обычный RQ-VAE по content/text embeddings, чтобы получить начальный codebook и hard SID mapping без recommendation gradients. **Initialize generator.** User histories переводятся в SID sequences, T5-style encoder-decoder обучается генерировать target SID как в TIGER-like pipeline. **Включить DRIL вместо STE.** На каждом quantization level считаются logits до codewords; forward использует hard Gumbel argmax, backward использует soft Gumbel-Softmax weighted sum. **Передать recommendation loss в tokenizer.** $\mathcal{L}_{\text{gen}}$ обновляет не только generator, но и tokenizer/codebook через soft path; $\mathcal{L}_{\text{vq}}$ и $\mathcal{L}_{\text{recon}}$ удерживают representation от разрушения. **Управлять exploration.** SDUD уменьшает noise через loss-dependent $\sigma$, а FrqUD применяет noise выборочно к hot codes, чтобы бороться с early code domination. **Мониторить collapse signals.** На каждой эпохе считаются code usage entropy, dead codes, incremental/cumulative SID drift и train-inference agreement. **Финализировать hard mapping.** На inference Gumbel noise выключается, item получает deterministic SID, который можно использовать в обычном constrained decoding / prefix tree. 5. Рисунки и что в них важно <img src="../../assets/diger/framework.png" alt="DIGER framework: conventional pipeline vs differentiable SID"> Figure 1 использует метафору "brickmaker-builder": в обычном pipeline tokenizer (brickmaker) производит кирпичи (SID) не зная, как их будет использовать recommender (builder). В DIGER recommendation loss влияет на производство кирпичей. Рисунок четко показывает, где в обычной схеме происходит gradient blocking и как DIGER его устраняет. <img src="../../assets/diger/diger_vs_ste.png" alt="DIGER vs STE: validation NDCG и code balance"> Figure 2 -- диагностический рисунок на B-Shop. Верхняя панель показывает validation NDCG@10 по эпохам: DIGER стабильно растет, STE хаотичен и нестабилен. Нижняя панель показывает code balance (mean coverage over 3 codebook levels): у STE существенно больший разброс (error bars), что указывает на volatile, inconsistent code usage. Серая пунктирная линия отмечает эпоху early-stop для STE. Этот рисунок является центральным доказательством того, что code balance -- ранний diagnostic signal, а не второстепенная статистика. 6. Эксперименты 6.1. Датасеты Dataset#Users#Items#InteractionsAvg. LengthSparsity B-Shop22,36312,101198,5028.880.9993 I-Shop24,7729,922206,1538.320.9992 Yelp30,43120,033304,52410.010.9995 B-Shop содержит косметические товары, I-Shop -- музыкальные товары, Yelp -- рестораны. Content features извлекаются через LLaMA-7B из titles/descriptions. Evaluation использует leave-one-out protocol с ranking against full item set. 6.2. Implementation details Tokenizer: pretrained RQ-VAE, codebook size $K = 256$, sequence length $m = 3$ (plus один conflict code). Recommender: T5-style encoder-decoder, 6+6 layers, hidden size 128. Optimizer: AdamW, weight decay 0.05. Gumbel temperature $\tau = 2.0$. Hardware: два H100 GPU. 6.3. Сравнение с двухстадийным pipeline и STE ModelB-Shop R@10B-Shop N@10I-Shop R@10I-Shop N@10Yelp R@10Yelp N@10 Two-Stage0.06100.03310.10580.07970.04070.0213 STE0.01340.0067—0.0077—— DIGER (FrqUD)0.06830.03720.11380.0844—— DIGER (SDUD)0.06570.03610.11240.08230.04390.0227 Результаты наглядно демонстрируют три ключевых вывода. Во-первых, STE catastrophically ухудшает performance: на B-Shop Recall@10 падает с 0.0610 до 0.0134, что подтверждает codebook collapse hypothesis. Во-вторых, DIGER consistently улучшает двухстадийный baseline на всех датасетах: B-Shop N@10 растет с 0.0331 до 0.0372 (+12.4%), I-Shop N@10 с 0.0797 до 0.0844 (+5.9%), Yelp N@10 с 0.0213 до 0.0227 (+6.6%). В-третьих, FrqUD слегка превосходит SDUD на B-Shop и I-Shop. 6.4. Сравнение с SOTA baselines ModelB-Shop R@10B-Shop N@10I-Shop R@10I-Shop N@10Yelp R@10Yelp N@10 SASRec0.05880.03130.09470.06900.02960.0152 P5-CID0.05970.03470.09870.07510.03470.0181 TIGER0.06100.03310.10580.07970.04070.0213 LETTER0.06720.03640.11220.08310.04260.0231 ETEGRec0.06150.03350.11060.08100.04150.0214 **DIGER****0.0683*****0.0372*****0.1138*****0.0844***0.04320.0227 DIGER достигает SOTA на B-Shop и I-Shop по обеим метрикам (p-value < 0.05). На Yelp DIGER показывает лучший R@10 (0.0432), но N@10 (0.0227) немного уступает LETTER (0.0231). Авторы объясняют это тем, что LETTER использует collaborative signals через SASRec-derived embeddings, тогда как DIGER работает только с text features. ETEGRec -- closest competitor по joint optimization approach, но требует примерно вдвое больше training time и все равно уступает DIGER. 7. Ablation study 7.1. Component ablation (B-Shop) ModelR@5R@10N@5N@10 Two-Stage0.03950.06100.02620.0331 Naive E2E (STE)0.00760.01340.00480.0067 DIGER w/ UD0.04400.06830.02940.0372 DIGER w/o UD0.04240.06790.02830.0365 w/o Gumbel Noise0.01680.02830.01040.0141 w/o soft update0.04220.06500.02810.0354 w/ Gumbel tau annealing0.04190.06540.02730.0348 w/ Gaussian Noise0.03890.06200.02530.0327 Ablation показывает иерархию компонентов. Удаление Gumbel noise -- самое разрушительное изменение: N@10 падает с 0.0365 до 0.0141, что близко к STE collapse. Замена Gumbel noise на Gaussian дает N@10 = 0.0327, что заметно хуже (Gumbel distribution лучше подходит для дискретного categorical assignment). Удаление uncertainty decay снижает N@10 с 0.0372 до 0.0365 -- improvement скромный, но consistent. Temperature annealing дает результат на уровне "без UD", подтверждая, что adaptive decay полезнее фиксированного schedule. 7.2. Codebook capacity и длина SID (B-Shop) KmR@10N@10 12830.06550.0352 25630.06830.0372 51230.06600.0359 2562—0.0301 25630.06830.0372 25640.06730.0373 Performance peaking при $K = 256$: меньший codebook (128) недостаточно выразителен, а больший (512) затрудняет exploration. По длине SID: $m = 2$ дает существенное падение (N@10 = 0.0301), $m = 4$ почти не улучшает относительно $m = 3$ (N@10 = 0.0373 vs 0.0372). Оптимум $K = 256, m = 3$ обеспечивает лучший баланс expressivity и stable training. 8. Динамика SID и стабильность Figure 6 анализирует три аспекта SID dynamics на B-Shop для четырех методов (STE, DIGER w/o UD, DIGER FrqUD, DIGER SDUD). Incremental SID drift (доля item'ов, меняющих SID между эпохами): STE показывает резкие, непредсказуемые скачки drift'а, особенно в начале обучения. DIGER варианты эволюционируют более плавно. Cumulative SID drift (доля item'ов, чей SID отличается от начального): DIGER с uncertainty decay показывает больший cumulative drift (до 40%), но это не деградация -- это полезная адаптация SID под recommendation objective. DIGER без UD имеет наименьший drift (ниже 15%), что означает ограниченную адаптацию и объясняет его чуть худшее качество. Train-inference agreement (доля item'ов, у которых training-time sampled SID совпадает с inference-time deterministic SID): STE достигает perfect agreement по конструкции, но совпадение тривиально из-за collapse. DIGER без UD имеет persistently low agreement. DIGER с FrqUD быстро достигает и поддерживает высокое agreement; SDUD постепенно увеличивает agreement через снижение $\sigma$. Figure 7 визуализирует code usage distribution как heatmaps ($256$ codebook entries на сетке $16 \times 16$) по трем quantization layers. DIGER с UD показывает наиболее равномерное использование; STE имеет severe collapse, особенно в глубоких layers. 9. Теоретический анализ Помимо уже упомянутых теорем A.1 и A.2 о suboptimality двухстадийного обучения, статья доказывает Теорему A.3: effective code count $\text{Eff}(q) = \exp(H(q))$ максимизируется при uniform distribution $q_i = 1/K$, достигая $\text{Eff}(q) = K$. Entropy bonus в objective pushes code distribution к более высокому Eff(q), т.е. лучшей codebook utilization. Это формально обосновывает, почему exploration через Gumbel noise помогает: оно действует как implicit entropy regularization на ранних этапах обучения. 10. Сильные стороны Статья четко формулирует и теоретически обосновывает objective mismatch между tokenizer'ом и recommender'ом. Это не просто empirical observation, а формализованная проблема с proven lower bounds на suboptimality gap. DIGER показывает, что наивная дифференцируемость (STE) может быть хуже замороженного tokenizer'а, что является контринтуитивным и практически важным предупреждением. Метод дает практически полезные diagnostics: code usage, SID drift, capacity sensitivity, early-stop behavior. Эти метрики полезны для любого проекта с semantic IDs, а не только для DIGER. Метод можно встроить в существующий RQ-VAE/TIGER-like pipeline без изменения inference API: hard SID mapping сохраняется для beam search и prefix tree, soft path используется только для обучения. Это снижает барьер adoption. Experimental comparison с тремя видами baselines (two-stage, STE, SOTA methods) дает convincing evidence. Ablation study изолирует вклад каждого компонента. 11. Слабые стороны, ограничения и спорные моменты Joint training усложняет versioning SID и rollout в production. Если SID меняются между training checkpoints, все downstream systems (prefix tree, item-to-SID table, cached embeddings) должны обновляться синхронно. В offline research это не проблема, но в production это серьезный engineering overhead. Метод требует подбора schedule шума: $\tau$, $\lambda$, $r$, $\beta$ -- четыре дополнительных гиперпараметра поверх обычных recommendation hyperparameters. Плохая настройка может вернуть collapse (слишком мало exploration) или train-serve mismatch (слишком много exploration поздно в обучении). Работа полностью offline: online стоимость frequent SID updates остается открытым вопросом. Для cold-start item'ов, у которых recommendation gradient еще слабый, joint optimization может не давать преимущества над content-based frozen tokenizer. Масштаб экспериментов скромный: максимум 30K users и 20K items. На industrial-scale catalogs с миллионами item'ов codebook collapse может проявляться иначе, и computational cost joint training может стать prohibitive. DIGER использует только text features (через LLaMA-7B) и не интегрирует collaborative signals. На Yelp, где LETTER (с collaborative signals) слегка превосходит DIGER, это может быть значимым ограничением. 12. Практические выводы При внедрении DIGER-подобного подхода стоит логировать usage entropy по каждому уровню codebook и fraction dead codes на каждой эпохе. Необходимо сравнивать three-way setup: frozen SID, STE, DIGER -- без STE baseline нельзя доказать пользу exploration. Важно проверять SID drift: какая доля items меняет код между checkpoints и меняются ли верхние уровни. В inference нужно использовать тот же hard mapping, что и в production, а soft path оставлять только для обучения. Для production систем рекомендуется начинать с shadow deployment: DIGER-SID генерируются параллельно с существующими frozen SID, и только после подтверждения стабильности и улучшения offline metrics переключать traffic. 13. Связь с соседними работами DIGER близок к ETEGRec как работа про joint optimization, но ETEGRec использует alternating distillation и не может directly backpropagate через discrete indexing. DIGER отвечает на другой вопрос: как пустить downstream gradients в SID и не схлопнуть codebook. CoST и ReSID меняют tokenizer objective (contrastive вместо reconstruction), но сохраняют frozen SID paradigm. MERGE меняет production indexing paradigm (streaming clusters вместо fixed VQ), но не решает joint optimization. LETTER добавляет collaborative signals в tokenizer, что ортогонально к differentiable SID -- в принципе, LETTER-style features можно использовать внутри DIGER framework. 14. Итог DIGER важен как предупреждение и решение одновременно. Предупреждение: differentiable semantic IDs не являются бесплатным улучшением -- наивная дифференцируемость через STE хуже frozen tokenizer'а из-за codebook collapse. Решение: controlled exploration через Gumbel noise с adaptive decay позволяет recommendation loss улучшать item-to-code mapping, сохраняя дискретность inference. Для практики статья задает minimum viable checklist для joint SID learning: мониторинг code balance, SID drift, train-inference agreement и capacity sensitivity.
