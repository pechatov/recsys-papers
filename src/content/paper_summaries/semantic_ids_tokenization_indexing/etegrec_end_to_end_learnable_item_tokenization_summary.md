---
title: "ETEGRec: Generative Recommender with End-to-End Learnable Item Tokenization"
category: "semantic_ids_tokenization_indexing"
slug: "etegrec_end_to_end_learnable_item_tokenization_summary"
catalogId: "paper-etegrec_end_to_end_learnable_item_tokenization_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/etegrec_end_to_end_learnable_item_tokenization_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2409.05546"
---
Подробное саммари статьи:

> **Авторы:** Enze Liu, Bowen Zheng, Cheng Ling, Lantao Hu, Han Li, Wayne Xin Zhao, Ji-Rong Wen.
>
> **Аффилиации:** Renmin University of China; Kuaishou Technology.
>
> **Публикация:** SIGIR 2025, Padua, Italy.

## 1. Коротко: о чем статья

ETEGRec (End-To-End Generative Recommender) решает проблему разрыва между item tokenizer'ом и генеративным recommender'ом, которые в стандартных pipeline'ах (TIGER, LETTER) обучаются раздельно. Статья предлагает dual encoder-decoder архитектуру, где RQ-VAE токенизатор и T5-like recommender связаны двумя alignment objectives: Sequence-Item Alignment (SIA), выравнивающим sequence hidden states с target item representation через symmetric KL divergence, и Preference-Semantic Alignment (PSA), использующим InfoNCE между decoder preference state и reconstructed semantic embedding. Alternating optimization обеспечивает стабильное совместное обучение. На трёх Amazon 2023 датасетах ETEGRec значимо превосходит все baselines, включая TIGER и LETTER.

## 2. Контекст

В генеративной рекомендации item'ы представляются как дискретные token sequences, и модель авторегрессионно генерирует токены следующего item'а. Существующие подходы делят pipeline на два независимых этапа: сначала обучается токенизатор (RQ-VAE), затем на фиксированных token'ах обучается recommender. Такое decoupling создаёт фундаментальную проблему: токенизатор оптимизирует reconstruction loss, не зная о downstream recommendation objective, а recommender не может влиять на качество токенизации.

TIGER использует Sentence-T5 embeddings и RQ-VAE, LETTER добавляет collaborative и diversity regularization. Но во всех случаях token sequences фиксируются до обучения recommender'а и не обновляются. ETEGRec -- первая работа, которая делает токенизацию и рекомендацию end-to-end совместимыми через alignment objectives.

## 3. Проблема

Авторы выделяют два аспекта decoupling problem. Во-первых, токенизатор не осведомлён о recommendation objective: он оптимизирует reconstruction, а не downstream Recall/NDCG. Item'ы с хорошими reconstruction но плохими recommendation properties получают одинаковые коды с item'ами, которые recommender'у было бы полезно различать. Во-вторых, recommender не может передать обратно знание о пользовательских предпочтениях в токенизатор: если encoder научился, что определённые item'ы часто следуют друг за другом, эта информация не попадает в token assignment.

## 4. Dual encoder-decoder архитектура

### 4.1. Item Tokenizer (RQ-VAE)

Вход: collaborative embedding $\mathbf{z} \in \mathbb{R}^{d_s}$ (256-мерный из SASRec). Encoder $\mathrm{Encoder}_T$ -- MLP, переводящий $\mathbf{z}$ в latent $\mathbf{r}$. Residual quantization через $L$ codebook'ов $\mathcal{C}_l = \{\mathbf{e}_k^l\}_{k=1}^K$:

$$
c_l = \arg\max_k P(k|\mathbf{v}_l), \quad P(k|\mathbf{v}_l) = \frac{\exp(-\|\mathbf{v}_l - \mathbf{e}_k^l\|^2)}{\sum_{j=1}^K \exp(-\|\mathbf{v}_l - \mathbf{e}_j^l\|^2)}
$$

где $\mathbf{v}_1 = \mathbf{r}$, $\mathbf{v}_l = \mathbf{v}_{l-1} - \mathbf{e}_{c_{l-1}}^l$. Quantized representation: $\tilde{\mathbf{r}} = \sum_{l=1}^L \mathbf{e}_{c_l}^l$. Decoder $\mathrm{Decoder}_T$ восстанавливает: $\tilde{\mathbf{z}} = \mathrm{Decoder}_T(\tilde{\mathbf{r}})$. Semantic quantization loss:

$$
\mathcal{L}_{\mathrm{SQ}} = \|\mathbf{z} - \tilde{\mathbf{z}}\|^2 + \sum_{l=1}^L \left(\|\mathrm{sg}[\mathbf{v}_l] - \mathbf{e}_{c_l}^l\|^2 + \beta\|\mathbf{v}_l - \mathrm{sg}[\mathbf{e}_{c_l}^l]\|^2\right)
$$

В экспериментах: $L = 3$ codebook'а, $K = 256$ кодов, dimension 128, $\beta = 0.25$. К каждому item'у добавляется uniqueness token для разрешения коллизий.

### 4.2. Generative Recommender

T5-like encoder-decoder: 6 encoder layers + 6 decoder layers, hidden 128, FFN 512, 4 heads, head dim 64. Encoder обрабатывает tokenized user history $X$, decoder авторегрессионно генерирует target token sequence $Y$. Recommendation loss:

$$
\mathcal{L}_{\mathrm{REC}} = -\sum_{j=1}^L \log P(Y_j | X, Y_{

5. Recommendation-oriented Alignment 5.1. Sequence-Item Alignment (SIA) Гипотеза: encoder hidden states $\mathbf{H}^E$ должны быть тесно связаны с collaborative embedding target item'а. Авторы проецируют mean-pooled encoder output в пространство токенизатора: \[\mathbf{z}^E = \mathrm{MLP}(\mathrm{mean\_pool}(\mathbf{H}^E))
$$

 Затем $\mathbf{z}^E$ и $\mathbf{z}$ (target item embedding) пропускаются через токенизатор, получая распределения $P_{\mathbf{z}}^l$ и $P_{\mathbf{z}^E}^l$ на каждом уровне. SIA loss -- symmetric KL divergence: 

$$
\mathcal{L}_{\mathrm{SIA}} = -\sum_{l=1}^L \bigl(D_{\mathrm{KL}}(P_{\mathbf{z}}^l \| P_{\mathbf{z}^E}^l) + D_{\mathrm{KL}}(P_{\mathbf{z}^E}^l \| P_{\mathbf{z}}^l)\bigr)
$$

 SIA заставляет encoder recommender'а формировать представления, совместимые с token space. Дополнительный эффект: SIA решает проблему bypass encoder, когда decoder игнорирует encoder и полагается только на уже сгенерированные токены. 5.2. Preference-Semantic Alignment (PSA) Гипотеза: первый hidden state decoder'а $\mathbf{h}^D$ кодирует user preference, а reconstructed semantic embedding $\tilde{\mathbf{z}}$ -- семантику target item'а. Они должны быть выровнены. PSA -- symmetric InfoNCE с in-batch negatives: 

$$
\mathcal{L}_{\mathrm{PSA}} = -\Bigl(\log\frac{\exp(s(\tilde{\mathbf{z}}, \mathbf{h}^D)/\tau)}{\sum_{\hat{\mathbf{h}} \in B} \exp(s(\tilde{\mathbf{z}}, \hat{\mathbf{h}})/\tau)} + \log\frac{\exp(s(\mathbf{h}^D, \tilde{\mathbf{z}})/\tau)}{\sum_{\hat{\mathbf{z}} \in B} \exp(s(\mathbf{h}^D, \hat{\mathbf{z}})/\tau)}\Bigr)
$$

 где $s(\cdot, \cdot)$ -- cosine similarity, $\tau$ -- temperature. 6. Alternating Optimization Обучение делится на циклы длины $C$. В первую эпоху каждого цикла обучается токенизатор (recommender заморожен): 

$$
\mathcal{L}_{\mathrm{IT}} = \mathcal{L}_{\mathrm{SQ}} + \mu\,\mathcal{L}_{\mathrm{SIA}} + \lambda\,\mathcal{L}_{\mathrm{PSA}}
$$

 В оставшиеся $C-1$ эпох обучается recommender (токенизатор заморожен): 

$$
\mathcal{L}_{\mathrm{GR}} = \mathcal{L}_{\mathrm{REC}} + \mu\,\mathcal{L}_{\mathrm{SIA}} + \lambda\,\mathcal{L}_{\mathrm{PSA}}
$$

 Чередование продолжается до сходимости токенизатора, после чего recommender дообучается. $C$ настраивается в $\{2, 4\}$. Этот механизм предотвращает нестабильность, которая возникает при полностью совместном обучении: слишком частые обновления token assignments дестабилизируют recommender. 6.1. Псевдокод обучения ETEGRec Практически ETEGRec - это не один backward pass через все компоненты, а controlled alternating schedule. Главный artifact, который постоянно меняется во время обучения, - mapping item -> token sequence, поэтому recommender нельзя обучать так, будто tokens статичны. `initialize tokenizer RQ-VAE from collaborative item embeddings initialize T5-like generative recommender for cycle in training_cycles: freeze(recommender) unfreeze(tokenizer) for epoch in first_epoch_of_cycle: z = collaborative_embedding(target_item) token_probs, token_ids, z_recon = tokenizer(z) encoder_state = recommender.encode(history_token_sequence) decoder_state = recommender.decode_prefix(target_token_sequence) loss = L_SQ(z, z_recon, token_ids) + mu * L_SIA(encoder_state, z, tokenizer) + lambda * L_PSA(decoder_state, z_recon) update(tokenizer) refresh_item_to_token_mapping() freeze(tokenizer) unfreeze(recommender) for epoch in remaining_epochs_of_cycle: target_tokens = lookup_tokens(target_item) logits = recommender(history_tokens, target_prefix=target_tokens) loss = L_REC(logits, target_tokens) + mu * L_SIA(recommender.encoder_state, target_item_embedding, tokenizer) + lambda * L_PSA(recommender.decoder_state, tokenizer.reconstruct(target_item)) update(recommender) after tokenizer_converges: freeze(tokenizer) train recommender to convergence on fixed final token mapping serve with cached item token assignments and beam search` 7. Рисунки <img src="../../assets/etegrec/framework.png" alt="ETEGRec framework: dual encoder-decoder with SIA and PSA alignment"> Рисунок показывает dual encoder-decoder architecture: item tokenizer (RQ-VAE) слева и generative recommender (T5) справа. SIA связывает encoder output recommender'а с item representation через tokenizer. PSA связывает decoder preference state с reconstructed semantic embedding. Стрелки alignment losses показывают, что information flow идёт в обоих направлениях. 8. Эксперименты и результаты 8.1. Датасеты Датасет#Users#Items#InteractionsSparsity Instrument57,43924,587511,83699.964% Scientific50,985~25,000--~99.969% Game94,76225,612814,58699.966% Amazon 2023 review data, 5-core filtering, max sequence length 50, leave-one-out evaluation, full ranking. Beam size 20. 8.2. Основные результаты МодельInst R@5Inst R@10Inst N@5Inst N@10Game R@10Game N@10 SASRec0.03410.05300.02170.02770.08210.0426 FDSA0.03640.05570.02330.02950.08570.0453 TIGER0.03680.05740.02420.03080.08950.0471 LETTER0.03720.05810.02430.03100.09010.0475 **ETEGRec****0.0402****0.0624****0.0260****0.0331****0.0947****0.0507** Все улучшения статистически значимы ($p < 0.01$). ETEGRec превосходит LETTER на 7-8% по Recall@10 на Instrument и на 5% на Game. Среди традиционных моделей лучше всех FDSA благодаря текстовым features. 8.3. Ablation study ВариантInst R@10Inst N@10Game R@10Game N@10 ETEGRec (full)0.06240.03310.09470.0507 w/o SIA----0.09170.0491 w/o PSA0.06090.03210.09330.0499 w/o SIA + PSA0.0601--0.08940.0478 w/o Alternating Training0.05290.02770.08100.0428 w/o End-to-End----0.08990.0475 Ключевые выводы. Удаление alternating training даёт самое сильное падение: без чередования recommender дестабилизируется из-за постоянных обновлений token assignments. Удаление обоих alignment losses приближает результат к TIGER/LETTER baseline. Важно: вариант w/o ETE (обучение отдельного recommender'а на финальных токенах) хуже полного ETEGRec, что доказывает: выигрыш не только от лучших идентификаторов, но и от интеграции prior knowledge токенизатора в recommender. 8.4. Generalizability На 5% unseen users (с наименьшей историей) ETEGRec также превосходит LETTER и TIGER, демонстрируя более robust preference modeling через alignment. Это важно для cold-start сценариев. 8.5. t-SNE визуализация Визуализация $\mathbf{h}^D$ (preference) и $\tilde{\mathbf{z}}$ (semantic) для 10 item'ов с 80 историями каждый показывает: preference points кластеризуются вокруг соответствующих target semantic points и отделены от чужих. Это подтверждает, что PSA действительно выравнивает preference space и semantic space. 9. Complexity analysis Item tokenization per item: $O(d^2 + LKd)$. Generative recommendation: $O(N^2d + Nd^2)$. Loss computation: REC -- $O(LKd)$, SIA -- $O(LKd)$, PSA -- $O(Md)$. Общий порядок training complexity совпадает с TIGER и LETTER. Inference complexity идентична TIGER, потому что token assignments кэшируются. 10. Hyperparameter analysis $\mu$ (SIA coefficient): оптимально $3 \times 10^{-4}$ на Instrument/Scientific, $10^{-3}$ на Game. Слишком большое $\mu$ мешает основному обучению. $\lambda$ (PSA coefficient): лучший результат при $\lambda = 10^{-4}$ на всех трёх датасетах. 11. Сильные стороны **Прямая атака на decoupling problem.** ETEGRec -- первая работа, которая формально связывает item tokenizer и generative recommender через alignment objectives. SIA и PSA работают с двух сторон: sequence-level и target-level. **Inference-compatible.** Благодаря кэшированию token assignments, inference cost идентичен TIGER. Alternating optimization добавляет overhead только в training. **Чистый ablation design.** Каждый компонент (SIA, PSA, alternating training, end-to-end) проверяется отдельно на трёх датасетах, что даёт ясную картину contribution. **Open-source.** Код доступен на GitHub (RUCAIBox/ETEGRec), что важно для воспроизводимости. 12. Ограничения **Operational complexity.** Alternating training требует аккуратного управления: переключение между optimization targets, версионирование token maps внутри training loop. В production это значительно сложнее, чем two-stage pipeline. **SID churn.** При каждом обновлении токенизатора token assignments меняются. В production частая смена SIDs требует синхронизации между моделью, индексом и serving stack. **Hyperparameter sensitivity.** $\mu$ и $\lambda$ требуют аккуратной настройки; оптимальные значения различаются между датасетами. Перенос на новый домен не гарантирован без дополнительного grid search. **Масштаб экспериментов.** Amazon 2023 subsets -- средний масштаб (50-95k users). Нет проверки на industrial-scale данных с миллионами item'ов и streaming обновлениями. **Нет online evaluation.** В отличие от PIT (той же группы из Kuaishou), ETEGRec не имеет A/B-теста в production. 13. Практические выводы ETEGRec показывает, что end-to-end alignment между токенизатором и recommender'ом даёт значимый прирост поверх LETTER-style regularization. Для практика ключевые takeaways: (1) alternating training стабильнее полностью совместного обучения; (2) alignment через distribution matching (KL) и contrastive (InfoNCE) работают комплементарно; (3) inference cost не увеличивается благодаря кэшированию токенов. 14. Связь с соседними работами ETEGRec -- прямой преемник TIGER и LETTER. От TIGER он отличается end-to-end обучением. От LETTER -- не только добавлением CF/diversity regularization, но и feedback loop от recommender'а к tokenizer'у. Conceptually ETEGRec ближе к PIT (Kuaishou), который также делает end-to-end, но через co-generative architecture с beam index вместо alternating optimization. SIIT (Snap) предлагает альтернативу: итеративную relabeling без differentiable joint optimization. 15. Итог ETEGRec -- важный шаг от decoupled к end-to-end генеративной рекомендации. Его главный вклад -- формализация двух alignment objectives (SIA и PSA), которые связывают sequence representation recommender'а с token space tokenizer'а. Alternating optimization обеспечивает стабильность. Статья хорошо дополняет LETTER: если LETTER показывает, что tokenizer должен быть CF-aware, ETEGRec показывает, что tokenizer и recommender должны быть aware друг друга. Ограничения связаны с operational complexity и отсутствием production validation.
