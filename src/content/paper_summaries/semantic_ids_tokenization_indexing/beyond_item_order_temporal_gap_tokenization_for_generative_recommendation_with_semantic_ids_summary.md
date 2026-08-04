---
title: "Beyond Item Order: Temporal Gap Tokenization for Generative Recommendation with Semantic IDs"
category: "semantic_ids_tokenization_indexing"
slug: "beyond_item_order_temporal_gap_tokenization_for_generative_recommendation_with_semantic_ids_summary"
catalogId: "paper-beyond_item_order_temporal_gap_tokenization_for_generative_recommendation_with_semantic_ids_summary"
paperUrl: "https://arxiv.org/abs/2607.03918"
---
> **Авторы:** Chengkai Huang, Tianqi Gao, Hongtao Huang, Quan Z. Sheng, Lina Yao.
>
> **Аффилиации:** University of New South Wales; Macquarie University; CSIRO's Data61; Independent Researcher.
>
> **Источник:** arXiv:2607.03918v1 от 2026-07-04. Venue и публичный code repository не указаны.

## 1. Коротко

ChronoSID атакует temporal blindness generative recommendation: две user histories с одинаковым порядком items, но с интервалами в минуты и месяцы обычно кодируются одинаково. Между тем короткий gap чаще означает продолжение текущего interest context, а длинный - drift или новый intent.

Метод добавляет время в ReSID-like pipeline в двух местах. На item-representation stage auxiliary regression заставляет encoder предсказывать log time gap до target interaction. На generator stage реальные gaps дискретизируются в пять понятных log-scale buckets и interleave'ятся с трёхуровневыми SID tuples. Decoder по-прежнему генерирует только три SID tokens target item, поэтому output space и decoding depth не меняются.

Главный эмпирический вывод: непосредственные gap tokens дают большую часть gain, а auxiliary temporal loss добавляет небольшой complementary effect. На восьми Amazon-2023 domains ChronoSID стабильно лучше ReSID по Recall/NDCG, но абсолютные improvements обычно малы. Более заметный эффект появляется в long-gap groups. Это полезная controlled paper про input representation, а не доказательство production scalability или универсального superiority time-aware GR.

<figure class="paper-figure">
  <img src="../../assets/chronosid/framework.png" alt="ChronoSID three-stage framework with temporal auxiliary task and gap token interleaving">
  <figcaption>Рисунок 1. ChronoSID сохраняет GAOQ tokenization ReSID, добавляя temporal auxiliary task в Stage 1 и gap-token interleaving в Stage 3.</figcaption>
</figure>

## 2. Мотивация

Авторы измеряют same-category purchase rate между соседними interactions и показывают его снижение с ростом gap во всех рассмотренных Amazon domains. Это не causal evidence preference drift, но разумный proxy: порядок items не содержит информации о скорости поведения.

Большинство SID papers оптимизирует сам item code - RQ-VAE, collaborative tokenization, aligned clusters, collision policy - однако generator получает history как плоскую последовательность static tuples. ChronoSID задаёт другой вопрос: можно ли сохранить compact SID output, но сделать encoder history time-aware?

Ключевое отличие от ChronoID: ChronoSID не делает ID item'а зависимым от времени. Semantic ID mapping остаётся статичным; gap - отдельный context token между interactions. Это проще для lookup, caching и backward compatibility.

## 3. Pipeline

### 3.1. Stage 1: TA-FAMAE

Каждый item описан structured fields: title/text, category, brand и item ID. Для target item случайно маскируется непустое подмножество fields; Transformer encoder восстанавливает их через field-specific embedding spaces. Число masked fields случайно выбирается от 1 до $F$.

ChronoSID добавляет regression head, который по target hidden state $h_L$ предсказывает log-transformed gap до предыдущего interaction:

$$
\Delta t_L=t_L-t_{L-1}, \qquad
\hat z_L=w_g^\top h_L+b_g,
$$

$$
\mathcal{L}_{time}=(\hat z_L-\log(1+\Delta t_L))^2.
$$

Итоговая objective:

$$
\mathcal{L}_{TA-FAMAE}=\mathcal{L}_{feat}+\lambda\mathcal{L}_{time},
$$

default $\lambda=0.1$. После обучения для каждого item извлекается deterministic embedding без masked fields и замораживается. Authors подчёркивают: item ID не становится dynamic; temporal task только regularizes representation.

### 3.2. Stage 2: GAOQ без изменений

Для controlled comparison используется quantizer ReSID - Globally Aligned Orthogonal Quantization:

1. balanced K-Means создаёт 32 coarse level-1 clusters;
2. внутри каждого cluster строятся level-2 subclusters, а Hungarian matching связывает local indices с global orthogonal anchors;
3. level 3 разрешает collisions внутри $(c_1,c_2)$ prefix и обеспечивает unique item IDs.

Item получает static tuple $\Phi(v)=(c_1^v,c_2^v,c_3^v)$. ChronoSID не меняет этот этап, поэтому разницу с ReSID можно в основном отнести к temporal additions.

### 3.3. Stage 3: gap tokens

Для первого history item используется special `g_start`. Остальные gaps попадают в пять fixed buckets:

- меньше 1 часа;
- от 1 часа до 1 дня;
- от 1 дня до 1 недели;
- от 1 недели до 1 месяца;
- не меньше 1 месяца.

Перед каждым SID tuple вставляется один gap token:

$$
x_u=[\delta_1,\tilde c_1^{v_1},\tilde c_2^{v_1},\tilde c_3^{v_1},\ldots,
\delta_L,\tilde c_1^{v_L},\tilde c_2^{v_L},\tilde c_3^{v_L}].
$$

Encoder length растёт с $3L$ до $4L$. Lightweight T5-style encoder-decoder обучается from scratch teacher-forced cross-entropy. Target остаётся трёхтокенным SID следующего item. Beam sizes на каждом уровне равны `[50, 50, 50]`; invalid tuples отбрасываются, duplicates merge'ятся по лучшему generation score.

## 4. Experimental setup

Используются восемь Amazon-2023 subsets после 5-core filtering и chronological leave-one-out: последний interaction - test, предпоследний - validation, остальные - train.

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>Users</th><th>Items</th><th>Interactions</th></tr></thead>
<tbody>
<tr><td>Musical Instruments</td><td>57,359</td><td>23,742</td><td>490,522</td></tr>
<tr><td>Video Games</td><td>94,515</td><td>24,685</td><td>772,218</td></tr>
<tr><td>Industrial & Scientific</td><td>50,886</td><td>25,142</td><td>394,989</td></tr>
<tr><td>Baby Products</td><td>150,642</td><td>35,024</td><td>1,189,171</td></tr>
<tr><td>Arts, Crafts & Sewing</td><td>196,980</td><td>87,449</td><td>1,706,484</td></tr>
<tr><td>Sports & Outdoors</td><td>409,309</td><td>151,411</td><td>3,333,753</td></tr>
<tr><td>Toys & Games</td><td>431,411</td><td>156,537</td><td>3,652,250</td></tr>
<tr><td>Beauty & Personal Care</td><td>712,259</td><td>193,383</td><td>5,785,124</td></tr>
</tbody>
</table>
</div>

Baselines включают HGN, SASRec, BERT4Rec, S3-Rec, их side-information variants, а также TIGER, LETTER, EAGER, UNGER, ETEGRec и ReSID. Main results averaged по пяти seeds на NVIDIA V100. Метрики: Recall@5/10 и NDCG@5/10.

## 5. Main results

Ниже показано компактное comparison с ReSID; полная paper table содержит 15 methods и четыре metrics на каждый dataset.

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>Method</th><th>R@5</th><th>R@10</th><th>N@5</th><th>N@10</th></tr></thead>
<tbody>
<tr><td>MI</td><td>ReSID</td><td>0.0388</td><td>0.0614</td><td>0.0253</td><td>0.0325</td></tr>
<tr><td>MI</td><td>ChronoSID</td><td><strong>0.0417</strong></td><td><strong>0.0645</strong></td><td><strong>0.0273</strong></td><td><strong>0.0346</strong></td></tr>
<tr><td>VG</td><td>ReSID</td><td>0.0571</td><td>0.0898</td><td>0.0375</td><td>0.0480</td></tr>
<tr><td>VG</td><td>ChronoSID</td><td><strong>0.0597</strong></td><td><strong>0.0927</strong></td><td><strong>0.0396</strong></td><td><strong>0.0501</strong></td></tr>
<tr><td>IS</td><td>ReSID</td><td>0.0305</td><td>0.0478</td><td>0.0194</td><td>0.0250</td></tr>
<tr><td>IS</td><td>ChronoSID</td><td><strong>0.0340</strong></td><td><strong>0.0512</strong></td><td><strong>0.0218</strong></td><td><strong>0.0273</strong></td></tr>
<tr><td>BPC</td><td>ReSID</td><td>0.0223</td><td>0.0346</td><td>0.0146</td><td>0.0185</td></tr>
<tr><td>BPC</td><td>ChronoSID</td><td><strong>0.0228</strong></td><td><strong>0.0348</strong></td><td><strong>0.0149</strong></td><td><strong>0.0189</strong></td></tr>
</tbody>
</table>
</div>

Gains consistent, но существенно отличаются по domain. На IS relative improvements заметные; на large sparse BPC абсолютный R@10 gain всего 0.0002. Кроме того, ChronoSID не всегда global winner против item-ID models: side-information SASRec лучше по R@10 на IS, SO и TG. Корректный claim - лучший среди tested SID-based generators, а не лучший во всех columns.

## 6. Что даёт основной gain

<div class="table-scroll">
<table>
<thead><tr><th>Method</th><th>MI R@10</th><th>MI N@10</th><th>VG R@10</th><th>VG N@10</th><th>IS R@10</th><th>IS N@10</th></tr></thead>
<tbody>
<tr><td>ReSID</td><td>0.0614</td><td>0.0325</td><td>0.0898</td><td>0.0480</td><td>0.0478</td><td>0.0250</td></tr>
<tr><td>+ TA-FAMAE</td><td>0.0620</td><td>0.0333</td><td>0.0904</td><td>0.0480</td><td>0.0482</td><td>0.0255</td></tr>
<tr><td>+ Gap Tokens</td><td>0.0643</td><td>0.0344</td><td>0.0945</td><td>0.0507</td><td>0.0499</td><td>0.0263</td></tr>
<tr><td>ChronoSID</td><td><strong>0.0651</strong></td><td><strong>0.0347</strong></td><td><strong>0.0952</strong></td><td><strong>0.0509</strong></td><td><strong>0.0512</strong></td><td><strong>0.0273</strong></td></tr>
</tbody>
</table>
</div>

Gap tokens - основной механизм. TA-FAMAE alone даёт небольшой gain, потому что temporal information влияет на generator лишь косвенно через static item embeddings. Full model обычно лучший, значит representation regularization всё же complementary.

Main table и ablation table содержат слегка разные full-model numbers, например MI R@10 0.0645 против 0.0651. Authors объясняют, что main results averaged over five seeds; ablation table используется как отдельный component study. Сравнивать absolute values между этими таблицами нужно осторожно.

## 7. Long-gap и popularity diagnostics

<figure class="paper-figure">
  <img src="../../assets/chronosid/temporal_robustness.png" alt="ChronoSID and ReSID performance by target-side time gap">
  <figcaption>Рисунок 2. ChronoSID лучше ReSID во всех gap groups; абсолютное качество обеих моделей падает при долгом отсутствии пользователя.</figcaption>
</figure>

Test cases группируются по gap между последним history item и target. Этот target-side gap используется только для analysis и не подаётся модели. Gains сохраняются во всех groups и становятся визуально заметнее для `[1 week, 1 month)` и `>=1 month`, что согласуется с drift motivation.

Popularity analysis определяет top-20% training-frequency items как popular. На Video Games relative gains на unpopular items выше: +10.17% R@10 и +13.59% N@10 против +5.86%/+5.74% на popular. На Musical Instruments картина обратная: larger gains на popular. Поэтому temporal tokens не выглядят простой popularity amplifier, но эффект domain-dependent.

<figure class="paper-figure">
  <img src="../../assets/chronosid/popularity.png" alt="Relative ChronoSID improvement for popular and unpopular target items">
  <figcaption>Рисунок 3. Temporal gain наблюдается в обоих popularity regimes, но соотношение меняется между domains.</figcaption>
</figure>

## 8. Sensitivity и cost

Лучший tested $\lambda=0.1$. Слишком большой temporal weight 0.2 слегка ухудшает quality: regression начинает отвлекать representation learner от item semantics.

Пять bins лучше 3, 7 и 10. Слишком мало bins смешивает разные rhythms, слишком много делает gap tokens sparse. Это поддерживает coarse interpretable tokenization, но paper не сравнивает её с learned bucket boundaries или continuous time embeddings.

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>Model</th><th>Train time</th><th>Inference</th><th>ms/sample</th><th>R@10</th></tr></thead>
<tbody>
<tr><td>MI</td><td>ReSID</td><td>174 min</td><td>39.61 s</td><td>0.69</td><td>0.0614</td></tr>
<tr><td>MI</td><td>ChronoSID</td><td>187 min (1.07x)</td><td>49.13 s</td><td>0.86</td><td>0.0651</td></tr>
<tr><td>VG</td><td>ReSID</td><td>275 min</td><td>156.39 s</td><td>1.68</td><td>0.0898</td></tr>
<tr><td>VG</td><td>ChronoSID</td><td>356 min (1.29x)</td><td>199.51 s</td><td>2.15</td><td>0.0952</td></tr>
</tbody>
</table>
</div>

Encoder sequence длиннее на 33%, но decoder target и beam depth те же. Measured overhead умеренный на этих datasets, хотя перенос на histories и catalogs production scale не проверен.

## 9. Сильные стороны

- Очень простой и совместимый temporal signal: static SID mapping не нужно обновлять при каждом time context.
- Controlled comparison с ReSID сохраняет quantizer, backbone и beam settings.
- Ablation ясно отделяет auxiliary representation learning от direct generator conditioning.
- Есть five-seed main results, long-gap, popularity, sensitivity, efficiency и paired output diagnostics.

## 10. Ограничения и вопросы

Все datasets - Amazon product reviews с 5-core filtering и leave-one-out. Это не temporal production split и не fast-changing feed; generalization на video/news/music, seasonality и repeated consumption не показана.

Нет сильных explicit time-aware baselines. ChronoSID сравнивается с static sequence models, но не с тем же T5, получающим continuous/log time embeddings, relative positional bias, learned buckets или Time2Vec. Поэтому доказано, что gap tokens лучше отсутствия времени, но не что fixed five-bin tokenization - лучший способ.

Main table не показывает variance или significance, хотя усредняется по пяти seeds. Положительные paired bootstrap intervals опубликованы только для output analysis на MI.

TA-FAMAE предсказывает gap до target item при representation learning, но после обучения каждый item получает один deterministic embedding. Механизм, через который такой context-dependent pretext task улучшает static embedding, раскрыт только через downstream ablation; нет representation-level temporal probing.

Код не опубликован. Reproduction требует реализации ReSID/GAOQ, точных preprocessing details и разрешения расхождения между main и ablation numbers.

Дополнительный token на item увеличивает encoder attention cost; при очень длинных histories рост может быть заметнее, чем на tested setup. Нет memory/throughput scaling curves.

## 11. Итог

ChronoSID показывает простой и правдоподобный способ устранить temporal blindness SID generator: не менять item identifiers, а interleave'ить history с gap tokens. Наиболее надёжный takeaway - прямой temporal context в generator полезнее, чем попытка спрятать время только в item representation. Evidence consistent, но gains небольшие и benchmark-bound; без time-aware baselines и public code paper следует воспринимать как сильную design ablation, а не окончательный ответ на temporal modeling в generative recommendation.

## Источники

- [arXiv:2607.03918](https://arxiv.org/abs/2607.03918)
