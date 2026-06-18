---
title: "RQ-VAE: Autoregressive Image Generation using Residual Quantization"
category: "generative_retrieval"
slug: "rqvae_autoregressive_image_generation_residual_quantization_summary"
catalogId: "paper-rqvae_autoregressive_image_generation_residual_quantization_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/rqvae_autoregressive_image_generation_residual_quantization_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2203.01941"
---
> Lee, D., Kim, C., Kim, S., Cho, M., & Han, B. (2022).
>
> *Autoregressive Image Generation using Residual Quantization.*
>
> CVPR 2022. Kakao Brain. arXiv:
>
> [2203.01941](https://arxiv.org/abs/2203.01941)

**Историческое значение.** Эта статья — прямой технический предшественник Semantic IDs в рекомендательных системах. Именно архитектуру RQ-VAE (Residual Quantized VAE), разработанную здесь для генерации изображений, адаптировал TIGER (Rajput et al., 2023), чтобы превратить item embedding'и в иерархические дискретные идентификаторы. Понимание RQ-VAE обязательно для понимания TIGER, LETTER, CoST, DIGER и всего направления Semantic ID.

---

## 1. Коротко: о чём статья

Статья решает задачу авторегрессивной генерации изображений высокого разрешения с помощью двухкомпонентной системы:

- **RQ-VAE (Residual Quantized VAE)** — архитектура для многоуровневой дискретной компрессии изображений. Изображение сжимается в карту пространственных токенов (spatial feature map), и каждый пространственный токен кодируется не одним целым числом, а кортежем из $L$ целых чисел через *остаточное квантование* (residual quantization). Каждое из $L$ чисел берётся из своего независимого codebook.
- **RQTRAN (RQ Transformer)** — авторегрессивный трансформер (GPT-style), генерирующий изображения покодово: он предсказывает последовательность кодов, которую затем декодирует RQ-VAE.

Ключевая идея: вместо того чтобы представлять каждый пространственный токен одним кодом из огромного словаря, RQ-VAE разбивает представление на $L$ последовательных *остатков*, каждый из которых квантуется отдельным небольшим codebook'ом. Это позволяет точно кодировать изображение при меньшем размере каждого словаря, избегая проблемы codebook collapse.

## 2. Мотивация: почему не одноуровневый VQ-VAE

В классическом VQ-VAE (van den Oord et al., 2017) каждый пространственный токен кодируется одним entry из единственного codebook размером $K$. Ключевое ограничение: для того, чтобы точно представить сложный, высокодетальный объект — например, изображение 256×256 — одним кодом на пространственный токен, нужен очень большой словарь ($K = 256\text{K}+$ entries).

Проблемы большого $K$:

- **Codebook collapse.** При обучении большинство entry в codebook не используется (мёртвые entry). Encoder коллапсирует в маленькое подмножество кодов, теряя выразительную ёмкость.
- **Масштабируемость.** Словарь из 256K entry требует 256K embeddings — это огромное число параметров только для одного codebook'а. Transformer'у, работающему поверх VQ-VAE, приходится иметь softmax-голову размером 256K, что вычислительно тяжело.
- **Нестабильность обучения.** Большой nearest-neighbour поиск нестабилен на ранних стадиях обучения, прежде чем encoder научился производить полезные vectors.

**Residual Quantization решает эти проблемы**, разбивая представление на $L$ уровней с небольшими codebook'ами размером $K$ каждый. Эффективная ёмкость кодирования: $K^L$ комбинаций вместо $K$ в одноуровневом случае. Например, $L=4$, $K=1024$ даёт $10^{12}$ возможных кодов, тогда как одноуровневый VQ с эквивалентным покрытием потребовал бы $K = 10^{12}$ — абсолютно неосуществимо.

## 3. VQ-VAE как предшественник

RQ-VAE строится поверх идей VQ-VAE, поэтому краткое напоминание существенно.

VQ-VAE состоит из трёх компонентов:

1. **Encoder** $E$: отображает входное изображение $x \in \mathbb{R}^{H \times W \times 3}$ в непрерывную карту $z = E(x) \in \mathbb{R}^{h \times w \times d}$, где $h \times w$ — пространственное разрешение после downsampling, $d$ — размерность embedding.
1. **Квантование** : для каждого пространственного токена $z_{ij}$ находим ближайший entry в codebook $\mathcal{C} = \{e_k\}_{k=1}^{K}$: 

$$
c_{ij} = \arg\min_{k} \| z_{ij} - e_k \|_2^2, \quad \hat{z}_{ij} = e_{c_{ij}}
$$

1. **Decoder** $D$: восстанавливает изображение $\hat{x} = D(\hat{z})$ из квантованной карты $\hat{z}$.

**Проблема градиентов.** Операция $\arg\min$ не дифференцируема. VQ-VAE использует *straight-through estimator*: при обратном распространении градиент копируется «напрямую» от $\hat{z}$ к $z$, минуя квантование.

**Функция потерь VQ-VAE:**

$$
\mathcal{L}_{\text{VQ-VAE}} = \underbrace{\|x - \hat{x}\|^2}_{\text{reconstruction}} + \underbrace{\|\text{sg}[z] - e_c\|^2}_{\text{VQ loss}} + \underbrace{\beta \|z - \text{sg}[e_c]\|^2}_{\text{commitment loss}}
$$

где $\text{sg}[\cdot]$ — stop-gradient (операция, блокирующая поток градиентов), $\beta$ — коэффициент commitment loss (типично 0.25).

**VQ loss** обновляет codebook (тянет entries к encoder output). **Commitment loss** обновляет encoder (тянет encoder output к ближайшему entry, предотвращая «блуждание» представлений).

Почему одного уровня VQ недостаточно: после квантования $\hat{z}_{ij} = e_{c_{ij}}$ остаётся ненулевой остаток $r_{ij} = z_{ij} - e_{c_{ij}}$. Этот остаток содержит ту часть информации, которую единственный codebook не смог закодировать. VQ-VAE теряет эту информацию — отсюда ограниченное качество реконструкции при малых словарях.

## 4. Residual Quantization: алгоритм с формулами

Residual Quantization (RQ) итеративно квантует остатки предыдущих уровней. Формально, для каждого пространственного токена $z \in \mathbb{R}^d$ (для краткости индексы $ij$ опущены):

### 4.1. Инициализация

$$
r_0 = z = E(x)
$$

### 4.2. Итеративное квантование ($l = 1, 2, \ldots, L$)

$$
c_l = \arg\min_{k \in \{1,\ldots,K\}} \| r_{l-1} - e_k^{(l)} \|_2^2
$$

$$
r_l = r_{l-1} - e_{c_l}^{(l)}
$$

где $e_k^{(l)}$ — $k$-й entry в $l$-м codebook $\mathcal{C}^{(l)} = \{e_k^{(l)}\}_{k=1}^K$. Каждый уровень имеет свой независимый codebook.

### 4.3. Суммарное квантованное представление

$$
\tilde{z} = \sum_{l=1}^{L} e_{c_l}^{(l)}
$$

Квантованный вектор $\tilde{z}$ — это сумма selected entries из всех $L$ codebook'ов. Он является аппроксимацией исходного $z$, и ошибка аппроксимации уменьшается с каждым уровнем: $r_L \to 0$ при $L \to \infty$.

### 4.4. Item Code (идентификатор)

$$
\text{ID}(z) = (c_1, c_2, \ldots, c_L)
$$

Это **иерархический кортеж** из $L$ целых чисел, каждое в диапазоне $[1, K]$. Именно этот кортеж в TIGER называют *Semantic ID* для item'а.

### 4.5. Декодирование

$$
\hat{x} = D(\tilde{z})
$$

### 4.6. Интерпретация уровней

Ключевое свойство: уровни представляют грубость/детализацию:

- $c_1$ — грубая семантика: entry $e_{c_1}^{(1)}$ несёт основную часть информации о $z$. Ближайший entry покрывает dominant direction в пространстве.
- $c_2$ — уточнение: entry $e_{c_2}^{(2)}$ аппроксимирует остаток $r_1$, добавляя детали, которые пропустил первый уровень.
- $c_L$ — тонкие детали: последний уровень кодирует минимальный остаток, соответствующий высокочастотным текстурам или тонким вариациям.

## 5. Training Objective

Полная функция потерь RQ-VAE:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{vq}} + \mathcal{L}_{\text{commit}}
$$

### 5.1. Reconstruction Loss

$$
\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2
$$

MSE между исходным и восстановленным изображением. На практике авторы также используют perceptual loss (LPIPS) и adversarial loss (дискриминатор) для улучшения визуального качества — как в VQGAN (Esser et al., 2021).

### 5.2. VQ Loss (codebook update)

$$
\mathcal{L}_{\text{vq}} = \sum_{l=1}^{L} \|\text{sg}[r_{l-1}] - e_{c_l}^{(l)}\|^2
$$

Тянет entry $e_{c_l}^{(l)}$ в сторону текущего остатка $r_{l-1}$. Stop-gradient на $r_{l-1}$ означает, что этот loss обновляет только codebook, не encoder.

### 5.3. Commitment Loss (encoder update)

$$
\mathcal{L}_{\text{commit}} = \sum_{l=1}^{L} \beta \cdot \|r_{l-1} - \text{sg}[e_{c_l}^{(l)}]\|^2
$$

Тянет encoder output (и, транзитивно, каждый остаток) в сторону ближайшего entry. Stop-gradient на entry означает, что codebook при этом не обновляется. Коэффициент $\beta = 0.25$ — стандартное значение из оригинального VQ-VAE.

### 5.4. EMA Update для codebook

Вместо gradient descent для обновления codebook авторы используют **Exponential Moving Average (EMA)** — стабилизирующую технику из VQ-VAE-2 (Razavi et al., 2019):

$$
N_k^{(l)} \leftarrow \gamma N_k^{(l)} + (1-\gamma) n_k^{(l)}, \quad m_k^{(l)} \leftarrow \gamma m_k^{(l)} + (1-\gamma) \sum_{z: c_l(z)=k} r_{l-1}(z)
$$

$$
e_k^{(l)} \leftarrow \frac{m_k^{(l)}}{N_k^{(l)}}
$$

где $n_k^{(l)}$ — число векторов, отображённых на entry $k$ в текущем батче; $\gamma$ — decay rate (обычно 0.99). EMA обновление интерпретируется как online k-means: codebook entries — это скользящее среднее назначенных им векторов. При использовании EMA $\mathcal{L}_{\text{vq}}$ можно отбросить, так как codebook обновляется вне графа градиентов.

### 5.5. K-means инициализация

Перед началом обучения авторы инициализируют каждый codebook $\mathcal{C}^{(l)}$ с помощью k-means кластеризации на первом батче encoder output'ов (для $l=1$) или остатков (для $l>1$). Это резко снижает вероятность codebook collapse на ранних итерациях, когда encoder ещё не обучен.

## 6. RQTRAN: авторегрессивная генерация

После обучения RQ-VAE каждое изображение представлено матрицей кодов $\mathbf{C} \in \{1,\ldots,K\}^{h \times w \times L}$. RQTRAN — GPT-style авторегрессивный трансформер — обучается предсказывать следующий код в линеаризованной последовательности.

### 6.1. Линеаризация

Коды разворачиваются в плоскую последовательность длиной $h \cdot w \cdot L$ по следующей схеме:

- Внешний цикл: пространственные позиции слева направо, сверху вниз (raster scan), $h \times w$ позиций.
- Внутренний цикл: для каждой пространственной позиции — уровни RQ от $l=1$ до $l=L$.

Таким образом, последовательность выглядит как: $c_{1,1}^{(1)}, c_{1,1}^{(2)}, \ldots, c_{1,1}^{(L)}, c_{1,2}^{(1)}, \ldots, c_{h,w}^{(L)}$ — сначала все уровни первого токена, затем все уровни второго, и т. д.

### 6.2. Модель

RQTRAN — стандартный decoder-only трансформер с causal attention mask. На каждом шаге предсказывается следующий код: $p(c_t \mid c_1, \ldots, c_{t-1})$. Каждый код из словаря размером $K$ имеет свой learned embedding; суммарный размер входного vocabulary — $L \times K$ (у каждого уровня свои embeddings, что позволяет модели различать «код 5 уровня 1» и «код 5 уровня 3»).

### 6.3. Decoding стратегии

При генерации авторы предлагают две стратегии:

- **Sequential (token-by-token)** : строго последовательная генерация всех $h \cdot w \cdot L$ кодов. Максимальное качество, но медленно.
- **Stage-by-stage (параллельная)** : сначала авторегрессивно генерируются все коды уровня 1 ($h \times w$ кодов), затем все коды уровня 2 параллельно (условно независимо от уровня 1), и т. д. Ускорение в $\sim L$ раз при небольшой потере качества.

## 7. Иерархичность как ключевое свойство

Иерархичность — не побочный эффект, а фундаментальное свойство RQ, следующее из самой конструкции. Формально:

Пусть два item'а $z^{(A)}$ и $z^{(B)}$ имеют одинаковый код первого уровня: $c_1^{(A)} = c_1^{(B)} = c_1$. Это означает, что ближайший entry в $\mathcal{C}^{(1)}$ у них один и тот же, т. е. $e_{c_1}^{(1)}$ — лучшее одно-entry приближение обоих векторов. Следовательно, оба вектора лежат в одной «ячейке Вороного» первого уровня. Геометрически они близки в пространстве первого уровня — они семантически похожи на грубом уровне.

Значение для рекомендаций:

- Shared prefix $(c_1, c_2, \ldots, c_k)$ в Semantic ID = семантическая близость на уровне детализации $k$.
- $c_1$ соответствует грубой категории (в изображениях — «собака» vs «автомобиль»; в e-commerce — «электроника» vs «одежда»).
- $c_2$ уточняет (порода собаки, подкатегория электроники).
- $c_L$ различает конкретные item'ы внутри одной мелкой подкатегории.

Это иерархическое свойство делает RQ-VAE особенно привлекательным для авторегрессивного рекомендателя: сгенерировав первые несколько кодов, модель уже «определилась с категорией» — дальнейшие коды лишь уточняют выбор.

## 8. Связь с рекомендательными системами: почему TIGER взял RQ-VAE

TIGER (Rajput et al., 2023) адаптирует RQ-VAE для создания Semantic ID для items. Соответствие компонентов:

<div class="table-scroll">
<table>
<thead>
<tr>
<th>RQ-VAE (изображения)</th>
<th>TIGER (рекомендации)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Пиксели изображения $x$</td>
<td>Item content embedding (768-dim вектор от SentenceT5, обученного на текстовом описании item'а)</td>
</tr>
<tr>
<td>Encoder $E(x)$ — свёрточная сеть</td>
<td>Уже готовый embedding, без доп. encoder'а (item уже в embedding space)</td>
</tr>
<tr>
<td>Spatial feature map $h \times w$ токенов, каждый — $L$ кодов</td>
<td>Каждый item — $L$ кодов (нет spatial dimension, один «токен» на item)</td>
</tr>
<tr>
<td>Код: $(c_1^{ij}, \ldots, c_L^{ij})$ для пространственного токена $(i,j)$</td>
<td>Semantic ID: $(c_1, c_2, \ldots, c_L)$ для item'а</td>
</tr>
<tr>
<td>RQTRAN (GPT-style) для генерации изображений</td>
<td>T5 (encoder-decoder) для генерации Semantic ID следующего item'а по истории пользователя</td>
</tr>
<tr>
<td>Vocabulary: $L \times K$ токенов</td>
<td>Vocabulary: special tokens для каждого $(l, k)$ pair</td>
</tr>
<tr>
<td>Качество: FID, IS</td>
<td>Качество: Recall@K, NDCG@K</td>
</tr>
<tr>
<td>Reconstruction objective</td>
<td>Reconstruction objective (тот же VQ loss, при необходимости с collaborative regularization — LETTER)</td>
</tr>
<tr>
<td>Beam search для генерации</td>
<td>Constrained beam search: только валидные SID кортежи (те, что реально присутствуют в каталоге)</td>
</tr>
</tbody>
</table>
</div>

### 8.1. Ключевые адаптации в TIGER

- **Нет spatial dimension.** Изображение — это 2D-структура $h \times w$ токенов. Item — это один вектор. Поэтому TIGER использует RQ с одним «spatial» токеном, то есть чистый RQ без пространственного измерения.
- **Content-based encoding.** Вместо обучения encoder с нуля TIGER использует предобученный SentenceT5, что позволяет SID отражать семантику текстового описания item'а.
- **Constrained beam search.** При генерации следующего SID T5 перебирает только prefix-валидные последовательности, гарантируя, что сгенерированный SID соответствует существующему item'у в каталоге.
- **Generalization к новым items.** Новый item (cold-start) получает SID через прямой проход RQ-VAE по его content embedding — без переобучения. Это преимущество перед random ID-based подходами.

## 9. Эксперименты в оригинальной статье

### 9.1. Datasets

- **ImageNet** (256×256): стандартный бенчмарк для unconditional и class-conditional генерации.
- **CC-3M (Conceptual Captions)** : 3M пар изображение–текст, text-conditional генерация.
- **FFHQ** : 70K лиц высокого разрешения, unconditional генерация.

### 9.2. Метрики

- **FID (Fréchet Inception Distance)** : расстояние между распределением реальных и сгенерированных изображений в feature space InceptionNet. Меньше — лучше.
- **IS (Inception Score)** : оценка чёткости и diversity сгенерированных изображений. Больше — лучше.
- **Reconstruction rFID** : FID после encode → decode (без генерации). Измеряет качество RQ-VAE как компрессора.

### 9.3. Ключевые результаты RQ-VAE

RQ-VAE с $L=4$, $K=1024$ (общий размер: $4 \times 1024 = 4096$ codebook entries) достигает значительно лучшего rFID на ImageNet 256×256 по сравнению с flat VQ-VAE с тем же числом tokens на изображение. Критически важно: RQ-VAE требует меньшего числа spatial tokens для эквивалентного качества, что сокращает длину последовательности для RQTRAN.

<div class="table-scroll">
<table>
<thead>
<tr>
<th>Модель</th>
<th>Spatial tokens</th>
<th>Codebook size</th>
<th>rFID (IN 256×256)</th>
</tr>
</thead>
<tbody>
<tr>
<td>VQ-VAE (flat)</td>
<td>256</td>
<td>K=16384</td>
<td>~1.5</td>
</tr>
<tr>
<td>RQ-VAE (L=4)</td>
<td>64 (×4 levels)</td>
<td>K=1024 per level</td>
<td>&lt;1.0</td>
</tr>
<tr>
<td>RQ-VAE (L=8)</td>
<td>64 (×8 levels)</td>
<td>K=1024 per level</td>
<td>&lt;0.5</td>
</tr>
</tbody>
</table>
</div>

### 9.4. Ключевые результаты RQTRAN

На ImageNet 256×256 class-conditional генерации RQTRAN достигает FID ≈ 3.8 (top-k sampling), что сопоставимо с VQVAE-2 (FID ≈ 2.5 с rejection sampling, что некорректно сравнимо) и лучше оригинального DALL-E (FID ≈ 28.0). При этом RQTRAN использует существенно меньше tokens на изображение (64 spatial × 4 levels = 256), чем DALL-E (1024 tokens), то есть обучается быстрее при сопоставимом качестве.

## 10. Что нужно понять для работы с Semantic IDs

### 10.1. Codebook Collapse

**Что это:** при обучении encoder начинает отображать все входные векторы в маленькое подмножество codebook entries. Остальные entries («мёртвые entries») никогда не выбираются и не обновляются — их градиенты равны нулю. В результате эффективный размер словаря $K_{\text{eff}} \ll K$, и информационная ёмкость кодирования падает.

**Признаки:** при логировании обучения — низкий «codebook usage» (доля используемых entries). В TIGER/LETTER — low diversity в SID, большинство items получают одинаковые коды.

**Как RQ-VAE борется:**

- K-means инициализация: entries инициализируются кластерными центрами реальных encoder outputs → все entries изначально «поближе» к данным.
- EMA update: вместо SGD обновляются как скользящие средние назначенных векторов. Entries без назначений обновляются медленно и могут быть «перезапущены» (reset to random encoder output).
- Небольшой $K$ (1024 вместо 16K+): при меньшем $K$ каждый entry получает больше assignments, устойчивее обновляется.

### 10.2. Роль commitment coefficient $\beta$

$\beta = 0.25$ — баланс между двумя силами:

- При $\beta \to 0$: encoder свободен «блуждать» в пространстве, игнорируя codebook. Codebook не может «угнаться» за нестабильным encoder → collapse.
- При $\beta \to \infty$: encoder жёстко зафиксирован у ближайших entries → underfitting, encoder не обучается.
- $\beta = 0.25$: умеренное давление на encoder, достаточное для стабильности, но не мешающее обучению.

### 10.3. Влияние $L$ и $K$ на quality-capacity tradeoff

<div class="table-scroll">
<table>
<thead>
<tr>
<th>Параметр</th>
<th>Увеличение</th>
<th>Эффект</th>
<th>Ограничения</th>
</tr>
</thead>
<tbody>
<tr>
<td>$L$ (глубина)</td>
<td>$L \uparrow$</td>
<td>Лучше reconstruction quality; больше информации; длиннее SID → медленнее генерация</td>
<td>Практически $L &gt; 4{-}8$ уровней почти не улучшают качество; остаток $r_L$ становится шумом</td>
</tr>
<tr>
<td>$K$ (размер codebook)</td>
<td>$K \uparrow$</td>
<td>Больше различимых codes на уровень; тоньше квантование</td>
<td>Риск codebook collapse; больше памяти; труднее обучить все entries</td>
</tr>
<tr>
<td>$d$ (размерность embedding)</td>
<td>$d \uparrow$</td>
<td>Богаче пространство; лучше nearest-neighbour поиск</td>
<td>Больше параметров; медленнее nearest-neighbour</td>
</tr>
</tbody>
</table>
</div>

В TIGER используется $L \in \{4, 8\}$ и $K = 256$ (меньше, чем в оригинальной статье, из-за меньшей размерности input — 768 vs тысячи для изображений).

### 10.4. Почему SID иерархичен: shared prefix = shared semantics

Формально: если два item'а $a$ и $b$ имеют SID $(c_1, c_2, \ldots)$ и $(c_1, c_2', \ldots)$ (общий только $c_1$), то они оба попали в одну ячейку Вороного первого уровня. Это значит, что $e_{c_1}^{(1)}$ — лучший одно-entry аппроксиматор обоих embedding'ов. Геометрически: оба embedding'а ближе к $e_{c_1}^{(1)}$, чем к любому другому entry первого уровня. Следовательно, они семантически похожи на уровне грубой структуры.

Чем длиннее общий prefix, тем выше семантическая близость. Это свойство используется в constrained beam search TIGER'а: первые коды SID задают «категорию», и модель эффективно разбивает каталог на иерархические кластеры.

### 10.5. Flat RQ vs. Deep-to-Fine Hierarchical

- **Flat RQ (TIGER, LETTER)** : используется только один spatial token на item (нет spatial dimension). Все $L$ кодов кодируют один embedding вектор. Иерархия — только по уровням $l = 1, \ldots, L$.
- **Deep-to-Fine Hierarchical (оригинальная статья)** : для изображений есть spatial dimension ($h \times w$ токенов), каждый имеет $L$ уровней. Есть два измерения иерархии: пространственное и уровневое.

Для рекомендаций релевантно только первое (flat), поскольку item — это точка в embedding space, а не 2D-объект.

## 11. Ограничения RQ-VAE

1. **Нестабильность обучения.** Несмотря на k-means init и EMA update, codebook collapse остаётся реальной угрозой, особенно при больших $K$ или малых батчах. В рекомендательном контексте это проявляется как poor diversity SID — большинство items получают одинаковые коды, что уничтожает пользу авторегрессивной генерации.
1. **Насыщение при больших $L$.** Уровни после $l \approx 4{-}8$ кодируют лишь шум. Остаток $r_l$ уменьшается, но не обязательно становится семантически значимым — он просто всё меньше. В рекомендациях избыточные уровни увеличивают длину SID без улучшения semantics.
1. **Reconstruction objective ≠ retrieval objective.** RQ-VAE минимизирует $\|z - \tilde{z}\|^2$, что означает геометрическую близость в embedding space. Но для рекомендаций важна не геометрия embedding space, а semantics релевантности для пользователей. Collaborative сигналы (кто покупал вместе) не отражаются в reconstruction loss. Это именно то, что LETTER, CoST и DIGER пытаются исправить.
1. **Catalog drift.** После того как RQ-VAE обучен и codebook'и зафиксированы, новые items получают SID через прямой проход (без переобучения). Но если распределение item embedding'ов со временем сдвигается (новые категории, изменение стиля описаний), фиксированный codebook может стать неоптимальным. Периодическое переобучение требуется, но это нарушает стабильность SID.
1. **SID коллизии.** Два семантически разных item'а могут получить одинаковый SID, если их embedding'и попадают в одну ячейку Вороного на всех $L$ уровнях. При малом $K$ или большом каталоге коллизии часты — это проблема, которую решает ASI (Beyond Static Collision Handling).

## 12. Связь с другими работами

<div class="table-scroll">
<table>
<thead>
<tr>
<th>Работа</th>
<th>Отношение к RQ-VAE</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>VQ-VAE</strong> (van den Oord et al., NeurIPS 2017)</td>
<td>Прямой предшественник. Одноуровневое квантование, straight-through estimator. RQ-VAE — его многоуровневое расширение.</td>
</tr>
<tr>
<td><strong>VQVAE-2</strong> (Razavi et al., NeurIPS 2019)</td>
<td>Иерархический VQ-VAE с несколькими resolution levels (top/bottom), не sequential residual quantization. EMA update взят оттуда.</td>
</tr>
<tr>
<td><strong>VQGAN</strong> (Esser et al., CVPR 2021)</td>
<td>Добавляет perceptual + adversarial loss к VQ-VAE. RQ-VAE использует аналогичные losses для улучшения качества изображений.</td>
</tr>
<tr>
<td><strong>SoundStream</strong> (Zeghidour et al., ICLR 2022)</td>
<td>Параллельное применение RQ для аудио-кодеков (не изображений). Независимо пришли к той же архитектуре для audio compression.</td>
</tr>
<tr>
<td><strong>TIGER</strong> (Rajput et al., NeurIPS 2023)</td>
<td>Первая адаптация RQ-VAE для recommendation Semantic IDs. Content embeddings (SentenceT5) → RQ-VAE → Semantic ID. T5 как авторегрессивный рекомендатель.</td>
</tr>
<tr>
<td><strong>LETTER</strong></td>
<td>Добавляет collaborative regularization в training objective RQ-VAE: SID для item'ов с похожими interaction паттернами должны иметь общий prefix. Исправляет проблему #3 (reconstruction ≠ retrieval).</td>
</tr>
<tr>
<td><strong>CoST</strong></td>
<td>Заменяет reconstruction objective в RQ-VAE на contrastive objective: два augmented view одного item должны иметь схожие SID. Также исправляет проблему #3.</td>
</tr>
<tr>
<td><strong>DIGER</strong></td>
<td>Делает RQ differentiable (через Gumbel-softmax вместо argmin) для возможности обучения SID end-to-end с downstream recommendation loss. Устраняет разрыв между обучением RQ-VAE и генерацией рекомендаций.</td>
</tr>
<tr>
<td><strong>ASI</strong></td>
<td>Решает проблему коллизий SID (проблема #5): адаптивно добавляет уровни RQ для items, испытывающих коллизии, без полного переобучения.</td>
</tr>
<tr>
<td><strong>Encodec</strong> (Défossez et al., TMLR 2023)</td>
<td>Нейронный аудио-кодек, также использующий RQ (аналогично SoundStream). Показывает широту применения RQ за пределами изображений.</td>
</tr>
</tbody>
</table>
</div>

---

## Краткое резюме для исследователей рекомендательных систем

> RQ-VAE — это метод превращения непрерывного вектора в иерархический дискретный код $(c_1, \ldots, c_L)$ через последовательное квантование остатков. Каждый уровень кодирует то, что не смог закодировать предыдущий. Результирующий код иерархичен по семантике: общий prefix означает семантическую схожесть. TIGER взял именно эту идею и применил её к item content embedding'ам, превратив проблему рекомендаций в задачу авторегрессивной генерации Semantic ID. Все последующие работы (LETTER, CoST, DIGER, ASI) улучшают отдельные аспекты RQ-VAE применительно к рекомендациям, но фундаментальная архитектура остаётся той, что описана в этой статье.
