---
title: "Unleash the Potential of Long Semantic IDs for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "unleash_the_potential_of_long_semantic_ids_for_generative_recommendation_summary"
catalogId: "paper-unleash_the_potential_of_long_semantic_ids_for_summary"
paperUrl: "https://arxiv.org/abs/2602.13573"
---
> **Авторы:** Ming Xia, Guoxin Ma, Zhiqin Zhou, Dongmin Huang (первые трое - equal contribution).
>
> **Аффилиации:** Southern University of Science and Technology; Xi'an Jiaotong University; Nanjing University.
>
> **Источник:** arXiv:2602.13573. Саммари написано по **v2 от 2026-08-02** (оформление AAAI 2027, статус "under review"); v1 вышла 2026-02-14. Между версиями заметные отличия - они собраны в разделе 12. Кода в открытом доступе на момент написания нет.

## 1. Коротко: о чем статья

Статья про **ACERec** (Adaptive Compression for Efficient Recommendation) - generative recommender, который пытается снять конфликт между *выразительностью* длинных Semantic IDs и *стоимостью* их обработки.

Конфликт выглядит так. RQ-based модели вроде [TIGER](../generative_retrieval/tiger_recommender_systems_with_generative_retrieval_summary.html) описывают item 3-4 токенами: это дешево для autoregressive decoding, но мало для описания item'а. OPQ-based модели вроде [RPG](generating_long_semantic_ids_in_parallel_for_recommendation_summary.html) описывают item 16-64 токенами и предсказывают их параллельно, но на входе вынуждены схлопывать все токены item'а в один вектор через mean pooling - иначе история пользователя из 50 item'ов превращается в 1600 токенов. Получается парадокс: tokenizer аккуратно раскладывает item по 32 подпространствам, а sequence model видит их среднее.

ACERec предлагает **развязать гранулярность токенизации и длину входа recommender'а**:

1. Item кодируется длинным OPQ-кодом ($m=32$ токена).
2. **Attentive Token Merger (ATM)** через cross-attention сжимает 32 токена в $k=4$ латентных токена. Queries генерируются из самого item'а, поэтому сжатие content-adaptive, а не фиксированное. Дополнительный reconstruction loss заставляет латенты сохранять информацию обо всех 32 токенах.
3. После каждого item'а в последовательность добавляется **Intent Token** - отдельная позиция, которая через step-wise causal attention собирает историю и служит точкой предсказания следующего item'а.
4. Обучение **dual-granularity**: token-level multi-token prediction всех 32 цифр целевого item'а (MTP) плюс item-level contrastive alignment между Intent Token и целевым item'ом (ISA).
5. Inference без beam search: одна прямая прогонка дает $m$ распределений по codebook'ам, затем каждый item каталога получает score как сумму log-вероятностей своих 32 цифр.

Заявленный результат: на девяти Amazon-датасетах ACERec лучше всех девяти baseline'ов, в среднем **+12.92% NDCG@10 и +7.49% Recall@10** к лучшему baseline на каждом датасете, примерно в 2.2 раза выше throughput, чем у TIGER/ActionPiece/RPG, и заметно лучше на cold-start item'ах.

Главная оговорка, к которой мы вернемся в разделах 8 и 14: по ablation самих авторов около двух третей прироста NDCG@10 дает Intent Token, а не работа с длинными ID. То есть статья скорее про удачную комбинацию "длинные OPQ-коды + хороший prediction anchor + item-level loss", чем про чистый эффект длины SID.

<figure class="paper-figure">
  <img src="../../assets/acerec/paradigms.png" alt="Three semantic ID paradigms: residual quantization with autoregressive decoding, product quantization with pooling, product quantization with token merging">
  <figcaption>Рисунок 1 (Figure 1 статьи). Три парадигмы. (a) RQ: короткие иерархические ID, вход без сжатия, но медленный autoregressive decoding. (b) OPQ + pooling: длинные ID и параллельное предсказание, но lossy-сжатие на входе. (c) ACERec: те же длинные ID, но сжатие делает обучаемый Token Merger.</figcaption>
</figure>

## 2. Контекст: granularity-efficiency dilemma

### 2.1. Зачем вообще Semantic IDs

Классический sequential recommender (SASRec, BERT4Rec) хранит отдельный embedding на каждый item ID. У этого две проблемы: таблица растет линейно с каталогом, а item'ы с похожими атрибутами никак не связаны - редкий item почти ничего не получает от популярного соседа. Semantic IDs заменяют атомарный ID на короткую последовательность токенов из общего словаря. Похожие item'ы делят токены, поэтому знания переносятся между ними, а размер словаря не зависит от размера каталога.

Дальше все упирается в то, **как** строить эти токены и **сколько** их.

### 2.2. RQ: короткие иерархические ID

Residual Quantization (RQ-VAE в TIGER, ETEGRec и др.) квантует embedding по уровням: первый codebook приближает сам вектор, второй - остаток, третий - остаток остатка. Получается иерархия "от грубого к тонкому".

Авторы выделяют две слабости:

- **последовательная зависимость кодов.** Токен уровня $l$ осмыслен только при известных предыдущих, поэтому декодировать приходится авторегрессивно, с beam search. Длина ID напрямую умножает latency, и на практике ID держат в 3-4 токена;
- **убывающая информативность.** Поздние уровни кодируют все более слабые остатки, и модель плохо их использует. Просто удлинить RQ-код - неэффективно.

### 2.3. OPQ: длинные параллельные ID

Optimized Product Quantization разбивает embedding на $m$ подпространств (после обучаемого ортогонального поворота) и квантует каждое своим codebook'ом независимо. Два следствия:

- токены равноправны и несут взаимодополняющую информацию, а не остатки друг друга;
- при условной независимости цифр их можно предсказывать параллельно: $P(\mathbf{c} \mid \mathcal{S}_u) \approx \prod_{k=1}^{m} P(c_k \mid \mathcal{S}_u)$.

Это путь RPG (и раньше VQ-Rec): ID длиной до 64 токенов, $m$ prediction heads, без авторегрессии.

### 2.4. Где ломается OPQ-подход

Проблема на входе. Если подать историю из $L$ item'ов как $L \times m$ токенов, при $L=50$, $m=32$ получится 1600 позиций и квадратичный attention по ним. Поэтому VQ-Rec и RPG усредняют embeddings всех токенов item'а в один вектор.

Авторы называют это **semantic blurring**: mean pooling одинаково взвешивает все подпространства у всех item'ов, и то, ради чего OPQ разносил атрибуты по подпространствам, смешивается обратно. Вопрос статьи сформулирован так: можно ли сохранить выразительность длинных ID и при этом оставить эффективными и вход, и выход модели?

<div class="table-scroll">
<table>
<thead><tr><th>Парадигма</th><th>Длина ID</th><th>Что видит recommender на входе</th><th>Decoding</th><th>Слабое место</th></tr></thead>
<tbody>
<tr><td>RQ (TIGER, ETEGRec)</td><td>3-4</td><td>все токены, $L \cdot m$ позиций</td><td>autoregressive + beam search</td><td>мало емкости, latency растет с длиной ID</td></tr>
<tr><td>OPQ + mean pooling (VQ-Rec, RPG)</td><td>16-64</td><td>1 вектор на item</td><td>параллельный MTP + graph decoding</td><td>усреднение стирает структуру подпространств</td></tr>
<tr><td>OPQ + ATM (ACERec)</td><td>32</td><td>$k=4$ латента + Intent Token на item</td><td>параллельный MTP + полный scoring каталога</td><td>полный проход по каталогу на inference</td></tr>
</tbody>
</table>
</div>

## 3. Постановка задачи

Стандартный next-item prediction: по истории $\mathcal{S}_u = \{i_1, \dots, i_L\}$ предсказать $i_{L+1}$. Каждый item представлен кортежем $\mathbf{c}_i = (c_{i,1}, \dots, c_{i,m})$, и задача переформулируется как генерация кортежа целевого item'а.

ACERec относится к семейству **parallel decoding**: все $m$ цифр предсказываются одновременно из одного состояния. Строго говоря, это уже не "генерация" в смысле LLM - авторегрессии по цифрам нет, и, как будет видно в разделе 4.5, нет и декодирования последовательности: модель выдает $m$ categorical-распределений, по которым скорится готовый каталог.

## 4. Метод

<figure class="paper-figure">
  <img src="../../assets/acerec/framework.png" alt="ACERec architecture: embedding table, attentive token merger, recommender with intent token, dual-granularity optimization and top-k retrieval">
  <figcaption>Рисунок 2 (Figure 2 статьи). Слева - поток кодирования: m-digit ID → Embedding Table → ATM → k латентов + Intent Token → Recommender. В центре - устройство ATM: queries из item embedding, cross-attention к токенам, reconstruction loss. Справа - два уровня обучения (MTP и ISA) и inference через сопоставление предсказанных распределений с кодами item'ов.</figcaption>
</figure>

### 4.1. Токенизация через OPQ

Текстовые метаданные item'а кодируются `sentence-t5-base` (768-мерный embedding), затем FAISS OPQ дает $m=32$ цифры, у каждой свой codebook размера $M$ (256 для Amazon-2014, 512 для Amazon-2023). При $m=32$ на одно подпространство приходится 24 измерения.

Каждая цифра $c_k$ отображается в обучаемый embedding $\mathbf{e}_k \in \mathbb{R}^d$; item описывается матрицей $\mathbf{E}_i \in \mathbb{R}^{m \times d}$. Всего в словаре $m \times M = 8192$ токена (или 16384 для датасетов 2023 года) - при $d=448$ это 3.7M / 7.3M параметров независимо от размера каталога. Токенизация **заморожена**: OPQ считается один раз до обучения recommender'а.

### 4.2. Attentive Token Merger

ATM - это один слой cross-attention в духе Perceiver / Q-Former, но с важной деталью: queries не общие обучаемые векторы, а функция от самого item'а.

**Шаг 1. Content-adaptive queries.** Сначала токены агрегируются в item-level summary, затем summary проецируется в $k$ queries:

$$
\mathbf{s}_i = f_s(\mathbf{e}_1, \dots, \mathbf{e}_m), \qquad \mathbf{Q}_i = f_q(\mathbf{s}_i) \in \mathbb{R}^{k \times d}.
$$

Вектор $\mathbf{s}_i$ дальше используется еще дважды: как инициализация Intent Token и как целевое представление item'а в ISA loss. Конкретный вид $f_s$ и $f_q$ в статье не раскрыт (названы просто "projection" и "projector").

**Шаг 2. Merging.** К токенам добавляются обучаемые позиционные embeddings $\mathbf{P} \in \mathbb{R}^{m \times d}$ - они кодируют *номер подпространства*, потому что одна и та же цифра "57" в 3-м и 20-м codebook'ах означает разное:

$$
\tilde{\mathbf{E}}_i = \mathbf{E}_i + \mathbf{P}, \qquad
\mathbf{Z}_i = f_{out}\big(f_{attn}(\mathbf{Q}_i, \tilde{\mathbf{E}}_i, \tilde{\mathbf{E}}_i)\big) \in \mathbb{R}^{k \times d}, \quad k \ll m.
$$

Здесь $f_{attn}$ - multi-head cross-attention, $f_{out}$ - MLP с layer norm. Каждый из $k$ латентов может смотреть на свой набор подпространств, причем для разных item'ов - на разный.

**Шаг 3. Reconstruction.** Чтобы сжатие не выбрасывало информацию, легкий upsampling-MLP восстанавливает из $k$ латентов все $m$ исходных token embeddings, а loss штрафует косинусное расстояние:

$$
\hat{\mathbf{E}}_i = \text{Decoder}(\mathbf{Z}_i) \in \mathbb{R}^{m \times d}, \qquad
\mathcal{L}_{\mathrm{Recon}} = 1 - \frac{1}{m} \sum_{r=1}^{m} \frac{\hat{\mathbf{e}}_{i,r} \cdot \mathbf{e}_{i,r}}{\|\hat{\mathbf{e}}_{i,r}\| \, \|\mathbf{e}_{i,r}\|}.
$$

По сути ATM + Decoder - это маленький autoencoder поверх token embeddings, обучаемый совместно с recommender'ом. Декодер нужен только при обучении. Этого компонента не было в v1 - он добавлен в v2.

### 4.3. Intent Token и step-wise causal attention

Латенты $\mathbf{Z}_t$ описывают *item*. Но для предсказания нужен вектор, описывающий *пользователя на шаге* $t$. В RPG эту роль играет выход Transformer'а на позиции последнего item'а; у ACERec на один item приходится $k$ позиций, и непонятно, с какой из них предсказывать.

Решение - добавить к каждому item'у отдельную позицию:

$$
\tilde{\mathbf{Z}}_t = [\mathbf{Z}_t; \mathbf{h}_t] \in \mathbb{R}^{(k+1) \times d}, \qquad \mathbf{H} = [\tilde{\mathbf{Z}}_1, \dots, \tilde{\mathbf{Z}}_L].
$$

Intent Token $\mathbf{h}_t$ **инициализируется summary текущего item'а** $\mathbf{s}_t$ - это не общий [CLS]-подобный вектор, а контентная позиция, своя на каждом шаге.

Маска внимания блочная:

- токены шага $t$ не видят будущие блоки $\tilde{\mathbf{Z}}_{>t}$;
- внутри блока латенты $\mathbf{Z}_t$ видят друг друга (двунаправленно);
- $\mathbf{h}_t$ видит латенты своего блока и все предыдущие блоки.

В итоге $\mathbf{h}_t$ на выходе - это "состояние пользователя после $t$ взаимодействий", и именно с него делается предсказание. Длина входа Transformer'а - $L(k+1)$: при $L=50$, $k=4$ это 250 позиций вместо 1600 у сырых токенов и 50 у RPG.

### 4.4. Dual-granularity objective

**Token level - MTP.** Финальное состояние $\mathbf{h}_i$ нормализуется и проецируется $m$ отдельными heads в $m$ подпространственных представлений $\mathbf{h}_i^{(k)}$. Для каждой цифры - softmax по своему codebook'у:

$$
P(c_k = v \mid \mathbf{h}_i) = \frac{\exp(\mathbf{h}_i^{(k)\top} \mathbf{e}_v^{(k)} / \gamma)}{\sum_{v'=1}^{M} \exp(\mathbf{h}_i^{(k)\top} \mathbf{e}_{v'}^{(k)} / \gamma)}, \qquad
\mathcal{L}_{\mathrm{MTP}} = -\frac{1}{m} \sum_{k=1}^{m} \log P(c_{tgt,k} \mid \mathbf{h}_i).
$$

Температура $\gamma = 0.03$. Это тот же objective, что у RPG: выходные logits считаются против тех же token embeddings, что используются на входе.

**Item level - Intent-Semantic Alignment (ISA).** MTP учит угадывать каждую цифру по отдельности, но не требует, чтобы intent-вектор был близок к целевому item'у *как целому*. ISA добавляет in-batch contrastive loss между $\mathbf{h}_i$ и summary следующего item'а $\mathbf{s}_{i+1}$:

$$
\phi(\mathbf{h}_i, \mathbf{s}_j) = \frac{\text{sim}(\mathbf{h}_i, \mathbf{s}_j)}{\tau} - b_j, \qquad
b_j = \beta \cdot \frac{\log f_j}{\max_{i' \in \mathcal{I}} \log f_{i'}},
$$

$$
\mathcal{L}_{\mathrm{ISA}} = -\log \frac{\exp \phi(\mathbf{h}_i, \mathbf{s}_{i+1})}{\sum_{j \in \mathcal{B}} \exp \phi(\mathbf{h}_i, \mathbf{s}_j)}.
$$

Поправка $b_j$ - стандартная logQ-style коррекция sampling bias: популярные item'ы чаще оказываются in-batch негативами, и без коррекции модель их излишне штрафует. Здесь она нормирована на максимум лог-частоты и масштабируется $\beta = 0.02$; $\tau = 0.07$.

Полезно заметить, что ISA - это по сути two-tower retrieval loss, встроенный в generative модель: user tower - Transformer с Intent Token, item tower - $f_s$ поверх token embeddings. На inference эта "dense-голова" не используется, она работает только как регуляризатор.

**Итоговый loss:**

$$
\mathcal{L} = \mathcal{L}_{\mathrm{MTP}} + \lambda \mathcal{L}_{\mathrm{ISA}} + \alpha \mathcal{L}_{\mathrm{Recon}}, \qquad \lambda \in \{0.1, 0.15, 0.2\}, \quad \alpha \in \{0.03, 0.05\}.
$$

### 4.5. Inference: holistic candidate scoring

Beam search не нужен. Inference состоит из двух векторизованных шагов.

**Parallel subspace matching.** Из финального $\mathbf{h}_i$ считается матрица log-вероятностей $\mathbf{P} \in \mathbb{R}^{m \times M}$:

$$
\mathbf{P}[k, v] = \log \frac{\exp(\mathbf{h}_i^{(k)} \cdot \mathbf{e}_v^{(k)} / \gamma)}{\sum_{v'=1}^{M} \exp(\mathbf{h}_i^{(k)} \cdot \mathbf{e}_{v'}^{(k)} / \gamma)}.
$$

Стоимость этого шага - $O(mMd)$, от размера каталога не зависит.

**Vectorized score gathering.** Коды всех item'ов предпосчитаны, поэтому score item'а - просто сумма $m$ значений из таблицы:

$$
\text{Score}(j) = \sum_{k=1}^{m} \mathbf{P}[k, c_{j,k}].
$$

Затем top-$K$ по всем $j \in \mathcal{I}$.

Это один в один **asymmetric distance computation (ADC)** из литературы по Product Quantization: для запроса строится lookup-таблица $m \times M$, и расстояние до каждого вектора базы - сумма $m$ обращений к таблице. Отличие только в том, что "расстояния" здесь - выученные log-вероятности. Отсюда же и главное ограничение: ADC без инвертированного индекса - это **линейный проход по каталогу**, $O(|\mathcal{I}| \cdot m)$ на запрос. Это в $d/m = 14$ раз дешевле, чем dense scoring с 448-мерными векторами, но сублинейным не является. Авторы формулируют осторожно ("сложность первого шага не зависит от $|\mathcal{I}|$"), однако второй шаг от каталога зависит. RPG как раз избегал полного прохода через graph-constrained decoding; ACERec от этого отказался и выиграл в простоте (ноль inference-гиперпараметров), но проверен только на каталогах до 64K item'ов.

Плюс такого scoring'а: модель по построению не может выдать невалидный код - ранжируются только существующие item'ы.

### 4.6. Сложность

<div class="table-scroll">
<table>
<thead><tr><th>Модель</th><th>Длина входа</th><th>Self-attention</th><th>При $L=50$</th></tr></thead>
<tbody>
<tr><td>Сырые токены (TIGER-style)</td><td>$L \cdot m$</td><td>$O(L^2 m^2 d)$</td><td>1600 позиций при $m=32$</td></tr>
<tr><td>RPG (mean pooling)</td><td>$L$</td><td>$O(L^2 d)$</td><td>50 позиций</td></tr>
<tr><td>ACERec</td><td>$L(k+1)$</td><td>$O(L^2 k^2 d)$ + ATM $O(Lmkd)$</td><td>250 позиций при $k=4$</td></tr>
</tbody>
</table>
</div>

То есть attention у ACERec примерно в 41 раз дешевле, чем по сырым 32 токенам, но в 25 раз дороже, чем у RPG. (В статье для TIGER написано $O(L^2 n^2 d)$ - по смыслу это опечатка, $n$ должно быть $m$.)

## 5. Экспериментальный сетап

**Датасеты.** Девять категорий Amazon Reviews: семь из издания 2014 года и две из издания 2023 года. 5-core фильтрация по пользователям, leave-last-out split, история обрезается до 50 последних взаимодействий, full ranking по всему каталогу.

<div class="table-scroll">
<table>
<thead><tr><th>Датасет</th><th>Издание</th><th>Users</th><th>Items</th><th>Interactions</th><th>Avg. length</th></tr></thead>
<tbody>
<tr><td>Sports</td><td>2014</td><td>18,357</td><td>35,598</td><td>260,739</td><td>8.32</td></tr>
<tr><td>Beauty</td><td>2014</td><td>22,363</td><td>12,101</td><td>176,139</td><td>8.87</td></tr>
<tr><td>Toys</td><td>2014</td><td>19,412</td><td>11,924</td><td>148,185</td><td>8.63</td></tr>
<tr><td>CDs</td><td>2014</td><td>75,258</td><td>64,443</td><td>1,022,334</td><td>14.58</td></tr>
<tr><td>Baby</td><td>2014</td><td>19,446</td><td>7,051</td><td>160,792</td><td>8.27</td></tr>
<tr><td>Pets</td><td>2014</td><td>19,856</td><td>8,510</td><td>157,836</td><td>7.95</td></tr>
<tr><td>Office</td><td>2014</td><td>4,906</td><td>2,421</td><td>53,258</td><td>10.86</td></tr>
<tr><td>Science</td><td>2023</td><td>50,986</td><td>25,849</td><td>412,947</td><td>8.10</td></tr>
<tr><td>Instruments</td><td>2023</td><td>57,440</td><td>24,588</td><td>511,836</td><td>8.91</td></tr>
</tbody>
</table>
</div>

Все датасеты маленькие по индустриальным меркам: самый большой каталог - 64K item'ов, средняя длина истории 8-15.

**Baselines.** Пять ID-based: HGN, SASRec, S³-Rec, ICLRec, ELCRec (последние два - intent learning через кластеризацию и contrastive learning). Четыре SID-based: [TIGER](../generative_retrieval/tiger_recommender_systems_with_generative_retrieval_summary.html), [ETEGRec](etegrec_end_to_end_learnable_item_tokenization_summary.html), [ActionPiece](../sequential_session_long_history/actionpiece_contextual_action_tokenization_summary.html), [RPG](generating_long_semantic_ids_in_parallel_for_recommendation_summary.html). Для всех SID-моделей используется один encoder `sentence-t5-base`, чтобы разница не объяснялась качеством исходных embeddings.

Важная деталь: для HGN, SASRec, S³-Rec и TIGER на Sports/Beauty/Toys цифры **взяты из статьи TIGER**, а не воспроизведены. Остальное воспроизведено через официальный код или RecBole.

**ACERec.** Конфигурация выровнена с RPG: 2-слойный Transformer decoder, $d=448$, FFN 1024, 4 heads; $m=32$, compression ratio $r = m/k = 8$, то есть $k=4$. Одна RTX 6000 Ada (48GB). Двухэтапный тюнинг: сначала learning rate $\{0.003, 0.005\}$ × batch size $\{64, 256\}$ при фиксированных $\lambda=0.1$, $\alpha=0.03$, затем сетка по $\lambda$ и $\alpha$ - всего 9 запусков на датасет. Checkpoint выбирается по validation NDCG@10. Результаты усреднены по пяти seeds; стандартные отклонения и тесты значимости не приводятся.

## 6. Основные результаты

NDCG@10 по всем методам и датасетам (жирным - лучший результат, подчеркнут лучший baseline; последняя строка - прирост ACERec к лучшему baseline):

<div class="table-scroll">
<table>
<thead><tr><th>Метод</th><th>Sports</th><th>Beauty</th><th>Toys</th><th>CDs</th><th>Baby</th><th>Pets</th><th>Office</th><th>Science</th><th>Instruments</th></tr></thead>
<tbody>
<tr><td>HGN</td><td>0.0159</td><td>0.0266</td><td>0.0277</td><td>0.0220</td><td>0.0167</td><td>0.0272</td><td>0.0389</td><td>0.0169</td><td>0.0215</td></tr>
<tr><td>SASRec</td><td>0.0192</td><td>0.0318</td><td>0.0374</td><td>0.0263</td><td>0.0089</td><td>0.0184</td><td>0.0379</td><td>0.0080</td><td>0.0073</td></tr>
<tr><td>S³-Rec</td><td>0.0204</td><td>0.0327</td><td>0.0376</td><td>0.0182</td><td>0.0211</td><td>0.0318</td><td><u>0.0509</u></td><td>0.0215</td><td>0.0283</td></tr>
<tr><td>ICLRec</td><td>0.0228</td><td>0.0403</td><td>0.0472</td><td>0.0394</td><td>0.0211</td><td>0.0363</td><td>0.0440</td><td>0.0222</td><td>0.0292</td></tr>
<tr><td>ELCRec</td><td>0.0225</td><td>0.0423</td><td><u>0.0474</u></td><td>0.0376</td><td>0.0209</td><td>0.0341</td><td>0.0403</td><td>0.0219</td><td>0.0294</td></tr>
<tr><td>TIGER</td><td>0.0225</td><td>0.0384</td><td>0.0432</td><td><u>0.0411</u></td><td>0.0133</td><td>0.0247</td><td>0.0303</td><td>0.0190</td><td>0.0276</td></tr>
<tr><td>ETEGRec</td><td>0.0209</td><td>0.0350</td><td>0.0373</td><td>0.0357</td><td>0.0118</td><td>0.0306</td><td>0.0304</td><td>0.0224</td><td><u>0.0311</u></td></tr>
<tr><td>ActionPiece</td><td>0.0238</td><td>0.0412</td><td>0.0391</td><td>0.0395</td><td><u>0.0225</u></td><td>0.0343</td><td>0.0444</td><td>0.0222</td><td>0.0304</td></tr>
<tr><td>RPG</td><td><u>0.0242</u></td><td><u>0.0436</u></td><td>0.0454</td><td>0.0400</td><td>0.0220</td><td><u>0.0374</u></td><td>0.0490</td><td><u>0.0234</u></td><td>0.0307</td></tr>
<tr><td><strong>ACERec</strong></td><td><strong>0.0287</strong></td><td><strong>0.0513</strong></td><td><strong>0.0591</strong></td><td><strong>0.0451</strong></td><td><strong>0.0263</strong></td><td><strong>0.0407</strong></td><td><strong>0.0572</strong></td><td><strong>0.0244</strong></td><td><strong>0.0321</strong></td></tr>
<tr><td>Improv.</td><td>+18.60%</td><td>+17.66%</td><td>+24.68%</td><td>+9.73%</td><td>+16.89%</td><td>+8.82%</td><td>+12.38%</td><td>+4.27%</td><td>+3.22%</td></tr>
</tbody>
</table>
</div>

Recall@10 и сравнение с прямым предшественником - RPG:

<div class="table-scroll">
<table>
<thead><tr><th>Датасет</th><th>Лучший baseline R@10</th><th>ACERec R@10</th><th>Improv.</th><th>RPG R@10</th><th>ACERec vs RPG, R@10</th><th>ACERec vs RPG, N@10</th></tr></thead>
<tbody>
<tr><td>Sports</td><td>0.0441 (ActionPiece)</td><td>0.0494</td><td>+12.02%</td><td>0.0435</td><td>+13.6%</td><td>+18.6%</td></tr>
<tr><td>Beauty</td><td>0.0762 (ActionPiece)</td><td>0.0856</td><td>+12.34%</td><td>0.0757</td><td>+13.1%</td><td>+17.7%</td></tr>
<tr><td>Toys</td><td>0.0815 (ELCRec)</td><td>0.0940</td><td>+15.34%</td><td>0.0777</td><td>+21.0%</td><td>+30.2%</td></tr>
<tr><td>CDs</td><td>0.0748 (TIGER)</td><td>0.0800</td><td>+6.95%</td><td>0.0708</td><td>+13.0%</td><td>+12.7%</td></tr>
<tr><td>Baby</td><td>0.0424 (ActionPiece)</td><td>0.0462</td><td>+8.96%</td><td>0.0394</td><td>+17.3%</td><td>+19.5%</td></tr>
<tr><td>Pets</td><td>0.0674 (RPG)</td><td>0.0706</td><td>+4.75%</td><td>0.0674</td><td>+4.7%</td><td>+8.8%</td></tr>
<tr><td>Office</td><td>0.0989 (S³-Rec)</td><td>0.1036</td><td>+4.75%</td><td>0.0917</td><td>+13.0%</td><td>+16.7%</td></tr>
<tr><td>Science</td><td>0.0429 (RPG)</td><td>0.0439</td><td>+2.33%</td><td>0.0429</td><td>+2.3%</td><td>+4.3%</td></tr>
<tr><td>Instruments</td><td>0.0579 (ETEGRec)</td><td>0.0579</td><td>+0.00%</td><td>0.0557</td><td>+3.9%</td><td>+4.6%</td></tr>
</tbody>
</table>
</div>

Столбцы "vs RPG" пересчитаны из таблиц статьи; в среднем это +14.8% NDCG@10 и +11.3% Recall@10.

Что здесь видно:

- **ACERec лучший везде**, кроме ничьей с ETEGRec по Recall@10 на Instruments. Средние +12.92% NDCG@10 и +7.49% Recall@10 сходятся с таблицами.
- **Выигрыш очень неоднороден.** На классической тройке Sports/Beauty/Toys средний прирост NDCG@10 - около +20%, на CDs/Baby/Pets/Office - около +12%, на двух датасетах Amazon-2023 - меньше +4% (по Recall@10 - чуть больше +1%). Среднее по девяти датасетам в основном держится на первой группе. Без дисперсий по seeds разницу в 0.001 на Science и Instruments трудно считать надежной.
- **NDCG растет сильнее Recall**, а @5 - сильнее @10 (например, Toys: +29.15% N@5 против +15.34% R@10). Модель в первую очередь лучше ранжирует верх списка, а не расширяет покрытие. Это согласуется с тем, что главный вклад дают Intent Token и ISA - компоненты про "точное попадание", а не про coverage.
- **Сильные ID-based intent-модели конкурентны.** ELCRec - лучший baseline на Toys, S³-Rec - на Office. Авторы делают из этого справедливый вывод: богатое представление item'а само по себе не гарантирует лучших рекомендаций, важно, как оно превращается в user intent.
- **Сравнение с RPG - самое чистое.** Та же токенизация, тот же backbone и MTP; отличаются сжатие входа, prediction anchor, два дополнительных loss'а и способ inference.

## 7. Анализ токенизации

### 7.1. Масштабирование по длине ID

При фиксированном $r=8$ длина меняется $m \in \{8, 16, 32, 64\}$ (то есть $k = 1, 2, 4, 8$).

<figure class="paper-figure">
  <img src="../../assets/acerec/sid_length_scaling.png" alt="NDCG@10 versus semantic ID length for ACERec and RPG on Beauty and Toys">
  <figcaption>Рисунок 3 (Figure 3 статьи). NDCG@10 при разной длине SID. ACERec выше RPG при любой длине; обе модели имеют максимум около m=32 (RPG на Toys - уже при m=16), при m=64 качество падает.</figcaption>
</figure>

ACERec выигрывает у RPG на всех длинах, сильно растет от 8 к 16, достигает максимума на 32 и слегка проседает на 64. RPG на Toys начинает деградировать уже после 16. Авторы читают это как "ATM и reconstruction loss лучше утилизируют длинные ID".

Аккуратнее было бы сказать так: обе кривые имеют одинаковую форму с насыщением, и ACERec сдвигает ее вверх примерно на константу. Убедительного "чем длиннее, тем лучше" здесь нет - при $m=64$ (12 измерений на подпространство, 64 prediction heads) хуже обеим моделям.

### 7.2. Длинный ID со сжатием против короткого ID

Ключевой эксперимент для главного тезиса. Все три модели подают в recommender **по 4 токена на item**:

- short-digit OPQ: item сразу квантуется в 4 OPQ-цифры;
- ETEGRec: 3 RQ-токена + 1 conflict token;
- ACERec: 32 OPQ-цифры → ATM → 4 латента.

<figure class="paper-figure">
  <img src="../../assets/acerec/short_id_comparison.png" alt="Bar charts comparing ETEGRec, short-digit OPQ and ACERec on Beauty, Toys and Pets">
  <figcaption>Рисунок 4 (Figure 4 статьи). При одинаковой длине входа (4 токена на item) ACERec заметно лучше обеих коротких альтернатив на Beauty, Toys и Pets.</figcaption>
</figure>

Два наблюдения авторов:

- при жестком бюджете RQ лучше мелкого OPQ: ETEGRec выше short-digit OPQ в среднем на 24.88% Recall@10 и 12.05% NDCG@10. Это логично - 4 подпространства по 192 измерения с 256 центроидами дают очень грубое квантование, а RQ тратит тот же бюджет на последовательные уточнения;
- ACERec выше short-digit OPQ на 56.18% / 62.76% и выше ETEGRec на 25.44% / 46.01% (Recall@10 / NDCG@10).

Это действительно поддерживает тезис "важна гранулярность источника, а не длина входа". Но эксперимент не изолирует *входную* сторону: у ACERec длинный ID работает еще и на выходе - 32 prediction heads и scoring по 32 цифрам против 4. Часть выигрыша почти наверняка идет от более точного output space, а не от того, что латенты "дистиллированы из 32 токенов". Варианта "вход - 4 короткие цифры, выход - 32 цифры" в статье нет.

### 7.3. Что важнее: $m$ или $k$

<figure class="paper-figure">
  <img src="../../assets/acerec/m_k_heatmap.png" alt="Heatmaps of NDCG@10 over semantic ID length m and latent token size k on Beauty and Toys">
  <figcaption>Рисунок 5 (Figure 11 статьи). NDCG@10 в зависимости от длины SID m и числа латентов k. Строка m=32 лучше остальных при любом k; оптимум на обоих датасетах - (m=32, k=4).</figcaption>
</figure>

Вывод авторов: разрешение источника доминирует над емкостью латентов. Конфигурация $(m=32, k=4)$ лучше $(m=16, k=8)$: 0.0513 против 0.0456 на Beauty и 0.0591 против 0.0564 на Toys, хотя во втором случае латентов вдвое больше. Недостаток исходной гранулярности нельзя компенсировать числом латентов.

Отдельно по compression ratio ($r \in \{2, 4, 8, 16\}$ при $m=32$): кривые почти плоские. На Toys NDCG@10 меняется в пределах 0.058-0.059, на Beauty - 0.049-0.051, оптимум при $r=8$. Даже $k=2$ теряет немного. (В v1 это описывалось как "inverted-V", в v2 - как "stable".)

У этой плоскости есть и менее лестное прочтение: если качеству почти все равно, 2 латента или 16, значит, латенты несут сильно избыточную информацию, и "тонкая структура 32 подпространств" на входе используется слабо. Это согласуется с картинкой внимания в разделе 9.

## 8. Ablation: откуда на самом деле прирост

### 8.1. Кумулятивный ablation

Путь от RPG к ACERec, компоненты добавляются по одному:

<div class="table-scroll">
<table>
<thead><tr><th>Вариант</th><th>Toys R@10</th><th>Toys N@10</th><th>Baby R@10</th><th>Baby N@10</th><th>Шаг по N@10 (Toys)</th></tr></thead>
<tbody>
<tr><td>Base (RPG)</td><td>0.0777</td><td>0.0454</td><td>0.0394</td><td>0.0220</td><td>-</td></tr>
<tr><td>+ ATM</td><td>0.0802</td><td>0.0470</td><td>0.0406</td><td>0.0226</td><td>+3.5%</td></tr>
<tr><td>+ Intent Token</td><td>0.0869</td><td>0.0562</td><td>0.0415</td><td>0.0242</td><td>+19.6%</td></tr>
<tr><td>+ ISA loss</td><td>0.0916</td><td>0.0579</td><td>0.0427</td><td>0.0249</td><td>+3.0%</td></tr>
<tr><td>+ Recon loss (ACERec)</td><td>0.0940</td><td>0.0591</td><td>0.0462</td><td>0.0263</td><td>+2.1%</td></tr>
</tbody>
</table>
</div>

Если разложить полный прирост NDCG@10 на Toys (0.0454 → 0.0591) по шагам, получится: ATM - 12%, **Intent Token - 67%**, ISA - 12%, Recon - 9%. По Recall@10 на Toys: 15% / 41% / 29% / 15%. На Baby картина другая: по Recall@10 больше половины прироста дает reconstruction loss (+8.2% на последнем шаге), по NDCG@10 вклад Intent Token и Recon примерно равен.

Это важная поправка к нарративу статьи. Заголовок и введение - про потенциал длинных ID и умное сжатие. Но замена mean pooling на ATM сама по себе дает около +3%, а основной скачок связан с тем, *с какой позиции делается предсказание*. Ablation кумулятивный, порядок добавления фиксирован, поэтому вклады нельзя считать строго аддитивными - но порядок величин показателен.

### 8.2. Стратегия сжатия (при прочих компонентах полной модели)

<div class="table-scroll">
<table>
<thead><tr><th>Merging</th><th>Toys R@10</th><th>Toys N@10</th><th>Baby R@10</th><th>Baby N@10</th></tr></thead>
<tbody>
<tr><td>MLP</td><td>0.0869</td><td>0.0518</td><td>0.0433</td><td>0.0236</td></tr>
<tr><td>Mean pooling</td><td>0.0890</td><td>0.0563</td><td>0.0428</td><td>0.0256</td></tr>
<tr><td>Convolution</td><td>0.0891</td><td>0.0570</td><td>0.0443</td><td>0.0259</td></tr>
<tr><td>ATM</td><td><strong>0.0940</strong></td><td><strong>0.0591</strong></td><td><strong>0.0462</strong></td><td><strong>0.0263</strong></td></tr>
</tbody>
</table>
</div>

ATM лучше mean pooling на +5.6% R@10 / +5.0% N@10 на Toys и на +7.9% / +2.7% на Baby. Это честный leave-one-out вклад ATM - заметный, но не определяющий. Показательно, что **полная модель с обычным mean pooling (0.0563 N@10 на Toys) уже на 24% лучше RPG (0.0454)**: большая часть отрыва от RPG достигается без адаптивного сжатия. Как именно mean pooling, MLP и convolution превращают 32 токена в $k$ латентов (и как для них определен reconstruction loss), в статье не описано.

### 8.3. Prediction anchor

Сравниваются Intent Token и три статических anchor'а, построенных из латентов последнего item'а $\mathbf{Z}_L$:

<div class="table-scroll">
<table>
<thead><tr><th>Anchor</th><th>Toys R@10</th><th>Toys N@10</th><th>Baby R@10</th><th>Baby N@10</th></tr></thead>
<tbody>
<tr><td>Last-Token</td><td>0.0802</td><td>0.0470</td><td>0.0406</td><td>0.0226</td></tr>
<tr><td>Mean-Pooling</td><td>0.0805</td><td>0.0496</td><td><strong>0.0420</strong></td><td>0.0227</td></tr>
<tr><td>MLP</td><td>0.0771</td><td>0.0466</td><td>0.0393</td><td>0.0214</td></tr>
<tr><td>Intent Token</td><td><strong>0.0869</strong></td><td><strong>0.0562</strong></td><td>0.0415</td><td><strong>0.0242</strong></td></tr>
</tbody>
</table>
</div>

На Toys Intent Token дает +8% R@10 и +13-20% N@10. В тексте сказано "consistently performs best", но на Baby по Recall@10 mean-pooling anchor чуть выше (0.0420 против 0.0415) - утверждение слегка завышено.

Почему Intent Token так помогает? Когда item занимает $k$ позиций, ни одна из них не обучена быть "сводкой пользователя": латенты $z_1..z_k$ специализируются на описании item'а. Выделенная позиция со своей ролью и с доступом ко всем латентам текущего блока снимает этот конфликт. Это наблюдение переносимо на любые multi-token-per-item архитектуры.

## 9. Что выучивает ATM

<figure class="paper-figure">
  <img src="../../assets/acerec/atm_attention.png" alt="ATM attention weight heatmaps for Beauty, Toys and Office: dataset average and two individual item examples">
  <figcaption>Рисунок 6 (Figure 13 статьи). Веса внимания ATM: строки - латенты z1..z4, столбцы - 32 подпространства OPQ. Слева среднее по датасету, справа два конкретных item'а. Шкала нелинейная.</figcaption>
</figure>

Авторы выделяют два свойства:

- **sparse-yet-focused.** В среднем ATM смотрит на небольшое число "горячих" подпространств (на Toys - индексы 17, 24, 27, 29, 30, 32), остальным дает почти нулевой вес;
- **content-adaptive.** Для конкретных item'ов паттерн отличается от среднего: на Beauty один item активирует подпространство 27, другой - 19. Каждый латент для конкретного item'а выбирает 3-4 подпространства.

Эта же картинка подсказывает два критических наблюдения, которых в статье нет:

- в усредненных картах **все четыре латента смотрят практически на одни и те же столбцы**. Специализации латентов по разным группам подпространств в среднем не видно - это еще одно указание на избыточность латентов (см. раздел 7.3);
- если на входе большинство из 32 подпространств получает около нулевого веса, то эффективное "входное разрешение" заметно меньше 32. При этом на выходе MTP и scoring используют все 32 цифры. Это усиливает гипотезу из раздела 7.2: длинный ID может быть полезен в первую очередь как output space.

Впрочем, reconstruction loss требует восстанавливать все 32 токена, так что игнорируемые вниманием подпространства могут частично попадать в латенты через summary-зависимые queries. Визуализация качественная, количественного анализа (например, энтропии внимания или reconstruction error по подпространствам) нет.

## 10. Cold-start

Тестовые item'ы разбиты по частоте в train: $[0,5]$, $[6,10]$, $[11,15]$, $[16,20]$.

<figure class="paper-figure">
  <img src="../../assets/acerec/cold_start.png" alt="NDCG@10 by item frequency bucket on Pets and Baby for TIGER, ActionPiece, RPG and ACERec">
  <figcaption>Рисунок 7 (Figure 5 статьи). NDCG@10 по частотным бакетам целевого item'а на Pets и Baby. В самом редком бакете TIGER и RPG близки к нулю; ACERec остается выше остальных на всех бакетах.</figcaption>
</figure>

Масштаб проблемы: в бакет $[0,5]$ попадает от 15% (Instruments) до 32% (Toys) тестовых item'ов, а вместе с $[6,10]$ - около половины на большинстве датасетов. (В тексте статьи назван диапазон "от 19.3% до 32.2%" - он не учитывает CDs с 17.5% и Instruments с 15.1% из их же таблицы; Pets в таблице отсутствует, хотя основной график построен именно на нем.)

Результаты: в бакете $[0,5]$ TIGER и RPG деградируют почти до нуля, ACERec сохраняет ненулевое качество; на Beauty и Office он примерно вдвое выше RPG в этом бакете. С ростом частоты отрыв не сокращается, а растет.

Авторы объясняют это двумя механизмами: ATM сохраняет информативные признаки длинного ID, а ISA привязывает intent к семантике целевого item'а. Второе объяснение выглядит убедительнее: у ISA целевое представление $\mathbf{s}_j$ строится из общих token embeddings, поэтому редкий item получает осмысленный вектор через токены, которые он делит с популярными. Но ablation по бакетам нет, и какой компонент отвечает за cold-start, напрямую не показано. Строго говоря, это не "настоящий" cold-start (item'ы, появившиеся после обучения), а long tail внутри фиксированного каталога.

## 11. Эффективность и сходимость

<figure class="paper-figure">
  <img src="../../assets/acerec/efficiency.png" alt="Scatter plot of inference throughput versus NDCG@10 on Toys for ActionPiece, RPG, TIGER and ACERec">
  <figcaption>Рисунок 8 (Figure 6 статьи). Throughput (samples/s, логарифмическая шкала) и NDCG@10 на Toys. ACERec - в правом верхнем углу.</figcaption>
</figure>

На Toys ACERec дает в среднем **в 2.2 раза больший throughput**, чем ActionPiece, TIGER и RPG: по рисунку это порядка 750-800 samples/s против 300-470. TIGER и ActionPiece тормозит beam search, RPG - итеративное graph decoding. TIGER с теми же 4 токенами на item тоже медленнее.

Оговорки:

- Toys - это 12K item'ов. Полный scoring каталога на таком размере почти бесплатен. Graph decoding в RPG придуман для каталогов, где полный проход невозможен, и сравнивать их throughput на 12K item'ов - значит сравнивать в режиме, удобном для ACERec. Как меняется картина на CDs (64K) и тем более на миллионах item'ов, не показано;
- нет latency на запрос, нет замеров памяти, нет стоимости обучения (вход у ACERec в 5 раз длиннее, чем у RPG);
- точка ACERec на графике соответствует NDCG@10 около 0.058 - это результат версии без reconstruction loss (v1), а не 0.0591. На скорость inference это не влияет, декодер на inference не используется.

<figure class="paper-figure">
  <img src="../../assets/acerec/convergence.png" alt="Validation NDCG@10 over training epochs on Toys for TIGER, ActionPiece, RPG and ACERec">
  <figcaption>Рисунок 9 (Figure 12 статьи). Validation NDCG@10 по эпохам на Toys. ACERec проходит 0.06 за 40 эпох и выходит на плато около 0.071; RPG насыщается около 0.058.</figcaption>
</figure>

По сходимости ACERec быстрее и выше остальных: RPG быстро растет, но рано упирается в 0.058, TIGER остается ниже 0.035. Кривая ActionPiece к 150-й эпохе еще явно растет, так что для него это сравнение при фиксированном бюджете, а не на сходимости.

## 12. Что изменилось между v1 и v2

Это полезно знать, потому что цифры из v1 уже разошлись по обзорам.

<div class="table-scroll">
<table>
<thead><tr><th></th><th>v1 (2026-02-14)</th><th>v2 (2026-08-02)</th></tr></thead>
<tbody>
<tr><td>Loss</td><td>$\mathcal{L}_{\mathrm{MTP}} + \lambda \mathcal{L}_{\mathrm{ISA}}$</td><td>добавлен $\alpha \mathcal{L}_{\mathrm{Recon}}$ и decoder в ATM</td></tr>
<tr><td>Датасеты</td><td>6 (Amazon-2014)</td><td>9: добавлены CDs, Pets, Science, а Instruments взят из Amazon-2023</td></tr>
<tr><td>Headline</td><td>+14.40% NDCG@10 в среднем</td><td>+12.92% NDCG@10, +7.49% Recall@10</td></tr>
<tr><td>Toys, ACERec</td><td>R@10 0.0916, N@10 0.0579</td><td>R@10 0.0940, N@10 0.0591</td></tr>
<tr><td>Short-ID сравнение</td><td>только short-digit OPQ (+44.48% / +56.91%)</td><td>добавлен ETEGRec; +56.18% / +62.76% к short OPQ</td></tr>
<tr><td>Compression ratio</td><td>"inverted-V"</td><td>"stable", плюс heatmap $m \times k$</td></tr>
<tr><td>Оформление</td><td>ACM-шаблон</td><td>AAAI 2027, under review</td></tr>
</tbody>
</table>
</div>

Результат v1 на Toys и Baby в точности совпадает со строкой "+ ISA loss" в ablation v2 - то есть v2 = v1 + reconstruction loss + расширенная оценка. Средний прирост в v2 ниже не потому, что модель стала хуже, а потому, что добавлены датасеты, где выигрыш меньше.

В исходниках v2 на arXiv лежит также не включенный в PDF файл с разделом Limitations. Авторы называют там три ограничения: двухстадийный pipeline с замороженной OPQ-токенизацией, одинаковое $k$ для всех item'ов независимо от их семантической сложности и использование только текстовых признаков.

## 13. Сильные стороны

- **Правильно поставленный вопрос.** "Гранулярность токенизации не обязана совпадать с длиной входа recommender'а" - простая и переносимая идея. Она ортогональна выбору квантизатора и применима к любому длинному item-коду.
- **Хорошо спроектированный контрольный эксперимент** с одинаковым бюджетом в 4 токена на item (раздел 7.2) и heatmap $m \times k$, разделяющий разрешение источника и емкость латентов.
- **Честное выравнивание с RPG**: тот же encoder, тот же backbone и размерности, тот же MTP. Сравнение с ближайшим предшественником чистое.
- **Intent Token** - практичная находка для multi-token-per-item архитектур; ablation по anchor'ам показывает, что эффект большой и не сводится к pooling'у.
- **Простой inference**: ни beam search, ни параметров graph decoding, гарантированно валидные item'ы, ноль inference-гиперпараметров.
- **Широкий набор baseline'ов**, включая сильные ID-based intent-модели, которые в SID-статьях часто опускают и которые здесь местами обыгрывают все предыдущие SID-методы.
- Скромный бюджет тюнинга (9 запусков на датасет) и пять seeds.

## 14. Ограничения и вопросы

**Атрибуция прироста.** Главный тезис статьи - про длинные ID и адаптивное сжатие, но по собственному ablation ATM дает около +3% поверх RPG и около +5% в leave-one-out, тогда как Intent Token - до +20% NDCG@10. Полная модель с mean pooling уже сильно лучше RPG. Название статьи обещает больше, чем показывает ablation.

**Вход или выход?** Не разделены два эффекта длинного ID: более богатый вход (через ATM) и более точный output space (32 heads + scoring по 32 цифрам). Плоская зависимость от $k$, почти одинаковые карты внимания у всех латентов и около нулевые веса у большинства подпространств говорят скорее в пользу второго.

**Inference линейный по каталогу.** Holistic scoring - это ADC без индекса. На 12K-64K item'ов это быстро, но для каталогов в десятки миллионов нужен либо IVF-подобный coarse stage, либо возврат к graph decoding из RPG. Throughput измерен только на самом удобном для метода размере.

**Масштаб и домены.** Только Amazon Reviews, каталоги до 64K, короткие истории. На двух более новых датасетах Amazon-2023 выигрыш падает до 2-4%, по Recall@10 на Instruments - ничья. Нет ни одного не-e-commerce домена и ни одного датасета с длинными историями, где разница между 50 и 250 позициями на входе стала бы существенной.

**Статистика.** Пять seeds заявлены, но дисперсий и тестов значимости нет. Для разниц в третьем знаке на Science/Instruments это критично.

**Baselines.** Часть цифр на Sports/Beauty/Toys перенесена из статьи TIGER - это старые и, по ряду reproducibility-работ, заниженные результаты SASRec. На выводы о лидерстве это не влияет (лучшие baselines там - RPG, ActionPiece и ELCRec, которые воспроизведены), но колонку SASRec не стоит воспринимать как сильный ID-baseline. Нет сравнения с методами, которые тоже борются с длиной входа: [variable-length SID](variable_length_semantic_ids_summary.html), IntRR, а также с простым baseline "RPG + Intent-подобный anchor".

**Замороженная контентная токенизация.** OPQ строится только по текстовым embeddings, без collaborative signal и без дообучения. Там, где предпочтения плохо объясняются текстом, потолок метода задан качеством `sentence-t5`. Для обновляемого каталога встает вопрос стабильности: новые item'ы кодируются старым OPQ без проблем, но переобучение OPQ меняет все коды и инвалидирует модель.

**Условная независимость цифр.** Score - сумма log-вероятностей по цифрам, то есть product-of-experts с предположением независимости. OPQ-поворот декоррелирует подпространства лишь в смысле ошибки квантования; насколько независимость выполняется для пользовательских предпочтений, не проверяется.

**Недоописанные детали.** Вид $f_s$ и $f_q$, архитектура декодера, реализация альтернативных merging-стратегий, то, с каких позиций считается loss при обучении (только финальный шаг или все $\mathbf{h}_t$), - в статье не раскрыты, а кода нет.

## 15. Практические выводы

- Если у вас multi-token-per-item вход (любой: RQ, OPQ, multimodal-токены), **заведите отдельную позицию для предсказания** вместо того, чтобы предсказывать с последнего токена item'а. По данным статьи это самый дешевый и самый результативный компонент.
- **Item-level contrastive loss поверх token-level loss** - недорогой регуляризатор, который, судя по всему, особенно полезен для long-tail item'ов. Item tower для него получается бесплатно из тех же token embeddings.
- Длинный OPQ-код стоит рассматривать прежде всего как **богатый output space с дешевым точным scoring'ом**. Для каталога до сотен тысяч item'ов полный ADC-проход проще и надежнее beam search; для больших каталогов его нужно комбинировать с coarse-этапом.
- Обучаемое сжатие вместо mean pooling дает умеренный, но стабильный плюс (порядка 5%), причем $k=2..4$ латентов достаточно. Ценой в $(k+1)$ раз более длинного входа.
- При чтении похожих статей полезно требовать ablation, отделяющий вклад "представления item'а" от вклада "головы предсказания" - здесь именно он меняет интерпретацию результата.

## 16. Вывод

ACERec - аккуратная инженерная работа поверх RPG, которая закрывает его очевидную слабость (mean pooling на входе) и добавляет два компонента, заимствованных из intent learning и two-tower retrieval. Вместе это дает устойчивый выигрыш на девяти Amazon-датасетах, особенно в NDCG и на редких item'ах, при более простом и быстром inference на небольших каталогах.

Главный переносимый тезис - **гранулярность токенизации и длина входа sequence model - это разные ручки, и их можно крутить независимо** - выглядит верным и полезным. Но эмпирически статья показывает скорее другое: при длинных параллельных ID основной резерв качества лежит не в том, как сжимать item на входе, а в том, откуда и с каким loss'ом делать предсказание. Вопрос о том, нужен ли длинный ID именно на входе, или достаточно длинного output space, остается открытым.
