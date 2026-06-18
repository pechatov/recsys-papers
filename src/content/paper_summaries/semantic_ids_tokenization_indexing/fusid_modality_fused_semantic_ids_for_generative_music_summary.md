---
title: "FusID: Modality-Fused Semantic IDs for Generative Music Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "fusid_modality_fused_semantic_ids_for_generative_music_summary"
catalogId: "paper-fusid_modality_fused_semantic_ids_for_generative_music_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/fusid_modality_fused_semantic_ids_for_generative_music_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2601.08764"
---
Подробное саммари статьи:

> **Авторы:** Haven Kim, Yupeng Hou, Julian McAuley.
>
> **Аффилиации:** University of California San Diego.

## 1. Коротко: о чем статья

FusID предлагает трехстадийный pipeline для построения мультимодальных semantic IDs для музыкальных рекомендаций. Вместо независимой токенизации каждой модальности (текстовые теги, метаданные, тексты песен, аудио, плейлистовое co-occurrence) FusID сначала объединяет все модальности через fusion network, затем обучает объединенное представление через contrastive и regularization losses, и наконец квантует через product quantization. Ключевой результат: на Million Playlist Dataset FusID достигает нулевого уровня коллизий ID (каждый token sequence соответствует ровно одной песне) и превосходит baselines по MRR и Recall@k.

Главная идея: для мультимодальных каталогов semantic ID должен кодировать не сумму независимых модальностей, а их взаимодействие. Если токенизировать каждую модальность отдельно, разные токены могут дублировать один и тот же сигнал, раздувая словарь и не улавливая кросс-модальные связи.

## 2. Контекст: почему музыка -- хороший testbed для мультимодальных SID

Музыкальный item почти всегда мультимодален: аудио характеристики (тембр, темп, гармоническая структура), текстовые метаданные (исполнитель, альбом, жанр), тексты песен, пользовательские теги и, критически, поведенческие сигналы -- какие песни часто появляются вместе в плейлистах. Если каждую модальность квантизовать отдельно (как в TalkPlay), последовательность SID становится длиннее, содержит дублирующую информацию и может не отражать то, как модальности совместно определяют музыкальное предпочтение.

Например, две песни могут иметь похожие аудио-признаки, но разные текстовые теги (instrumental vs vocal). Или наоборот -- похожие теги, но разное звучание. Fusion до квантования позволяет формировать единое пространство, где inter-modal interactions уже учтены, а product quantization делает компактные токены.

## 3. Проблема

FusID формулирует две ключевые проблемы существующих подходов к мультимодальным semantic IDs. Первая -- redundancy: независимое кодирование модальностей дублирует семантический контент, раздувает размер словаря и нагружает языковую модель. Вторая -- потеря inter-modal interactions: изолированная обработка модальностей пропускает комплементарные и синергетические связи между ними.

Дополнительно FusID выделяет проблему codebook underutilization: при неудачной квантизации значительная часть codeword'ов остается неиспользованной, снижая эффективную емкость SID пространства.

## 4. Метод: трехстадийный pipeline

### 4.1. Stage 1: Multimodal Fusion

Из pretrained энкодеров извлекаются модально-специфичные features для пяти модальностей:

<div class="table-scroll">
<table>
<tr><th>Модальность</th><th>Размерность</th><th>Энкодер</th></tr>
<tr><td>Tags</td><td>$\mathbb{R}^{1024}$</td><td>Qwen3 (text)</td></tr>
<tr><td>Metadata</td><td>$\mathbb{R}^{1024}$</td><td>Qwen3 (text)</td></tr>
<tr><td>Lyrics</td><td>$\mathbb{R}^{1024}$</td><td>Qwen3 (text)</td></tr>
<tr><td>Audio</td><td>$\mathbb{R}^{512}$</td><td>CLAP (audio)</td></tr>
<tr><td>Playlist co-occurrence</td><td>$\mathbb{R}^{128}$</td><td>Word2vec-style</td></tr>
</table>
</div>

Features конкатенируются и подаются в fusion network $f_\theta$:

$$
\mathbf{E} = f_\theta([\mathbf{M}_1; \ldots; \mathbf{M}_m])
$$

где $\mathbf{E} \in \mathbb{R}^{n \times d}$ -- выход, разбитый на $n$ sub-embeddings, каждый из которых впоследствии квантуется в один токен SID. Архитектура fusion network: Linear(3072 $\to$ 2048) $\to$ BatchNorm $\to$ ReLU $\to$ Linear(2048 $\to$ $n \times 128$) $\to$ LayerNorm. В экспериментах $n = 5$, $d = 128$.

### 4.2. Stage 2: Representation Learning

Обучение fusion network происходит через два типа loss.

Contrastive loss сближает embedding'и песен, часто встречающихся вместе в плейлистах, и разводит редко совместные:

$$
\mathcal{L}_{\mathrm{cont}} = \mathrm{MSE}(|1 - y|, D(\mathbf{E}_i, \mathbf{E}_j))
$$

где $D$ -- евклидово расстояние, $y \in \{0, 1\}$ -- метка положительной (co-occurring) или отрицательной пары. Пары добываются из нормализованных co-occurrence scores в плейлистах.

Regularization loss, вдохновленный VICReg, состоит из двух компонентов. Covariance loss штрафует корреляции между sub-embeddings, обеспечивая, что каждый sub-embedding захватывает комплементарную информацию:

$$
\mathcal{L}_{\mathrm{cov}} = \frac{2}{n(n-1)} \sum_i \sum_{j>i} \frac{1}{d} \|\mathrm{Cov}(\mathbf{e}_i, \mathbf{e}_j)\|_F^2
$$

Variance loss предотвращает embedding collapse, гарантируя достаточную вариацию по каждому измерению:

$$
\mathcal{L}_{\mathrm{var}} = \frac{1}{n \cdot d} \sum_c \max(0, \gamma - \sqrt{\mathrm{Var}(\mathbf{e}_{\cdot,c}) + \epsilon})
$$

где $\gamma = 1$ -- целевое стандартное отклонение.

Итоговая loss:

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{cont}} + \alpha \cdot (\mathcal{L}_{\mathrm{cov}} + \mathcal{L}_{\mathrm{var}})
$$

с $\alpha = 0.2$.

### 4.3. Stage 3: Product Quantization

После обучения каждый fused embedding $\mathbf{E} \in \mathbb{R}^{n \times d}$ разбивается на $n$ sub-embeddings. Каждый $\mathbf{e}_i$ назначается к одному из $K = 1024$ кластеров через k-means в своем подпространстве (обучение k-means только на training songs). Песня получает token sequence $(c_1, \ldots, c_n)$, где каждая позиция имеет собственный codebook размера 1024.

При $n = 5$ позициях комбинаторное ID пространство составляет:

$$
1024^5 \approx 1.1 \times 10^{15}
$$

Это обеспечивает уникальный mapping для каталога из 537K песен с огромным запасом.

## 5. Экспериментальная установка

### 5.1. Датасет

Million Playlist Dataset (MPD): исходно 2,262,292 уникальных песни. После фильтрации по наличию полных мультимодальных features остается 537,042 песни (~23.7% исходного каталога). Плейлисты с менее чем 6 валидными песнями удалены. Итого: 926,689 плейлистов (в среднем 36.81 песни/плейлист). Split: 80% train, 10% validation, 10% test.

### 5.2. Downstream модель

GPT-2-style decoder-only transformer. Задача: дать первые $l-1$ треков плейлиста (или 30, что меньше) как контекст, предсказать следующую песню.

### 5.3. Baselines

SASRec -- autoregressive sequential recommendation (не генеративный SID подход). TalkPlay -- мультимодальная музыкальная рекомендация с LLM, использующая отдельные per-modality tokenizers.

## 6. Основные результаты

### 6.1. Качество Semantic ID

<div class="table-scroll">
<table>
<tr><th>Метод</th><th>CUR(All)</th><th>CUR(Test)</th><th>Cardinality(All)</th><th>Cardinality(Test)</th><th>Conflict(All)</th><th>Conflict(Test)</th></tr>
<tr><td>TalkPlay</td><td>0.43%</td><td>11.78%</td><td>536,080</td><td>192,836</td><td>0.31%</td><td>0.27%</td></tr>
<tr><td>FusID (w/o reg)</td><td>0.00%</td><td>0.00%</td><td>514,929</td><td>178,661</td><td>6.23%</td><td>11.02%</td></tr>
<tr><td><strong>FusID (full)</strong></td><td><strong>0.00%</strong></td><td><strong>0.02%</strong></td><td><strong>537,042</strong></td><td><strong>193,124</strong></td><td><strong>0.00%</strong></td><td><strong>0.00%</strong></td></tr>
</table>
</div>

FusID достигает нулевого conflict rate и максимальной cardinality на всем датасете и на тестовой подвыборке. На тестовом наборе остается неиспользованным лишь один codeword (0.02% CUR). Это принципиальный результат: каждая песня получает уникальный SID, что делает generated token sequence однозначно grounded в каталоге.

Критически важно: без regularization loss (ablation variant) conflict rate взлетает до 6.23% (All) и 11.02% (Test), а cardinality падает с 537,042 до 514,929. Это означает потерю уникального представления для ~22,000 песен.

### 6.2. Generative recommendation

<div class="table-scroll">
<table>
<tr><th>Метод</th><th>MRR</th><th>R@1</th><th>R@5</th><th>R@10</th><th>R@20</th></tr>
<tr><td>SASRec</td><td>2.60</td><td>1.05</td><td>4.02</td><td>6.85</td><td>9.52</td></tr>
<tr><td>TalkPlay</td><td>9.02</td><td>6.96</td><td>11.61</td><td>13.56</td><td>14.79</td></tr>
<tr><td>FusID (w/o reg)</td><td>7.54</td><td>5.90</td><td>9.53</td><td>11.06</td><td>12.51</td></tr>
<tr><td><strong>FusID (full)</strong></td><td><strong>9.58</strong></td><td><strong>7.40</strong></td><td><strong>12.36</strong></td><td><strong>14.41</strong></td><td><strong>15.69</strong></td></tr>
</table>
</div>

FusID превосходит TalkPlay по всем метрикам: MRR +6.21%, R@1 +6.32%, R@5 +6.46%, R@10 +6.27%, R@20 +6.09%. Улучшения последовательные и стабильные по всем $k$.

## 7. Ablation: роль regularization loss

Удаление regularization loss ($\mathcal{L}_{\mathrm{cov}} + \mathcal{L}_{\mathrm{var}}$) имеет драматический эффект. По качеству SID: conflict rate взлетает с 0% до 6.23%/11.02%. По recommendation: MRR падает на 21.29%, R@5 на 22.90%, R@10 на 23.25%. Ablated модель оказывается хуже TalkPlay (MRR 7.54 vs 9.02), несмотря на архитектурное преимущество fusion.

Это подчеркивает критическую роль regularization: без нее sub-embeddings коллапсируют или становятся избыточными, многие песни получают одинаковые коды, и downstream генерация страдает от ambiguity.

## 8. Анализ codebook utilization

Отдельная per-modality токенизация TalkPlay приводит к 11.78% CUR на тестовых данных -- почти 12% codebook entries потрачены впустую. Product quantization FusID с fused embeddings достигает почти идеальной утилизации (0.00% на All, 0.02% на Test). Ключевой механизм: regularization loss (covariance + variance) предотвращает feature collapse, который иначе заставил бы многие item'ы маппиться в одни и те же кластеры.

## 9. Почему fusion до квантования принципиально

FusID принципиально делает fusion до дискретизации. Если сначала построить отдельные IDs для audio, text и metadata, модель получает длинную последовательность, где разные токены могут кодировать один и тот же музыкальный сигнал. Audio features могут различать tracks с похожими titles; text metadata может различать tracks с похожим звучанием, но разными artists/genres. Co-occurrence в playlists добавляет collaborative signal поверх content modalities.

Fusion сначала формирует единое пространство, где inter-modal interactions уже учтены. PQ после fusion снижает collision probability без раздувания одного словаря. При этом covariance loss гарантирует, что разные sub-embeddings кодируют разные аспекты fused representation, а не дублируют информацию.

## 9.1. Пошаговый алгоритм FusID

1. **Извлечь пять модальностей.** Tags, metadata и lyrics кодируются Qwen3 в 1024-dimensional vectors; audio - CLAP в 512 dimensions; playlist co-occurrence - word2vec-style embedding в 128 dimensions.
1. **Сконкатенировать features.** Все modality vectors объединяются в один вход fusion network, чтобы inter-modal interactions появились до дискретизации.
1. **Получить $n$ sub-embeddings.** MLP $3072 \to 2048 \to n \times 128$ с BatchNorm/ReLU/LayerNorm выдает $n=5$ subspaces.
1. **Обучить fused representation.** Contrastive loss сближает co-occurring playlist songs, covariance loss снижает redundancy между sub-embeddings, variance loss предотвращает collapse.
1. **Обучить PQ codebooks.** Для каждого subspace отдельно запускается k-means с $K=1024$ clusters только на training songs.
1. **Назначить SID.** Song получает sequence $(c_1,\ldots,c_5)$, где каждый $c_j$ - ближайший centroid в своем subspace.
1. **Проверить grounding.** До обучения downstream GPT-2 нужно посчитать cardinality, conflict rate и codeword utilization; zero conflict является ключевым требованием для generative retrieval.
1. **Обучить next-song generator.** Playlist prefix переводится в SID sequences, decoder предсказывает SID следующей песни, затем SID маппится обратно в unique song.

```
for song in catalog:
    features = concat(tags_qwen, metadata_qwen, lyrics_qwen, audio_clap, playlist_w2v)
    E = fusion_network(features)  # shape: n sub-embeddings x 128

train fusion_network with:
    contrastive_loss(co_occurring_song_pairs)
    + alpha * (covariance_loss(E) + variance_loss(E))

for subspace j in 1..n:
    codebook[j] = kmeans(E[:, j], K=1024)

SID(song) = [nearest(codebook[j], E_song[j]) for j in 1..n]
train GPT-style recommender on playlist SID sequences
```

## 10. Сильные стороны

- **Доменная постановка точна для музыки.** Аудио, теги, тексты, metadata и playlist co-occurrence несут разные аспекты preference; single-modality ID объективно беден.
- **Zero-conflict mapping практически важен.** Каждый generated SID соответствует ровно одной песне, поэтому serving не требует disambiguation между несколькими tracks.
- **Tokenizer quality отделен от downstream quality.** FusID показывает и conflict/cardinality metrics, и MRR/Recall, поэтому gain нельзя списать только на GPT-2 backbone.
- **Regularization имеет доказанный вклад.** Без covariance+variance conflict rate растет до 6.23%/11.02%, и recommendation падает ниже TalkPlay.
- **PQ масштабирует capacity.** $1024^5 \approx 10^{15}$ комбинаций покрывают большой каталог без огромного словаря на каждом уровне.

## 11. Ограничения

- **Сильная фильтрация каталога.** Только 537,042 из 2,262,292 песен (~23.7%) имели все модальности; перенос на полный каталог не проверен.
- **Один датасет и один decoder.** MPD и GPT-2-style downstream не доказывают переносимость на другие музыкальные платформы и generative architectures.
- **Cold songs слабо покрыты co-occurrence.** Новые песни без playlist history получат SID почти только из content modalities.
- **Fusion network простая.** Двухслойный MLP не сравнен с attention-based или cross-modal transformer fusion.
- **PQ post-training не end-to-end.** K-means codebooks не оптимизируются совместно с downstream recommendation loss.
- **Production refresh не раскрыт.** Неясно, как обновлять SID при росте каталога и какова latency с 5-token sequences.

## 12. Практические выводы

Для мультимодальных каталогов fusion до квантования предпочтительнее независимой per-modality токенизации. Regularization (covariance + variance) критически важна: без нее fusion degrades в redundancy и collisions. Product quantization предоставляет огромное комбинаторное пространство при умеренном словаре каждого уровня.

При реализации стоит: проверять retrieval в continuous fused space до квантования; после PQ измерять conflict rate, code entropy по каждому subspace и долю dead codewords; downstream проверять MRR/Recall@k против independent-modality tokenizer и text-only/audio-only baselines.

## 13. Связь с другими работами

FusID родственно CoST и ReSID по идее tokenizer-first improvement, но его ключевой axis -- multimodal fusion. С MMQ его объединяет мультимодальная природа, но подходы различаются: MMQ использует shared-specific экспертов с отдельными кодбуками, FusID -- единую fusion network с product quantization. С TalkPlay это прямое сравнение: per-modality vs fused tokenization. Для music recsys FusID -- аккуратный мост между multimodal representation learning и semantic ID generation.

## 14. Итоговая оценка

FusID -- концептуально чистая и хорошо выполненная работа для музыкального домена. Ее ценность -- в ясной демонстрации того, что multimodal fusion до квантования + regularization для предотвращения collapse = conflict-free и downstream-полезные semantic IDs. Ограничения связаны с масштабом оценки (один датасет, 23.7% каталога) и простотой fusion архитектуры. Для исследователей это хороший baseline и концептуальный пример того, как правильно строить мультимодальные SID; для практиков -- аргумент в пользу fusion-first подхода при наличии множественных модальностей.
