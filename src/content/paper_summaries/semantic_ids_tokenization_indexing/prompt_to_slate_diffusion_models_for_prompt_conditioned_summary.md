---
title: "Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation"
category: "semantic_ids_tokenization_indexing"
slug: "prompt_to_slate_diffusion_models_for_prompt_conditioned_summary"
catalogId: "paper-prompt_to_slate_diffusion_models_for_prompt_conditioned_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/prompt_to_slate_diffusion_models_for_prompt_conditioned_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2408.06883"
---
**Авторы:** Federico Tomasi, Francesco Fabbri, Justin Carter, Elias Kalomiris, Mounia Lalmas, Zhenwen Dai.

**Аффилиации:** Spotify.

**Индустрия:** prompt-conditioned playlist/slate generation и e-commerce bundles.

**Первичный источник:** arXiv source 2408.06883.

## Коротко

- DMSG генерирует целый slate по natural language prompt.
- Вместо autoregressive SID используется continuous diffusion over concatenated item embeddings.
- Подходит для editorial/cold-start settings, где нет персональной истории.

## Контекст

- Slate recommendation требует coherence между itemами, а не только individual relevance.
- Playlists и bundles имеют higher-order structure.
- Prompt-conditioned generation должен давать несколько валидных diverse answers.

## Проблема

- Комбинаторный slate space огромен.
- Retrieval baselines плохо моделируют joint distribution.
- Diffusion inference может быть дорогим без DDIM/latency optimization.

## Метод/архитектура

- Item sequence кодируется fixed encoder в continuous latent space.
- Text encoder дает condition vector.
- Diffusion transformer с cross-attention предсказывает noise.
- Nearest-neighbor rounding превращает latent vectors обратно в catalog items.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Objective/алгоритм

- Forward process добавляет Gaussian noise.
- Reverse process обучается denoising objective.
- DDIM позволяет использовать fewer steps на inference.
- Multiple samples одного prompt дают diversity/freshness.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

### Подробная схема алгоритма

DMSG заменяет autoregressive generation of item IDs на denoising в continuous slate space: модель сразу восстанавливает набор embedding'ов будущего slate, а затем округляет их до каталога.

1. **Закодировать обучающие slates.** Playlist/bundle превращается в concatenated sequence item embeddings фиксированной длины; prompt кодируется text encoder'ом.
1. **Добавить forward noise.** На шаге $t$ к slate latent добавляется Gaussian noise, создавая зашумленное представление.
1. **Обучить denoising transformer.** Модель получает noisy slate, timestep и prompt condition, затем через cross-attention предсказывает noise или clean latent.
1. **Сэмплировать на inference.** Стартуя с noise, DDIM/denoising steps постепенно строят slate latent, conditioned on prompt.
1. **Округлить к items.** Каждый latent vector маппится к ближайшему item embedding в каталоге; затем применяются post-processing rules против duplicates/invalid candidates.
1. **Получить diversity.** Несколько запусков с разным seed для одного prompt дают разные coherent slates, что важно для freshness/exposure.

```
train:
    slate_latent = concatenate(item_embeddings(slate))
    condition = text_encoder(prompt)
    noisy, noise = add_gaussian_noise(slate_latent, timestep)
    predicted = diffusion_transformer(noisy, timestep, condition)
    minimize denoising_loss(predicted, noise)

inference:
    latent = random_noise()
    for t in DDIM_steps:
        latent = denoise(latent, t, text_encoder(prompt))
    slate = nearest_neighbor_rounding(latent, catalog_embeddings)
    postprocess duplicates and invalid items
```

## Эксперименты

- Datasets: Spotify MPD subset 100K playlists и Amazon Bundle 11K items/5,444 bundles.
- Metrics: NDCGSim, MAPSim, BertScore, CategorySim.
- Curated: до +17% NDCGSim и +12.9% MAPSim над BM25.
- Online: NVIDIA L4, 50 diffusion steps, P99 150 ms против 500 ms baseline.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Рисунки/таблицы

- Training/inference overview DMSG.
- Transformer details figure.
- Tables compare baselines on Curated/MPD/Bundle.
- Exposure/freshness plots показывают diversity.

## Ablation conclusions

- Post-processing влияет на MPD quality.
- BM25 может выигрывать по NDCGSim на MPD, но DMSG лучше по MAPSim/BertScore.
- Freshness/overlap analysis показывает stochastic benefit.

## Сильные стороны

- **Моделирует joint slate distribution.** Метод оптимизирует совместимость item'ов внутри playlist/bundle, а не независимый top-K ranking.
- **Prompt conditioning естественно подходит editorial сценариям.** Natural language prompt может задавать mood/use-case, где пользовательской истории мало или нет.
- **Stochastic generation дает diversity/freshness.** Несколько samples одного prompt создают альтернативные slates без ручного reranking diversity.
- **Continuous latent decouples model from fixed token IDs.** При обновлении каталога можно обновлять nearest-neighbor mapping, не обязательно переобучая decoder с нуля.

## Ограничения

- **Качество ограничено item embeddings.** Если embedding space плохо отражает slate compatibility, diffusion будет генерировать гладкие, но нерелевантные vectors.
- **Nearest-neighbor rounding может сломать coherence.** Latent vectors могут быть согласованными до округления, но ближайшие catalog items окажутся duplicates или semantically off.
- **Diffusion latency остается trade-off.** 50 steps на NVIDIA L4 дают P99 150 ms в summary, но другой catalog/encoder может потребовать другой budget.
- **Similarity metrics не равны satisfaction.** NDCGSim/MAPSim/BertScore/CategorySim полезны, но не заменяют пользовательское acceptance, skips и saves.

## Как реализовать/проверять

1. Зафиксировать версию каталога, train/eval split и mapping item/document -> identifier; без этого невозможно понять, улучшает ли метод саму модель или только меняет пространство кандидатов.
1. Считать отдельно качество tokenization и качество generator/ranker: collision rate, utilization, Gini, valid-path rate, Recall/HR/NDCG/MRR и latency должны лежать в одном отчете.
1. Делать ablation не только по среднему качеству, но и по head/torso/tail, cold-start, new users/new items, long-history и категориям с похожими объектами.
1. Проверять деградацию при обновлении каталога: semantic IDs могут устаревать, а генератор может помнить старые пути.
1. Сохранять обратный индекс identifier -> item/document и явно логировать случаи many-to-one collisions.
1. Для генеративного вывода использовать constrained decoding или post-validation, иначе invalid identifiers будут маскироваться в offline метриках.
1. В production считать стоимость не только модели, но и перестроения codebook, trie/index, feature pipeline и мониторинга drift.
1. В отчетах отделять retrieval-stage gains от downstream ranking/business metrics, потому что рост HR@K не всегда дает CTR/CVR uplift.

## Failure modes и мониторинг

- Identifier collapse: малая часть кодов получает большую долю объектов, а генератор начинает переиспользовать популярные пути.
- Semantic-collaborative mismatch: похожие по тексту объекты имеют разные user intents, или наоборот, совместно потребляемые объекты текстово далеки.
- Exposure bias autoregressive generation: ошибка раннего токена полностью меняет candidate path.
- Popularity bias: модель учится генерировать frequent IDs и выглядит хорошо на aggregate, но теряет novelty и tail coverage.
- Evaluation leakage: если ID/tokenizer обучался на будущем каталоге или post-training видел target-like сигналы, offline gain завышен.
- Serving mismatch: offline beam шире и медленнее production beam, поэтому качество не переносится напрямую.
- Специфично для этой статьи: Зависит от item embeddings и rounding.
- Специфично для этой статьи: Similarity metrics не полностью заменяют user satisfaction.
- Специфично для этой статьи: Diffusion steps дают latency trade-off.

## Связь

- Контрастирует с SID autoregressive retrieval.
- Близок к generative recommendation как output-level генерация slate, а не item ID.

## Итог

- Для slate задач diffusion может быть лучше autoregressive ID generation.
- Главная идея - учить joint distribution of slates conditioned on prompt.

## Минимальный план воспроизведения

<div class="table-scroll">
<table>
<thead>
<tr><th>Шаг</th><th>Что сделать</th><th>Что считать успехом</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>Собрать исходные item/document features и interaction/query logs в той же временной схеме, что у статьи.</td><td>Нет leakage, воспроизводимы splits и отрицательные примеры.</td></tr>
<tr><td>2</td><td>Построить identifiers/token representations и сохранить mapping plus collision report.</td><td>Utilization и collision metrics понятны до обучения generator.</td></tr>
<tr><td>3</td><td>Обучить baseline без нового компонента и полный метод с тем же budget.</td><td>Сравнение честное по compute, beam, candidates и preprocessing.</td></tr>
<tr><td>4</td><td>Запустить ablations по каждому заявленному компоненту.</td><td>Каждый компонент дает объяснимый вклад или честно признается избыточным.</td></tr>
<tr><td>5</td><td>Проверить production-like constraints: latency, invalid IDs, refresh, monitoring.</td><td>Offline gain не исчезает при реальных ограничениях serving.</td></tr>
</tbody>
</table>
</div>

Примечание: если в источнике не раскрыты приватные production details, summary явно помечает такие ограничения и не выдумывает закрытые числа.
