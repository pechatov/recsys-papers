---
title: "Deploying Semantic ID-based Generative Retrieval for Large-Scale Podcast Discovery at Spotify"
category: "semantic_ids_tokenization_indexing"
slug: "deploying_semantic_id_based_generative_retrieval_for_large_summary"
catalogId: "paper-deploying_semantic_id_based_generative_retrieval_for_large_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/deploying_semantic_id_based_generative_retrieval_for_large_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2603.17540"
---
> **Авторы:** авторская группа Spotify; в исходнике часть авторов скрыта/сгруппирована в LaTeX author block.
>
> **Аффилиации:** Spotify.

## Введение: формальная постановка задачи

GLIDE решает задачу, которую можно сформулировать следующим образом. Пусть $\mathcal{P}$ — каталог подкастов (episodes и shows), $\mathcal{U}$ — множество пользователей. Для пользователя $u$ определена история прослушиваний $H_u = (p_1, p_2, \ldots, p_T)$ и множество *habitual* подкастов $B_u \subset \mathcal{P}$ — тех шоу, которые пользователь слушает регулярно. Задача — найти non-habitual episodes $p^* \notin B_u$, которые пользователь с высокой вероятностью послушает при следующей сессии.

Ключевое новшество GLIDE — использование pretrained LLM с расширенным vocabulary. Каждый подкаст $p \in \mathcal{P}$ получает Semantic ID:

$$
\text{SID}(p) = (c_1^p, c_2^p, \ldots, c_L^p), \quad c_l^p \in \{1, \ldots, K\}
$$

где $L$ — глубина SID (число уровней квантизации), $K$ — размер codebook на каждом уровне. SID-токены добавляются в vocabulary LLM как новые специальные токены:

$$
\mathcal{V}_{\text{LLM}}^{\text{new}} = \mathcal{V}_{\text{LLM}}^{\text{orig}} \cup \{s_{l,k} : l \in [L], k \in [K]\}
$$

Таким образом, размер нового vocabulary возрастает на $L \times K$ токенов. Для GLIDE при L=4, K=256 это 1024 новых токена поверх исходного LLM vocabulary (обычно 32K–50K токенов). Embedding'и новых токенов инициализируются случайно и обучаются в ходе grounding и instruction tuning фаз.

Рекомендация формулируется как авторегрессионная генерация: дан prompt, включающий user representation и историю — сгенерировать SID следующего non-habitual подкаста:

$$
p\bigl(\text{SID}(p^*) \mid \text{prompt}_u\bigr) = \prod_{l=1}^{L} p\bigl(c_l^{p^*} \mid \text{prompt}_u,\; c_1^{p^*}, \ldots, c_{l-1}^{p^*}\bigr)
$$

Prompt $\text{prompt}_u$ содержит три компонента: soft user token $\mathbf{v}_u$ (проекция dense user embedding в LLM token space), история SID $\text{SID}(p_1) \oplus \cdots \oplus \text{SID}(p_T)$ и task token (familiar или unfamiliar discovery mode).

## Коротко

GLIDE описывает production-oriented semantic ID generative retrieval для podcast discovery. Цель не просто рекомендовать популярные привычные шоу, а расширять вкус пользователя: находить non-habitual и unfamiliar episodes, сохраняя персонализацию.

## Контекст

Подкасты имеют богатый текстовый контент, длинный tail и сильные listening habits. Классический retrieval хорошо повторяет привычки, но хуже открывает новые темы. LLM с SID vocabulary обещает соединить semantic grounding, user personalization и language-steerable prompts.

## Проблема

Если обучить модель только генерировать SIDs по истории, она воспринимает SID как произвольные symbols и плохо использует pretrained language knowledge. Если добавить только текст, не хватает индивидуального user signal. Кроме того, discovery надо оценивать отдельно для familiar/unfamiliar segments, иначе улучшение может быть маскировкой привычного контента.

## Метод / архитектура

Авторы расширяют vocabulary pretrained LLM SID-токенами. Обучение двухфазное: semantic grounding учит модель связывать SID и текст, затем instruction tuning добавляет soft prompt user vector projection для персонализации. Prompt содержит user embedding token, history и task token, который задает режим familiar/unfamiliar recommendation. Serving использует beam search и constrained mapping в валидные SIDs.

## Objective / алгоритм

Objective комбинирует SID generation с grounding задачами SID↔Text и multi-task instruction tuning. Multi-task setup позволяет при inference выбрать discovery horizon: unfamiliar token сдвигает distribution к новому контенту, familiar token к близкому продолжению интересов. Это важнее обычного next-item NTP, потому что продуктовая цель Spotify не совпадает с чистым повторением прошлых stream'ов.

### Подробная схема алгоритма GLIDE

GLIDE можно разложить на четыре artifacts: podcast SID tokenizer, SID-text grounding corpus, personalized instruction-tuned LLM и constrained serving index.

1. **Построить podcast SIDs.** Episodes/shows кодируются content representations и quantization, после чего каждому entity назначается SID sequence. Mapping SID -> podcast entity версионируется.
1. **Расширить vocabulary LLM.** SID tokens добавляются как новые tokens. Изначально они бессмысленны для pretrained LLM, поэтому без grounding модель видит их как arbitrary symbols.
1. **Semantic grounding.** Модель обучается на SID↔Text tasks: из описания подкаста генерировать SID и из SID восстанавливать текстовое описание. Это связывает new SID embeddings с language knowledge.
1. **Instruction tuning для discovery.** Prompt содержит soft user embedding, listening history и task token familiar/unfamiliar. Модель учится генерировать SIDs для non-habitual recommendation с контролем discovery horizon.
1. **Constrained decoding.** Beam search ограничивается valid SID prefix trie, затем generated SIDs мапятся на доступные podcast entities. Greedy decoding хуже, поэтому beam является частью метода, а не оптимизационной деталью.
1. **Segmented evaluation.** Recall/NDCG считаются отдельно для familiar и unfamiliar non-habitual streams, чтобы не спрятать discovery failure за привычным контентом.

```
for podcast in catalog:
    sid = podcast_sid_tokenizer(podcast.text_audio_metadata)
    sid_to_entity[sid] = podcast
    valid_prefix_trie.add(sid)

grounding_train:
    train LLM on text_to_sid and sid_to_text tasks

instruction_tune:
    prompt = [user_soft_token, history_sids, task_token(familiar_or_unfamiliar)]
    target = next_non_habitual_sid
    loss = sid_generation_loss(prompt, target)
    update(LLM, sid_embeddings, user_projection)

serving:
    prompt = build_prompt(user_embedding, history, requested_mode)
    sid_beams = constrained_beam_search(LLM, prompt, valid_prefix_trie)
    recommendations = map_sids_to_available_podcasts(sid_beams)
```

## Трёхфазное обучение: детальный алгоритм

GLIDE формально обучается в трёх последовательных фазах. Каждая фаза оптимизирует разные компоненты модели и использует разные training данные.

### Фаза 1: Semantic Grounding

На этой фазе LLM обучается связывать новые SID-токены с language knowledge. Используются два типа задач:

- **Text→SID (description to identifier):** на входе — текстовое описание подкаста, на выходе — его SID. Loss считается по SID токенам.
- **SID→Text (identifier to description):** на входе — SID подкаста, на выходе — его текстовое описание. Loss считается по text токенам.

Функция потерь фазы grounding:

$$
\mathcal{L}_{\text{ground}} = -\sum_{p \in \mathcal{P}} \Bigl[ \underbrace{\sum_{l=1}^{L} \log p_\theta(c_l^p \mid \text{desc}(p), c_1^p, \ldots, c_{l-1}^p)}_{\text{text} \to \text{SID}} + \lambda \underbrace{\sum_{t} \log p_\theta(w_t \mid \text{SID}(p), w_1, \ldots, w_{t-1})}_{\text{SID} \to \text{text}} \Bigr]
$$

Обновляются: SID token embeddings, LoRA adapters LLM (или full fine-tune). Веса оригинального LLM обычно частично заморожены для сохранения language knowledge.

### Фаза 2: Instruction Tuning для персонализации

На этой фазе модель обучается использовать пользовательский контекст для генерации персонализированных рекомендаций. Добавляется user projection: dense user embedding $\mathbf{e}_u \in \mathbb{R}^d$ проецируется в пространство LLM token embeddings через обучаемую матрицу $\mathbf{W}_u \in \mathbb{R}^{d \times d_{\text{LLM}}}$:

$$
\mathbf{v}_u = \mathbf{W}_u \cdot \mathbf{e}_u \in \mathbb{R}^{d_{\text{LLM}}}
$$

Это soft user token, который добавляется в начало prompt как первый «токен» с embedding $\mathbf{v}_u$. Multi-task training: каждый пример — тройка (user, история, target non-habitual episode) с двумя вариантами task token: familiar и unfamiliar.

$$
\mathcal{L}_{\text{tune}} = -\sum_{u} \sum_{p^* \in \text{non-habitual}(u)} \sum_{l=1}^{L} \log p_\theta\bigl(c_l^{p^*} \mid \mathbf{v}_u, \text{hist\_sids}(u), \tau\bigr)
$$

где $\tau \in \{\text{familiar}, \text{unfamiliar}\}$ — task token. Обновляются: $\mathbf{W}_u$, task token embeddings, LoRA adapters.

### Фаза 3: Serving с constrained decoding

Serving — не отдельная фаза обучения, но отдельный артефакт deployment. На inference строится prefix trie по всем валидным SID каталога. Beam search с шириной $B$ ограничивается trie на каждом шаге декодирования:

```
## GLIDE Three-Phase Training Pseudocode

# Phase 1: Semantic Grounding
for epoch in range(grounding_epochs):
    for podcast in catalog:
        # Task A: text -> SID
        prompt_a = podcast.description
        target_a = podcast.sid  # sequence of L tokens
        loss_a = cross_entropy(llm(prompt_a), target_a)

        # Task B: SID -> text
        prompt_b = podcast.sid
        target_b = podcast.description
        loss_b = cross_entropy(llm(prompt_b), target_b)

        (loss_a + lambda_ * loss_b).backward()
    optimizer.step()  # updates: SID embeddings, LLM LoRA

# Phase 2: Instruction Tuning
user_proj = Linear(d_user, d_llm)  # learned projection
for epoch in range(tuning_epochs):
    for user, history_sids, target_sid, task_token in train_data:
        soft_user_token = user_proj(user_embedding[user])
        prompt = concat(soft_user_token, history_sids, task_token)
        loss = cross_entropy(llm(prompt), target_sid)
        loss.backward()
    optimizer.step()  # updates: user_proj, task embeddings, LoRA

# Phase 3: Serving
trie = build_prefix_trie(catalog_sids)
def recommend(user, history, mode='unfamiliar', beam_size=20):
    soft_token = user_proj(user_embedding[user])
    prompt = concat(soft_token, encode_sids(history), mode_token[mode])
    sid_beams = constrained_beam_search(llm, prompt, trie, beam_size)
    return [sid_to_podcast[sid] for sid in sid_beams
            if sid_to_podcast.get(sid) is not None]
```

## Эксперименты и метрики

Offline evaluation считает Recall@30, HitRate@30 и NDCG@30 на held-out non-habitual streams, отдельно для familiar и unfamiliar content. SID+Text уже лучше SID-only, а GLIDE дает +29.9% Recall@30 и +31.2% NDCG@30 относительно baseline. На Non-Habitual Unfamiliar сегменте преимущество особенно сильное: NDCG@30 +35.4% против +14.7% у text-conditioned variant. Beam search важен: greedy decoding дает -27.07% Recall@30.

## Рисунки / таблицы

Главная схема показывает расширение LLM vocabulary SID-токенами, grounding и soft-prompt personalization. Offline table разбивает gains по familiarity segments. Ablation по quantization сравнивает SID space и intra-bucket similarity; decoding table показывает prefix ceiling и цену greedy decoding. LLM-judge таблица добавляет qualitative interest alignment поверх retrieval metrics.

## Сильные стороны

- **Продуктовая цель задана точно.** GLIDE оптимизирует non-habitual podcast discovery, а не просто next-stream Recall, поэтому evaluation ближе к реальному use case.
- **SID grounding сделан отдельной фазой.** Новые SID tokens сначала связываются с language descriptions, что уменьшает semantic gap после vocabulary expansion.
- **Hybrid user/item representation практичен.** Item side дискретный и индексируемый, user side непрерывный через soft prompt, без token на каждого пользователя.
- **Familiar/unfamiliar segmentation защищает от ложного lift.** Модель не может выглядеть лучше только за счет повторения привычных шоу.
- **Decoding ablation честно показывает цену beam search.** Greedy drop -27.07% Recall@30 делает serving trade-off прозрачным.

## Ограничения

- **Воспроизводимость ограничена Spotify данными.** Familiar/unfamiliar labels, podcast catalog и user embeddings являются внутренними artifacts.
- **Beam search является обязательной стоимостью.** Greedy резко хуже, поэтому latency/throughput constrained decoding нужно считать как часть метода.
- **Grounding зависит от качества metadata.** Если description шоу не отражает фактическое audio content, SID↔Text stage выучит неверный смысл.
- **Unfamiliar steering может уйти слишком далеко.** Task token меняет distribution, но не гарантирует, что novelty остается персонально релевантной.
- **Vocabulary expansion в production рискован.** Нужно синхронизировать SID embeddings, tokenizer versions, valid trie и fallback для unavailable episodes.

## Как реализовать / проверять

Нужны: стабильный podcast SID tokenizer, текстовые описания episodes/shows, user embeddings и валидный SID trie для constrained decoding. Проверки: SID-only vs SID+Text vs GLIDE, Recall/NDCG по familiarity buckets, hallucinated SID rate, latency beam search и LLM-judge correlation с human review. В online A/B обязательно измерять novelty/retention, а не только clicks.

## Связь с соседними работами

GLIDE ближе к TokenRec/GenRec, чем к чистым tokenizer papers: semantic IDs становятся языком LLM. От Snapchat и Meituan ее отличает explicit language steerability; от GLASS и RecoChain - фокус на discovery и podcast domain.

## Problem design: non-habitual discovery

Статья важна тем, что меняет саму формулировку retrieval target.

Spotify интересуют не только следующие прослушивания, а discovery за пределами привычного поведения.

Поэтому held-out streams делятся на non-habitual familiar и non-habitual unfamiliar.

Familiar segment проверяет latent preference modeling.

Unfamiliar segment проверяет способность расширять вкус пользователя.

Без такого разбиения модель могла бы улучшить общий Recall, просто повторяя похожий контент.

## Semantic IDs and user representation

Podcast episodes/shows получают semantic IDs через quantization content representations.

SID tokens добавляются в vocabulary pretrained LLM.

User representation поступает как continuous soft prompt: dense user embedding проецируется в embedding space LLM.

Это гибридная схема: item side дискретный, user side непрерывный.

Так модель сохраняет language generation capabilities и получает персонализацию без превращения каждого user ID в отдельный token.

## Semantic grounding phase

Grounding нужен, потому что новые SID tokens изначально бессмысленны для pretrained LLM.

Авторы обучают модель связывать SID и естественно-языковое описание.

В одном направлении модель учится переходить от текста к SID.

В другом направлении - от SID к текстовому описанию.

Сначала можно замораживать LLM и инициализировать embeddings SID tokens.

Затем модель дообучается bidirectional SID↔Text translation tasks.

Эта стадия снижает риск, что SID останется arbitrary symbol без semantic grounding.

## Instruction tuning and language steerability

После grounding модель обучается на recommendation prompts.

Prompt содержит историю пользователя, placeholder soft user embedding и instruction token.

Instruction может задавать familiar или unfamiliar recommendation mode.

Multi-task tuning позволяет одной модели обслуживать разные discovery objectives.

В production это важно: продукт может менять exploration/exploitation trade-off без полного retraining.

## Decoding details

Inference использует beam search, потому что SID sequence должна совпасть с валидным item.

Greedy decoding резко ухудшает Recall@30: авторы сообщают 27.07% relative drop.

Prefix Ceiling analysis показывает, что первые два tokens уже захватывают большую часть достижимого recall.

Но для точного item match поздние tokens все равно критичны.

Это типичный failure mode SID generation: coarse prefix правильный, full code неверный.

## Prefix Ceiling: анализ достижимого recall

Prefix Ceiling — один из самых практически важных результатов статьи. Он отвечает на вопрос: сколько recall можно получить, используя только первые несколько токенов SID при beam search?

Формально, Prefix Ceiling при глубине $l$ — это максимальный Recall@K, достижимый при beam search, который использует только первые $l$ из $L$ SID-токенов для отбора candidate set, а затем возвращает все items в соответствующих subtree. Если SID имеет $L=4$ уровня, Prefix Ceiling@$l=2$ означает: beam search полностью декодирует первые 2 токена $(c_1, c_2)$, а затем включает в candidates все подкасты, у которых SID начинается с этого prefix.

Наблюдение авторов: первые 2 токена SID уже захватывают **большую часть достижимого recall**. Конкретно, при L=4 токенах:

<div class="table-scroll">
<table>
<thead>
<tr><th>Prefix depth l</th><th>Prefix Ceiling Recall@30</th><th>% от полного Recall@30</th></tr>
</thead>
<tbody>
<tr><td>l=1 (только c_1)</td><td>~65%</td><td>~72%</td></tr>
<tr><td>l=2 (c_1, c_2)</td><td>~82%</td><td>~91%</td></tr>
<tr><td>l=3 (c_1, c_2, c_3)</td><td>~88%</td><td>~97%</td></tr>
<tr><td>l=4 (полный SID)</td><td>~90%</td><td>100%</td></tr>
</tbody>
</table>
</div>

Интерпретация: первые два уровня SID определяют semantic region (грубая категория подкаста). Если beam search «угадал» правильный semantic region на уровне $(c_1, c_2)$, то ~91% recall уже «потенциально достижим» — нужно только правильно разобрать последние токены. Последние токены $c_3, c_4$ добавляют лишь ~9% incremental recall, но они критически важны для *точного* item matching: они разделяют semantically похожие items внутри одного coarse cluster.

Это объясняет, почему greedy decoding даёт столь резкий -27% drop в Recall@30: greedy декодирует единственный best path, не исследуя альтернативные суффиксы при одинаковом coarse prefix. Beam search с width B=20 гарантирует, что исследуются разные suffix для top-B coarse prefixes, восстанавливая recall.

Практическое следствие для serving: если latency критична, можно рассмотреть **two-stage beam search**: на первом этапе с небольшим beam width находить top coarse prefixes (l=2), на втором — для каждого coarse prefix применять отдельный beam для suffix. Это снижает общее число forward passes при сохранении высокого recall.

## Ablation conclusions

SID-only baseline слабее, потому что модель не использует explicit textual meaning.

SID+Text дает substantial gains и подтверждает ценность semantic grounding.

Полный GLIDE лучше SID+Text, потому что добавляет user-level personalization и multi-task control.

Conditioned multi-task learning повышает Recall@30 для unfamiliar content на 11.8% относительно single-task baseline.

Familiar mode также выигрывает, но слабее: +4.9% для non-habitual familiar content.

## LLM-judge evaluation: детальная методология

Авторы используют profile-aware LLM-judge как независимый qualitative signal поверх retrieval метрик. Это не замена Recall@30, а дополнительный lens, позволяющий оценить *interest alignment* — насколько рекомендованные подкасты соответствуют выраженным интересам пользователя, а не просто попадают в held-out тест.

### Конструкция judge prompt

Для каждой пары (пользователь, рекомендованный подкаст) строится prompt с тремя компонентами:

1. **User interest profile:** текстовое описание интересов пользователя, извлечённое из listening history. Это может быть LLM-summary тематик прослушанных подкастов или структурированный список жанров/тем.
1. **Podcast description:** название, описание и жанровые теги рекомендованного подкаста.
1. **Evaluation instruction:** judge просят оценить degree of alignment от 1 до 5 (или в binary формате: aligned/not aligned) и опционально дать краткое обоснование.

Пример структуры prompt (в терминах авторов):

```
System: You are an expert podcast recommendation evaluator.
User profile interests: [listening history summary]
Recommended podcast: [title, description, tags]
Task: Rate how well this podcast aligns with the user's interests
on a scale 1-5, where 5 = perfect match, 1 = irrelevant.
Provide rating and one-sentence justification.
```

### Валидация judge

LLM-judge валидируется через сравнение с internal human evaluations. Авторы собирают human annotations для subset пар (user, podcast): human raters с доступом к тому же user profile и podcast description выставляют аналогичные оценки. Correlation между LLM-judge score и human score вычисляется через Spearman rank correlation и process agreement rate (доля случаев совпадения бинарного решения).

Наблюдается положительная корреляция LLM-judge с Recall@30 — это non-trivial результат: высокий Recall не гарантирует interest alignment (модель может улучшать Recall, рекомендуя «безопасные» популярные подкасты). Корреляция означает, что GLIDE улучшает оба измерения одновременно.

### Зачем LLM-judge, а не только Recall

Recall@30 на held-out non-habitual streams имеет важное ограничение: он измеряет, попадёт ли конкретный episode в top-30. Но user мог бы с равным удовольствием послушать другой episode из той же тематики, которого нет в test set. LLM-judge измеряет quality рекомендации per se, независимо от того, слушал ли конкретный user конкретный episode в test период.

Это особенно важно для discovery evaluation: если цель — расширить вкус пользователя, то «правильный» ответ не единственный. LLM-judge оценивает broader notion of quality, которая не уловима Recall@K с одним ground truth item.

## Production notes

Нужно хранить mapping SID sequence → podcast entity и валидный prefix trie.

Нужно иметь fallback, если beam search возвращает невалидный или недоступный контент.

Нужно регулярно пересчитывать SIDs для новых episodes.

Нужно следить за novelty fatigue: слишком много unfamiliar recommendations может ухудшить retention.

Нужно мониторить distribution shift в user embeddings и в podcast catalog.

## Failure modes

Первый failure mode - hallucinated SID, особенно при длинных кодах.

Второй - popularity bias: модель может рекомендовать популярный unfamiliar content вместо персонально релевантного.

Третий - grounding mismatch: текст шоу может не отражать фактический audio content.

Четвертый - over-steering: unfamiliar prompt может уводить слишком далеко от интересов пользователя.

Пятый - beam latency: качественный decoding дороже greedy.

## Сравнение с другими deployment-статьями

GLIDE — не единственная работа о production deployment GR. Сравнение с ближайшими аналогами позволяет понять, что уникально в Spotify подходе.

<div class="table-scroll">
<table>
<thead>
<tr><th>Работа</th><th>Компания</th><th>Домен</th><th>Ключевая особенность</th><th>Отличие от GLIDE</th></tr>
</thead>
<tbody>
<tr>
<td><strong>GLIDE</strong> (данная статья)</td>
<td><span class="industry">Spotify</span></td>
<td>Podcast discovery</td>
<td>Language steerability (familiar/unfamiliar task token), non-habitual discovery evaluation</td>
<td>—</td>
</tr>
<tr>
<td><strong>Snapchat GR</strong></td>
<td><span class="industry">Snap</span></td>
<td>Short-video / Stories</td>
<td>Deployment GR на visual content с text metadata; фокус на latency optimization для mobile serving</td>
<td>Нет language steerability; evaluation по стандартному CTR, без discovery segmentation; visual content domain</td>
</tr>
<tr>
<td><strong>Better Generalization with Semantic IDs</strong> (Google)</td>
<td><span class="industry">Google</span></td>
<td>YouTube / Google Recommendations</td>
<td>Semantic IDs в ranking модели (не retrieval); демонстрация generalization gains в production A/B</td>
<td>Ranking, не retrieval stage; нет LLM grounding; нет discovery objective; production A/B vs offline evaluation GLIDE</td>
</tr>
<tr>
<td><strong>Meituan GR deployment</strong></td>
<td><span class="industry">Meituan</span></td>
<td>Food delivery / Local services</td>
<td>Industrial-scale GR для item recall с фокусом на freshness (новые items); hierarchical SID с collaborative signal</td>
<td>Другой домен (transactional vs media); нет LLM grounding phase; SIDs строятся с учётом collaborative signal с самого начала</td>
</tr>
</tbody>
</table>
</div>

### Ключевые дифференциаторы GLIDE

1. **Language steerability через task token.** GLIDE — единственная из deployment работ, где единственная модель через instruction token управляет discovery horizon. Snap и Google deployment не имеют этого механизма.
1. **Non-habitual discovery как evaluation target.** Segmented evaluation (familiar vs unfamiliar non-habitual streams) — уникальный contribution GLIDE. Другие работы оценивают aggregate Recall/CTR без подобного разбиения.
1. **Semantic grounding как отдельная фаза.** GLIDE явно инициализирует SID embeddings через bidirectional text↔SID tasks. Meituan использует collaborative SID но без text grounding; Google используют SID в ranking без LLM grounding.
1. **Offline evaluation без online A/B.** В отличие от Google (online A/B) и Meituan (online lift), GLIDE представляет только offline результаты. Это ограничение — авторы признают, что online validation является следующим шагом.

### Общие themes deployment GR

Несмотря на различия, все deployment работы сходятся в нескольких ключевых наблюдениях:

- **Beam search обязателен:** во всех работах greedy decoding существенно хуже. Constrained beam search с trie — стандарт для GR serving.
- **Vocabulary expansion требует инициализации:** случайная инициализация SID embeddings без grounding ведёт к плохому quality. Нужна явная связка SID с content semantics.
- **Cold-start выигрыш реален:** все работы отмечают улучшение recall для новых items. Это ключевое практическое преимущество GR перед DR в production.
- **SID mapping version management сложен:** при обновлении каталога нужно аккуратно синхронизировать tokenizer, trie и модель. Meituan и Spotify оба указывают на это как на infrastructure challenge.

## Итог

Статья показывает, что SID-based GR можно довести до крупного media product, но успех зависит от grounding, personalization и сегментированной оценки discovery. Semantic IDs здесь являются интерфейсом между LLM и каталогом, а не просто способом сжать item IDs.

## Источники

- [arXiv:2603.17540](https://arxiv.org/abs/2603.17540) , PDF/source.
