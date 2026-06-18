---
title: "PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations"
category: "semantic_ids_tokenization_indexing"
slug: "plum_adapting_pre_trained_language_models_for_industrial_summary"
catalogId: "paper-plum_adapting_pre_trained_language_models_for_industrial_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/plum_adapting_pre_trained_language_models_for_industrial_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2510.07784"
---
**Авторы:** Ruining He, Lukasz Heldt, Lichan Hong, Raghunandan Keshavan, Shifan Mao, Nikhil Mehta, Zhengyang Su, Alicia Tsai, Yueqi Wang, Shao-Chuan Wang, Xinyang Yi, Lexi Baugher, Baykal Cakici, Ed Chi, Cristos Goodrow, Ningren Han, He Ma, Romer Rosales, Abby Van Soest, Devansh Tandon, Su-Lin Wu и др.

**Аффилиации:** Google DeepMind; YouTube.

**Индустрия:** YouTube candidate generation на миллиардном видеокаталоге.

**Первичный источник:** arXiv source 2510.07784.

## Коротко

- PLUM адаптирует pretrained LLM к industrial generative recommendation.
- Pipeline: Semantic IDs, continued pre-training, task-specific SFT для generative retrieval.
- Фреймворк уже используется в YouTube для long-form и short-form recommendations.

## Контекст

- YouTube retrieval обычно опирается на large embedding models и multi-stage serving.
- LLM sequence modeling может заменить часть candidate generation, если научить ее item space.
- Semantic IDs становятся интерфейсом между видео-каталогом и language model.

## Проблема

- Pretrained LLM не знает динамический YouTube corpus.
- Fine-tuning без CPT может плохо освоить доменные co-occurrences.
- Serving миллиардов пользователей требует constrained generation, latency engineering и continuous monitoring.

## Метод/архитектура

- Item tokenizer строит Semantic IDs для videos.
- CPT обучает модель на domain-specific sequence data.
- SFT target - SID tokens next watch.
- Prompt включает watch history, user features и context video features.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Objective/алгоритм

- Autoregressive loss применяется к target SID sequence.
- CPT может использовать sequences без instruction overhead, чтобы модель быстро освоила SID structure.
- Scaling study проверяет Gemini-1.5 MoE family: 110M, 370M, 900M, 3B activated parameters.
- Inference methods адаптированы к production launch.

## Детальный алгоритм PLUM

PLUM - это full-stack pipeline, где pretrained language model сначала учится языку YouTube Semantic IDs, а затем дообучается на конкретную next-watch retrieval задачу. Важный момент: SID vocabulary, train/eval time split и constrained serving являются частью алгоритма, а не деталями вокруг него.

1. **Зафиксировать corpus и SID mapping.** Миллиардный video catalog превращается в semantic identifiers; mapping video -> SID и обратный индекс должны быть версионированы.
1. **Собрать sequence corpus для CPT.** Модель получает YouTube watch sequences/user-context sequences в формате, близком к будущему retrieval input, но без лишнего instruction overhead.
1. **Continued pre-training.** Pretrained Gemini-1.5 MoE model адаптируется к распределению SID tokens, co-watch patterns и доменным sequence conventions.
1. **Собрать SFT examples.** Prompt включает до 100 recent watches, user features и context video features; target - SID next watch.
1. **Task-specific SFT.** Autoregressive loss применяется к target SID tokens. Scaling study сравнивает 110M/370M/900M/3B activated parameters, чтобы найти compute-quality trade-off.
1. **Constrained generation для retrieval.** На inference model генерирует SID candidates; outputs валидируются через SID mapping/trie и передаются в downstream YouTube retrieval/ranking stack.
1. **Production monitoring.** Отдельно считаются valid-path rate, stale IDs, head/tail coverage, freshness, latency и downstream business metrics, потому что HR@K сам по себе недостаточен.

```
sid_map = build_semantic_ids(video_catalog_version)
cpt_sequences = build_watch_sequences(logs, sid_map, without_instruction_overhead=True)
model = continued_pretrain(pretrained_LM, cpt_sequences)

sft_examples = []
for session in training_logs:
    prompt = pack_recent_watches_user_features_context(session, max_tokens=1536)
    target = sid_map[session.next_watch]
    sft_examples.append((prompt, target))

model = supervised_finetune(model, sft_examples, loss="autoregressive_sid")
candidates = constrained_generate(model, serving_prompt, valid_sid_index)
return map_sids_to_videos(candidates)
```

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Эксперименты

- Dataset: production YouTube surface для next-watch recommendation.
- Corpus содержит миллиарды videos.
- Train: 7 continuous days from July 2025; evaluation: Day 8.
- Experiments conducted with 1024 Google v6e TPUs; 4 trainers по 256 TPUs.

### Что важно при чтении

В этой секции статьи стоит отделять заявленный conceptual contribution от инженерного механизма: для PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations качество возникает из сочетания representation design, training objective и evaluation protocol.

Если переносить метод в другую систему, нельзя копировать только название компонента: нужно воспроизвести входные данные, формат идентификаторов, ограничения decoding и slices, на которых авторы показывают выигрыш.

## Рисунки/таблицы

- Framework figures раскрывают PLUM stages.
- Scaling plots связывают loss/retrieval metric/model size/compute.
- CPT ablations показывают, зачем нужна доменная адаптация.
- Production notes обсуждают online A/B сравнение с LEM-based retrieval.

## Ablation conclusions

- Без CPT модель хуже адаптируется к SID distribution.
- Semantic ID enhancements нужны для quality/freshness.
- Scaling дает не только lower loss, но и retrieval metric improvements до определенного compute-optimal point.

## Сильные стороны

- **Редкий full-stack industrial paper.** PLUM покрывает SID construction, CPT, SFT, scaling, inference и production rollout, а не только model architecture.
- **Показывает роль continued pre-training.** Pretrained LLM не знает YouTube SID language; CPT явно закрывает этот gap.
- **Масштаб реалистичен.** Billions of videos, 100 recent watches, 1536 input tokens и 1024 v6e TPUs дают production-level контекст.
- **Scaling study полезен инженерно.** Сравнение activated parameters помогает выбирать compute-optimal point, а не просто "больше модель лучше".
- **Связано с реальным serving.** В summary есть inference/production notes, а не только offline Recall.

## Ограничения

- **Многие details закрыты.** Production data, exact online metrics, reward/serving details и часть SID enhancements недоступны для независимой проверки.
- **Compute threshold очень высокий.** 1024 Google v6e TPUs и multiple trainers делают прямое воспроизведение недоступным большинству команд.
- **SID freshness критична.** YouTube catalog меняется быстро; stale IDs и old mappings могут разрушить retrieval even if model loss низкий.
- **Popularity bias может усилиться.** Autoregressive LM легко учит frequent paths и может терять novelty/tail coverage.
- **Offline quality не равна production impact.** Latency, constrained decoding, ranking handoff и freshness filters могут съесть retrieval gains.

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
- Специфично для этой статьи: Многие детали production data и online metrics закрыты.
- Специфично для этой статьи: Метод требует огромной compute infrastructure.
- Специфично для этой статьи: Риски: invalid IDs, stale IDs, popularity bias, serving latency.

## Связь

- Связан с Better Generalization with Semantic IDs как следующий YouTube-scale шаг.
- Связан с FORGE и UniSearch через production SID serving.

## Итог

- PLUM показывает, что LLM для recommendation требует доменного pretraining, а не только prompt engineering.
- Semantic IDs превращают каталог в язык, но production успех зависит от всей системы.

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
