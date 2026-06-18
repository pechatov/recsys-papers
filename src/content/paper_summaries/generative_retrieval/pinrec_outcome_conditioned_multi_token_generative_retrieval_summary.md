---
title: "PinRec: Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale Recommendation Systems"
category: "generative_retrieval"
slug: "pinrec_outcome_conditioned_multi_token_generative_retrieval_summary"
catalogId: "paper-pinrec_outcome_conditioned_multi_token_generative_retrieval_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/pinrec_outcome_conditioned_multi_token_generative_retrieval_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2504.10507"
---
> **Авторы:**
>
> Prabhat Agarwal, Anirudhan Badrinath, Laksh Bhasin, Jaewon Yang, Edoardo Botta, Jiajing Xu, Charles Rosenberg.
>
>   
>
>
> **Аффилиация:**
>
> Pinterest
>
> , Palo Alto, CA, USA.
>
>   
>
>
> **Источник:**
>
> arXiv:2504.10507, версия v6 от 2026-03-04. В статье заявлено, что код будет опубликован после публикации; публичного production-датасета нет.

## 1. Коротко

PinRec - industrial generative retrieval system для Pinterest Homefeed, Search и Related Pins. Это не SID-only работа: модель генерирует dense item embeddings, а затем через ANN достает реальные Pins. Главный вклад - показать, как генеративный retrieval можно довести до web-scale production, где важны не только Recall@K, но и controllability по разным outcomes, diversity, latency, cost и интеграция с существующим ranking/blending stack.

Две ключевые идеи: **outcome-conditioned generation** и **temporal multi-token generation**. Outcome conditioning позволяет на инференсе сказать модели, какой тип результата нужен: repin/save, grid click, long grid click, outbound click или surface/context. Multi-token generation заставляет модель за один decode step выдавать несколько будущих item representations на разных temporal offsets, что одновременно снижает стоимость autoregressive rollout и расширяет разнообразие candidate set.

Для линии semantic IDs работа особенно важна как сильный industrial counterexample. Авторы прямо пишут, что пробовали Semantic IDs, но столкнулись с representational collapse и худшими offline metrics, поэтому выбрали dense embeddings. Это не доказывает, что SID всегда хуже, но задает жесткую планку: compact discrete identifiers должны сохранять capacity на очень большом, визуально-семантическом и быстро меняющемся каталоге.

## 2. Контекст и мотивация

Pinterest описывает масштаб как более 550M monthly active users и более 1.5B saved Pins per week. Retrieval stage должен выбрать тысячи кандидатов для последующего ranking. Если retrieval stage не нашел свежие, разнообразные и релевантные объекты, downstream ranker уже не сможет их показать.

Классический industrial baseline - two-tower retrieval: user/query tower строит embedding запроса, item tower строит embedding объекта, ANN ищет ближайшие items. Generative retrieval заменяет это на sequence model: по истории пользователя модель генерирует representation следующего item или последовательности items. Проблема в том, что академические GR-методы часто проверяются на малых benchmark'ах, обычно оптимизируют один target action и не показывают, как обслуживать модель в реальном latency/cost budget.

PinRec формулирует три production-требования:

- **Multi-objective controllability.** Система должна управлять балансом между saves/repins, clicks, long clicks, outbound clicks и surface-specific goals.
- **Diversity.** Один next-item prediction легко переобучается на самый недавний intent и дает узкий candidate set; feed требует покрытия разных пользовательских интересов.
- **Efficiency.** Autoregressive генерация нескольких embeddings дороже, чем один forward pass two-tower retrieval; без serving оптимизаций GR не пройдет production constraints.

## 3. Problem setup

Система имеет пользователей, items, actions и surfaces. История пользователя - хронологическая последовательность tuple'ов вида item, action, surface, timestamp. Items гетерогенны: Pins, search queries, boards и потенциально другие сущности. Surfaces тоже разные: Homefeed, Search, Related Pins, board pages и т.п.

Задача модели - по истории пользователя и request context сгенерировать ordered list candidate item representations. Эти representations не являются финальной выдачей: они идут в nearest-neighbor retrieval, а найденные items далее проходят ranking и blending. Поэтому PinRec нужно оценивать как candidate generator, а не как end-to-end recommender.

## 4. Архитектура PinRec

### 4.1. Backbone

Backbone - causal Transformer decoder. На вход подается featurized user journey с learned positional embeddings. Causal mask запрещает смотреть в future interactions, поэтому hidden state на позиции $t$ кодирует только историю до $t$. На выходе каждый hidden state $h_t$ передается в output head, который генерирует representation будущего item.

В основном offline setup используется 12-layer Transformer с 12 attention heads и hidden size 768. Это размер примерно уровня GPT-2 base для dense transformer part, но большая часть capacity вынесена в item vocabulary representations: hash-based learned ID embeddings для Pins имеют примерно 10B sparse parameters.

### 4.2. Outcome-conditioned head

Обычная next-item модель генерирует один representation без явного контроля, какой engagement должна поддержать рекомендация. PinRec добавляет conditioning embeddings для target action и surface/context. Output head получает hidden state $h_t$ и набор conditioning embeddings $c$, после чего выдает outcome-specific item representation:

$$
\hat{e}_{t,c} = f_\theta(h_t, c).
$$

На training model condition'ится на фактический target action. На inference modeler задает бюджет по outcomes. Например, часть retrieve budget можно выделить под repin-oriented candidates, часть под grid-click-oriented candidates, часть под outbound-click-oriented candidates. Это важное отличие от post-hoc reranking: controllability встроена прямо в candidate generation.

### 4.3. Temporal multi-token generation

В language modeling multi-token prediction обычно означает несколько output heads для позиций $t+1, t+2,...$. В recommender systems позиционный индекс менее надежен: у активного пользователя следующий action может случиться через секунды, у dormant user - через дни. Поэтому PinRec condition'ит output head не только на future position, но и на relative temporal distance.

Операционно это выглядит так: из одного hidden state модель генерирует несколько embeddings для разных temporal offsets, например now, 30 seconds later, 1 minute later и дальше. Для autoregressive continuation используется representation immediate next token, а остальные generated embeddings идут в retrieval budget. Это позволяет получить больше candidate embeddings за меньше decode steps.

### 4.4. Input sequence representation

PinRec подробно разбирает, как кодировать гетерогенные события:

- **Pins.** Pin representation объединяет pretrained OmniSage semantic embedding и learned hash-based ID embeddings. OmniSage кодирует visual/textual content и engagement through Pin-board graph. ID embeddings добавляют memorization/collaborative capacity для конкретных Pins.
- **Search queries.** Для queries авторы используют OmniSearchSage semantic representation; ID embeddings для queries не дали offline value.
- **Temporal features.** Абсолютное время кодируется через sinusoidal transforms с daily, weekly и seasonal periods. Relative time since previous action кодируется log-scale frequencies с learnable phase parameters.
- **Surface embeddings.** Surface добавляется к sequence element, чтобы модель различала, где случилось взаимодействие и на какой surface делается запрос.

Важный engineering choice: авторы сравнивали dense embeddings и Semantic IDs. Semantic IDs часто схлопывали много items в один identifier и давали ниже metrics; dense representation оказался более надежным для Pinterest-scale каталога.

## 5. Training objective

Обучение использует teacher forcing и sampled softmax. Для позиции $t$ модель предсказывает future item representation, затем считается similarity между predicted representation и candidate item embedding. В similarity добавляется bias correction term, оцененный через count-min sketches, чтобы учитывать неравномерность sampled/in-batch negatives.

Базовая цель - максимизировать similarity target item относительно negative samples. Для multi-token prediction targets берутся из нескольких будущих позиций. Для outcome-conditioned generation conditioning values берутся из actual target item action/surface.

Отдельная важная деталь - **intra-feed relaxation**. В естественном языке next token обычно строго определен. В feed interaction последовательности слабее: пользователь может кликнуть несколько элементов внутри одной feed session в почти произвольном порядке. Поэтому PinRec допускает target set из future items в той же feed session и оптимизирует best match внутри этого set, а не только immediate next item. Практически это ближе к вопросу: может ли модель предсказать любой будущий engaged item в текущей session?

## 6. Inference: как из embeddings получаются кандидаты

1. **Собрать историю.** Signal Service возвращает batch + real-time события пользователя; для contextual surfaces добавляется request context, например search query или closeup Pin.
1. **Сгенерировать embeddings.** UC variant генерирует без outcome conditioning. OC variant генерирует embedding для каждого requested outcome. MT variant дополнительно генерирует несколько temporal-offset embeddings за один step.
1. **Разложить budget.** Общий retrieval budget распределяется между generated embeddings. Для OC budget задается по action types, чтобы, например, repin и click candidates получили нужную долю.
1. **Сжать похожие embeddings.** Если generated embeddings имеют cosine similarity выше threshold, они объединяются, а budgets суммируются. Это снижает overlap в ANN retrieval.
1. **ANN retrieval.** Faiss IVF-HNSW index на CPU ищет ближайшие Pins по inner product. Retrieval делает overfetch, после чего candidates дедуплицируются и идут в downstream ranking/blending.

## 7. Serving architecture

PinRec serving - один из самых ценных разделов статьи, потому что он показывает, какие части нужны, чтобы generative retrieval работал в production.

- **Signal fetching.** Lambda architecture: daily Spark batch хранит год positive engagements и search queries; real-time RocksDB store добавляет события после batch cutoff. Batch и real-time signals дедуплицируются.
- **Transport compression.** Sequence embeddings передаются как INT8 и dequantize'ятся в FP16 перед inference, чтобы снизить transport cost.
- **ID embedding service.** Pin ID embedder требует много памяти, поэтому вынесен в отдельный CPU memory-optimized TorchScript service, чтобы не конкурировать с main model за GPU memory. Заявленная latency - single-digit milliseconds.
- **Transformer inference.** Main model served на NVIDIA L40S через NVIDIA Triton ensemble и Python backend. Используются CUDA Graphs, torch.compile и KV cache: prefill keys/values для истории кешируются, decode steps пересчитывают только новые sequence elements.
- **ANN retrieval.** Faiss IVF-HNSW index работает на CPU hosts с batching через Triton.

Appendix дает полезные serving numbers: PinRec variants используют около 1.6GiB GPU memory с KV cache и CUDA graphs; без этих служебных структур main dense model порядка 250MiB. На 80 QPS PinRec-OC / OC,MT работают около p50 40ms и p90 65ms для 6 steps; PinRec-UC с 3 steps около p50 23ms и p90 46ms. Это примерно 3-4x p50 latency two-tower retrieval, но авторы утверждают, что parallel RPC execution дает менее 1% added end-to-end latency.

## 8. Offline evaluation

Offline evaluation построен на users, disjoint from training users. Модель autoregressively генерирует output representations без teacher forcing, затем проверяется, поднимает ли generated representation будущие target item embeddings из index с random negatives. Основная метрика - unordered recall@10: target считается найденным, если хотя бы один из generated embeddings достал его в top-10. Diversity оценивается как proportion of unique retrieved items across users.

### 8.1. Main comparison

<div class="table-scroll">
<table>
<thead><tr><th>Model</th><th>Homefeed R@10</th><th>Related Pins R@10</th><th>Search R@10</th><th>Комментарий</th></tr></thead>
<tbody>
<tr><td>SASRec</td><td>0.382</td><td>0.426</td><td>0.142</td><td>Transformer sequential baseline.</td></tr>
<tr><td>PinnerFormer</td><td>0.461</td><td>0.412</td><td>0.257</td><td>Pinterest user representation baseline.</td></tr>
<tr><td>TIGER</td><td>0.208</td><td>0.230</td><td>0.090</td><td>SID-based generative recommendation baseline.</td></tr>
<tr><td>HSTU-OC</td><td>0.596</td><td>0.539</td><td>0.179</td><td>Architecture ablation with HSTU-like block.</td></tr>
<tr><td>PinRec-UC</td><td>0.608</td><td>0.521</td><td>0.350</td><td>Unconditioned, no multi-token.</td></tr>
<tr><td>PinRec-OC</td><td>0.625</td><td>0.537</td><td>0.352</td><td>Outcome-conditioned, no multi-token.</td></tr>
<tr><td><strong>PinRec-MT,OC</strong></td><td><strong>0.676</strong></td><td><strong>0.631</strong></td><td><strong>0.450</strong></td><td>Outcome-conditioned + temporal multi-token.</td></tr>
</tbody>
</table>
</div>

PinRec-MT,OC - strongest offline model on all three surfaces. Relative to PinRec-UC it gives about +11.2% on Homefeed, +21.1% on Related Pins и +28.6% on Search by unordered recall@10. TIGER особенно слаб на Pinterest data, что согласуется с тезисом авторов о SID collapse/capacity issue в этом setup.

### 8.2. Outcome conditioning

Проверка controllability устроена просто: весь budget выделяется одному action outcome, и авторы смотрят, растет ли recall именно для этого фактического action. Matching outcome conditioning дает lift: true repins +1.9%, grid clicks +1.4%, long grid clicks +4.8%, outbound clicks +6.2%. Если conditioning не совпадает с actual action, recall падает, особенно для outbound clicks. Это важно: conditioning не просто добавляет шумный feature, а реально двигает generated representation в разные outcome-specific области.

### 8.3. Multi-token trade-off

Авторы держат total number of generated embeddings fixed и меняют число embeddings per generation step. При 16 generations per step они сообщают примерно 10x latency reduction относительно PinRec-OC, одновременно +16.0% unordered recall и +21.3% diversity. Интерпретация: multi-token generation не только ускоряет rollout, но и снижает recency bias, потому что temporal offsets заставляют модель покрывать несколько будущих horizons.

## 9. Online A/B experiments

Online PinRec работает как дополнительный candidate generator: найденные candidates проходят через existing ranking и blending без специальных изменений. Это делает результаты ближе к реальному product impact, но также означает, что observed lifts зависят от всего production stack, а не только от retrieval model.

### 9.1. Homefeed

Для Homefeed авторы тестируют пять budget configurations для PinRec-OC и пять для PinRec-MT,OC. Fixed budget оставлен для outbound click, long grid clicks и unspecified actions; remaining budget делится между repin и grid click с шагом 0.2. Pareto plots показывают ожидаемый control: больше repin budget - выше repin rate и ниже grid click rate.

<div class="table-scroll">
<table>
<thead><tr><th>Metric</th><th>PinRec-UC</th><th>PinRec-OC</th><th>PinRec-MT,OC</th></tr></thead>
<tbody>
<tr><td>Fulfilled Sessions</td><td>+0.02%</td><td>+0.21%</td><td><strong>+0.28%</strong></td></tr>
<tr><td>Time Spent</td><td>-0.02%</td><td>+0.16%</td><td><strong>+0.55%</strong></td></tr>
<tr><td>Unfulfilled Sessions</td><td>-0.28%</td><td><strong>-0.89%</strong></td><td>-0.78%</td></tr>
<tr><td>Site-wide Grid Clicks</td><td>+0.58%</td><td><strong>+1.76%</strong></td><td>+1.73%</td></tr>
<tr><td>Homefeed Grid Clicks</td><td>+1.87%</td><td><strong>+4.01%</strong></td><td>+3.33%</td></tr>
</tbody>
</table>
</div>

Outcome conditioning улучшает fulfilled/unfulfilled session metrics и clicks. Multi-token variant сильнее по fulfilled sessions и time spent, но чуть ниже OC по Homefeed grid clicks. Это нормальный production trade-off: diversity и session-level quality могут не совпадать с максимизацией одного click metric.

### 9.2. Search и promoted products

<div class="table-scroll">
<table>
<thead><tr><th>Search metric</th><th>PinRec-UC lift</th></tr></thead>
<tbody>
<tr><td>Search Fulfillment Rate</td><td>+2.27%</td></tr>
<tr><td>Search Repins</td><td>+4.21%</td></tr>
<tr><td>Search Grid Clicks</td><td>+3.00%</td></tr>
</tbody>
</table>
</div>

В promoted product retrieval те же embeddings дают CPA -1.83% и shopping conversion volume +1.87% в US. Это полезный сигнал, что generated representations несут не только organic engagement signal, но и commercial relevance, хотя детали ads/ranking stack закрыты.

### 9.3. Related Pins

<div class="table-scroll">
<table>
<thead><tr><th>Related Pins metric</th><th>PinRec-OC lift</th></tr></thead>
<tbody>
<tr><td>Fulfilled Sessions</td><td>+0.26%</td></tr>
<tr><td>Time Spent</td><td>+0.30%</td></tr>
<tr><td>Unfulfilled Sessions</td><td>-1.07%</td></tr>
<tr><td>Site-wide Grid Clicks</td><td>+0.50%</td></tr>
<tr><td>Sitewide Repins</td><td>+0.63%</td></tr>
</tbody>
</table>
</div>

## 10. Appendix ablations and diagnostics

### 10.1. Architecture: Transformer vs HSTU

Outcome conditioning и multi-token prediction не завязаны на vanilla Transformer. Авторы проверяют HSTU-style architecture: Homefeed 0.596, Related Pins 0.539, Search 0.179 против PinRec-OC 0.625/0.537/0.352. Related Pins примерно конкурентен, но Search резко хуже. Вывод осторожный: идеи PinRec architecture-agnostic, но конкретная architecture важна для surface mix.

### 10.2. Negative sampling

Random negatives и in-batch negatives оба важны для sampled softmax. Увеличение random negatives с 8K до 16K дает +2.5% Homefeed recall и +1.3% all-surface recall. Масштабирование in-batch negatives сильнее: 1.5x дает +4.2% Homefeed, 2x +8.4%, 12x +18.3%. Это подчеркивает, что GR retrieval quality сильно зависит от training negative corpus, а не только от architecture.

### 10.3. Parameter scaling

Transformer scaling почти не окупается: medium -0.8%, large -0.9%, XL +2.2% recall относительно base примерно 100M transformer parameters, но XL становится непрактичным для real-time serving. Зато vocabulary scaling через hash-based ID embeddings дает большой эффект: +14.0% Homefeed и +14.4% all-surface recall. Для Pinterest это означает, что sparse item-specific capacity важнее, чем просто увеличивать dense Transformer.

### 10.4. Tail users and tail items

Outcome conditioning не ломает tail/new cohorts. Для new users PinRec-UC/OC дают grid clicks +2.60%/+2.62% и outbound clickthroughers +3.98%/+5.78%. Для resurrected users grid clicks +1.56%/+1.86%, outbound CTs +1.98%/+1.72%. По corpus coverage PinRec-UC/OC дают +6.2%/+17.3% unique pins против production two-tower; для 1-impression tail Pins +5.8%/+7.7%. Это важный контраргумент против опасения, что outcome conditioning будет усиливать только head/popular content.

### 10.5. Qualitative multi-token examples

В qualitative figures авторы показывают, что PinRec-OC сильнее привязан к недавнему intent, например только nails или только food, а PinRec-MT,OC возвращает mix, который учитывает более старые interests вроде tattoo/nursery. Это не доказательство само по себе, но оно согласуется с offline diversity и online session-level lifts.

## 11. Что реально является вкладом

- **Production formulation of controllable GR.** Outcome conditioning встроен в generation stage, а не только в ranking/blending.
- **Temporal multi-token retrieval.** Multi-token prediction адаптирован под recommender time dynamics, а не перенесен буквально из language modeling.
- **Dense generative retrieval at Pinterest scale.** Работа показывает, что GR может генерировать dense embeddings и использовать ANN, не обязательно discrete item tokens.
- **Serving disclosure.** Triton ensemble, ID embedding service, INT8 transport, FP16 inference, CUDA Graphs, KV cache и CPU Faiss описаны достаточно конкретно, чтобы понять production constraints.
- **Online evidence.** Есть не только offline Recall@10, но и A/B на Homefeed, Search, Related Pins и promoted product retrieval.

## 12. Сильные стороны

- **Сильная production валидность.** Это редкая GR-статья с online A/B, несколькими surfaces и serving деталями.
- **Честный фокус на trade-offs.** Авторы не скрывают, что GR дороже two-tower; вместо этого показывают, какие optimizations делают latency приемлемой.
- **Controllability проверена напрямую.** Outcome conditioning проверяется action-stratified offline и budget-controlled online Pareto behavior.
- **Appendix полезен.** Negative sampling, parameter scaling, tail cohorts и serving efficiency дают больше информации, чем типичный top-line-only industrial paper.
- **Важный результат для SID literature.** TIGER/SID baseline проигрывает сильно; authors explicitly report SID collapse, что делает работу обязательной для чтения рядом с semantic-tokenization papers.

## 13. Ограничения и открытые вопросы

- **Reproduction почти невозможен.** Данные, production surfaces, Signal Service, OmniSage/OmniSearchSage embeddings, ranking/blending stack и traffic allocation закрыты.
- **SID diagnosis недостаточно глубокий.** Статья говорит о representational collapse и низких metrics, но не дает полного анализа tokenizer setup, codebook size, collision distribution, retraining schedule и SID-specific mitigations.
- **Offline index uses random negatives.** Это стандартный промышленный компромисс, но random-negative retrieval легче full-catalog retrieval; абсолютные recall numbers нельзя напрямую переносить на полный Pinterest catalog.
- **Online attribution смешанная.** PinRec candidates проходят existing ranking/blending. Lifts показывают product impact, но не полностью изолируют retrieval contribution.
- **Statistical details ограничены.** Для части online tables нет confidence intervals / p-values в таблицах, хотя на Pareto plots показаны error bars.
- **Cost model неполный.** Есть latency и memory numbers, но нет полного cost breakdown по GPU/CPU/Faiss/embedding-service capacity и traffic share.
- **Outcome control может конфликтовать с долгосрочным quality.** Budget по clicks/repins управляемый, но не до конца понятно, как он влияет на long-term satisfaction, creator ecosystem и filter bubbles.

## 14. Итоговая оценка

PinRec - одна из самых полезных industrial работ по generative retrieval, потому что она переносит разговор из "может ли Transformer предсказать next item" в "может ли generative retrieval быть controllable, diverse, efficient и online-positive в реальной системе". Главный технический урок: production GR - это не только model architecture, а совместный дизайн representations, objectives, budget allocation, ANN retrieval, serving и downstream integration.

Доверять работе как evidence for production feasibility можно достаточно сильно: есть offline comparisons, ablations и online A/B на нескольких surfaces. Доверять ей как evidence against all Semantic IDs нужно осторожно: показан провал конкретного SID setup на Pinterest, но не исчерпывающая оптимизация SID family. Тем не менее для semantic-ID исследований PinRec задает сильный baseline: dense generative retrieval с large sparse ID vocabulary, outcome conditioning и multi-token temporal generation.

## 15. Первичные источники

- arXiv abstract: [https://arxiv.org/abs/2504.10507](https://arxiv.org/abs/2504.10507)
- arXiv HTML v6: [https://arxiv.org/html/2504.10507](https://arxiv.org/html/2504.10507)
