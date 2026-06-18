---
title: "IDNP: Interest Dynamics Modeling Using Generative Neural Processes for Sequential Recommendation"
category: "generative_retrieval"
slug: "idnp_interest_dynamics_generative_neural_processes_summary"
catalogId: "paper-idnp_interest_dynamics_generative_neural_processes_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/idnp_interest_dynamics_generative_neural_processes_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1145/3539597.3570373"
---
Подробное саммари статьи:

> **Авторы:** Jing Du, Zesheng Ye, Lina Yao, Bin Guo, Zhiwen Yu.
>
> **Аффилиации:** The University of New South Wales; CSIRO Data61 and UNSW; Northwestern Polytechnical University.
>
> **Публикация:** WSDM 2023, Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining, pp. 481-489.
>
> **Источники:** [ACM DOI](https://doi.org/10.1145/3539597.3570373), [arXiv:2208.04600](https://arxiv.org/abs/2208.04600), [IR Anthology](https://ir.webis.de/anthology/2023.wsdm_conference-2023.56/), [DBLP](https://dblp.org/rec/conf/wsdm/DuY00023).

## 1. Коротко

IDNP переносит **generative Neural Processes** в sequential recommendation. Авторы критикуют стандартную постановку, где пользовательский интерес моделируется как последовательность близких во времени кликов: реальные интересы перемешаны, пользователь может перескакивать между темами, а долгосрочная динамика наблюдается редко и дискретно.

Ключевая идея: долгосрочный интерес пользователя можно рассматривать как **непрерывную функцию во времени**, а отдельные short-term sequences - как наблюдения этой функции. Neural Process учится семейству таких функций по пользователям и затем восстанавливает интерес в query timestep даже при малом числе непоследовательных взаимодействий.

## 2. Контекст

Последовательные модели вроде GRU4Rec, Caser, SASRec и NextItNet обычно хорошо ловят локальные паттерны, но их предположение о плотной хронологической зависимости ограничивает few-shot сценарии. Пример из статьи: пользователь может смотреть романтические фильмы из-за актрисы Anne Hathaway, затем переключиться на другой жанр, но долгосрочный интерес к актрисе всё равно объясняет следующий Sci-Fi фильм с её участием. Такая динамика плохо укладывается в один монотонный short-term interest.

IDNP ставит задачу не просто предсказать next item, а восстановить функцию интереса пользователя из контекстных наблюдений. Это сближает recommendation с probabilistic function approximation: модель не задаёт вручную prior, а учит его из набора пользователей.

## 3. Метод и pipeline

1. **Attentive Interest Encoder.** Short-term subsequences кодируются dilated convolutions, чтобы расширить receptive field и поймать skip behaviors. Attention взвешивает важность разных time frames.
1. **User context.** Представления short-term interests агрегируются в permutation-invariant context set. Это важно для Neural Process: context observations не обязаны быть строго последовательными.
1. **Dual Dynamics Inference.** Есть deterministic path, который учитывает query time, и latent path с глобальной latent interest variable. Вместе они задают dynamics-aware представление долгосрочного интереса.
1. **Interest Decoder.** Decoder восстанавливает форму long-term interest function и предсказывает item interactions для query timestep.
1. **Optimization.** IDNP оптимизирует likelihood next-item predictions и расстояние между восстановленной и наблюдаемой функцией интереса; в ablation отдельно проверяется вклад Wasserstein distance.

## 4. Результаты и evidence

Эксперименты идут в few-shot setting на четырёх датасетах:

<div class="table-scroll">
<table>
<thead><tr><th>Датасет</th><th>Users</th><th>Items</th><th>Interactions</th><th>Sparsity</th><th>Тип</th></tr></thead>
<tbody>
<tr><td>MovieLens</td><td>6,040</td><td>3,400</td><td>999,620</td><td>95.16%</td><td>explicit</td></tr>
<tr><td>Gowalla</td><td>29,858</td><td>40,981</td><td>1,027,370</td><td>99.92%</td><td>implicit</td></tr>
<tr><td>Yelp</td><td>31,668</td><td>38,048</td><td>1,561,406</td><td>99.87%</td><td>implicit</td></tr>
<tr><td>Amazon-book</td><td>52,643</td><td>91,599</td><td>2,984,108</td><td>99.94%</td><td>explicit</td></tr>
</tbody>
</table>
</div>

Baselines: Caser, GRU4Rec, SASRec, NextItNet, GRec, MeLU и MetaTL. Метрики: Hit@K, Recall@K, NDCG@K для K = 1, 5, 10. Для few-shot авторы ограничивают длину истории: 20 взаимодействий для MovieLens, 16 для Gowalla, 24 для Yelp и Amazon-book.

IDNP показывает лучший результат по всем датасетам. На MovieLens при K=10 он достигает Hit@10 = 0.3722, Recall@10 = 0.0424 и NDCG@10 = 0.6499; в тексте указаны улучшения над вторыми лучшими методами на 0.98%, 2.92% и 5.57% соответственно. На Yelp особенно заметен прирост NDCG: +7.95% при K=1 и +9.15% при K=10 относительно второго лучшего baseline.

Ablation на MovieLens подтверждает вклад модулей: plain CNN даёт NDCG@10 около 0.5460, добавление dilated convolutions и Neural Process inference постепенно улучшает результат, а полный IDNP достигает 0.7097 NDCG@10 в ablation table.

## 5. Ограничения

- IDNP нацелен на few-shot sequential recommendation, но не решает задачу масштабного retrieval по каталогу через generated IDs.
- Модель сложнее классических sequential baselines: нужно выбирать context size, subsequence length, embedding dimension и training protocol для NP-компонента.
- Эксперименты offline; нет данных о latency и поведении при постоянном online-update пользовательских функций.
- Функциональная интерпретация интереса полезна, но остаётся latent: модель не выдаёт человекочитаемые причины интереса без дополнительного слоя объяснений.

## 6. Связь с GR/SID

IDNP относится к generative recommendation, но не к semantic-ID generative retrieval. Он генерирует/восстанавливает представления пользовательского интереса как функции, а не генерирует item identifiers. Для GR/SID направления статья полезна как напоминание: генеративный retriever с SID обычно фокусируется на output space item'ов, а IDNP показывает ортогональную проблему - как лучше моделировать динамический user context, который потом мог бы служить encoder-side сигналом для генерации SID.
