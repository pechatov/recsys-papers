---
title: "Learning Without Missing-At-Random Prior Propensity-A Generative Approach for Recommender Systems"
category: "generative_retrieval"
slug: "learning_without_mnar_prior_propensity_generative_recommender_summary"
catalogId: "paper-learning_without_mnar_prior_propensity_generative_recommender_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/learning_without_mnar_prior_propensity_generative_recommender_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1109/TKDE.2024.3490593"
---
Подробное саммари статьи:

> **Авторы:** Yuanbo Xu, Fuzhen Zhuang, En Wang, Chaozhuo Li, Jie Wu.
>
> **Аффилиации:** Jilin University; Beihang University; Beijing University of Posts and Telecommunications; Temple University.
>
> **Публикация:** IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 2, pp. 754-765, 2025. DOI зарегистрирован в 2024 году.
>
> **Источники:** [DOI / IEEE](https://doi.org/10.1109/TKDE.2024.3490593), [DBLP](https://dblp.org/rec/journals/tkde/XuZWLW25), [J-GLOBAL metadata](https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=202502269580116397), [авторская PDF-версия](https://cis.temple.edu/~jiewu/research/publications/Publication_files/Learning_from_Miss_Not_At_Random_data_without_Missing_At_Random_Prior_Propensity_in_Recommender_Systems_TKDE__Copy_.pdf).

## 1. Коротко

Статья разбирает проблему **MNAR** в recommender systems: наблюдаемые ratings не являются случайной выборкой из всех потенциальных ratings. Пользователи чаще ставят оценки item'ам, к которым у них сильное отношение, поэтому обучение только на observed ratings даёт biased preference estimates и усиливает Matthew effect.

Типичный способ борьбы с MNAR - оценивать propensity через отдельные MAR feedbacks. Авторы считают это непрактичным: MAR-данные дорого собирать, они есть только в небольших публичных датасетах, а многие методы проверялись на synthetic или small-scale setups. Предложение статьи: рассматривать **user preference** как common prior propensity, которая связывает MAR и MNAR generative processes. Практическая реализация - lightweight iterative probabilistic matrix factorization, **lightIPMF**.

## 2. Контекст

MAR assumption говорит, что вероятность наблюдать rating не зависит от самого потенциального rating value. В реальных рекомендациях это обычно неверно: пользователь скорее поставит 5 или 1, чем нейтральную оценку, а absence rating может означать не отсутствие интереса, а отсутствие экспозиции. Поэтому direct matrix completion по observed matrix `Ro` восстанавливает не ground-truth preference distribution, а biased distribution.

Статья формулирует missing mechanism через observation indicator matrix `O`. В MNAR rating distribution и observation process нельзя разорвать: rating value влияет на вероятность того, что рейтинг будет наблюдаться. Авторы предлагают явно моделировать три фактора: user preference `X`, observed ratings `Ro` и latent prior propensity `Z`.

## 3. Метод и pipeline

1. **Data consistency hypothesis.** Между MAR и MNAR есть общий фактор: пользовательское preference representation. Его можно извлекать даже из MNAR и использовать как приближение prior propensity.
1. **User Preference Model.** Модель оценивает latent user preference `X` , не требуя отдельного MAR supervision.
1. **Observation Prediction Model.** Observation `O` моделируется как зависящий от observed rating estimates, user preference и latent prior `Z` . Для explicit ratings continuous estimate разбивается на 5 уровней.
1. **Rating Prediction Model.** Полная rating matrix предсказывается через low-rank user/item factors и latent prior. Это остаётся MF-подобной моделью, но оптимизируется вместе с missing mechanism.
1. **Iterative optimization.** lightIPMF использует EM-like итерации: обновляет preference-related параметры, observation-related параметры и rating estimates, минимизируя joint likelihood для ratings, observations и preferences.

## 4. Результаты и evidence

Эксперименты поставлены на четырёх real-world datasets: Yahoo, Coat, ML10M и Amazon. Yahoo и Coat содержат MAR/MNAR setup, а ML10M и Amazon представляют сценарии без MAR prior propensity. Baselines: PMF, MF-MNAR, MF-IPS, MF-DRJL, TPMF, PVAE, not-MIWAE и GINA. Метрики: MSE, AUC и NDCG для top-10 recommendations.

<div class="table-scroll">
<table>
<thead><tr><th>Датасет</th><th>Ключевые результаты lightIPMF</th></tr></thead>
<tbody>
<tr><td>Yahoo</td><td>MSE 1.295, AUC 0.721, NDCG 0.830; gain над лучшим baseline: 4.78%, 2.86%, 6.41%.</td></tr>
<tr><td>Coat</td><td>MSE 0.714, AUC 0.844, NDCG 0.754; gain: 4.03%, 4.12%, 2.72%.</td></tr>
<tr><td>ML10M</td><td>MSE 1.623, AUC 0.621, NDCG 0.651; особенно сильный gain по NDCG: 14.6%.</td></tr>
<tr><td>Amazon</td><td>MSE 1.801, AUC 0.661, NDCG 0.613; gain по AUC 10.4%, по NDCG 9.27%.</td></tr>
</tbody>
</table>
</div>

Авторы отдельно подчёркивают: методы, которым нужен MAR prior propensity, выглядят сильнее на Yahoo/Coat, но их точность ухудшается на ML10M/Amazon, где такой prior недоступен. VAE-based методы могут неявно извлекать знание из MNAR, но lightIPMF выигрывает за счёт явного preference modeling.

## 5. Ограничения

- Центральная гипотеза о user preference как common prior propensity эмпирически поддержана, но всё равно является modeling assumption.
- Модель ориентирована на explicit ratings и matrix factorization style; она не решает cold-start item understanding через content features.
- lightIPMF улучшает MNAR rating prediction, но не является генеративным retriever'ом и не снимает инфраструктурную проблему candidate generation при большом каталоге.
- Полный IEEE final PDF был недоступен напрямую, но авторская PDF-версия и метаданные совпадают по названию, авторам и DOI.

## 6. Связь с GR/SID

Эта работа полезна для GR/SID не архитектурой output IDs, а bias-постановкой. Generative retrieval с semantic IDs часто оптимизируется по observed interaction logs, а такие логи тоже MNAR: item не был выбран не обязательно потому, что пользователь его не хотел. lightIPMF показывает, что generative recommendation pipeline должен различать preference, exposure/observation и rating/relevance. Для SID-систем это может означать отдельное моделирование propensity или debiasing loss перед обучением генератора SID.
