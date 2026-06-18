---
title: "Creating synthetic datasets for collaborative filtering recommender systems using generative adversarial networks"
category: "generative_retrieval"
slug: "synthetic_datasets_collaborative_filtering_gans_summary"
catalogId: "paper-synthetic_datasets_collaborative_filtering_gans_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/synthetic_datasets_collaborative_filtering_gans_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1016/j.knosys.2023.111016"
---
Подробное саммари статьи:

> **Авторы:** Jesus Bobadilla, Abraham Gutierrez, Raciel Yera, Luis Martinez.
>
> **Аффилиации:** Universidad Politecnica de Madrid; Universidad de Jaen; Universidad de Ciego de Avila.
>
> **Публикация:** Knowledge-Based Systems, vol. 280, article 111016, 2023. Open access.
>
> **Источники:** [DOI / Elsevier](https://doi.org/10.1016/j.knosys.2023.111016), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705123007669), [arXiv:2303.01297](https://arxiv.org/abs/2303.01297), [страница проекта SINBAD2](https://sinbad2.ujaen.es/en/bibcite/keyword/765).

## 1. Коротко

Статья предлагает **GANRS** - метод генерации synthetic collaborative filtering datasets. Цель не улучшить конкретный recommender, а создать параметризуемые наборы данных, похожие на реальные CF-матрицы, чтобы исследователи могли тестировать модели при разных числах users, items, samples и разных уровнях stochastic variability.

Ключевой технический ход: GAN не обучается на огромных sparse rating vectors. Сначала DeepMF переводит users и items в короткие dense continuous embeddings. GAN генерирует samples в этом dense space, а затем k-means clustering возвращает их в discrete sparse формат user-id/item-id/rating, пригодный для обычных CF-экспериментов.

## 2. Контекст

Большинство исследований recommender systems зависит от нескольких популярных датасетов: MovieLens, Netflix, MyAnimeList, FilmTrust и т.п. Это ограничивает эксперименты: трудно отдельно менять число users, число items или sparsity и смотреть, как модель деградирует или улучшается. Synthetic data может дать controlled experimental families, но для CF это сложнее, чем для изображений: качество нельзя оценить визуально.

Авторы подчёркивают, что предыдущие GAN-подходы в recommender systems обычно использовались для data augmentation, denoising, attack/defense или prediction quality. GANRS, наоборот, генерирует полный synthetic dataset как экспериментальный объект.

## 3. Метод и pipeline

1. **DeepMF training.** На исходном CF dataset обучается DeepMF: отдельные embedding layers кодируют users и items, prediction строится через dot product user/item embeddings.
1. **Dense sample extraction.** После обучения берутся compact continuous representations users и items, связанные с rating samples.
1. **GAN generation.** GAN учится генерировать dense fake samples. Random noise std становится управляемым параметром variability.
1. **Discretization.** Так как итоговый CF dataset должен иметь discrete user IDs и item IDs, dense generated user/item vectors группируются двумя k-means процедурами: отдельно для users и items.
1. **Parameter control.** Число clusters задаёт желаемое число synthetic users/items. Число generated samples задаёт размер датасета.
1. **Cleanup.** Повторяющиеся discrete samples удаляются; авторы также обсуждают редкие случаи нескольких ratings для одной пары user-item.

## 4. Результаты и evidence

Три source datasets:

<div class="table-scroll">
<table>
<thead><tr><th>Source</th><th>Users</th><th>Items</th><th>Ratings</th><th>Scores</th><th>Sparsity</th></tr></thead>
<tbody>
<tr><td>MovieLens 100K</td><td>943</td><td>1,682</td><td>99,831</td><td>1-5</td><td>93.71%</td></tr>
<tr><td>Netflix*</td><td>23,012</td><td>1,750</td><td>535,421</td><td>1-5</td><td>98.68%</td></tr>
<tr><td>MyAnimeList</td><td>19,179</td><td>2,692</td><td>548,967</td><td>1-10</td><td>98.94%</td></tr>
</tbody>
</table>
</div>

На Netflix* авторы генерируют family of datasets с разными settings: 100-8000 users, 2000-8000 items, 150K-3M generated samples и разными std noise. Проверяются distributions users-vs-ratings, distribution of rating values, number of repeated samples, MAE/accuracy, precision/recall/F1.

Главный evidence: synthetic datasets воспроизводят expected CF behavior. Например, при росте числа users улучшается accuracy и снижается MAE; precision/recall curves для generated datasets похожи на source Netflix*. Для MovieLens и MyAnimeList авторы также показывают, что generated datasets mimic behavior source datasets. Код и generated datasets заявлены как доступные для исследователей.

## 5. Ограничения

- GANRS не генерирует семантические item descriptions или user profiles; это synthetic collaborative signal, а не полноценный мультимодальный catalog simulator.
- Качество оценивается косвенно через распределения и downstream CF metrics. В CF нет простого визуального теста качества generated samples.
- K-means discretization создаёт повторяющиеся samples, которые приходится удалять; это может менять итоговую density и distribution.
- Метод зависит от source dataset: generated family наследует паттерны исходного MovieLens/Netflix/MyAnimeList.

## 6. Связь с GR/SID

GANRS не является generative retrieval и не создаёт semantic IDs. Но для исследований GR/SID он полезен как инфраструктурная работа: controlled synthetic CF datasets могут помочь проверять, как semantic-ID tokenizers и generative recommenders ведут себя при изменении catalog size, sparsity, user/item count и noise. Особенно это релевантно для scaling experiments, где реальных открытых датасетов часто недостаточно.
