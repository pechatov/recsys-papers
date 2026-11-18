---
title: "Mitigating Popularity Bias in Sequential Recommendations with Sparse Prior-Regularized Objective"
category: "fairness_privacy_robustness_debiasing_evaluation"
slug: "pr_entmax_sparse_prior_regularized_objective_summary"
catalogId: "paper-pr_entmax_sparse_prior_regularized_objective_summary"
paperUrl: "https://arxiv.org/abs/2610.17823"
---
> **Авторы:** Anton Pembek, Anton Klenitskiy, Alexander Savchenko, Alexey Vasilev.
>
> **Аффилиации:** Sber AI Lab; Lomonosov Moscow State University; HSE University.
>
> **Публикация:** CIKM 2026, pp. 3952–3963, DOI 10.1145/3799682.3841006; опубликована 2026-11-06. Прочитан arXiv:2610.17823v1 от 2026-10-13, включая приложение.
>
> **Код:** ветка `cikm_2026`, ревизия `0811dbab3a1f3733bfcae06de65429896f021470`; статическая проверка 2026-11-18. Обучение моделей не воспроизводилось.

## 1. Коротко: о чём статья

PR-entmax объединяет разреженное вероятностное преобразование entmax и поправку логитов на популярность. При обучении используется $z_i+\gamma\log q_i$, при ранжировании — исходный $z_i$. Entmax присваивает части объектов точную нулевую вероятность. Поэтому prior меняет не только веса конкурентов, но и состав активного множества.

Метод проверяется на SASRec и GRU4Rec, пяти датасетах и нескольких способах коррекции popularity bias. Центральный экспериментальный вопрос — позволяет ли prior внутри sparse objective улучшать компромисс между релевантностью и распределением рекомендаций по каталогу. Основные таблицы поддерживают полезность подхода, но содержат случаи падения NDCG, а проценты абляции местами не согласуются с абсолютными результатами. [Полный текст](https://arxiv.org/html/2610.17823v1).

## 2. Контекст: плотная и разреженная конкуренция

В next-item prediction модель выдаёт логит каждого кандидата. Softmax распределяет положительную вероятность между всеми кандидатами, а cross-entropy даёт отрицательному объекту градиент, равный его вероятности. Очень слабые кандидаты получают маленький, но ненулевой сигнал.

Entmax заменяет это распределение разреженным. Объекты ниже порога получают ровно нулевую вероятность. Для отрицательных объектов исчезает непосредственный выходной градиент, и модель различает target с более узким набором конкурентов. Это не отменяет вычисления исходных scores и само по себе не гарантирует ускорения.

Обычный entmax выбирает конкурентов по contextual logits. PR-entmax дополнительно учитывает глобальную популярность. Популярный объект с подходящим контексту score с большей вероятностью остаётся конкурентом; слишком слабый score всё ещё может исключить его из support.

## 3. Вариационная постановка и вероятностное преобразование

Пусть $z_i(x)$ — логит объекта $i$ для последовательности $x$, а $q_i\gt0$ — фиксированный prior, оценённый по train. Авторы определяют распределение через

$$
p^*(z)=\arg\max_{p\in\Delta}
\left\{\langle p,z\rangle+\theta H_\alpha(p)-\gamma H(p,q)\right\},
$$

где $\Delta$ — вероятностный симплекс, а энтропия Цаллиса и cross-entropy с prior равны

$$
H_\alpha(p)=\frac{1-\sum_i p_i^\alpha}{\alpha(\alpha-1)},
\qquad H(p,q)=-\sum_i p_i\log q_i.
$$

Параметр $\alpha\gt1$ управляет разреженностью, $\theta\gt0$ — масштабом энтропийного члена, $\gamma$ — силой prior. Подстановка $H(p,q)$ показывает, что prior входит как линейная поправка:

$$
\langle p,z\rangle-\gamma H(p,q)
=\sum_i p_i(z_i+\gamma\log q_i).
$$

Отсюда вычислительная форма метода:

$$
p^*(z)=\operatorname{entmax}_\alpha
\left(\frac{z+\gamma\log q}{\theta}\right).
$$

Это entmax над скорректированными логитами. Вклад работы заключается в использовании prior для выбора активных конкурентов и экспериментальном изучении этой конструкции в recommendation; новой архитектуры encoder здесь нет.

Приложение выводит решение через условия Каруша—Куна—Таккера:

$$
p_i^*=
\left[(\alpha-1)\left(\frac{z_i+\gamma\log q_i}{\theta}-\tau\right)\right]_+^{1/(\alpha-1)},
\qquad \sum_i p_i^*=1.
$$

Порог $\tau$ зависит от всего вектора скорректированных логитов. Активное множество имеет вид

$$
\operatorname{supp}(p^*)=
\{i:z_i+\gamma\log q_i\gt\theta\tau\}.
$$

Prior глобален и фиксирован. Контекстная зависимость коррекции возникает через $z_i(x)$ и порог $\tau(x)$, а не через отдельную модель персонализированной популярности.

<figure class="paper-figure">
  <img loading="lazy" decoding="async" src="../../assets/pr_entmax/training_pipeline.png" alt="PR-entmax: contextual logits плюс popularity prior, разреженное распределение и градиент p минус y">
  <figcaption>Рисунок 1 статьи. Prior добавляется до entmax и влияет на активное множество. Для target градиент содержит отдельный член −1, поэтому нулевая вероятность target не означает отсутствие обучения. Источник: Pembek et al., <a href="https://arxiv.org/pdf/2610.17823">arXiv:2610.17823v1</a>.</figcaption>
</figure>

## 4. Loss, градиенты и inference

Соответствующая Fenchel–Young loss записывается как

$$
L(z,y)=\langle p^*-y,z\rangle
+\theta H_\alpha(p^*)-\gamma H(p^*,q).
$$

В этой записи, как и в статье, опущена константа $-\gamma\log q_y$, не зависящая от параметров модели. Градиент по исходным логитам:

$$
\nabla_z L=p^*-y.
$$

Для negative вне support он равен нулю. Для target вне support градиент равен $-1$, и обучение продолжается. Это уточняет встречающуюся в статье формулировку о том, что только объекты с ненулевой вероятностью участвуют в обновлениях. Кроме того, shared embeddings могут получать сигнал через входную последовательность, даже если непосредственный выходной градиент конкретного negative равен нулю.

Нельзя заменить softmax на entmax внутри обычного $-\log p_y$ и считать это тем же методом: при нулевой вероятности target такая формула расходится. Нужна согласованная Fenchel–Young loss.

При $\gamma\gt0$ популярный объект получает менее отрицательную поправку, чем редкий. Поэтому popular negatives сильнее конкурируют с target, а редкому positive требуется более высокий contextual score для компенсации prior. На inference используют

$$
s_i^{\mathrm{inference}}(x)=z_i(x).
$$

«Убрать prior» означает не добавлять его к выходу модели. Повторное вычитание $\gamma\log q_i$ из уже исходного $z_i$ внесло бы дополнительную, не предусмотренную этим правилом коррекцию. Интерпретировать $z_i$ как причинно очищенную релевантность нельзя: objective не запрещает модели кодировать популярность другими путями.

<div class="table-scroll">

| Параметры | Частный случай |
|---|---|
| $\alpha\to1$, $\gamma=0$, $\theta=1$ | Softmax cross-entropy |
| $\alpha\to1$, $\gamma\gt0$, $\theta=1$ | Dense logit adjustment |
| $\alpha\gt1$, $\gamma=0$ | Entmax |
| $\alpha\gt1$, $\gamma\gt0$ | PR-entmax |
| $\alpha=2$ | Sparsemax-преобразование |

</div>

## 5. Данные, протокол и метрики

В таблице 2 приведены размеры после preprocessing:

<div class="table-scroll">

| Датасет | Пользователи | Объекты | Взаимодействия | Средняя длина |
|---|---:|---:|---:|---:|
| ML-1M | 6 040 | 3 373 | 999 295 | 165 |
| Tmall | 44 759 | 173 427 | 3 721 625 | 83 |
| Yambda-50M | 9 006 | 38 399 | 888 332 | 99 |
| Yoochoose | 142 954 | 16 068 | 1 212 835 | 9 |
| Zvuk | 9 272 | 109 346 | 4 695 928 | 506 |

</div>

Используются global temporal cutoff на 90% взаимодействий и 5-core filtering train. Validation строится на случайно выбранных 10% пользователей training data; для validation/test последняя interaction становится target, предыдущие образуют контекст. Для Zvuk и Tmall предварительно выбирают 10 000 и 50 000 пользователей, для Yambda-50M истории обрезают до 128 ещё до preprocessing. Поэтому эксперимент на Yambda-50M не является обучением на 50 млн событий.

Обучение: максимальная входная длина 128, batch size 64, Adam, learning rate $10^{-3}$, три seed. Для Tmall используются 50 000 негативов в loss; это число нельзя переносить на evaluation candidate set. В основных опытах $\alpha=1.5$. Для CE и Entmax отдельно выбирают backbone/config, затем для основанных на них методов настраивают специфические параметры. Это важно при оценке равенства tuning budget.

Baselines: Cross-Entropy, Inverse Propensity Weighting, Popularity Deconfounding, popularity-based negative sampling и Logit Adjustment. Entmax без prior вынесен в абляцию. Проверяются SASRec и GRU4Rec.

NDCG измеряет ranking quality, Coverage — долю каталога в рекомендациях, Novelty — нормированную self-information объекта:

$$
\operatorname{Novelty}(i)=1-\frac{\log_2 u_i}{\log_2|\mathcal U|},
$$

где $u_i$ — число train users объекта; для cold items принимается $u_i=1$. Метрики рассчитываются на глубинах 10 и 100. Novelty описывает редкость объектов, а не неожиданность или полезность рекомендации для конкретного пользователя.

## 6. Основные результаты

Ниже NDCG@10 из таблиц 3–4. Проценты пересчитаны по округлённым абсолютным значениям; это сравнение с CE, а не с лучшим конкурентом.

<div class="table-scroll">

| Датасет | SASRec: CE → PR-entmax | Изменение | GRU4Rec: CE → PR-entmax | Изменение |
|---|---:|---:|---:|---:|
| ML-1M | 0,047 → 0,058 | +23,4% | 0,054 → 0,062 | +14,8% |
| Tmall | 0,274 → 0,285 | +4,0% | 0,216 → 0,291 | +34,7% |
| Yambda-50M | 0,089 → 0,098 | +10,1% | 0,056 → 0,086 | +53,6% |
| Yoochoose | 0,203 → 0,209 | +3,0% | 0,199 → 0,197 | −1,0% |
| Zvuk | 0,218 → 0,222 | +1,8% | 0,163 → 0,224 | +37,4% |

</div>

Лучшие из остальных строк основных таблиц по NDCG@10 для SASRec: 0,049 / 0,274 / 0,092 / 0,203 / 0,218 в том же порядке датасетов; для GRU4Rec: 0,054 / 0,223 / 0,057 / 0,199 / 0,165. PR-entmax превосходит эти точки, кроме GRU4Rec на Yoochoose. Это ещё не сравнение с plain Entmax, отсутствующим в основных таблицах.

Для SASRec одновременно растут метрики распределения выдачи:

<div class="table-scroll">

| Датасет | Novelty@10: CE → PR-entmax | Coverage@10: CE → PR-entmax |
|---|---:|---:|
| ML-1M | 0,274 → 0,331 | 0,463 → 0,600 |
| Tmall | 0,657 → 0,703 | 0,351 → 0,499 |
| Yambda-50M | 0,553 → 0,671 | 0,440 → 0,785 |
| Yoochoose | 0,606 → 0,646 | 0,753 → 0,876 |
| Zvuk | 0,550 → 0,686 | 0,163 → 0,216 |

</div>

При этом PR-entmax не максимизирует каждую debiasing-метрику среди baseline: например, на SASRec/Tmall Novelty@10 у Pop Sampling равна 0,729 против 0,703 у PR-entmax. Смысл сравнения — совместный trade-off с NDCG.

На глубине 100 есть ухудшения относительно CE:

<div class="table-scroll">

| Backbone / датасет | NDCG@100: CE → PR-entmax |
|---|---:|
| SASRec / Yoochoose | 0,230 → 0,228 |
| SASRec / Zvuk | 0,256 → 0,246 |
| GRU4Rec / ML-1M | 0,105 → 0,100 |
| GRU4Rec / Yoochoose | 0,225 → 0,213 |

</div>

Результаты подтверждают полезность objective на этих задачах, но не гарантируют улучшения любой ranking-метрики. Три seed усреднены; основные таблицы не показывают разброс. Звёздочки обозначают значимость относительно CE, а не доказанное превосходство над каждым конкурентом. [Таблицы 3–4](https://arxiv.org/pdf/2610.17823).

## 7. Абляции, компромисс качества и распределение показов

Таблица 5 сравнивает PR-entmax с его двумя компонентами. Здесь сохранены именно опубликованные проценты, включая несогласованности с основной таблицей:

<div class="table-scroll">

| Датасет | Метод | Δ NDCG@10 | Δ Novelty@10 | Δ Coverage@10 |
|---|---|---:|---:|---:|
| ML-1M | PR-entmax | +18% | +17% | +23% |
| ML-1M | Logit Adjustment | −1% | +6% | +6% |
| ML-1M | Entmax | +18% | −7% | −5% |
| Tmall | PR-entmax | +4% | +7% | +30% |
| Tmall | Logit Adjustment | −5% | +8% | +18% |
| Tmall | Entmax | +4% | +4% | +25% |
| Yambda-50M | PR-entmax | +9% | +18% | +44% |
| Yambda-50M | Logit Adjustment | −1% | +12% | +20% |
| Yambda-50M | Entmax | +9% | +9% | +34% |
| Yoochoose | PR-entmax | +3% | +6% | +14% |
| Yoochoose | Logit Adjustment | −5% | +7% | +10% |
| Yoochoose | Entmax | +4% | −1% | +2% |
| Zvuk | PR-entmax | +2% | +20% | +24% |
| Zvuk | Logit Adjustment | −2% | +14% | +20% |
| Zvuk | Entmax | +6% | +9% | +15% |

</div>

Разреженность сама по себе повышает accuracy, но не обязательно novelty. Добавление prior может обменивать часть accuracy-выигрыша на изменение распределения выдачи: на Zvuk plain Entmax имеет +6% NDCG@10, PR-entmax — +2%.

Есть конкретные расхождения: ML-1M 0,047→0,058 из таблицы 3 даёт около +23,4%, тогда как таблица 5 указывает +18%. Coverage@10 на Yambda-50M 0,440→0,785 означает примерно +78,4%, а не +44%. Округление до трёх знаков их не объясняет. Разные конфигурации или версии результатов возможны, но явного пояснения не найдено. Эти две таблицы нельзя считать одной согласованной числовой серией.

<figure class="paper-figure">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:12px">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/tmall_absolute_values.png" alt="Tmall: NDCG@10 и Novelty@10 при разных objectives">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/yambda_absolute_values.png" alt="Yambda-50M: NDCG@10 и Novelty@10 при разных objectives">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/yoochoose_absolute_values.png" alt="Yoochoose: NDCG@10 и Novelty@10 при разных objectives">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/zvuk_absolute_values.png" alt="Zvuk: NDCG@10 и Novelty@10 при разных objectives">
  </div>
  <figcaption>Рисунок 2 статьи, панели Tmall, Yambda-50M, Yoochoose и Zvuk. Правее и выше — лучше. Точки показывают разные режимы настройки, а не доверительные интервалы. Источник: <a href="https://arxiv.org/html/2610.17823v1#S5.F2">Pembek et al.</a></figcaption>
</figure>

Графики показывают более выгодные достижимые сочетания NDCG и Novelty для PR-entmax, но не доминирование любой его настройки. При сильной коррекции остаётся обмен accuracy на novelty. Бюджет поиска и критерий выбора operating point поэтому важны не меньше одной лучшей строки таблицы.

<figure class="paper-figure">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:12px">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/tmall_curve.png" alt="Tmall: доля показов самых популярных объектов">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/yambda_curve.png" alt="Yambda-50M: доля показов самых популярных объектов">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/yoochoose_curve.png" alt="Yoochoose: доля показов самых популярных объектов">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/zvuk_new_curve.png" alt="Zvuk: доля показов самых популярных объектов">
  </div>
  <figcaption>Рисунок 3. По горизонтали — доля каталога от самых популярных объектов, по вертикали — накопленная доля рекомендательных показов. Более низкая кривая означает меньшую концентрацию показов в head. Источник: <a href="https://arxiv.org/html/2610.17823v1#S5.F3">Pembek et al.</a></figcaption>
</figure>

На Yambda-50M и Zvuk PR-entmax заметно снижает концентрацию показов в популярной части каталога. На других данных порядок методов меняется по участкам кривой. Диагональ равномерной экспозиции — ориентир распределения, а не доказанный оптимум пользовательской полезности.

## 8. Чувствительность и стоимость

<figure class="paper-figure">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:12px">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/tmall_alpha.png" alt="Tmall: влияние alpha на NDCG и Novelty">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/yambda_alpha.png" alt="Yambda-50M: влияние alpha на NDCG и Novelty">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/yoochoose_alpha.png" alt="Yoochoose: влияние alpha на NDCG и Novelty">
    <img loading="lazy" decoding="async" src="../../assets/pr_entmax/zvuk_alpha.png" alt="Zvuk: влияние alpha на NDCG и Novelty">
  </div>
  <figcaption>Рисунок 4. Увеличение α меняет разреженность и компромисс качества. Чрезмерная разреженность резко ухудшает NDCG на части данных, несмотря на рост Novelty. Источник: <a href="https://arxiv.org/html/2610.17823v1#S5.F4">Pembek et al.</a></figcaption>
</figure>

Авторы исследуют $\alpha$ от 1,05 до 2 и выделяют область примерно 1,2–1,6 как более устойчивую. Основное значение 1,5 находится внутри неё. Это эмпирический диапазон для выбранных конфигураций, а не универсальная настройка для любого каталога.

На Zvuk обучение с entmax примерно на 20% медленнее CE; добавление prior почти не меняет стоимость относительно самого Entmax. Нулевые вероятности не устраняют необходимость вычислить scores и support. Поэтому sparse output distribution нельзя автоматически трактовать как вычислительно разреженный retrieval.

## 9. Проверка опубликованной реализации

В `nn/modules.py` prior добавляется до деления логитов на `temp`, затем вызывается `entmax15_loss` или `entmax_bisect_loss`. В sampled-ветке prior выбирается для positive и sampled negatives. Inference использует model scores без этой loss-side поправки. [Проверенная ревизия файла](https://github.com/Antondfger/Sparse-Prior-Regularized/blob/0811dbab3a1f3733bfcae06de65429896f021470/nn/modules.py).

Найдены два существенных замечания к воспроизведению. Это статические наблюдения, не результаты повторного обучения.

Во-первых, Popularity Deconfounding обозначается как `pd` в инициализации, но полнокаталожный `compute_loss` проверяет `pda`. При выборе `pd` эта ветка не применит ожидаемое преобразование логитов. Sampled-ветка проверяет `pd` корректно. Неизвестно, совпадает ли версия всех опубликованных запусков с текущей ревизией; до уточнения сравнения с PD требуют осторожности.

Во-вторых, код вызывает loss на $(z+\gamma\log q)/\theta$ без внешнего множителя $\theta$. По chain rule градиент относительно $z$ тогда равен $(p^*-y)/\theta$, а не $p^*-y$ из нормировки формулы статьи. Для фиксированной температуры это положительное масштабирование objective, но при её подборе меняется и масштаб обучения. Это не опровергает метод; такой фактор нужно учитывать при сравнении температур и learning rate.

Формула Pop Sampling (23) в тексте также записывает знаменатель только по множеству негативов, хотя оно определено без target. В sampled-реализации positive и negatives конкатенируются перед cross-entropy, поэтому код включает target в нормировку. Для воспроизведения корректнее опираться на эту реализацию и явно зафиксировать расхождение записи.

## 10. Оценка доказательств и связь с GD-SSM

Сильная сторона — проверка objective на двух последовательных архитектурах и нескольких доменах, с временным разделением данных и абляциями двух компонентов. Формулы позволяют точно локализовать механизм: prior влияет на состав активных отрицательных конкурентов.

Ограничения связаны с интерпретацией bias и качеством экспериментальной отчётности. Coverage/Novelty не доказывают причинного устранения exposure bias, provider fairness или online-полезности. Offline targets сами происходят из логов с неравномерной экспозицией. Основные таблицы не показывают variance, а расхождения абляции и PD-ветки требуют проверки до доверия точным величинам выигрыша.

Размер каталога и пользовательские подвыборки ограничивают выводы о масштабировании. Следующие содержательные проверки — качество по popularity/user-activity группам, распределение размера support, влияние sampler, устаревание prior и online-метрики. При одинаковом бюджете настройки особенно важно сравнивать с plain Entmax: часть выигрыша accuracy получается уже от разреженного objective.

[GD-SSM](gd_ssm_norm_removed_bias_remains_summary.html) исследует зависимость обновлений от норм и частот взаимодействий. PR-entmax действует на распределение выходных градиентов. Их можно сравнивать на одном backbone, но предположение о совместной пользе проекции и двух prior пока остаётся гипотезой.

## 11. Источники

- [Полный текст arXiv:2610.17823v1](https://arxiv.org/html/2610.17823v1), [PDF с таблицами и приложением](https://arxiv.org/pdf/2610.17823v1).
- [Публикация CIKM 2026](https://doi.org/10.1145/3799682.3841006); дата сверена с [Crossref](https://api.crossref.org/works/10.1145/3799682.3841006).
- [Код: проверенная ревизия](https://github.com/Antondfger/Sparse-Prior-Regularized/tree/0811dbab3a1f3733bfcae06de65429896f021470).

Рисунки принадлежат авторам статьи. Панели 2–4 сохранены из arXiv HTML без изменения; рисунок 1 извлечён из страницы PDF. Указанная в статье лицензия — CC BY-NC-ND 4.0.
