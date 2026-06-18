---
title: "GS 2 -RS: A Generative Approach for Alleviating Cold Start and Filter Bubbles in Recommender Systems"
category: "generative_retrieval"
slug: "gs2rs_generative_cold_start_filter_bubbles_summary"
catalogId: "paper-gs2rs_generative_cold_start_filter_bubbles_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/gs2rs_generative_cold_start_filter_bubbles_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1109/TKDE.2023.3290140"
---
Подробное саммари статьи:

> **Авторы:** Yuanbo Xu, En Wang, Yongjian Yang, Hui Xiong.
>
> **Аффилиации:** Mobile Intelligent Computing Lab, Key Laboratory of Symbolic Computation and Knowledge Engineering, Department of Computer Science and Technology, Jilin University; Thrust of Artificial Intelligence, HKUST (Guangzhou); Department of Computer Science and Engineering, HKUST.
>
> **Публикация:** IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 2, pp. 668-681. DOI опубликован в 2023 году, журнальный выпуск вышел в феврале 2024.
>
> **Источники:** [DOI / IEEE](https://doi.org/10.1109/TKDE.2023.3290140), [страница Rutgers](https://www.researchwithrutgers.com/en/publications/gssup2sup-rs-a-generative-approach-for-alleviating-cold-start-and/), авторская PDF-копия из IEEE Xplore.

## 1. Коротко

GS2-RS решает две связанные проблемы классических recommender systems: **cold start** и **filter bubble**. Авторы утверждают, что эти проблемы нельзя лечить независимо: рекомендация популярных item'ов cold-start пользователям усиливает популярность уже видимых item'ов, а пузырь похожих рекомендаций, в свою очередь, мешает новым пользователям и редким item'ам получить шанс на экспозицию.

Метод не требует side information: входом является только user-item rating matrix. Вместо генерации самих рейтингов GS2-RS генерирует более интерпретируемые **fine-grained preferences**: interest и satisfaction. Затем из них строит serendipity-сигнал: item считается неожиданно полезным, если у пользователя низкий interest, но высокая predicted satisfaction. На выходе получается enhanced rating matrix, которую можно подать в обычный recommender как preprocessing.

## 2. Контекст

Cold start в статье трактуется как высокая разреженность взаимодействий: у пользователя или item'а мало рейтингов, поэтому CF/MF-модели вынуждены опираться на слабый сигнал. Filter bubble возникает, когда модель всё сильнее замыкается на исторически похожих и популярных item'ах. В отличие от подходов, использующих профиль пользователя, social links, item descriptions или reviews, GS2-RS специально ограничивается одним источником данных: матрицей рейтингов.

Важный сдвиг относительно GAN-based data completion: авторы не хотят просто заполнить матрицу вероятными рейтингами. Они сначала раскладывают поведение пользователя на interest и satisfaction, потому что высокий интерес не равен высокой удовлетворённости, а serendipity возникает именно на их несоответствии.

## 3. Метод и pipeline

1. **Preference extraction.** Из исходной матрицы рейтингов строятся две матрицы: interest matrix и satisfaction matrix. Они задают два разных взгляда на пользовательскую оценку item'а.
1. **Twin-CGAN.** Две conditional GAN-модели обучаются отдельно: I-CGAN для interest и S-CGAN для satisfaction. Для каждого пользователя генераторы производят несколько synthetic preference vectors, которые авторы называют self-neighbors.
1. **Preference filtering.** Сгенерированные self-neighbors сравниваются с исходным пользователем; затем используется агрегирование top-3 похожих виртуальных предпочтений.
1. **Gated serendipity fusion.** По двум порогам, обычно `theta_in = theta_sa = 0.5` , item помечается как serendipity, если predicted satisfaction высокая, а predicted interest низкий.
1. **Negative matrix injection.** Вместо рискованного заполнения матрицы положительными рейтингами метод инжектит нули для явно невозможных item'ов: низкий interest и низкая satisfaction либо один неизвестный сигнал и второй низкий. Это снижает неопределённость и разреженность.
1. **Reuse as preprocessing.** Enhanced matrix используется как вход для CF, WMF, NCF, JoVA, CSII и других базовых моделей без изменения их архитектуры.

## 4. Результаты и evidence

Эксперименты поставлены на Movielens, Amazon, Yelp и Book-Crossing; авторы подчёркивают, что все четыре набора имеют разреженность выше 95%. Базовые методы включают CF, WMF, NCF, JoVA, AR-CF, CSII, GAZRec и VELF. Для общей точности используются Precision, Recall, NDCG и MRR; для cold start - Exposure Ratio; для filter bubble - diversity и serendipity.

По overall recommendation GS2-RS оказывается сильнее baseline'ов; в тексте отдельно отмечено, что на трёх датасетах, кроме Movielens, он превосходит JoVA примерно на 12% по Rec@10. Как preprocessing, GS2-RS повышает Precision@10 и NDCG@10 у нескольких существующих моделей, особенно у CF и NCF, потому что даёт им менее разреженный и более информативный input.

По cold-start экспериментам GS2-RS повышает exposure ratio редких item'ов, при этом не проваливая accuracy. По filter-bubble экспериментам вариант ранжирования по serendipity даёт максимальные diversity/serendipity, особенно на sparse datasets. Ablation показывает, что удаление preference extraction, satisfaction branch, indicator vector или negative matrix injection ухудшает качество; лучшая практическая настройка для числа виртуальных соседей - `N = 2`.

## 5. Ограничения

- Метод работает с explicit ratings и всё ещё зависит от качества исходной rating matrix; при полностью новых users/items без каких-либо следов он не создаёт контентное понимание item'а.
- Пороговая логика serendipity интерпретируема, но груба: разные домены могут требовать других порогов и калибровки interest/satisfaction.
- Negative injection добавляет сильное предположение: помеченные нулём item'ы действительно невозможны для пользователя. Ошибочные нули могут ухудшить recall.
- Это не generative retrieval в смысле генерации item identifier токен за токеном; генеративность находится в data/preference completion.

## 6. Связь с GR/SID

Для карты generative recommendation эта статья важна как ранний пример **generative preprocessing**: генератор не декодирует semantic ID, но создаёт латентные предпочтения, которые меняют candidate space. В отличие от TIGER/semantic-ID подходов, здесь нет дискретной токенизации item'ов, constrained decoding или autoregressive retrieval. Связь с GR скорее концептуальная: генеративная модель используется для расширения наблюдаемого пространства и борьбы с bias/exposure проблемами, которые потом всплывают и в generative retrieval с SID.
