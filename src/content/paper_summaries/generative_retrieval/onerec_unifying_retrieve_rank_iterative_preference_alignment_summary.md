---
title: "OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment"
category: "generative_retrieval"
slug: "onerec_unifying_retrieve_rank_iterative_preference_alignment_summary"
catalogId: "paper-onerec_unifying_retrieve_rank_iterative_preference_alignment_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/onerec_unifying_retrieve_rank_iterative_preference_alignment_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2502.18965"
---
> **Авторы:**
>
> Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, Guorui Zhou.
>
>   
>
>
> **Аффилиации:**
>
> Kuaishou; в пользовательской карточке также указан CAS, но публичный arXiv author block перечисляет Kuaishou Inc.
>
>   
>
>
> **Источник:**
>
> arXiv:2502.18965, 2025.

## 1. Коротко

OneRec пытается заменить каскад retrieve-then-rank единой generative recommender model. Вместо того чтобы сначала выбрать кандидатов, а потом ранжировать их отдельной моделью, OneRec генерирует целую session/list short videos и затем улучшает генерацию через Iterative Preference Alignment.

Ключевой сдвиг: модель предсказывает не один следующий item, а набор видео для следующего запроса страницы. Это ближе к реальному recommender serving, где система возвращает 5-10 роликов за один request.

## 2. Контекст

В short-video системах retrieval, pre-ranking и ranking оптимизируются раздельно, что создает objective mismatch и потери качества. Generative recommendation обещает одно end-to-end распределение по items, но на практике часто остается recall-компонентом. OneRec заявляет, что доводит GR до deployed single-stage recommendation.

## 3. Метод / pipeline

- **Semantic IDs:** item embeddings строятся из multimodal representations, выровненных с user-item behavior distribution. Residual balanced k-means quantization создает semantic IDs; в экспериментах K=8192 clusters per codebook и L=3 levels.
- **Backbone:** encoder-decoder Transformer/T5-like architecture. Encoder читает положительную историю пользователя, decoder autoregressively генерирует session semantic IDs.
- **MoE decoder:** OneRec-1B использует sparse MoE с 24 experts и top-2 activation; на инференсе активны около 13% параметров.
- **Session-wise generation:** target session выбирается из качественных сессий: минимум 5 watched videos, threshold по watch duration и дополнительные сигналы like/collect/share.
- **IPA:** reward model оценивает generated session по swt, vtr, wtr, ltr. Beam search генерирует N=128 responses; лучший ответ становится winner, худший loser; следующая модель обучается DPO-style loss. Для эффективности используется 1% DPO sample ratio.

Production stack включает XLA, bfloat16, KV cache, float16 quantization и beam size 128.

## 4. Результаты и evidence

Offline quality измеряется reward-model метриками: session watch time (swt), view-through rate (vtr), follow/watch target (wtr) и like target (ltr). В тексте указано, что OneRec-1B дает +1.78% max swt и +3.36% max ltr к TIGER-1B, а OneRec-1B+IPA улучшает базовый OneRec-1B еще на +4.04% max swt и +5.43% max ltr. 1% DPO sample достигает около 95% максимального наблюдаемого качества при примерно 20% compute относительно 5% sample ratio.

<div class="table-scroll">
<table>
<thead><tr><th>Online variant</th><th>Total watch time</th><th>Average view duration</th></tr></thead>
<tbody>
<tr><td>OneRec-0.1B</td><td>+0.57%</td><td>+4.26%</td></tr>
<tr><td>OneRec-1B</td><td>+1.21%</td><td>+5.01%</td></tr>
<tr><td>OneRec-1B + IPA</td><td>+1.68%</td><td>+6.56%</td></tr>
</tbody>
</table>
</div>

Online A/B проводился на главной странице Kuaishou с 1% трафика. Абстракт формулирует главный lift как около 1.6% watch-time gain; таблица online experiments дает +1.68% для OneRec-1B+IPA.

## 5. Ограничения

- Основные данные и serving details промышленно закрыты; public reproducibility отсутствует.
- Offline оценка сильно зависит от reward model, который сам обучен на платформенных целях.
- Авторы признают, что watch time улучшен сильнее, чем interactive indicators вроде likes; дальнейшая работа нужна для multi-objective alignment.
- Session definition и quality thresholds платформенно-специфичны и могут плохо переноситься на другие домены.

## 6. Связь с GR/SID

OneRec использует semantic IDs, но главное отличие от TIGER-like retrieval - генерация списка/сессии и DPO alignment по платформенным наградам. Это пример того, как SID-based GR можно поднять из candidate generation в полноценный recommender/ranker, если добавить list objective и preference alignment.
