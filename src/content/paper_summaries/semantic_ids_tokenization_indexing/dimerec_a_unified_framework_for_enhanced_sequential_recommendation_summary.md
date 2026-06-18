---
title: "DimeRec: A Unified Framework for Enhanced Sequential Recommendation via Generative Diffusion Models"
category: "semantic_ids_tokenization_indexing"
slug: "dimerec_a_unified_framework_for_enhanced_sequential_recommendation_summary"
catalogId: "paper-dimerec_a_unified_framework_for_enhanced_sequential_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/dimerec_a_unified_framework_for_enhanced_sequential_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2408.12153"
---
> **Авторы:** авторская группа Kuaishou/Alibaba/USTC/Sun Yat-sen University из arXiv source.
>
> **Аффилиации:** Kuaishou; Alibaba; University of Science and Technology of China; Sun Yat-sen University.

## Коротко

DimeRec - diffusion-based sequential recommender для matching/retrieval stage.

В отличие от SID papers, он не строит semantic IDs как главный объект, но находится рядом с ними: это generative recommendation, где модель восстанавливает next-interest embedding через diffusion process.

Главные компоненты: Guidance Extraction Module, Diffusion Aggregation Module и Geodesic Random Walk noise на сфере.

## Контекст

Sequential recommendation часто обучается предсказывать один next item.

Но пользователь может иметь несколько потенциальных следующих интересов.

Diffusion model дает способ моделировать distribution next interests, а не точечный embedding.

Для matching stage важнее широкий candidate recall, чем идеальный top-1 ranking.

## Проблема

Diffusion recommenders могут страдать от optimization inconsistency: objectives guidance, denoising и reconstruction тянут model в разные стороны.

Кроме того, raw sequence encoder может не выделять multi-interest structure.

DimeRec пытается объединить multi-interest guidance и diffusion denoising в одной multi-task схеме.

## Метод / архитектура

Guidance Extraction Module извлекает multi-interest guidance из behavior sequence.

В основной версии используется ComiRec-SA как self-attention multi-interest backbone.

Diffusion Aggregation Module добавляет noise к target item embedding и восстанавливает original embedding под guidance.

Noise задается Geodesic Random Walk на сфере, чтобы сохранять geometry normalized embeddings.

## Objective / алгоритм

Training включает несколько losses.

`L_gem` обучает guidance extraction.

`L_ssm` связан с score/semantic similarity modeling diffusion process.

`L_recon` восстанавливает target item embedding.

GRW помогает согласовать noise process с spherical embedding space.

Serving выполняет reverse denoising: от noisy representation под guidance модель приходит к next-interest embedding, затем retrieve candidates.

## Детальный алгоритм DimeRec

DimeRec генерирует не item ID и не semantic ID, а continuous next-interest embedding. Алгоритм состоит из conditional diffusion: multi-interest guidance задает направление, а reverse denoising восстанавливает target item embedding на сфере.

1. **Нормализовать item embeddings.** Item vectors лежат на сфере, поэтому noise process должен сохранять spherical geometry.
1. **Извлечь multi-interest guidance.** Guidance Extraction Module, в основной версии ComiRec-SA, превращает behavior sequence в несколько interest vectors, а не один усредненный user state.
1. **Выбрать target embedding для обучения.** Для user sequence target item embedding становится clean sample, который diffusion module должен восстановить.
1. **Добавить Geodesic Random Walk noise.** Forward process двигает clean target по геодезическим шагам на сфере, а не обычным Gaussian noise в Euclidean space.
1. **Обучить denoising под guidance.** Diffusion Aggregation Module получает noisy target, timestep и guidance, затем предсказывает clean/score direction.
1. **Сбалансировать losses.** `L_gem` обучает guidance, `L_ssm` стабилизирует diffusion semantic/score modeling, `L_recon` напрямую восстанавливает target embedding.
1. **На serving выполнить reverse diffusion.** Из noisy/init representation под user guidance получается next-interest embedding, затем ANN index возвращает candidate items.

```
for user_sequence, target_item in batch:
    guidance = GEM(user_sequence)  # multi-interest vectors
    x0 = normalize(item_embedding[target_item])
    t = sample_timestep()
    xt = geodesic_random_walk_noise(x0, t)

    pred = DAM(xt, t, guidance)
    loss = L_gem(guidance, target_item) + mu * L_ssm(pred, x0) + lambda * L_recon(pred, x0)
    update(loss)

at serving:
    guidance = GEM(user_history)
    generated_embedding = reverse_denoise(guidance, steps=S)
    candidates = ANN_search(normalize(generated_embedding))
```

## Эксперименты и метрики

Датасеты: YooChoose, KuaiRec и ML-10M.

Метрики: HR@10/20/50 и NDCG@10/20/50.

Эксперименты проводятся пять раз, таблицы показывают mean/std, значимость через t-test `p<.05`.

DimeRec превосходит baselines по HR@20, HR@50, NDCG@20 и NDCG@50 на всех трех датасетах.

Для HR@50 gains относительно second-best: 18.29% на YooChoose, 22.14% на KuaiRec и 39.13% на ML-10M.

Для NDCG@50 gains: 20.75%, 9.81% и 32.90% соответственно.

## Рисунки / таблицы

Figure architecture показывает GEM слева и DAM справа.

Optimization inconsistency figure объясняет, зачем нужны специальные balancing choices.

Table baseline comparison содержит HR/NDCG на трех datasets.

Guidance module ablation сравнивает SASRec, MLP, ComiRec-DR и ComiRec-SA.

Дополнительные графики показывают влияние GRW на три losses и sensitivity по diffusion steps, `lambda` и `mu`.

## Сильные стороны

- **Четкая ориентация на matching stage.** DimeRec оптимизирует широкий candidate recall, поэтому HR@50 gains интерпретируются честнее, чем попытка назвать модель top-ranker.
- **Multi-interest guidance решает проблему одного user vector.** ComiRec-SA дает diffusion module несколько возможных future interests.
- **GRW согласован с normalized embeddings.** Noise process учитывает spherical geometry, а не ломает ее Euclidean perturbations.
- **Ablations показывают вклад блоков.** Сравнение SASRec/MLP/ComiRec-DR/ComiRec-SA и удаление losses помогают понять, что не является декоративным.
- **Метод не требует SID vocabulary.** Это полезный контраст к generative retrieval через discrete tokens: можно оставить ANN serving.

## Ограничения

- **Не решает discrete vocabulary.** Каталог все равно обслуживается через ANN по generated embedding, а не через валидные item tokens.
- **Reverse diffusion дороже обычного encoder retrieval.** Max diffusion steps напрямую бьют по latency, поэтому нужен quality-latency tuning.
- **Top ranking остается слабее matching.** HR@50 может расти сильнее, чем NDCG@10; downstream ranker все равно нужен.
- **Guidance quality является bottleneck.** Если GEM не выделяет реальные multi-interest modes, DAM будет восстанавливать усредненный или шумный intent.
- **ANN index должен быть строго согласован.** Generated embeddings и item index должны иметь одну normalization/embedding версию.

## Как реализовать / проверять

1. Подготовить normalized item embeddings.
1. Обучить GEM: SASRec/ComiRec baseline и выбранный multi-interest extractor.
1. Реализовать GRW noise schedule на сфере.
1. Обучить DAM с multi-task losses.
1. На serving запускать reverse denoising и ANN search.
1. Сравнивать HR/NDCG на разных K, отдельно K=50 для matching.

## Связь с соседними работами

DimeRec отличается от SID retrieval: он генерирует continuous interests, а не discrete semantic IDs.

Но его можно рассматривать как альтернативный generative retrieval backbone.

В production SID-подход может заменить ANN lookup на token generation, а DimeRec оставляет ANN, но улучшает distribution modeling.

## Ablation conclusions

Замена Guidance Extraction Module на SASRec или простой MLP ухудшает Recall@20.

Dynamic-routing ComiRec-DR сильнее простых вариантов, но ComiRec-SA оказывается лучшим в основной конфигурации.

Это подтверждает, что multi-interest extraction является не декоративным блоком, а источником качества diffusion module.

Отдельная ablation на ML-10M показывает, что удаление каждого компонента снижает Recall@20/NDCG@20.

Full DimeRec в этой таблице дает Recall@20 0.0823 и NDCG@20 0.0516.

Variants без отдельных блоков падают до 0.0598-0.0776 Recall@20.

## Hyperparameter sensitivity

Appendix анализирует коэффициент `lambda` при `L_recon`.

Также анализируется коэффициент `mu` при `L_ssm`.

Отдельно проверяется Max Diffusion Steps.

Слишком малое число steps может недовосстанавливать target interest.

Слишком большое число steps увеличивает serving cost и может давать diminishing returns.

Для production matching нужно выбрать точку на quality-latency curve, а не максимум offline HR.

## Production notes

Online A/B table показывает запуск experimental group with DimeRec против base group without DimeRec.

Это важно, потому что diffusion retrieval может выглядеть сильным offline, но проиграть из-за latency или candidate distribution shift.

В serving strategy нужно заранее подготовить item ANN index.

Generated embeddings должны быть normalized так же, как embeddings index.

Нужен fallback на классический retrieval, если reverse diffusion не уложился в latency budget.

Также нужен мониторинг diversity: diffusion может расширять candidate set, но может и генерировать слишком похожие embeddings.

## Итог

DimeRec полезен как контраст к semantic ID работам: generative recommendation не обязана быть дискретной.

Если задача - candidate recall на matching stage, diffusion over embeddings может дать сильные HR@50 gains, но потребует отдельного решения latency и ANN serving.

## Источники

- [arXiv:2408.12153](https://arxiv.org/abs/2408.12153) , PDF/source.
