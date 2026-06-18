---
title: "R3-VAE: Reference Vector-Guided Rating Residual Quantization VAE for Generative Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "r3_vae_reference_vector_guided_rating_residual_quantization_vae_for_generative_recommendation_summary"
catalogId: "paper-r3_vae_reference_vector_guided_rating_residual_quantization_vae_for_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/r3_vae_reference_vector_guided_rating_residual_quantization_vae_for_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2604.11440"
---
> **Авторы:** Qiang Wan, Ze Yang, Dawei Yang, Ying Fan, Xin Yan, Siyang Liu, Yicong Liu, Chenwei Zhang, Wei Xu, Jiahao Qin, Ke Wang.
>
> **Код:** <https://github.com/wwqq/R3-VAE>.

## 1. Коротко: о чем статья

R3-VAE предлагает новый tokenizer для generative recommendation с semantic IDs. Авторы утверждают, что стандартный RQ-VAE страдает от двух проблем: unstable training через straight-through estimator и дорогая оценка качества SID, когда каждый вариант tokenizer-а нужно проверять через полное downstream GR обучение.

Метод добавляет **reference vector projection** и **rating residual quantization**. Reference vector служит semantic anchor: embedding сначала проецируется относительно обучаемого reference vector, чтобы residual quantization начинался из более устойчивой геометрии. Rating quantization заменяет жесткий STE-style выбор codeword на dot-product rating и soft weights, через которые лучше проходят градиенты.

Дополнительно R3-VAE вводит SID quality metrics: **Semantic Cohesion** и **Preference Discrimination**. Они должны дать быстрый proxy для отбора tokenizer-а без полного обучения генеративного recommender-а.

<figure class="paper-figure">
  <img src="../../assets/r3_vae/framework.png" alt="R3-VAE pipeline with reference vector projection and rating residual quantization">
  <figcaption>Рисунок 1. Pipeline R3-VAE: reference vector projection задает anchor, hierarchical rating quantization выбирает codewords через soft dot-product scores, decoder восстанавливает embedding, а SC/PD regularization связывает token quality с recommendation utility.</figcaption>
</figure>

## 2. Контекст: что не так с RQ-VAE

RQ-VAE в semantic-ID recommendation обычно делает residual quantization: каждый следующий codebook квантует residual error предыдущего уровня. Это дает compact hierarchical identifier, но training зависит от discrete assignment. Straight-through estimator пропускает градиент приближенно, поэтому codebook usage и reconstruction могут быть нестабильными.

В production есть еще одна проблема: tokenizer iteration дорогой. Чтобы понять, хороший ли SID, часто нужно построить item-to-SID map, обучить GR model и только потом измерить Recall/NDCG. R3-VAE пытается дать proxy metrics, которые коррелируют с downstream quality.

## 3. Метод

### 3.1. Reference vector projection

Input item embedding $\mathbf{x}$ сначала сравнивается с reference vector. Идея: вместо того чтобы напрямую квантизовать исходное пространство, модель задает направляющий semantic anchor и получает более равномерное angular distribution. Это должно уменьшить sensitivity к initialization и улучшить cluster separability.

### 3.2. Rating quantization

На уровне $l$ есть residual $\mathbf{e}^{(l-1)}$ и codebook $C^l$. Для каждого codeword считается normalized dot-product score. После softmax получается распределение весов:

$$
w_k^l = \frac{\exp(s_k^l)}{\sum_{k'=1}^{M}\exp(s_{k'}^l)}.
$$

Quantized representation строится как weighted sum codewords:

$$
\mathbf{e}^{(l)} = \sum_{k=1}^{M} w_k^l \mathbf{c}_k^l.
$$

Semantic ID на inference можно получить через argmax по каждому уровню, но training идет через differentiable rating weights. Это ключевое отличие от hard STE updates.

## 4. SID quality metrics

R3-VAE предлагает два proxy:

- **Semantic Cohesion (SC):** насколько item'ы внутри одного SID cluster семантически близки.
- **Preference Discrimination (PD):** насколько коды различают preference-relevant groups.

Идея хорошая практически: tokenizer нужно быстро сравнивать до полного GR retraining. Но proxy полезен только если его корреляция с downstream Recall/NDCG сохраняется на новых доменах, а не только в авторском setup.

## 5. Эксперименты

Авторы проверяют R3-VAE на шести public datasets: Beauty, Sports, Toys, Clothing, LastFM и ML1M, плюс промышленный Toutiao setup. Baselines включают VQ-VAE, RQ-VAE, RQ-KMeans, TIGER, LETTER, CoST и OneRec-style systems.

По авторскому claim, R3-VAE дает в среднем +14.5% Recall@10 и +15.5% NDCG@10 на public benchmarks. На Toutiao есть offline MRR и online A/B: рост StayTime/U и cold-start click volume, включая +15.36% для cold-start click volume в discriminative online evaluation.

Отдельно проверяется, что SC/PD коррелируют с UAUC или Recall@10. Это важный результат, потому что он делает tokenizer search дешевле.

## 6. Как пользоваться SC/PD на практике

SC и PD лучше воспринимать как early-stop / model-selection metrics, а не как замену downstream evaluation.

Рабочий цикл может быть таким: обучить несколько tokenizer variants с разными codebook sizes, depth, temperature и reference-vector settings; посчитать utilization, collision rate, SC и PD; отбросить явно слабые варианты; только top-k вариантов прогонять через полное TIGER/GR обучение. Это экономит время, но не отменяет финальный Recall/NDCG/AUC check.

Для интерпретации важно смотреть SC и PD вместе. Высокий SC при низком PD может означать, что коды собирают семантически похожие items, но плохо различают preference-relevant alternatives. Высокий PD при низком SC может означать, что tokenizer ловит поведенческий сигнал, но разрушает content coherence и ухудшает cold-start/fresh item behavior. Поэтому полезны scatter plots SC vs PD vs downstream metric, а не один scalar score.

## 7. Сильные стороны

- Хорошо формулирует pain point tokenizer iteration: полный downstream GR retraining слишком дорог.
- Rating quantization дает более гладкий gradient path, чем hard STE.
- Reference vector projection пытается исправить geometry до квантизации, а не только добавить loss после.
- Есть public code, что повышает воспроизводимость.
- Industrial evidence показывает, что метод рассматривается не только как academic tokenizer.

## 8. Ограничения и риски

Proxy metrics SC/PD надо валидировать отдельно. Если они коррелируют с Recall на одном наборе доменов, это не гарантирует перенос на другой catalog или business objective.

Soft rating training и hard argmax inference могут расходиться. Нужно измерять train-inference gap: насколько soft assignments во время обучения соответствуют реальным discrete SIDs.

Reference vector projection добавляет новый источник sensitivity. Если reference vector плохо обучается или домен мультимодальный, один anchor может быть слишком грубым.

Industrial results закрыты в деталях: traffic mix, duration, guardrails и variance не полностью воспроизводимы.

Для production также нужен monitoring hard-code distribution после argmax inference. Soft rating weights могут выглядеть хорошо в обучении, но если hard argmax концентрируется в небольшом числе codewords, downstream generator получит тот же imbalance, от которого R3-VAE пытается уйти.

## 9. Вывод

R3-VAE полезна как работа про **стабилизацию tokenizer-а и быстрые proxy-метрики SID quality**. Ее главный практический takeaway: при разработке semantic IDs нужно логировать не только downstream Recall/NDCG, но и cohesion/discrimination/utilization metrics, чтобы быстрее отбрасывать слабые tokenizer variants до дорогого GR обучения.
