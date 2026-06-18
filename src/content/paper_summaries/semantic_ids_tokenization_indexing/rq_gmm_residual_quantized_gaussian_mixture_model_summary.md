---
title: "RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR Prediction"
category: "semantic_ids_tokenization_indexing"
slug: "rq_gmm_residual_quantized_gaussian_mixture_model_summary"
catalogId: "paper-rq_gmm_residual_quantized_gaussian_mixture_model_summary"
paperUrl: "https://arxiv.org/abs/2602.12593"
---
> **Авторы:** Ziye Tong, Jiahao Liu, Weimin Zhang, Hongji Ruan, Derick Tang, Zhanpeng Zeng, Qinsong Zeng, Peng Zhang, Tun Lu, Ning Gu.
>
> **Аффилиации:** Tencent; Fudan University; Beijing Jiaotong University.

## 1. Коротко: о чем статья

RQ-GMM переносит semantic discretization в CTR prediction. В отличие от TIGER-style generative retrieval, здесь semantic IDs используются не как generated target sequence, а как дискретные признаки для CTR model. Мотивация: multimodal embeddings из pretrained encoders полезны, но если просто добавить continuous embeddings в CTR модель, возникает mismatch objectives и разная скорость сходимости между CTR backbone и модальным encoder'ом.

Авторы предлагают дискретизировать multimodal embeddings в semantic IDs и подавать эти IDs в CTR model. Но стандартные VQ/RQ методы страдают ограниченным codebook utilization, reconstruction error и слабой semantic discriminability. RQ-GMM заменяет hard clustering на Gaussian Mixture Model внутри residual quantization: вместо "назначить ближайший centroid" модель оценивает probabilistic assignment к Gaussian components.

Главный результат: RQ-GMM улучшает CTR prediction на Amazon Review datasets и дает +1.502% Advertiser Value в online A/B на крупной short-video платформе, где метод полностью развернут.

## 2. Контекст: semantic IDs не только для generative retrieval

Большая часть semantic ID литературы обсуждает autoregressive recommendation: модель генерирует tokens следующего item'а. RQ-GMM показывает другой сценарий: semantic IDs как compact categorical features для discriminative CTR prediction.

Это важный use case. В промышленном CTR stack categorical IDs хорошо совместимы с embedding tables, feature crosses и production serving. Если multimodal information можно превратить в стабильные semantic IDs, то CTR model получает content signal без необходимости совместно обучать тяжелый image/text encoder.

## 3. Preliminaries: VQ, RQ-VAE и RQ-KMeans

VQ-VAE назначает embedding ближайшему code vector. RQ-VAE делает это многоуровнево: первый codebook приближает исходный vector, следующий квантует residual, и так далее. RQ-KMeans похож по духу, но использует residual KMeans без neural VAE.

Ограничение hard assignment: пространства multimodal embeddings часто имеют сложную форму. Один centroid с Euclidean nearest-neighbor assignment может плохо описывать density, а редкие области могут недоиспользовать codebook.

RQ-GMM заменяет centroid-only view на mixture distribution. Каждый cluster описывается Gaussian component, а assignment учитывает вероятность принадлежности с учетом mean, covariance и mixture weight.

## 4. Метод RQ-GMM

На каждом residual quantization level модель обучает Gaussian Mixture Model на текущих residual vectors. Для embedding $\mathbf{x}$ GMM оценивает posterior probability принадлежности к компонентам. Semantic token выбирается из наиболее вероятной компоненты, а reconstruction subtracts component mean from residual.

Интуитивно это дает три преимущества:

- **better density modeling** - Gaussian components лучше описывают distribution, чем голые centroids;
- **soft assignment during learning** - EM-style updates используют responsibilities, а не только жесткую принадлежность;
- **higher codebook utilization** - компоненты меньше схлопываются, потому что probabilistic learning лучше распределяет массу.

<figure class="paper-figure">
  <img src="../../assets/rq_gmm/convergence.png" alt="RQ-GMM and RQ-KMeans RMSE convergence curves">
  <figcaption>Рисунок 1. RQ-GMM быстрее и ниже сходится по RMSE на промышленном датасете, чем RQ-KMeans. Это поддерживает claim, что probabilistic residual quantization лучше подстраивается под multimodal embedding distribution.</figcaption>
</figure>

## 5. Как semantic IDs используются в CTR

После RQ-GMM item получает sequence of discrete semantic IDs. Эти IDs подаются в CTR backbone как categorical features через embedding lookup. Авторы проверяют два режима:

- **w/o Emb** - CTR model получает semantic IDs без исходного continuous modal embedding;
- **w/ Emb** - semantic IDs добавляются вместе с continuous modal embedding.

Это хорошая проверка: если RQ-GMM полезен только как compression, он должен помогать w/o Emb; если он дополняет continuous features, gain должен сохраняться w/ Emb.

## 6. Offline результаты

Эксперименты идут на Amazon Review Appliances, Beauty и Automotive, с CTR backbones FNN и IPNN. RQ-GMM стабильно дает лучший AUC и LogLoss среди None, VQ-VAE, RQ-VAE и RQ-KMeans.

<div class="table-scroll">
<table>
<thead>
<tr><th>Backbone</th><th>Mode</th><th>Method</th><th>Appliances AUC</th><th>Beauty AUC</th><th>Automotive AUC</th></tr>
</thead>
<tbody>
<tr><td>FNN</td><td>w/o Emb</td><td>RQ-KMeans</td><td>0.665</td><td>0.613</td><td>0.632</td></tr>
<tr><td>FNN</td><td>w/o Emb</td><td>RQ-GMM</td><td>0.674</td><td>0.619</td><td>0.638</td></tr>
<tr><td>FNN</td><td>w/ Emb</td><td>RQ-KMeans</td><td>0.667</td><td>0.621</td><td>0.635</td></tr>
<tr><td>FNN</td><td>w/ Emb</td><td>RQ-GMM</td><td>0.678</td><td>0.628</td><td>0.644</td></tr>
<tr><td>IPNN</td><td>w/o Emb</td><td>RQ-KMeans</td><td>0.675</td><td>0.620</td><td>0.642</td></tr>
<tr><td>IPNN</td><td>w/o Emb</td><td>RQ-GMM</td><td>0.680</td><td>0.630</td><td>0.646</td></tr>
<tr><td>IPNN</td><td>w/ Emb</td><td>RQ-KMeans</td><td>0.680</td><td>0.628</td><td>0.647</td></tr>
<tr><td>IPNN</td><td>w/ Emb</td><td>RQ-GMM</td><td>0.688</td><td>0.634</td><td>0.654</td></tr>
</tbody>
</table>
</div>

Discretization quality также улучшается. На Appliances RMSE падает с 0.121 у RQ-KMeans до 0.117 у RQ-GMM, utilization растет с 86.7/87.1 до 89.5/89.3. На Beauty RMSE падает 0.132 -> 0.127, utilization растет 77.3/78.9 -> 80.3/81.2.

## 7. Online результат

Самый сильный claim статьи - deployment на large-scale short-video platform. Авторы сообщают +1.502% Advertiser Value относительно сильных baselines. Для CTR/recommendation advertising такой uplift обычно значим, особенно если система обслуживает сотни миллионов пользователей.

Ограничение: детали production setup закрыты. Неясно, какие exact guardrails, latency budget, traffic split, duration и statistical protocol использовались. Поэтому online result важен как evidence of deployability, но не как полностью воспроизводимое сравнение.

## 8. Сильные стороны

RQ-GMM хорошо поднимает вопрос, который часто теряется в generative recommendation: semantic IDs полезны не только как targets for generation, но и как categorical features для rankers/CTR models.

Метод технически прост для production: после offline discretization downstream CTR model видит обычные token IDs. Это гораздо легче встроить в существующий feature pipeline, чем заменить retrieval на autoregressive generation.

Probabilistic quantization - осмысленная замена hard KMeans. Улучшения видны не только в CTR metrics, но и в reconstruction/utilization diagnostics.

## 9. Ограничения и открытые вопросы

RQ-GMM не решает весь lifecycle semantic IDs. В статье меньше внимания уделено incremental updates, drift, item churn и совместимости codebooks между retraining runs.

Метод проверен для CTR prediction, а не для autoregressive generative retrieval. Нельзя автоматически переносить выводы на TIGER-style next-item generation.

GMM complexity выше KMeans: нужно хранить и обновлять covariance/mixture parameters. Для очень больших catalogs важно измерять training cost, memory и stability EM updates.

## 10. Вывод

RQ-GMM - полезная работа для тех, кто хочет начать с более безопасного внедрения semantic IDs: не заменять весь recommender на generator, а добавить discrete multimodal semantic features в CTR/ranking model. Главный технический takeaway: residual quantization выигрывает от probabilistic density modeling, особенно когда multimodal embedding space плохо описывается hard centroids.
