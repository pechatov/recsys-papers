---
title: "End-to-end Learnable Clustering for Intent Learning in Recommendation"
category: "generative_retrieval"
slug: "end_to_end_learnable_clustering_intent_learning_recommendation_summary"
catalogId: "paper-end_to_end_learnable_clustering_intent_learning_recommendation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/end_to_end_learnable_clustering_intent_learning_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2401.05975"
---
Подробное саммари статьи:

> **Авторы:** Yue Liu, Shihao Zhu, Jun Xia, Yingwei Ma, Jian Ma, Xinwang Liu, Shengju Yu, Kejun Zhang, Wenliang Zhong.
>
> **Аффилиации:** Ant Group; Alibaba Group; Westlake University; National University of Singapore; National University of Defense Technology; Zhejiang University.
>
> **Публикация:** NeurIPS 2024. arXiv:2401.05975. Код: github.com/yueliu1999/ELCRec.

## 1. Коротко

ELCRec решает задачу intent learning в sequential recommendation: нужно выделить скрытые пользовательские intents из историй поведения и использовать их для next-item prediction. Авторы утверждают, что многие intent-learning методы работают как обобщенный EM: сначала кластеризуют user behavior embeddings на всем датасете, затем обновляют representation model, и повторяют. Это дорого, плохо масштабируется и разделяет clustering и behavior learning.

Предложение ELCRec: сделать cluster centers обычными learnable parameters нейросети и обучать их end-to-end на mini-batches. Тогда intent learning становится частью общей loss, а не внешним offline clustering step. Для GR/SID это близко к проблеме tokenization/indexing: вместо фиксированного k-means/RQ-VAE индекса модель учит дискретно-подобную структуру вместе с поведением.

## 2. Контекст

В рекомендациях “intent” обычно означает не явную категорию, а латентную причину поведения: например, пользователь взаимодействует с обувью, сумками и ракетками, а система выводит intent “badminton”. Методы вроде MIND, ComiRec, ICLRec пытаются разложить complex user interests на несколько latent units.

Проблема EM-подхода особенно заметна в индустриальных сценариях: clustering по всем пользователям требует хранить и обрабатывать полный embedding matrix. На приложениях с миллионами или миллиардами пользователей это либо медленно, либо приводит к out-of-memory. Кроме того, offline cluster assignment отстает от меняющихся behavior patterns.

## 3. Метод

ELCRec состоит из трех частей:

- **Behavior encoding.** История пользователя кодируется Transformer encoder'ом. Последовательность длины до T агрегируется через concatenate pooling в behavior embedding.
- **End-to-end Learnable Cluster Module (ELCM).** Cluster centers C размерности k x d являются обучаемыми параметрами. Loss одновременно раздвигает разные centers и притягивает behavior embeddings к centers на unit sphere.
- **Intent-assisted Contrastive Learning (ICL).** Используются две augmented views истории пользователя: mask, crop, reorder. Sequence contrastive loss учит behavior representations, а intent contrastive loss использует ближайший cluster center как self-supervision.

Общая objective: next-item prediction loss + 0.1 * intent-assisted contrastive loss + alpha * clustering loss. В отличие от offline k-means, вычисление cluster loss зависит от batch size b и числа clusters k, а не от полного числа пользователей |U|.

## 4. Эксперименты/результаты

Основные offline benchmarks: Sports, Beauty, Toys, Yelp. Метрики: HR@5/20 и NDCG@5/20. Baselines включают BPR-MF, GRU4Rec, Caser, SASRec, BERT4Rec, S3-Rec, CL4SRec, DCRec, MAERec, IOCRec, ICLRec. ELCRec обычно лучший или near-best; исключение: Toys HR@5, где ICLRec 0.0586 слегка выше ELCRec 0.0585.

<div class="table-scroll">
<table>
<thead><tr><th>Срез</th><th>Результат</th></tr></thead>
<tbody>
<tr><td>Beauty</td><td>NDCG@5 0.0355 против 0.0326 у runner-up; +8.90%.</td></tr>
<tr><td>Sports</td><td>HR@5 0.0286 против 0.0263; +8.75%.</td></tr>
<tr><td>Efficiency</td><td>В среднем time -7.18% и GPU memory -9.58% относительно ICLRec; на Beauty time -22.49%.</td></tr>
<tr><td>Industrial A/B</td><td>Alipay live-streaming recommendation: PVCTR/VV/PVCTR merchandise растут примерно на 2.3-2.5%; UVCTR +1.62%.</td></tr>
</tbody>
</table>
</div>

Аblation показывает, что и ELCM, и ICL полезны отдельно. Комбинация лучшая на трех из четырех датасетов; на Toys авторы прямо отмечают возможный конфликт ICL с уже сильными intent representations от ELCM.

## 5. Ограничения

- **Число clusters задается вручную.** Авторы называют predefined cluster number одним из будущих направлений; k=256/512 работает в их setup, но не является универсальным.
- **Домены ограничены.** Основные benchmarks - buying/business recommendation; дополнительные ML-1M и MIND-small есть в appendix, но это все еще ограниченный набор.
- **Training dynamics могут быть нестабильны.** На Sports время не уменьшается: авторы связывают это с неудачным направлением оптимизации cluster embeddings.
- **Это не generative retrieval модель.** ELCRec учит intents для sequential recommendation, но не генерирует SID или docid напрямую.

## 6. Как читать для GR/SID

ELCRec стоит читать как paper про **learnable clustering вместо frozen index construction**. В TIGER-like системах semantic IDs часто строятся отдельно через RQ-VAE/k-means и затем замораживаются. ELCRec показывает другой принцип: центры можно сделать параметрами и обучать их вместе с recommendation objective.

Для SID-tokenization это поднимает прямой вопрос: должны ли item/user clusters быть offline artifacts или частью end-to-end loss? Ответ ELCRec не переносится механически на item SID, но дает полезные идеи: mini-batch cluster loss, push-pull centers, contrastive supervision и проверка memory/time на больших данных.
