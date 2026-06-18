---
title: "Killing Two Birds with One Stone: Unifying Retrieval and Ranking with a Single Generative Recommendation Model"
category: "generative_retrieval"
slug: "killing_two_birds_unifying_retrieval_ranking_single_gr_model_summary"
catalogId: "paper-killing_two_birds_unifying_retrieval_ranking_single_gr_model_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/killing_two_birds_unifying_retrieval_ranking_single_gr_model_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2504.16454"
---
> **Авторы:**
>
> Luankang Zhang, Kenan Song, Yi Quan Lee, Wei Guo, Hao Wang, Yawen Li, Huifeng Guo, Yong Liu, Defu Lian, Enhong Chen.
>
>   
>
>
> **Аффилиации:**
>
> University of Science and Technology of China, Huawei Noah's Ark Lab, Beijing University of Posts and Telecommunications.
>
>   
>
>
> **Источник:**
>
> arXiv:2504.16454, 2025.

## 1. Коротко

UniGRF - framework, который обучает одну generative recommendation модель одновременно на retrieval и ranking. Вместо двух отдельных сетей модель читает sequence из items и feedback, предсказывает следующий item для retrieval и score click/no-click для ranking.

Главный инженерный тезис: если backbone один и параметры разделяются, time/space complexity примерно уменьшается относительно двух раздельных моделей, а ranking feedback можно использовать для улучшения retrieval negatives.

## 2. Контекст

В промышленных recommender systems retrieval отвечает за высокий recall, а ranking - за точный порядок. Разделение удобно, но создает mismatch: retrieval учится на sampled softmax/next-item objective, ranking - на CTR labels. UniGRF проверяет, можно ли использовать один autoregressive backbone и совместный loss без деградации ranking.

## 3. Метод / pipeline

- **Input sequence:** история задается как чередование item $i_k$ и feedback $b_k$, например click/no-click.
- **Retrieval head:** hidden state позиции следующего item обучается через SampledSoftmax предсказывать next item embedding.
- **Ranking head:** hidden state feedback-позиции проходит через MLP и BCE loss для click probability.
- **Backbone-agnostic:** авторы применяют framework к HSTU и Llama-like generative recommenders.
- **Ranking-driven enhancer:** hard negatives - items с высоким retrieval score и низким ranking score; potential favorite items - candidates с ranking score выше threshold alpha, которые можно relabel как positive.
- **Gradient-guided adaptive weighter:** веса retrieval/ranking loss меняются по отношению current/previous loss; более медленно сходящийся objective получает больший вес.

## 4. Результаты и evidence

Датасеты: MovieLens-1M (6,040 users, 3,706 items, 1,000,209 interactions), MovieLens-20M (138,493 / 26,744 / 20,000,263) и Amazon-Books (694,897 / 686,623 / 10,053,086). Evaluation: leave-one-out по последнему interaction; retrieval candidate set - полный item set.

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>Лучшая UniGRF retrieval конфигурация</th><th>Ключевые метрики</th></tr></thead>
<tbody>
<tr><td>MovieLens-1M</td><td>UniGRF-Llama</td><td>NDCG@10 0.1765, NDCG@50 0.2368, HR@10 0.3219, HR@50 0.5921; UniGRF-HSTU дает лучший MRR 0.1484.</td></tr>
<tr><td>MovieLens-20M</td><td>UniGRF-Llama</td><td>NDCG@10 0.1891, NDCG@50 0.2464, HR@10 0.3270, HR@50 0.5846, MRR 0.1652.</td></tr>
<tr><td>Amazon-Books</td><td>UniGRF-HSTU</td><td>NDCG@10 0.0354, NDCG@50 0.0522, HR@10 0.0638, HR@50 0.1415, MRR 0.0319.</td></tr>
</tbody>
</table>
</div>

Ranking AUC тоже улучшается: UniGRF-Llama достигает 0.7932 на ML-1M; UniGRF-HSTU - 0.7941 на ML-20M и 0.7672 на Amazon-Books. Ablation на ML-1M показывает, что без enhancer и adaptive weighting ranking может проседать: HSTU AUC 0.7621, w/o Both 0.7293, UniGRF-HSTU 0.7832.

## 5. Ограничения

- Нет online deployment; результаты только на публичных датасетах.
- Авторы сами указывают future work: перенести framework на industrial datasets и объединить также pre-ranking/reranking.
- Retrieval все еще завязан на sampled softmax и top-k candidate scoring; это не полная генерация человеко-читаемых IDs.
- Hard negative strategy чувствительна к числу negatives: на ML-1M слишком большой m приводит к переобучению.

## 6. Связь с GR/SID

Статья важна для generative recommendation как попытка сделать GR не только recall model, но и ranker. В отличие от SID-центричных работ, фокус здесь на multi-task objective и feedback loop между retrieval/ranking, а не на дизайне идентификаторов.
