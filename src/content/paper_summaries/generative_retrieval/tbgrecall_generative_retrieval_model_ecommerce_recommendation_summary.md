---
title: "TBGRecall: A Generative Retrieval Model for E-commerce Recommendation Scenarios"
category: "generative_retrieval"
slug: "tbgrecall_generative_retrieval_model_ecommerce_recommendation_summary"
catalogId: "paper-tbgrecall_generative_retrieval_model_ecommerce_recommendation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/tbgrecall_generative_retrieval_model_ecommerce_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2508.11977"
---
> **Авторы:**
>
> Zida Liang, Changfa Wu, Dunxian Huang, Weiqiang Sun, Ziyang Wang, Yuliang Yan, Jian Wu, Yuning Jiang, Bo Zheng, Ke Chen, Silu Zhou, Yu Zhang.
>
>   
>
>
> **Аффилиации:**
>
> Shanghai Jiao Tong University; Alibaba Inc.
>
>   
>
>
> **Источник:**
>
> arXiv:2508.11977, 2025.

## 1. Коротко

TBGRecall - генеративная recall-модель для Taobao "Guess You Like". Она формулирует e-commerce homepage recommendation как *Next Session Prediction*: модель читает последовательность прошлых сессий и выдает user latent vector для ANN retrieval товаров следующей сессии.

Работа важна тем, что не пытается декодировать semantic IDs. В e-commerce item pool относительно стабилен, поэтому авторы используют sparse item IDs и масштабируют Transformer/MoE до industrial setting.

## 2. Контекст

В Taobao homepage backend возвращает набор товаров на каждый open/scroll request, а внутри сессии у товаров нет естественного causal order. Поэтому обычное next-item modeling плохо совпадает с задачей. TBGRecall делает causal modeling по сессиям, но запрещает causal связи между items внутри одной сессии.

## 3. Метод / pipeline

- **Session-wise autoregression:** sequence состоит из context token и unordered items каждой сессии; hidden state context token следующей сессии становится user representation для retrieval.
- **Token representation:** сумма ID embedding, action embedding (click/exposure), side information (category, seller, price, time) и context/scenario embedding.
- **Session Mask:** causal mask сохраняется между сессиями, но скрывает взаимодействия items внутри одной сессии.
- **Session-wise RoPE:** все tokens одной сессии получают одинаковую позиционную фазу.
- **TSN:** token-specific network с отдельными линейными преобразованиями для context и item tokens на входе Transformer.
- **MSP и MoE:** Multi-Session Prediction по session dimension; DeepSeekMoE с 24 routed experts, 1 shared expert и top-2 activation.
- **PIT:** stochastic partial incremental training для ежедневного обновления без полного переобучения.

## 4. Результаты и evidence

<div class="table-scroll">
<table>
<thead><tr><th>Dataset / model</th><th>HR@20</th><th>HR@500</th><th>HR@1000</th><th>HR@4000</th></tr></thead>
<tbody>
<tr><td>RecFlow HSTU</td><td>0.06%</td><td>1.21%</td><td>2.19%</td><td>7.38%</td></tr>
<tr><td>RecFlow TBGRecall</td><td><strong>0.26%</strong></td><td><strong>2.45%</strong></td><td><strong>3.85%</strong></td><td><strong>8.20%</strong></td></tr>
<tr><td>Taobao online baseline</td><td>1.03%</td><td>9.37%</td><td>13.36%</td><td>23.57%</td></tr>
<tr><td>Taobao TBGRecall</td><td><strong>1.53%</strong></td><td><strong>11.81%</strong></td><td><strong>16.62%</strong></td><td><strong>29.45%</strong></td></tr>
</tbody>
</table>
</div>

Ablation на Taobao показывает вклад компонентов: full model HR@4000 29.45%; без TSN 28.88%; без MSP 27.28%; без MoE 28.47%; без session-wise RoPE 27.36. MSP и session-wise positional modeling оказываются наиболее заметными.

Training comparison: Normal training 5 дней на 128 GPUs дает HR@4000 23.76%; Incre - 16 часов на 1280 GPUs и 29.76%; SL - 48 часов на 128 GPUs и 26.50%; PIT - 11 часов на 128 GPUs и 29.45%. Online A/B в Taobao GUL на сотнях миллионов daily exposures, 5% randomized traffic, 7 дней: PVR +23.94%, Transaction Count +0.60%, Transaction Amount +2.16%.

## 5. Ограничения

- Ключевые результаты завязаны на закрытый Taobao corpus и production infra.
- Модель не генерирует IDs напрямую в serving: она выдает user vector, затем ANN ищет товары по item embeddings.
- Sparse item IDs хорошо подходят стабильному e-commerce каталогу, но хуже применимы к domains с высокой cold-start/freshness динамикой.
- Next Session Prediction специализирована под unordered homepage sessions и не заменяет универсальную sequence modeling постановку.

## 6. Связь с GR/SID

TBGRecall расширяет industrial GR за пределы SID decoding: генеративность здесь в session-level Transformer objective, а retrieval реализуется через ANN по hidden representation. Для SID-дискуссии это аргумент, что на production scale иногда важнее корректная temporal/session постановка и incremental training, чем дискретизация items.
