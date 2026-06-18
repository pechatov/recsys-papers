---
title: "Large Language Model as Universal Retriever in Industrial-Scale Recommender System"
category: "generative_retrieval"
slug: "llm_universal_retriever_industrial_scale_recommender_summary"
catalogId: "paper-llm_universal_retriever_industrial_scale_recommender_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/llm_universal_retriever_industrial_scale_recommender_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2502.03041"
---
> **Авторы:**
>
> Junguang Jiang, Yanwen Huang, Bin Liu, Xiaoyu Kong, Xinhang Li, Ziru Xu, Han Zhu, Jian Xu, Bo Zheng.
>
>   
>
>
> **Аффилиация:**
>
> Taobao & Tmall Group of Alibaba, China.
>
>   
>
>
> **Источник:**
>
> arXiv:2502.03041, 2025.

## 1. Коротко

URM предлагает использовать LLM как universal retriever: одна модель получает user features и текстовое описание retrieval objective, затем возвращает candidates для разных целей и сценариев. В отличие от SID-decoding моделей, URM не генерирует item token sequence; она делает один LLM forward и скорит десятки миллионов items через factorized item distribution и ANN/sampling.

## 2. Контекст

Industrial recommender обычно содержит много retrieval channels: click prediction retrieval, scenario-specific retrieval, long-tail retrieval, purchase retrieval, query-aware retrieval. Отдельные модели дешевле, но плохо переносят знания между задачами и требуют много serving/maintenance. URM формулирует задачу как $P(v \mid u,o)$, где $u$ - пользовательский контекст, $o$ - retrieval objective на естественном языке, $v$ - item.

## 3. Метод / pipeline

- **LLM backbone:** Qwen-7B полностью fine-tuned. User features и objective сериализуются как текст; item IDs добавляются как special tokens через distributed hashtable и one-layer MLP.
- **Multi-query representation:** к input добавляют $M$ learnable query tokens; item score берется как максимум inner product по $M$ user representations. В production-like setting используют M=128.
- **Matrix decomposition:** item matrix $W$ раскладывается как $U V^\top$ с H=128. $V_{dis}$ учится по item IDs, $V_{trans}$ строится из item text через General Text Embedding LLM и помогает unseen items.
- **Probabilistic sampling:** HNSW по $W$, iterative subset+neighbor sampling. При T=4 precision относительно full retrieval достигает 91.0%, а compute снижается с примерно 5000G FLOPs до около 2G.
- **Serving:** asynchronous workflow: запуск по user actions, 10-minute window, vLLM single prefill, результаты сохраняются для downstream retrieval.

## 4. Результаты и evidence

<div class="table-scroll">
<table>
<thead><tr><th>Public dataset</th><th>URM HR@5</th><th>URM NDCG@5</th><th>Сравнение с лучшим baseline</th></tr></thead>
<tbody>
<tr><td>Sports</td><td>0.0733</td><td>0.0488</td><td>против IDGenRec 0.0429 / 0.0326</td></tr>
<tr><td>Beauty</td><td>0.0929</td><td>0.0671</td><td>против 0.0618 / 0.0486</td></tr>
<tr><td>Toys</td><td>0.0888</td><td>0.0619</td><td>HR выше IDGenRec 0.0655; NDCG выше P5 0.0567</td></tr>
<tr><td>Yelp</td><td>0.0724</td><td>0.0476</td><td>против P5 0.0574 / 0.0403</td></tr>
</tbody>
</table>
</div>

Авторы суммируют public results как средние +46% HR@5 и +29% NDCG@5; в appendix для HR@10/NDCG@10 указаны +49% и +37%.

На промышленном multi-objective benchmark URM MTL достигает average R@1000 0.403, лучше Attention-DNN PLE 0.363 и Transformer MTL 0.308. По 9 objectives URM лучший на 6 и дает +11.0% average uplift over best baseline. Online A/B с 28 апреля по 14 мая 2025: Revenue +3.01%, CTR +0.78%, CVR +1.24%, number of long-tail items +2.23%.

Ablations: $V_{dis}$ alone дает 0.256 overall / 0.116 unseen, $V_{trans}$ - 0.152 / 0.101, combined - 0.263 / 0.130. Sampling precision растет от 0.2% при T=1 до 91.0% при T=4 и почти не улучшается при T=5.

## 5. Ограничения

- LLM retrieval дороже традиционных retrieval моделей; авторы используют 5% hourly sample для daily training within 24h и asynchronous inference.
- Новые objectives не появляются "бесплатно": нужны training mappings для текстового objective и соответствующего behavior signal.
- Индустриальные данные и код не раскрыты; production latency описана в общих терминах, LLM inference остается dominant cost.
- Модель заменяет SID decoding другим способом скоринга items, поэтому выводы не являются прямым доказательством превосходства LLM-generated IDs.

## 6. Связь с GR/SID

URM - сильная альтернатива semantic-ID generative retrieval. Она сохраняет LLM как универсальный conditional retriever, но уходит от autoregressive SID decoding к matrix-factorized item scoring. Это адресует две типовые проблемы SID: discriminability крупных каталогов и cold-start/unseen items через $V_{trans}$.
