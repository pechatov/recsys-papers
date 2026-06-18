---
title: "Alleviating LLM-based Generative Retrieval Hallucination in Alipay Search"
category: "generative_retrieval"
slug: "alleviating_llm_generative_retrieval_hallucination_alipay_search_summary"
catalogId: "paper-alleviating_llm_generative_retrieval_hallucination_alipay_search_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/alleviating_llm_generative_retrieval_hallucination_alipay_search_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2503.21098"
---
> **Авторы:**
>
> Yedan Shen, Kaixin Wu, Yuechen Ding, Jingyuan Wen, Hong Liu, Mingjie Zhong, Zhouhan Lin, Linjian Mo, Jia Xu.
>
>   
>
>
> **Аффилиации:**
>
> Ant Group, Hangzhou; LUMIA Lab, Shanghai Jiao Tong University.
>
>   
>
>
> **Источник:**
>
> arXiv:2503.21098, 2025; исходник содержит SIGIR 2025 metadata.

## 1. Коротко

Статья рассматривает конкретную production-проблему LLM-based generative retrieval: модель может вернуть валидный, но нерелевантный финансовый/страховой продукт. В Alipay Search это называется hallucination и напрямую бьет по trust, clicks и conversion.

Решение двухступенчатое: Knowledge Distillation Reasoning улучшает само SFT обучение GR, а Decision Agent фильтрует сгенерированные результаты через структурированную LLM-проверку релевантности.

## 2. Контекст

Домены - Fund Search и Insurance Search в Alipay. Пользовательские запросы часто fuzzy или complex: например, не точное название фонда, а желаемая тема, риск, срок или страховой сценарий. В базовой постановке product title используется как DocID, а Qwen2.5-14B fine-tuned как generative retriever.

## 3. Метод / pipeline

- **Preliminary GR:** SFT на annotated relevant query-document pairs и structured product knowledge <d,k>; output identifier - product title.
- **Negative mining:** предварительный GR retrieves docs; ensemble of open-source LLM judges relevance; только docs, признанные нерелевантными всеми judges, становятся negative pairs.
- **Reasoning distillation:** Qwen2.5-72B генерирует объяснения релевантности/нерелевантности для positive и negative pairs; эти reasoning records добавляются к SFT. На сценарий используется около 20k reasoning records.
- **Decision Agent:** для каждого GR result BM25 находит top-m related docs, затем Qwen2.5-32B оценивает relevance по structured perspectives: company, product type, duration/risk и т.п. Сохраняются только релевантные результаты.
- **Top-K selection:** по ACC curve выбран K=5: Fund ACC 87.17, Insurance ACC 90.05.

## 4. Результаты и evidence

<div class="table-scroll">
<table>
<thead><tr><th>Model</th><th>Fund ACC</th><th>Insurance ACC</th></tr></thead>
<tbody>
<tr><td>BM25</td><td>69.33%</td><td>47.33%</td></tr>
<tr><td>GR baseline</td><td>83.83%</td><td>85.83%</td></tr>
<tr><td>Ours</td><td><strong>87.17%</strong></td><td><strong>90.05%</strong></td></tr>
<tr><td>w/o reasoning</td><td>85.17%</td><td>87.66%</td></tr>
<tr><td>w/o decision agent</td><td>85.50%</td><td>86.17%</td></tr>
</tbody>
</table>
</div>

Dataset scale: около 250k fund records, 200k insurance records и 20k reasoning records per scenario. Test queries разделены по frequency buckets из logs и размечены annotators как relevant/irrelevant.

Online A/B: Variant A - current baseline recall system с sparse+dense retrieval; Variant B добавляет optimized GR как дополнительный recall path. Fund overall: Click_PV +2.84%, Click_UV +1.45%, Trade_Count +1.31%, Trade_UV +0.99%. На broad fund queries effect крупнее: Click_PV +12.71%, Click_UV +11.06%, Trade_Count +11.75%, Trade_UV +10.63%. Insurance: Trade_Count +1.89%, Trade_UV +2.07%. Статья указывает significance 95%, p < 0.05.

## 5. Ограничения

- Reasoning data строится LLM judges и larger LLM, то есть hallucination mitigated LLMs контролируется другими LLMs.
- Даже после reasoning нужен Decision Agent, значит проблема не решена только обучением retriever.
- Structured perspectives доменно-специфичны: фондовый и страховой поиск имеют свои поля риска, срока, компании и типа продукта.
- Промышленные logs, exact candidate sizes и latency/cost Decision Agent не раскрыты полностью.

## 6. Связь с GR/SID

Работа показывает важную сторону GR, которая не видна в чистых SID-бенчмарках: валидный идентификатор еще не гарантирует релевантность. Даже если title/DocID существует, LLM может "галлюцинировать" соответствие query-product. Поэтому production GR требует relevance reasoning, hard negatives и post-generation verification.
