---
title: "Generative Retrieval and Alignment Model: A New Paradigm for E-commerce Retrieval"
category: "generative_retrieval"
slug: "generative_retrieval_alignment_model_ecommerce_retrieval_summary"
catalogId: "paper-generative_retrieval_alignment_model_ecommerce_retrieval_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/generative_retrieval_alignment_model_ecommerce_retrieval_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2504.01403"
---
> **Авторы:**
>
> Ming Pang, Chunyuan Yuan, Xiaoyu He, Zheng Fang, Donghao Xie, Fanyi Qu, Xue Jiang, Changping Peng, Zhangang Lin, Ching Law, Jingping Shao.
>
>   
>
>
> **Аффилиация:**
>
> JD.com, Beijing, China.
>
>   
>
>
> **Источник:**
>
> arXiv:2504.01403, 2025.

## 1. Коротко

GRAM предлагает generative retrieval для e-commerce search, где модель генерирует не opaque item IDs, а структурированные текстовые коды товара/запроса. Главная идея: использовать LLM для генерации атрибутивных codes, а затем выровнять query-code-product пространство так, чтобы retrieval и pre-ranking работали как единый pipeline.

В отличие от SID-подходов с beam search по learned token IDs, GRAM опирается на осмысленные product attributes: бренд, категория, серия, модель, материал, стиль, цвет, спецификации, аудитория, сценарии применения, marketing terms и т.п.

## 2. Контекст

Авторы критикуют sparse/dense retrieval в e-commerce за слабую работу с fine-grained interaction и memory. ID-based GR требует изучать новую систему идентификаторов и дорого декодировать top-k через beam search. Строковые/n-gram идентификаторы используют знания LLM, но query/product language mismatch и контроль качества кодов остаются проблемой.

## 3. Метод / pipeline

<figure class="paper-figure">
  <img src="../../assets/gram/architecture.png" alt="GRAM generative retrieval and alignment model architecture">
  <figcaption>Рисунок 1. Архитектура GRAM: query и product переводятся в structured text codes, затем co-training и co-alignment связывают генерацию кодов с online relevance.</figcaption>
</figure>

### Structured text codes

Коды строятся из 16 типов атрибутов. Coarse code содержит 1-2 атрибута, medium - 3, fine - более 3. Начальные коды получают из экспертно размеченных запросов: BERT NER обучается на примерно 3 млн запросов и извлекает attributes из query/product/click logs. Начальное обучение покрывает около 6 млн уникальных запросов и 8 млн товаров.

### Query-code и product-code generation

Открытая/внутренняя 0.5B LLM обучается генерировать codes для запросов и товаров. Product generator создает коды для активных товаров; query generator генерирует до 10 codes на связанный запрос; relevance filter добавляет в обучение только качественные пары.

### Alignment

- **Code co-alignment:** DPO на query-product pair; positive code - пересечение query/product tags, negative - разность.
- **Query-product alignment through code:** после code matching товар скорится через token-level Jensen-Shannon divergence между вероятностями query-code и product-code.
- **Pre-ranking weights:** code-granular weights обучаются pairwise margin loss.

В production GRAM объединяет intent recognition, retrieval и pre-ranking; новые товары получают codes nearline.

## 4. Результаты и evidence

Данные SFT: 184.8M query-code pairs, 459.7M product-code pairs, 6.2M unique queries, 8.4M products, 7.4M codes. Alignment: 67.4M query-code, 134.8M product-code, 1.5M queries, 15.6M products, 452.7K codes.

<div class="table-scroll">
<table>
<thead><tr><th>Model</th><th>Recall@10</th><th>Recall@100</th><th>Recall@300</th><th>RelR</th></tr></thead>
<tbody>
<tr><td>BM25</td><td>3.01%</td><td>10.52%</td><td>15.23%</td><td>35.78%</td></tr>
<tr><td>DPR</td><td>3.89%</td><td>11.26%</td><td>17.92%</td><td>30.96%</td></tr>
<tr><td>LC-Rec</td><td>4.35%</td><td>7.16%</td><td>7.33%</td><td>23.94%</td></tr>
<tr><td><strong>GRAM</strong></td><td>2.85%</td><td><strong>12.54%</strong></td><td><strong>21.13%</strong></td><td><strong>40.18%</strong></td></tr>
</tbody>
</table>
</div>

GRAM не лучший на Recall@10, но выигрывает Recall@100/300 и relevance rate. Ablation показывает вклад alignment: full GRAM 2.85/12.54/21.13/40.18; без code alignment 1.80/11.37/20.23/33.51; без co-training и code alignment 1.57/7.64/12.89/33.36.

Online A/B на JD search: 5% трафика против 5% baseline, минимум неделя, beam size 10, до 300 ads. GRAM дал +0.74% ad impressions, +1.27% CTR, +0.45% CPC и +2.46% ad revenue, p < 0.05.

Главный нюанс интерпретации: GRAM не пытается победить все retrieval baselines на top-10. Его сильная зона - расширить candidate set релевантными товарами и поднять качество pre-ranking/revenue. Поэтому Recall@100/300 и RelR здесь важнее, чем единичный Recall@10.

## 5. Что проверять при реализации

GRAM требует отдельного контроля качества на трех уровнях.

Первый уровень - качество structured codes. Нужно логировать coverage атрибутов, долю пустых/слишком общих codes, распределение coarse/medium/fine codes, количество товаров на code и количество codes на товар. Если код вроде "phone accessory" покрывает слишком много объектов, retrieval станет широким, но плохо управляемым.

Второй уровень - query-product-code alignment. Здесь важны не только Recall@K, но и exact/partial overlap между query codes и product codes, доля generated codes, отсутствующих в каталоге, и error analysis по intent types: брендовые запросы, compatibility queries, long-tail спецификации, seasonal terms.

Третий уровень - serving. GRAM добавляет генерацию и code matching перед pre-ranking, поэтому нужно считать latency на генерацию codes, размер candidate set до и после filters, deduplication rate, conflict rate между несколькими codes одного товара и revenue/CTR отдельно по head/tail queries.

## 6. Ограничения

- Данные, taxonomy атрибутов и traffic logs закрытые; воспроизводимость ограничена.
- Recall@10 ниже LC-Rec и DPR, то есть модель полезнее для широкого candidate generation, чем для сверхточного top-10.
- Качество зависит от NER, product attributes и relevance filter; ошибки в каталожной разметке напрямую влияют на retrieval.
- В тексте статьи масштаб deployment описан не вполне единообразно: встречается формулировка про millions и hundreds of millions retrievals.
- Метод сильнее всего подходит для товарного поиска с богатыми атрибутами. В доменах без устойчивой taxonomy structured text codes могут стать просто noisy keywords.
- Fuzzy alignment через похожие attributes может увеличить recall, но ухудшить precision на совместимых, но не взаимозаменяемых товарах, например аксессуары против основного устройства.

## 7. Связь с GR/SID

GRAM - альтернатива learned semantic IDs: идентификатор остается генерируемой последовательностью, но она человеко-читаемая и привязана к e-commerce attributes. Для SID это полезный контраст: интерпретируемый code может упростить alignment и контроль relevance, но требует сильной доменной схемы.

По отношению к TIGER/LETTER GRAM делает другой trade-off. Learned SID компактнее и лучше подходит для constrained decoding по item trie, но плохо интерпретируем. Structured text code дороже и зависит от taxonomy, зато дает понятный debugging surface: можно посмотреть, какие именно attributes привели товар в выдачу.
