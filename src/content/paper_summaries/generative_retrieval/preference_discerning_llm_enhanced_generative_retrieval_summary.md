---
title: "Preference Discerning with LLM-Enhanced Generative Retrieval"
category: "generative_retrieval"
slug: "preference_discerning_llm_enhanced_generative_retrieval_summary"
catalogId: "paper-preference_discerning_llm_enhanced_generative_retrieval_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/preference_discerning_llm_enhanced_generative_retrieval_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2412.08604"
---
> **Авторы:**
>
> Fabian Paischer, Liu Yang, Linfeng Liu, Shuai Shao, Kaveh Hassani, Jiacheng Li, Ricky Chen, Zhang Gabriel Li, Xiaoli Gao, Wei Shao, Xue Feng, Nima Noorshams, Sem Park, Bo Long, Hamid Eghbalzadeh.
>
>   
>
>
> **Аффилиации:**
>
> ELLIS Unit / LIT AI Lab / JKU Linz, University of Wisconsin-Madison, AI at Meta.
>
>   
>
>
> **Источник:**
>
> arXiv:2412.08604, 2024; в исходнике также указан OpenReview/TMLR accepted 2025.

## 1. Коротко

Статья вводит задачу *Preference Discerning*: генеративная рекомендательная модель должна не просто предсказывать следующий item ID, а учитывать явно сформулированные пользовательские предпочтения на естественном языке. Это важно для generative retrieval: semantic-ID модели вроде TIGER хорошо генерируют идентификаторы товаров, но плохо объясняют и контролируют, *какие* аспекты интересов пользователя были использованы.

Авторы предлагают Mender, LLM-enhanced generative retrieval framework. LLM сначала аппроксимирует предпочтения пользователя из отзывов и истории, затем encoder-decoder модель генерирует semantic IDs товаров, условно на истории и текстовых предпочтениях.

## 2. Контекст

Классический sequential recommendation учится на последовательности взаимодействий. В generative retrieval товар кодируется последовательностью semantic tokens, и модель декодирует эти токены как текст. Такой подход масштабируется к большому каталогу, но обычно не различает положительные/отрицательные, тонкие/грубые и конфликтующие предпочтения пользователя.

Авторы специально конструируют тестовые оси: preference-based recommendation, sentiment following, fine-grained steering, coarse-grained steering и history consolidation. Поэтому оценка выходит за пределы обычного Recall/NDCG: проверяется, может ли модель следовать явно заданному вкусу, например избегать нежелательных свойств или менять рекомендации при изменении preference prompt.

## 3. Метод / pipeline

### Preference approximation

Для каждого user-time step LLaMA-3-70B-Instruct получает историю, отзывы и item metadata и генерирует ровно пять предпочтений. Prompt просит фокусироваться на свойствах товара и вкусе пользователя, включая aversions, и игнорировать нерелевантные факторы вроде доставки или цены. Всего получено около 5 млн preference records.

### Mender

- Базовый item representation следует TIGER: Sentence-T5 item embeddings с RQ-VAE/semantic IDs, которые затем декодируются как token sequence.
- Encoder использует pretrained FLAN-T5-Small; decoder инициализируется случайно и генерирует semantic IDs через cross-attention к истории и preferences.
- **MenderTok** кодирует историю и preferences как одну текстовую последовательность токенов.
- **MenderEmb** кодирует каждый item/preference отдельно через Instructor embeddings, чтобы ускорить обучение и инференс.

## 4. Результаты и evidence

Основные датасеты: Amazon Beauty, Toys, Sports и Steam; в таблице статистики также указан Yelp. Размеры: Beauty 22,363 users / 12,101 items / 198,502 actions; Toys 19,412 / 11,924 / 167,597; Sports 35,598 / 18,357 / 296,337; Steam 47,761 / 10,403 / 599,620. Число preference records: Beauty 992,510; Toys 837,985; Sports 1,481,685; Steam 2,026,225.

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>MenderTok Recall@10</th><th>MenderTok NDCG@10</th><th>SASRec Recall@10</th><th>SASRec NDCG@10</th></tr></thead>
<tbody>
<tr><td>Beauty</td><td>0.0937</td><td>0.0508</td><td>0.0528</td><td>0.0227</td></tr>
<tr><td>Sports</td><td>0.0427</td><td>0.0234</td><td>0.0271</td><td>0.0118</td></tr>
<tr><td>Toys</td><td>0.0799</td><td>0.0432</td><td>0.0615</td><td>0.0267</td></tr>
<tr><td>Steam</td><td>0.2040</td><td>0.1560</td><td>0.1781</td><td>0.1469</td></tr>
</tbody>
</table>
</div>

По тексту статьи, MenderTok дает до 45% относительного улучшения на recommendation axis и до 70.5% на fine-grained steering. При этом sentiment following и coarse-grained steering остаются трудными без специальной data augmentation: модель лучше реагирует на тонкие предпочтения, чем на обобщенные команды типа "больше такого жанра" или "избегай отрицательных предпочтений".

Цена качества высокая: MenderTok обучался на A100 от 1021 до 2350 минут в зависимости от датасета, а инференс занимал примерно 210-562 мс, тогда как SASRec работал за 5-9 мс. MenderEmb существенно быстрее, но обычно уступает по качеству.

## 5. Ограничения

- Preference approximation зависит от крупного LLM и наличия отзывов; ручная инспекция/постобработка остаются частью pipeline.
- Language encoder и Transformer дают высокую стоимость инференса; MenderTok трудно напрямую переносить в latency-sensitive production retrieval.
- Coarse steering и sentiment following требуют отдельного синтетического обучения; "понимание предпочтений" не возникает автоматически из semantic IDs.
- Оценка демонстрирует controllability, но не показывает online A/B на промышленном трафике.

## 6. Связь с GR/SID

Работа расширяет TIGER-style generative retrieval: semantic IDs остаются выходным пространством, но вход модели обогащается явными natural-language preferences. Для SID-направления это важный сигнал: качество идентификаторов решает не все, потому что управляемость и preference alignment требуют отдельного слоя данных и обучения.
