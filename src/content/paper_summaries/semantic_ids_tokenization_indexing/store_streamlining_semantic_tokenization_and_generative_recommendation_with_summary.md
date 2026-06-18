---
title: "STORE: Streamlining Semantic Tokenization and Generative Recommendation with A Single LLM"
category: "semantic_ids_tokenization_indexing"
slug: "store_streamlining_semantic_tokenization_and_generative_recommendation_with_summary"
catalogId: "paper-store_streamlining_semantic_tokenization_and_generative_recommendation_with_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/store_streamlining_semantic_tokenization_and_generative_recommendation_with_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2409.07276"
---
> **Авторы:** Qijiong Liu, Jieming Zhu, Lu Fan, Zhou Zhao, Xiao-Ming Wu.
>
> **Аффилиации:** Hong Kong Polytechnic University; Huawei Noah's Ark Lab; Zhejiang University.
>
> **Первичный источник:** arXiv:2409.07276v1 для статьи STORE. Важно: текущая версия arXiv v3 переименована и заменена на LAMIA, поэтому для разбора STORE использована версия v1 PDF.

## 1. Коротко: о чем статья

STORE предлагает упростить pipeline generative recommendation с semantic tokenization. Типичная схема TIGER-like систем состоит из нескольких независимых модулей: content embedder, quantizer вроде RQ-VAE и autoregressive recommender. STORE пытается заменить этот набор одним LLM backbone, который одновременно умеет делать semantic tokenization и generative recommendation.

Ключевая формулировка: semantic tokenization превращается в **text-to-token** generation, а recommendation - в **token-to-token** generation. Чтобы не потерять связь между item text и discrete tokens, STORE добавляет **token-to-text reconstruction** для tokenizer stage и **text-to-token auxiliary task** для recommendation stage.

## 2. Контекст: почему multi-stage semantic tokenization неудобен

В generative recommendation item представляется sequence of semantic tokens. Эти tokens затем генерируются моделью как next-item identifier. Такой подход компактнее atomic item IDs и лучше переносится на long-tail/cold-start, но в классическом pipeline есть несколько слабых мест.

- **Embedder domain gap.** Item embeddings часто берутся из frozen general-domain text encoder, который не обязательно понимает товарные, новостные или ресторанные признаки в нужном домене.
- **Quantizer fragility.** RQ-VAE может страдать codebook collapse, collisions и чувствительностью к hyperparameters.
- **Disjoint optimization.** Embedder, quantizer и recommender оптимизируются отдельно; knowledge не проходит свободно между ними.
- **Operational complexity.** В production нужно тренировать, версионировать и обслуживать несколько моделей и mapping tables.

## 3. Главная идея STORE

STORE сохраняет смысл semantic tokens, но формулирует все вокруг generation tasks внутри одного LLM. Вместо "embedder -> quantizer -> recommender" предлагается unified instruction-based pipeline:

1. LLM учится генерировать dense semantic tokens по item content.
1. Непараметрический clusterer переводит dense tokens в discrete tokens.
1. Тот же LLM backbone используется для next-item generation по token sequence.
1. Auxiliary tasks помогают LLM понимать соответствие между text и tokens.

Таким образом STORE не просто заменяет RQ-VAE другой quantizer, а меняет границу между tokenization и recommendation: обе задачи становятся разными instruction tasks одного generative model.

## 4. Dense tokenizer

Dense tokenizer в STORE берет item content и производит token embeddings. В v1 статье это описано как text-to-token conversion через cascaded attention mask и self-supervised tasks. Идея в том, что модель не должна просто взять frozen embedding и квантовать его; она должна адаптировать representation к recommendation domain.

Важная деталь: авторы не делают сложный differentiable VQ внутри LLM. После получения dense tokens они используют более простой clusterer. Это снижает риск нестабильности RQ-VAE, хотя переносит часть ответственности на качество dense tokenizer и clustering.

## 5. Clusterer и discrete tokens

Для discretization STORE использует PCA для уменьшения размерности dense tokens и k-means для получения discrete token IDs. В paper приведены settings: 1024-dimensional representations редуцируются перед clustering, а затем discrete codes используются как item identifiers. Это проще, чем обучать RQ-VAE с residual codebooks, commitment losses и codebook updates.

Плюс такого решения - простота и воспроизводимость. Минус - k-means остается offline step: если dense tokenizer меняется, token map тоже надо обновлять и синхронизировать с recommender.

## 6. Post-pretraining и instruction tuning

STORE добавляет post-pretraining tasks, чтобы LLM лучше понимала связь между item content и semantic tokens. Основные задачи:

- **Text-to-token.** По item text сгенерировать соответствующие semantic tokens.
- **Token-to-text reconstruction.** По semantic tokens восстановить item text или content signal, снижая information loss.
- **Token-to-token recommendation.** По user history в token space сгенерировать next item tokens.
- **Text-token alignment.** Auxiliary task во время recommendation tuning, чтобы LLM не воспринимала tokens как бессмысленные новые symbols.

Training реализуется через instruction templates. Это важно: STORE пытается использовать сильную сторону LLM - unified generative interface - не только для recommendation, но и для tokenizer learning.

## 7. Conditional beam search

На inference STORE строит code tree, который содержит все валидные token combinations. При beam search logits невалидных branches маскируются, чтобы модель генерировала только существующие item identifiers. Это аналог constrained decoding в generative retrieval.

Авторы противопоставляют conditional beam search более мягким ограничениям LC-Rec: soft constraints могут оставить path, который не соответствует реальному item; code-tree masking делает generated identifier валидным по конструкции.

## 7.1. Пошаговый алгоритм STORE

1. **Сформировать instruction data для tokenizer-а.** Item text превращается в prompts text-to-token, а semantic token slots учатся как dense outputs одного LLM backbone.
1. **Обучить dense tokenizer.** Cascaded attention mask и self-supervised tasks заставляют LLM создавать token embeddings, адаптированные к recommendation domain, а не просто повторять frozen text encoder.
1. **Дискретизировать dense tokens.** PCA уменьшает размерность 1024D representations, k-means превращает dense slots в discrete semantic token IDs.
1. **Добавить token-to-text reconstruction.** По discrete/dense semantic tokens LLM восстанавливает item content, чтобы снизить information loss и сохранить связь token space с текстом.
1. **Обучить token-to-token recommender.** User history как sequence of semantic tokens подается тому же LLM, target - tokens следующего item.
1. **Добавить text-token alignment во время recommendation tuning.** Auxiliary task не дает LLM воспринимать semantic tokens как произвольные новые symbols без content grounding.
1. **Инференс через code tree.** Conditional beam search маскирует невалидные branches и возвращает только token paths, существующие в item catalog.

## 8. Экспериментальный setup

STORE оценивается на двух real-world datasets:

<div class="table-scroll">
<table>
<tr><th>Dataset</th><th>Домен</th><th>Почему важен</th></tr>
<tr><td>MIND</td><td>news recommendation</td><td>item text информативен, user histories длиннее, меньше items относительно Yelp</td></tr>
<tr><td>Yelp</td><td>restaurant recommendation</td><td>больше items, короче истории, ниже average item appearance</td></tr>
</table>
</div>

Baselines включают traditional sequential recommenders и LLM/recommendation baselines: SASRec, BERT4Rec, P5, TIGER, LC-Rec. Для retrieval используются Recall@1/5/10/20 и NDCG@1/5/10/20. Также есть scoring scenario с AUC и ranking-style metrics.

## 9. Основные результаты

В retrieval results semantic-code group в целом выигрывает у classic ID/text-only approaches. STORE показывает лучшие результаты среди сравниваемых baselines на MIND и Yelp. Авторы отдельно отмечают, что MIND оказывается более благоприятным: item content news titles более информативен, а catalog меньше и плотнее. Yelp сложнее из-за большего item space и меньшей частоты появления item.

В scoring scenario STORE также превосходит TIGER и LC-Rec. Это важно: метод не ограничивается только exact token generation, а может использовать semantic tokens и LLM backbone для score prediction / CTR-like settings.

## 10. Абляции

Абляции в Table 4 особенно важны, потому что они проверяют не только final architecture, но и причины выигрыша:

- **STORE w/o dense tokenizer.** Возвращает pipeline ближе к стандартной semantic tokenization через RQ-VAE. Полный STORE лучше, значит text-to-token dense tokenizer действительно полезен.
- **STORE w/o conditional beam search.** Использует soft beam constraints. Качество ниже, что подтверждает важность code-tree constrained decoding.
- **STORE w/o text-token alignment task.** Убирает auxiliary alignment между textual content и discrete tokens. Просадка показывает, что LLM нужно учить понимать новые semantic tokens.

## 11. Что важно в рисунках

- **Figure 1.** Показывает разницу между conventional unique identifiers и semantic identifiers: semantic codes разделяются между похожими items.
- **Figure 2.** Главная схема paper: старый pipeline embedder -> quantizer -> recommender против STORE pipeline, где tokenizer и recommender опираются на один LLM.
- **Figure 3.** Детализирует dense tokenizer: item content сжимается в embeddings, затем через attention mask и self-supervised tasks формируются dense tokens.
- **Figure 5.** Instruction templates показывают, что STORE формулирует tokenizer и recommender tasks в одном generative interface.
- **Figure 6.** Code tree для conditional beam search; это serving-critical часть, без которой generated token sequence может не соответствовать item.

## 12. Сильные стороны

- STORE прямо атакует pipeline complexity, а не только improves tokenizer loss.
- Использование одного LLM backbone уменьшает disjoint optimization между semantic tokenization и recommendation.
- Text-token auxiliary tasks делают semantic tokens менее "чужими" для language model.
- Conditional beam search дает валидность generated identifiers.
- Метод применим не только к retrieval, но и к scoring/CTR-like setup.

## 13. Слабые стороны и ограничения

- **Не полностью end-to-end.** k-means clustering остается отдельным offline step, поэтому discrete token map все равно нужно версионировать.
- **LLM cost.** Единый LLM упрощает архитектуру концептуально, но может быть дороже, чем маленький tokenizer + recommender.
- **Content dependence.** На доменах со слабым item text преимущества могут быть меньше, что видно по разнице MIND/Yelp.
- **Version drift.** arXiv paper позднее был существенно переработан в LAMIA; для воспроизведения STORE нужно фиксировать именно v1 source.
- **Cluster count sensitivity.** k-means capacity влияет на collisions и item distinguishability.

## 14. Как реализовать и проверять

1. Зафиксировать версию paper/source: для STORE использовать arXiv v1.
1. Начать с TIGER/LC-Rec baseline на MIND или Yelp.
1. Реализовать dense tokenizer как instruction task: item text -> dense token slots.
1. Добавить token-to-text reconstruction и проверить, падает ли information loss.
1. После clustering посчитать code utilization, collision rate, head/tail distribution и item validity.
1. Обучить token-to-token recommender с тем же backbone и text-token alignment auxiliary task.
1. На inference использовать code-tree constrained beam search, а не unconstrained generation.

## 15. Связь с соседними работами

<div class="table-scroll">
<table>
<tr><th>Работа</th><th>Связь со STORE</th></tr>
<tr><td>TIGER</td><td>классический RQ-VAE tokenizer + generative recommendation baseline</td></tr>
<tr><td>CoST</td><td>улучшает tokenizer objective contrastive loss, но сохраняет multi-stage pipeline</td></tr>
<tr><td>LETTER</td><td>добавляет collaborative/diversity regularization к tokenizer</td></tr>
<tr><td>ETEGRec</td><td>связывает tokenizer и recommender через end-to-end alignment</td></tr>
<tr><td>DIGER / UniGRec</td><td>идут дальше в сторону differentiable или soft semantic IDs</td></tr>
<tr><td>LAMIA</td><td>поздняя переработка этой arXiv-линии: multi-aspect item palette вместо STORE unified LLM framing</td></tr>
</table>
</div>

## 16. Итог

STORE - важная работа про упрощение semantic tokenization pipeline. Ее главный тезис: если semantic tokens нужны LLM-рекомендателю, то их стоит учить в том же generative/instruction framework, а не получать из отдельного frozen embedder + RQ-VAE. Практически самая ценная часть - сочетание dense tokenizer, text-token alignment и conditional beam search.

Для будущих исследований STORE задает хороший вопрос: можно ли сделать semantic tokenizer не отдельным preprocessing module, а частью единой LLM-native recommendation system, сохранив при этом controlled decoding, стабильные item maps и приемлемую стоимость inference.
