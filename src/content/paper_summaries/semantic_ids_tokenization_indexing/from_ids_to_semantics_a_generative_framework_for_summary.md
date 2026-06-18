---
title: "From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization"
category: "semantic_ids_tokenization_indexing"
slug: "from_ids_to_semantics_a_generative_framework_for_summary"
catalogId: "paper-from_ids_to_semantics_a_generative_framework_for_summary"
sourceHtml: "summaries/paper_summaries/semantic_ids_tokenization_indexing/from_ids_to_semantics_a_generative_framework_for_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1609/aaai.v40i17.38508"
---
Подробное саммари статьи:

> **Авторы:** Peiyu Hu, Wayne Lu, Jia Wang.
>
> **Аффилиации:** Xi'an Jiaotong-Liverpool University; University of Liverpool.

## 1. Коротко

GenCDR переносит semantic tokenization в cross-domain recommendation. Вместо того чтобы полагаться на shared user/item IDs между доменами, модель строит domain-adaptive semantic IDs и генерирует рекомендации авторегрессионно. Ключевой механизм - dynamic routing между universal encoder и domain-specific adapters: item получает код, который одновременно сохраняет общую семантику и доменные отличия.

## 2. Контекст

Cross-domain recommendation часто ломается, когда нет надежных overlapping IDs или когда один и тот же термин имеет разные смыслы в разных доменах. Авторы приводят пример “Apple”: в lifestyle domain это может быть еда/бренд повседневного товара, а в technology domain - электроника. Поэтому простой text encoding или общие ID не дают хорошего переноса.

## 3. Проблема

- Item ID tokenization dilemma: атомарные IDs раздувают vocabulary и плохо переносят collaborative knowledge.
- Недостаточное domain-specific modeling: универсальная модель сглаживает отличия доменов, а отдельные модели теряют transfer.
- CDR нуждается в генерации кандидатов с учетом домена, иначе prefix может вести к item из неправильного item pool.
- LLM/full fine-tuning дорог, поэтому нужен adapter/LoRA-friendly подход.

## 4. Метод и архитектура

Framework состоит из двух стадий. Domain-adaptive Tokenization строит disentangled semantic IDs через hierarchical adapter system: universal path отвечает за shared semantics, domain adapters - за специфичные признаки. Cross-Domain Autoregressive Recommendation симметрично моделирует user preferences как смесь universal и domain-specific interests. На inference используется Domain-aware Prefix-tree, который ограничивает decoding допустимыми item IDs нужного домена.

## 5. Objective и алгоритм

Routing module выбирает, сколько информации брать из universal encoder и domain adapters. Semantic IDs обучаются так, чтобы быть пригодными для autoregressive generation, а не только для восстановления текста. Recommendation objective оптимизирует предсказание следующего item'а в multi-domain sequence, а domain-aware prefix tree превращает sequence generation в контролируемый retrieval: beam search не уходит в недопустимые ветки и снижает grounding errors.

### 5.1. Пошаговая схема GenCDR

1. **Разметить домены и собрать item features.** Для каждого item известен domain label; item text/features проходят через общий encoder и domain-specific adapter.
1. **Посчитать universal representation.** Universal branch извлекает переносимую семантику, которая должна помогать там, где нет shared item IDs.
1. **Посчитать domain-specific correction.** Domain adapter добавляет локальные значения терминов, категорий и user intent. Пример из summary: “Apple” может иметь разные смыслы в lifestyle и technology domains.
1. **Смешать branches routing module.** Dynamic routing выбирает веса universal/domain-specific information для item-а; эти weights являются важной диагностикой collapse.
1. **Сгенерировать domain-adaptive semantic ID.** Disentangled representation квантуется/кодируется в SID, который сохраняет shared semantics и domain correction.
1. **Обучить cross-domain autoregressive recommender.** User history из нескольких доменов представляется как SID sequence; decoder предсказывает next item SID с учетом target domain.
1. **Ограничить inference prefix tree.** Domain-aware prefix tree оставляет в beam search только валидные item paths нужного домена, снижая domain leakage и invalid generation.
1. **Проверить transfer и leakage.** Нужны source-only/target-only/joint comparisons, valid generation rate и routing weights по domain/category.

```
for item in catalog:
    u = universal_encoder(item.features)
    d = domain_adapter[item.domain](item.features)
    alpha = routing_module(item.features, item.domain)
    z = alpha * u + (1 - alpha) * d
    sid[item] = semantic_tokenizer(z)

for sequence in multi_domain_logs:
    input_sids = [sid[i] for i in sequence.history]
    target_domain = sequence.target_domain
    loss = autoregressive_loss(model, input_sids, sid[sequence.target_item], target_domain)
    update(loss)

at inference:
    trie = domain_aware_prefix_tree(target_domain)
    candidates = constrained_beam_search(model, history_sids, trie)
```

## 6. Эксперименты и метрики

Эксперименты AAAI/arXiv версии проведены на нескольких cross-domain наборах; метрики - Recall@K и NDCG@K. GenCDR сравнивается с sequential, generative и cross-domain baselines, а также с вариантами без компонентов framework. Таблицы показывают устойчивое превосходство GenCDR по всем наборам. В efficiency analysis LoRA-based GenCDR требует существенно меньше trainable parameters, training time и peak GPU memory, чем full fine-tuning Qwen2.5-7B; также анализируется runtime memory/inference time относительно item pool size.

## 7. Что важно в рисунках и таблицах

Figure со “story” про Apple важен как интуитивное объяснение domain ambiguity. Архитектурный рисунок показывает три слоя: tokenization module, recommendation module и детальную структуру adapters/routing. t-SNE визуализация демонстрирует, что adaptive tokenization лучше разделяет доменные item embeddings. Таблица ablation по NDCG@10 показывает вклад domain-adaptive tokenizer, cross-domain autoregression и prefix-tree constraints.

## 8. Сильные стороны

- **Хорошо связывает semantic IDs с реальной проблемой CDR без shared IDs.** Identifier нужен не просто для сжатия item'а, а для переноса между доменами.
- **Adapters дают инженерно разумный компромисс.** Universal branch сохраняет transfer, domain adapters не дают сгладить специфичные значения категорий и терминов.
- **Domain-aware prefix tree делает генерацию grounded.** Beam search не должен возвращать item из неправильного домена, даже если SID prefix похож.
- **Есть efficiency analysis.** LoRA/full fine-tuning и runtime scaling по item pool size важны для практического CDR, где домены могут расти независимо.
- **Routing weights диагностируемы.** Можно увидеть, действительно ли item использует domain adapter или вся модель collapse'ится в universal path.

## 9. Ограничения

- **Сложность выше single-domain TIGER.** Нужны domain labels, adapters, routing diagnostics и отдельные prefix trees.
- **Transfer зависит от общей семантической основы.** Если домены слишком разнородны, universal branch не сможет переносить полезный signal.
- **Есть риск adapter collapse или over-specialization.** Routing может почти всегда выбирать universal branch либо, наоборот, разорвать transfer через слишком сильные domain adapters.
- **Prefix constraints могут скрывать слабую модель.** Высокий valid generation rate не гарантирует хороший ranking внутри domain item pool.
- **Воспроизводимость зависит от preprocessing.** В AAAI версии код указан как supplementary; без точных domain splits и merging правил результаты трудно повторить.

## 10. Как реализовать и проверять

- Начинать с двух доменов и явно измерять transfer: source-only, target-only, joint, GenCDR.
- Логировать routing weights по доменам и по item categories, чтобы увидеть collapse в universal или domain-specific branch.
- Проверять prefix tree constraints отдельно: valid generation rate, domain leakage, beam utilization.
- В low-resource domain оценивать не только Recall/NDCG, но и cold-start buckets.

## 11. Связь с соседними работами

GenCDR близок к Structured Term Identifiers тем, что пытается заменить голые IDs на semantic identifiers, совместимые с LLM/generative modeling. От DOS и PIT его отличает cross-domain постановка; от UniGRec и DIGER - меньший акцент на differentiable codebook collapse и больший на domain-adaptive representation.

## 12. Итог

GenCDR важен как пример того, что semantic ID не должен быть глобально одинаковым для всех контекстов. В cross-domain recommendation полезный identifier обязан содержать и shared semantics, и domain-specific correction, а decoding должен знать домен назначения.

## 13. Детальный разбор механизмов статьи

### 13.1. Domain-adaptive Tokenization

DAT является главным отличием GenCDR от обычной semantic tokenization. Universal encoder извлекает переносимую семантику, а domain-specific adapters добавляют поправку для конкретного домена. Dynamic routing нужен, потому что не все item'ы одинаково domain-specific: часть semantics общая, часть меняет смысл при переходе между доменами.

- Universal branch отвечает за shared collaborative/semantic structure.
- Domain adapters кодируют локальные значения категорий, терминов и user intent.
- Routing weights дают диагностический сигнал: модель действительно использует domain branch или collapse'ится в universal.
- Disentangled semantic IDs помогают переносить knowledge без shared item IDs.
- Tokenization module обучается в связке с recommendation module, а не как изолированный text encoder.

### 13.2. Cross-domain autoregressive recommendation

- История пользователя представляется как последовательность semantic IDs из нескольких доменов.
- Preference representation также разделяет universal и domain-specific interests.
- Autoregressive decoder генерирует next item identifier, но должен учитывать target domain.
- Domain-aware prefix tree ограничивает beam search валидными путями и снижает domain leakage.
- Такой prefix tree одновременно ускоряет decoding и улучшает grounding.

### 13.3. Эксперименты и efficiency

AAAI/arXiv версия сравнивает GenCDR с sequential, generative и cross-domain baselines. R@K и N@K используются как основные метрики, а отдельный efficiency block сравнивает LoRA-based GenCDR с Full Fine-Tuning на Qwen2.5-7B. Это важно: GenCDR заявляет не только accuracy, но и более реалистичный training cost.

- Dataset statistics включают users, items, interactions, sparsity и overlap после merging.
- Ablation по NDCG@10 показывает падение при удалении domain-adaptive tokenization.
- t-SNE visualization показывает лучшее разделение item embeddings в adaptive setting.
- LoRA analysis показывает меньше trainable parameters, training time и GPU memory.
- Runtime comparison по item pool size сравнивает TriCDR, TIGER и GenCDR.

### 13.4. Failure modes

- Domain leakage: decoder генерирует item из неправильного домена при слабом prefix constraint.
- Adapter collapse: routing почти всегда выбирает universal branch, и domain-specific modeling исчезает.
- Over-specialization: domain adapters перестают переносить shared semantics и ухудшают low-resource domains.
- Ambiguous terms: одно слово или бренд имеет разные значения, и без routing semantic ID становится misleading.
- Overlap bias: если evaluation domains имеют слишком большой overlap, модель может выглядеть лучше, чем в реальном no-shared-ID сценарии.

## 14. Первичные источники

- Официальная AAAI страница/DOI: [AAAI 40(17), 14874-14882](https://ojs.aaai.org/index.php/AAAI/article/view/38508) .
- arXiv preprint/source: [2511.08006](https://arxiv.org/abs/2511.08006) .
