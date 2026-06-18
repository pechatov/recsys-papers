---
title: "GenRec: Large Language Model for Generative Recommendation"
category: "generative_retrieval"
slug: "genrec_large_language_model_for_generative_recommendation_summary"
catalogId: "paper-genrec_large_language_model_for_generative_recommendation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/genrec_large_language_model_for_generative_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1007/978-3-031-56063-7_42"
---
Подробное саммари статьи:

> **Авторы:** Jianchao Ji, Zelong Li, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Juntao Tan, Yongfeng Zhang.
>
> **Аффилиации:** Rutgers University.
>
> **Публикация:** ECIR 2024, Advances in Information Retrieval, LNCS, pages 494-502. DOI: 10.1007/978-3-031-56063-7_42; также доступен arXiv:2307.00457.

## 1. Коротко

GenRec - ранняя и довольно минималистичная попытка напрямую использовать LLaMA как генеративный sequential recommender. Вместо того чтобы считать score для каждого кандидата или строить ANN-индекс, модель получает историю пользователя в виде текстовых названий item'ов и генерирует название следующего item'а как текстовый output.

Главный вклад не в сложной архитектуре, а в постановке: рекомендация формулируется как instruction-following задача. Для каждого пользователя строится prompt из трех частей: инструкция, список прошлых взаимодействий и целевой item. Затем LLaMA дообучается через LoRA, чтобы по истории порождать следующий объект. Это ближе к text-to-text recommendation, чем к TIGER-like semantic-ID generative retrieval.

## 2. Контекст

До GenRec LLM-based recommendation часто использовала либо item ID как специальные токены, либо multi-task prompt learning вроде P5. Авторы критикуют такой подход за слабое использование текстовой семантики item'ов: если модель видит только внутренние ID, она не может использовать знания pretrained LLM о названиях фильмов, товаров и категорий.

GenRec ставит вопрос проще: если LLM уже умеет работать с текстом, можно ли дать ей историю пользователя как последовательность названий и попросить сгенерировать следующий title? Для GR/SID литературы это важная контрольная точка: не всякая generative recommendation требует дискретного semantic ID; иногда identifier - это сам естественно-языковой item name.

## 3. Метод

Pipeline состоит из двух частей. Сначала interaction history превращается в instruction-style training sequence. На примере MovieLens prompt выглядит как вопрос о вероятном следующем фильме, input содержит несколько предыдущих фильмов пользователя, а output - последний фильм в последовательности.

Затем авторы дообучают LLaMA через LLaMA-LoRA. LoRA нужна, потому что даже LLaMA-7B дорог для полного fine-tuning: в статье отмечено, что LoRA позволяет обучаться на одной 24GB GPU, хотя для ускорения экспериментов использовались несколько GPU. Максимальная длина input - 256 токенов, batch size - 128, optimizer - AdamW, learning rate - 3e-4, 5 epochs.

Важно: модель не строит отдельный constrained decoder и не гарантирует, что сгенерированная строка будет точно соответствовать item из каталога. Это делает подход концептуально простым, но практически более хрупким, чем генерация коротких SID с trie lookup.

## 4. Эксперименты/результаты

Эксперименты проведены на MovieLens 25M и Amazon Toys. Разбиение sequential: самый последний item идет в test, предпоследний - в validation, остальные - в train. Метрики: HR@5/10 и NDCG@5/10.

<div class="table-scroll">
<table>
<thead><tr><th>Датасет</th><th>Ключевой результат</th><th>Интерпретация</th></tr></thead>
<tbody>
<tr><td>MovieLens 25M</td><td>GenRec лучше P5: HR@5 0.1034 vs 0.0688; NDCG@10 0.0837 vs 0.0577.</td><td>На большом и богатом interaction dataset LLaMA-LoRA лучше использует текстовые названия и историю.</td></tr>
<tr><td>Amazon Toys</td><td>P5 лучше GenRec: HR@10 0.0411 vs 0.0251; NDCG@10 0.0201 vs 0.0157.</td><td>Результат не универсален; на более разреженном товарном датасете простой title-generation проигрывает P5.</td></tr>
</tbody>
</table>
</div>

Baseline в основной таблице фактически один - P5, поэтому утверждения о superiority нужно читать осторожно. Работа показывает feasibility LLM-to-title generation, но не закрывает сравнение с сильными sequential baselines, dense retrievers или semantic-ID GR.

## 5. Ограничения

- **Catalog grounding не решен.** Если модель генерирует title с вариацией, опечаткой или неоднозначным названием, нужен внешний matching слой.
- **Слабая экспериментальная сетка.** Основное сравнение с P5 не показывает, как метод конкурирует с SASRec/BERT4Rec/TIGER-like системами при одинаковом candidate universe.
- **Разный результат на двух датасетах.** GenRec выигрывает на MovieLens 25M, но проигрывает на Amazon Toys, что указывает на чувствительность к домену и плотности взаимодействий.
- **Нет semantic-ID механики.** Подход не решает identifier collision, constrained decoding и эффективный large-catalog retrieval так, как это делают SID-based методы.

## 6. Как читать для GR/SID

GenRec полезен как baseline идеи “LLM напрямую генерирует item text”. Для построения SID-систем он скорее показывает границу применимости text identifiers: естественный язык удобен для cold-start и интерпретации, но плохо подходит как надежный production identifier без lookup, нормализации и constrained decoding.

Читать рядом с GPT4Rec и TIGER: GPT4Rec генерирует поисковые запросы и делегирует grounding BM25, GenRec генерирует item title напрямую, а TIGER генерирует компактный semantic ID. Это три разные точки в design space генеративной рекомендации.
