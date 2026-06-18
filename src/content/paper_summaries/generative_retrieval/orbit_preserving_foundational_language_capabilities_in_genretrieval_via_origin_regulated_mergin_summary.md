---
title: "ORBIT: Preserving Foundational Language Capabilities in GenRetrieval via Origin-Regulated Merging"
category: "generative_retrieval"
slug: "orbit_preserving_foundational_language_capabilities_in_genretrieval_via_origin_regulated_mergin_summary"
catalogId: "paper-orbit_preserving_foundational_language_capabilities_in_genretrieval_via_origin_regulated_mergin_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/orbit_preserving_foundational_language_capabilities_in_genretrieval_via_origin_regulated_mergin_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2605.12419"
---
> **Авторы:** Neha Verma, Nikhil Mehta, Shao-Chuan Wang, Naijing Zhang, Alicia Tsai, Li Wei, Lukasz Heldt, Lichan Hong, Ed Chi, Xinyang Yi.
>
> **Аффилиации:** Johns Hopkins University; Google DeepMind; Google.

## 1. Коротко: о чем статья

ORBIT изучает проблему catastrophic forgetting при fine-tuning LLM под generative retrieval. Если LLM дообучать на GenRetrieval task, она начинает лучше генерировать item/SID sequences, но быстро теряет general language/reasoning capabilities. Это важно для LLM-based recommenders, где одна и та же модель может использоваться не только для retrieval, но и для language understanding, explanations, query rewriting или instruction following.

Метод ORBIT прост: во время fine-tuning отслеживается distance между текущими weights и исходными pretrained weights. Когда distance превышает threshold $\epsilon$, текущие weights усредняются с origin weights. Это **origin-regulated back-merging** ограничивает drift и уменьшает forgetting.

<figure class="paper-figure">
  <img src="../../assets/orbit/overview.png" alt="ORBIT origin-regulated merging overview">
  <figcaption>Рисунок 1. ORBIT отслеживает удаление fine-tuned модели от исходных weights и запускает back-merging, когда drift становится слишком большим. В отличие от фиксированного schedule, intervention зависит от фактического расстояния.</figcaption>
</figure>

## 2. Контекст: почему GenRetrieval ломает LLM abilities

GenRetrieval превращает sequential recommendation в generation problem: user history становится prompt/context, следующий item представлен sequence IDs, и модель учится генерировать completion. Для LLM это узкая специализированная задача с большим числом искусственных tokens и паттернов.

Fine-tuning на такой задаче может быстро переориентировать параметры модели. Авторский эксперимент показывает падение text benchmark performance после GenRetrieval fine-tuning. Это не просто "немного забыли знания": если recommender должен оставаться универсальным LLM-компонентом, degradation может быть product blocker.

<figure class="paper-figure">
  <img src="../../assets/orbit/genretrieval_setup.png" alt="ORBIT LLM-based GenRetrieval setup with text and SID vocabularies">
  <figcaption>Рисунок 2. GenRetrieval setup использует отдельные SID tokens вместе с text tokens. Именно смешение recommendation-specific tokens и general language vocabulary делает forgetting важной проблемой.</figcaption>
</figure>

## 3. Метод: алгоритм ORBIT

Пусть $\theta_{init}$ - исходные LLM weights, $\theta_t$ - текущие weights на шаге fine-tuning, $d(\theta_t, \theta_{init})$ - выбранная distance function. ORBIT задает threshold $\epsilon$.

На каждом шаге optimizer предлагает обновление $\theta^*_{t+1}$. Если расстояние до origin превышает threshold, выполняется back-merging:

$$
\theta^*_{t+1} \leftarrow \frac{\theta^*_{t+1} + \theta_{init}}{2}.
$$

Если после одного усреднения distance все еще слишком большая, операция повторяется в цикле. После этого обновленные weights становятся $\theta_{t+1}$.

Ключевое отличие от weight averaging baselines: merge schedule не фиксирован. Модель вмешивается только когда drift реально превышает допустимую границу.

## 4. Почему distance to origin

Авторы мотивируют ORBIT наблюдением: text performance коррелирует с расстоянием от initial parameters. Это не строгий закон для всех моделей, но полезный engineering signal. Если fine-tuning уводит weights слишком далеко, вероятность забывания базовых capabilities растет.

Distance может быть разной: L2, sign dissimilarity и другие варианты. Статья анализирует sign dissimilarity как сигнал, связанный с text performance.

## 5. Эксперименты

Fine-tuning проводится на GenRetrieval tasks с Amazon Review datasets вроде Beauty, Sports and Outdoors, Toys and Games. Base model - instruction-tuned Gemma3-1B. Retrieval quality оценивается Recall/NDCG, language quality - набором text/reasoning benchmarks. Для выбора balanced model авторы используют Pareto view и Distance To Ideal Point (DTIP).

Baselines включают no-intervention fine-tuning, L2 weight decay, continual-learning/regularization methods и related weight-averaging methods. По авторскому claim, ORBIT лучше сохраняет text performance при сопоставимой retrieval quality.

Важно читать results как Pareto-задачу, а не как single-metric leaderboard. Если baseline дает чуть лучший Recall, но резко теряет MMLU/BBH/GSM-like text benchmarks, он может быть хуже для платформы, где recommender LLM должна также понимать запросы, объяснять рекомендации или работать в диалоге. ORBIT явно вводит этот критерий: хорошая GenRetrieval модель должна лежать ближе к Pareto frontier по двум осям.

## 6. Как проверять метод на своей модели

Минимальная проверка ORBIT должна включать четыре набора метрик.

1. **Retrieval metrics:** Recall@K, NDCG@K, valid generated SID rate, duplicate/invalid rate и latency beam search.
1. **Language retention:** фиксированный набор text/reasoning/instruction benchmarks до fine-tuning, после обычного fine-tuning и после ORBIT.
1. **Drift diagnostics:** distance-to-origin по layers, sign dissimilarity, доля параметров с измененным знаком, cosine между update direction и origin direction.
1. **Trade-off curves:** несколько значений $\epsilon$, чтобы увидеть, где specialization начинает вредить language performance.

Практически ORBIT особенно полезен, если recommender обучается через LoRA/full fine-tuning одного LLM backbone, а не через отдельный маленький decoder. Если retrieval model полностью отдельная и никогда не используется как language model, сохранение foundation capabilities может быть менее важным.

## 7. Что доказывает статья

ORBIT показывает, что для LLM-based GenRetrieval нельзя смотреть только на Recall@10. Если модель после fine-tuning перестает быть хорошей языковой моделью, это может быть неприемлемо для multi-purpose recommender platform.

Методологически статья полезна тем, что вводит двухосевую оценку: recommendation performance и language performance. Это сильнее, чем обычный recsys-only benchmark.

## 8. Сильные стороны

- Простая реализация: ORBIT не требует replay data или сложной Fisher-information оценки.
- Intervention schedule зависит от measured drift, а не от произвольной частоты merge.
- Paper явно оценивает language capabilities alongside retrieval metrics.
- Хорошо подходит для сценариев, где LLM должна оставаться универсальной после recommendation fine-tuning.

## 9. Ограничения и вопросы

Threshold $\epsilon$ становится критичным hyperparameter. Слишком маленький threshold мешает specialization, слишком большой не предотвращает forgetting.

Метод сохраняет близость к origin, но не объясняет, какие именно capabilities нужно сохранять. Text benchmark suite может не совпадать с реальными product tasks.

ORBIT не решает проблемы semantic ID tokenizer-а, invalid generation, latency и serving cost. Это regularization/fine-tuning method, а не полный GenRetrieval system.

Эксперименты на Gemma3-1B не гарантируют перенос на большие LLMs, MoE модели или heavily instruction-tuned production backbones.

Еще один риск - layer heterogeneity. Разные слои могут отвечать за разные способности: нижние слои за lexical/semantic representation, верхние за task adaptation. Глобальный distance threshold может быть слишком грубым. Для production полезно проверить layer-wise ORBIT или разные thresholds для attention/MLP/embedding blocks.

## 10. Вывод

ORBIT стоит читать как предупреждение: fine-tuning LLM для GenRetrieval может разрушить foundation capabilities быстрее, чем это видно по recommendation metrics. Главный takeaway: для LLM-based recommenders нужно вести dual evaluation - retrieval quality и retained language ability - и контролировать weight drift во время обучения.
