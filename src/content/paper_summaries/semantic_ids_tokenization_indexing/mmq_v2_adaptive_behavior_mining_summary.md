---
title: "MMQ-v2: Align, Denoise, and Amplify: Adaptive Behavior Mining for Semantic IDs Learning in Recommendation"
category: "semantic_ids_tokenization_indexing"
slug: "mmq_v2_adaptive_behavior_mining_summary"
catalogId: "paper-mmq_v2_adaptive_behavior_mining_summary"
paperUrl: "https://arxiv.org/abs/2510.25622"
---
> **Авторы:** Yi Xu, Moyu Zhang, Chaofan Fan, Jinxin Hu, Xiaochen Li, Yu Zhang, Xiaoyi Zeng, Jing Zhang.
>
> **Источник:** arXiv:2510.25622. Важно: в v3 статья переименована в **Taming the Long Tail: Denoising Collaborative Information for Robust Semantic ID Generation**; в этом обзоре сохранено имя `MMQ-v2`, потому что оно соответствует v1/v2 и пользовательскому списку.

## 1. Коротко: о чем статья

MMQ-v2 развивает линию multimodal/behavior-aware semantic IDs. Базовая проблема такая: content-based SIDs хорошо переносятся на long-tail и cold-start, потому что строятся из текста/изображений, но им часто не хватает collaborative signal. Поведенческие сигналы полезны для recommendation quality, но сильно skewed: у head items много надежных взаимодействий, у tail items поведенческий embedding шумный или почти пустой.

Авторы утверждают, что простое behavior-content alignment опасно. Если одинаково сильно выравнивать content embedding и collaborative embedding для всех item'ов, то шумная collaborative информация tail item'ов начинает портить устойчивое content representation. MMQ-v2 поэтому предлагает **ADC-SID**: adaptive denoising/alignment framework, который отдельно моделирует shared и modality-specific части, регулирует силу alignment по надежности поведения и динамически взвешивает behavioral tokens.

Главное практическое обещание: semantic ID должен уметь использовать collaborative signal там, где он надежен, и опираться на content там, где behavior noisy. На промышленном e-commerce датасете авторы показывают прирост в generative retrieval и discriminative ranking, а в online A/B получают +3.50% revenue / +1.15% CTR для retrieval scenario и +1.56% revenue / +3.04% CTR для ranking scenario.

## 2. Контекст: почему MMQ-v2 не просто еще один tokenizer

Обычные content-based SIDs квантуют item title/image embedding. Это дает компактный token sequence и semantic clustering, но не гарантирует, что соседство в SID-space соответствует user preference. Behavior-aware методы добавляют collaborative embeddings или user-item interaction signals, чтобы semantic IDs лучше совпадали с recommendation objective.

Проблема в distribution mismatch. Content features относительно плотные и доступны почти для всех item'ов. Collaborative features распределены по power law: head items получают качественные behavioral embeddings, tail items получают noisy estimates. Поэтому behavior-content alignment должен быть **неравномерным**. Для популярного item'а collaborative information может быть сильным teacher'ом; для редкого item'а тот же teacher может вносить шум.

<figure class="paper-figure">
  <img src="../../assets/mmq_v2/sid_paradigms.png" alt="MMQ-v2 comparison of content-based, behavior-content aligned, and shared-specific SID paradigms">
  <figcaption>Рисунок 1. Статья сравнивает три парадигмы: content-only SIDs, behavior-content aligned SIDs и shared/specific SID learning. Важный сдвиг MMQ-v2: поведенческий сигнал нужно не просто добавить, а очистить и дозировать.</figcaption>
</figure>

## 3. Метод: ADC-SID

ADC-SID состоит из двух ключевых блоков.

Первый блок - **Adaptive Behavior-Content Alignment**. Он отвечает за то, чтобы shared representation действительно ловил пересечение между behavior и content, но не заставлял tail items слепо копировать шумный collaborative signal.

Второй блок - **Dynamic Behavioral Weighting Mechanism**. Он отвечает за то, чтобы разные behavioral components не имели одинаковый вес по умолчанию. В отличие от equal-weight SID paradigm, модель учит dynamic gate и sparsely activated training strategy, чтобы усиливать информативные behavioral dimensions и подавлять шумные.

<figure class="paper-figure">
  <img src="../../assets/mmq_v2/architecture.png" alt="Overall ADC-SID architecture from MMQ-v2">
  <figcaption>Рисунок 2. Архитектура ADC-SID: shared experts извлекают behavior-content common signal, behavior/content specific experts сохраняют модально-специфичную информацию, alignment controller регулирует силу contrastive alignment, а dynamic behavioral gate выбирает вклад behavioral SIDs.</figcaption>
</figure>

### 3.1. Behavior-content mixture-of-quantization

Модель работает с двумя видами признаков: content features и behavior features. Content part обычно формируется из item text/image encoders. Behavior part формируется из user-item interactions и поэтому несет collaborative information.

Вместо раннего слияния ADC-SID использует shared-specific experts:

- **behavior-content shared experts** должны извлекать общий сигнал, который согласуется между content и behavior;
- **behavior-specific experts** сохраняют collaborative details, которые не обязаны быть видны в content;
- **content-specific experts** сохраняют семантику item'а, особенно важную для tail/cold-start.

Дальше representations квантуются в semantic IDs. Смысл не в том, чтобы получить один максимально точный reconstruction vector, а в том, чтобы получить дискретные коды, полезные для downstream retrieval/ranking и устойчивые к popularity skew.

### 3.2. Adaptive alignment strength

Ключевой риск behavior-content contrastive learning: если positive pair строится между content и noisy behavior embedding, contrastive objective начинает тянуть content space в неправильную сторону. Поэтому ADC-SID вводит alignment strength controller.

Интуитивно controller должен вести себя так:

- для item'ов с надежной behavior information alignment можно усилить;
- для long-tail/noisy item'ов alignment нужно ослабить;
- при слишком слабом alignment модель снова станет content-only и потеряет collaborative gain.

<figure class="paper-figure">
  <img src="../../assets/mmq_v2/alignment_controller.png" alt="MMQ-v2 alignment strength controller sensitivity">
  <figcaption>Рисунок 3. Alignment strength controller показывает, что сила behavior-content alignment должна меняться, а не быть глобальной константой. Это центральная защита от collaborative noise в long tail.</figcaption>
</figure>

### 3.3. Dynamic behavioral weighting

В equal-weight SID paradigm все behavioral components получают одинаковый вклад. MMQ-v2 критикует это как слишком грубое предположение: разные поведенческие сигналы имеют разную надежность, а для tail item'ов часть сигналов может быть фактически шумом.

Dynamic Behavioral Weighting Gate учит веса для behavioral SIDs. Sparsely-activated training заставляет модель не размазывать внимание по всем behavior channels, а выбирать наиболее полезные. Это похоже на MoE-логику: не каждый expert должен активно участвовать в каждом item.

<figure class="paper-figure">
  <img src="../../assets/mmq_v2/dynamic_weight_sid.png" alt="Equal-weight versus dynamic-weight behavioral SID paradigm">
  <figcaption>Рисунок 4. Dynamic-weight SID paradigm заменяет одинаковые веса на обучаемые веса behavioral components. Это особенно важно в long-tail режиме, где часть collaborative signal статистически ненадежна.</figcaption>
</figure>

## 4. Эксперименты

Статья проверяет ADC-SID на промышленном e-commerce датасете Alibaba-scale и на Amazon Beauty. Индустриальный датасет намного больше публичного: 35.15M users, 48.11M items и 75.73B interactions против 22,363 users, 12,101 items и 198,360 interactions на Beauty.

Оценка идет в двух downstream tasks:

- **generative retrieval**, где semantic IDs используются как target sequence для генеративного retrieval;
- **discriminative ranking**, где SID quality проверяется через ranking metrics вроде AUC/GAUC.

<div class="table-scroll">
<table>
<thead>
<tr><th>Setup</th><th>Ключевой результат</th><th>Интерпретация</th></tr>
</thead>
<tbody>
<tr><td>Generative retrieval, industrial</td><td>ADC-SID: R@50 0.2774, R@100 0.2927, N@50 0.1688, N@100 0.1744</td><td>Существенно выше content-only и behavior-aligned baselines; выигрыш связан не только с reconstruction.</td></tr>
<tr><td>Ablation</td><td>Удаление alignment controller или dynamic gate снижает retrieval metrics</td><td>Оба блока несут самостоятельный вклад: denoising alignment и behavioral weighting не дублируют друг друга.</td></tr>
<tr><td>Online retrieval</td><td>+3.50% revenue, +1.15% CTR</td><td>Авторский claim о production utility подтвержден A/B, но детали трафика и guardrails закрыты.</td></tr>
<tr><td>Online ranking</td><td>+1.56% revenue, +3.04% CTR</td><td>SID полезен не только для generation, но и как ranking feature.</td></tr>
</tbody>
</table>
</div>

## 5. Что доказывают ablations

Ablation table важнее top-line, потому что MMQ-v2 добавляет несколько механизмов сразу. В полном ADC-SID на industrial dataset авторы показывают R@50 0.2774 и N@100 0.1744. Без Alignment Strength Controller R@50 падает до 0.2701, без Behavior-Content Contrastive Learning до 0.2733, без Dynamic Behavioral Weighting Gate до 0.2705.

Интерпретация: adaptive alignment помогает не потому, что добавляет параметры, а потому что меняет режим использования collaborative signal. Dynamic gate отдельно помогает, потому что даже после alignment разные behavior components имеют разную ценность.

## 6. Сильные стороны

Главная сильная сторона статьи - точная постановка long-tail failure mode. Многие SID работы говорят "добавим collaborative signal", но не различают reliable behavior для head items и noisy behavior для tail. MMQ-v2 делает это центральным объектом метода.

Второй плюс - проверка на двух задачах. Если SID улучшает и generative retrieval, и discriminative ranking, меньше риск, что метод подогнан только под один decoder.

Третий плюс - online A/B. Абсолютные details закрыты, но наличие production deployment повышает доверие к тому, что проблема не является артефактом маленького public benchmark.

## 7. Ограничения и риски

Главное ограничение - воспроизводимость. Самые важные результаты получены на внутреннем industrial dataset, а public Beauty слишком мал, чтобы полностью проверить long-tail dynamics в масштабе десятков миллионов items.

Второй риск - сложность pipeline. ADC-SID требует content encoders, behavior embeddings, shared/specific experts, alignment controller, dynamic gate и downstream evaluation. Это сложнее, чем RQ-VAE или RQ-KMeans, и требует мониторинга drift/noise в behavior features.

Третий риск - causality of gains. Улучшение может идти не только от denoising идеи, но и от большего capacity, meilleure tuning или специфики Alibaba traffic. Для независимой проверки нужны сильные baselines с сопоставимым числом параметров и одинаковым downstream model.

## 8. Вывод

MMQ-v2 полезна как paper про **когда collaborative signal стоит добавлять в semantic IDs, а когда его нужно ослаблять**. В отличие от content-only tokenization, ADC-SID пытается сделать SID одновременно preference-aware и robust к long tail. Самый переносимый takeaway: behavior-content alignment не должен быть глобальным и одинаковым для всех item'ов; его нужно регулировать надежностью поведенческого сигнала.
