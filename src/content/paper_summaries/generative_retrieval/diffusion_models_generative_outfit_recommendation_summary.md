---
title: "Diffusion Models for Generative Outfit Recommendation"
category: "generative_retrieval"
slug: "diffusion_models_generative_outfit_recommendation_summary"
catalogId: "paper-diffusion_models_generative_outfit_recommendation_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/diffusion_models_generative_outfit_recommendation_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1145/3626772.3657719"
---
Подробное саммари статьи:

> **Авторы:** Yiyan Xu, Wenjie Wang, Fuli Feng, Yunshan Ma, Jizhi Zhang, Xiangnan He.
>
> **Аффилиации:** University of Science and Technology of China; National University of Singapore.
>
> **Публикация:** SIGIR 2024, Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1350-1359.
>
> **Источники:** [ACM DOI](https://doi.org/10.1145/3626772.3657719), [arXiv:2402.17279](https://arxiv.org/abs/2402.17279), [DBLP](https://dblp.org/rec/conf/sigir/XuWFMZ024), [официальный GitHub](https://github.com/YiyanXu/DiFashion).

## 1. Коротко

Статья вводит задачу **Generative Outfit Recommendation (GOR)**: вместо выбора готового outfit'а или сборки из существующих fashion items система должна **сгенерировать новые изображения предметов одежды** и составить из них персонализированный, совместимый outfit.

Предложенная модель **DiFashion** адаптирует diffusion models, чтобы параллельно генерировать несколько fashion images. Три цели GOR: high fidelity, compatibility внутри outfit'а и personalization под пользователя. Для этого DiFashion использует category condition, mutual condition через encoder совместимости и history condition из пользовательских прошлых взаимодействий; на inference применяется classifier-free guidance.

## 2. Контекст

Outfit recommendation обычно развивался в два этапа: pre-defined outfit recommendation и personalized outfit composition. Первый выбирает готовый комплект, второй собирает комплект из существующих items. Оба ограничены каталогом: если нужного сочетания нет среди текущих products, recommender не может предложить truly new outfit.

Diffusion models дают другой режим: recommendation как генерация новых visual items. Это сильнее обычного retrieval, но и сложнее: нужно одновременно соблюдать визуальное качество, категорийные constraints, совместимость всех частей комплекта и пользовательские вкусы.

## 3. Метод и pipeline

1. **GOR formulation.** На входе user information, history и designated categories. На выходе набор fashion images, образующих совместимый outfit.
1. **Diffusion backbone.** DiFashion основан на Stable Diffusion style denoising: forward process добавляет Gaussian noise к outfit images, reverse process восстанавливает изображения через U-Net.
1. **Category condition.** Категория предмета направляет fidelity и соответствие нужному типу одежды.
1. **Mutual encoder.** Для каждого item'а модель смотрит на incomplete outfit из остальных noisy items и кодирует compatibility information. Это условие помогает генерировать предмет, согласованный с остальными.
1. **History encoder.** Пользовательская история взаимодействий кодируется как personalization signal.
1. **Classifier-free guidance.** Условия иногда маскируются на training, а на inference guidance scales усиливают alignment с category, mutual и history conditions.
1. **Tasks.** Проверяются Personalized Fill-In-The-Blank (PFITB), где нужно сгенерировать missing item, и полный GOR, где генерируется целый outfit.

## 4. Результаты и evidence

Два датасета:

<div class="table-scroll">
<table>
<thead><tr><th>Dataset</th><th>Users</th><th>Outfits</th><th>Items</th><th>Interactions</th></tr></thead>
<tbody>
<tr><td>iFashion</td><td>12,806</td><td>19,882</td><td>344,186</td><td>107,396</td></tr>
<tr><td>Polyvore-U</td><td>517</td><td>33,906</td><td>119,202</td><td>33,908</td></tr>
</tbody>
</table>
</div>

Generative baselines: OutfitGAN, SD-v1.5, SD-v2, SD-naive и ControlNet. Retrieval-based baselines: Random, HFN, HFGN, BGN и BGN-Trans. Метрики разделены на image generation quality (FID, IS, CLIP Score), fashion metrics (CIS, LPIPS, compatibility, personalization) и retrieval grounding accuracy.

В generative comparison DiFashion получает лучший или близкий к лучшему результат на большинстве метрик. Например, на iFashion PFITB FID снижается до 34.06 против 42.47 у fine-tuned SD-v1.5, IS растёт до 29.99, compatibility достигает 0.58, personalization - 55.86. На iFashion GOR FID равен 20.21 против 28.31 у SD-v1.5, personalization - 55.54.

В сравнении с retrieval-based методами на PFITB DiFashion конкурентоспособен даже в zero-shot retrieval setting: на iFashion Retrieval = 0.76, Comp. = 0.78, Per. = 64.93; на Polyvore-U Retrieval ниже HFGN, но personalization выше. Human evaluation на iFashion показывает, что DiFashion чаще предпочитают SD-v1.5/SD-v2 по fidelity, compatibility и personalization.

## 5. Ограничения

- Training дорогой: авторы указывают, что DiFashion обычно требует около одного A100 GPU day.
- Для iFashion оставлены outfits длины 4 ради вычислительных ресурсов; более длинные outfits заявлены как поддерживаемые, но основной эксперимент ограничен.
- Оценка generated fashion images опирается на proxy metrics и trained evaluators; они не полностью заменяют реальные покупки или пользовательские A/B tests.
- Модель генерирует изображения, а не обязательно доступные к покупке SKU; grounding generated image в существующий catalog остаётся отдельной задачей.

## 6. Связь с GR/SID

DiFashion - это generative recommendation в буквальном смысле: output не ID, а новые изображения items. С SID-based GR связь опосредованная, но важная: обе линии пытаются выйти за пределы retrieve-from-fixed-catalog. TIGER-подобные модели генерируют discrete semantic item identifiers, а DiFashion генерирует visual item content. Для будущих систем возможен гибрид: diffusion model создаёт candidate visual concepts, затем SID/GR слой связывает их с реальными SKU или catalog regions.
