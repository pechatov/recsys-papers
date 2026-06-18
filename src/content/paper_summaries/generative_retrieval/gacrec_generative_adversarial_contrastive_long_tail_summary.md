---
title: "GACRec: Generative adversarial contrastive learning for improved long-tail item recommendation"
category: "generative_retrieval"
slug: "gacrec_generative_adversarial_contrastive_long_tail_summary"
catalogId: "paper-gacrec_generative_adversarial_contrastive_long_tail_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/gacrec_generative_adversarial_contrastive_long_tail_summary.html"
generatedFromHtml: true
paperUrl: "https://doi.org/10.1016/j.knosys.2024.112146"
---
Подробное саммари статьи:

> **Авторы:** Bingjun Qin, Zhenhua Huang, Xing Tian, Yunwen Chen, Wenguang Wang.
>
> **Аффилиации:** School of Artificial Intelligence, South China Normal University; School of Computer Science, South China Normal University; DataGrand Inc.; Da Er Guan Data (Chengdu) Co., Ltd.
>
> **Публикация:** Knowledge-Based Systems, vol. 300, article 112146, 2024.
>
> **Источники:** [DOI / Elsevier](https://doi.org/10.1016/j.knosys.2024.112146), [ScienceDirect article page](https://www.sciencedirect.com/science/article/pii/S0950705124007809), [J-GLOBAL metadata](https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=202402271002942531), [DBLP](https://dblp.org/rec/journals/kbs/QinHTCW24).

## 1. Коротко

GACRec решает long-tail item recommendation: у большого числа item'ов мало interaction records, поэтому их embedding'и слабы, noisy и плохо конкурируют с head item'ами. Авторы комбинируют **graph contrastive learning** и **generative adversarial zero-shot learning**, чтобы создать virtual feature representations для long-tail item'ов.

Главная идея: не просто перенести знания популярных item'ов на tail, а сгенерировать для tail item'ов виртуальные представления через shared attributes между popular и long-tail item'ами. Для генерации используется conditional Wasserstein GAN; затем generated virtual embeddings заменяют слабые long-tail embeddings во время обучения recommender'а.

## 2. Контекст

Long-tail distribution типична для рекомендаций: немного популярных item'ов собирает большую часть взаимодействий, а большинство item'ов видны редко. Transfer learning и meta-learning пытаются переносить знания с head на tail, но авторы выделяют две проблемы: transferred knowledge может плохо соответствовать конкретному tail item'у, а learned representations tail item'ов содержат много interaction noise.

Contrastive learning помогает обучать более устойчивые представления, но стандартные augmentations вроде edge dropping или masking могут вредить tail item'ам: там и так мало данных. Поэтому GACRec вводит специальные graph augmentations для sparse region и использует GAN, чтобы сгенерировать дополнительные признаки, а не только выбрасывать часть графа.

## 3. Метод и pipeline

1. **Problem formulation.** Входом является user-item interaction matrix `R` и auxiliary matrix `T` , отсортированная по числу взаимодействий item'ов. В implicit setting взаимодействия кодируются как 1, отсутствие - как 0.
1. **Head/tail split.** Item'ы делятся по частоте взаимодействий. Popular item'ы дают более надёжные представления; long-tail item'ы требуют augmentation.
1. **Graph contrastive learning.** Сначала обучаются представления popular и long-tail item'ов. Для tail авторы добавляют nodes и edges в graph structure, чтобы усилить sparse representations.
1. **Shared attributes.** Извлекаются common interaction records между popular и long-tail item'ами. В тексте ScienceDirect conclusion это описано как maximum common interaction records.
1. **Generative adversarial zero-shot learning.** Conditional Wasserstein GAN генерирует virtual embeddings на основе shared attributes. Эти embeddings должны быть близки к полезным popular-item representations, но адаптированы к tail.
1. **Training replacement.** Во время обучения long-tail item features заменяются generated virtual representations, чтобы recommender получил более устойчивые признаки tail item'ов.

## 4. Результаты и evidence

Publisher page сообщает эксперименты на трёх benchmark datasets: MovieLens100K, FilmTrust и Last.FM. Метрики: Precision, Recall и NDCG. Сравнение сделано против state-of-the-art методов для top-K recommendation и long-tail recommendation; GACRec превосходит comparative methods по указанным метрикам.

Важные evidence points из страницы статьи:

- GACRec позиционируется как zero-shot-learning-based recommendation framework для long-tail item'ов.
- Две новые augmentation-идеи работают на graph level: добавление nodes и edges повышает качество sparse representations.
- Сгенерированные GAN features используются не как отдельный reranker, а прямо в обучении модели, чтобы улучшить generalization ability.
- Авторы также проводят theoretical analysis и ablation/settings experiments, включая проверку эффективности augmentation и sensitivity к настройкам.

## 5. Ограничения

- Полный текст KBS закрыт на ScienceDirect; в открытом доступе были доступны abstract, highlights, section snippets, DBLP и J-GLOBAL metadata, поэтому точные численные таблицы в этом саммари не приводятся.
- Метод зависит от корректности head/tail разбиения и качества shared attributes; если пересечение interaction records слабое, генератору сложнее создать полезные tail features.
- Conditional WGAN добавляет нестабильность и гиперпараметры GAN-обучения поверх graph recommender'а.
- Постановка использует user/item IDs и interaction graph; content/multimodal признаки явно не являются центральной частью метода.

## 6. Связь с GR/SID

GACRec не строит semantic IDs и не выполняет autoregressive item generation. Но для generative recommendation он важен как пример **feature-level generation**: генератор создаёт item representations для областей каталога, где collaborative signal беден. В SID-based GR long-tail проблема проявляется иначе - редкие item'ы получают слабые token assignments или редко встречаются как targets. Идея GACRec может быть полезна как предтокенизационный или auxiliary этап: улучшать representations long-tail item'ов до построения semantic IDs.
