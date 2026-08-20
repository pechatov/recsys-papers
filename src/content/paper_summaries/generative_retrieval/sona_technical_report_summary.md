---
title: "Sona Technical Report"
category: "generative_retrieval"
slug: "sona_technical_report_summary"
catalogId: "paper-sona_technical_report_summary"
paperUrl: "https://arxiv.org/abs/2608.11015"
---
> **Авторы:** Sona Team, Alexandr Udeneev, Aleksei Krasilnikov, Alexey Nadtochiy, Andrey Semenov, Andrey Tsyrkunov, Anna Krivonos, Anna Lipkina, Artem Matveev, Daniil Burlakov, Daniil Leshchev, Daria Tikhonovich, Denis Burshtein, Ekaterina Dmitrieva, Eugene Krofto, Grigorii Khlystov, Ilya Murzin, Kirill Golovko, Ksenia Sycheva, Leonid Dmitriev, Mariia Rozaeva, Mariia Ulianova, Mikhail Sandul, Nikolai Savushkin, Oleg Sorokin, Roman Odobesku, Semyon Panenko, Sergei Liamaev, Sergei Makeev, Vadim Shilov, Veronika Ivanova, Viktor Yanush, Vladimir Baikalov, Vladislav Dodonov, Vladislav Tytskiy.
>
> **Аффилиации:** Yandex Music.
>
> **Источник:** arXiv:2608.11015v1 от 2026-08-11. Публичные код, данные и модель не заявлены.

## Коротко: о чем статья

Sona объединяет генерацию кандидатов и ранжирование вокруг общего user encoder; технический отчёт описывает online A/B на My Vibe для умных колонок.

## Abstract

We introduce Sona, a single-model generative recommender for Yandex Music. In an online A/B test, Sona replaced the entire production cascade, comprising more than 15 candidate generators followed by pre-ranking and ranking models that consume hundreds of features, including signals from large transformer models such as Argus and target-attention scorers, while significantly improving key engagement metrics. The architecture of Sona unifies candidate generation and ranking around a shared user representation. Its encoder transforms the user's chronological sequence of logged engagement events into hidden states consumed by both the autoregressive decoder and the Ranking Module. The next-token-prediction and distillation objectives jointly update the encoder, coupling generation and ranking through the same user state. Neither Sona nor its Teacher Ranker uses hand-engineered features; both operate on logged event fields and learned item representations. In the final Sona configuration, the larger teacher supplies ranking targets during training but is absent from serving, leaving the encoder, decoder, and Ranking Module as a single deployed model. We evaluate Sona in an online A/B experiment using live traffic from My Vibe on smart speakers, one of Yandex Music's largest recommendation surfaces. Relative to the production control, Sona produced statistically significant uplifts of 4.53% in Active Users, the primary metric, 6.30% in Total Listening Time, and 11.42% in Likes. These effects were incremental to improvements retained from preceding deployments. The Active Users uplift was 2.35 times the increment previously delivered by Argus, the strongest model deployed on this surface before Sona. These results show that a single jointly trained model can replace a mature multi-stage recommendation cascade while improving recommendation quality on live traffic.

## Ограничения

Это свежий препринт: заявленные результаты и сравнения следует интерпретировать в границах раскрытых авторами данных, протоколов и baseline-ов.
