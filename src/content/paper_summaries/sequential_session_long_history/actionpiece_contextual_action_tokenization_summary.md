---
title: "ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation"
category: "sequential_session_long_history"
slug: "actionpiece_contextual_action_tokenization_summary"
catalogId: "paper-actionpiece_contextual_action_tokenization_summary"
sourceHtml: "summaries/paper_summaries/sequential_session_long_history/actionpiece_contextual_action_tokenization_summary.html"
generatedFromHtml: true
---
Подробное саммари статьи:

> **Авторы:** Yupeng Hou, Jianmo Ni, Zhankui He, Noveen Sachdeva, Wang-Cheng Kang, Ed H. Chi, Julian McAuley, Derek Zhiyuan Cheng.
>
> **Аффилиации:** University of California San Diego; Google DeepMind.

## 1. Коротко: о чем статья

ActionPiece переносит идею subword tokenization из NLP в generative recommendation, но применяет ее не к item IDs, а к action sequences. Главная претензия к существующим GR tokenizer'ам: они токенизируют один и тот же action одинаково во всех контекстах. В рекомендациях это плохо, потому что один товар или действие может иметь разный смысл в разных user sequences.

ActionPiece делает context-aware tokenization: action представляется как set item features, а vocabulary строится через merge feature patterns по co-occurrence внутри action set и между соседними action sets.

## 2. Пример токенизации

<img src="../../assets/actionpiece/tokenization_case.png" alt="ActionPiece tokenization process example">

В отличие от fixed SID подходов, ActionPiece может создать токены, которые покрывают не только части одного item, но и устойчивые паттерны между соседними действиями. Поэтому одинаковый action может получить разное разбиение в зависимости от surrounding context.

## 3. Как строится vocabulary

Каждый action начинается как unordered set item features. Затем алгоритм считает co-occurrence token pairs:

- внутри одного feature set;
- между соседними action sets;
- с весами, которые учитывают близость и структуру последовательности.

<img src="../../assets/actionpiece/cooccurrence_weights.png" alt="ActionPiece co-occurrence weighting during vocabulary construction">

Наиболее частые и полезные pairs сливаются в новые tokens, как в BPE/SentencePiece, но с учетом структуры recommendation sequence.

## 4. Set permutation regularization

Поскольку item features внутри action не упорядочены, один и тот же action set можно сегментировать несколькими способами без изменения semantics. ActionPiece использует set permutation regularization: модель видит несколько tokenization variants для той же последовательности. Это работает как data augmentation и снижает зависимость от произвольного порядка features.

## 5. Пошаговый алгоритм ActionPiece

1. **Представить action как set features.** Каждый user action раскладывается не в один item ID, а в unordered set признаков item'а.
1. **Инициализировать vocabulary атомарными признаками.** На старте каждый feature является отдельным token, как символы в BPE.
1. **Посчитать co-occurrence внутри action.** Частые пары feature tokens внутри одного action set получают вес как кандидаты для merge.
1. **Посчитать co-occurrence между соседними actions.** Пары tokens из adjacent action sets тоже учитываются, чтобы vocabulary захватывал transition patterns, а не только item semantics.
1. **Выбрать лучший merge.** Самая полезная weighted pair объединяется в новый ActionPiece token; vocabulary пополняется этим pattern token.
1. **Повторять до budget vocabulary.** Алгоритм итеративно строит tokens, которые могут покрывать feature subsets и cross-action patterns.
1. **Токенизировать sequence context-aware способом.** Один и тот же item/action может получить разное разбиение в зависимости от соседних actions.
1. **Применить set permutation regularization.** Для одного unordered action set генерируются несколько valid tokenization variants, чтобы recommender не зависел от произвольного порядка features.
1. **Обучить generative recommender.** Модель получает ActionPiece-tokenized sequences и предсказывает следующие action tokens/items.

```
vocab = atomic_item_features
while len(vocab) < vocab_budget:
    scores = {}
    for user_sequence in logs:
        for action_set in user_sequence:
            scores += intra_action_pair_counts(action_set)
        for adjacent_actions in user_sequence:
            scores += inter_action_pair_counts(adjacent_actions, distance_weights)
    best_pair = argmax(scores)
    vocab.add(merge(best_pair))
    retokenize_sequences(vocab)

for training sequence:
    variants = set_permutation_tokenizations(sequence, vocab)
    train_generative_recommender(variants)
```

## 6. Эксперименты

Авторы сравнивают ActionPiece с sequential recommendation и generative recommendation baselines на Amazon Reviews datasets. ActionPiece стабильно обгоняет baseline'ы и улучшает NDCG@10 на 6.00%-12.82% относительно лучшего baseline. Отдельный вывод: feature-based methods обычно лучше item ID-only, а generative models с sub-item semantics лучше используют item features.

В абляциях авторы показывают, что выигрыш не объясняется только большим vocabulary size. Контекстные merges и set permutation regularization вносят отдельный вклад.

## 7. Сильные стороны

- **Сдвигает фокус с item tokenization на action-sequence tokenization.** Метод кодирует поведенческий контекст, а не только объект каталога.
- **Context-aware merges происходят до обучения recommender.** Tokenizer уже содержит устойчивые intra-action и inter-action patterns.
- **Учитывает unordered природу feature sets.** Set permutation regularization снижает зависимость от искусственного порядка признаков.
- **Хорош для доменов, где действие важнее item semantics.** Один и тот же item может означать разные intents в разных histories, и ActionPiece допускает разную сегментацию.
- **Ablations отделяют vocabulary size от механизма.** Выигрыш объясняется contextual merges и regularization, а не просто большим словарем.

## 8. Ограничения

- Алгоритм vocabulary construction сложнее обычного RQ-VAE tokenizer.
- Нужно иметь качественные item features; raw item IDs сами по себе не раскрывают потенциал метода.
- Context-aware tokens могут быть менее стабильными при сильном изменении behavior distribution.
- Production decoding и item grounding требуют аккуратной реализации, потому что output tokens больше не соответствуют одному item напрямую.
- Vocabulary refresh может быть дорогим: изменение frequent action patterns меняет merges и token meanings.

## 9. Связь с Semantic ID работами

TIGER/CoST/LETTER обычно спрашивают: как токенизировать item? ActionPiece спрашивает: как токенизировать действие внутри последовательности? Это делает работу особенно полезной для sequential recommendation, где intent зависит от соседних действий.

## 10. Вывод

ActionPiece - сильная идея для generative sequential recommendation: токенизировать не item изолированно, а action в контексте. Если semantic IDs дают компактный язык объектов, ActionPiece строит язык поведенческих паттернов. Это особенно важно для длинных историй, где один и тот же item может означать разные intents.
