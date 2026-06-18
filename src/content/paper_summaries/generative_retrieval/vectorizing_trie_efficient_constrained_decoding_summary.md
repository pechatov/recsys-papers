---
title: "Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators"
category: "generative_retrieval"
slug: "vectorizing_trie_efficient_constrained_decoding_summary"
catalogId: "paper-vectorizing_trie_efficient_constrained_decoding_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/vectorizing_trie_efficient_constrained_decoding_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2602.22647"
---
> YouTube / Google DeepMind / Yale University — arXiv:2602.22647 — 2026 —
>
> Production

## 1. Коротко: о чём статья

Статья решает фундаментальную инженерную проблему, которая блокировала production-деплой generative retrieval (GR) при высоком трафике: стандартный trie-based constrained decoding **не векторизуется на GPU/TPU**.

В системах типа TIGER или DSI модель генерирует токен за токеном, и на каждом шаге нужно ограничить множество допустимых следующих токенов теми, которые соответствуют реальным item SID-ам. Классический способ — обход дерева trie. Проблема: обход дерева — это последовательная операция с ветвлением, и GPU/TPU не умеют эффективно её параллелизовать по элементам батча.

Ключевой вклад статьи: **представить всё дерево trie как разреженную матрицу** допустимых переходов, после чего constrained decoding превращается в одну батчевую матричную операцию на весь batch одновременно. Результат — ускорение в 10–100× при идентичном качестве beam search, что делает GR production-feasible при миллионах items и высоком QPS.

## 2. Контекст: зачем нужен constrained decoding

В generative retrieval — будь то TIGER, DSI или GENRE — модель генерирует item identifier (SID) авторегрессионно, токен за токеном:

$$
p(\text{SID} \mid u) = \prod_{t=1}^{L} p(s_t \mid s_{<t}, u)
$$

где $u$ — пользовательский контекст, $s_1, \ldots, s_L$ — токены SID длины $L$.

Без каких-либо ограничений языковая модель может сгенерировать произвольную последовательность токенов, которая не соответствует ни одному реальному item в каталоге — так называемая **галлюцинация SID**. Это катастрофично для рекомендательной системы: такой item просто не существует.

Trie (prefix tree) хранит все валидные SID-ы в виде дерева, где каждый путь от корня к листу — один корректный идентификатор. При декодировании на шаге $t$:

1. Найти узел в trie, соответствующий уже сгенерированному префиксу $s_{<t}$.
1. Получить множество допустимых дочерних токенов для этого узла.
1. Обнулить логиты всех остальных токенов (маскирование).
1. Применить softmax и выбрать top-K для beam search.

Таким образом, beam search с constrained decoding **гарантирует**, что все сгенерированные последовательности являются валидными SID-ами из каталога. Это необходимое условие корректной работы GR.

## 3. Проблема с обычным trie

Стандартный Python/C++ trie при обслуживании запросов имеет следующую вычислительную структуру:

```
for каждый элемент в batch:           # B итераций
    for каждый beam:                   # W итераций
        for каждый шаг декодирования:  # L итераций
            node = trie.lookup(prefix)
            allowed = node.children_tokens()
            mask[...] = allowed
```

Общее число последовательных lookup-операций: $O(B \times W \times L)$, где:

- $B$ — размер батча (обычно 32–512 в production)
- $W$ — ширина beam search (обычно 5–50)
- $L$ — длина SID (обычно 4–8 токенов)

На GPU/TPU это создаёт несколько принципиальных проблем:

- **Warp divergence:** разные элементы батча находятся в разных узлах trie → разные ветви выполнения → GPU threads не могут работать синхронно и эффективно.
- **Нерегулярный доступ к памяти:** обход дерева — это pointer-chasing, враждебный cache hierarchy и memory coalescing.
- **Линейное масштабирование с batch size:** увеличение $B$ прямо пропорционально увеличивает latency constrained decoding, тогда как forward pass LLM масштабируется сублинейно.
- **CPU–GPU синхронизация:** в типичных реализациях trie живёт на CPU, что требует копирования масок между CPU и GPU на каждом из $L$ шагов декодирования.

В production-измерениях авторы показывают, что constrained decoding занимал **30–70% от суммарной latency** GR serving, причём эта доля росла с увеличением каталога и batch size.

## 4. Метод: Vectorized Trie как sparse matrix

Ключевая идея: представить всё дерево trie как **матрицу допустимых переходов**:

$$
M \in \{0, 1\}^{|N| \times |V|}
$$

где $|N|$ — число узлов trie, $|V|$ — размер словаря. Элемент $M[i, v] = 1$ тогда и только тогда, когда из узла $i$ есть переход по токену $v$ (то есть токен $v$ является допустимым следующим токеном из узла $i$).

Тогда constrained decoding на шаге $t$ для всего батча выполняется в два шага:

1. **Lookup узлов:** для каждого элемента батча и каждого beam определить индекс текущего узла через *perfect hashing* по текущему префиксу. Результат — матрица индексов $\mathbf{n} \in \mathbb{Z}^{B \times W}$.
1. **Маскирование логитов:** выбрать строки $M[\mathbf{n}]$ через gather-операцию и применить element-wise маску к тензору логитов $\text{logits} \in \mathbb{R}^{B \times W \times |V|}$:

$$
\text{logits\_masked}[b, w, :] = \text{logits}[b, w, :] \cdot M[\mathbf{n}[b, w], :]
$$

Оба шага — gather и element-wise multiply — **полностью векторизованы** и выполняются как единая тензорная операция на весь батч. Нет ни циклов по batch, ни ветвлений по структуре дерева.

### Perfect hashing для O(1) prefix lookup

Самый нетривиальный момент — шаг 1: как за $O(1)$ сопоставить текущий префикс индексу узла в матрице $M$?

Авторы используют **perfect hashing** по последовательности токенов префикса. При построении trie каждому узлу присваивается хеш-ключ, и строится perfect hash table (без коллизий, поскольку набор ключей известен статически). При декодировании:

- Хеш текущего префикса вычисляется инкрементально (rolling hash): $h_t = f(h_{t-1}, s_t)$.
- Lookup в hash table даёт индекс строки в $M$ за $O(1)$ без обхода дерева.
- Rolling hash полностью параллелизуется по элементам батча — это арифметическая операция, не pointer-chasing.

## 5. Детали реализации

### Sparse representation

Матрица $M$ крайне разреженная: каждый узел trie обычно имеет лишь несколько допустимых дочерних токенов при размере словаря $|V| \sim 30\,000$. Плотность $M$ порядка $O(|\text{trie edges}|) / (|N| \times |V|)$, что обычно составляет менее 0.1%.

Реализовано в формате CSR (Compressed Sparse Row) для хранения. При lookup конкретной строки под batched gather производится конвертация нужных строк в dense маску — это и есть единственный compute-интенсивный шаг, но он полностью векторизован.

### Фреймворки

- **JAX/XLA для TPU:** операции gather + multiply выражены через `jax.lax.gather` и `jnp.multiply` , JIT-компилируются XLA в эффективный TPU kernel. Perfect hash table реализована как статический массив на ускорителе.
- **CUDA для GPU:** custom CUDA kernel для batched gather + mask, с учётом memory coalescing и warp-level parallelism.

### Memory footprint

Trie как sparse matrix занимает $O(|\text{trie edges}|)$ памяти. Для каталога из 1 млн видео с SID длиной 8 токенов и словарём из 256 токенов (RQ-VAE кодирование) типичный размер trie — порядка нескольких миллионов рёбер, что составляет десятки MB в sparse представлении — вполне помещается в HBM современных TPU/GPU.

## 6. Результаты

<div class="table-scroll">
<table>
<thead>
<tr>
<th>Метрика</th>
<th>Sequential trie (baseline)</th>
<th>Vectorized trie</th>
<th>Изменение</th>
</tr>
</thead>
<tbody>
<tr>
<td>Latency constrained decode (batch 64)</td>
<td>100% (baseline)</td>
<td>1–10% от baseline</td>
<td>10–100× быстрее</td>
</tr>
<tr>
<td>Recall@K (beam search)</td>
<td>Baseline</td>
<td>Идентично</td>
<td>0% (эквивалентно)</td>
</tr>
<tr>
<td>Доля latency от суммарного GR serving</td>
<td>30–70%</td>
<td>&lt;5%</td>
<td>Устранён как bottleneck</td>
</tr>
<tr>
<td>Масштаб деплоя</td>
<td>—</td>
<td>YouTube (млн видео, высокий QPS)</td>
<td>—</td>
</tr>
</tbody>
</table>
</div>

Качество beam search при использовании vectorized trie **математически идентично** sequential trie — это не приближение, а точно та же маскировочная операция, реализованная через матричный умножение вместо обхода дерева.

**Latency breakdown.** Авторы приводят детальный профиль serving pipeline. До vectorization constrained decoding был самым тяжёлым компонентом при любом batch size > 1. После vectorization его доля в суммарной latency стала пренебрежимо малой по сравнению с forward pass LLM.

## 7. Связь с production GR

Статья является ключевым «infrastructure paper», который делает GR production-feasible. До неё существовала дилемма:

- Без constrained decoding: GR генерирует невалидные SID-ы с вероятностью, растущей с размером каталога — непригодно для production.
- С sequential trie: latency неприемлема при batch size > нескольких единиц и каталоге > сотен тысяч items.

Vectorized trie снимает это противоречие: constrained decoding работает *корректно* и *эффективно* одновременно. Это объясняет, почему YouTube смог задеплоить GR в production (см. статью «Actions Speak Louder Than Words», 2024) — Vectorized Trie является частью той же production infrastructure stack.

Без данного вклада latency GR на GPU/TPU остаётся неприемлемой для high-QPS рекомендательных систем с крупным каталогом: именно этот инфраструктурный gap и закрывает статья.

## 8. Ограничения

- **Dynamic catalog:** при добавлении новых items trie нужно перестраивать, а матрицу $M$ и perfect hash table — обновлять. Это batch-операция, не инкрементальная. Для каталогов с частыми обновлениями (новые видео каждую минуту) требуется careful refresh strategy с версионированием trie.
- **Memory overhead при очень большом vocab:** если SID использует большой словарь токенов, dense представление строк $M$ при gather может быть дорогим. При $|V| = 30\,000$ и batch 512 gather даёт 512 строк × 30K float16 ≈ 30 MB за шаг декодирования.
- **Очень большой trie:** при каталоге 10M+ items trie может не помещаться в GPU/TPU HBM и требует sharding по нескольким устройствам с соответствующим communication overhead при lookup.
- **Переменная длина SID:** метод оптимален при фиксированной длине SID. При переменных длинах нужен дополнительный механизм обработки EOS-токенов и padding.

## 9. Практические выводы

При деплое GR в production с Vectorized Trie рекомендуется мониторить следующие метрики:

<div class="table-scroll">
<table>
<thead>
<tr>
<th>Метрика</th>
<th>Что сигнализирует</th>
<th>Порог для внимания</th>
</tr>
</thead>
<tbody>
<tr>
<td>Trie build time при обновлении каталога</td>
<td>Задержка между появлением нового item и его доступностью в GR</td>
<td>Превышает SLA обновления каталога</td>
</tr>
<tr>
<td>Memory footprint trie (MB)</td>
<td>Нагрузка на SRAM/HBM ускорителя</td>
<td>&gt;20% доступной памяти ускорителя</td>
</tr>
<tr>
<td>Latency constrained decode vs unconstrained</td>
<td>Overhead векторизованной маскировки относительно «сырого» forward pass</td>
<td>&gt;10% от суммарной latency serving</td>
</tr>
<tr>
<td>Valid/invalid generated SID ratio</td>
<td>Корректность constrained decoding — все генерируемые SID-ы должны быть в каталоге</td>
<td>Invalid rate &gt; 0% означает баг в реализации</td>
</tr>
<tr>
<td>Покрытие новых items в trie</td>
<td>Свежесть trie относительно актуального каталога</td>
<td>Stale items &gt; допустимого окна задержки</td>
</tr>
</tbody>
</table>
</div>

### Checklist для production деплоя

1. Убедиться, что trie полностью покрывает весь каталог, включая только что добавленные items.
1. Настроить pipeline инкрементального (или периодического) обновления trie при catalog refresh.
1. Профилировать memory layout $M$ под конкретный ускоритель (TPU и GPU имеют разные оптимальные форматы sparse хранения).
1. Убедиться, что beam size $W$ и batch size $B$ тюнируются совместно с memory budget trie.
1. Мониторить долю latency constrained decoding в общем serving pipeline — после vectorization она должна стремиться к нулю.
