---
title: "Neural Discrete Representation Learning"
category: "generative_retrieval"
slug: "neural_discrete_representation_learning_summary"
catalogId: "paper-neural_discrete_representation_learning_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/neural_discrete_representation_learning_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/1711.00937"
---
Подробное саммари статьи:

> **Авторы:** Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu.
>
> **Аффилиации:** Google DeepMind.
>
> **Venue / статус:** NeurIPS 2017; arXiv:1711.00937, версия v2 от 2018-05-30.
>
> **Первичный источник:** [arXiv](https://arxiv.org/abs/1711.00937).

## 1. Коротко: о чем статья

Статья вводит **VQ-VAE** - Vector Quantised Variational AutoEncoder, один из базовых механизмов дискретного representation learning. В обычном VAE latent representation чаще всего непрерывное, например Gaussian latent vector. VQ-VAE вместо этого пропускает output encoder'а через nearest-neighbor lookup в learnable codebook: каждый continuous latent заменяется ближайшим embedding vector из конечного словаря. Получается bottleneck из дискретных кодов, но decoder по-прежнему обучается end-to-end.

Для semantic IDs в рекомендациях эта работа важна не потому, что она про recommender systems напрямую, а потому что она задает техническую основу для всей линии VQ/RQ tokenization. TIGER и многие наследники используют RQ-VAE как способ превратить item embedding в последовательность discrete tokens; RQ-VAE - это многослойное расширение идеи VQ-VAE, где каждый следующий quantizer кодирует residual предыдущего уровня. CoST, LETTER, ETEGRec и другие работы спорят уже не с фактом дискретизации, а с тем, какой objective и какие constraints делают эти codes полезными для retrieval.

Главная инженерная ценность VQ-VAE - простой способ обучать discrete latent variables без REINFORCE-like high-variance estimator. В forward pass модель делает hard nearest-neighbor assignment, а в backward pass использует straight-through estimator: gradient decoder'а копируется в output encoder'а, как будто quantization operation была identity. Codebook при этом обучается отдельным vector-quantization loss, а encoder стабилизируется commitment loss.

## 2. Контекст: почему дискретные latents важны

До VQ-VAE сильные generative models часто оказывались в неприятной развилке. Если decoder слабый, latent variables нужны, но качество samples ограничено. Если decoder мощный, например autoregressive PixelCNN/WaveNet, он может моделировать локальные зависимости сам и игнорировать latent code. Это известная проблема **posterior collapse**: encoder формально существует, но decoder фактически не использует его информацию.

Авторы делают ставку на дискретность: многие реальные структуры ближе к символам, чем к гладкому Gaussian space. Речь раскладывается на фонемы и слова, действия - на discrete decisions, изображения можно описывать объектами и атрибутами. Дискретный codebook заставляет модель выбирать из конечного набора состояний и тем самым может выучить более абстрактные, устойчивые признаки, а не хранить noise-level детали.

Для рекомендаций аналогия прямая. Item ID в production recommender'е тоже дискретен, но random ID не несет семантики. Semantic ID пытается заменить random ID на learnable discrete code, который наследует content/collaborative structure. VQ-VAE объясняет, как вообще можно получить такой codebook differentiably enough, а последующие RecSys-работы уже адаптируют objective под ranking/retrieval.

## 3. Модель VQ-VAE

На входе есть объект $x$: изображение, waveform, video frame sequence или другой сигнал. Encoder производит continuous representation:

$$
z_e(x)
$$

Есть learnable embedding table $e \in \mathbb{R}^{K \times D}$, где $K$ - число codebook entries, а $D$ - размер каждого code vector. Quantizer выбирает ближайший code vector:

$$
k = \arg\min_j \lVert z_e(x) - e_j \rVert_2, \qquad z_q(x) = e_k.
$$

<figure class="paper-figure">
  <img src="../../assets/vqvae/figure1_vqvae_architecture.png" alt="VQ-VAE architecture and embedding-space nearest-neighbor quantization">
  <figcaption><strong>Figure 1.</strong> Ключевая схема VQ-VAE: encoder выдает continuous $z_e(x)$, quantizer выбирает ближайший codebook vector, decoder восстанавливает объект по $z_q(x)$. Правая часть показывает, почему gradient через decoder не меняет сам hard assignment напрямую, а только двигает encoder output в следующем forward pass. Источник: van den Oord et al., arXiv:1711.00937.</figcaption>
</figure>

В терминах posterior distribution это deterministic categorical posterior: вероятность 1 у ближайшего code и 0 у остальных. Decoder получает не continuous encoder output, а выбранный code vector $z_q(x)$, и учится восстанавливать $x$.

Если вход имеет пространственную или временную структуру, discrete latent не один. Для ImageNet авторы используют grid из discrete codes, например $32 \times 32 \times 1$. Для speech latent sequence получается после strided convolutions. Это важно для semantic IDs: item code в рекомендациях обычно тоже не один token, а tuple или path.

## 4. Как обучается discrete bottleneck

Ключевой трюк - разделить обучение на три потока: decoder reconstruction, codebook update и commitment encoder'а к выбранному code. В статье objective записан как:

$$
\mathcal{L} = \log p(x \mid z_q(x)) + \lVert \operatorname{sg}[z_e(x)] - e \rVert_2^2 + \beta \lVert z_e(x) - \operatorname{sg}[e] \rVert_2^2.
$$

С точностью до знака reconstruction term в реализации обычно минимизируют negative log-likelihood или reconstruction loss. $\operatorname{sg}$ - stop-gradient operator: в forward pass это identity, а в backward pass gradient через operand не проходит.

- **Reconstruction/data term.** Обучает decoder восстанавливать input по выбранному discrete code. Gradient от decoder'а straight-through образом копируется в encoder output.
- **Codebook loss.** Двигает выбранный embedding vector к encoder output. Это похоже на dictionary learning или k-means centroid update.
- **Commitment loss.** Заставляет encoder не менять масштаб и не убегать от выбранных embeddings. В статье стандартно используют $\beta = 0.25$, при этом авторы отмечают устойчивость в диапазоне 0.1-2.0.

В forward pass nearest-neighbor assignment жесткий. В backward pass нет настоящего gradient через argmin, поэтому используется straight-through estimator. Это biased estimator, но практически он работает лучше, чем высокодисперсные score-function estimators для больших discrete spaces.

В appendix авторы также описывают альтернативное обновление codebook через exponential moving averages. Идея та же, что в online k-means: для каждого code vector поддерживать счетчик назначений и сумму encoder outputs, сглаженные с коэффициентом $\gamma$. Это стало важной практической деталью в последующих VQ/VQ-VAE реализациях, потому что EMA часто стабилизирует codebook utilization.

## 5. Prior поверх discrete codes

VQ-VAE разделяет обучение autoencoder'а и prior. Во время обучения VQ-VAE prior по $z$ считается uniform, поэтому KL term становится константой и не давит на encoder. После того как discrete latent space выучен, поверх code sequence/grid обучается отдельная autoregressive model:

- для изображений - PixelCNN по grid'у latent codes;
- для raw audio - WaveNet по latent sequence;
- для video/action-conditioned generation - prior по latent states во времени.

Эта двухстадийность очень похожа на pipeline semantic-ID recommendation: сначала строится tokenizer/codebook, затем downstream generative model учится генерировать sequence of codes. Разница в objective: в VQ-VAE tokenizer оптимизирован на reconstruction/generation качества исходного сигнала, а в RecSys tokenizer должен быть полезен для retrieval/ranking. Именно поэтому CoST заменяет MSE-реконструкцию contrastive retrieval-oriented signal, а другие работы добавляют collaborative regularization, end-to-end alignment или code balancing.

## 6. Эксперименты и evidence

### 6.1. CIFAR10: сравнение с continuous VAE и VIMCO

Первый эксперимент проверяет, насколько discrete latent VAE конкурентен continuous latent VAE. Используется CIFAR10, одинаковая convolutional VAE architecture, ADAM с learning rate $2 \cdot 10^{-4}$, 250k steps и batch size 128. Результаты в bits/dim:

<div class="table-scroll">
<table>
<thead><tr><th>Модель</th><th>Bits/dim</th><th>Комментарий</th></tr></thead>
<tbody>
<tr><td>Continuous VAE</td><td>4.51</td><td>Сильный baseline с continuous latent variables.</td></tr>
<tr><td>VQ-VAE</td><td>4.67</td><td>Немного хуже continuous VAE, но существенно лучше discrete VIMCO setup.</td></tr>
<tr><td>VIMCO</td><td>5.14</td><td>Discrete latent training через multi-sample objective; хуже в этом setup.</td></tr>
</tbody></table>
</div>

Вывод авторов: VQ-VAE закрывает большую часть разрыва между discrete latent models и continuous VAEs, при этом дает символический bottleneck, удобный для последующего autoregressive prior.

### 6.2. Images: ImageNet и DeepMind Lab

Для ImageNet 128x128x3 encoder сжимает изображение в $32 \times 32 \times 1$ discrete latent grid с $K=512$. Авторы оценивают это как примерно 42.6x reduction в битах относительно pixel representation. Figure 2 показывает, что reconstructions остаются похожими на оригиналы, хотя становятся немного более размытыми.

<figure class="paper-figure">
  <img src="../../assets/vqvae/figure2_imagenet_reconstructions.png" alt="ImageNet originals and VQ-VAE reconstructions from a 32 by 32 by 1 latent grid">
  <figcaption><strong>Figure 2.</strong> Слева оригинальные ImageNet изображения, справа reconstruction из $32 \times 32 \times 1$ discrete latent grid. Это важный sanity check: bottleneck сильно сжимает сигнал, но сохраняет object-level structure, что и делает VQ-подход полезным как tokenizer.</figcaption>
</figure>

После этого обучается PixelCNN prior поверх latent grid. Samples из PixelCNN затем декодируются обычным VQ-VAE decoder'ом в pixel space. Это важная демонстрация разнесения ролей: VQ-VAE учит compact discrete representation, а expensive autoregressive model работает уже не по пикселям, а по короткому latent grid.

На DeepMind Lab авторы повторяют идею для frames 84x84x3, получая $21 \times 21 \times 1$ latent space. Отдельно показан двухстадийный VQ-VAE: второй уровень моделирует целую сцену всего тремя discrete latents с $K=512$ каждый, то есть примерно 27 bits. Смысл этого эксперимента - показать, что даже с мощным PixelCNN decoder latents не схлопываются и продолжают хранить глобальную структуру сцены.

### 6.3. Audio: speech, speaker conversion и phoneme-like codes

На VCTK, где есть записи 109 speakers, encoder состоит из 6 strided convolutions и дает latent sequence с downsampling factor 64. Codebook size - 512. Decoder conditioned на speaker ID, поэтому модель может разделить content и speaker-specific voice characteristics.

Реконструкции не совпадают sample-by-sample с исходной waveform, но сохраняют произносимый content. Это как раз желательное свойство abstract discrete code: он не обязан хранить waveform noise, pitch/prosody details и микроскопические особенности сигнала, если decoder может восстановить правдоподобную речь.

Для unconditional speech generation авторы обучают модель на более крупном dataset с 460 speakers. При downsampling factor 128 chunk 40960 timesteps превращается в 320 latent timesteps. Авторы утверждают, что samples VQ-VAE содержат clear words и part-sentences, тогда как прямые raw-audio models вроде раннего WaveNet часто звучат как babbling в аналогичной unconditional постановке.

Самый сильный диагностический результат - сопоставление discrete codes с ground-truth phonemes, которые не использовались при обучении. При 128-dimensional discrete space на 25 Hz каждый latent value маппится к наиболее вероятной из 41 phoneme categories. Accuracy получается 49.3% против 7.2% для случайного latent space по most-likely phoneme baseline. Это не supervised phoneme recognizer, но хороший сигнал, что codebook выучил high-level speech descriptors.

### 6.4. Video: action-conditioned latent rollout

В video experiment используется DeepMind Lab. Модель получает первые frames и action sequence, затем генерирует future frames через latent-space rollout. Важно, что generation идет в latent space $z_t$, а pixel images получаются уже после decoding. Авторы показывают sequences для repeated action "move forward" и "move right"; локальная geometry и visual quality сохраняются без явной генерации каждого пикселя autoregressively.

## 7. Пошаговый алгоритм

1. **Выбрать encoder и decoder.** Для images это convolutional encoder/decoder; для audio - dilated convolutional/WaveNet-like decoder; для RecSys-аналогии это может быть item encoder и decoder/token objective.
1. **Инициализировать codebook.** Задать $K$ discrete embeddings размерности $D$.
1. **Encode.** Получить $z_e(x)$ для каждого input.
1. **Quantize.** Найти ближайший code vector $e_k$ и заменить $z_e(x)$ на $z_q(x)=e_k$.
1. **Decode.** Восстановить input или likelihood distribution $p(x \mid z_q(x))$.
1. **Backprop через straight-through.** Передать gradient decoder'а в encoder output, несмотря на hard assignment.
1. **Обновить codebook и commitment.** Codebook подтягивается к encoder outputs, encoder штрафуется за уход от выбранного code.
1. **После обучения VQ-VAE обучить prior.** PixelCNN/WaveNet/autoregressive model учится генерировать discrete latent sequence или grid.

## 8. Почему это фундаментально для semantic IDs

VQ-VAE вводит три решения, которые почти напрямую переносятся в semantic ID литературу:

- **Discrete codebook как learned vocabulary.** В рекомендациях codebook entries становятся semantic tokens или building blocks для item IDs.
- **Nearest-neighbor assignment.** Item embedding получает token через ближайший code vector. В RQ-VAE этот процесс повторяется по residual levels.
- **Straight-through training.** Позволяет соединить hard discrete tokenization с neural encoder training.

Но есть и важное расхождение. VQ-VAE хорош для compression/generation исходного modality signal. Recommender retrieval требует другого: сохранить различимость items, collaborative neighborhoods, head-tail behavior, collision profile и downstream generation compatibility. Поэтому reconstruction loss, достаточный для images/audio, может быть плохим proxy для recommendation. CoST как раз атакует этот gap: вместо "восстановить embedding" - "отличить свой item от других".

## 9. Сильные стороны

- **Простота механизма.** Hard nearest-neighbor quantization плюс straight-through estimator намного проще большинства discrete VAE estimators.
- **Latents не игнорируются мощным decoder'ом.** Эксперименты с PixelCNN decoder показывают, что discrete bottleneck помогает избежать posterior collapse.
- **Широкая проверка.** В статье есть images, speech, speaker conversion и video/action-conditioned generation, а не один toy dataset.
- **Полезная abstraction boundary.** Отдельное обучение tokenizer/autoencoder и prior стало шаблоном для многих generative-tokenization pipelines.

## 10. Ограничения и риски

- **Objective не retrieval-aware.** Для рекомендаций reconstruction quality не гарантирует хороший semantic ID. Два item'а могут быть реконструируемы, но плохо различимы для next-item generation.
- **Codebook utilization требует контроля.** Статья показывает механизм, но production tokenizer должен дополнительно мониторить dead codes, collapse, occupancy и collisions.
- **Two-stage training создает mismatch.** Prior обучается после tokenizer. Если downstream model нуждается в другой геометрии code space, VQ-VAE сам это не исправит.
- **Evidence по likelihood не равно evidence по utility.** CIFAR bits/dim и speech samples доказывают generative usefulness, но не доказывают качество retrieval/ranking.
- **Нет RecSys-экспериментов.** Для semantic IDs это foundational method paper, а не recommender baseline.

## 11. Как реализовать и проверять

- Логировать codebook occupancy, dead-code rate, assignment entropy и nearest-centroid distances.
- Проверять sensitivity к $K$, $D$, $\beta$, initialization и EMA vs loss-based codebook updates.
- В RecSys не останавливаться на reconstruction loss: отдельно считать collisions, item distinctiveness, head/tail slices и downstream Recall/NDCG.
- Сохранять versioned mapping от item/document к code sequence; иначе generator и catalog быстро рассинхронизируются.
- Если используется residual quantization, смотреть не только суммарную reconstruction error, но и вклад каждого residual level.

## 12. Итог

Neural Discrete Representation Learning - базовая статья для понимания semantic ID tokenization. Она объясняет, как можно обучать neural model с hard discrete bottleneck, почему codebook может хранить high-level structure, и как поверх discrete latents строится autoregressive prior.

Для генеративных рекомендаций главный урок двойной. С одной стороны, VQ-VAE дает технический фундамент: encoder, codebook, nearest-neighbor assignment, commitment loss и straight-through gradients. С другой стороны, сама статья не решает recommender-specific проблемы: retrieval-aware objective, semantic/collaborative mismatch, collisions, code utilization и catalog freshness. Поэтому ее нужно читать как основу для TIGER/RQ-VAE/CoST, а не как готовую рецептуру semantic IDs.
