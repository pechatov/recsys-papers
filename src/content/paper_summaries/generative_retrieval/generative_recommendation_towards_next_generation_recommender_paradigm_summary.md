---
title: "Generative Recommendation: Towards Next-generation Recommender Paradigm"
category: "generative_retrieval"
slug: "generative_recommendation_towards_next_generation_recommender_paradigm_summary"
catalogId: "paper-generative_recommendation_towards_next_generation_recommender_paradigm_summary"
sourceHtml: "summaries/paper_summaries/generative_retrieval/generative_recommendation_towards_next_generation_recommender_paradigm_summary.html"
generatedFromHtml: true
paperUrl: "https://arxiv.org/abs/2304.03516"
---
Подробное саммари статьи:

> **Авторы:** Wenjie Wang, Xinyu Lin, Fuli Feng, Xiangnan He, Tat-Seng Chua.
>
> **Аффилиации:** National University of Singapore; University of Science and Technology of China.
>
> **Статус:** arXiv:2304.03516, версия v2 от 2024-02-25; позиционная работа с feasibility study.
>
> **Первичный источник:** [arXiv](https://arxiv.org/abs/2304.03516). Код и данные авторы публикуют в [GitHub-репозитории GeneRec](https://github.com/Linxyhaha/GeneRec).

## 1. Коротко: о чем статья

Статья предлагает **GeneRec** - генеративную парадигму рекомендательных систем, где recommender не только выбирает item из фиксированного каталога, но может инициировать генерацию или редактирование контента под конкретного пользователя. Это не статья про semantic ID retrieval в узком смысле: здесь "generative recommendation" означает более широкий сдвиг от ranking-only pipeline к системе, которая умеет выполнять три действия: retrieval существующих items, repurposing существующих items и creation новых items.

Авторы начинают с критики классического retrieval-based recommender paradigm. В ней human uploaders или experts создают item corpus, recommender ранжирует элементы, пользователь дает пассивную обратную связь вроде кликов и dwell time, а модель оптимизирует будущую выдачу. Такой цикл хорошо масштабируется, но имеет два ограничения. Во-первых, в каталоге может не быть item'а, который точно удовлетворяет запрос пользователя. Во-вторых, пользователь обычно управляет рекомендациями косвенно, через noisy passive feedback, а не через точные инструкции.

GeneRec добавляет второй цикл: пользователь взаимодействует с AI generator через инструкции и feedback, а генератор создает или редактирует контент. Сгенерированный item может быть сразу показан пользователю, если запрос был явным, или отправлен в общий item corpus и дальше ранжироваться вместе с human-generated items. Центральная идея статьи: AIGC и LLM-интерфейсы позволяют recommender'у стать не только фильтром каталога, но и content-generation system.

## 2. Что именно авторы называют GeneRec

GeneRec строится вокруг двух новых целей для next-generation recommender systems:

1. **Автоматически repurpose или create items.** Система должна не только извлекать существующие объекты, но и менять их форму, стиль, фрагмент, обложку, описание или создавать новый объект с нуля.
1. **Интегрировать user instructions.** Пользователь может выразить запрос текстом, голосом, изображением, видео или мультимодальным сообщением; система должна объединять это с историческим feedback.

Важное различие: авторы не предлагают выбросить традиционный recommender. GeneRec скорее дополняет старый цикл. Human-generated content остается нужен, ranking остается нужен, item corpus остается нужен. Новое звено - AI generator, который получает user instructions и feedback, затем производит personalized content. Итоговый контент идет одним из двух путей: direct recommendation без ranking или добавление в corpus для последующего ранжирования.

<figure class="paper-figure">
  <img src="../../assets/generec/figure2_generec_paradigm.png" alt="GeneRec paradigm with AI generator, item corpus, recommender ranking, users, instructions, and feedback loops">
  <figcaption><strong>Figure 2.</strong> Общий цикл GeneRec. Важно, что AI generator не заменяет ranking-систему: он либо напрямую отдает generated item пользователю, либо добавляет item в corpus, где он дальше конкурирует с human-generated content через обычный recommender. Источник: Wang et al., arXiv:2304.03516.</figcaption>
</figure>

Пример из статьи - micro-video recommendation. Пользователь просит музыкальное видео по песне, система сначала может предложить существующий клип, затем пользователь уточняет стиль исполнения, и AI generator создает новую версию под этот запрос. Этот пример демонстрирует главную мотивацию: иногда правильного item'а в каталоге нет, но его можно синтезировать или адаптировать.

## 3. Архитектура: три модуля

Для инстанцирования парадигмы авторы выделяют три модуля: **instructor**, **AI editor** и **AI creator**. Это не готовая production architecture с конкретными API, а conceptual decomposition, через которую удобно разложить research tasks.

<figure class="paper-figure">
  <img src="../../assets/generec/figure4_modules.png" alt="GeneRec module view with instructor, AI editor, AI creator, item corpus, facts and knowledge, users, and uploaders">
  <figcaption><strong>Figure 4.</strong> Модульная декомпозиция GeneRec. Instructor решает, нужно ли запускать генерацию и как преобразовать user instructions/history в guidance. AI editor repurpose'ит существующий item, AI creator создает новый item, а facts/knowledge нужны для grounding и fidelity checks.</figcaption>
</figure>

### 3.1. Instructor

Instructor - слой, который стоит между пользователем, историей взаимодействий и генератором. Его входы:

- мультимодальные conversational instructions пользователя;
- пассивный feedback по историческим рекомендациям, например clicks, dwell time, likes, dislikes;
- контекст взаимодействия, если он доступен.

Instructor решает две задачи. Первая - понять, нужно ли вообще запускать AI generator. Например, если пользователь явно просит сгенерировать контент или много раз отвергает существующие items, генерация становится оправданной. Вторая - преобразовать инструкции и feedback в guidance signals, которые понимает генератор: prompt, instruction embedding, conditioning vector, выбор reference items или набор constraints.

Авторы подчеркивают, что instructor может быть интерактивным. Если пользовательское намерение неясно, он должен задавать дополнительные вопросы. Это отличает GeneRec от простого text-to-image prompt box: recommender имеет историю пользователя и может уточнять потребность, а не просто исполнять один prompt.

### 3.2. AI editor

AI editor отвечает за **personalized item editing**: он берет существующий item из каталога и меняет его под пользователя. Примеры из статьи: выбрать персонализированную thumbnail для micro-video, сгенерировать новую thumbnail, обрезать видео до интересного пользователю фрагмента, изменить стиль ролика, переписать новостной текст или подготовить personalized version рекламного креатива.

Входы AI editor:

- guidance signals от instructor;
- существующий item из item corpus;
- facts and knowledge из внешних источников, если задача требует factual grounding, правил или актуальных событий.

Выход - edited item, который должен лучше удовлетворять пользователя, чем исходный item. Важная деталь: AI editor может быть полезен не только конечному пользователю, но и human uploader'у. Например, автор видео может с помощью editor'а создать несколько версий обложки или стиля, а платформа потом ранжирует их для разных аудиторий.

### 3.3. AI creator

AI creator делает более сильный шаг: создает новый item без существующего исходного объекта. Он получает guidance signals и, при необходимости, внешние факты/знания, затем генерирует контент. В micro-video примере это может быть новый ролик под песню, исполнителя и стиль, извлеченные из user instructions и user history.

Этот модуль концептуально самый амбициозный и самый рискованный. Для него сложнее обеспечить factual correctness, авторские права, безопасность, качество и контроль соответствия запросу. В feasibility study статьи именно creation показывает худшую картину по качеству: personalized signals улучшают similarity к предпочтениям, но видео получаются визуально хуже, чем unconditional generation.

## 4. Fidelity checks

Авторы явно выносят trustworthiness в архитектурный контур. Для generated content недостаточно оптимизировать engagement: перед показом нужен набор проверок качества и надежности. В статье перечислены шесть групп checks.

- **Bias and fairness.** Генератор не должен воспроизводить стереотипы, hate speech, discrimination или другие вредные смещения. Особенно чувствительны news и политические темы.
- **Privacy.** Контент, созданный на основе персональных данных пользователя, не должен раскрывать sensitive information другим пользователям.
- **Safety.** Система должна снижать риски физического и психологического вреда, учитывать возрастные ограничения и быть устойчивой к атакам вроде shilling attacks.
- **Authenticity.** Для factual domains необходимо проверять факты, статистику и claims по надежным источникам, иначе recommender становится каналом масштабирования misinformation.
- **Legal compliance.** Нужны проверки соответствия законам, отраслевым правилам, healthcare/finance restrictions, copyright и ownership constraints.
- **Identifiability.** Авторы предлагают водяные знаки или другие способы отличать AI-generated, human-generated и AI-assisted content.

Для evaluation авторы разделяют item-side и user-side оценки. Item-side включает качество самого контента, релевантность запросу и fidelity checks. User-side оценивает satisfaction через explicit feedback, conversational feedback, CTR, dwell time и retention. Это важный момент: традиционные recommender metrics не исчезают, но становятся недостаточными, потому что generated content может быть кликабельным и одновременно небезопасным или недостоверным.

## 5. Roadmap статьи

Roadmap GeneRec разбит на три измерения.

**User-system interaction.** Исторически recommender'ы в основном использовали passive feedback. GeneRec добавляет active instructions и multimodal conversations. Авторы ожидают, что LLM-like interface позволит пользователю проще и точнее выражать потребности, а система будет объединять разговор с пассивной историей.

**Content generation.** Авторы описывают переход от expert-generated content к user-generated content и далее к AI-assisted/AI-created content. GeneRec не предполагает одномоментной замены людей генераторами. Более реалистичный путь - сначала AI помогает создателям и пользователям редактировать контент, затем в некоторых областях начинает создавать новые items.

**Algorithms.** Традиционные discriminative models для ranking остаются важными, но рядом с ними появляются generative models. Авторы ожидают сосуществование discriminative и generative approaches, а в перспективе - unified LLM-based recommender, который покрывает retrieval, repurposing и creation.

## 6. Домены применения

Статья приводит несколько доменных сценариев, чтобы показать, что GeneRec не привязан к micro-video.

- **News recommendation.** Генерация или персонализация новостных материалов по событиям дня. Главные checks: misinformation, misleading headlines, satire, bias, factuality.
- **Fashion recommendation.** AI может помогать дизайнерам создавать варианты продуктов или генерировать personalized fashion items. Checks связаны с реализмом, детализацией, цветом, стилевой совместимостью и потенциальным переходом к manufacturing.
- **Music recommendation.** Генератор может учитывать предпочтения по артистам, текстам, мелодиям и стилю исполнения. Центральные риски - copyright и legal compliance.
- **Micro-video recommendation.** Самый подробно рассмотренный домен. Он сложен из-за мультимодальности: subtitles, cover images, video, background music, ambient sounds.

Общий вывод авторов: зрелость GeneRec будет сильно зависеть от домена. Для текста и изображений многие building blocks уже работают лучше; для видео и мультимодального long-form content качество и контроль пока существенно сложнее.

## 7. Research tasks, которые формулирует статья

Статья не ограничивается vision-level схемой и перечисляет задачи, которые нужно решать, чтобы GeneRec стал реальной системой.

- **Instruction tuning for LLM-based instructor.** Нужно научить instructor понимать recommender-specific намерения и выбирать действия: отвечать, рекомендовать, запускать editor или creator.
- **Activation control.** Система должна решать, когда генерировать контент, когда редактировать существующий, когда продолжать ranking обычного каталога, а когда direct recommendation оправдана.
- **Personalized item editing.** Нужно интегрировать implicit feedback в генераторы, потому что пользователь часто не может или не хочет описывать все предпочтения в prompt.
- **Personalized item creation.** Это более сложная задача, которую авторы предлагают начинать с доменов, где генеративные технологии уже зрелее, например news и music.
- **Domain-specific fidelity checks.** Проверки должны зависеть от предметной области: одно дело thumbnail, другое - medical advice, financial report или политическая новость.

## 8. Сравнение с соседними направлениями

### 8.1. Conversational recommendation

Conversational recommender systems тоже общаются с пользователем, но GeneRec отличается двумя пунктами. Во-первых, авторы считают, что LLM-like instructor должен лучше понимать инструкции и intention, чем старые conversational recommenders. Во-вторых, GeneRec не только спрашивает preference и достает item из каталога, но может repurpose или create content.

### 8.2. Traditional AIGC

По сравнению с обычной генерацией контента GeneRec добавляет recommender-specific user modeling. Генератор получает не только prompt, но и implicit preference из истории пользователя. Это важно, потому что пользователь не всегда осознает или формулирует свои предпочтения. Например, длинный dwell time на определенном типе видео может быть сильным сигналом, даже если пользователь никогда не написал это словами.

Вторая разница - coexistence с human-generated content. Авторы не утверждают, что AI должен генерировать все. В некоторых ситуациях human content незаменим: уникальный авторский опыт, latest reports с места события, профессиональная экспертиза, реальные фотографии и видео.

## 9. Feasibility study: micro-video experiments

Практическая часть статьи проверяет, можно ли собрать простые версии AI editor и AI creator для micro-video. Instructor в экспериментах упрощен: вместо полноценного ChatGPT-like диалога авторы используют MMGCN как recommender model, извлекающий user embeddings или исторические liked items. Поэтому experiments стоит читать как feasibility demonstration, а не как полноценную реализацию всей GeneRec-схемы.

Датасет: 64,643 interactions, 7,895 users, 4,570 micro-videos. У каждого видео есть thumbnail примерно 1934 x 1080. После preprocessing каждое micro-video представлено 400 frames с разрешением 64 x 64. Домены видео включают разные жанры, например news и celebrities.

<figure class="paper-figure">
  <img src="../../assets/generec/figure5_editing_tasks.png" alt="AI editor tasks for thumbnail selection, thumbnail generation, and micro-video clipping">
  <figcaption><strong>Figure 5.</strong> Три простые editing-задачи из feasibility study. Они показывают более реалистичный первый шаг GeneRec: не создавать весь item с нуля, а персонализировать часть существующего micro-video через thumbnail selection, thumbnail generation или clipping.</figcaption>
</figure>

### 9.1. Thumbnail selection и thumbnail generation

Задача: для заданного video и истории пользователя выбрать или сгенерировать thumbnail, который лучше соответствует предпочтениям пользователя. Для selection авторы используют image encoder CLIP: усредняют representations thumbnails из historically liked videos пользователя и выбирают frame из candidate video с максимальным dot product. Для generation используют Retrieval-augmented Diffusion Model (RDM), conditioning которого строится на input video и historically liked videos.

Метрики:

- **Cosine@K.** Средняя cosine similarity между выбранными/сгенерированными thumbnails рекомендованных items и thumbnails из истории пользователя.
- **PS@K.** Prediction Score от обученного MMGCN по features выбранных/сгенерированных thumbnails и user representation.

<div class="table-scroll">
<table class="results-table">
<thead>
<tr><th>Метод</th><th>Cosine@5</th><th>Cosine@10</th><th>PS@5</th><th>PS@10</th></tr>
</thead>
<tbody>
<tr><td>Random Frame</td><td>0.4796</td><td>0.4786</td><td>22.6735</td><td>23.1950</td></tr>
<tr><td>Original</td><td>0.4978</td><td>0.4970</td><td>22.2606</td><td>22.7445</td></tr>
<tr><td>CLIP</td><td>0.5142</td><td>0.5134</td><td>22.7682</td><td>23.2854</td></tr>
<tr><td>RDM</td><td>0.5369</td><td>0.5347</td><td>23.0145</td><td>23.3712</td></tr>
</tbody>
</table>
</div>

Результат: CLIP лучше random/original за счет personalized selection, а RDM лучше всех по обеим метрикам. Но case study показывает важное ограничение: generated thumbnail может хуже сохранять fidelity исходного видео. То есть personalization score растет, но визуальная достоверность не гарантирована.

### 9.2. Micro-video clipping

Задача: выбрать короткий consecutive clip из длинного micro-video, чтобы показать пользователю наиболее релевантный фрагмент. Механизм похож на thumbnail selection: CLIP кодирует frames, clip representation считается как среднее по $L$ frames, а выбранный clip максимизирует similarity с user representation. Авторы подбирают $L$ из нескольких вариантов и используют $L=8$.

<div class="table-scroll">
<table class="results-table">
<thead>
<tr><th>Метод</th><th>Cosine@5</th><th>Cosine@10</th><th>PS@5</th><th>PS@10</th></tr>
</thead>
<tbody>
<tr><td>Random</td><td>0.4864</td><td>0.4851</td><td>22.1483</td><td>23.1401</td></tr>
<tr><td>1st Clip</td><td>0.4910</td><td>0.4899</td><td>22.1509</td><td>23.1657</td></tr>
<tr><td>Unclipped</td><td>0.4969</td><td>0.4976</td><td>22.1685</td><td>23.1700</td></tr>
<tr><td>CLIP</td><td>0.5052</td><td>0.5038</td><td>22.1863</td><td>23.1758</td></tr>
</tbody>
</table>
</div>

Здесь улучшения небольшие, но последовательные. Это скорее доказательство возможности personalized editing без генерации нового видео: система может адаптировать existing item через selection внутри объекта.

### 9.3. Micro-video content editing

Content editing разбит на два подтипа. Первый - style transfer по user instruction, где авторы используют VToonify для превращения видео в заданный стиль. Второй - revision по user feedback, где используется Masked Conditional Video Diffusion (MCVD). В MCVD input video сначала зашумляется, затем denoising process восстанавливает edited video с conditioning на user feedback.

<figure class="paper-figure">
  <img src="../../assets/generec/figure9_revision_creation.png" alt="Micro-video content revision and creation with forward corruption, reverse denoising, user embeddings, and instruction representation">
  <figcaption><strong>Figure 9.</strong> Схема двух diffusion-задач. Для revision модель начинает с существующего micro-video и редактирует его через denoising под user feedback. Для creation стартом служит noise, а conditioning включает user feedback и instruction representation; именно эта ветка в статье показывает худшее качество по FVD.</figcaption>
</figure>

Для user feedback авторы сравнивают два conditioning источника:

- **User_Emb.** User embedding из pretrained MMGCN.
- **User_Hist.** Усредненные features historically liked micro-videos.

Качество видео оценивается через FVD. Метрика сравнивает распределения real и generated videos по features I3D и учитывает temporal coherence. Чем ниже FVD, тем лучше:

$$
\mathrm{FVD} = \lVert \mu_R - \mu_G \rVert^2 + \operatorname{Tr}\left(\Sigma_R + \Sigma_G - 2(\Sigma_R \Sigma_G)^{1/2}\right).
$$

<div class="table-scroll">
<table class="results-table">
<thead>
<tr><th>Метод</th><th>Cosine@5</th><th>Cosine@10</th><th>PS@5</th><th>PS@10</th><th>FVD</th></tr>
</thead>
<tbody>
<tr><td>Original</td><td>0.5010</td><td>0.5083</td><td>25.8900</td><td>24.6800</td><td>-</td></tr>
<tr><td>User_Hist</td><td>0.5166</td><td>0.5127</td><td>25.9012</td><td>24.7107</td><td>783.7505</td></tr>
<tr><td>User_Emb</td><td>0.5273</td><td>0.5200</td><td>26.0200</td><td>24.7900</td><td>646.7156</td></tr>
</tbody>
</table>
</div>

Авторы также дают reference: unconditional micro-video revision имеет FVD 745.9443. Поэтому User_Emb не только лучше по preference metrics, но и лучше unconditional revision по FVD. User_Hist, наоборот, улучшает preference metrics, но имеет хуже FVD. Интерпретация авторов: compressed user embeddings могут содержать более чистый preference signal, тогда как raw history features шумнее.

### 9.4. Micro-video content creation

AI creator должен создать новое micro-video с нуля, используя instructions и user feedback. Для image synthesis авторы показывают примеры Stable Diffusion по single-turn instructions. Для video creation используют MCVD: starting point - random noise, conditioning - CLIP-encoded instruction плюс User_Emb или User_Hist.

<div class="table-scroll">
<table class="results-table">
<thead>
<tr><th>Метод</th><th>Cosine@5</th><th>Cosine@10</th><th>FVD</th></tr>
</thead>
<tbody>
<tr><td>Original</td><td>0.4883</td><td>0.4907</td><td>-</td></tr>
<tr><td>User_Hist</td><td>0.4902</td><td>0.4915</td><td>735.0413</td></tr>
<tr><td>User_Emb</td><td>0.5356</td><td>0.5376</td><td>743.1090</td></tr>
</tbody>
</table>
</div>

Reference для unconditional creation: FVD 727.8236. Это ключевой результат feasibility study: personalized conditioning повышает Cosine@K, особенно для User_Emb, но ухудшает FVD относительно unconditional generation. Case studies показывают артефакты: искаженные лица, размытие, неверные детали одежды и тела. Поэтому создание нового video content пока существенно слабее editing и selection.

Авторы предлагают три направления улучшения: использовать ChatGPT-like instructor для более богатых инструкций, лучше кодировать implicit preference, применять более сильные pretrained generative models с широким world knowledge в image/video/audio.

## 10. Как читать evidence статьи

Статья полезна прежде всего как agenda и decomposition, а не как строгий benchmark нового алгоритма. В feasibility study есть реальные numbers, но они проверяют несколько отдельных компонентов, а не всю GeneRec-систему end-to-end. Instructor сильно упрощен: нет полноценного conversational loop, нет реальной проверки activation decision, нет online user study и нет сравнения бизнес-метрик после интеграции generated content в recommender.

Сильная часть evidence - демонстрация, что existing multimodal models уже могут выполнять локальные editing tasks: thumbnail selection, thumbnail generation, clipping, simple style transfer, revision. Слабая часть - content creation: quality остается проблемой, а personalization metrics могут конфликтовать с visual fidelity. Это хорошо согласуется с архитектурной мыслью статьи: generation без fidelity checks и post-processing нельзя считать recommendation-ready.

## 11. Сильные стороны

- **Широкая постановка.** Статья отделяет generative recommendation как генерацию/редактирование content от более узкой линии generative retrieval по identifiers.
- **Хорошая модульная декомпозиция.** Instructor, AI editor и AI creator помогают разложить будущие системы на понятные research problems.
- **Явные trustworthiness checks.** Bias, privacy, safety, authenticity, legal compliance и identifiability встроены в постановку, а не оставлены как послесловие.
- **Feasibility experiments.** В статье есть не только vision, но и микровидео-демо с CLIP, RDM, VToonify, MCVD и численными результатами.
- **Корректное признание ограничений.** Авторы прямо показывают, что creation quality пока слабая, а copyright/regulation остаются открытыми вопросами.

## 12. Ограничения и риски

- **Нет end-to-end GeneRec evaluation.** Эксперименты не доказывают, что полный цикл instructor -> generator -> fidelity checks -> recommendation улучшит пользовательский опыт или бизнес-метрики.
- **Instructor почти не реализован.** Вместо реального conversational instructor используется MMGCN-derived guidance, поэтому центральная LLM-интерактивная часть остается vision-level.
- **Personalization metrics не равны content quality.** В creation experiments Cosine@K улучшается, но FVD хуже unconditional generation. Это предупреждение против оптимизации только на preference similarity.
- **Много нерешенных governance-вопросов.** Copyright, authorship, watermarking, privacy leakage и legal responsibility описаны как важные, но не решены технически.
- **Масштабирование не раскрыто.** Не обсуждены latency, serving cost, moderation throughput, storage generated items, cache policy и взаимодействие generated items с ranking infrastructure.
- **Связь с catalog retrieval неполная.** Статья утверждает coexistence retrieval и generation, но не дает строгого алгоритма выбора между direct generation, retrieval, editing и ranking.

## 13. Итог

Generative Recommendation: Towards Next-generation Recommender Paradigm - это foundation/position paper, который расширяет термин "generative recommendation" за пределы генерации item IDs. Его главный вклад - формулировка GeneRec как recommender paradigm, где пользовательские инструкции и исторический feedback могут запускать AI editor или AI creator, а generated content затем либо напрямую показывается пользователю, либо попадает в item corpus.

Статью стоит читать как карту будущих системных задач: activation control, multimodal instruction understanding, personalization of generative models, fidelity checks, copyright, evaluation и coexistence human-generated/AI-generated content. Но как method paper она ограничена: нет полноценного end-to-end сравнения, instructor не реализован в заявленном виде, а micro-video creation показывает, что сильная персонализация может идти вразрез с качеством генерации.
