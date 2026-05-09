# Статьи по рекомендательным системам с ключевых конференций (2023-2025)

Составлено 2026-05-08. Период: 2023-2025, то есть последние три полных года конференционных публикаций на момент составления. В список включены значимые и явно релевантные recommender systems статьи из основных площадок: ACM RecSys, SIGIR, KDD, The Web Conference / WWW, WSDM, CIKM, AAAI, IJCAI, NeurIPS, ICLR и ICML. Воркшопы, демо и doctoral papers не включались, если работа не стала заметной для направления.

Аффилиации указаны без имён авторов и сведены до уникальных организаций. Источники для аффилиаций: страницы accepted papers конференций, ACM DL, arXiv/OpenReview/PMLR/PDF-тексты статей и уже проверенные записи в локальной базе.

## Сводка по тематикам

| Тематика | Количество |
|---|---:|
| LLM, агенты и языковые профили в рекомендациях | 18 |
| Generative retrieval, semantic IDs и диффузионные рекомендации | 15 |
| Sequential/session recommendation и long user history | 18 |
| Graph, knowledge-aware, cross-domain и multi-behavior recommendation | 12 |
| Multimodal, content-aware и cold-start recommendation | 12 |
| Fairness, privacy, robustness, debiasing и evaluation | 13 |
| Industrial RecSys, CTR, ads, ranking infrastructure | 12 |

## LLM, агенты и языковые профили в рекомендациях

| Название | Конференция | Год | Аффилиации авторов |
|---|---:|---:|---|
| [TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation](https://arxiv.org/abs/2305.00447) | RecSys | 2023 | University of Science and Technology of China |
| [Uncovering ChatGPT's Capabilities in Recommender Systems](https://arxiv.org/abs/2305.02182) | RecSys | 2023 | Renmin University of China |
| [Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation](https://arxiv.org/abs/2305.07609) | RecSys | 2023 | University of Science and Technology of China |
| [Leveraging Large Language Models for Sequential Recommendation](https://arxiv.org/abs/2309.09261) | RecSys | 2023 | Delft University of Technology; Athens University of Economics and Business |
| [Large Language Models are Competitive Near Cold-start Recommenders for Language- and Item-based Preferences](https://arxiv.org/abs/2307.14225) | RecSys | 2023 | University of Toronto; Google |
| [Prompt Distillation for Efficient LLM-based Recommendation](https://doi.org/10.1145/3583780.3615017) | CIKM | 2023 | Hong Kong Baptist University; Rutgers University |
| [CALRec: Contrastive Alignment of Generative LLMs For Sequential Recommendation](https://arxiv.org/abs/2405.02429) | RecSys | 2024 | University of Cambridge; Google; Google Research |
| [IDGenRec: LLM-RecSys Alignment with Textual ID Learning](https://arxiv.org/abs/2403.19021) | SIGIR | 2024 | Rutgers University |
| [Data-efficient Fine-tuning for LLM-based Recommendation](https://arxiv.org/abs/2401.17197) | SIGIR | 2024 | National University of Singapore; Hong Kong Polytechnic University |
| [OpenP5: An Open-Source Platform for Developing, Training, and Evaluating LLM-based Recommender Systems](https://arxiv.org/abs/2306.11134) | SIGIR | 2024 | Rutgers University |
| [Representation Learning with Large Language Models for Recommendation](https://arxiv.org/abs/2310.15950) | WWW | 2024 | University of Hong Kong; Baidu |
| [ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation](https://arxiv.org/abs/2308.11131) | WWW | 2024 | Shanghai Jiao Tong University; Huawei Technologies |
| [Harnessing Large Language Models for Text-Rich Sequential Recommendation](https://doi.org/10.1145/3589334.3645358) | WWW | 2024 | University of Science and Technology of China; Hong Kong University of Science and Technology; University of Hong Kong; Venus Medtech |
| [Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System](https://arxiv.org/abs/2404.11343) | KDD | 2024 | KAIST; NAVER |
| [Large Language Model driven Policy Exploration for Recommender Systems](https://arxiv.org/abs/2501.13816) | WSDM | 2025 | University of Glasgow; Google DeepMind; Telefonica Research |
| [Reindex-Then-Adapt: Improving Large Language Models for Conversational Recommendation](https://arxiv.org/abs/2405.12119) | WSDM | 2025 | University of California San Diego; Netflix |
| [STARec: An Efficient Agent Framework for Recommender Systems via Autonomous Deliberate Reasoning](https://arxiv.org/abs/2508.18812) | CIKM | 2025 | Renmin University of China; Huawei |
| [Local Large Language Models for Recommendation](https://doi.org/10.1145/3746252.3761280) | CIKM | 2025 | Seoul National University |

## Generative Retrieval, Semantic IDs и диффузионные рекомендации

| Название | Конференция | Год | Аффилиации авторов |
|---|---:|---:|---|
| [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) | NeurIPS | 2023 | University of Wisconsin-Madison; Google; Google DeepMind |
| [Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion](https://arxiv.org/abs/2310.20453) | NeurIPS | 2023 | University of Science and Technology of China; Hong Kong Polytechnic University |
| [Generative Flow Network for Listwise Recommendation](https://arxiv.org/abs/2306.02239) | KDD | 2023 | Kuaishou Technology; University of California San Diego; Peking University |
| [Bridging Search and Recommendation in Generative Retrieval: Does One Task Help the Other?](https://arxiv.org/abs/2410.16823) | RecSys | 2024 | Spotify |
| [Learnable Item Tokenization for Generative Recommendation](https://arxiv.org/abs/2405.07314) | SIGIR | 2024 | National University of Singapore |
| [EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration](https://arxiv.org/abs/2406.14017) | KDD | 2024 | Zhejiang University; Huawei Noah's Ark Lab |
| [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) | ICML | 2024 | Meta |
| [Generative News Recommendation](https://arxiv.org/abs/2403.03424) | WWW | 2024 | University of Electronic Science and Technology of China; Shandong University; Renmin University of China; Leiden University |
| [DimeRec: A Unified Framework for Enhanced Sequential Recommendation via Generative Diffusion Models](https://arxiv.org/abs/2408.12153) | WSDM | 2025 | University of Science and Technology of China; Kuaishou; Sun Yat-sen University; Alibaba |
| [Generative Recommender with End-to-End Learnable Item Tokenization](https://arxiv.org/abs/2409.05546) | SIGIR | 2025 | Renmin University of China; Kuaishou |
| [GenSAR: Unifying Balanced Search and Recommendation with Generative Retrieval](https://arxiv.org/abs/2504.05730) | RecSys | 2025 | Renmin University of China; Kuaishou Technology |
| [GRACE: Generative Recommendation via Journey-Aware Sparse Attention on Chain-of-Thought Tokenization](https://arxiv.org/abs/2507.14758) | RecSys | 2025 | Walmart Global Tech; Walmart |
| [Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation](https://arxiv.org/abs/2408.06883) | RecSys | 2025 | Spotify |
| [Entity-Aware Generative Retrieval for Personalized Contexts](https://doi.org/10.1145/3746252.3761211) | CIKM | 2025 | Seoul National University; Samsung Electronics |
| [Reinforcement Learning-Driven Generative Retrieval with Semantic-aligned Multi-Layer Identifiers](https://doi.org/10.1145/3746252.3761136) | CIKM | 2025 | Dalian University of Technology; City University of Hong Kong |

## Sequential/Session Recommendation и Long User History

| Название | Конференция | Год | Аффилиации авторов |
|---|---:|---:|---|
| [A Multi-view Graph Contrastive Learning Framework for Cross-Domain Sequential Recommendation](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | Shenzhen University |
| [Contrastive Learning with Frequency-Domain Interest Trends for Sequential Recommendation](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | Harbin Engineering University |
| [Distribution-based Learnable Filters with Side Information for Sequential Recommendation](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | Hebei University; Northeastern University |
| [Equivariant Contrastive Learning for Sequential Recommendation](https://arxiv.org/abs/2211.05290) | RecSys | 2023 | HKUST; Upstage; Peking University; Harvard Medical School; University of Technology Sydney; University of Hong Kong |
| [STRec: Sparse Transformer for Sequential Recommendations](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | City University of Hong Kong; Xi'an Jiaotong University; Michigan State University; Wuhan University; Hong Kong Polytechnic University |
| [A Pre-trained Zero-shot Sequential Recommendation Framework via Popularity Dynamics](https://recsys.acm.org/recsys24/accepted-contributions) | RecSys | 2024 | University of Illinois Urbana-Champaign |
| [Scaling Law of Large Sequential Recommendation Models](https://arxiv.org/abs/2311.11351) | RecSys | 2024 | Renmin University of China; University of California San Diego; Tencent |
| [Sequential Recommendation with Latent Relations based on Large Language Model](https://doi.org/10.1145/3626772.3657762) | SIGIR | 2024 | Tsinghua University |
| [Ada-Retrieval: An Adaptive Multi-Round Retrieval Paradigm for Sequential Recommendations](https://doi.org/10.1609/aaai.v38i8.28712) | AAAI | 2024 | Renmin University of China; Microsoft Research Asia |
| [Plug-In Diffusion Model for Sequential Recommendation](https://arxiv.org/abs/2401.02913) | AAAI | 2024 | Shandong University; Tencent |
| [An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention](https://arxiv.org/abs/2312.10325) | AAAI | 2024 | Yonsei University |
| [Temporal Graph Contrastive Learning for Sequential Recommendation](https://doi.org/10.1609/aaai.v38i8.28789) | AAAI | 2024 | University of Science and Technology of China; Hong Kong University of Science and Technology; Guangzhou HKUST Fok Ying Tung Research Institute; University of Hong Kong |
| [Temporal Linear Item-Item Model for Sequential Recommendation](https://arxiv.org/abs/2412.07382) | WSDM | 2025 | Sungkyunkwan University |
| [Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation](https://arxiv.org/abs/2411.01457) | WSDM | 2025 | Nanyang Technological University |
| [Oracle-guided Dynamic User Preference Modeling for Sequential Recommendation](https://arxiv.org/abs/2412.00813) | WSDM | 2025 | Fudan University; Microsoft Research Asia; Amazon |
| [MUFFIN: Mixture of User-Adaptive Frequency Filtering for Sequential Recommendation](https://arxiv.org/abs/2508.13670) | CIKM | 2025 | Sungkyunkwan University |
| [Lost in Sequence: Do Large Language Models Understand Sequential Recommendation?](https://arxiv.org/abs/2502.13909) | KDD | 2025 | KAIST; NAVER Corporation; University of California San Diego |
| [Revisiting Self-attention for Cross-domain Sequential Recommendation](https://arxiv.org/abs/2505.21811) | KDD | 2025 | Snap Inc. |

## Graph, Knowledge-Aware, Cross-Domain и Multi-Behavior Recommendation

| Название | Конференция | Год | Аффилиации авторов |
|---|---:|---:|---|
| [BVAE: Behavior-aware Variational Autoencoder for Multi-Behavior Multi-Task Recommendation](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | Shenzhen University |
| [KGTORe: Tailored Recommendations through Knowledge-aware GNN Models](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | Polytechnic University of Bari |
| [Knowledge-based Multiple Adaptive Spaces Fusion for Recommendation](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | Beihang University; University of Chinese Academy of Sciences; Beijing Academy of Blockchain and Edge Computing |
| [Multi-Relational Contrastive Learning for Recommendation](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | University of Hong Kong |
| [Adaptive Fusion of Multi-View for Graph Contrastive Recommendation](https://recsys.acm.org/recsys24/accepted-contributions) | RecSys | 2024 | Zhejiang University |
| [DGR: A General Graph Desmoothing Framework for Recommendation via Global and Local Perspectives](https://arxiv.org/abs/2403.04287) | IJCAI | 2024 | Beijing Jiaotong University; Griffith University; Augusta University; Northwestern Polytechnical University |
| [S-Diff: An Anisotropic Diffusion Model for Collaborative Filtering in Spectral Domain](https://arxiv.org/abs/2501.00384) | WSDM | 2025 | Nanjing University of Aeronautics and Astronautics; Kuaishou Technology |
| [LightGNN: Simple Graph Neural Network for Recommendation](https://arxiv.org/abs/2501.03228) | WSDM | 2025 | University of Hong Kong |
| [Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation](https://recsys.acm.org/recsys25/accepted-contributions/) | RecSys | 2025 | Chinese University of Hong Kong; Tencent; Fuzhou University |
| [A Self-Supervised Mixture-of-Experts Framework for Multi-behavior Recommendation](https://arxiv.org/abs/2508.19507) | CIKM | 2025 | KAIST |
| [Collaborative Interest Mining Network for Knowledge Graph-based Recommendation](https://doi.org/10.1145/3746252.3761258) | CIKM | 2025 | Guangxi University |
| [LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture](https://arxiv.org/abs/2506.10347) | KDD | 2025 | Zhejiang University; Singapore University of Technology and Design |

## Multimodal, Content-Aware и Cold-Start Recommendation

| Название | Конференция | Год | Аффилиации авторов |
|---|---:|---:|---|
| [Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited](https://arxiv.org/abs/2303.13835) | SIGIR | 2023 | Westlake University; Zhejiang Lab |
| [Goal-Oriented Multi-Modal Interactive Recommendation with Verbal and Non-Verbal Relevance Feedback](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | University of Glasgow |
| [Multi-task Item-attribute Graph Pre-training for Strict Cold-start Item Recommendation](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | University of Illinois Chicago; Salesforce; Beihang University; Yale University |
| [A Multi-modal Modeling Framework for Cold-start Short-video Recommendation](https://recsys.acm.org/recsys24/accepted-contributions) | RecSys | 2024 | Kuaishou Technology; Chinese Academy of Sciences |
| [A Multimodal Single-Branch Embedding Network for Recommendation in Cold-start and Missing Modality Scenarios](https://recsys.acm.org/recsys24/accepted-contributions) | RecSys | 2024 | Johannes Kepler University Linz; Linz Institute of Technology |
| [MARec: Metadata Alignment for cold-start Recommendation](https://recsys.acm.org/recsys24/accepted-contributions) | RecSys | 2024 | Amazon; University of Adelaide |
| [Prompt Tuning for Item Cold-start Recommendation](https://recsys.acm.org/recsys24/accepted-contributions) | RecSys | 2024 | Kuaishou Technology; Peking University |
| [FineRec: Exploring Fine-grained Sequential Recommendation](https://arxiv.org/abs/2404.12975) | SIGIR | 2024 | Dalian University of Technology; Pennsylvania State University |
| [Who To Align With: Feedback-Oriented Multi-Modal Alignment in Recommendation Systems](https://sigir-2024.github.io/proceedings.html) | SIGIR | 2024 | Xiamen University; Jinan University; Zhejiang University |
| [FindRec: Stein-Guided Entropic Flow for Multi-Modal Sequential Recommendation](https://arxiv.org/abs/2507.04651) | KDD | 2025 | City University of Hong Kong; Beihang University; University of Technology Sydney; University of Queensland; Fudan University |
| [Hypercomplex Prompt-aware Multimodal Recommendation](https://arxiv.org/abs/2508.10753) | CIKM | 2025 | Hong Kong Polytechnic University; University of Hong Kong; Carnegie Mellon University; University College Dublin |
| [Do Recommender Systems Really Leverage Multimodal Content? A Comprehensive Analysis on Multimodal Representations for Recommendation](https://arxiv.org/abs/2508.04571) | CIKM | 2025 | Politecnico di Bari |

## Fairness, Privacy, Robustness, Debiasing и Evaluation

| Название | Конференция | Год | Аффилиации авторов |
|---|---:|---:|---|
| [Towards Robust Fairness-aware Recommendation](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | Renmin University of China; Ant Group |
| [Two-sided Calibration for Quality-aware Responsible Recommendation](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | Tsinghua University; China Mobile Research |
| [Popularity Debiasing from Exposure to Interaction in Collaborative Filtering](https://sigir.org/sigir2023/program/accepted-papers/short-papers/) | SIGIR | 2023 | Chinese Academy of Sciences |
| [Fair Sequential Recommendation without User Demographics](https://sigir-2024.github.io/proceedings.html) | SIGIR | 2024 | ETH Zurich; University of Amsterdam |
| [Configurable Fairness for New Item Recommendation Considering Entry Time of Items](https://sigir-2024.github.io/proceedings.html) | SIGIR | 2024 | Tsinghua University; China Mobile Research |
| [Putting Popularity Bias Mitigation to the Test: A User-Centric Evaluation in Music Recommenders](https://recsys.acm.org/recsys24/accepted-contributions) | RecSys | 2024 | Delft University of Technology; Utrecht University |
| [Can We Trust Recommender System Fairness Evaluation? The Role of Fairness and Relevance](https://arxiv.org/abs/2405.18276) | SIGIR | 2024 | University of Amsterdam; Radboud University |
| [Federated Recommendation with Additive Personalization](https://openreview.net/forum?id=xkXdE81mOK) | ICLR | 2024 | University of Technology Sydney; University of Maryland, College Park |
| [Auditing Recommender Systems for User Empowerment in Very Large Online Platforms under the Digital Services Act](https://recsys.acm.org/recsys25/accepted-contributions/) | RecSys | 2025 | IMT School for Advanced Studies Lucca; University of Cagliari |
| [Integrating Individual and Group Fairness for Recommender Systems through Social Choice](https://recsys.acm.org/recsys25/accepted-contributions/) | RecSys | 2025 | University of Colorado Boulder; Comenius University Bratislava; Tulane University |
| [PriviRec: Confidential and Decentralized Graph Filtering for Recommender Systems](https://doi.org/10.1145/3746252.3761152) | CIKM | 2025 | INSA Lyon; Inria; McGill University; LIRIS CNRS |
| [LeadFairRec: LLM-enhanced Discriminative Counterfactual Debiasing for Two-sided Fairness in Recommendation](https://doi.org/10.1145/3746252.3761126) | CIKM | 2025 | Northeastern University; RMIT University; Liaoning University |
| [Evaluating and Addressing Fairness Across User Groups in Negative Sampling for Recommender Systems](https://doi.org/10.1145/3746252.3761263) | CIKM | 2025 | RMIT University; ETH Zurich |

## Industrial RecSys, CTR, Ads, Ranking Infrastructure

| Название | Конференция | Год | Аффилиации авторов |
|---|---:|---:|---|
| [AutoOpt: Automatic Hyperparameter Scheduling and Optimization for Deep Click-through Rate Prediction](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | Huawei Noah's Ark Lab |
| [Deep Situation-Aware Interaction Network for Click-Through Rate Prediction](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | Institute of Software, Chinese Academy of Sciences; Meituan |
| [Online Matching: A Real-time Bandit System for Large-scale Recommendations](https://recsys.acm.org/recsys23/accepted-contributions/) | RecSys | 2023 | Google |
| [AIE: Auction Information Enhanced Framework for CTR Prediction in Online Advertising](https://recsys.acm.org/recsys24/accepted-contributions) | RecSys | 2024 | Huawei Noah's Ark Lab |
| [FLIP: Fine-grained Alignment between ID-based Models and Pretrained Language Models for CTR Prediction](https://arxiv.org/abs/2310.19453) | RecSys | 2024 | Shanghai Jiao Tong University; Huawei Noah's Ark Lab |
| [LARR: Large Language Model Aided Real-time Scene Recommendation with Semantic Understanding](https://recsys.acm.org/recsys24/accepted-contributions) | RecSys | 2024 | Meituan |
| [RPAF: A Reinforcement Prediction-Allocation Framework for Cache Allocation in Large-Scale Recommender Systems](https://recsys.acm.org/recsys24/accepted-contributions) | RecSys | 2024 | Kuaishou Technology; Tsinghua University |
| [ReLand: Integrating Large Language Models' Insights into Industrial Recommenders via a Controllable Reasoning Pool](https://recsys.acm.org/recsys24/accepted-contributions) | RecSys | 2024 | Ant Group; Zhejiang University |
| [Exploring Scaling Laws of CTR Model for Online Performance Improvement](https://recsys.acm.org/recsys25/accepted-contributions/) | RecSys | 2025 | Institute of Software, Chinese Academy of Sciences; Meituan |
| [LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](https://arxiv.org/abs/2505.04421) | RecSys | 2025 | ByteDance |
| [PinFM: Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery Platform](https://arxiv.org/abs/2507.12704) | RecSys | 2025 | Pinterest |
| [GREAT: Guiding Query Generation with a Trie for Recommending Related Search about Video at Kuaishou](https://arxiv.org/abs/2507.15267) | KDD | 2025 | Kuaishou Technology |

## Примечания по покрытию

- RecSys включён шире остальных площадок, потому что это специализированная конференция по рекомендательным системам.
- Для SIGIR, KDD, WWW, WSDM, CIKM, AAAI, IJCAI, NeurIPS, ICLR и ICML включались статьи, где recommender systems являются основной задачей, а не только вторичным примером применения.
- Для части свежих работ 2025 года DOI ещё не индексировались в OpenAlex/ACM на момент проверки; в таких случаях ссылка ведёт на официальную страницу accepted papers или arXiv.
- Аффилиации нормализованы до организаций, поэтому факультеты, лаборатории и повторяющиеся подразделения обычно объединены в один вуз/компанию.
