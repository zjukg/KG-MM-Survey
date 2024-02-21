# KG-MM-Survey
[![Awesome](https://awesome.re/badge.svg)](https://github.com/zjukg/KG-MM-Survey) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/zjukg/KG-MM-Survey/blob/main/LICENSE)
![](https://img.shields.io/github/last-commit/zjukg/KG-MM-Survey?color=green) 
![](https://img.shields.io/badge/PRs-Welcome-red) 

![Task](figures/task.jpg)

> 🙌 This repository collects papers integrating Knowledge Graphs (KGs) and Multi-Modal Learning, focusing on research in two principal aspects: **KG-driven Multi-Modal (KG4MM) learning**, where KGs support multi-modal tasks, and **Multi-Modal Knowledge Graph (MM4KG)**, which extends KG studies into the MMKG realm.

😎 Welcome to recommend missing papers through **`Adding Issues`** or **`Pull Requests`**. 
<details>
   <summary>👈 🔎 Roadmap </summary>
   
![Roadmap](figures/roadmap.jpg)
</details>



## 🔔 News
- **`2024-02` We preprint our Survey [Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey](http://arxiv.org/abs/2402.05391)  [[`Repo`](https://github.com/zjukg/KG-MM-Survey)].**

*Todo:*
1. - [ ] `Finish updating papers`
         
## 📜 Content

- [KG-MM-Survey](#kg-mm-survey)
  - [🔔 News](#-news)
  - [📜Content](#-content)
  - [🤖🌄 KG-driven Multi-modal Learning (KG4MM)](#-kg-driven-multi-modal-learning-kg4mm)
    - [Understanding & Reasoning Tasks](#understanding--reasoning-tasks)
      - [Visual Question Answering](#visual-question-answering)
      - [Visual Question Generation](#visual-question-generation)
      - [Visual Dialog](#visual-dialog)
    - [Classification Tasks](#classification-tasks)
      - [Image Classification](#image-classification)
      - [Fake News Detection](#fake-news-detection)
      - [Movie Genre Classification](#movie-genre-classification)
    - [Content Generation Tasks](#content-generation-tasks)
      - [Image Captioning](#image-captioning)
      - [Visual Storytelling](#visual-storytelling)
      - [Conditional Text-to-Image Generation](#conditional-text-to-image-generation)
      - [Scene Graph Generation](#scene-graph-generation)
    - [Retrieval Tasks](#retrieval-tasks)
      - [Cross-Modal Retrieval](#cross-modal-retrieval)
      - [Visual Referring Expressions & Grounding](#visual-referring-expressions--grounding)
    - [KG-aware Mutli-modal Pre-training](#kg-aware-mutli-modal-pre-training)
      - [Structure Knowledge aware Pre-training](#structure-knowledge-aware-pre-training)
      - [Knowledge Graph aware Pre-training](#knowledge-graph-aware-pre-training)
  - [🌄🤖 Multi-modal Knowledge Graph (MM4KG)](#-multi-modal-knowledge-graph-mm4kg)
    - [MMKG Resources](#mmkg-resources)
      - [Public MMKGs](#public-mmkgs)
      - [MMKG Construction Methods](#mmkg-construction-methods)
    - [MMKG Acquisition](#mmkg-acquisition)
      - [Multi-modal Named Entity Recognition](#multi-modal-named-entity-recognition)
      - [Multi-modal Relation Extraction](#multi-modal-relation-extraction)
      - [Multi-modal Event Extraction](#multi-modal-event-extraction)
    - [MMKG Fusion](#mmkg-fusion)
      - [Multi-modal Entity Alignment](#multi-modal-entity-alignment)
      - [Multi-modal Entity Linking & Disambiguation](#multi-modal-entity-linking--disambiguation)
    - [MMKG Inference](#mmkg-inference)
      - [Multi-modal Knowledge Graph Completion](#multi-modal-knowledge-graph-completion)
      - [Multi-modal Knowledge Graphs Reasoning](#multi-modal-knowledge-graphs-reasoning)
    - [MMKG-driven Tasks](#mmkg-driven-tasks)
      - [Retrieval](#retrieval)
      - [Reasoning & Generation](#reasoning--generation)
      - [Pre-training](#pre-training)
      - [AI for Science](#ai-for-science)
      - [Industry Application](#industry-application)
  - [Contribution](#contribution)
    - [👥 Contributors](#-contributors)
    - [🔖 Contact](#-contact)
    - [🤝 Citation:](#-cite)

---

## 🤖🌄 KG-driven Multi-modal Learning (KG4MM)

### Understanding & Reasoning Tasks
<details>
   <summary>👈 🔎 Pipeline </summary>
   
![KG4MMR](figures/kg4mmr.jpg)
</details>

#### Visual Question Answering
<details>
    <summary>👈 🔎 Benchmarks </summary>

![VQA](figures/vqatab.jpg)
</details>

- \[[arXiv 2024](https://arxiv.org/abs/2402.12728)\] Modality-Aware Integration with Large Language Models for Knowledge-based Visual Question Answering.
- \[[arXiv 2023](https://arxiv.org/abs/2312.12723)\] Multi-Clue Reasoning with Memory Augmentation for Knowledge-based Visual Question Answering.
- \[[arXiv 2023](https://arxiv.org/abs/2310.08148)\] Open-Set Knowledge-Based Visual Question Answering with Inference Paths.
- \[[arXiv 2023](https://arxiv.org/abs/2308.15851)\] Prompting Vision Language Model with Knowledge from Large Language Model for Knowledge-Based VQA.
- \[[EMNLP 2023](https://arxiv.org/abs/2310.20159)\] Language Guided Visual Question Answering: Elevate Your Multimodal Language Model Using Knowledge-Enriched Prompts.
- \[[EMNLP 2023](https://arxiv.org/abs/2310.13570)\] A Simple Baseline for Knowledge-Based Visual Question Answering.
- \[[EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.437/)\] MM-Reasoner: A Multi-Modal Knowledge-Aware Framework for Knowledge-Based Visual Question Answering.
- \[[NeurIPS 2023](https://openreview.net/forum?id=bW1uwPV3im)\] LoRA: A Logical Reasoning Augmented Dataset for Visual Question Answering.
- \[[CVPR 2023](https://arxiv.org/abs/2303.01903)\] Prompting Large Language Models with Answer Heuristics for Knowledge-Based Visual Question Answering.
- \[[EACL 2023](https://arxiv.org/abs/2303.10699)\] FVQA 2.0: Introducing Adversarial Samples into Fact-based Visual Question Answering.
- \[[WACV 2023](https://arxiv.org/abs/2210.13626)\] VLC-BERT: Visual Question Answering with Contextualized Commonsense Knowledge.
- \[[ICASSP 2023](https://ieeexplore.ieee.org/abstract/document/10096074/)\] Outside Knowledge Visual Question Answering Version 2.0.
- \[[ICME 2023](https://ieeexplore.ieee.org/abstract/document/10219862/)\] A Retriever-Reader Framework with Visual Entity Linking for Knowledge-Based Visual Question Answering.
- \[[TIP 2023](https://ieeexplore.ieee.org/abstract/document/10268338/)\] Semantic-Aware Modular Capsule Routing for Visual Question Answering.
- \[[ACM MM 2023](https://dl.acm.org/doi/abs/10.1145/3503161.3548387)\] AI-VQA: Visual Question Answering based on Agent Interaction with Interpretability.
- \[[SIGIR 2023](https://dl.acm.org/doi/abs/10.1145/3539618.3591629)\] A Symmetric Dual Encoding Dense Retrieval Framework for Knowledge-Intensive Visual Question Answering.
- \[[ICMR 2023](https://dl.acm.org/doi/abs/10.1145/3591106.3592227)\] Explicit Knowledge Integration for Knowledge-Aware Visual Question Answering about Named Entities.
- \[[TMM 2023](https://ieeexplore.ieee.org/abstract/document/10163831/)\] Resolving Zero-shot and Fact-based Visual Question Answering via Enhanced Fact Retrieval.
- \[[ESA 2023](https://www.sciencedirect.com/science/article/pii/S0957417422017055)\] Image captioning for effective use of language models in knowledge-based visual question answering.
- \[[EMNLP 2022](https://arxiv.org/abs/2210.03809)\] Retrieval Augmented Visual Question Answering with Outside Knowledge.
- \[[EMNLP 2022](https://arxiv.org/abs/2210.10176)\] Entity-Focused Dense Passage Retrieval for Outside-Knowledge Visual Question Answering.
- \[[IJCKG 2022](https://arxiv.org/abs/2207.12888)\] LaKo: Knowledge-driven Visual Question Answering via Late Knowledge-to-Text Injection.
- \[[NeurIPS 2022](https://arxiv.org/abs/2206.01201)\] REVIVE: Regional Visual Representation Matters in Knowledge-Based Visual Question Answering.
- \[[CVPR 2022](https://arxiv.org/abs/2203.09138)\] MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering.
- \[[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Gao_Transform-Retrieve-Generate_Natural_Language-Centric_Outside-Knowledge_Visual_Question_Answering_CVPR_2022_paper.html)\] Transform-Retrieve-Generate: Natural Language-Centric Outside-Knowledge Visual Question Answering.
- \[[ECCV 2022](https://arxiv.org/abs/2206.01718)\] A-OKVQA: A Benchmark for Visual Question Answering Using World Knowledge.
- \[[ICCV 2022](https://arxiv.org/abs/2205.11501)\] VQA-GNN: Reasoning with Multimodal Semantic Graph for Visual Question Answering.
- \[[AAAI 2022](https://arxiv.org/abs/2203.02985)\] Dynamic Key-Value Memory Enhanced Multi-Step Graph Reasoning for Knowledge-Based Visual Question Answering.
- \[[AAAI 2022](https://arxiv.org/abs/2109.05014)\] An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA.
- \[[ACM MM 2022](https://arxiv.org/abs/2206.14989)\] A Unified End-to-End Retriever-Reader Framework for Knowledge-based VQA.
- \[[ACL 2022](https://arxiv.org/abs/2204.10448)\] Hypergraph Transformer: Weakly-Supervised Multi-hop Reasoning for Knowledge-based Visual Question Answering.
- \[[WWW 2022](https://arxiv.org/abs/2112.06888)\] Improving and Diagnosing Knowledge-Based Visual Question Answering via Entity Enhanced Knowledge Injection.
- \[[SITIS 2022](https://ieeexplore.ieee.org/abstract/document/10090046/)\] Multimodal Knowledge Reasoning for Enhanced Visual Question Answering.
- \[[KBS 2022](https://www.sciencedirect.com/science/article/pii/S0950705121009126)\] Fact-based visual question answering via dual-process system.
- \[[ISWC 2021](https://arxiv.org/abs/2107.05348)\] Zero-Shot Visual Question Answering Using Knowledge Graph.
- \[[ISWC 2021](https://arxiv.org/abs/2107.06325)\] Graphhopper: Multi-hop Scene Graph Reasoning for Visual Question Answering.
- \[[ACL 2021](https://aclanthology.org/2021.acl-short.60/)\] In Factuality: Efficient Integration of Relevant Facts for Visual Question Answering.
- \[[KDD 2021](https://arxiv.org/abs/2103.05568)\] Select, Substitute, Search: A New Benchmark for Knowledge-Augmented Visual Question Answering.
- \[[CVPR 2021](https://arxiv.org/abs/2012.11014)\] KRISP: Integrating Implicit and Symbolic Knowledge for Open-Domain Knowledge-Based VQA.
- \[[PR 2021](https://www.sciencedirect.com/science/article/pii/S003132032100340X)\] Knowledge base graph embedding module design for Visual question answering model.
- \[[SIGIR 2021](https://arxiv.org/abs/2105.03938)\] Passage Retrieval for Outside-Knowledge Visual Question Answering.
- \[[TNNLS 2021](https://ieeexplore.ieee.org/abstract/document/9199272/)\] Rich Visual Knowledge-Based Augmentation Network for Visual Question Answering.
- \[[COLING 2020](https://arxiv.org/abs/2111.08531)\] Towards Knowledge-Augmented Visual Question Answering.
- \[[arXiv 2020](https://arxiv.org/abs/2012.15484)\] Seeing is Knowing! Fact-based Visual Question Answering using Knowledge Graph Embeddings.
- \[[ACM MM 2020](https://dl.acm.org/doi/abs/10.1145/3394171.3413943)\] Boosting Visual Question Answering with Context-aware Knowledge Aggregation.
- \[[EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.44/)\] ConceptBert: Concept-Aware Representation for Visual Question Answering.
- \[[PR 2020](https://arxiv.org/abs/2009.00145)\] Cross-modal knowledge reasoning for knowledge-based visual question answering.
- \[[IJCAI 2020](https://arxiv.org/abs/2006.09073)\] Mucko: Multi-Layer Cross-Modal Knowledge Reasoning for Fact-based Visual Question Answering.
- \[[AAAI 2020](https://arxiv.org/abs/1910.10706)\] KnowIT VQA: Answering Knowledge-Based Questions about Videos.
- \[[AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4915)\] KVQA: Knowledge-Aware Visual Question Answering.
- \[[CVPR 2019](https://arxiv.org/abs/1906.00067)\] OK-VQA: Visual Question Answering Benchmark Requiring External Knowledge.
- \[[NeurIPS 2018](https://arxiv.org/abs/1811.00538)\] Out of the Box: Reasoning with Graph Convolution Nets for Factual Visual Question Answering.
- \[[ECCV 2018](https://arxiv.org/abs/1809.01124)\] Straight to the Facts: Learning Knowledge Base Retrieval for Factual Visual Question Answering.
- \[[CVPR 2018](https://arxiv.org/abs/1806.04860)\] Learning Visual Knowledge Memory Networks for Visual Question Answering.
- \[[KDD 2018](https://arxiv.org/abs/1805.09701)\] R-VQA: Learning Visual Relation Facts with Semantic Attention for Visual Question Answering.
- \[[TPAMI 2018](https://arxiv.org/abs/1606.05433)\] FVQA: Fact-Based Visual Question Answering.
- \[[IJCAI 2017](https://arxiv.org/abs/1511.02570)\] Explicit Knowledge-based Reasoning for Visual Question Answering.
- \[[CVPR 2016](https://arxiv.org/abs/1511.06973)\] Ask Me Anything: Free-Form Visual Question Answering Based on Knowledge from External Sources.


#### Visual Question Generation
- \[[ICTIR 2023](https://arxiv.org/abs/2306.16478)\] Pre-Training Multi-Modal Dense Retrievers for Outside-Knowledge Visual Question Answering.
- \[[ACM MM 2023](https://dl.acm.org/doi/abs/10.1145/3581783.3612536)\] Deconfounded Visual Question Generation with Causal Inference.
- \[[WACV 2023](https://arxiv.org/abs/2203.07890)\] K-VQG: Knowledge-aware Visual Question Generation for Common-sense Acquisition.
- \[[TCSVT 2022](https://ieeexplore.ieee.org/abstract/document/9826805/)\] Knowledge-Based Visual Question Generation.

  
#### Visual Dialog
- \[[ICMR 2023](https://dl.acm.org/doi/abs/10.1145/3591106.3592272)\] Knowledge-Aware Causal Inference Network for Visual Dialog.
- \[[TCSVT 2023](https://ieeexplore.ieee.org/abstract/document/9893870/)\] Heterogeneous Knowledge Network for Visual Dialog.
- \[[TMM 2023](https://ieeexplore.ieee.org/abstract/document/10147343/)\] Counterfactual Visual Dialog: Robust Commonsense Knowledge Learning from Unbiased Training.
- \[[TPAMI 2022](https://ieeexplore.ieee.org/abstract/document/9444809/)\] Context-Aware Graph Inference With Knowledge Distillation for Visual Dialog.
- \[[ACM MM 2022](https://arxiv.org/abs/2211.13235)\] Unified Multimodal Model with Unlikelihood Training for Visual Dialog.
- \[[arXiv 2022](https://arxiv.org/abs/2207.07934)\] Multimodal Dialog Systems with Dual Knowledge-enhanced Generative Pretrained Language Model.
- \[[CVPR 2022 Workshops](https://arxiv.org/abs/2204.04680)\] Reasoning with Multi-Structure Commonsense Knowledge in Visual Dialog.
- \[[EMNLP 2021](https://arxiv.org/abs/2004.06698)\] Reasoning Visual Dialog with Sparse Graph Learning and Knowledge Transfer.
- \[[ICME 2021](https://ieeexplore.ieee.org/abstract/document/9428279/)\] SKANet: Structured Knowledge-Aware Network for Visual Dialog.
- \[[ACM MM 2020](https://arxiv.org/abs/2008.04858)\] KBGN: Knowledge-Bridge Graph Network for Adaptive Vision-Text Reasoning in Visual Dialogue.
- \[[CVPR 2020](https://arxiv.org/abs/2004.02194)\] Iterative Context-Aware Graph Inference for Visual Dialog.


  
### Classification Tasks
<details>
   <summary>👈 🔎 Comparison </summary>
   
![IMGC](figures/imgc.jpg)
</details>

#### Image Classification
<details>
    <summary>👈 🔎 Benchmarks </summary>
<div align="center">
    <img src="figures/imgctab.jpg" width="45%" height="auto" />
</div>
</details>

- \[[arXiv 2023](https://arxiv.org/abs/2401.00608)\] Bringing Back the Context: Camera Trap Species Identification as Link Prediction on Multimodal Knowledge Graphs.
- \[[AAAI 2023](https://arxiv.org/abs/2207.01328)\] DUET: Cross-Modal Semantic Grounding for Contrastive Zero-Shot Learning.
- \[[KDD 2023](https://arxiv.org/abs/2306.08487)\] Recognizing Unseen Objects via Multimodal Intensive Knowledge Graph Propagation.
- \[[JWS 2023](https://arxiv.org/abs/2106.15047)\] Benchmarking knowledge-driven zero-shot learning.
- \[[ICTAI 2022](https://ieeexplore.ieee.org/abstract/document/10098100/)\] A Zero-shot Learning Method with a Multi-Modal Knowledge Graph.
- \[[KDD 2022](https://arxiv.org/abs/2206.03739)\] Disentangled Ontology Embedding for Zero-shot Learning.
- \[[Trans. Mach. Learn. Res. 2022](https://arxiv.org/abs/2006.10713)\] Zero-Shot Learning with Common Sense Knowledge Graphs.
- \[[Cogn. Comput. 2022](https://arxiv.org/abs/2012.06236)\] Improving Zero-Shot Learning Baselines with Commonsense Knowledge.
- \[[ICCV 2021 Workshop](https://openaccess.thecvf.com/content/ICCV2021W/GSP-CV/html/Wang_Zero-Shot_Learning_via_Contrastive_Learning_on_Dual_Knowledge_Graphs_ICCVW_2021_paper.html)\] Zero-Shot Learning via Contrastive Learning on Dual Knowledge Graphs.
- \[[WWW 2021](https://arxiv.org/abs/2102.07339)\] OntoZSL: Ontology-enhanced Zero-shot Learning.
- \[[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6626)\] Zero-Shot Ingredient Recognition by Multi-Relational Graph Convolutional Network.
- \[[KR 2020](https://arxiv.org/abs/2006.16917)\] Ontology-guided Semantic Composition for Zero-shot Learning.
- \[[AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4843)\] I Know the Relationships: Zero-Shot Action Recognition via Two-Stream Graph Convolutional Networks and Knowledge Graphs.
- \[[CVPR 2019](https://arxiv.org/abs/1805.11724)\] Rethinking Knowledge Graph Propagation for Zero-Shot Learning.
- \[[CVPR 2018](https://arxiv.org/abs/1803.08035)\] Zero-Shot Recognition via Semantic Embeddings and Knowledge Graphs.


#### Fake News Detection
- \[[ADMA 2023](https://link.springer.com/chapter/10.1007/978-3-031-46661-8_24)\] EmoKnow: Emotion- and Knowledge-Oriented Model for COVID-19 Fake News Detection.
- \[[TKDE 2023](https://arxiv.org/abs/2306.02137)\] Inconsistent Matters: A Knowledge-Guided Dual-Consistency Network for Multi-Modal Rumor Detection.
- \[[TOMCCAP 2021](https://dl.acm.org/doi/abs/10.1145/3451215)\] Knowledge-aware Multi-modal Adaptive Graph Convolutional Networks for Fake News Detection.
- \[[ICMR 2020](https://dl.acm.org/doi/abs/10.1145/3372278.3390713)\] Fake News Detection via Knowledge-driven Multimodal Graph Convolutional Networks.


#### Movie Genre Classification
- \[[ACM MM 2023](https://dl.acm.org/doi/abs/10.1145/3581783.3612085)\] Incorporating Domain Knowledge Graph into Multimodal Movie Genre Classification with Self-Supervised Attention and Contrastive Learning.


### Content Generation Tasks 
<details>
   <summary>👈 🔎 Case </summary>
<div align="center">
    <img src="figures/VGG.jpg" width="45%" height="auto" />
</div>
</details>

#### Image Captioning
- \[[ISPP 2023](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12707/127070W/Image-caption-generation-based-on-object-detection-and-knowledge-enhancement/10.1117/12.2680966.short)\] Image caption generation based on object detection and knowledge enhancement.
- \[[arXiv 2022](https://arxiv.org/abs/2210.04806)\] Generating image captions with external encyclopedic knowledge.
- \[[PR Lett. 2021](https://www.sciencedirect.com/science/article/pii/S0167865521000040)\] Image captioning with transformer and knowledge graph.
- \[[arXiv 2021](https://arxiv.org/abs/2107.11970)\] Boosting Entity-aware Image Captioning with Multi-modal Knowledge Graph.
- \[[arXiv 2020](https://arxiv.org/abs/2007.11690)\] Integrating Image Captioning with Rule-based Entity Masking.
- \[[AAAI 2020](https://arxiv.org/abs/2002.08277)\] When Radiology Report Generation Meets Knowledge Graph.
- \[[CIKM 2020](https://dl.acm.org/doi/abs/10.1145/3340531.3411948)\] Image Captioning with Internal and External Knowledge.
- \[[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6731)\] Joint Commonsense and Relation Reasoning for Image and Video Captioning.
- \[[arXiv 2019](https://arxiv.org/abs/1906.01290)\] Relational Reasoning using Prior Knowledge for Visual Captioning.
- \[[WACV 2019](https://arxiv.org/abs/1901.08942)\] Improving Image Captioning by Leveraging Knowledge Graphs.
- \[[TMM 2019](https://ieeexplore.ieee.org/abstract/document/8630068/)\] Know More Say Less: Image Captioning Based on Scene Graphs.
- \[[EMNLP 2018](https://arxiv.org/abs/1804.07889)\] Entity-aware Image Caption Generation.
- \[[arXiv 2015](https://arxiv.org/abs/1511.03292)\] From Images to Sentences through Scene Description Graphs using Commonsense Reasoning and Knowledge.

  
#### Visual Storytelling
- \[[TPAMI 2023](https://arxiv.org/abs/2203.05346)\] Knowledge-Enriched Attention Network With Group-Wise Semantic for Visual Storytelling.
- \[[AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16410)\] Imagine, Reason and Write: Visual Storytelling with Graph Knowledge and Relational Reasoning.
- \[[ACL 2021](https://arxiv.org/abs/2105.06950)\] Plot and Rework: Modeling Storylines for Visual Storytelling.
- \[[AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16410)\] Imagine, Reason and Write: Visual Storytelling with Graph Knowledge and Relational Reasoning.
- \[[AAAI 2021](https://arxiv.org/abs/2102.02963)\] Commonsense Knowledge Aware Concept Selection For Diverse and Informative Visual Storytelling.
- \[[AAAI 2020](https://arxiv.org/abs/1912.01496)\] Knowledge-Enriched Visual Storytelling.
- \[[IJCAI 2019](https://www.ijcai.org/Proceedings/2019/0744.pdf)\] Knowledgeable Storyteller: A Commonsense-Driven Generative Model for Visual Storytelling.

  
#### Conditional Text-to-Image Generation
- \[[arXiv 2023](https://arxiv.org/abs/2305.15753)\] T2TD: Text-3D Generation Model based on Prior Knowledge Guidance.
- \[[Pattern Recognition Letters 2023](https://www.sciencedirect.com/science/article/pii/S0167865523003501)\] Prior knowledge guided text to image generation.
- \[[TCSVT 2022](https://ieeexplore.ieee.org/abstract/document/9656731/)\] RiFeGAN2: Rich Feature Generation for Text-to-Image Synthesis From Constrained Prior Knowledge.
- \[[CSAE 2021](https://dl.acm.org/doi/abs/10.1145/3487075.3487155)\] Enriching Attributes from Knowledge Graph for Fine-grained Text-to-Image Synthesis.
- \[[ISCID 2021](https://ieeexplore.ieee.org/abstract/document/9679242/)\] KnHiGAN: Knowledge-enhanced Hierarchical Generative Adversarial Network for Fine-grained Text-to-Image Synthesis.
- \[[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_RiFeGAN_Rich_Feature_Generation_for_Text-to-Image_Synthesis_From_Prior_Knowledge_CVPR_2020_paper.html)\] RiFeGAN: Rich Feature Generation for Text-to-Image Synthesis From Prior Knowledge.
- \[[NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/d18f655c3fce66ca401d5f38b48c89af-Abstract.html)\] Learn, Imagine and Create: Text-to-Image Generation from Prior Knowledge.

  
#### Scene Graph Generation
- \[[ACM MM 2023](https://dl.acm.org/doi/abs/10.1145/3581783.3612249)\] Prior Knowledge-driven Dynamic Scene Graph Generation with Causal Inference.
- \[[WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Chen_More_Knowledge_Less_Bias_Unbiasing_Scene_Graph_Generation_With_Explicit_WACV_2023_paper.html)\] More Knowledge, Less Bias: Unbiasing Scene Graph Generation with Explicit Ontological Adjustment.
- \[[ICME 2022](https://ieeexplore.ieee.org/abstract/document/9859944/)\] Zero-Shot Scene Graph Generation with Knowledge Graph Completion.
- \[[ESWC 2022](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_6)\] Expressive Scene Graph Generation Using Commonsense Knowledge Infusion for Visual Understanding and Reasoning.
- \[[IJCAI 2020](https://arxiv.org/abs/2006.07585)\] Learning from the Scene and Borrowing from the Rich: Tackling the Long Tail in Scene Graph Generation.
- \[[ECCV 2020](https://arxiv.org/abs/2006.09623)\] Learning Visual Commonsense for Robust Scene Graph Generation.
- \[[ECCV 2020](https://arxiv.org/abs/2001.02314)\] Bridging Knowledge Graphs to Generate Scene Graphs.
- \[[CVPR 2019](https://arxiv.org/abs/1904.00560)\] Scene Graph Generation With External Knowledge and Image Reconstruction.
- \[[CVPR 2019](https://arxiv.org/abs/1903.03326)\] Knowledge-Embedded Routing Network for Scene Graph Generation.
- \[[IJCAI 2018](https://www.ijcai.org/Proceedings/2018/0132.pdf)\] Representation Learning for Scene Graph Completion via Jointly Structural and Visual Embedding.
- \[[CVPR 2018](https://arxiv.org/abs/1711.06640)\] Neural Motifs: Scene Graph Parsing With Global Context.

  
### Retrieval Tasks
<details>
   <summary>👈 🔎 Case </summary>
<div align="center">
    <img src="figures/CMR.jpg" width="50%" height="auto" />
</div>
</details>

#### Cross-Modal Retrieval
- \[[WSDM 2023](https://dl.acm.org/doi/abs/10.1145/3539597.3570481)\] AGREE: Aligning Cross-Modal Entities for Image-Text Retrieval Upon Vision-Language Pre-trained Models.
- \[[ACM MM 2023](https://dl.acm.org/doi/abs/10.1145/3581783.3613786)\] External Knowledge Dynamic Modeling for Image-text Retrieval.
- \[[TOMCCAP 2023](https://dl.acm.org/doi/abs/10.1145/3580501)\] MKVSE: Multimodal Knowledge Enhanced Visual-semantic Embedding for Image-text Retrieval.
- \[[TPAMI 2023](https://arxiv.org/abs/2206.08842)\] Entity-Graph Enhanced Cross-Modal Pretraining for Instance-Level Product Retrieval.
- \[[TMM 2023](https://ieeexplore.ieee.org/abstract/document/10169896/)\] Commonsense-Guided Semantic and Relational Consistencies for Image-Text Retrieval.
- \[[NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3379ce104189b72d5f7baaa03ae81329-Abstract-Conference.html)\] MACK: Multimodal Aligned Conceptual Knowledge for Unpaired Image-text Matching.
- \[[ECCV 2022](https://arxiv.org/abs/2208.09843)\] CODER: Coupled Diversity-Sensitive Momentum Contrastive Learning for Image-Text Retrieval.
- \[[ECCV 2020](https://arxiv.org/abs/2007.08883)\] Consensus-Aware Visual-Semantic Embedding for Image-Text Matching.
- \[[IJCAI 2019](https://www.ijcai.org/proceedings/2019/0720.pdf)\] Knowledge Aware Semantic Concept Expansion for Image-Text Matching.

  
#### Visual Referring Expressions & Grounding
- \[[CVPR 2023](https://arxiv.org/abs/2307.11558)\] Advancing Visual Grounding with Scene Knowledge: Benchmark and Method.
- \[[ACL 2023](https://aclanthology.org/2023.findings-acl.557/)\] Segment-Level and Category-Oriented Network for Knowledge-Based Referring Expression Comprehension.
- \[[EACL 2023](https://arxiv.org/abs/2302.09027)\] CK-Transformer: Commonsense Knowledge Enhanced Transformers for Referring Expression Comprehension.
- \[[AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20123)\] Improving Zero-Shot Phrase Grounding via Reasoning on External Knowledge and Spatial Relations.
- \[[ACM MM 2020](https://arxiv.org/abs/2006.01629)\] Give Me Something to Eat: Referring Expression Comprehension with Commonsense Knowledge.
- \[[CVPR 2018](https://arxiv.org/abs/1803.03879)\] Knowledge Aided Consistency for Weakly Supervised Phrase Grounding.

  
### KG-aware Mutli-modal Pre-training  
#### Structure Knowledge aware Pre-training
- \[[AAAI 2024](https://arxiv.org/abs/2305.06152)\] Structure-CLIP: Enhance Multi-modal Language Representations with Structure Knowledge.
- \[[ACM MM 2021](https://arxiv.org/abs/2108.07073)\] ROSITA: Enhancing Vision-and-Language Semantic Alignments via Cross- and Intra-modal Knowledge Integration.
- \[[ACL 2021](https://arxiv.org/abs/2101.00419)\] KM-BART: Knowledge Enhanced Multimodal BART for Visual Commonsense Generation.
- \[[AAAI 2021](https://arxiv.org/abs/2006.16934)\] ERNIE-ViL: Knowledge Enhanced Vision-Language Representations through Scene Graphs.
  
#### Knowledge Graph aware Pre-training
- \[[WWW 2023](https://arxiv.org/abs/2303.03922)\] Structure Pretraining and Prompt Tuning for Knowledge Graph Transfer.
- \[[CVPR 2023](https://arxiv.org/abs/2211.16504)\] Improving Commonsense in Vision-Language Models via Knowledge Graph Riddles.
- \[[ACM MM 2022](https://arxiv.org/abs/2209.07118)\] Align, Reason and Learn: Enhancing Medical Vision-and-Language Pre-training with Knowledge.
  
---

##  🌄🤖 Multi-modal Knowledge Graph (MM4KG)
<details>
    <summary>👈 🔎 N-MMKG Ontology </summary>

![MMKGOnto](figures/mmkgonto.jpg)
</details>

<details>
   <summary>👈 🔎 Taxonomy </summary>
<div align="center">
    <img src="figures/mmkgtask.jpg" width="90%" height="auto" />
</div>
</details>

### MMKG Resources

#### Public MMKGs

<details>
    <summary>👈 🔎 MMKG Overview </summary>

![MMKG](figures/mmkgtab.jpg)
</details>

- \[[arXiv 2023](https://arxiv.org/abs/2312.10417)\] M2ConceptBase: A Fine-grained Aligned Multi-modal Conceptual Knowledge Base.
- \[[arXiv 2023](https://arxiv.org/abs/2311.03783)\] Scene-Driven Multimodal Knowledge Graph Construction for Embodied AI.
- \[[EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.488/)\] VISTA: Visual-Textual Knowledge Graph Representation Learning.
- \[[ISWC 2023](https://link.springer.com/chapter/10.1007/978-3-031-47243-5_2)\] MMpedia: A Large-Scale Multi-modal Knowledge Graph.
- \[[ACM MM 2023](https://dl.acm.org/doi/abs/10.1145/3581783.3612266)\] TIVA-KG: A Multimodal Knowledge Graph with Text, Image, Video and Audio.
- \[[CIKM 2023](https://arxiv.org/abs/2308.04992)\] AspectMMKG: A Multi-modal Knowledge Graph with Aspect-aware Entities.
- \[[ISWC 2023](https://arxiv.org/abs/2307.16210)\] Rethinking Uncertainly Missing and Ambiguous Visual Modality in Multi-Modal Entity Alignment.
- \[[arXiv 2023](https://arxiv.org/abs/2302.06891)\] UKnow: A Unified Knowledge Protocol for Common-Sense Reasoning and Vision-Language Pre-training.
- \[[ICASSP 2023](https://arxiv.org/abs/2302.08774)\] Vision, Deduction and Alignment: An Empirical Study on Multi-Modal Knowledge Graph Alignment.
- \[[ICLR 2023](https://arxiv.org/abs/2210.00312)\] Multimodal Analogical Reasoning over Knowledge Graphs.
- \[[ACM MM 2022](https://dl.acm.org/doi/abs/10.1145/3503161.3548388)\] Relation-enhanced Negative Sampling for Multimodal Knowledge Graph Completion.
- \[[NAACL 2021](https://aclanthology.org/2021.naacl-demos.16/)\] RESIN: A Dockerized Schema-Guided Cross-document Cross-lingual Cross-media Information Extraction and Event Tracking System.
- \[[AAAI 2021](https://arxiv.org/abs/2009.13603)\] Visual Pivoting for (Unsupervised) Entity Alignment.
- \[[EMNLP 2021](https://arxiv.org/abs/2008.09150)\] VisualSem: a high-quality knowledge graph for vision and language.
- \[[BDR 2020](https://www.sciencedirect.com/science/article/pii/S2214579620300277)\] Richpedia: A Large-Scale, Comprehensive Multi-Modal Knowledge Graph.
- \[[ACL 2020](https://aclanthology.org/2020.acl-demos.11/)\] GAIA: A Fine-grained Multimedia Knowledge Extraction System.
- \[[ESWC 2019](https://link.springer.com/chapter/10.1007/978-3-030-21348-0_30)\] MMKG: Multi-modal Knowledge Graphs.
- \[[ISWC 2017](https://link.springer.com/chapter/10.1007/978-3-319-68204-4_8)\] IMGpedia: A Linked Dataset with Content-Based Analysis of Wikimedia Images.
- \[[TIP 2017](https://ieeexplore.ieee.org/abstract/document/7835116/)\] Robust ImageGraph: Rank-Level Feature Fusion for Image Search.
- \[[IJCV 2017](https://arxiv.org/abs/1602.07332)\] Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations.
- \[[ICCV 2013](http://openaccess.thecvf.com/content_iccv_2013/html/Chen_NEIL_Extracting_Visual_2013_ICCV_paper.html)\] NEIL: Extracting Visual Knowledge from Web Data.


#### MMKG Construction Methods

- \[[DASFAA 2022](https://link.springer.com/chapter/10.1007/978-3-031-00123-9_14)\] Visualizable or Non-visualizable? Exploring the Visualizability of Concepts in Multi-modal Knowledge Graph.
- \[[ACM MM 2020](https://dl.acm.org/doi/abs/10.1145/3394171.3416292)\] Towards Using Semantic-Web Technologies for Multi-Modal Knowledge Graph Construction.
- \[[AKBC 2019](https://arxiv.org/abs/1709.02314)\] Answering Visual-Relational Queries in Web-Extracted Knowledge Graphs.
- \[[IJCAI 2017](https://www.ijcai.org/Proceedings/2017/0238.pdf)\] Extracting Visual Knowledge from the Web with Multimodal Learning.
- \[[IJCAI 2017](https://arxiv.org/abs/1609.07028)\] Image-embodied Knowledge Representation Learning.
- \[[ECCV 2016](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_51)\] Visual Relationship Detection with Language Priors.
- \[[NAACL 2016](https://aclanthology.org/N16-1023.pdf)\] Stating the Obvious: Extracting Visual Common Sense Knowledge.
- \[[CVPR 2015](http://openaccess.thecvf.com/content_cvpr_2015/html/Johnson_Image_Retrieval_Using_2015_CVPR_paper.html)\] Image retrieval using scene graphs.
- \[[CVPR 2014](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Chen_Enriching_Visual_Knowledge_2014_CVPR_paper.html)\] Enriching Visual Knowledge Bases via Object Discovery and Segmentation.

### MMKG Acquisition
<details>
   <summary>👈 🔎 Case </summary>
<div align="center">
    <img src="figures/MMIE.jpg" width="50%" height="auto" />
</div>
</details>

#### Multi-modal Named Entity Recognition
<details>
    <summary>👈 🔎 Benchmarks </summary>
<div align="center">
    <img src="figures/mnertab.jpg" width="45%" height="auto" />
</div>
</details>

- \[[arXiv 2023](https://arxiv.org/abs/2305.08372)\] A Novel Framework for Multimodal Named Entity Recognition with Multi-level Alignments.
- \[[ACM MM 2023](https://arxiv.org/abs/2310.16822)\] Prompt Me Up: Unleashing the Power of Alignments for Multimodal Entity and Relation Extraction.
- \[[EMNLP 2023](https://arxiv.org/abs/2305.12212)\] Prompting ChatGPT in MNER: Enhanced Multimodal Named Entity Recognition with Auxiliary Refined Knowledge.
- \[[CIKM 2023](https://dl.acm.org/doi/abs/10.1145/3583780.3614967)\] MGICL: Multi-Grained Interaction Contrastive Learning for Multimodal Named Entity Recognition.
- \[[EMNLP 2022](https://arxiv.org/abs/2212.01612)\] Named Entity and Relation Extraction with Multi-Modal Retrieval.
- \[[ICME 2022](https://ieeexplore.ieee.org/abstract/document/9859972/)\] CAT-MNER: Multimodal Named Entity Recognition with Knowledge-Refined Cross-Modal Attention.
- \[[DASFAA 2022](https://link.springer.com/chapter/10.1007/978-3-031-00129-1_24)\] PromptMNER: Prompt-Based Entity-Related Visual Clue Extraction and Integration for Multimodal Named Entity Recognition.
- \[[NAACL 2022](https://arxiv.org/abs/2112.06482)\] ITA: Image-Text Alignments for Multi-Modal Named Entity Recognition.
- \[[TASLP 2023](https://ieeexplore.ieee.org/abstract/document/10413604/)\] Enhancing Multimodal Entity and Relation Extraction with Variational Information Bottleneck.
- \[[TCSS 2023](https://ieeexplore.ieee.org/abstract/document/10223602/)\] Dynamic Graph Construction Framework for Multimodal Named Entity Recognition in Social Media.
- \[[SIGIR 2022](https://arxiv.org/abs/2205.02357)\] Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion.
- \[[WSDM 2023](https://dl.acm.org/doi/abs/10.1145/3539597.3570485)\] Reducing the Bias of Visual Objects in Multimodal Named Entity Recognition.
- \[[AAAI 2023](https://arxiv.org/abs/2211.14739)\] MNER-QG: An End-to-End MRC Framework for Multimodal Named Entity Recognition with Query Grounding.
- \[[TAI 2023](https://ieeexplore.ieee.org/abstract/document/10292546/)\] Text-Image Scene Graph Fusion for Multi-Modal Named Entity Recognition.
- \[[CIKM 2023](https://dl.acm.org/doi/abs/10.1145/3583780.3614975)\] MPMRC-MNER: A Unified MRC framework for Multimodal Named Entity Recognition based Multimodal Prompt.
- \[[ACM MM 2022](https://dl.acm.org/doi/abs/10.1145/3503161.3548228)\] Learning from Different text-image Pairs: A Relation-enhanced Graph Convolutional Network for Multimodal NER.
- \[[IJMLC 2023](https://link.springer.com/article/10.1007/s13042-023-02037-8)\] ‘What’and ‘Where’both matter: dual cross-modal graph convolutional networks for multimodal named entity recognition.
- \[[EMNLP 2022](https://aclanthology.org/2022.findings-emnlp.473/)\] Entity-level Interaction via Heterogeneous Graph for Multimodal Named Entity Recognition.
- \[[ACM MM 2022](https://dl.acm.org/doi/abs/10.1145/3503161.3548427)\] Query Prior Matters: A MRC Framework for Multimodal Named Entity Recognition.
- \[[COLING 2022](https://arxiv.org/abs/2208.11039)\] Flat Multi-modal Interaction Transformer for Named Entity Recognition.
- \[[NAACL 2022](https://arxiv.org/abs/2205.03521)\] Good Visual Guidance Make A Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction.
- \[[EMNLP 2022](https://aclanthology.org/2022.emnlp-main.422/)\] A Span-based Multimodal Variational Autoencoder for Semi-supervised Multimodal Named Entity Recognition.
- \[[WSDM 2022](https://dl.acm.org/doi/abs/10.1145/3488560.3498475)\] MAF: A General Matching and Alignment Framework for Multimodal Named Entity Recognition.
- \[[Appl. Intell. 2022](https://link.springer.com/article/10.1007/s10489-021-02546-5)\] UAMNer: uncertainty-aware multimodal named entity recognition in social media posts.
- \[[IPM 2024](https://www.sciencedirect.com/science/article/pii/S0306457323002832)\] Multi-granularity cross-modal representation learning for named entity recognition on social media.
- \[[IEICE Trans. Inf. Syst. 2023](https://search.ieice.org/bin/summary.php?id=e106-d_4_545)\] Multimodal Named Entity Recognition with Bottleneck Fusion and Contrastive Learning.
- \[[AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17687)\] Multi-modal Graph Fusion for Named Entity Recognition with Targeted Visual Guidance.
- \[[ACL 2020](https://aclanthology.org/2020.acl-main.306/)\] Improving Multimodal Named Entity Recognition via Entity Span Detection with Unified Multimodal Transformer.
- \[[COLING 2020](https://aclanthology.org/2020.coling-main.168/)\] RIVA: A Pre-trained Tweet Multimodal Model Based on Text-image Relation for Multimodal NER.
- \[[DASFAA 2021](https://link.springer.com/chapter/10.1007/978-3-030-73197-7_12)\] Multimodal Named Entity Recognition with Image Attributes and Image Knowledge.
- \[[ACM MM 2021](https://dl.acm.org/doi/abs/10.1145/3474085.3476968)\] Multimodal Relation Extraction with Efficient Graph Alignment.
- \[[AAAI 2021](https://arxiv.org/abs/2102.02967)\] RpBERT: A Text-image Relation Propagation-based BERT Model for Multimodal NER.
- \[[ACM MM 2020](https://dl.acm.org/doi/abs/10.1145/3394171.3413650)\] Multimodal Representation with Embedded Visual Guiding Objects for Named Entity Recognition in Social Media Posts.


#### Multi-modal Relation Extraction
<details>
    <summary>👈 🔎 Benchmarks </summary>
<div align="center">
    <img src="figures/mmretab.jpg" width="45%" height="auto" />
</div>
</details>

- \[[ACM MM 2023](https://arxiv.org/abs/2310.16822)\] Prompt Me Up: Unleashing the Power of Alignments for Multimodal Entity and Relation Extraction.
- \[[EMNLP 2023](https://arxiv.org/abs/2305.13805)\] Towards Zero-shot Relation Extraction in Web Mining: A Multimodal Approach with Relative XML Path.
- \[[ACL 2023](https://arxiv.org/abs/2305.16166)\] Multimodal Relation Extraction with Cross-Modal Retrieval and Synthesis.
- \[[arXiv 2023](https://arxiv.org/abs/2304.02328)\] Enhancing Multimodal Entity and Relation Extraction with Variational Information Bottleneck.
- \[[ACL 2023](https://arxiv.org/abs/2306.11020)\] Dual-Gated Fusion with Prefix-Tuning for Multi-Modal Relation Extraction.
- \[[ACL 2023](https://arxiv.org/abs/2305.11719)\] Information Screening whilst Exploiting! Multimodal Relation Extraction with Feature Denoising and Multimodal Topic Modeling.
- \[[AAAI 2023 (Student Abstract)](https://arxiv.org/abs/2211.07504)\] On Analyzing the Role of Image for Visual-Enhanced Relation Extraction.
- \[[SIGIR 2022](https://arxiv.org/abs/2205.02357)\] Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion.
- \[[NAACL 2022](https://arxiv.org/abs/2205.03521)\] Good Visual Guidance Make A Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction.
- \[[EMNLP 2022](https://arxiv.org/abs/2212.01612)\] Named Entity and Relation Extraction with Multi-Modal Retrieval.
- \[[ACM MM 2021](https://dl.acm.org/doi/abs/10.1145/3474085.3476968)\] Multimodal Relation Extraction with Efficient Graph Alignment.
- \[[ICME 2021](https://ieeexplore.ieee.org/abstract/document/9428274/)\] MNRE: A Challenge Multimodal Dataset for Neural Relation Extraction with Visual Evidence in Social Media Posts.

#### Multi-modal Event Extraction
<details>
    <summary>👈 🔎 Benchmarks </summary>
<div align="center">
    <img src="figures/mmeetab.jpg" width="45%" height="auto" />
</div>
</details>

 Image-Text:
- \[[ACM MM 2023](https://arxiv.org/abs/2306.08966)\] Training Multimedia Event Extraction With Generated Images and Captions.
- \[[ACM MM 2022](https://dl.acm.org/doi/abs/10.1145/3503161.3548132)\] Multimedia Event Extraction From News With a Unified Contrastive Learning Framework.
- \[[CVPR 2022](https://arxiv.org/abs/2201.05078)\] CLIP-Event: Connecting Text and Images with Event Structures.
- \[[ACL 2020](https://arxiv.org/abs/2005.02472)\] Cross-media Structured Common Space for Multimedia Event Extraction.


 Video-Text:
- \[[FUSION 2023](https://ieeexplore.ieee.org/abstract/document/10224213/)\] Few-shot Domain-Adaptative Visually-fused Event Detection from Text.
- \[[EMNLP 2023](https://aclanthology.org/2023.emnlp-main.103/)\] Three Stream Based Multi-level Event Contrastive Learning for Text-Video Event Extraction.
- \[[DASFAA 2023](https://link.springer.com/chapter/10.1007/978-3-031-30675-4_51)\] Cross-Modal Contrastive Learning for Event Extraction.
- \[[EMNLP 2021](https://arxiv.org/abs/2109.12776)\] Joint Multimedia Event Extraction from Video and Article.


### MMKG Fusion

#### Multi-modal Entity Alignment
<details>
    <summary>👈 🔎 Benchmarks </summary>
<div align="center">
    <img src="figures/mmeatab.jpg" width="45%" height="auto" />
</div>
</details>

- \[[ICLR 2024](https://arxiv.org/abs/2305.14651)\] Revisit and Outstrip Entity Alignment: A Perspective of Generative Models.
- \[[arXiv 2024](https://arxiv.org/abs/2401.17859)\] Towards Semantic Consistency: Dirichlet Energy Driven Robust Multi-Modal Entity Alignment.
- \[[EMNLP 2023](https://arxiv.org/abs/2310.06365)\] Multi-Modal Knowledge Graph Transformer Framework for Multi-Modal Entity Alignment.
- \[[arXiv 2023](https://arxiv.org/abs/2310.05364)\] Universal Multi-modal Entity Alignment via Iteratively Fusing Modality Similarity Paths.
- \[[ACM MM 2023](https://arxiv.org/abs/2212.14454)\] MEAformer: Multi-modal Entity Alignment Transformer for Meta Modality Hybrid.
- \[[ACM MM 2023](https://dl.acm.org/doi/abs/10.1145/3581783.3612151)\] Cross-Modal Graph Attention Network for Entity Alignment.
- \[[ACM MM 2023](https://dl.acm.org/doi/abs/10.1145/3581783.3611778)\] PSNEA: Pseudo-Siamese Network for Entity Alignment between Multi-modal Knowledge Graphs.
- \[[ISWC 2023](https://arxiv.org/abs/2307.16210)\] Rethinking Uncertainly Missing and Ambiguous Visual Modality in Multi-Modal Entity Alignment.
- \[[WWW 2023](https://arxiv.org/abs/2304.01563)\] Attribute-Consistent Knowledge Graph Representation Learning for Multi-Modal Entity Alignment.
- \[[ICASSP 2023](https://arxiv.org/abs/2302.08774)\] Vision, Deduction and Alignment: An Empirical Study on Multi-Modal Knowledge Graph Alignment.
- \[[COLING 2023](https://arxiv.org/abs/2209.00891)\] Multi-modal Contrastive Representation Learning for Entity Alignment.
- \[[DSE 2023](https://link.springer.com/chapter/10.1007/978-3-031-25198-6_20)\] Probing the Impacts of Visual Context in Multimodal Entity Alignment.
- \[[KSEM 2023](https://link.springer.com/chapter/10.1007/978-3-031-40286-9_18)\] Enhanced Entity Interaction Modeling for Multi-Modal Entity Alignment.
- \[[Inf. Fusion 2023](https://www.sciencedirect.com/science/article/pii/S1566253523002518)\] MMIEA: Multi-modal Interaction Entity Alignment model for knowledge graphs.
- \[[ESWA 2023](https://www.sciencedirect.com/science/article/pii/S0957417423008655)\] Leveraging multimodal features for knowledge graph entity alignment based on dynamic self-attention networks.
- \[[Neurocomput. 2022](https://www.sciencedirect.com/science/article/pii/S0925231222006099)\] MultiJAF: Multi-modal joint entity alignment framework for multi-modal knowledge graph.
- \[[KDD 2022](https://dl.acm.org/doi/abs/10.1145/3534678.3539244)\] Multi-modal Siamese Network for Entity Alignment.
- \[[Neurocomput. 2021](https://arxiv.org/abs/2106.03619)\] Multi-modal entity alignment in hyperbolic space.
- \[[AAAI 2021](https://arxiv.org/abs/2009.13603)\] Visual Pivoting for (Unsupervised) Entity Alignment.
- \[[KSEM 2020](https://link.springer.com/chapter/10.1007/978-3-030-55130-8_12)\] MMEA: Entity Alignment for Multi-modal Knowledge Graph.


#### Multi-modal Entity Linking & Disambiguation
<details>
    <summary>👈 🔎 Benchmarks </summary>
<div align="center">
    <img src="figures/mmeltab.jpg" width="45%" height="auto" />
</div>
</details>

- \[[arXiv 2023](https://arxiv.org/abs/2312.11816)\] A Dual-way Enhanced Framework from Text Matching Point of View for Multimodal Entity Linking.
- \[[arXiv 2023](https://arxiv.org/abs/2306.12725)\] Generative Multimodal Entity Linking.
- \[[ACL 2023](https://arxiv.org/abs/2305.17337)\] Benchmarking Diverse-Modal Entity Linking with Generative Models.
- \[[arXiv 2022](https://arxiv.org/abs/2211.00732)\] Kuaipedia: a Large-scale Multi-modal Short-video Encyclopedia.
- \[[KDD 2023](https://arxiv.org/abs/2307.09721)\] Multi-Grained Multimodal Interaction Network for Entity Linking.
- \[[UAI 2023](https://proceedings.mlr.press/v216/yang23d.html)\] MMEL: A Joint Learning Framework for Multi-Mention Entity Linking.
- \[[CVPR 2022](https://arxiv.org/abs/2111.02387)\] An Empirical Study of Training End-to-End Vision-and-Language Transformers.
- \[[EMNLP 2022](https://aclanthology.org/2022.findings-emnlp.230/)\] Multimodal Knowledge Learning for Named Entity Disambiguation.
- \[[SIGIR 2022](https://dl.acm.org/doi/abs/10.1145/3477495.3531867)\] Multimodal Entity Linking with Gated Hierarchical Fusion and Contrastive Training.
- \[[ICLR 2021](https://arxiv.org/abs/2010.00904)\] Autoregressive Entity Retrieval.
- \[[NeurIPS 2021](https://arxiv.org/abs/2107.07651)\] Align before Fuse: Vision and Language Representation Learning with Momentum Distillation.
- \[[ECIR 2020](https://arxiv.org/abs/2104.03236)\] Multimodal Entity Linking for Tweets.
- \[[EMNLP 2020](https://arxiv.org/abs/1911.03814)\] Scalable Zero-shot Entity Linking with Dense Entity Retrieval.
- \[[ACL 2018](https://aclanthology.org/P18-1186/)\] Multimodal Named Entity Disambiguation for Noisy Social Media Posts.


### MMKG Inference

#### Multi-modal Knowledge Graph Completion
<details>
    <summary>👈 🔎 Benchmarks </summary>
<div align="center">
    <img src="figures/mkgctab.jpg" width="45%" height="auto" />
</div>
</details>

- \[[EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.488/)\] VISTA: Visual-Textual Knowledge Graph Representation Learning.
- \[[ACM MM 2023](https://dl.acm.org/doi/abs/10.1145/3581783.3612266)\] TIVA-KG: A Multimodal Knowledge Graph with Text, Image, Video and Audio.
- \[[TOMCCAP 2023](https://dl.acm.org/doi/abs/10.1145/3545573)\] Hyper-node Relational Graph Attention Network for Multi-modal Knowledge Graph Completion.
- \[[arXiv 2023](https://arxiv.org/abs/2307.03591)\] Structure Guided Multi-modal Pre-trained Transformer for Knowledge Graph Reasoning.
- \[[TKDE 2023](https://ieeexplore.ieee.org/abstract/document/9942333/)\] Contrastive Multi-Modal Knowledge Graph Representation Learning.
- \[[ICDE 2023](https://ieeexplore.ieee.org/abstract/document/10231041/)\] Multimodal Biological Knowledge Graph Completion via Triple Co-Attention Mechanism.
- \[[ECML/PKDD 2022](https://link.springer.com/chapter/10.1007/978-3-031-26390-3_11)\] MULTIFORM: Few-Shot Knowledge Graph Completion via Multi-modal Contexts.
- \[[WWW 2023](https://arxiv.org/abs/2303.10816)\] IMF: Interactive Multimodal Fusion Model for Link Prediction.
- \[[SIGIR 2022](https://arxiv.org/abs/2205.02357)\] Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion.
- \[[ACM MM 2022](https://dl.acm.org/doi/abs/10.1145/3503161.3548388)\] Relation-enhanced Negative Sampling for Multimodal Knowledge Graph Completion.
- \[[IJCNN 2023](https://arxiv.org/abs/2304.11618)\] Modality-Aware Negative Sampling for Multi-modal Knowledge Graph Embedding.
- \[[Appl. Intell. 2022](https://link.springer.com/article/10.1007/s10489-021-02693-9)\] MMKRL: A robust embedding approach for multi-modal knowledge graph representation learning.
- \[[EMNLP 2022](https://arxiv.org/abs/2210.08821)\] MoSE: Modality Split and Ensemble for Multimodal Knowledge Graph Completion.
- \[[NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ffdb280e7c7b4c4af30e04daf5a84b98-Abstract-Conference.html)\] OTKGE: Multi-modal Knowledge Graph Embeddings via Optimal Transport.
- \[[KDD 2022](https://arxiv.org/abs/2209.07084)\] Knowledge Graph Completion with Pre-trained Multimodal Transformer and Twins Negative Sampling.
- \[[ACM MM 2021](https://arxiv.org/abs/2209.07084)\] Is Visual Context Really Helpful for Knowledge Graph? A Representation Learning Perspective.
- \[[IJCNN 2019](https://ieeexplore.ieee.org/abstract/document/8852079/)\] Multimodal Data Enhanced Representation Learning for Knowledge Graphs.
- \[[EMNLP 2018](https://arxiv.org/abs/1809.01341)\] Embedding Multimodal Relational Data for Knowledge Base Completion.
- \[[NAACL 2018](https://aclanthology.org/S18-2027/)\] A Multimodal Translation-Based Approach for Knowledge Graph Representation Learning.
- \[[IJCAI 2017](https://arxiv.org/abs/1609.07028)\] Image-embodied Knowledge Representation Learning.


#### Multi-modal Knowledge Graphs Reasoning
- \[[ICLR 2023](https://arxiv.org/abs/2210.00312)\] Multimodal Analogical Reasoning over Knowledge Graphs.
- \[[arXiv 2023](https://arxiv.org/abs/2306.10345)\] Do as I can, not as I get: Topology-aware multi-hop reasoning on multi-modal knowledge graphs.
- \[[ICDE 2023](https://arxiv.org/abs/2209.01416)\] MMKGR: Multi-hop Multi-modal Knowledge Graph Reasoning.

### MMKG-driven Tasks
<details>
   <summary>👈 🔎 Case </summary>
<div align="center">
    <img src="figures/mmkg4mm.jpg" width="50%" height="auto" />
</div>
</details>

#### Retrieval
##### Image Retrieval:
- \[[EMNLP 2021](https://arxiv.org/abs/2008.09150)\] VisualSem: a high-quality knowledge graph for vision and language.
- \[[ISWC 2017](https://link.springer.com/chapter/10.1007/978-3-319-68204-4_8)\] IMGpedia: A Linked Dataset with Content-Based Analysis of Wikimedia Images.
- \[[TIP 2017](https://ieeexplore.ieee.org/abstract/document/7835116/)\] Robust ImageGraph: Rank-Level Feature Fusion for Image Search.

##### Cross-modal Retrieval:
- \[[AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25445)\] Multi-Modal Knowledge Hypergraph for Diverse Image Retrieval.
- \[[TOMCCAP 2023](https://dl.acm.org/doi/abs/10.1145/3580501)\] MKVSE: Multimodal Knowledge Enhanced Visual-semantic Embedding for Image-text Retrieval.
- \[[arXiv 2022](https://arxiv.org/abs/2206.13163)\] Endowing Language Models with Multimodal Knowledge Graph Representations.
- \[[DASFAA 2021](https://link.springer.com/chapter/10.1007/978-3-030-73197-7_12)\] Multimodal Named Entity Recognition with Image Attributes and Image Knowledge.

#### Reasoning & Generation
- \[[arXiv 2023](https://arxiv.org/abs/2312.10417)\] M2ConceptBase: A Fine-grained Aligned Multi-modal Conceptual Knowledge Base.
- \[[JIIS 2023](https://link.springer.com/article/10.1007/s10844-023-00812-1)\] Self-supervised opinion summarization with multi-modal knowledge graph.
- \[[TMM 2023](https://arxiv.org/abs/2107.11970)\] Boosting Entity-aware Image Captioning with Multi-modal Knowledge Graph.


#### Pre-training
##### Triple-level:
- \[[NeurIPS 2022](https://arxiv.org/abs/2210.08901)\] Contrastive Language-Image Pre-Training with Knowledge Graphs.

##### Graph-level:
- \[[NeurIPS 2023](https://arxiv.org/abs/2309.13625)\] GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph.
- \[[arXiv 2023](https://arxiv.org/abs/2302.06891)\] UKnow: A Unified Knowledge Protocol for Common-Sense Reasoning and Vision-Language Pre-training.

#### AI for Science
- \[[AAAI 2023 (Student Abstract)](https://arxiv.org/abs/2207.10080)\] Multi-Modal Protein Knowledge Graph Construction and Applications.
- \[[ICDE 2023](https://ieeexplore.ieee.org/abstract/document/10231041/)\] Multimodal Biological Knowledge Graph Completion via Triple Co-Attention Mechanism.
- \[[Nat. Mach. Intell. 2023](https://www.nature.com/articles/s42256-023-00654-0)\] Knowledge graph-enhanced molecular contrastive learning with functional prompt.
- \[[AAAI 2022](https://arxiv.org/abs/2112.00544)\] Molecular Contrastive Learning with Chemical Element Knowledge Graph.
- \[[ICLR 2022](https://arxiv.org/abs/2201.11147)\] OntoProtein: Protein Pretraining With Gene Ontology Embedding.
- \[[IJCAI 2020](https://xuanlin1991.github.io/files/publications/ijcai20.pdf)\] KGNN: Knowledge Graph Neural Network for Drug-Drug Interaction Prediction.

#### Industry Application
<details>
   <summary>👈 🔎 Case </summary>
<div align="center">
    <img src="figures/mmkg4indus.jpg" width="50%" height="auto" />
</div>
</details>

- \[[PRCV 2023](https://dl.acm.org/doi/abs/10.1145/3503161.3548273)\] Enhancing Recommender System with Multi-modal Knowledge Graph.
- \[[ACL 2023](https://aclanthology.org/2023.acl-industry.16/)\] FashionKLIP: Enhancing E-Commerce Image-Text Retrieval with Fashion Multi-Modal Conceptual Knowledge Graph.
- \[[ACM MM 2022](https://dl.acm.org/doi/abs/10.1145/3503161.3548273)\] Cross-modal Knowledge Graph Contrastive Learning for Machine Learning Method Recommendation.
- \[[CIKM 2021](https://dl.acm.org/doi/abs/10.1145/3459637.3481983)\] AliMe MKG: A Multi-modal Knowledge Graph for Live-streaming E-commerce.
- \[[ACM MM 2021](https://dl.acm.org/doi/abs/10.1145/3474085.3475648)\] Knowledge Perceived Multi-modal Pretraining in E-commerce.
- \[[CIKM 2020](https://dl.acm.org/doi/abs/10.1145/3340531.3411947)\] Multi-modal Knowledge Graphs for Recommender Systems.





## Contribution
### 👥 Contributors

<a href="https://github.com/zjukg/KG-MM-Survey/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zjukg/KG-MM-Survey" />
</a>

### 🎉 Contributing ( welcome ! )

- ✨ Add a new paper or update an existing KG4MM or MM4KG paper.
- 🧐 Use the same format as existing entries to describe the work.
- 😄 A very brief explanation why you think a paper should be added or updated is recommended (Not Neccessary) via **`Adding Issues`** or **`Pull Requests`**.

**Don't worry if you put something wrong, they will be fixed for you. Just feel free to contribute and promote your awesome work here! 🤩 We'll get back to you in time ~ 😉**

---

### 🔖 Contact

> 📫 zhuo.chen@zju.edu.cn 


### 🤝 Cite:
If this Repo is helpful to you, please consider citing our paper. We would greatly appreciate it :)
```bigquery
@article{chen2024knowledge,
  author       = {Zhuo Chen and
                  Yichi Zhang and
                  Yin Fang and
                  Yuxia Geng and
                  Lingbing Guo and
                  Xiang Chen and
                  Qian Li and
                  Wen Zhang and
                  Jiaoyan Chen and
                  Yushan Zhu and
                  Jiaqi Li and
                  Xiaoze Liu and
                  Jeff Z. Pan and
                  Ningyu Zhang and
                  Huajun Chen},
  title        = {Knowledge Graphs Meet Multi-Modal Learning: {A} Comprehensive Survey},
  journal      = {CoRR},
  volume       = {abs/2402.05391},
  year         = {2024}
}
```

![Star History Chart](https://api.star-history.com/svg?repos=zjukg/KG-MM-Survey&type=Date)


