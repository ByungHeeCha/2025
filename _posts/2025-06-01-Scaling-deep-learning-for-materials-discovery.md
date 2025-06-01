---
layout: distill
title: AI810 Blog Post (20244491) Scaling deep learning for materials discovery
description: Deep learning has transformed fields like language and vision—can it do the same for materials discovery? In this blog post, we explore how scaling graph neural networks with massive first-principles data unlocks unprecedented generalization capabilities in inorganic crystal prediction. Building on over 48,000 known stable materials, the model identifies 2.2 million new structures, many beyond human chemical intuition. We discuss how this order-of-magnitude leap in materials discovery efficiency enables rapid screening for energy and electronic applications, and how the resulting models pave the way for accurate interatomic potentials and zero-shot predictions of complex properties like ionic conductivity.
date: 2025-06-01
future: true
htmlwidgets: true
hidden: false

authors:
  - name: ByungHee Cha
    url: "https://sites.google.com/view/byungheecha/home?authuser=0"
    affiliations:
      name: KAIST
# must be the exact same name as your blogpost
bibliography: 2025-06-01-Scaling-deep-learning-for-materials-discovery.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
    - name: Introduction
    - name: Method
    - name: Experiments
      subsections:
      - name: Python (Code) Concept
      - name: Language Concept
        subsections:
        - name: French Concept
        - name: Simplfied/Traditional Chinese Concept
        - name: Arabic Concept
      - name: Areas where CAV Excels and Does Not
    - name: Discussion
      subsections:
      - name: Is PSA-induced CAV the same as IT-induced?
      - name: Can expand to multi-behavior steering?
    - name: Conclusion


# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

# Introduction

The discovery of novel, stable inorganic materials underpins progress in a wide range of technologies—from batteries and solar cells to semiconductors and solid-state electrolytes. Traditionally, this process has relied on expensive and time-consuming trial-and-error experiments or computational methods such as density functional theory (DFT), which—despite their accuracy—remain computationally intensive and limited in scope. Over the past decade, databases like the Inorganic Crystal Structure Database (ICSD), the Materials Project, and OQMD have grown to include tens of thousands of stable materials. Yet, these repositories represent only a fraction of the possible chemical space.

As deep learning continues to demonstrate powerful generalization across domains like language and vision, a natural question arises: can these techniques similarly accelerate scientific discovery in materials science? In this work, we present a large-scale application of graph neural networks (GNNs) to materials exploration, using an active learning framework to dramatically expand the known landscape of stable inorganic crystals. Our method—GNoME (Graph Networks for Materials Exploration)—iteratively trains on DFT-labeled data and proposes candidate structures, improving prediction accuracy and efficiency through each cycle.

By scaling both model size and training data, GNoME enables the discovery of over 2.2 million previously unknown structures, including many with complex compositions beyond the reach of earlier approaches. This order-of-magnitude leap in coverage represents one of the most significant expansions in stable materials to date. Beyond discovery, we show that the GNoME-generated dataset supports downstream applications, including training interatomic potentials and predicting ionic conductivity with high fidelity—demonstrating how large-scale deep learning can reshape the tools and reach of materials science.

---

# Method

<img src="{{'assets/img/2025-06-01-Scaling-deep-learning-for-materials-discovery/Gnome_pipeline.png' | relative_url}}" alt="GNoME Pipeline Illustration" width="70%">

<div class="caption">
  <strong>Figure:</strong> The pipeline of GNOME.
</div>

<!-- ![The pipeline of GNOME](assets/img/2025-06-01-Scaling-deep-learning-for-materials-discovery/Gnome_pipeline.png) -->

### Generation and Filtration at Scale

The vastness of crystal space makes unbiased sampling intractable. Historically, guided searches have relied on human chemical intuition and restrictive substitution schemes—effective for narrowing the search, but inherently biased and limited in diversity. With neural networks like GNoME, we unlock the ability to perform large-scale candidate generation and filtration in a data-driven and automated manner.

Two complementary frameworks underpin this approach:

1. **Structure-based framework**: Candidates are generated by modifying known crystals, vastly expanded using symmetry-aware partial substitutions (SAPS). Ionic substitution probabilities are adjusted to prioritize unexplored compositions, allowing over 10⁹ unique candidates throughout active learning cycles. These structures are then filtered by GNoME using volume-based test-time augmentation and deep ensemble uncertainty estimates.
2. **Composition-based framework**: Structures are generated from reduced chemical formulas using relaxed oxidation-state balancing. Predictions from composition-only GNoME models guide AIRSS-based structural sampling, evaluating 100 random structures per composition via DFT.

In both cases, neural predictions guide the selection of candidates for DFT validation, where success is measured by the number of stable materials discovered and precision of the predictions.


### GNoME: Scalable GNN Energy Predictor

GNoME models are graph neural networks predicting total energy from atomic graphs. All models are trained on a curated snapshot of \~69,000 materials from the 2018 Materials Project. Using improved architectures and message normalization, GNoME reduces MAE from the previous 28 meV/atom to 21 meV/atom.

The architecture is fixed for consistency, with all scaling experiments building on this foundation. Deep ensembles are used to improve both performance and uncertainty calibration. These GNNs form the backbone of both structural and compositional candidate filtering.


### Active Learning Pipeline

Each round of active learning improves GNoME’s accuracy and expands the candidate space:

* Filtered structures are evaluated with DFT
* DFT results are used to retrain models
* New candidates are generated using the updated models

Initially, hit rates were below 6% for structural and 3% for compositional predictions. After six rounds, final models reach:

* **11 meV/atom** MAE on relaxed structures
* **>80% hit rate** (structural)
* **>33% hit rate** (compositional)

This iterative learning loop is key to unlocking the benefits of model scaling.


### Neural Scaling and Generalization

As more data is added, GNoME models follow neural scaling laws: prediction error decreases as a power law with dataset size. Unlike other domains (e.g., NLP), materials science allows for continued data generation through simulation—removing a core limitation.

We observe generalization to out-of-distribution tasks, such as predicting energies of high-energy structures generated via random search. Structural models trained purely on substitution-generated structures still perform well, highlighting strong extrapolation ability from scale.


### Substitution and SAPS for Diversity

Substitution-based candidate generation is vastly expanded:

* Modified the original ionic substitution model to prioritize novelty
* Removed restrictions on rare elements and enabled more compositions by lifting charge-balancing and threshold constraints

SAPS extends these ideas by allowing **partial substitutions** on symmetry-equivalent sites. This leads to discovery of new structure types (e.g., double perovskites) while avoiding combinatorial explosion.

SAPS contributed to **>230,000** out of the **381,000** stable materials discovered by GNoME, showing its central importance.


### Composition Framework and AIRSS

For compositions without known structures:

* Relaxed oxidation-state constraints enable richer chemical spaces
* AIRSS is used to generate plausible structures
* 100 DFT relaxations per composition increase the chance of finding stable configurations

This framework adds flexibility and diversity beyond prototype-based approaches, though it remains bottlenecked by AIRSS convergence issues, particularly for challenging compositions.

---

# Results

### Discovery of Stable Crystals

<!-- ![Statistics of discovered crystals using GNoME.](assets/img/2025-06-01-Scaling-deep-learning-for-materials-discovery/gnome_statistics.webp) -->
<!-- relative_url -->
<img src="{{'assets/img/2025-06-01-Scaling-deep-learning-for-materials-discovery/gnome_statistics.webp' | relative_url}}" alt="GNoME Pipeline Illustration" width="70%">

<div class="caption">
  <strong>Figure:</strong> Statistics of discovered crystals using GNoME.
</div>


Scaling deep learning with GNoME has resulted in a transformative expansion of the known crystal universe. The GNoME pipeline identified over **2.2 million** candidate structures stable with respect to the Materials Project, with **381,000** of these residing on the updated convex hull as newly discovered stable materials. These results increase the stable crystal catalog by nearly an order of magnitude.

GNoME also displaced at least 5,000 previously labeled ‘stable’ materials from the Materials Project and OQMD, suggesting both the sensitivity and superiority of its predictions. Notably, GNoME discovered a large number of structures with **more than four unique elements**, historically a challenging space for previous methods.

Clustering analysis identified **45,500+ new structural prototypes**, significantly exceeding the \~8,000 in the original Materials Project. This diversity confirms the efficacy of SAPS and substitution-guided generation in exploring uncharted crystal spaces.

### Experimental and Functional Validation

Out of newly discovered crystals, **736 match ICSD experimental structures**, supporting the physical relevance of GNoME’s predictions. Additionally, **91%** of 2,202 new Materials Project compositions since 2021 align structurally with GNoME predictions.

Further validation was performed using the r2SCAN functional. For GNoME predictions:

* **84%** of binaries and ternaries remain stable under r2SCAN
* **86.8%** of quaternaries remain on the r2SCAN convex hull

These metrics show that the model's stability predictions hold under more accurate energy functionals.

### Application-Specific Material Families

With GNoME's expanded catalogue, domain-specific filtering yields striking results:

* **52,000 stable layered materials** (vs. \~1,000 previously)
* **528 promising Li-ion conductors** (25× increase over prior studies)
* **15 new stable Li/Mn transition-metal oxides**, important for next-gen batteries

These advances underscore the practical impact of a massive, high-quality material database.

### Scaling Machine-Learned Interatomic Potentials (MLIPs)

Beyond static stability, GNoME generates vast data on structural relaxations. This enables training of a **general-purpose MLIP** using NequIP, pretrained on diverse ionic relaxation data.

Key findings:

* Scaling pretraining data leads to **power-law improvements** in downstream accuracy
* **Zero-shot performance** surpasses prior general-purpose MLIPs
* Pretrained potentials transfer well to out-of-distribution settings (e.g., AIMD at higher temperatures)

In transferability tests, fine-tuning from the GNoME-pretrained potential consistently outperformed training from scratch—even with 1,000+ samples.

### Accelerating Discovery of Solid-State Electrolytes

To evaluate zero-shot capability, GNoME-pretrained potentials were tested on **623 unseen compositions** via molecular dynamics. These models:

* Accurately classify superionic conductors
* Generalize across temperatures
* Scale to compositional spaces unreachable by AIMD

This establishes GNoME MLIPs as robust tools for high-throughput screening of solid-state electrolytes—offering predictive power at a fraction of the cost.

---
# Conclusion
This work demonstrates how large-scale training of graph neural networks (GNNs) on diverse first-principles datasets can significantly accelerate the discovery of inorganic materials. By leveraging the GNoME framework, we have discovered 2.2 million stable crystals—an order-of-magnitude increase over existing materials databases. Beyond discovery, the generated data enables the development of machine-learned interatomic potentials that exhibit robust, zero-shot performance on previously unseen materials in molecular dynamics simulations.

These results highlight a key shift in the role of deep learning in scientific discovery. Traditionally, machine learning models struggle when deployed on out-of-distribution data—yet exploration, by its nature, requires generalizing far beyond the training distribution. Our findings show that scaling up model size and data diversity allows for emergent generalization, both across chemical space (e.g., in compositions with five or more elements) and across new physical tasks (e.g., predicting ionic conductivity or thermodynamic stability).

Despite these advances, several challenges remain. Translating computational stability into real-world synthesis depends on understanding phase competition, polymorphism, vibrational stability, and entropy-driven effects. But the foundation laid by GNoME—massive-scale prediction, transferable potentials, and general-purpose pretrained models—paves the way for a new era of materials discovery powered by AI.

We envision GNoME not just as a single tool, but as an evolving platform. As the field grows, pretrained general-purpose GNoME models may become the backbone for research across energy storage, electronics, catalysis, and beyond—offering scientists the ability to explore chemical space with unprecedented breadth, speed, and confidence.