---
layout: distill
title: AI810 Blog Post (20244491) CRYSTALFORMER:Infinitely Connected Attention for Periodic Structure Encoding
description: Transformer-based models have shown remarkable success in molecular property prediction, but applying them to crystal structures presents a unique challenge due to the periodic and infinite nature of crystals. In this blog post, we explore CRYSTALFORMER, a novel architecture that introduces infinitely connected attention to effectively model periodic atomic structures. By interpreting this infinite attention as a form of neural potential summation, CRYSTALFORMER enables tractable and expressive encoding of periodicity in crystals. Despite its architectural simplicity and reduced parameter count, the model achieves state-of-the-art performance across multiple benchmark datasets, offering a powerful yet efficient approach to crystal property prediction.
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
bibliography: 2025-06-01-crystalformer.bib

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

Predicting the physical properties of materials—like conductivity, stability, or energy band gaps—directly from their crystal structures can dramatically accelerate materials discovery. Instead of relying on slow, compute-heavy simulations like density functional theory (DFT), machine learning models offer a fast, scalable alternative. But while modeling molecules with neural networks has seen great progress, crystals present a unique challenge: periodicity. Unlike molecules, crystal structures repeat infinitely in space, and that makes direct application of Transformer-based models far from trivial.

Transformer architectures like Graphormer have achieved impressive results in molecular property prediction using fully connected attention. But in crystals, where atoms are connected not just to nearby neighbors but also to infinitely repeating copies, this approach leads to a formulation we call infinitely connected attention. It raises a natural question: Can we adapt standard Transformer attention to respect periodicity without blowing up computation or losing physical fidelity?

In this blog post, we introduce CRYSTALFORMER, a new Transformer-based encoder that interprets infinitely connected attention as a form of neural potential summation—a physics-inspired, computationally tractable formulation of long-range interatomic interactions. With this view, we preserve the simplicity of the original Transformer while efficiently encoding periodic structures. CRYSTALFORMER not only reduces the parameter count by over 70% compared to prior work like MATFORMER, but also achieves state-of-the-art performance on benchmark datasets like Materials Project and JARVIS-DFT.

Let’s explore how CRYSTALFORMER brings Transformer power to the world of crystals—without breaking symmetry or the GPU.

---

# Preliminaries

In this section, we introduce the basic structure of crystal systems and review how standard self-attention mechanisms operate in Transformer encoders—especially when augmented with relative positional information. These foundations are essential for understanding how CRYSTALFORMER extends attention to periodic, infinitely repeating structures.

### Crystal Structures and Periodicity

A crystal is defined by a **unit cell**: a small, repeating 3D structure composed of atoms. The full crystal structure is formed by infinitely translating this unit cell across space using lattice vectors.

Each unit cell contains a finite set of atoms, with their Cartesian coordinates and atomic attributes represented as:

- **Coordinates**: $\mathcal{P} = \{\mathbf{p}_1, \mathbf{p}_2, ..., \mathbf{p}_N\}$
- **Initial Features**: $\mathcal{X}^0 = \{\mathbf{x}_1^0, \mathbf{x}_2^0, ..., \mathbf{x}_N^0\}$

The full structure is generated by translating each atom by a combination of three lattice vectors $\mathbf{l}_1, \mathbf{l}_2, \mathbf{l}_3 \in \mathbb{R}^3$:

$$
\mathbf{p}_{i(\mathbf{n})} = \mathbf{p}_i + n_1 \mathbf{l}_1 + n_2 \mathbf{l}_2 + n_3 \mathbf{l}_3,
$$

where $\mathbf{n} = (n_1, n_2, n_3) \in \mathbb{Z}^3$ denotes a 3D integer offset identifying the cell’s periodic image.

<img src="{{'assets/img/2025-06-01-crystalformer/crystal.png' | relative_url}}" alt="Crystal Structure Illustration" width="70%">

<div class="caption">
  <strong>Figure:</strong> A 2D illustration of a crystal structure and its infinitely connected periodic attention.
</div>

In what follows, we use:
- **i** to index atoms in the unit cell,
- **i(n)** to refer to periodic images of atom *i*,
- $\sum_{\mathbf{n}}$ to denote summation over all periodic copies of the unit cell.

### Self-Attention with Relative Position Representations

Transformers rely on self-attention mechanisms to model interactions between all pairs of tokens—or, in our case, atoms. The standard self-attention layer transforms input feature vectors $(\mathbf{x}_1, \dots, \mathbf{x}_N)$ into output vectors $(\mathbf{y}_1, \dots, \mathbf{y}_N)$ as:

$$
\mathbf{y}_i = \frac{1}{Z_i} \sum_{j=1}^N \exp\left(\frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_K}} + \phi_{ij} \right) \left( \mathbf{v}_j + \mathbf{\psi}_{ij} \right),
$$

where:
- $\mathbf{q}_i, \mathbf{k}_j, \mathbf{v}_j$ are query, key, and value vectors computed from inputs $\mathbf{x}_i$ and $\mathbf{x}_j$,
- $\phi_{ij}$ and $\bm{\psi}_{ij}$ encode scalar and vector biases based on relative positions (e.g., distance or direction),
- $Z_i$ is a normalization factor.

These position-aware augmentations allow the Transformer to capture spatial relationships—crucial for physical systems—but are designed for finite structures. To apply them to crystals, we need to extend attention to an **infinite** setting, where atoms interact not only with others in the same unit cell, but also with all periodic copies across space.

---

# CRYSTALFORMER

We now present the main methodology of **CRYSTALFORMER**, a Transformer-based model designed to encode periodic crystal structures using a physics-inspired attention mechanism.

### Problem Setup

We aim to predict physical properties of a given crystal structure. Each structure is represented by its unit cell, defined by:

- Atom coordinates: \${\mathbf{p}\_1, ..., \mathbf{p}\_N}\$
- Atom species (initial states): \${\mathbf{x}\_1^0, ..., \mathbf{x}\_N^0}\$
- Lattice vectors: \${\mathbf{l}\_1, \mathbf{l}\_2, \mathbf{l}\_3}\$

We model the physical state of the entire periodic structure using the finite set of unit-cell atom states \$\mathcal{X} = {\mathbf{x}\_1, ..., \mathbf{x}\_N}\$. The task is to evolve the symbolic input \$\mathcal{X}^0\$ into \$\mathcal{X}'\$, a set of high-level features used for property prediction.


### Infinitely Connected Attention as Neural Potential Summation

Inspired by Graphormer's dense attention over molecules, we generalize the attention mechanism to infinite periodic crystal structures:

$$
\mathbf{y}_i = \frac{1}{Z_i} \sum_{j=1}^N \sum_{\mathbf{n}} \exp\left(\frac{\mathbf{q}_i^T \mathbf{k}_{j(\mathbf{n})}}{\sqrt{d_K}} + \phi_{ij(\mathbf{n})}\right) (\mathbf{v}_{j(\mathbf{n})} + \boldsymbol{\psi}_{ij(\mathbf{n})})
$$

This formulation resembles potential summation in physical simulations, where atoms exert influence across all periodic images. The scalar term \$\phi\_{ij(\mathbf{n})}\$ and vector \$\boldsymbol{\psi}\_{ij(\mathbf{n})}\$ encode relative spatial relations.

### Pseudo-Finite Periodic Attention

<img src="{{'assets/img/2025-06-01-crystalformer/attention.png' | relative_url}}" alt="Pseudo-Finite Periodic Attention" width="70%">

<div class="caption">
  <strong>Figure:</strong> A 2D illustration of Pseudo-Finite Periodic Attention
</div>

To make the infinite summation tractable, we reformulate the attention as:

$$
\mathbf{y}_i = \frac{1}{Z_i} \sum_{j=1}^N \exp\left(\frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_K}} + \alpha_{ij}\right) (\mathbf{v}_j + \boldsymbol{\beta}_{ij})
$$

Where:

$$\alpha\_{ij} = \log \sum\_{\mathbf{n}} \exp(\phi\_{ij(\mathbf{n})})$$

$$\boldsymbol{\beta}_{ij} = \frac{1}{Z_{ij}} \sum_{\mathbf{n}} \exp(\phi_{ij(\mathbf{n})}) \boldsymbol{\psi}_{ij(\mathbf{n})}$$

These new terms, \$\alpha\_{ij}\$ and \$\boldsymbol{\beta}_{ij}\$, encode periodic spatial and edge information, allowing us to reduce infinite attention to a finite operation.

### Distance Decay Attention

We adopt a Gaussian kernel to define spatial dependencies:

$$
\exp(\phi_{ij(\mathbf{n})}) = \exp\left(-\frac{\|\mathbf{p}_{j(\mathbf{n})} - \mathbf{p}_i\|^2}{2 \sigma_i^2}\right)
$$

This formulation ensures fast convergence of the infinite sum, and we bound \$\sigma\_i\$ to control tail behavior. An adaptive window selects image cells to include in the summation, depending on \$\sigma\_i\$ and lattice geometry.

### Position Encoding with Radial Basis Functions

To preserve lattice information, we define:

$$
\boldsymbol{\psi}_{ij(\mathbf{n})} = W^E \cdot \mathbf{b}(\|\mathbf{p}_{j(\mathbf{n})} - \mathbf{p}_i\|)
$$

Where \$\mathbf{b}(r)\$ is a vector of Gaussian radial basis functions:

$$
 b_k(r) = \exp\left(-\frac{(r - \mu_k)^2}{2(r_{\text{max}} / K)^2}\right)
$$

This allows the model to differentiate between structures with identical atoms but different lattice constants.

### Implementation Details

- Image cell summation:
    $$
    |\mathbf{n}|_\infty \le 2,\quad
    \text{with adaptive extension up to }3.5\,\sigma_i\text{ (Å).}
    $$
- Gaussian RBF: \$K = 64\$, \$r\_{\text{max}} = 14\text{\AA}\$.

### Network Architecture

CRYSTALFORMER uses a Transformer encoder with four self-attention blocks. Each block includes:

* Multi-head pseudo-finite periodic attention
* Feed-forward layer
* Residual connections

**LayerNorm is removed**, following prior work showing improved stability with specialized initialization. Final material-level features are obtained via global average pooling and passed through a small MLP for property regression.

<!-- ![Network architecture of Crystalformer](assets/img/architecture.png) -->
<img src="{{'assets/img/2025-06-01-crystalformer/architecture.png' | relative_url}}" alt="Network architecture" width="70%">

<div class="caption">
  <strong>Figure:</strong> Network architecture of Crystalformer
</div>

---

# Experimental Results

### Datasets

We evaluate CRYSTALFORMER on two benchmark datasets:

* **Materials Project (MEGNet)**: 69,239 materials with DFT-calculated formation energy, bandgap, bulk modulus, and shear modulus.
* **JARVIS-DFT (3D 2021)**: 55,723 materials with DFT-derived properties including formation energy, total energy, bandgap (OPT and MBJ), and energy above hull (E-hull).

We follow established dataset splits and evaluation protocols used in Matformer and PotNet for fair comparison.

### Training Setup

We optimize mean absolute error (MAE) using AdamW and stochastic weight averaging (SWA). Hyperparameters include:

* Batch size: 128
* Epochs: 500 (extended for JARVIS)
* Learning rate: \$5 \times 10^{-4}\$ with decay
* SWA: Averaging over last 50 epochs

### Performance on Crystal Property Prediction

CRYSTALFORMER achieves state-of-the-art or competitive performance across all regression tasks. Notably:

* On **Materials Project**, CRYSTALFORMER outperforms all previous models on 3/4 tasks.
* On **JARVIS-DFT**, it is highly competitive with PotNet, while using fewer parameters.

We observe robust results even without SWA, indicating stable optimization.

### Model Efficiency

Compared to PotNet and Matformer:

* CRYSTALFORMER has **48.6% fewer parameters** than PotNet and **only 29.4%** of Matformer's parameter count.
* It achieves **faster inference** (6.6 ms/material) despite a longer training time.

This efficiency stems from the inductive bias introduced by the neural potential summation.

### Ablation Study

We examine the impact of removing the value position encoding \$\boldsymbol{\psi}\$. Results show:

* \$\boldsymbol{\psi}\$ significantly boosts formation and total energy prediction.
* It helps distinguish lattice configurations even when atom types are constant.

---

# Discussion and Limitations

### Angular and Directional Information

Currently, CRYSTALFORMER relies on distance-only position encodings (\$\phi\$ and \$\boldsymbol{\psi}\$) to ensure SE(3) invariance. While this makes the model invariant to rotation and translation, it limits expressive power. Prior studies show that such purely distance-based representations are incomplete. Future improvements could incorporate angular or directional features using:

* Three-body interactions (as in ALIGNN or M3GNet)
* Plane-wave edge encodings
* SE(3)-equivariant architectures (e.g., Equiformer, Faenet)
* Higher-order attention mechanisms (e.g., hypergraph Transformers)

### Modeling Realistic Interatomic Potentials

The current use of Gaussian decay in \$\exp(\phi)\$ serves as a universal approximator for distance-dependent interactions. While effective, it is a simplification of true physical potentials like:

* Coulomb potential (\$1/r\$)
* van der Waals potential (\$1/r^6\$)

These forms could be more faithfully modeled by mixing multiple functional forms across attention heads. Multi-head attention with varied decay types could allow each head to specialize, enhancing accuracy and interpretability.

### Long-Range Interactions and Reciprocal Space Attention

To limit computation, the Gaussian width \$\sigma\_i\$ is capped (e.g., \$\sigma\_{ub} \approx 2\text{\AA}\$), potentially underrepresenting long-range effects like Coulomb interactions. One way to address this is to leverage **Fourier-space attention**:

* Represent \$\alpha\_{ij}\$ using reciprocal space (Fourier domain)
* Capture long-range effects efficiently with lower-bounded \$\sigma\_i > \sigma\_{lb}\$
* Combine real and reciprocal space heads within MHA

A preliminary experiment using dual-space attention shows that:

* It improves E-hull prediction
* It degrades formation and total energy predictions

This indicates reciprocal-space modeling is task-sensitive. A promising direction is to **adaptively route attention heads** to real or reciprocal space depending on the property being predicted.

---

# Conclusions

We introduced **CRYSTALFORMER**, a Transformer-based encoder tailored for periodic crystal structures. By leveraging the concept of **infinitely connected attention**, our model treats all periodic replicas of atoms as part of the attention mechanism, and interprets the resulting formulation as a **neural potential summation**. We apply a simple Gaussian distance-decay to make this attention computationally feasible while retaining physical plausibility.

CRYSTALFORMER achieves strong predictive performance across a range of material property tasks while being significantly more parameter-efficient than competing methods. We also demonstrate its flexibility through extensions into reciprocal space attention to better handle long-range interactions.

We hope this simple yet effective Transformer framework bridges machine learning and materials science, providing new perspectives for both communities. Looking forward, an exciting challenge is scaling CRYSTALFORMER with large material datasets—drawing inspiration from the evolution of large language models. How such models can encode generalizable physical knowledge from massive materials data remains an open and ambitious question.