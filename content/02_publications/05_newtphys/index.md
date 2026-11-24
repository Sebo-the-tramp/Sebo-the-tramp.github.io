---
title: "NewtPhys"
description: "A physics-grounded benchmark for evaluating foundation models on Newtonian reasoning."
slug: newtphys
date: 2026-04-20T10:00:00+02:00
draft: false
image: newtphys.png

tags:
    - Publications

comments: false

---

### Preprint in preparation

**NewtPhys: Do Foundation Models Understand Newtonian Physics?**

*Sebastian Cavada, Soumava Paul, Tuan-Hung Vu, Andrei Bursuc, Raoul de Charette*

## Abstract

Previous work has evaluated physics reasoning in foundation models using synthetic or semi-synthetic scenes and visual question-answering tasks.
However, these benchmarks emphasize high-level events and lack the visual fidelity required to assess true low-level Newtonian understanding.
We introduce NewtPhys a 4D physically annotated dataset built from multiview images of real-world scenes with physics-grounded simulations.
The dataset provides dense, fine-grained annotations across timesteps --- including 3D forces and amodal per-pixel quantities covering physics, tracking, semantics and geometry --- bridging the gap between simplistic synthetic setups and realistic visual complexity.
Using NewtPhys, we systematically evaluate 56 VLMs, {including 54 open-weight models and 2 closed-source frontier models}, and 10 VFMs and reveal limitations in low-level physics reasoning.
Beyond benchmarking, our dataset enables future research in physics-grounded vision and the development of next-generation physics-aware evaluations.
Code and datasets are available at https://astra-vision.github.io/NewtPhys.
