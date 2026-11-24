---
title: "FungiTastic Baseline"
description: "Training-free fine-grained semantic segmentation in low-data regimes."
slug: fungitastic-baseline
date: 2026-05-21T15:44:52+02:00
image: fungitastic.png
draft: false

tags:
    - Publications

comments: false
links:
  - title: Paper
    description: arXiv preprint
    website: https://arxiv.org/abs/2605.22492

---

### CVPR 2026 FGVC Workshop

**Training-Free Fine-Grained Semantic Segmentations in Low Data Regimes: A FungiTastic Baseline**

*Sebastian Cavada, Francesco Pelosin, Lapo Faggi*

## Abstract

Fine-grained semantic segmentation in FungiTastic requires accurate mask localization while distinguishing between visually similar mushroom categories under long-tailed data and variable capture conditions. This work introduces a training-free two-stage baseline: SAM3 first generates class-agnostic mushroom masks from macro-taxonomic prompts, then DINOv3 assigns fine-grained labels using prototype matching in feature space. A simple feature-space transformation improves the prototype classifier, making the approach more scalable than class-specific prompting while keeping segmentation cost low. The paper reports performance from one-shot to few-hundred-shot settings and establishes an initial baseline for low-data fine-grained semantic segmentation.
