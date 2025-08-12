---
description: CAD-Assistant Tool-Augmented VLLMs as Generic CAD Task Solvers
slug: cad_assistant
date: 2024-12-18T20:21:13+02:00
image: cad.png
draft: false
    
tags:
    - Publications

comments: false
links:
  - title: Link to the paper
    description: Arxiv link of the paper
    website: https://arxiv.org/abs/2412.13810

---

### ICCV 2025

**CAD-Assistant: Tool-Augmented VLLMs as Generic CAD Task Solvers**

*Dimitrios Mallis, Ahmet Serdar Karadeniz, Sebastian Cavada, Danila Rukhovich, Niki Foteinopoulou, Kseniya Cherenkova, Anis Kacem, Djamila Aouada*

## Abstract

We propose CAD-Assistant, a general-purpose CAD agent for AI-assisted design. Our approach is based on a powerful Vision and Large Language Model (VLLM) as a planner and a tool-augmentation paradigm using CAD-specific tools. CAD-Assistant addresses multimodal user queries by generating actions that are iteratively executed on a Python interpreter equipped with the FreeCAD software, accessed via its Python API. Our framework is able to assess the impact of generated CAD commands on geometry and adapts subsequent actions based on the evolving state of the CAD design. We consider a wide range of CAD-specific tools including a sketch image parameterizer, rendering modules, a 2D cross-section generator, and other specialized routines. CAD-Assistant is evaluated on multiple CAD benchmarks, where it outperforms VLLM baselines and supervised task-specific methods. Beyond existing benchmarks, we qualitatively demonstrate the potential of tool-augmented VLLMs as general-purpose CAD solvers across diverse workflows.