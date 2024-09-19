---
description: Does complexity Pay Off?
slug: complexity_iaria
date: 2024-09-17T20:21:13+02:00
image: image.png
draft: false
    
tags:
    - Publications

comments: false
links:
  - title: Link to the paper
    description: Arxiv link of the paper
    website: https://arxiv.org/

---

**Does Complexity Pay Off? Applying Advanced Algorithms to Depression Detection on the GLOBEM Dataset**

*Sebastian Cavada, Alvaro Berobide, Yevheniia Kryklyvets*

## Abstract

This manuscript evaluates the performance of state-of-the-art time series analysis algorithms for depression detection on the GLOBEM dataset. We assess TSMixer, Crossformer, GRU, CNN_LSTM and introduce a novel self-developed algorithm with the goal of increasing accuracy over the original Reorder. While these models demonstrate robust out-of-domain generalization, they fail to surpass the accuracy of the baseline Reorder algorithm, which was specifically developed for in-domain analysis by the GLOBEM team. Our findings reveal consistently low performance across all models, suggesting limitations inherent in the dataset rather than the algorithms themselves. We hypothesize that the dataset’s absence of critical variables and insufficient granularity likely limits model convergence. This hypothesis is supported by similar studies that achieved higher accuracy using more frequent data points with similar architecture approaches. Based on these insights, we suggest that future studies might benefit from incorporating more granular sensor measurements and more sophisticated data types such as, but not limited to, Heart Rate Variability (HRV)