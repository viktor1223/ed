---
title: "Experiment 04: Thompson Sampling vs Greedy Modality Selection"
description: Monte Carlo evaluation of intervention modality selection policies
author: Viktor Ciroski
ms.date: 2025-07-17
ms.topic: reference
---

## Overview

Tests whether Thompson sampling outperforms greedy modality selection for
assigning intervention types (visual, concrete, pattern, verbal, peer) to
students with latent modality preferences.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Students | 1 000 (main), 500 (scalability) |
| Interactions per student | 50 |
| Modalities | 5 (main), 3-10 (scalability) |
| Preference generation | Dirichlet(1,...,1), rescaled to [0.15, 0.85] |
| Seed | 42 |

## Results

### Main Experiment (5 modalities, 50 interactions)

| Policy | Resolution Rate | SD | Regret vs Oracle | Median Convergence |
|--------|:--------------:|:--:|:----------------:|:-----------------:|
| Thompson | 0.3442 | 0.0967 | 0.1184 | 4 interactions |
| Greedy | 0.3637 | 0.1358 | 0.1003 | 3 interactions |
| Uniform | 0.2938 | 0.0605 | 0.1767 | 5 interactions |

### Scalability (Resolution Rate by Action Space Size)

| Modalities | Thompson | Greedy | Uniform |
|:----------:|:--------:|:------:|:-------:|
| 3 | 0.3801 | 0.3886 | 0.3312 |
| 4 | 0.3553 | 0.3618 | 0.3061 |
| 5 | 0.3393 | 0.3503 | 0.2929 |
| 6 | 0.3204 | 0.3318 | 0.2729 |
| 8 | 0.3007 | 0.3059 | 0.2462 |
| 10 | 0.2826 | 0.2865 | 0.2227 |

## Interpretation

1. **Greedy beats Thompson in this setting.** With only 5 modalities and 50
   interactions, the action space is small enough that greedy converges
   quickly (3 interactions) without needing principled exploration. Thompson's
   exploration cost outweighs its benefit.

2. **Both substantially beat uniform.** The 17% relative improvement of
   greedy over uniform (0.364 vs 0.294) confirms that personalized modality
   selection matters - students genuinely differ in which intervention type
   works best.

3. **Thompson's value is theoretical, not practical at this scale.** Thompson
   sampling is designed for settings where (a) the action space is large,
   (b) the horizon is long, or (c) preference distributions shift over time.
   With 5 modalities and 50 interactions, greedy's simplicity wins.

4. **Greedy has higher variance.** SD of 0.136 vs 0.097 for Thompson. Greedy
   occasionally locks into a suboptimal modality early, while Thompson's
   exploration prevents this - but the mean outcome still favors greedy.

5. **Scalability gap narrows at larger action spaces.** At 10 modalities,
   the greedy-Thompson gap shrinks from 5.7% to 1.4% relative. In domains
   with more intervention types, Thompson's exploration would likely become
   net-positive.

## Design Recommendation

For the Phase 1 implementation with 5 modalities:

- **Use greedy with epsilon-exploration (epsilon = 0.10).** This captures
  greedy's simplicity while maintaining a small exploration budget.
- **Log what Thompson would have chosen** alongside the actual selection.
  This generates counterfactual data for off-policy evaluation without
  paying the exploration cost.
- **Revisit Thompson when the action space grows** (e.g., per-misconception
  modality variants or cross-domain intervention types).

## Limitations

- Dirichlet(1,...,1) preference vectors are uniformly distributed. Real
  students likely have more peaked preferences (e.g., strongly visual
  learners), which could favor Thompson.
- Resolution is binary (success/fail) per interaction. Real misconception
  resolution is gradual, which changes the exploration-exploitation tradeoff.
- The oracle baseline redraws a separate random outcome each timestep, so
  oracle regret includes randomness in the reward realization.
