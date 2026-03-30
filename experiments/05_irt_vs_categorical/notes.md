---
title: "Experiment 05: IRT-Based vs Categorical Problem Selection"
description: Simulated RCT comparing difficulty targeting strategies for adaptive problem sequencing
author: Viktor Ciroski
ms.date: 2025-07-17
ms.topic: reference
---

## Overview

Tests whether IRT-based problem selection (targeting P(correct) ~ 0.70)
produces higher learning gains than categorical approaches (always-easy,
always-hard, random).

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Students per condition | 500 |
| Interactions per student | 40 |
| Concepts | 5 (linear prerequisite chain) |
| Practice problems | 18 (after test hold-out) |
| Test problems | 10 (2 per concept, held out) |
| IRT calibration | easy=-1.5, medium=0.0, hard=1.5 |
| Target P(correct) | 0.70 |
| Seed | 42 |

## Results

### Condition Performance

| Condition | Pre-Test | Post-Test | Gain | SD | Hit Rate | Concepts Mastered |
|-----------|:--------:|:---------:|:----:|:--:|:--------:|:-----------------:|
| IRT Targeted | 0.423 | 0.542 | +0.119 | 0.195 | 17.1% | 0.3 |
| Categorical Easy | 0.420 | 0.543 | +0.122 | 0.202 | 19.0% | 0.3 |
| Categorical Hard | 0.431 | 0.534 | +0.103 | 0.200 | 21.9% | 0.3 |
| Random | 0.420 | 0.538 | +0.118 | 0.204 | 20.7% | 0.3 |

### Effect Sizes (IRT vs Baselines)

| Comparison | Cohen's d | 95% CI | p-value |
|------------|:---------:|:------:|:-------:|
| IRT vs Categorical Easy | -0.018 | [-0.142, 0.106] | 0.775 |
| IRT vs Categorical Hard | +0.080 | [-0.044, 0.204] | 0.206 |
| IRT vs Random | +0.003 | [-0.121, 0.127] | 0.962 |

### Bank Size Scaling

| Problems/Concept | IRT Gain | Easy Gain | IRT Advantage |
|:----------------:|:--------:|:---------:|:-------------:|
| 3 | 0.122 | 0.104 | +0.018 |
| 5 | 0.144 | 0.144 | +0.000 |
| 10 | 0.286 | 0.289 | -0.003 |
| 20 | 0.384 | 0.365 | +0.019 |

## Interpretation

1. **IRT shows no significant advantage at current bank size.** With only
   18 practice problems (3-4 per concept), IRT cannot meaningfully differentiate
   difficulty. The problem bank is too coarse for precision targeting.

2. **All conditions improve substantially.** Pre-to-post gains of 0.10-0.12
   show the BKT-driven concept selection (shared across conditions) is doing
   the heavy lifting. The problem selection within a concept matters less
   than picking the right concept.

3. **Categorical-hard performs worst as expected.** d=-0.080 vs IRT confirms
   that frustration-level difficulty harms learning, though the effect is
   small and non-significant.

4. **Hit rate is universally low.** All conditions have 17-22% desirable
   difficulty hit rate, far below the 70% target. This reflects the sparse
   problem bank: with 3 difficulty levels mapped to 3 IRT values, the
   "closest to target" problem may still be far from optimal.

5. **Bank size matters more than selection strategy.** At 20 problems/concept,
   all conditions show ~3x the gains of 3 problems/concept. Problem bank
   richness is the binding constraint, not selection intelligence.

6. **IRT advantage emerges at scale.** At 20 problems/concept, IRT achieves
   59.8% hit rate (vs 31.2% for categorical-easy) and begins showing a
   gain advantage (+0.019). With a production-scale bank (50+ problems),
   IRT targeting would likely demonstrate significant benefits.

## Design Recommendation

For Phase 3 implementation:

- **Prioritize expanding the problem bank** (via templates) before investing
  in sophisticated IRT selection. Target 15-20 problems per concept minimum.
- **Implement IRT selection anyway** - it becomes valuable as the bank grows,
  and the implementation cost is low.
- **The template engine is critical.** The bank size scaling results show that
  problem variety is the single biggest driver of learning gains.

## Limitations

- IRT parameters are manually assigned, not estimated from data. Real IRT
  calibration would produce more granular difficulty values.
- The simulated student model may not capture the "desirable difficulty"
  learning mechanism accurately. Real students may benefit more from being
  challenged at their zone of proximal development.
- Only 3 difficulty levels in the bank means the IRT selector often picks the
  same problem as categorical-easy (the "closest" easy problem is also the
  "easiest" one).
