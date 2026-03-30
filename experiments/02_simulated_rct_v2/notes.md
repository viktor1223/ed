---
title: "Experiment 02: Simulated RCT v2 (Decoupled Assessment)"
description: Findings and interpretation from the redesigned simulated RCT using held-out test problems as the primary metric
author: Viktor Ciroski
ms.date: 2026-03-29
ms.topic: reference
---

## Overview

This experiment redesigns the v1 evaluation to address four structural flaws:
circular measurement, no held-out assessment, the fixed-sequence coverage
paradox, and structurally impossible misconception resolution.

The primary metric is now **proportion correct on a held-out test set** of
problems the student never sees during tutoring.

## Changes from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Primary metric | Internal `p_know` (BKT state) | Held-out test score |
| Assessment independence | BKT evaluates BKT | Test decoupled from learning model |
| Concept coverage | Adaptive gets stuck on 1-2 concepts | Coverage floor: max 6 consecutive on one concept |
| Misconception resolution | `targeted_resolution = 0.30` (nearly impossible) | `targeted_resolution = 0.50` (achievable in 40 interactions) |
| Diagnostic accuracy | Not measured | Tracked as sensitivity metric |
| Sensitivity analysis | Varies BKT only | Varies both BKT learning rate and resolution rate |

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Students per condition | 500 |
| Interactions per student | 40 |
| Concepts | 5 (linear prerequisite chain) |
| Misconceptions | 19 across all concepts |
| Problem bank split | 18 practice, 10 held-out test (2 per concept) |
| Coverage floor | 6 max consecutive interactions on one concept |
| Targeted resolution | 0.50 per intervention |
| Random seed | 42 |

## Results

### Condition Performance

| Condition | Pre-Test | Post-Test | Gain | SD | Resolution | Concepts Touched |
|-----------|:--------:|:---------:|-----:|---:|-----------:|-----------------:|
| Adaptive | 0.429 | 0.635 | +0.206 | 0.204 | 37.7% | 5.0 |
| Random | 0.427 | 0.584 | +0.157 | 0.197 | 0.0% | 5.0 |
| Fixed Sequence | 0.441 | 0.596 | +0.155 | 0.192 | 0.0% | 5.0 |
| No Remediation | 0.435 | 0.586 | +0.152 | 0.191 | 0.3% | 4.8 |

Diagnostic sensitivity (adaptive only): **94.2%** (when a student uses a
misconception, the system correctly identifies and targets it 94.2% of the
time).

### Effect Sizes (Adaptive vs Baseline)

| Comparison | Cohen's d | 95% CI | p (adjusted) | Significant |
|------------|----------:|-------:|-------------:|:-----------:|
| vs Random | 0.247 | [0.117, 0.374] | 0.0003 | Yes |
| vs Fixed Sequence | 0.260 | [0.132, 0.386] | 0.0001 | Yes |
| vs No Remediation | 0.278 | [0.156, 0.402] | < 0.0001 | Yes |

### v1 to v2 Effect Size Comparison

| Comparison | v1 (inflated) | v2 (decoupled) | Change |
|------------|:-------------:|:--------------:|:------:|
| vs Random | 0.423 | 0.247 | -42% |
| vs Fixed Sequence | 0.328 | 0.260 | -21% |
| vs No Remediation | 0.478 | 0.278 | -42% |

Effect sizes dropped 21-42% when measured independently. All remain
statistically significant but are now in the **small** range (d = 0.25-0.28)
rather than the small-to-medium range v1 reported.

## Key Findings

### 1. The System Works, Modestly

Adaptive targeted remediation produces a real (simulated) advantage over all
baselines on an independent metric. The advantage is small but consistent: about
5 percentage points on a held-out test (0.206 vs 0.152-0.157 gain).

### 2. The Fixed-Sequence Paradox is Resolved

With the coverage floor, adaptive now touches all 5 concepts (5.0/5), matching
fixed sequence. In v1, fixed sequence mastered more concepts (0.886 vs 0.688)
because adaptive got stuck. The coverage floor eliminates this distortion.

### 3. Misconception Resolution is Now Meaningful

Resolution improved from 12.4% (v1) to 37.7% (v2). This is the direct result
of raising `targeted_resolution` from 0.30 to 0.50. At 0.50, a misconception
starting at `p_active = 0.80` drops below 0.10 after 4 targeted interventions
instead of 7.

### 4. Diagnostic Sensitivity is High

94.2% sensitivity means the adaptive system correctly identifies and targets a
misconception nearly every time one occurs. This is the strongest finding: the
detection component works well. The gap is in what happens after detection (see
Limitations).

### 5. Robustness Fails at Low Learning Rates

The sensitivity analysis reveals that adaptive does **not** always win:

| Failure case | What happened |
|------|------|
| `bkt_0.5, res_0.30` | no_remediation (0.110) >= adaptive (0.103) |
| `bkt_0.5, res_0.70` | no_remediation (0.113) >= adaptive (0.096) |
| `bkt_1.0, res_0.30` | fixed_sequence (0.183) >= adaptive (0.172) |
| `bkt_1.0, res_0.30` | no_remediation (0.198) >= adaptive (0.172) |
| `bkt_1.0, res_0.50` | no_remediation (0.198) >= adaptive (0.189) |

Pattern: at low BKT learning rates (slow learners) or low resolution rates,
adaptive loses. The interpretation: **targeted misconception remediation
requires enough learning capacity to benefit from it.** For very slow learners,
broad exposure (fixed sequence) or simple advancement (no remediation) may be
more effective than repeatedly targeting misconceptions that resist resolution.

This is a genuine finding, not a methodological artifact. It suggests that an
ideal system would adapt its strategy based on learner speed: use targeted
remediation for students who respond to it, and broader coverage for students
who do not.

## Limitations

These results remain **simulated**. They validate system behavior, not real
student learning. Specific limitations:

1. The simulated student's learning model is still simple (BKT with
   misconception states). Real students have richer learning dynamics.
2. The test set is small (2 problems per concept, 10 total). A real pre/post
   test would use 5-10 problems per concept.
3. The problem bank (28 problems) is too small for robust adaptive sequencing.
   Students encounter repeats.
4. Interventions are still hardcoded strings. The simulation models "targeted
   intervention" as a probability modifier, not an actual pedagogical
   interaction.
5. Evidence level: **Stage 2 (system validation)**. A human pilot is needed for
   effect size claims.

## Next Steps

1. Run this evaluation framework with different student models (not just BKT)
   to see if findings hold under model misspecification.
2. Expand the problem bank to 100+ problems to reduce repeat exposure.
3. Implement Layer 1 from the agentic roadmap (intervention intelligence with
   tracking) and re-evaluate.
4. Prepare for a Stage 3 Wizard-of-Oz pilot with 5-10 real students.

## Reproducibility

```bash
cd <project_root>
python experiments/02_simulated_rct_v2/run.py
```

All outputs land in `experiments/02_simulated_rct_v2/artifacts/`. The v1-vs-v2
comparison plot requires experiment 01 to have been run first.
