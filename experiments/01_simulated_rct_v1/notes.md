---
title: "Experiment 01: Simulated RCT v1 (BKT-Coupled Assessment)"
description: Findings and interpretation from the original simulated randomized controlled trial using internal BKT mastery as the primary metric
author: Viktor Ciroski
ms.date: 2026-03-29
ms.topic: reference
---

## Overview

This experiment ran 500 simulated students per condition through 40 tutoring
interactions under four strategies: adaptive (BKT-driven concept selection with
targeted misconception remediation), random (random concept, generic feedback),
fixed sequence (linear curriculum, generic feedback), and no remediation
(adaptive concept selection, no misconception targeting).

The primary metric was **mean mastery gain** measured by reading each simulated
student's internal `p_know` values (BKT state) before and after tutoring.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Students per condition | 500 |
| Interactions per student | 40 |
| Concepts | 5 (linear prerequisite chain) |
| Misconceptions | 19 across all concepts |
| Student archetypes | 5 (strong, strong-arith-weak-algebra, specific-gap, weak, mixed) |
| Random seed | 42 |
| Simulated student model | Tier 2: BKT-driven with misconception states |

## Results

### Condition Performance

| Condition | Mean Mastery Gain | SD | Concepts Mastered | Resolution Rate |
|-----------|------------------:|---:|------------------:|----------------:|
| Adaptive | 0.2036 | 0.059 | 0.688 / 5 | 12.4% |
| Random | 0.1788 | 0.059 | 0.806 / 5 | 0.0% |
| Fixed Sequence | 0.1849 | 0.058 | 0.886 / 5 | 0.0% |
| No Remediation | 0.1758 | 0.056 | 0.588 / 5 | 0.1% |

### Effect Sizes (Adaptive vs Baseline)

| Comparison | Cohen's d | 95% CI | p (adjusted) |
|------------|----------:|-------:|-------------:|
| vs Random | 0.423 | [0.299, 0.542] | < 0.001 |
| vs Fixed Sequence | 0.328 | [0.207, 0.450] | < 0.001 |
| vs No Remediation | 0.478 | [0.348, 0.608] | < 0.001 |

### Sensitivity Analysis

BKT learning rates varied from 0.5x to 2.0x. The system reported "robust"
(adaptive outperformed all baselines at every scale).

## Identified Flaws

This experiment has four structural problems that undermine the validity of its
findings.

### 1. Circular Measurement

The primary metric reads `student.p_know`, which is the simulated student's
internal BKT state. The tutor also uses BKT to track mastery. Both the
generation model and the evaluation metric use the same framework. This is
analogous to grading students by asking the teaching model what grade they
deserve, rather than giving them an independent test.

The effect sizes (d = 0.33 to 0.48) measure "how much does BKT-driven tutoring
improve BKT-estimated mastery" rather than "how much does the student actually
learn."

### 2. No Held-Out Assessment

In a real classroom, you would administer a pre-test and post-test composed of
problems the student has not seen during tutoring. This experiment has no such
independent assessment. The "mastery gain" is entirely model-internal.

### 3. Fixed Sequence Paradox

Fixed sequence masters **more concepts** (0.886) than adaptive (0.688), yet
adaptive reports a higher "mastery gain." This happens because:

- `next_action()` remediates weak concepts first, so adaptive spends most of
  its 40 interactions revisiting 1 or 2 struggling concepts
- Fixed sequence cycles evenly through all 5 concepts (8 interactions each),
  giving broad coverage

Adaptive gets deeper gains on the concepts it touches but covers fewer. The
primary metric (mean gain averaged across concepts) masks this tradeoff.

### 4. Structurally Impossible Misconception Resolution

With `targeted_resolution = 0.30`, each targeted intervention reduces a
misconception's activation probability by 30%. Starting from `p_active = 0.80`:

| Interventions | p_active |
|--------------:|---------:|
| 0 | 0.800 |
| 1 | 0.560 |
| 2 | 0.392 |
| 3 | 0.274 |
| 4 | 0.192 |
| 5 | 0.134 |
| 6 | 0.094 |

It takes 6-7 targeted interventions on a single misconception to drop below the
0.1 active threshold. With 40 interactions spread across 5 concepts and an
average of 3.7 active misconceptions per student, any specific misconception
receives at most 2-3 targeted interventions. The 12.4% resolution rate is baked
into the math, not a meaningful finding about the system's effectiveness.

## Conclusion

These results **validate that the system runs correctly** (the adaptive strategy
does produce higher internal mastery estimates). They do **not** validate that
the system helps students learn. The effect sizes are inflated by circular
measurement and should not be compared to literature benchmarks (VanLehn 2011,
Kulik & Fletcher 2016) that used independent pre/post assessments with real
students.

Experiment 02 addresses all four flaws with a decoupled assessment methodology.

## Reproducibility

```bash
cd <project_root>
python experiments/01_simulated_rct_v1/run.py
```

All outputs land in `experiments/01_simulated_rct_v1/artifacts/`.
