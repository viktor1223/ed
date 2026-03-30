---
title: "Experiment 06: Escalation State Machine Convergence"
description: Absorbing Markov chain analysis and Monte Carlo validation of the Phase 1 escalation system
author: Viktor Ciroski
ms.date: 2025-07-17
ms.topic: reference
---

## Overview

Analyzes the Phase 1 escalation state machine as an absorbing Markov chain.
Computes analytical absorption probabilities, validates them with Monte Carlo
simulation (10 000 episodes), and sweeps over resolution probability and
max attempts to characterize system behavior.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Episodes (simulation) | 10 000 |
| Default max attempts | 4 modalities before escalation |
| P(prerequisite issue) | 0.30 |
| Resolution probability sweep | 0.10 to 0.90 (step 0.05) |
| Max attempts sweep | 2, 3, 4, 5, 6, 8 |
| Seed | 42 |

## Results

### Resolution Probability Sweep (max_attempts = 4)

| p_resolve | P(resolved) | P(escalated) | E[steps] | E[modalities] |
|:---------:|:-----------:|:------------:|:--------:|:-------------:|
| 0.10 | 0.344 | 0.656 | 5.49 | 3.43 |
| 0.20 | 0.590 | 0.410 | 4.78 | 2.96 |
| 0.30 | 0.760 | 0.240 | 4.17 | 2.53 |
| 0.40 | 0.870 | 0.130 | 3.64 | 2.18 |
| **0.50** | **0.938** | **0.063** | **3.20** | **1.89** |
| 0.60 | 0.974 | 0.026 | 2.83 | 1.63 |
| 0.70 | 0.992 | 0.008 | 2.53 | 1.40 |
| 0.80 | 0.998 | 0.002 | 2.30 | 1.24 |
| 0.90 | 1.000 | 0.000 | 2.12 | 1.11 |

### Analytical-Simulation Agreement

Maximum discrepancy in P(resolved) across all 17 sweep points: **0.0032**.
The Markov chain model is an accurate representation of the state machine.

## Interpretation

1. **At p=0.50 (v2 RCT setting), 93.75% of misconceptions resolve before
   teacher escalation.** Only 6.25% require human intervention. This is a
   healthy ratio: teachers handle the genuinely difficult cases while the
   system resolves most issues autonomously.

2. **The safety net works.** At p=0.20 (low resolution), 41% of misconceptions
   escalate to teacher conference. This is exactly the desired behavior:
   when interventions are ineffective, the system escalates rather than
   cycling endlessly.

3. **Expected steps are reasonable.** At p=0.50, the system reaches a terminal
   state in 3.2 steps (interactions), trying 1.89 modalities on average.
   Most students experience 1-2 modality switches, not all 4-5.

4. **Diminishing returns beyond 4 attempts.** The attempts sweep (in
   results.json) shows that going from 4 to 6 max attempts at p=0.50
   increases P(resolved) from 0.938 to 0.969 - a 3.1 percentage point
   gain at the cost of keeping students in remediation longer. The default
   of 4 attempts is a reasonable balance.

5. **The prerequisite check at attempt 2 is well-placed.** At p=0.30, 30%
   of students who reach attempt 2 are redirected to prerequisite
   remediation. This catches foundational gaps early rather than wasting
   all 4 modality attempts.

## Design Recommendations

- **Keep max_attempts=4 as the default.** It resolves 94% at p=0.50 while
  keeping the escalation path short.
- **Monitor the real p_resolve empirically.** If observed resolution rates
  are below 0.30, the interventions need improvement (content quality
  issue, not a system design issue).
- **Consider adaptive max_attempts per student.** Students with learning
  profiles indicating persistence could get 5-6 attempts; students showing
  frustration signals could escalate after 2-3.

## Limitations

- The model assumes resolution probability is constant across modalities
  for a given student. In reality, different modalities have different
  success rates (that is the point of Thompson sampling in Phase 1).
- The prerequisite check probability (0.30) is a parameter, not derived
  from data. Real classrooms will have different prerequisite gap rates.
- The model does not account for time between interactions (spaced
  repetition effects) or student fatigue from repeated failed attempts.
