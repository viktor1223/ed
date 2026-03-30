# Experiment 08: BKT Estimation Fidelity - Notes

## Status: COMPLETE - CONFIRMS SIMULATION VALIDITY ISSUE

## Research Questions

1. How accurately does BKT track the student's true p_know?
2. How does BKT parameter misspecification degrade concept selection and learning?
3. Which BKT parameter is the system most sensitive to?

## Results Summary

### Part A: BKT Tracking Accuracy

| Metric                           | Value  |
|----------------------------------|--------|
| Concept selection accuracy (vs oracle) | 26.4%  |
| Random baseline (1/5 concepts)   | 20.0%  |
| Initial RMSE (BKT vs true)      | 0.376  |
| Final RMSE (after 40 steps)     | 0.358  |

BKT barely improves its estimate of the student's true state over 40 interactions. The concept selector is only 6.4 percentage points above random.

Per-concept RMSE:
- integer_sign_ops: 0.239
- order_of_operations: 0.361
- distributive_property: 0.415 (worst - highest level)
- combining_like_terms: 0.369
- solving_linear_equations: 0.361

### Part B: Parameter Sensitivity

| Parameter | Baseline Gain | Worst Gain | Worst Scale | Max Degradation |
|-----------|-------------|-----------|------------|----------------|
| p_learn   | 0.189       | 0.167     | 3.0x       | 0.022          |
| p_guess   | 0.189       | 0.153     | 1.25x      | 0.036          |
| p_slip    | 0.189       | 0.164     | 1.5x       | 0.025          |

Most sensitive parameter: **p_guess** (but only 0.036 gain difference).

The range of gains across ALL perturbation levels is 0.153 to 0.221 - far smaller than the within-condition standard deviation (~0.20). No perturbation produces a statistically significant change.

### Part C: Joint Perturbation

All BKT parameters perturbed simultaneously:

| Scale | Mean Gain |
|-------|----------|
| 0.25x | 0.194    |
| 0.50x | 0.187    |
| 1.00x | 0.189    |
| 2.00x | 0.193    |
| 3.00x | 0.177    |

Gains essentially flat. Perturbing ALL parameters by 3x simultaneously produces only a 6% drop.

## Critical Finding: BKT Is Decorative

The BKT model is not driving tutoring decisions in a meaningful way:

1. **Concept selection is near-random**: 26.4% accuracy vs 20% random baseline. The tutor barely outperforms pure random concept selection.

2. **BKT doesn't converge**: RMSE drops from 0.376 to 0.358 over 40 interactions. The model is not learning the student's state. This is because:
   - BKT only observes correct/incorrect (binary signal)
   - 8 observations per concept (40/5) is insufficient for convergence
   - The student's true state is changing (learning) while BKT tries to estimate it - a moving target

3. **Misspecification doesn't matter**: If the model were actually driving decisions, 3x parameter errors should cause catastrophic failure. The near-zero sensitivity means BKT could be replaced with a random number generator with minimal impact.

### Why This Confirms the Simulation Issue

From Experiment 07, we learned that the simulated student's learning is dominated by interaction count, not intervention quality. Experiment 08 confirms the other side:

**The adaptive system's decisions (which concept to teach) have almost no impact on learning outcomes because the student learns regardless.**

The BKT model, the concept selection algorithm, and the problem selection strategy are all decorative - they make the system look adaptive, but the learning gains come from the simulated student's built-in `receive_instruction()` learning curve, not from the tutor's decisions.

## Implications

1. **The adaptive loop is broken**: Sense (BKT) -> Decide (concept selection) -> Act (intervention) -> Learn (student). But the Learn step dominates everything, making Sense and Decide irrelevant.

2. **Need fewer interactions**: Test with 10-15 interactions (1x mastery horizon) instead of 40 (3-4x). At tighter budgets, routing quality should matter.

3. **Need harder discrimination**: The 5-concept linear chain is too small. A 20-50 concept graph would make routing quality matter because random routing wastes more budget.

4. **BKT needs richer signals**: Binary correct/incorrect to 5 concepts over 40 observations isn't enough. Need response time, partial credit, or misconception-aware observations.

## Artifacts

- [results.json](artifacts/results.json)
- [tracking_accuracy.png](artifacts/tracking_accuracy.png)
- [parameter_sensitivity.png](artifacts/parameter_sensitivity.png) - flat sensitivity curves
- [concept_selection_error.png](artifacts/concept_selection_error.png) - per-concept RMSE
- [joint_perturbation.png](artifacts/joint_perturbation.png) - joint perturbation also flat
