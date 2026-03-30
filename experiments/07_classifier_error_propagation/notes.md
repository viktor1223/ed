# Experiment 07: Classifier Error Propagation - Notes

## Status: COMPLETE - CRITICAL SIMULATION VALIDITY FINDING

## Research Questions

1. How do classifier errors propagate through the tutoring pipeline?
2. What minimum classifier accuracy does the system need?
3. Which error types are most destructive?

## Results Summary

| Error Type        | Gain at 0% Error | Gain at 20% Error | Gain at 50% Error | Degradation at 50% |
|-------------------|------------------|-------------------|-------------------|-------------------|
| Misidentification | 0.189            | 0.184             | 0.172             | 9%                |
| False Negative    | 0.189            | 0.181             | 0.186             | 2%                |
| False Positive    | 0.189            | 0.221             | 0.255             | -35% (IMPROVED)   |
| Concept Misroute  | 0.189            | 0.181             | 0.165             | 13%               |

**No threshold found**: Even at 50% error rate, no error type drops gain below 80% of baseline.

## Critical Finding: The Simulation Is Not a Valid Test Bed

These results do NOT mean the classifier doesn't matter. They mean the simulation is saturated and cannot discriminate between good and bad tutoring.

### Root Cause Analysis

The simulated student's `receive_instruction()` applies learning on EVERY call regardless of targeting quality:

- **Generic instruction**: `p_new = p_L + (1 - p_L) * p_T` where p_T = 0.15
- **Targeted instruction** (even with wrong misconception): `p_new = p_L + (1 - p_L) * p_T * 2.0`

Starting from p_know = 0.10:
- Generic: reaches mastery (0.85) in ~15 interactions
- Targeted (2x bonus): reaches mastery in ~5 interactions
- Budget: 40 interactions / 5 concepts = 8 per concept minimum

Since BKT learning is compounding (`p_L + (1-p_L) * rate`), even generic instruction saturates within the interaction budget. The classifier errors change WHICH misconception gets resolved, but `p_know` rises regardless.

### Why False Positives IMPROVE Gains

When the classifier hallucinates a misconception on a correct answer:
1. The student gets `targeted_misconception != None` in `receive_instruction()`
2. This triggers the 2x learning bonus
3. Extra "targeted" instruction accelerates p_know growth
4. The wasted misconception resolution (on a hallucinated misconception) has no downside

This is a simulation artifact: real students don't learn 2x faster just because a tutor incorrectly claims they have a misconception.

### What This Means

The experiment correctly executed its protocol. The finding is:

**The simulated student model cannot distinguish between high-quality and low-quality tutoring because learning is dominated by the interaction count, not the intervention quality.**

This invalidates ALL learning-gain-based conclusions from experiments 01-06 that use this simulation.

## Implications for the Project

1. **Simulation redesign needed**: The `receive_instruction()` method should condition learning rate on whether the intervention actually matches the student's state
2. **Wrong metric**: Test score gain is dominated by p_know growth, which happens regardless. A better metric would be misconception resolution rate or wasted-interaction ratio
3. **Resolution rate IS informative**: The resolution rate column does show proper degradation (38% to 17% for concept_misroute) because it directly measures whether the RIGHT misconception was targeted. This metric should be primary.

## Artifacts

- [results.json](artifacts/results.json)
- [gain_degradation.png](artifacts/gain_degradation.png) - flat lines confirm saturation
- [bkt_error.png](artifacts/bkt_error.png) - BKT error unchanged by classifier error (expected - BKT uses correct/incorrect, not misconception ID)
- [error_type_comparison.png](artifacts/error_type_comparison.png)
- [threshold_analysis.png](artifacts/threshold_analysis.png) - no thresholds found
