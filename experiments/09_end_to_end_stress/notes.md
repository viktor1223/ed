# Experiment 09: End-to-End Pipeline Stress Test - Notes

## Status: COMPLETE - DEFINITIVE SIMULATION INVALIDITY PROOF

## Research Questions

1. Do errors compound multiplicatively (catastrophic) or linearly (graceful)?
2. Which combination of subsystem failures is most destructive?
3. What is the minimum acceptable operating envelope?

## Results Summary

### Headline Result

**All 80 conditions meet the 80% baseline threshold. The "worst" condition loses only 12%.**

| Condition | Gain | vs Baseline |
|-----------|------|-------------|
| Baseline (no degradation) | 0.192 | 100% |
| Worst (0% cls, 3.0x BKT, 50% noise) | 0.169 | 88% |
| Best (40% cls, 1.0x BKT, 0% noise) | 0.263 | 137% |

Higher classifier error rates INCREASE gains. The "best" condition has 40% classifier errors.

### Error Compounding

| Condition | Individual Losses | Combined Loss | Mode |
|-----------|------------------|---------------|------|
| cls=20% + bkt=2.0x | -0.048 (gains UP) | 0.010 | sub-additive |
| cls=20% + bkt=3.0x | -0.042 (gains UP) | -0.013 (gain UP) | sub-additive |
| cls=40% + bkt=2.0x | -0.079 (gains UP) | -0.067 (gain UP) | sub-additive |
| cls=40% + bkt=3.0x | -0.072 (gains UP) | -0.025 (gain UP) | sub-additive |

**Negative individual losses mean classifier errors and BKT misspecification IMPROVE outcomes.** This is nonsensical from a systems perspective.

### Operating Envelope

Max tolerable (within 80% of baseline): ALL conditions. The system has no operating envelope - it cannot fail because it cannot succeed differentially.

## Definitive Diagnosis: The Simulation Is Invalid for Systems Evaluation

Three experiments now converge on the same conclusion:

| Experiment | Finding |
|-----------|---------|
| 07: Classifier Error | 50% error rate loses only 9% gain; false positives HELP |
| 08: BKT Fidelity | 3x parameter misspecification loses 6%; BKT selection is near-random |
| 09: End-to-End | All 80 conditions within 12% of baseline; errors IMPROVE outcomes |

### Root Causes (in order of impact)

**1. Oversaturated interaction budget**  
40 interactions / 5 concepts = 8 per concept. With p_learn = 0.15, generic instruction reaches mastery in ~15 steps. Even with random routing, each concept gets enough practice. Reducing to 10-15 total interactions would create the constraint needed for routing quality to matter.

**2. Learning is unconditional**  
`receive_instruction()` always increases p_know. There is no mechanism for wrong instruction to have zero or negative effect. In reality, teaching the wrong misconception can confuse students. The simulation should model: (a) zero learning when instruction is irrelevant, (b) negative transfer when instruction targets the wrong misconception.

**3. Test score measures p_know, not misconception resolution**  
`administer_test()` samples responses using BKT p_know (which always rises). Misconceptions only trigger when `knows=False`. As p_know approaches 1.0 over 40 interactions, misconceptions become irrelevant regardless of whether they were explicitly resolved. A test that specifically probes misconceptions (not just correctness) would be more discriminating.

**4. Knowledge graph is too small**  
5 concepts in a linear chain with 40 interactions means the routing problem is trivially easy. A larger graph (20-50 concepts) with branching prerequisites would make concept selection genuinely challenging.

**5. Classifier errors give free 2x learning bonus**  
When the classifier returns ANY misconception ID (even wrong), `receive_instruction()` applies the 2x `remediation_bonus`. The bonus should only apply when the identified misconception matches the student's actual active misconception.

## What Would Fix This

To make the simulation a valid evaluation test bed:

1. **Conditional learning**: `receive_instruction()` should check if the targeted misconception matches an active misconception. If mismatched, apply `p_T * confusion_penalty` (0.5x or less) instead of `p_T * remediation_bonus` (2x).

2. **Tighter budget**: 10-15 total interactions instead of 40. This forces the system to route efficiently.

3. **Misconception-aware testing**: Test items that specifically trigger misconceptions, not just overall correctness. Measure whether students can distinguish correct from misconception-consistent answers.

4. **Larger concept graph**: 15-25 concepts with branching prerequisites. This makes routing non-trivial.

5. **Negative transfer**: Wrong instruction should have a chance of activating or strengthening misconceptions.

## Verdict on Experiments 01-09

| Exp | Claim | Status |
|-----|-------|--------|
| 01 | Adaptive > random (d=0.33-0.48) | INVALID - simulation saturated |
| 02 | Decoupled assessment works | PARTIALLY VALID - assessment was the focus |
| 03 | Catalog > fine-tuned | INVALID - apples-to-oranges comparison |
| 04 | Greedy slightly beats Thompson | INVALID - modality selection irrelevant in saturated sim |
| 05 | IRT = categorical | INVALID - measurement irrelevant in saturated sim |
| 06 | Escalation converges at p=0.50 | PARTIALLY VALID - measured state machine, not learning |
| 07 | Classifier errors don't matter | INVALID (correct data, wrong conclusion) - sim can't discriminate |
| 08 | BKT is decorative | VALID as a critique - BKT genuinely adds nothing here |
| 09 | System can't fail | INVALID (correct data, wrong conclusion) - sim can't discriminate |

## Artifacts

- [results.json](artifacts/results.json) - full 80-condition factorial results
- [heatmap_classifier_bkt.png](artifacts/heatmap_classifier_bkt.png) - essentially monochrome (no gradient)
- [heatmap_classifier_noise.png](artifacts/heatmap_classifier_noise.png) - same
- [degradation_surface.png](artifacts/degradation_surface.png) - 3D scatter cloud with no structure
- [failure_ranking.png](artifacts/failure_ranking.png) - many conditions ABOVE baseline
