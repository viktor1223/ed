---
title: "Experiment 03: LLM-Catalog (Heuristic Proxy) vs Fine-Tuned Classifier"
description: Head-to-head evaluation of four classifier approaches on misconception detection
author: Viktor Ciroski
ms.date: 2025-07-17
ms.topic: reference
---

> **RETRACTED.** This experiment's conclusions are invalid. The comparison is
> apples-to-oranges: the fine-tuned model predicts concept-level labels while
> the catalog classifier predicts misconception IDs (a different, harder task).
> Additionally, experiments 07-09 demonstrated that the simulated student model
> cannot discriminate between classifier quality levels, so downstream
> validation would also be unreliable. The raw accuracy numbers are preserved
> below for reference, but the implied conclusion ("catalog-based classification
> is viable for new domains") is not supported.

## Overview

This experiment tests whether a zero-training catalog-based classifier (proxy
for the LLM-catalog approach described in the roadmap) can achieve practical
accuracy on misconception detection compared to the fine-tuned DistilBERT model.

The heuristic catalog classifier uses string similarity between the student's
wrong answer and known wrong-answer examples from the knowledge graph - a
conservative lower bound on what a real LLM would achieve.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Test set | `data/dataset/test.json`, 107 examples |
| Classes | 20 (19 misconceptions + "correct") |
| Concepts | 5 |
| Knowledge graph examples | 2 per misconception (38 total) |
| Seed | 42 |

## Results

### Classifier Performance

| Classifier | Misconception Accuracy | Concept Accuracy | Macro F1 | ECE | Latency |
|------------|:---------------------:|:----------------:|:--------:|:---:|--------:|
| Fine-tuned DistilBERT | N/A (concept-level only) | 89.72% | N/A | 0.30 | 264 ms |
| Heuristic Catalog | 76.64% | 98.13% | 0.7447 | 0.34 | 0.8 ms |
| Majority Baseline | 15.89% | 15.89% | 0.0137 | 0.00 | - |
| Random Baseline | 18.69% | 71.03% | 0.1653 | 0.02 | - |

**Key finding:** The classifiers operate at different granularity levels. The
fine-tuned DistilBERT predicts concept-level labels (e.g., "combining_like_terms")
at 89.72% accuracy but cannot distinguish *which* misconception within that
concept. The catalog classifier predicts specific misconception IDs (e.g.,
"clt_multiply_variables") at 76.64% accuracy, and gets the concept right
98.13% of the time.

### Per-Concept Breakdown (Catalog Classifier)

| Concept | Accuracy | N |
|---------|:--------:|:-:|
| order_of_operations | 93% | 15 |
| distributive_property | 86% | 22 |
| solving_linear_equations | 76% | 25 |
| integer_sign_ops | 73% | 22 |
| combining_like_terms | 61% | 23 |

### Catalog Scaling

| Max Examples per Misconception | Top-1 Accuracy | Concept Accuracy |
|:------------------------------:|:--------------:|:----------------:|
| 0 (no examples) | 15.89% | 15.89% |
| 1 | 76.64% | 98.13% |
| 2 (all available) | 76.64% | 98.13% |

Just one example per misconception is sufficient for the heuristic classifier
to reach its ceiling accuracy. This suggests the bottleneck is the string
similarity method, not catalog richness.

## Interpretation

1. **The catalog approach works.** 76.64% misconception-level accuracy with
   zero training data and only 2 reference examples per class validates the
   core premise of the LLM-catalog design. A real LLM reasoning semantically
   rather than by string matching should exceed this floor.

2. **The fine-tuned model has a granularity gap.** It achieves near-90% at
   concept detection but provides no actionable misconception information.
   For the intervention system (Phase 1), knowing *which* misconception
   occurred is essential for selecting the right modality.

3. **Combined approach is optimal.** A hybrid routing system (fine-tuned for
   concept detection, LLM-catalog for misconception identification) would
   exploit the strengths of both.

4. **Combining_like_terms is hardest for the catalog.** At 61%, this concept's
   misconceptions have similar wrong-answer patterns (e.g., `5x^2` vs `5x^4`
   differ by one character), making string similarity less discriminative.

## Limitations

- The heuristic catalog is a lower bound. It uses Levenshtein distance, not
  semantic understanding. A real LLM-catalog classifier would reason about
  the mathematical error, likely closing much of the 23.4% gap.
- Test set is small (107 examples). Per-concept accuracy has wide confidence
  intervals.
- The fine-tuned model was trained on concept labels, not misconception IDs.
  A model fine-tuned on misconception labels would provide a fairer comparison.
