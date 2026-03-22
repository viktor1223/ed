---
title: "Adaptive Algebra Tutoring Through Misconception Detection: System Design, Training Methodology, and Evaluation"
description: "Comprehensive technical report for the design, implementation, training, evaluation, and productionization of an AI-assisted adaptive algebra tutoring system that combines transformer-based misconception classification with Bayesian Knowledge Tracing. Written at a level of detail sufficient to reproduce the full system from scratch."
ms.date: 2026-03-22
author: "Viktor Ciroski"
ms.topic: "technical-report"
keywords:
  - intelligent tutoring systems
  - misconception detection
  - Bayesian Knowledge Tracing
  - transformer classification
  - adaptive learning
  - algebra education
  - DistilBERT
  - knowledge tracing
  - educational AI
  - personalized learning
---

## Abstract

We present a fully reproducible adaptive algebra tutoring system that detects student misconceptions from free-form text responses and dynamically adjusts instruction using Bayesian Knowledge Tracing (BKT). The system covers five algebra concepts (integer sign operations, order of operations, distributive property, combining like terms, solving linear equations) spanning 19 misconception categories that target middle-school students.

Our fine-tuned DistilBERT classifier (66M parameters) achieves 91.1% accuracy and 88.6% macro F1 on a held-out test set of 90 examples, a 156% relative improvement over a TF-IDF + Logistic Regression baseline (35.6% accuracy, 34.5% F1). Simulated student evaluations across five learner profiles demonstrate that the BKT-guided adaptive strategy correctly identifies weak concepts 80% of the time and targets remediation toward those concepts 76% of the time, outperforming both random selection and fixed-sequence baselines. Mastery estimates converge within 5-6 interaction rounds for most student profiles.

This report provides exact file structures, data schemas, training hyperparameters, BKT derivations, evaluation protocols, code walkthroughs, and cost analysis at a level of detail sufficient to reconstruct the entire system. We also lay out a concrete path from prototype to classroom-scale production, covering model serving, teacher-facing interfaces, federated learning, and multilingual expansion.

## Introduction

### The Problem

Algebra is the gateway to higher mathematics. Students who fail to build solid algebraic foundations carry misconceptions forward into geometry, calculus, and STEM coursework. The challenge is that misconceptions are specific and persistent: a student who believes "a negative times a negative is negative" will not correct that belief by receiving generic feedback ("try again"). Targeted remediation requires knowing which misconception the student holds, not merely that they answered incorrectly.

Traditional Intelligent Tutoring Systems (ITS) detect misconceptions through hand-coded production rules: for every problem, a curriculum designer writes pattern-matching logic that identifies each possible error. This approach works for constrained multiple-choice formats but collapses when students respond in free-form text, where the same misconception can manifest in hundreds of phrasings ("I got 12", "x=12", "I think the answer is 12 because I multiplied them", "twelve").

### Our Approach

We replace hand-coded error detection with a fine-tuned transformer classifier that reads a (question, student_response) pair and predicts one of 19 misconception categories. The classifier feeds into a structured Bayesian model (BKT) that maintains per-concept mastery estimates and drives an adaptive engine selecting the next instructional action.

The core contribution is an end-to-end system with three properties:

1. It classifies free-form student algebra responses into 19 misconception categories with 91.1% accuracy.
2. It updates per-concept mastery estimates using a four-parameter BKT model extended with confidence-scaled penalties.
3. It selects the next instructional action (start, practice, remediate, advance, review) through an adaptive engine that enforces prerequisite relationships.

### Scope

We scope the system to five concepts forming a linear prerequisite chain. These concepts were selected based on three criteria from the MaE benchmark (Otero, Druga, & Lan, 2025):

1. Classroom frequency: cited by over 80% of surveyed teachers
2. Data availability: sufficient labeled examples in MaE to bootstrap classifier training
3. Prerequisite connectivity: they form a connected subgraph enabling meaningful adaptive transitions

The five concepts and their 19 associated misconceptions:

| Concept | Level | Misconceptions | MaE IDs |
|---------|-------|----------------|---------|
| Integer & Sign Operations | 1 | sign_sum_negatives, sign_neg_times_neg, sign_sub_negative, sign_always_subtract_smaller | MaE06-10 |
| Order of Operations | 2 | oo_left_to_right, oo_exponent_after_add, oo_parentheses_ignored | MaE20-22, MaE31 |
| Distributive Property | 3 | dist_first_term_only, dist_square_over_addition, dist_sign_error_negative, dist_drop_parens | MaE31-34 |
| Combining Like Terms | 4 | clt_combine_unlike, clt_multiply_variables, clt_constant_as_variable, clt_add_exponents | MaE45-48 |
| Solving Linear Equations | 5 | leq_reverse_operation, leq_divide_wrong_direction, leq_subtract_wrong_side, leq_move_without_sign_change | MaE49-55 |

## System Architecture

The system has four layers: a domain knowledge layer (the knowledge graph), a learner model layer (BKT), a classification layer (DistilBERT), and an orchestration layer (the adaptive session engine). Each layer is implemented as a separate Python module with a well-defined interface so that any component can be replaced independently.

### Repository Structure

```
ed/
├── data/
│   ├── knowledge_graph.json          # Domain graph: concepts, misconceptions, BKT params
│   ├── problem_bank.json             # 28 problems for adaptive problem selection
│   ├── dataset/
│   │   ├── train.json                # 479 examples (414 with non-null misconception_id)
│   │   ├── val.json                  # 101 examples (91 usable)
│   │   ├── test.json                 # 107 examples (90 usable)
│   │   ├── full.json                 # 738 pre-dedup merged set
│   │   └── dataset_card.json         # Dataset documentation
│   └── annotated-bibliography.md     # 28 literature sources
├── src/
│   ├── knowledge_graph.py            # KnowledgeGraph, StudentState, next_action
│   ├── classifier.py                 # MisconceptionClassifier inference wrapper
│   ├── train_classifier.py           # HuggingFace Trainer training script
│   ├── baseline_tfidf.py             # TF-IDF + LogReg baseline
│   ├── build_dataset.py              # Dataset assembly + synthetic generation
│   ├── tutor_session.py              # Integration: classifier + KG + BKT + hints
│   ├── tutor_cli.py                  # Interactive CLI
│   ├── evaluate.py                   # Phase 5 evaluation suite
│   └── validate_dataset.py           # Data quality checks
├── models/
│   └── classifier/
│       ├── best/                     # Fine-tuned DistilBERT checkpoint
│       └── training_results.json     # Training metrics
├── results/
│   ├── baseline_tfidf.json           # Baseline results
│   └── phase5_evaluation.json        # Full evaluation results
├── tests/
│   ├── test_knowledge_graph.py       # 31 unit tests
│   └── smoke_test.py                 # End-to-end integration test
├── web/
│   └── index.html                    # Interactive knowledge graph visualization
├── IRB.md                            # Original research proposal
├── PLAYBOOK.md                       # 7-phase implementation playbook
└── TECHNICAL_REPORT.md               # This document
```

### Layer 1: Knowledge Graph

The knowledge graph is stored as a single JSON file (`data/knowledge_graph.json`) and loaded into Python dataclasses at runtime. The schema is:

```json
{
  "metadata": {
    "version": "1.0.0",
    "mastery_threshold": 0.85,
    "mastery_initial": 0.5
  },
  "concepts": [
    {
      "id": "integer_sign_ops",
      "name": "Integer & Sign Operations",
      "description": "...",
      "level": 1,
      "prerequisites": [],
      "mae_ids": ["MaE06", "MaE07", "MaE08", "MaE09", "MaE10"],
      "bkt_params": {
        "p_init": 0.15,
        "p_learn": 0.15,
        "p_guess": 0.10,
        "p_slip": 0.10
      },
      "misconceptions": [
        {
          "id": "sign_sum_negatives",
          "label": "Sum of negatives becomes positive",
          "description": "Student adds two negative numbers and gets a positive result.",
          "examples": [
            {"problem": "Simplify: -6 - 3", "wrong": "9", "correct": "-9"}
          ]
        }
      ]
    }
  ],
  "edges": [
    {"from": "integer_sign_ops", "to": "order_of_operations", "type": "prerequisite"}
  ]
}
```

The prerequisite chain enforces a strict ordering:

```
integer_sign_ops (L1) → order_of_operations (L2) → distributive_property (L3)
    → combining_like_terms (L4) → solving_linear_equations (L5)
```

This linear topology is the simplest structure that supports prerequisite gating. When a student struggles with combining like terms, the system checks mastery of the distributive property (the prerequisite) and remediates there if needed. The linearity constraint is deliberate: branching prerequisite paths would increase the BKT state space and evaluation test matrix without foundational evidence that the additional complexity improves learning outcomes.

The Python representation uses three dataclasses:

```python
@dataclass
class Misconception:
    id: str           # e.g. "sign_sum_negatives"
    label: str        # Human-readable, e.g. "Sum of negatives becomes positive"
    description: str  # Detailed explanation
    examples: list[dict[str, str]]  # Worked examples with problem/wrong/correct

@dataclass
class Concept:
    id: str
    name: str
    description: str
    level: int                    # 1-5, determines prerequisite ordering
    prerequisites: list[str]      # IDs of prerequisite concepts
    mae_ids: list[str]            # MaE dataset misconception IDs
    bkt_params: dict[str, float]  # p_init, p_learn, p_guess, p_slip
    misconceptions: list[Misconception]

@dataclass
class KnowledgeGraph:
    concepts: dict[str, Concept]
    edges: list[dict[str, str]]
    mastery_threshold: float = 0.85
    mastery_initial: float = 0.5
```

Key methods on `KnowledgeGraph`:

- `from_json(path)`: Loads the JSON file and constructs the graph.
- `misconception_to_concept(misconception_id)`: Reverse lookup from a misconception ID to its parent concept. Used by the adaptive engine when the classifier detects a misconception to determine which concept to remediate.
- `label_list()`: Returns all 20 classification labels (19 misconception IDs + "correct"), sorted alphabetically. This sorting ensures consistent label-to-index mapping across training and inference.
- `concepts_by_level()`: Returns concepts sorted by level, used by the adaptive engine for progression.

### Layer 2: Bayesian Knowledge Tracing

#### Mathematical Foundation

For each student-concept pair, we maintain a mastery probability $P(L_t)$ estimating whether the student has learned the concept by interaction $t$. BKT is a two-state Hidden Markov Model where the hidden state is "learned" vs "unlearned," and the observed emission is "correct" vs "incorrect."

The model has four concept-level parameters:

| Parameter | Symbol | Meaning | Our Value |
|-----------|--------|---------|-----------|
| Initial knowledge | $P(\text{init})$ | Prior probability the student already knows the concept | 0.05-0.15 |
| Learning rate | $P(T)$ | Probability of transitioning from unlearned to learned on any given opportunity | 0.10-0.20 |
| Guess rate | $P(G)$ | Probability of answering correctly without having learned | 0.05-0.10 |
| Slip rate | $P(S)$ | Probability of answering incorrectly despite having learned | 0.10-0.15 |

The posterior update on a correct observation uses Bayes' theorem:

$$P(L_t \mid \text{correct}) = \frac{P(\text{correct} \mid L_t) \cdot P(L_t)}{P(\text{correct})}$$

Expanding the likelihood and marginal:

$$P(\text{correct} \mid L_t) = 1 - P(S) \quad \text{(learned student does not slip)}$$
$$P(\text{correct} \mid \neg L_t) = P(G) \quad \text{(unlearned student guesses)}$$
$$P(\text{correct}) = P(L_t) \cdot (1 - P(S)) + (1 - P(L_t)) \cdot P(G)$$

Therefore:

$$P(L_t \mid \text{correct}) = \frac{P(L_{t-1}) \cdot (1 - P(S))}{P(L_{t-1}) \cdot (1 - P(S)) + (1 - P(L_{t-1})) \cdot P(G)}$$

For an incorrect observation:

$$P(L_t \mid \text{incorrect}) = \frac{P(L_{t-1}) \cdot P(S)}{P(L_{t-1}) \cdot P(S) + (1 - P(L_{t-1})) \cdot (1 - P(G))}$$

After the posterior update, we apply a learning transition (the student may learn from the interaction regardless of correctness):

$$P(L_t) \leftarrow P(L_t \mid \text{obs}) + (1 - P(L_t \mid \text{obs})) \cdot P(T)$$

This transition ensures mastery can only increase (or stay the same) through the learning opportunity itself. The posterior update handles the evidence; the transition handles the learning.

#### Confidence-Scaled Penalty Extension

Standard BKT treats all incorrect responses identically. Our system extends this by using the classifier's confidence score to modulate the mastery penalty on incorrect answers:

```python
if not correct and confidence > 0.5:
    penalty = 0.05 * confidence
    p_new = max(0.01, p_new - penalty)
```

The rationale: a classifier that reports 0.9 confidence in a specific misconception likely identified a genuine, systematic error. A classifier at 0.3 confidence may indicate a careless mistake, partial understanding, or an ambiguous response. By scaling the penalty by confidence, we:

- Avoid over-penalizing students for careless errors (low confidence = small penalty)
- Appropriately weight genuine misconceptions (high confidence = larger penalty)
- Maintain a floor at 0.01 to prevent mastery from reaching zero (which would make recovery require many correct answers)

The 0.5 confidence threshold was chosen because uniformly random prediction across 20 classes would yield 0.05 confidence per class, and our model averages 0.33. The threshold triggers the penalty only when the model is meaningfully more confident than its baseline, indicating a recognized pattern rather than noise.

#### Per-Concept Parameter Values

| Concept | $P(\text{init})$ | $P(T)$ | $P(G)$ | $P(S)$ | Rationale |
|---------|-------------------|---------|---------|---------|-----------|
| Integer sign ops | 0.15 | 0.15 | 0.10 | 0.10 | Moderate prior (students have some exposure), standard learning rate |
| Order of operations | 0.10 | 0.10 | 0.10 | 0.15 | Lower prior (commonly confused), higher slip (procedural errors are common) |
| Distributive property | 0.05 | 0.15 | 0.05 | 0.10 | Low prior (frequently misunderstood), low guess rate (hard to guess correctly) |
| Combining like terms | 0.10 | 0.20 | 0.05 | 0.10 | Higher learning rate (pattern recognition clicks quickly once taught) |
| Solving linear equations | 0.10 | 0.15 | 0.05 | 0.15 | Low guess rate (multi-step), higher slip (procedural complexity) |

These values were initialized from ranges reported in Baker et al. (2008) for algebra domains. In production, they should be fit per-concept from empirical student data using expectation-maximization.

#### Mastery Threshold

A concept is considered mastered when $P(L_t) \geq 0.85$. We chose 0.85 over the more commonly used 0.95 because our prototype operates with uniform BKT parameters; a stricter threshold would require more interactions to achieve and could frustrate students in a prototype where parameters have not been calibrated to actual learning curves.

#### Implementation Walkthrough

The `StudentState` class maintains per-student state:

```python
class StudentState:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        # Initialize mastery from BKT p_init parameter per concept
        self.mastery = {
            cid: kg.concepts[cid].bkt_params.get("p_init", kg.mastery_initial)
            for cid in kg.concepts
        }
        self.attempts = {cid: 0 for cid in kg.concepts}

    def update(self, concept_id, correct, confidence=1.0):
        params = self.kg.concepts[concept_id].bkt_params
        p_L = self.mastery[concept_id]
        p_G = params.get("p_guess", 0.10)
        p_S = params.get("p_slip", 0.10)
        p_T = params.get("p_learn", 0.15)

        # Posterior update (Bayes)
        if correct:
            p_correct = p_L * (1 - p_S) + (1 - p_L) * p_G
            p_L_given_obs = (p_L * (1 - p_S)) / p_correct
        else:
            p_incorrect = p_L * p_S + (1 - p_L) * (1 - p_G)
            p_L_given_obs = (p_L * p_S) / p_incorrect

        # Learning transition
        p_new = p_L_given_obs + (1 - p_L_given_obs) * p_T

        # Confidence-scaled penalty for high-confidence incorrect predictions
        if not correct and confidence > 0.5:
            penalty = 0.05 * confidence
            p_new = max(0.01, p_new - penalty)

        self.mastery[concept_id] = p_new
        self.attempts[concept_id] += 1
        return p_new
```

To verify BKT correctness, we maintain 31 unit tests covering initialization, update mechanics, mastery thresholds, prerequisite gating, the adaptive engine's action selection, and edge cases (empty graphs, single concepts, all concepts mastered).

### Layer 3: Misconception Classifier

#### Architecture Choice

We evaluated two transformer architectures:

| Model | Parameters | Architecture Distinction | Training Outcome |
|-------|------------|--------------------------|------------------|
| DeBERTa-v3-base | 86M | Disentangled attention (separate content and position embeddings) | NaN gradients on Apple MPS |
| DistilBERT-base-uncased | 66M | Knowledge-distilled 6-layer BERT | Trained successfully, 90.1% val accuracy |

DeBERTa-v3-base's disentangled attention mechanism uses relative position encoding that interacts poorly with Apple MPS's mixed-precision arithmetic. The NaN gradients appear during the backward pass and are not recoverable with gradient clipping. This is a known limitation of MPS (not the model itself); DeBERTa should train correctly on CUDA hardware.

We proceeded with DistilBERT because it met our accuracy targets and trained reliably on available hardware. DistilBERT is a 6-layer, 768-hidden-dimension, 12-attention-head transformer distilled from BERT-base-uncased using a combination of language modeling loss, distillation loss, and cosine embedding loss during pre-training. Despite having 40% fewer parameters than BERT-base, it retains 97% of BERT's language understanding capabilities on GLUE benchmarks (Sanh et al., 2019).

#### Input Formatting

Each training example is formatted as:

```
Question: {problem_text}
Student answer: {student_response}
```

This two-field format was chosen over alternatives (concatenation with [SEP], single-text, JSON-structured) because:

1. It provides a clear separator ("Student answer:") that the model can attend to for locating the student's response.
2. It preserves the ordering relationship (question context precedes response).
3. It matches common instruction-following formats that DistilBERT encounters in pre-training data.

Maximum sequence length is 256 tokens. The longest training example is under 100 tokens, so this provides substantial padding for future concept expansion where problem descriptions may be longer.

#### Label Space

The classifier maps inputs to one of 20 classes:

- 19 misconception IDs (e.g., `sign_sum_negatives`, `dist_first_term_only`)
- 1 `correct` class

Labels are sorted alphabetically and assigned integer indices 0-19. This sorting is deterministic and is enforced by `KnowledgeGraph.label_list()`, which generates the label-to-ID mapping used by both the training script and the inference wrapper. Any change to the misconception taxonomy requires rerunning this method to produce a consistent mapping.

#### Training Methodology

##### Hyperparameters

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Base model | `distilbert-base-uncased` | Reliable on MPS; sufficient capacity for 20-class problem |
| Max epochs | 15 | Upper bound; early stopping typically triggers at epoch 5-7 |
| Learning rate | $2 \times 10^{-5}$ | Standard for transformer fine-tuning; within the $1 \times 10^{-5}$ to $5 \times 10^{-5}$ range recommended by Devlin et al. (2019) |
| Batch size | 16 | Fits in MPS memory (8GB Apple M-series); 32 works on 8GB CUDA |
| Max sequence length | 256 tokens | Comfortably covers all examples; headroom for expansion |
| Warmup ratio | 0.1 | 10% of training steps use linear warmup from 0 to lr |
| Label smoothing | 0.1 | Softens hard targets to $[0.005, 0.905]$ distribution; regularizes against overconfident predictions |
| Early stopping patience | 3 epochs | Stops training if val F1 (macro) does not improve for 3 consecutive epochs |
| Metric for best model | f1_macro | Prioritizes balanced performance across all 19 misconception classes, not just high-frequency ones |
| Weight decay | 0.0 (AdamW default) | Not explicitly tuned; default is sufficient for this dataset size |
| Optimizer | AdamW | Default HuggingFace Trainer optimizer |
| Random seed | 42 | Fixed for reproducibility across data splitting, model initialization, and shuffling |
| FP16 | True on CUDA, False on MPS/CPU | MPS does not reliably support mixed-precision; CUDA benefits significantly |

##### Training Script Walkthrough

The training pipeline (`src/train_classifier.py`) follows this sequence:

1. Load 20 labels from the knowledge graph (`load_labels()`), ensuring consistent label-to-ID mapping.
2. Load train and val splits from JSON files, filtering out examples where `misconception_id is None` (65 MaE examples that map to our concepts but not to our specific 19 misconception categories).
3. Tokenize each example as `"Question: {q}\nStudent answer: {r}"`.
4. Initialize a `AutoModelForSequenceClassification` from the DistilBERT checkpoint with `num_labels=20`, passing `label2id` and `id2label` dicts that get baked into the model config.
5. Configure `TrainingArguments` with the hyperparameters above.
6. Train using the HuggingFace `Trainer` with `EarlyStoppingCallback`.
7. After training, produce a `classification_report` on the validation set and save the best model checkpoint to `models/classifier/best/`.

##### Dataset Class

Custom `MisconceptionDataset` wrapping `torch.utils.data.Dataset`:

```python
class MisconceptionDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }
```

We use `padding="max_length"` (pad to 256 tokens) rather than dynamic padding per-batch because the dataset is small enough that the extra padding tokens are negligible in memory and simplify the data loading pipeline.

##### Metric Function

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1}
```

Macro F1 is the primary metric because it gives equal weight to all 19 misconception classes, regardless of their training set size. Weighted F1 would favor high-frequency classes and potentially mask poor performance on rare misconceptions.

##### Training Results

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 90.1% | 91.1% |
| F1 (macro) | 88.2% | 88.6% |
| F1 (weighted) | - | 89.8% |

That test performance slightly exceeds validation performance indicates the model is not overfitting; the marginal improvement is within expected random variation for sets of this size.

#### Inference Wrapper

The `MisconceptionClassifier` class in `src/classifier.py` provides a clean inference interface:

```python
class MisconceptionClassifier:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        # Auto-detect best device: CUDA > MPS > CPU
        self.model.to(self.device)

    def predict(self, question, student_response):
        text = f"Question: {question}\nStudent answer: {student_response}"
        enc = self.tokenizer(text, truncation=True, padding="max_length",
                            max_length=256, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_idx = probs.argmax().item()
        return {
            "label": self.id2label[pred_idx],
            "confidence": probs[pred_idx].item(),
            "all_probs": {self.id2label[i]: p.item() for i, p in enumerate(probs)},
        }
```

The `all_probs` dictionary returns the full 20-class probability distribution. This is used by the BKT confidence-scaled penalty and could also power a teacher-facing visualization showing the model's uncertainty across misconception categories.

### Layer 4: Adaptive Session Engine

#### Action Selection Algorithm

The `next_action(state, kg)` function implements a priority-chain decision procedure:

```
Priority 1: COLD START
  If no concepts have been attempted →
    Select the lowest-level unmastered concept.
    Action: "start"

Priority 2: REMEDIATE
  Scan concepts by level (lowest first).
  If any concept has been attempted AND mastery < threshold →
    Select it for remediation.
    Action: "remediate"

Priority 3: PROGRESS
  Scan concepts by level (lowest first).
  If any concept has NOT been attempted AND its prerequisites are all mastered →
    Select it for advancement.
    Action: "progress"

Priority 4: REVIEW
  All concepts are mastered.
  Select the concept with the lowest mastery for review/maintenance.
  Action: "review"
```

The by-level scanning ensures that remediation targets foundational concepts first. If a student fails a linear equations problem and the classifier detects a distributive property misconception, the next_action engine will surface the distributive property for remediation because it appears earlier in the level ordering and its mastery has dropped below threshold.

#### Problem Selection

Within the selected concept, the problem bank provides 5-6 problems at three difficulty levels (easy, medium, hard). The session engine:

1. Filters out problems that appeared in the last 5 interactions (recency filter)
2. If all problems are in the recency window, resets to the full set
3. Selects randomly from the available set

This prevents the student from seeing the exact same problem consecutively while keeping the selection stochastic enough to test true understanding rather than rote memorization of specific answers.

#### Hint System

Each of the 19 misconceptions has a hand-written, targeted hint stored in the `HINTS` dictionary within `tutor_session.py`. Hints are pedagogically structured to:

1. State the rule the student violated
2. Provide a concrete correction strategy
3. Give an example that demonstrates the correct approach

For instance, for `dist_first_term_only`:

> "When distributing, multiply the factor by EVERY term inside the parentheses, not just the first one. In 2(x + 3), both x and 3 get multiplied by 2."

Hints are displayed only when the classifier identifies a specific misconception (not for the generic "correct" label or when confidence is very low).

#### Answer Matching

The `_check_correct` method in `TutorSession` determines whether a student's free-form text response matches the expected correct answer. This is separate from the classifier (which predicts misconception type, not correctness).

The matching pipeline:

```python
def _check_correct(student_text, correct_answer):
    1. normalize(s):
       - Lowercase
       - Remove all whitespace
       - Convert Unicode math symbols (×→*, ÷→/, ²→^2, etc.)

    2. extract_value(s):
       - Apply normalize()
       - Strip common prose phrases:
         "I think the answer is", "my answer is", "I got", "the answer is"
       - Strip variable assignment prefixes: "x=", "m=", etc.

    3. Compare:
       a. If extract_value(student) == extract_value(correct) → True
       b. If correct_val appears in normalized student text with safe boundaries
          (character before the match is not a digit, period, or minus sign) → True
       c. Otherwise → False
```

The boundary check in step 3b prevents `"-4"` from matching `"4"` (the minus sign before "4" is in the blocked character set) while allowing `"x = 5"` to match `"5"` (the space before "5" is safe).

We tested this against 11 edge cases:

| Student Input | Correct Answer | Expected | Result |
|---------------|----------------|----------|--------|
| "5" | "x = 5" | Match | Pass |
| "x=5" | "x = 5" | Match | Pass |
| "I think the answer is 5" | "x = 5" | Match | Pass |
| "x = 5" | "x = 5" | Match | Pass |
| "The answer is definitely 5" | "x = 5" | Match | Pass |
| "42" | "42" | Match | Pass |
| "wrong" | "42" | No match | Pass |
| "-4" | "4" | No match | Pass |
| "4" | "-4" | No match | Pass |
| "-9" | "-9" | Match | Pass |
| "x = -9" | "-9" | Match | Pass |

A limitation: this approach is string-based, not algebraic. Equivalent expressions like "2x + 4" and "4 + 2x" would not match. For the current problem bank (which uses numeric or simple single-variable answers), this is acceptable. Expanding to support algebraic equivalence would require integrating a Computer Algebra System (CAS) like SymPy.

## Data Pipeline

### Step 1: MaE Dataset Ingestion

The MaE dataset (Otero, Druga, & Lan, 2025) is hosted on HuggingFace at `nanote/algebra_misconceptions` under the MIT license. It contains 220 examples across 55 misconception categories for middle-school algebra.

We download using the `huggingface_hub` library:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    "nanote/algebra_misconceptions", "data/data.json", repo_type="dataset"
)
```

Filtering to our 23 target MaE IDs (mapped to our 5 concepts) yields 92 examples. Each example contains:

- `Misconception ID`: MaE's category ID (e.g., "MaE06")
- `Question`: The algebra problem text
- `Incorrect Answer`: The student's wrong answer
- `Correct Answer`: The expected answer
- `Explanation`: Why the answer is wrong

We then map each MaE ID to our internal concept and misconception taxonomy:

```python
CONCEPT_MAE_MAP = {
    "integer_sign_ops": {
        "mae_ids": ["MaE06", "MaE07", "MaE08", "MaE09", "MaE10"],
        "topic_filter": "Number operations",
    },
    "order_of_operations": {
        "mae_ids": ["MaE20", "MaE21", "MaE22"],
        "topic_filter": "Number operations",
    },
    "distributive_property": {
        "mae_ids": ["MaE31", "MaE32", "MaE33", "MaE34"],
        "topic_filter": "Properties of numbers and operations",
    },
    "combining_like_terms": {
        "mae_ids": ["MaE45", "MaE46", "MaE47", "MaE48"],
        "topic_filter": "Variables, expressions, and operations",
    },
    "solving_linear_equations": {
        "mae_ids": ["MaE49", "MaE50", "MaE51", "MaE52", "MaE53", "MaE54", "MaE55"],
        "topic_filter": "Equations and inequalities",
    },
}
```

Of the 92 MaE examples, 65 map to a concept but not to any specific one of our 19 misconception IDs (some MaE IDs cover misconceptions we excluded or grouped differently). These 65 examples have `misconception_id: null` in the dataset and are excluded from classifier training and evaluation.

### Step 2: Synthetic Data Generation

With only 27 usable MaE examples (92 minus 65 with null misconception IDs), synthetic augmentation is essential. The generation pipeline in `src/build_dataset.py` uses randomized templates for each misconception.

#### Template Structure

Each generator function produces a tuple: `(question, student_response, wrong_answer, correct_answer)`.

Example for the `sign_sum_negatives` misconception:

```python
def _gen_sign_sum_neg():
    a = random.randint(2, 12)
    b = random.randint(2, 12)
    question = f"Simplify: -{a} - {b}"
    wrong = str(a + b)            # Student incorrectly gets positive
    correct = str(-(a + b))       # Correct answer is negative
    return question, _pick_phrasing(question, wrong), wrong, correct
```

Example for the `dist_first_term_only` misconception:

```python
def _gen_dist_first_only():
    coeff = random.randint(2, 9)
    v = random.choice(["x", "y", "n", "m", "a", "b", "k", "t"])
    const = random.randint(1, 10)
    question = f"Expand: {coeff}({v} + {const})"
    wrong = f"{coeff}{v} + {const}"           # Only multiplied first term
    correct = f"{coeff}{v} + {coeff * const}"  # Correctly distributed
    return question, _pick_phrasing(question, wrong), wrong, correct
```

There are 19 generator functions (one per misconception) plus a correct-answer generator per concept. Each generator:

1. Samples numeric parameters from a constrained random range (typically 1-12 for operands, 2-9 for coefficients)
2. Computes the correct answer algebraically
3. Computes the misconception-consistent wrong answer by applying the specific error pattern
4. Wraps the wrong answer in a randomly chosen phrasing style

The numeric ranges are chosen to produce answers within the integer range that middle-school students would encounter. We avoid edge cases (0, 1, very large numbers) that could create degenerate problems where the wrong answer equals the right answer.

#### Phrasing Styles

Six phrasing registers create linguistic diversity:

```python
PHRASING = {
    "math_only":  lambda q, a: f"{a}",
    "short":      lambda q, a: f"I got {a}",
    "with_work":  lambda q, a: f"My answer is {a}. I worked it out step by step.",
    "uncertain":  lambda q, a: f"I think the answer is {a} but I'm not sure",
    "confident":  lambda q, a: f"The answer is {a}",
    "explain":    lambda q, a: f"I solved it and got {a}. Here's what I did:",
}
```

This variation is critical for two reasons:

1. Real students do not respond uniformly. Some type bare numbers, some explain their reasoning, some express uncertainty. A classifier trained only on clean answers would fail when a student writes "I think maybe it's 12?"
2. It forces the model to learn the mathematical relationship between the question and the answer value, rather than memorizing the syntactic pattern of the response.

#### Generation Volume

Each misconception generator produces approximately 30-35 examples (varied by the numeric parameter ranges), yielding roughly 595 synthetic examples across all 19 misconceptions. Combined with the 92 MaE examples, the pre-deduplication corpus is 738 examples.

### Step 3: Deduplication and Splitting

#### Fingerprinting

Each example receives a fingerprint constructed from the normalized question text and incorrect answer:

```python
fingerprint = hashlib.md5(
    f"{normalize(question)}::{normalize(incorrect_answer)}".encode()
).hexdigest()
```

Duplicates (same question and incorrect answer with different phrasings) are removed, reducing 738 to 687 unique examples.

#### Stratified Splitting

The 687 examples are split 70/15/15 stratified by `concept_id` using a random shuffle with seed 42:

| Split | Total | With non-null misconception_id |
|-------|-------|--------------------------------|
| Train | 479 | 414 |
| Val | 101 | 91 |
| Test | 107 | 90 |

Cross-split leakage is verified at zero by checking that no fingerprint appears in more than one split.

### Problem Bank

Separate from the training data, 28 problems serve the adaptive engine during tutoring sessions:

- 5-6 problems per concept
- Three difficulty levels: easy, medium, hard
- Each problem has: `problem_id`, `concept`, `difficulty`, `problem_text`, `correct_answer`

These problems are distinct from training examples and are used only at inference time. They represent the "tests" the tutor administers, while the training examples represent historical student responses used to teach the classifier.

## Baseline: TF-IDF + Logistic Regression

To validate that transformer-level understanding is needed (and that surface-level features are insufficient), we train a TF-IDF baseline:

```python
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
```

Configuration:

- Up to 5,000 features using unigram and bigram token counts
- Sublinear TF scaling ($1 + \log(\text{tf})$) to dampen the impact of high-frequency terms
- L2-regularized logistic regression with one-vs-rest multiclass
- No hyperparameter tuning (deliberately kept simple to serve as a lower bound)

Results:

| Metric | Val | Test |
|--------|-----|------|
| Accuracy | 23.1% | 35.6% |
| F1 (macro) | 21.4% | 34.5% |

The baseline's near-chance performance on 19 classes (random would be 5.3%) confirms that bag-of-words features capture some signal but cannot reliably distinguish misconceptions. The improvement from val to test suggests the val set may be slightly harder or noisier, not that the model generalized well.

The 156% improvement from baseline to DistilBERT justifies the transformer complexity: the task genuinely requires semantic understanding of the relationship between the question and the student's response.

## Evaluation Results

### Classification on Held-Out Test Set

| Metric | DistilBERT | TF-IDF + LogReg | Relative Improvement |
|--------|------------|------------------|----------------------|
| Test accuracy | 91.1% (82/90) | 35.6% (32/90) | +155.6% |
| Test F1 (macro) | 88.6% | 34.5% | +156.8% |
| Test F1 (weighted) | 89.8% | - | - |
| Mean confidence | 0.327 | - | - |

#### Per-Concept Accuracy

| Concept | Accuracy | Correct/Total | Notes |
|---------|----------|---------------|-------|
| Integer sign operations | 100.0% | 20/20 | Perfect classification across all 4 misconceptions |
| Order of operations | 100.0% | 15/15 | Perfect classification across all 3 misconceptions |
| Combining like terms | 100.0% | 21/21 | Perfect classification across all 4 misconceptions |
| Distributive property | 76.5% | 13/17 | 4 errors within concept-internal confusion pair |
| Solving linear equations | 76.5% | 13/17 | 4 errors within concept-internal confusion pair |

Three of five concepts achieve perfect classification. The two lower-performing concepts exhibit specific confusion patterns.

#### Error Analysis: The Eight Misclassifications

All eight errors fall into two confusion pairs:

##### Confusion Pair 1: `dist_drop_parens` vs `dist_first_term_only`

3 `dist_drop_parens` examples misclassified as `dist_first_term_only`, 1 `dist_first_term_only` misclassified as `dist_drop_parens`.

These misconceptions are semantically almost identical. "Dropping parentheses without distributing" and "distributing to the first term only" both produce the answer `coeff * var + constant` (wrong because the constant was not multiplied). For the expression `5(x + 3)`, both misconceptions yield `5x + 3`. The distinction between them is the student's reasoning process, not the observable output. A single-answer format cannot disambiguate them without additional evidence (such as the student's work shown step by step).

Three possible remediations:

1. Merge the two categories into a single "incomplete distribution" class
2. Collect step-by-step work data that distinguishes the two reasoning paths
3. Implement hierarchical classification: first classify at the concept level, then disambiguate within concept

We recommend option 1 for immediate productionization (simpler, fewer classes, same remediation hint) and option 3 for a research extension.

##### Confusion Pair 2: `leq_reverse_operation` vs `leq_move_without_sign_change`

4 `leq_reverse_operation` examples misclassified as `leq_move_without_sign_change`.

Both misconceptions involve incorrect manipulation of equation terms. The distinction: "reverse operation" means the student applied the wrong operation entirely (subtracting when they should add), while "move without sign change" means they moved a term to the other side without flipping its sign. For simple one-step equations, the numeric output can be identical. The student says "m + 2 = 19, so m = 19 + 2 = 21" (either they added instead of subtracting, or they moved +2 without changing its sign, producing +2 on the right side).

The same three remediations apply. For linear equations, merger is the pragmatic choice: both misconceptions receive the same hint ("use the opposite operation on both sides").

##### Impact of Merging on Metrics

If we merge each confusion pair into a single class (reducing from 19 to 17 misconception categories), the test set accuracy rises to 100% (all 8 errors become intra-class confusions that are no longer counted as errors). The trade-off is reduced diagnostic granularity: the system would report "incomplete distribution" rather than distinguishing "drop parens" from "first term only." From a pedagogical standpoint, both mapped errors receive the same corrective hint, so the loss is primarily in research-grade misconception logging rather than tutoring effectiveness.

#### Confidence Distribution

Mean classifier confidence is 0.327 across all 90 test predictions. By comparison:

- Uniform random across 20 classes: 0.05
- Perfect classifier with sharp logits: approach 1.0
- Our model: 0.327 (roughly 6.5x random)

The moderate confidence suggests the model distributes probability mass across related misconception categories rather than concentrating on a single prediction. We verified that for correctly classified examples, confidence averages 0.34, while for misclassified examples it averages 0.25. The gap is small but in the expected direction, and our BKT confidence-scaled penalty exploits this signal.

The low absolute confidence reflects genuine uncertainty in the task: a student who writes "9" in response to "Simplify: -6 - 3" could be exhibiting `sign_sum_negatives` (dropped both negatives) or could have made a sign error in a different step. The model's probability distribution captures this ambiguity, and the BKT layer handles it gracefully by using the confidence to modulate the penalty rather than treating all incorrect answers as high-certainty misconceptions.

### Ablation: Topic Metadata

We tested whether prepending the concept name to the input changes classification performance. The model was trained without topic metadata, so this is an out-of-distribution test:

| Condition | Accuracy | F1 (macro) |
|-----------|----------|------------|
| Standard (trained format) | 91.1% | 88.6% |
| With topic prefix: "[Distributive Property] Expand: 3(x+2)" | 78.9% | 71.1% |

Topic metadata decreased accuracy by 12.2 percentage points. Two implications:

1. The model learns misconception patterns from mathematical content, not concept keywords. It does not associate "Expand" with distributive property errors; it analyzes the relationship between the question structure and the response value.
2. Input format sensitivity is real. Any change to the input template (adding metadata, changing separators, adding instructions) requires retraining. Deploying the model with a different input format than it was trained on will degrade performance.

A topic-aware variant would require training with topic metadata included in randomly sampled examples (perhaps 50% with metadata, 50% without) so the model learns to use the signal when present while remaining reliable when absent.

### Simulated Student Evaluation

#### Methodology

We created five student profiles, each defined by a per-concept probability of answering correctly:

| Profile | Description | Sign | OoO | Dist | CLT | LEQ |
|---------|-------------|------|-----|------|-----|-----|
| A: Strong, one weak | Good except distributive property | 95% | 90% | 20% | 85% | 80% |
| B: Weak overall | Below mastery on everything | 40% | 30% | 25% | 20% | 15% |
| C: Mixed | Strong arithmetic, weak algebra | 90% | 85% | 40% | 35% | 25% |
| D: Ceiling | Near-perfect everywhere | 95% | 95% | 90% | 95% | 90% |
| E: Random noise | Coin flip on everything | 50% | 50% | 50% | 50% | 50% |

Each profile was simulated through 20 rounds of tutoring using three strategies:

- Adaptive: uses `next_action()` (BKT-guided concept selection)
- Random: uniform random concept selection
- Fixed sequence: round-robin through concepts in level order

Each profile-strategy combination ran 10 times with different random seeds (42-51) to produce stable averages.

#### Concept Identification Accuracy

Measures whether the system's bottom-2 mastery-ranked concepts match the student's true bottom-2 weakness-ranked concepts.

| Strategy | Accuracy (avg across profiles) |
|----------|-------------------------------|
| Adaptive | 80% |
| Random | 60% |
| Fixed sequence | 60% |

The adaptive strategy's 20-point advantage comes from concentrating observations on weak concepts: by spending more rounds on lower-mastery areas, BKT receives more data points and produces higher-fidelity estimates.

Profile-specific behavior:

- Profile A (strong, one weak): Adaptive achieves 100%, confirming it reliably identifies a single isolated weakness.
- Profile B (weak overall): Adaptive achieves 100%. Even though all concepts are weak, the system correctly identifies the two weakest.
- Profile D (ceiling): Adaptive drops to 50%. When all concepts are at 90-95% correct rate, distinguishing the "weakest" from 20 observations is statistically impossible. This is correct behavior: there is nothing to remediate.
- Profile E (random noise): 50% for adaptive. Uniform noise provides no signal for BKT to detect.

#### Weak-Concept Targeting Rate

For profiles with genuinely weak concepts (correct rate below 50%), what fraction of tutoring rounds does each strategy spend on those concepts?

| Strategy | Targeting rate (avg) |
|----------|---------------------|
| Adaptive | 76% |
| Random | 61% |
| Fixed sequence | 60% |

The adaptive strategy allocates three-quarters of its effort to weak concepts. The baselines are near 60% because with 5 concepts and 3 weak ones, random selection hits a weak concept 60% of the time by chance.

The 16-point improvement from adaptive targeting means students receive approximately 3 additional practice opportunities on their weakest areas per 20-round session compared to non-adaptive approaches.

### Mastery Convergence

We measured the round at which no concept's mastery estimate changes by more than 0.05 in a single step:

| Profile | Convergence round | Real-time equivalent |
|---------|-------------------|----------------------|
| Strong, one weak | 5 | ~75 seconds |
| Weak overall | 6 | ~90 seconds |
| Mixed | 5 | ~75 seconds |
| Ceiling | 11 | ~165 seconds |
| Random noise | 6 | ~90 seconds |

Real-time estimates assume 15 seconds per student interaction (reading the problem, thinking, typing an answer).

Most profiles converge within 5-6 rounds (under 2 minutes). The ceiling profile requires 11 rounds because BKT is conservatively slow to declare mastery when the student is already near-threshold. The convergence results indicate the system achieves a stable, actionable estimate of student knowledge within the first few minutes of any session.

## Design Decisions and Their Consequences

### Why a Fine-Tuned Transformer Instead of an LLM API

We chose to fine-tune a 66M parameter model rather than call a general-purpose LLM (GPT-4, Claude) for several reasons:

1. Latency: DistilBERT inference is ~200ms on MPS, ~50ms on CUDA. LLM API calls typically take 500ms-2s, which degrades the interactive tutoring experience.
2. Cost: After training, inference is free (runs on local hardware). API-based classification at $0.01-0.03 per prediction would cost $0.20-0.60 per 20-round session per student.
3. Privacy: Student response data never leaves the school's infrastructure. This is critical for FERPA compliance and district adoption.
4. Consistency: A fine-tuned model produces deterministic outputs (with temperature=0 or greedy decoding). LLMs can vary across API versions and have occasional hallucinations.
5. Offline capability: The system can run entirely without internet access, enabling deployment in schools with unreliable connectivity.

The tradeoff: fine-tuning requires labeled training data, which limits how quickly we can add new misconception categories. An LLM with the misconception taxonomy in its system prompt could potentially handle new categories zero-shot. We recommend evaluating an LLM-based classifier as a comparison point during the pilot phase.

### Why Synthetic Data Dominates

The training set is 86% synthetic (595/687 examples). This was a necessity, not a preference: only 27 MaE examples have both a concept mapping and a specific misconception mapping to our 19 categories. However, the synthetic approach has two positive side effects:

1. Controlled misconception distribution: we can generate exactly balanced class counts, preventing the classifier from developing frequency bias.
2. Linguistic diversity: the six phrasing styles introduce variation that real classroom data (collected from a single school) might not provide.

The risk is domain shift: synthetic phrasings follow templates, while real students may use slang, code-switching, emoji, voice-to-text artifacts, or multi-step reasoning that our templates do not cover. We mitigate this during the pilot phase by collecting real student data and using it to progressively replace synthetic examples.

### Why Linear Prerequisite Chain

Algebra has a genuinely complex dependency structure. Fractions feed into equation solving. Proportional reasoning feeds into graphing. We constrained to a linear chain because:

1. It is the simplest structure that enables meaningful prerequisite gating.
2. Five concepts with a linear chain have $5! = 120$ possible mastery orderings, which is tractable for exhaustive testing.
3. It matches the most common textbook chapter ordering for this topic sequence.

Expanding to a DAG (e.g., adding fractions as a parallel branch) is architecturally trivial (the `KnowledgeGraph` class already supports any DAG; the linear chain is just the data, not a code constraint) but would require additional evaluation work to verify the adaptive engine makes sensible decisions at branching points.

### Why Label Smoothing

We apply label smoothing at 0.1, meaning the hard target $[0, 0, ..., 1, ..., 0]$ becomes $[0.005, 0.005, ..., 0.905, ..., 0.005]$. For a 20-class problem, this has two effects:

1. Prevents the model from becoming infinitely confident on training examples (which causes sharp logits and poor calibration).
2. Acts as a soft regularizer, encouraging the model to maintain some probability mass on related classes, which benefits the confusion pairs where multiple labels are plausible.

Without label smoothing, we observed sharper logits and marginally higher training accuracy but lower validation F1. This matches the expected behavior: label smoothing trades a small amount of peak accuracy for better generalization and calibration.

## Scalability and Productionization

### Scaling the Concept Graph

| Expansion Target | Concepts | Misconceptions (est.) | Training Examples Needed | Estimated Effort |
|-----------------|----------|----------------------|-------------------------|-----------------|
| Current prototype | 5 | 19 | 414 (achieved) | Complete |
| Full pre-algebra | 12-15 | 45-60 | 1,200-1,800 | 3-4 weeks |
| Full algebra I | 20-25 | 80-100 | 2,400-3,000 | 6-8 weeks |
| Algebra I + II | 35-45 | 140-180 | 4,200-5,400 | 3-4 months |

The per-concept expansion effort:

1. Domain expert identifies 3-5 misconceptions per concept (1-2 hours)
2. Write generator templates for each misconception (2-3 hours each)
3. Generate 30-40 synthetic examples per misconception
4. Collect 5-10 real examples per misconception from MaE or classroom data (optional but recommended)
5. Retrain the classifier from the existing checkpoint with the expanded label set
6. Add 5-6 problems to the problem bank per concept
7. Write targeted hints per misconception (30 minutes each)
8. Update knowledge graph JSON with new nodes and edges

Retraining from the existing checkpoint (rather than from scratch) preserves learned representations for existing misconceptions. The classifier head's weight matrix expands to accommodate new labels, and the new class-specific weights are initialized randomly while existing class weights transfer from the checkpoint.

### Serving Architecture for Production

#### Single Classroom (30 students)

```
Student Browser ──→ Web Server (FastAPI) ──→ DistilBERT (single GPU)
                                          ├── BKT State (SQLite)
                                          └── Knowledge Graph (JSON in memory)
```

Components:

- FastAPI web server handling HTTP requests
- DistilBERT loaded into GPU memory (256MB)
- SQLite database for student state persistence
- JSON knowledge graph loaded at startup

This handles 30 concurrent students with ~200ms response time per request. Total hardware cost: any machine with a modern GPU or Apple Silicon.

#### School-Wide (500 students)

```
Students ──→ Load Balancer ──→ [API Server 1] ──────→ ONNX Runtime
                            ├── [API Server 2] ──────→ (shared model)
                            └── [API Server 3] ──────→
                                                       └── PostgreSQL (student state)
```

Changes from single-classroom:

- ONNX-exported model for faster CPU inference (eliminating GPU requirement)
- Multiple API server instances behind a load balancer
- PostgreSQL for shared student state across servers
- Estimated: 3 CPU-only servers can handle 500 concurrent students

ONNX export eliminates the PyTorch dependency at inference time, reducing container size from ~2GB to ~300MB and enabling deployment on baseline server hardware without GPU.

#### District-Scale (5,000+ students)

```
Students ──→ CDN/Edge ──→ Regional API Cluster ──→ Model Serving (Triton)
                                                ├── Redis (session cache)
                                                ├── PostgreSQL (persistent state)
                                                └── Analytics Pipeline (Kafka → Spark)
```

Additional components:

- Triton Inference Server for batched GPU inference (100-500 predictions/second per GPU)
- Redis for session state caching (sub-millisecond reads)
- Kafka + Spark analytics pipeline for aggregating learning analytics across schools
- Federated learning pipeline for model improvement from distributed student data (see below)

### Latency Budget

| Component | Single GPU (MPS) | Single GPU (CUDA) | ONNX (CPU) | Triton (batched GPU) |
|-----------|-------------------|--------------------|------------|---------------------|
| Tokenization | ~5ms | ~5ms | ~5ms | ~2ms (batched) |
| Model inference | ~200ms | ~50ms | ~80ms | ~10ms/request |
| BKT update | <1ms | <1ms | <1ms | <1ms |
| Adaptive engine | <1ms | <1ms | <1ms | <1ms |
| Network overhead | 0 (local) | 0 (local) | ~20ms | ~50ms |
| Total | ~210ms | ~60ms | ~110ms | ~65ms |

All configurations deliver sub-250ms response times, well within the interactive threshold for educational software (students typically take 5-30 seconds to read and answer a problem).

### Cost Analysis

| Deployment Scale | Infrastructure | Monthly Cost (est.) |
|-----------------|----------------|---------------------|
| Single classroom | Teacher's laptop with MPS/CUDA | $0 (existing hardware) |
| Single classroom | Cloud VM with T4 GPU | ~$200/month |
| School-wide | 3 CPU VMs (ONNX) | ~$300/month |
| School-wide | 1 GPU VM (Triton) | ~$400/month |
| District (5K students) | Auto-scaling cluster | ~$1,500-3,000/month |

Compare to commercial ITS platforms: $5-15 per student per month. For a school of 500 students, that is $2,500-7,500/month vs our estimated $300-400/month. The cost advantage compounds at district scale and is particularly relevant for under-resourced schools.

## Expanding Impact: Paths to Reaching More Learners

### Teacher-Facing Analytics Dashboard

The system currently exposes student mastery data only through a CLI summary. A teacher-facing web interface would multiply impact by enabling:

1. Class-level misconception heatmaps: which misconceptions are most prevalent across the class? This guides whole-class instruction decisions.
2. Individual student timelines: mastery progression over time, with specific misconception triggers highlighted.
3. Session assignment: teachers create assignments scoped to specific concepts or misconception areas.
4. Early warning system: students whose mastery drops below threshold after previously achieving it could be flagged for intervention.
5. Data export: CSV/JSON exports for integration with school grading systems.

The backend data is already structured for this (per-student mastery arrays, per-interaction misconception logs). The implementation requires a web frontend consuming the same API that the CLI uses.

### LLM-Enhanced Classification

A hybrid approach combining the fine-tuned classifier with an LLM could address the error analysis findings:

1. Use DistilBERT as the primary classifier (fast, cheap, FERPA-compliant).
2. When DistilBERT confidence is below 0.3, route the input to an LLM with the misconception taxonomy in the prompt for disambiguation.
3. The LLM's classification can also be logged and used to generate additional training data for the fine-tuned model.

This hybrid approach would particularly help with the two confusion pairs identified in error analysis, where the fine-tuned model's limited context prevents disambiguation but an LLM's broader reasoning capabilities could infer the student's reasoning process.

Expected impact: resolving even half of the 8 current test errors would push accuracy from 91.1% to 95.6%.

### Active Learning for Data Efficiency

Instead of generating more synthetic data, we can use active learning to maximize the value of each real student interaction:

1. During deployment, when the classifier confidence falls below a threshold (e.g., 0.3), flag the example for human review.
2. A teacher or annotator labels the flagged example with the correct misconception.
3. The labeled example is added to the training set and the classifier is periodically retrained.

This approach preferentially collects examples from the decision boundary where the classifier is weakest, producing the most informative training signal per labeled example. Research shows active learning can match random-sampling performance with 50-70% fewer labeled examples (Settles, 2009).

### Federated Learning for Privacy-Preserving Model Improvement

Each school deployment generates student interaction data that could improve the classifier. Federated learning enables this without centralizing student data:

1. Each school trains a local model update on its student interaction data.
2. Local model weight updates (not student data) are sent to a central aggregation server.
3. The central server averages the weight updates (FedAvg, McMahan et al., 2017) and distributes the improved model back to all schools.

This approach:

- Preserves FERPA compliance (student data never leaves the school network)
- Reduces synthetic data dependency over time
- Captures regional and demographic linguistic variations
- Enables continuous model improvement without manual data collection

Implementation requires: a model update API endpoint at each school, a secure aggregation server, and differential privacy guarantees on the weight updates to prevent model inversion attacks.

### Multilingual Expansion

The architecture separates language-dependent components (classifier, problem bank, hints, phrasing styles) from language-independent components (knowledge graph structure, BKT, adaptive engine).

Expansion path:

1. Replace `distilbert-base-uncased` with `distilbert-base-multilingual-cased` or `xlm-roberta-base`.
2. Translate the problem bank and hints for each target language.
3. Create language-specific synthetic generator templates (phrasing styles vary by language).
4. Generate training data in the target language using the translated templates.
5. Fine-tune a multilingual or language-specific classifier.

The BKT parameters, knowledge graph edges, misconception taxonomy, and adaptive engine logic do not change across languages. A student who believes "negative times negative is negative" holds the same misconception regardless of whether they express it in English, Spanish, or Mandarin.

Priority languages for global impact: Spanish (400M+ native speakers, large Latin American education market), Mandarin Chinese (900M+ speakers), Hindi (600M+ speakers), Arabic (300M+ speakers).

### Hierarchical Classification

The error analysis reveals that confusion occurs within concepts, not across them. A two-level hierarchical classifier could exploit this:

| Level | Task | Classes | Expected Accuracy |
|-------|------|---------|-------------------|
| Level 1 | Concept identification | 5 concepts + correct | >98% (all cross-concept classifications are perfect today) |
| Level 2 | Misconception disambiguation | 3-4 per concept | Higher accuracy due to reduced class space |

Implementation: two models (or a single model with two heads) where Level 1 routes to the appropriate Level 2 classifier. The total parameter count doubles (~130M), but the per-request latency only increases by one additional forward pass (~100ms on MPS), keeping the system well within interactive thresholds.

### Accessibility and Equity Considerations

For the system to reach underserved populations:

1. Offline mode: the ONNX-exported model with the web interface should run entirely on a single low-cost device (Chromebook, tablet) without internet access.
2. Voice input: integrating speech-to-text would enable use by students with limited typing ability or visual impairments. The classifier's phrasing styles already accommodate informal language; speech-to-text output falls within the "uncertain" or "explain" registers.
3. Reduced data requirements: the system should work with limited bandwidth and storage. The full deployment package (model + knowledge graph + problem bank) is under 500MB.
4. Cultural adaptation: problem contexts (e.g., word problems about currency, measurement) should reflect the student's cultural context. The knowledge graph and misconception structures are universal, but the problem bank should be localized.

### Integration with Learning Standards

Each misconception category can be mapped to Common Core State Standards (CCSS) for mathematics:

| Concept | CCSS Standards |
|---------|----------------|
| Integer sign operations | 7.NS.A.1, 7.NS.A.2 |
| Order of operations | 6.EE.A.1, 6.EE.A.2 |
| Distributive property | 6.EE.A.3, 7.EE.A.1 |
| Combining like terms | 7.EE.A.1, 7.EE.A.2 |
| Solving linear equations | 8.EE.C.7 |

This mapping enables the teacher dashboard to report mastery in terms that align with existing assessment frameworks, lowering the adoption barrier for schools that already report against CCSS.

## Reproducing This Work

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch transformers datasets scikit-learn \
            huggingface_hub sentencepiece protobuf pytest
```

Tested on: Python 3.14.3, transformers 5.3.0, torch 2.10.0, scikit-learn 1.8.0. Apple M-series (MPS) and CPU backends confirmed working.

### Step-by-Step Reproduction

```bash
# 1. Build the dataset (downloads MaE, generates synthetic, splits)
python src/build_dataset.py

# 2. Train the baseline (produces results/baseline_tfidf.json)
python src/baseline_tfidf.py

# 3. Train the classifier (produces models/classifier/best/)
python src/train_classifier.py \
    --model_name distilbert-base-uncased \
    --epochs 15 \
    --lr 2e-5 \
    --batch_size 16 \
    --output_dir models/classifier

# 4. Run the evaluation suite (produces results/phase5_evaluation.json)
python src/evaluate.py

# 5. Run unit tests (31 tests covering BKT, knowledge graph, adaptive engine)
pytest tests/test_knowledge_graph.py -v

# 6. Run the interactive CLI
python src/tutor_cli.py
```

Expected training time: 5-10 minutes on Apple MPS, 2-3 minutes on RTX 3070. Early stopping typically triggers at epoch 5-7.

### Validating Your Reproduction

Compare your results against these targets:

| Metric | Expected Value | Acceptable Range |
|--------|---------------|------------------|
| Val accuracy | 90.1% | 87-93% |
| Val F1 (macro) | 88.2% | 85-91% |
| Test accuracy | 91.1% | 88-94% |
| Test F1 (macro) | 88.6% | 85-91% |
| Baseline test accuracy | 35.6% | 30-40% |
| Unit tests passing | 31/31 | 31/31 |

Variation within the acceptable range arises from hardware-specific floating-point differences and non-determinism in MPS operations. CUDA with `torch.backends.cudnn.deterministic = True` should produce exact matches.

### Key Files for Reproducibility

| Artifact | File | Critical For |
|----------|------|-------------|
| Domain knowledge | `data/knowledge_graph.json` | BKT params, misconception taxonomy, label list |
| Training data | `data/dataset/train.json` | Exact training examples |
| Data generator | `src/build_dataset.py` | Regenerating dataset from scratch |
| Training script | `src/train_classifier.py` | Hyperparameters, training loop |
| Evaluation suite | `src/evaluate.py` | All metrics reported in this document |
| BKT + adaptive engine | `src/knowledge_graph.py` | Core algorithms |
| Integration layer | `src/tutor_session.py` | Answer matching, hints, session logic |
| Unit tests | `tests/test_knowledge_graph.py` | Correctness verification |

## Known Limitations

### Data Quality

1. The 414-example training set is small for transformer fine-tuning. More data would improve both accuracy and confidence calibration. Target: 1,000-2,000 examples for robust production deployment.
2. Synthetic data constitutes 86% of the corpus. Template-based generation cannot capture the full distribution of real student language (slang, code-switching, emoji, voice-to-text artifacts, multi-language responses).
3. The two confusion pairs in the error analysis account for all eight test errors. These are genuinely ambiguous classification tasks where the observable output (the numeric answer) is identical for two different reasoning errors. Resolution requires either category merging or additional input signal (e.g., student work shown step by step).

### BKT Calibration

Uniform BKT parameters across all concepts are an acknowledged simplification. Per-concept parameters fitted from empirical data via expectation-maximization would improve:

- Absolute mastery estimate accuracy (important for teacher reporting)
- Convergence speed for concepts with unusual learning curves
- Detection of the transition from "struggling" to "learned" (currently uniform at $P(T) = 0.15$; some concepts have faster learning transitions in practice)

### Evaluation Gaps

This offline evaluation measures classification accuracy and adaptive engine behavior in simulation. It does not measure:

1. Learning outcomes: Does interaction with the system improve student performance on independent assessments?
2. Engagement: Do students find the interaction useful and appropriately paced?
3. Teacher utility: Do teachers find the misconception reports accurate and actionable?
4. Long-term retention: Do mastery estimates predict performance days or weeks later?
5. Fairness: Does the system perform equitably across student demographics (gender, race, socioeconomic status, English proficiency)?

Each of these requires a classroom pilot study with appropriate experimental design.

### Answer Matching Fragility

The string-based answer matching cannot verify algebraic equivalence. "2x + 4" and "4 + 2x" would be marked as different answers. For the current problem bank (numeric or simple algebraic answers), this is acceptable. Expanding to more complex expressions (multi-term polynomials, rational expressions) would require integrating SymPy or a similar CAS.

## Conclusion

This system demonstrates that a modestly sized transformer (66M parameters) fine-tuned on a compact dataset (414 examples) can classify algebra misconceptions with 91.1% accuracy, and that pairing this classifier with BKT produces adaptive tutoring behavior that reliably identifies and targets student weaknesses.

The architecture separates concerns cleanly: the knowledge graph encodes domain structure, BKT tracks learner state, the classifier reads natural language, and the adaptive engine makes instructional decisions. This separation means each component can be improved, replaced, or scaled independently. A school with CUDA hardware can swap in DeBERTa for higher accuracy. A district can replace SQLite with PostgreSQL. A researcher can plug in an LLM-based classifier without touching the BKT or adaptive engine code.

The primary barriers to deployment are operational, not algorithmic: collecting real student data, building teacher-facing and student-facing interfaces, and conducting the classroom evaluation needed to establish learning efficacy. The path from prototype to production is well-defined, and the estimated cost ($300-400/month for school-wide deployment) makes this accessible to schools that cannot afford commercial ITS platforms.

The greatest potential for impact lies in three directions: expanding the concept graph to cover the full algebra curriculum, enabling federated learning across school deployments to continuously improve the classifier from real student data without compromising privacy, and deploying multilingually to reach the hundreds of millions of students worldwide who study algebra outside of English-speaking contexts.

## References

1. Baker, R. S., Corbett, A. T., & Aleven, V. (2008). More accurate student modeling through contextual estimation of slip and guess probabilities in Bayesian Knowledge Tracing. *Proceedings of the 9th International Conference on Intelligent Tutoring Systems*, 406-415.
2. Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253-278.
3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*, 4171-4186.
4. He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. *International Conference on Learning Representations*.
5. McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *Proceedings of AISTATS*, 1273-1282.
6. Otero, N., Druga, S., & Lan, A. (2025). A benchmark dataset for math misconceptions across education levels. *Discover Education*, 4, Article 42.
7. Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L. J., & Sohl-Dickstein, J. (2015). Deep knowledge tracing. *Advances in Neural Information Processing Systems*, 28.
8. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.
9. Settles, B. (2009). Active learning literature survey. *Computer Sciences Technical Report 1648*, University of Wisconsin-Madison.
10. VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems. *Educational Psychologist*, 46(4), 197-221.
11. Woolf, B. P. (2009). *Building intelligent interactive tutors: Student-centered strategies for revolutionizing e-learning*. Morgan Kaufmann.
