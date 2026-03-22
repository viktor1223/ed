"""Phase 5: Offline Evaluation Suite.

Produces:
  1. Test-set classification metrics (per-class F1, confusion matrix)
  2. Ablation: with vs without topic metadata
  3. Simulated student profiles (5 profiles x 20 rounds)
  4. Adaptive vs random vs fixed-sequence comparison

Usage:
    python src/evaluate.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_graph import KnowledgeGraph, StudentState, next_action

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
KG_PATH = PROJECT_ROOT / "data" / "knowledge_graph.json"
RESULTS_DIR = PROJECT_ROOT / "results"

random.seed(SEED)
np.random.seed(SEED)


# ─── 1. Classification Evaluation ────────────────────────────────────────────

def load_test_data():
    with open(DATA_DIR / "test.json") as f:
        data = json.load(f)
    return [d for d in data if d["misconception_id"] is not None]


def evaluate_classifier():
    """Load trained model and evaluate on held-out test set."""
    print("=" * 70)
    print("  SECTION 1: Classification Evaluation on Held-Out Test Set")
    print("=" * 70)

    # Lazy import to avoid slow transformers load blocking everything
    from classifier import MisconceptionClassifier

    model_dir = PROJECT_ROOT / "models" / "classifier" / "best"
    clf = MisconceptionClassifier(model_dir)

    test_data = load_test_data()
    print(f"Test examples: {len(test_data)}")

    true_labels = []
    pred_labels = []
    confidences = []

    for item in test_data:
        pred = clf.predict(item["question"], item["student_response"])
        true_labels.append(item["misconception_id"])
        pred_labels.append(pred["label"])
        confidences.append(pred["confidence"])

    # Overall metrics
    acc = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    print(f"\nOverall Accuracy: {acc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"Mean Confidence: {np.mean(confidences):.4f}")

    # Per-class report
    all_labels = sorted(set(true_labels) | set(pred_labels))
    report = classification_report(true_labels, pred_labels, labels=all_labels, zero_division=0)
    print(f"\nPer-Class Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)

    # Error analysis
    errors = []
    for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
        if t != p:
            errors.append({
                "true": t,
                "predicted": p,
                "question": test_data[i]["question"],
                "response": test_data[i]["student_response"],
                "confidence": confidences[i],
            })

    print(f"\nError Analysis: {len(errors)} misclassified out of {len(test_data)}")
    # Group errors by true label
    error_by_true = defaultdict(list)
    for e in errors:
        error_by_true[e["true"]].append(e)
    for label, errs in sorted(error_by_true.items()):
        print(f"  {label}: {len(errs)} errors")
        for e in errs[:2]:
            print(f"    Q: {e['question'][:60]} -> predicted {e['predicted']} (conf {e['confidence']:.2f})")

    # Per-concept accuracy
    print("\nPer-Concept Accuracy:")
    kg = KnowledgeGraph.from_json(KG_PATH)
    concept_correct = defaultdict(int)
    concept_total = defaultdict(int)
    for item, t, p in zip(test_data, true_labels, pred_labels):
        cid = item["concept_id"]
        concept_total[cid] += 1
        if t == p:
            concept_correct[cid] += 1
    for cid in sorted(concept_total):
        acc_c = concept_correct[cid] / concept_total[cid]
        print(f"  {cid}: {acc_c:.2%} ({concept_correct[cid]}/{concept_total[cid]})")

    return {
        "test_size": len(test_data),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "mean_confidence": float(np.mean(confidences)),
        "num_errors": len(errors),
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": all_labels,
        "errors": errors[:20],
    }


# ─── 2. Ablation: Topic Metadata ─────────────────────────────────────────────

def evaluate_ablation():
    """Compare classification with and without topic/concept metadata in the prompt."""
    print("\n" + "=" * 70)
    print("  SECTION 2: Ablation - Topic Metadata Impact")
    print("=" * 70)

    from classifier import MisconceptionClassifier

    model_dir = PROJECT_ROOT / "models" / "classifier" / "best"
    clf = MisconceptionClassifier(model_dir)
    kg = KnowledgeGraph.from_json(KG_PATH)

    test_data = load_test_data()

    # Standard: "Question: ... Student answer: ..."
    standard_preds = []
    # With topic: "Topic: ... Question: ... Student answer: ..."
    topic_preds = []
    true_labels = []

    for item in test_data:
        true_labels.append(item["misconception_id"])

        # Standard prediction (what the model was trained on)
        pred_std = clf.predict(item["question"], item["student_response"])
        standard_preds.append(pred_std["label"])

        # Topic-augmented: prepend concept name to the question
        concept_name = kg.concepts[item["concept_id"]].name
        topic_question = f"[{concept_name}] {item['question']}"
        pred_topic = clf.predict(topic_question, item["student_response"])
        topic_preds.append(pred_topic["label"])

    std_f1 = f1_score(true_labels, standard_preds, average="macro", zero_division=0)
    topic_f1 = f1_score(true_labels, topic_preds, average="macro", zero_division=0)
    std_acc = accuracy_score(true_labels, standard_preds)
    topic_acc = accuracy_score(true_labels, topic_preds)

    print(f"\nStandard (no topic):    Acc={std_acc:.4f}  F1={std_f1:.4f}")
    print(f"With topic metadata:    Acc={topic_acc:.4f}  F1={topic_f1:.4f}")
    print(f"Delta:                  Acc={topic_acc - std_acc:+.4f}  F1={topic_f1 - std_f1:+.4f}")

    return {
        "standard_accuracy": std_acc,
        "standard_f1": std_f1,
        "topic_accuracy": topic_acc,
        "topic_f1": topic_f1,
        "delta_accuracy": topic_acc - std_acc,
        "delta_f1": topic_f1 - std_f1,
    }


# ─── 3. Simulated Student Profiles ───────────────────────────────────────────

STUDENT_PROFILES = {
    "A_strong_one_weak": {
        "description": "Strong overall, weak on distributive property",
        "correct_rates": {
            "integer_sign_ops": 0.95,
            "order_of_operations": 0.90,
            "distributive_property": 0.20,
            "combining_like_terms": 0.85,
            "solving_linear_equations": 0.80,
        },
    },
    "B_weak_overall": {
        "description": "Weak overall, all concepts below mastery",
        "correct_rates": {
            "integer_sign_ops": 0.40,
            "order_of_operations": 0.30,
            "distributive_property": 0.25,
            "combining_like_terms": 0.20,
            "solving_linear_equations": 0.15,
        },
    },
    "C_mixed": {
        "description": "Strong arithmetic, weak algebra",
        "correct_rates": {
            "integer_sign_ops": 0.90,
            "order_of_operations": 0.85,
            "distributive_property": 0.40,
            "combining_like_terms": 0.35,
            "solving_linear_equations": 0.25,
        },
    },
    "D_ceiling": {
        "description": "Strong everywhere (ceiling test)",
        "correct_rates": {
            "integer_sign_ops": 0.95,
            "order_of_operations": 0.95,
            "distributive_property": 0.90,
            "combining_like_terms": 0.95,
            "solving_linear_equations": 0.90,
        },
    },
    "E_random_noise": {
        "description": "Random performance (noise test)",
        "correct_rates": {
            "integer_sign_ops": 0.50,
            "order_of_operations": 0.50,
            "distributive_property": 0.50,
            "combining_like_terms": 0.50,
            "solving_linear_equations": 0.50,
        },
    },
}


def simulate_session(kg, profile, strategy="adaptive", n_rounds=20):
    """Simulate n_rounds of tutoring with a student profile.

    strategy: "adaptive" (uses next_action), "random", "fixed_sequence"
    """
    state = StudentState(kg)
    concepts_list = [c.id for c in kg.concepts_by_level()]
    history = []
    fixed_idx = 0

    for round_num in range(n_rounds):
        # Select concept
        if strategy == "adaptive":
            action = next_action(state, kg)
            concept_id = action["concept"]
        elif strategy == "random":
            concept_id = random.choice(concepts_list)
        else:  # fixed_sequence
            concept_id = concepts_list[fixed_idx % len(concepts_list)]
            fixed_idx += 1

        # Simulate student response
        correct_rate = profile["correct_rates"][concept_id]
        correct = random.random() < correct_rate

        # Update state
        state.update(concept_id, correct=correct, confidence=0.8 if not correct else 1.0)

        history.append({
            "round": round_num,
            "concept": concept_id,
            "correct": correct,
            "mastery_snapshot": {cid: round(state.mastery[cid], 4) for cid in concepts_list},
        })

    return state, history


def evaluate_simulated_students():
    """Run all 5 student profiles through 20 rounds with each strategy."""
    print("\n" + "=" * 70)
    print("  SECTION 3: Simulated Student Evaluation")
    print("=" * 70)

    kg = KnowledgeGraph.from_json(KG_PATH)
    strategies = ["adaptive", "random", "fixed_sequence"]
    results = {}

    for profile_name, profile in STUDENT_PROFILES.items():
        print(f"\n--- Profile {profile_name}: {profile['description']} ---")
        results[profile_name] = {}

        for strategy in strategies:
            # Run 10 simulations to get stable averages
            all_final_masteries = []
            all_correct_counts = []
            all_weak_targeted = []

            for trial in range(10):
                random.seed(SEED + trial)
                state, history = simulate_session(kg, profile, strategy=strategy)

                final_mastery = {cid: state.mastery[cid] for cid in kg.concepts}
                all_final_masteries.append(final_mastery)
                all_correct_counts.append(sum(1 for h in history if h["correct"]))

                # Measure weak-concept targeting
                weak_concepts = {
                    cid for cid, rate in profile["correct_rates"].items() if rate < 0.5
                }
                if weak_concepts:
                    weak_rounds = sum(1 for h in history if h["concept"] in weak_concepts)
                    all_weak_targeted.append(weak_rounds / len(history))

            # Average metrics
            avg_mastery = {
                cid: np.mean([m[cid] for m in all_final_masteries])
                for cid in kg.concepts
            }
            avg_correct = np.mean(all_correct_counts)
            avg_weak_targeting = np.mean(all_weak_targeted) if all_weak_targeted else 0

            # Concept identification accuracy: did the system correctly rank weak concepts lowest?
            true_weak = sorted(profile["correct_rates"], key=lambda c: profile["correct_rates"][c])[:2]
            system_weak = sorted(avg_mastery, key=lambda c: avg_mastery[c])[:2]
            concept_id_accuracy = len(set(true_weak) & set(system_weak)) / len(true_weak)

            results[profile_name][strategy] = {
                "avg_correct": float(avg_correct),
                "avg_mastery": {k: round(v, 4) for k, v in avg_mastery.items()},
                "concept_id_accuracy": concept_id_accuracy,
                "weak_targeting_rate": float(avg_weak_targeting),
            }

            print(f"  {strategy:16s}: correct={avg_correct:.1f}/20  "
                  f"concept_id_acc={concept_id_accuracy:.0%}  "
                  f"weak_targeting={avg_weak_targeting:.0%}")

    # Summary comparison
    print("\n--- Strategy Comparison (averaged across all profiles) ---")
    for strategy in strategies:
        avg_concept_id = np.mean([
            results[p][strategy]["concept_id_accuracy"] for p in results
        ])
        avg_weak = np.mean([
            results[p][strategy]["weak_targeting_rate"] for p in results
            if results[p][strategy]["weak_targeting_rate"] > 0
        ])
        print(f"  {strategy:16s}: concept_id_acc={avg_concept_id:.0%}  weak_targeting={avg_weak:.0%}")

    return results


# ─── 4. Mastery Convergence Analysis ─────────────────────────────────────────

def evaluate_convergence():
    """Measure how quickly mastery estimates stabilize for each profile."""
    print("\n" + "=" * 70)
    print("  SECTION 4: Mastery Convergence Analysis")
    print("=" * 70)

    kg = KnowledgeGraph.from_json(KG_PATH)
    results = {}

    for profile_name, profile in STUDENT_PROFILES.items():
        random.seed(SEED)
        state, history = simulate_session(kg, profile, strategy="adaptive", n_rounds=20)

        # Find round where mastery estimates stop changing by > 0.05
        convergence_round = 20
        for i in range(2, 20):
            max_delta = max(
                abs(history[i]["mastery_snapshot"][cid] - history[i - 1]["mastery_snapshot"][cid])
                for cid in kg.concepts
            )
            if max_delta < 0.05:
                convergence_round = i
                break

        results[profile_name] = {
            "convergence_round": convergence_round,
            "trajectory": [
                {cid: h["mastery_snapshot"][cid] for cid in kg.concepts}
                for h in history
            ],
        }
        print(f"  {profile_name}: converges at round {convergence_round}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run all evaluations
    clf_results = evaluate_classifier()
    ablation_results = evaluate_ablation()
    sim_results = evaluate_simulated_students()
    conv_results = evaluate_convergence()

    # Compile and save
    full_results = {
        "classification": {
            "test_size": clf_results["test_size"],
            "accuracy": clf_results["accuracy"],
            "f1_macro": clf_results["f1_macro"],
            "f1_weighted": clf_results["f1_weighted"],
            "mean_confidence": clf_results["mean_confidence"],
            "num_errors": clf_results["num_errors"],
            "labels": clf_results["labels"],
            "confusion_matrix": clf_results["confusion_matrix"],
        },
        "ablation_topic_metadata": ablation_results,
        "simulated_students": sim_results,
        "convergence": {
            name: {"convergence_round": r["convergence_round"]}
            for name, r in conv_results.items()
        },
        "baseline_comparison": _load_baseline(),
    }

    with open(RESULTS_DIR / "phase5_evaluation.json", "w") as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "=" * 70)
    print("  RESULTS SAVED to results/phase5_evaluation.json")
    print("=" * 70)

    return full_results


def _load_baseline():
    path = RESULTS_DIR / "baseline_tfidf.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    main()
