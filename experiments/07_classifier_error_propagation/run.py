"""Experiment 07: Classifier Error Propagation Through the Tutoring Pipeline.

The system pipeline is:
  Classifier -> BKT Update -> Concept Selection -> Problem Selection -> Intervention -> Learning

This experiment injects classifier errors at controlled rates and measures
how each error type propagates through the pipeline to degrade end-to-end
learning outcomes.

Error types:
  1. Misconception misidentification: classifier returns the wrong misconception
     (wrong targeted remediation - teaches the wrong thing)
  2. False negative: classifier misses a misconception entirely, returns "correct"
     (student doesn't get remediation they need)
  3. False positive: classifier reports a misconception when the student was correct
     (wastes time on unnecessary remediation, corrupts BKT)
  4. Concept misrouting: classifier assigns misconception to wrong concept
     (BKT updates the wrong concept's mastery)

For each error type, we sweep the error rate from 0% to 50% and measure:
  - Held-out test score gain (primary)
  - BKT mastery estimation error (vs ground-truth student p_know)
  - Misconception resolution rate
  - Wasted interventions (targeted at wrong misconception)
  - Effect size degradation vs error-free baseline

Usage:
    cd <project_root>
    python experiments/07_classifier_error_propagation/run.py

Outputs:
    experiments/07_classifier_error_propagation/artifacts/results.json
    experiments/07_classifier_error_propagation/artifacts/gain_degradation.png
    experiments/07_classifier_error_propagation/artifacts/bkt_error.png
    experiments/07_classifier_error_propagation/artifacts/error_type_comparison.png
    experiments/07_classifier_error_propagation/artifacts/threshold_analysis.png
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from knowledge_graph import KnowledgeGraph, StudentState, next_action
from simulated_student import SimulatedStudent, generate_students, load_problem_bank
from simulated_rct_v2 import (
    split_problem_bank,
    administer_test,
    adaptive_strategy_v2,
    MAX_CONSECUTIVE_SAME_CONCEPT,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
KG_PATH = PROJECT_ROOT / "data" / "knowledge_graph.json"
SEED = 42
N_STUDENTS = 300
N_INTERACTIONS = 40


# ─── Error Injection ──────────────────────────────────────────────────────────

class ErrorInjector:
    """Simulates classifier errors of different types."""

    def __init__(
        self,
        error_rate: float,
        error_type: str,
        kg: KnowledgeGraph,
        rng: random.Random,
    ):
        self.error_rate = error_rate
        self.error_type = error_type
        self.kg = kg
        self.rng = rng
        self.all_misconceptions = kg.all_misconception_ids()
        self.concept_misconceptions: dict[str, list[str]] = {}
        for concept in kg.concepts.values():
            self.concept_misconceptions[concept.id] = [
                m.id for m in concept.misconceptions
            ]

    def inject(
        self,
        true_misconception: str | None,
        concept_id: str,
        correct: bool,
    ) -> dict:
        """Apply error injection to a classifier prediction.

        Args:
            true_misconception: The actual misconception used (None if correct)
            concept_id: The concept being assessed
            correct: Whether the student's answer was correct

        Returns:
            dict with:
                detected_misconception: str | None (what the system thinks)
                was_injected: bool (whether an error was injected this time)
                error_category: str | None (type of error if injected)
        """
        if self.rng.random() >= self.error_rate:
            # No error - pass through truth
            return {
                "detected_misconception": true_misconception,
                "was_injected": False,
                "error_category": None,
            }

        if self.error_type == "misidentification":
            return self._misidentify(true_misconception, concept_id, correct)
        elif self.error_type == "false_negative":
            return self._false_negative(true_misconception, correct)
        elif self.error_type == "false_positive":
            return self._false_positive(true_misconception, concept_id, correct)
        elif self.error_type == "concept_misroute":
            return self._concept_misroute(true_misconception, concept_id, correct)
        else:
            raise ValueError(f"Unknown error type: {self.error_type}")

    def _misidentify(self, true_misc, concept_id, correct):
        """Return a wrong misconception from the same concept."""
        if true_misc is None:
            # Can't misidentify a correct answer
            return {"detected_misconception": None, "was_injected": False, "error_category": None}

        same_concept = [
            m for m in self.concept_misconceptions.get(concept_id, [])
            if m != true_misc
        ]
        if not same_concept:
            return {"detected_misconception": true_misc, "was_injected": False, "error_category": None}

        return {
            "detected_misconception": self.rng.choice(same_concept),
            "was_injected": True,
            "error_category": "misidentification",
        }

    def _false_negative(self, true_misc, correct):
        """Miss a real misconception - return None."""
        if true_misc is None:
            return {"detected_misconception": None, "was_injected": False, "error_category": None}
        return {
            "detected_misconception": None,
            "was_injected": True,
            "error_category": "false_negative",
        }

    def _false_positive(self, true_misc, concept_id, correct):
        """Hallucinate a misconception when the student was correct."""
        if not correct:
            # Student was wrong - no false positive to inject
            return {"detected_misconception": true_misc, "was_injected": False, "error_category": None}
        options = self.concept_misconceptions.get(concept_id, [])
        if not options:
            return {"detected_misconception": None, "was_injected": False, "error_category": None}
        return {
            "detected_misconception": self.rng.choice(options),
            "was_injected": True,
            "error_category": "false_positive",
        }

    def _concept_misroute(self, true_misc, concept_id, correct):
        """Assign the misconception to a wrong concept's misconception."""
        if true_misc is None:
            return {"detected_misconception": None, "was_injected": False, "error_category": None}
        other_concepts = [
            cid for cid in self.concept_misconceptions
            if cid != concept_id and self.concept_misconceptions[cid]
        ]
        if not other_concepts:
            return {"detected_misconception": true_misc, "was_injected": False, "error_category": None}
        wrong_concept = self.rng.choice(other_concepts)
        wrong_misc = self.rng.choice(self.concept_misconceptions[wrong_concept])
        return {
            "detected_misconception": wrong_misc,
            "was_injected": True,
            "error_category": "concept_misroute",
        }


# ─── Simulation with Error Injection ─────────────────────────────────────────

def run_student_with_errors(
    student: SimulatedStudent,
    kg: KnowledgeGraph,
    practice_bank: dict[str, list[dict]],
    test_bank: dict[str, list[dict]],
    injector: ErrorInjector,
    n_interactions: int,
) -> dict:
    """Run one student through the pipeline with classifier error injection."""

    # Pre-test
    pre_test = administer_test(student, test_bank)
    pre_p_know = dict(student.p_know)
    pre_misconceptions = set(student.active_misconceptions())

    # Tutoring with error injection
    tutor_state = StudentState(kg)
    consecutive_same = 0
    last_concept = None

    errors_injected = 0
    total_classifications = 0
    correct_remediations = 0
    wrong_remediations = 0
    missed_remediations = 0
    false_alarm_remediations = 0
    bkt_errors_over_time = []

    for i in range(n_interactions):
        # Concept selection (based on tutor's BKT estimate - may be corrupted)
        concept_id = adaptive_strategy_v2(
            tutor_state, kg, consecutive_same, last_concept,
        )
        if concept_id == last_concept:
            consecutive_same += 1
        else:
            consecutive_same = 0
        last_concept = concept_id

        # Pick a problem
        problems = practice_bank.get(concept_id, [])
        if not problems:
            continue
        problem = random.choice(problems)

        # Student responds (ground truth)
        response = student.respond(problem)
        true_misconception = response.get("misconception_used")

        # ── CLASSIFIER WITH ERROR INJECTION ──
        injection = injector.inject(true_misconception, concept_id, response["correct"])
        detected = injection["detected_misconception"]
        if injection["was_injected"]:
            errors_injected += 1
        total_classifications += 1

        # BKT update based on (potentially corrupted) classifier output
        if detected and not response["correct"]:
            # System thinks there's a specific misconception
            tutor_state.update(concept_id, correct=False, confidence=0.8)
        elif detected and response["correct"]:
            # False positive: system thinks wrong but student was right
            tutor_state.update(concept_id, correct=False, confidence=0.6)
        else:
            tutor_state.update(concept_id, correct=response["correct"], confidence=0.8)

        # Instruction based on detected (not true) misconception
        student.receive_instruction(concept_id, targeted_misconception=detected)

        # Track remediation accuracy
        if true_misconception and detected == true_misconception:
            correct_remediations += 1
        elif true_misconception and detected and detected != true_misconception:
            wrong_remediations += 1
        elif true_misconception and detected is None:
            missed_remediations += 1
        elif not true_misconception and detected:
            false_alarm_remediations += 1

        # Track BKT estimation error
        bkt_error = np.mean([
            abs(tutor_state.mastery.get(c, 0.5) - student.p_know.get(c, 0.1))
            for c in kg.concepts
        ])
        bkt_errors_over_time.append(float(bkt_error))

    # Post-test
    post_test = administer_test(student, test_bank)
    post_misconceptions = set(student.active_misconceptions())

    return {
        "test_gain": post_test["aggregate"]["proportion"] - pre_test["aggregate"]["proportion"],
        "pre_score": pre_test["aggregate"]["proportion"],
        "post_score": post_test["aggregate"]["proportion"],
        "misconceptions_resolved": len(pre_misconceptions - post_misconceptions),
        "misconceptions_initial": len(pre_misconceptions),
        "errors_injected": errors_injected,
        "total_classifications": total_classifications,
        "correct_remediations": correct_remediations,
        "wrong_remediations": wrong_remediations,
        "missed_remediations": missed_remediations,
        "false_alarm_remediations": false_alarm_remediations,
        "mean_bkt_error": float(np.mean(bkt_errors_over_time)),
        "final_bkt_error": bkt_errors_over_time[-1] if bkt_errors_over_time else 0.0,
        "bkt_error_trajectory": bkt_errors_over_time,
    }


# ─── Sweep Runner ────────────────────────────────────────────────────────────

def run_error_sweep(
    error_type: str,
    error_rates: list[float],
    n_students: int,
    n_interactions: int,
    seed: int,
) -> list[dict]:
    """Sweep error rates for a single error type."""
    kg = KnowledgeGraph.from_json(KG_PATH)
    full_bank = load_problem_bank()
    practice_bank, test_bank = split_problem_bank(full_bank, n_test_per_concept=2, seed=seed)

    results = []
    for rate in error_rates:
        rng = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)

        students = generate_students(n=n_students, seed=seed)
        for s in students:
            s.targeted_resolution = 0.50

        injector = ErrorInjector(rate, error_type, kg, rng)

        student_results = []
        for s in students:
            sr = run_student_with_errors(
                s, kg, practice_bank, test_bank, injector, n_interactions,
            )
            student_results.append(sr)

        gains = [s["test_gain"] for s in student_results]
        bkt_errors = [s["mean_bkt_error"] for s in student_results]
        resolution_rates = [
            s["misconceptions_resolved"] / max(s["misconceptions_initial"], 1)
            for s in student_results
        ]
        wasted = [
            (s["wrong_remediations"] + s["false_alarm_remediations"]) / max(s["total_classifications"], 1)
            for s in student_results
        ]

        results.append({
            "error_type": error_type,
            "error_rate": rate,
            "mean_gain": round(float(np.mean(gains)), 4),
            "std_gain": round(float(np.std(gains)), 4),
            "mean_bkt_error": round(float(np.mean(bkt_errors)), 4),
            "mean_resolution_rate": round(float(np.mean(resolution_rates)), 4),
            "mean_wasted_fraction": round(float(np.mean(wasted)), 4),
            "n_students": n_students,
        })

    return results


# ─── Threshold Analysis ──────────────────────────────────────────────────────

def compute_minimum_accuracy_thresholds(all_sweeps: dict[str, list[dict]]) -> dict:
    """For each error type, find the error rate at which learning gain
    drops below 50% and 80% of the error-free baseline."""
    thresholds = {}
    for error_type, sweep in all_sweeps.items():
        baseline_gain = sweep[0]["mean_gain"]  # error_rate=0
        if baseline_gain <= 0:
            thresholds[error_type] = {"threshold_50pct": None, "threshold_80pct": None}
            continue

        t50, t80 = None, None
        for point in sweep:
            ratio = point["mean_gain"] / baseline_gain if baseline_gain > 0 else 0
            if t80 is None and ratio < 0.80:
                t80 = point["error_rate"]
            if t50 is None and ratio < 0.50:
                t50 = point["error_rate"]

        thresholds[error_type] = {
            "baseline_gain": baseline_gain,
            "threshold_80pct": t80,
            "threshold_50pct": t50,
            "implied_min_accuracy_80pct": round(1.0 - t80, 2) if t80 else "> 0.50",
            "implied_min_accuracy_50pct": round(1.0 - t50, 2) if t50 else "> 0.50",
        }
    return thresholds


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_gain_degradation(all_sweeps: dict[str, list[dict]]) -> None:
    """Learning gain vs error rate for each error type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "misidentification": "#F44336",
        "false_negative": "#FF9800",
        "false_positive": "#9C27B0",
        "concept_misroute": "#2196F3",
    }
    labels = {
        "misidentification": "Misconception Misidentification",
        "false_negative": "False Negative (missed misconception)",
        "false_positive": "False Positive (hallucinated misconception)",
        "concept_misroute": "Concept Misrouting",
    }

    for error_type, sweep in all_sweeps.items():
        rates = [p["error_rate"] for p in sweep]
        gains = [p["mean_gain"] for p in sweep]
        stds = [p["std_gain"] for p in sweep]
        ax.plot(rates, gains, "o-", label=labels[error_type],
                color=colors[error_type], linewidth=2, markersize=5)
        ax.fill_between(
            rates,
            [g - s for g, s in zip(gains, stds)],
            [g + s for g, s in zip(gains, stds)],
            alpha=0.1, color=colors[error_type],
        )

    ax.set_xlabel("Classifier Error Rate")
    ax.set_ylabel("Mean Test Score Gain")
    ax.set_title("Learning Gain Degradation by Error Type")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.51)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "gain_degradation.png", dpi=150)
    plt.close()


def plot_bkt_error(all_sweeps: dict[str, list[dict]]) -> None:
    """BKT estimation error vs classifier error rate."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "misidentification": "#F44336",
        "false_negative": "#FF9800",
        "false_positive": "#9C27B0",
        "concept_misroute": "#2196F3",
    }
    labels = {
        "misidentification": "Misidentification",
        "false_negative": "False Negative",
        "false_positive": "False Positive",
        "concept_misroute": "Concept Misroute",
    }

    for error_type, sweep in all_sweeps.items():
        rates = [p["error_rate"] for p in sweep]
        bkt_errs = [p["mean_bkt_error"] for p in sweep]
        ax.plot(rates, bkt_errs, "o-", label=labels[error_type],
                color=colors[error_type], linewidth=2, markersize=5)

    ax.set_xlabel("Classifier Error Rate")
    ax.set_ylabel("Mean BKT Estimation Error (|estimated - true| mastery)")
    ax.set_title("BKT Mastery Estimation Corruption by Error Type")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "bkt_error.png", dpi=150)
    plt.close()


def plot_error_type_comparison(all_sweeps: dict[str, list[dict]]) -> None:
    """Side-by-side comparison of all degradation metrics at 20% error."""
    error_types = list(all_sweeps.keys())
    target_rate = 0.20

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ("mean_gain", "Test Score Gain", "Learning Outcome"),
        ("mean_resolution_rate", "Misconception Resolution Rate", "Resolution Effectiveness"),
        ("mean_wasted_fraction", "Wasted Intervention Fraction", "Resource Waste"),
    ]

    colors = ["#F44336", "#FF9800", "#9C27B0", "#2196F3"]

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        vals = []
        for et in error_types:
            sweep = all_sweeps[et]
            point_20 = next((p for p in sweep if abs(p["error_rate"] - target_rate) < 0.01), sweep[-1])
            vals.append(point_20[metric])

        bars = ax.bar(range(len(error_types)), vals, color=colors)
        ax.set_xticks(range(len(error_types)))
        ax.set_xticklabels([
            "Mis-ID", "False Neg", "False Pos", "Misroute"
        ], rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} at 20% Error Rate")

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "error_type_comparison.png", dpi=150)
    plt.close()


def plot_threshold_analysis(all_sweeps: dict[str, list[dict]], thresholds: dict) -> None:
    """Show where each error type crosses 80% and 50% of baseline gain."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "misidentification": "#F44336",
        "false_negative": "#FF9800",
        "false_positive": "#9C27B0",
        "concept_misroute": "#2196F3",
    }

    for error_type, sweep in all_sweeps.items():
        baseline = sweep[0]["mean_gain"]
        if baseline <= 0:
            continue

        rates = [p["error_rate"] for p in sweep]
        ratios = [p["mean_gain"] / baseline for p in sweep]
        ax.plot(rates, ratios, "o-", label=error_type.replace("_", " ").title(),
                color=colors[error_type], linewidth=2, markersize=5)

    ax.axhline(y=0.80, color="orange", linestyle="--", alpha=0.7, label="80% of baseline")
    ax.axhline(y=0.50, color="red", linestyle="--", alpha=0.7, label="50% of baseline")
    ax.set_xlabel("Classifier Error Rate")
    ax.set_ylabel("Gain as Fraction of Error-Free Baseline")
    ax.set_title("Minimum Classifier Accuracy Thresholds")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.51)
    ax.set_ylim(-0.1, 1.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "threshold_analysis.png", dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 07: Classifier Error Propagation Through Tutoring Pipeline")
    print("=" * 70)

    error_types = ["misidentification", "false_negative", "false_positive", "concept_misroute"]
    error_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    all_sweeps: dict[str, list[dict]] = {}

    for et in error_types:
        print(f"\n--- Error type: {et} ---")
        sweep = run_error_sweep(et, error_rates, N_STUDENTS, N_INTERACTIONS, SEED)
        all_sweeps[et] = sweep

        for p in sweep:
            print(f"  rate={p['error_rate']:.2f}  gain={p['mean_gain']:.4f}  "
                  f"bkt_err={p['mean_bkt_error']:.4f}  "
                  f"resolution={p['mean_resolution_rate']:.4f}  "
                  f"wasted={p['mean_wasted_fraction']:.4f}")

    # Threshold analysis
    print("\n--- Minimum Accuracy Thresholds ---")
    thresholds = compute_minimum_accuracy_thresholds(all_sweeps)
    for et, t in thresholds.items():
        print(f"  {et}:")
        print(f"    80% baseline retained at error rate: {t['threshold_80pct']}")
        print(f"    50% baseline retained at error rate: {t['threshold_50pct']}")
        print(f"    Implied min accuracy (80%): {t['implied_min_accuracy_80pct']}")

    # Save
    full_output = {
        "metadata": {
            "experiment": "07_classifier_error_propagation",
            "n_students": N_STUDENTS,
            "n_interactions": N_INTERACTIONS,
            "error_rates_swept": error_rates,
            "seed": SEED,
        },
        "sweeps": all_sweeps,
        "thresholds": thresholds,
    }

    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nResults saved to {ARTIFACTS / 'results.json'}")

    # Plots
    print("\nGenerating plots...")
    plot_gain_degradation(all_sweeps)
    plot_bkt_error(all_sweeps)
    plot_error_type_comparison(all_sweeps)
    plot_threshold_analysis(all_sweeps, thresholds)
    print("Plots saved.")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 07 SUMMARY")
    print("=" * 70)
    baseline = all_sweeps["misidentification"][0]["mean_gain"]
    print(f"\nBaseline (0% error) gain: {baseline:.4f}")
    for et in error_types:
        sweep = all_sweeps[et]
        at_20 = next((p for p in sweep if abs(p["error_rate"] - 0.20) < 0.01), None)
        if at_20:
            pct = (at_20["mean_gain"] / baseline * 100) if baseline > 0 else 0
            print(f"\n  {et} at 20% error:")
            print(f"    Gain: {at_20['mean_gain']:.4f} ({pct:.0f}% of baseline)")
            print(f"    BKT error: {at_20['mean_bkt_error']:.4f}")
            print(f"    Wasted interventions: {at_20['mean_wasted_fraction']:.1%}")


if __name__ == "__main__":
    main()
