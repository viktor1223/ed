"""Experiment 09: End-to-End Pipeline Stress Test.

Questions:
  1. When multiple subsystems degrade simultaneously, do errors compound
     multiplicatively (catastrophic) or linearly (graceful)?
  2. Which combination of subsystem failures is most destructive to learning?
  3. What is the minimum acceptable operating envelope for the full system?

Design:
  - Run the FULL pipeline with simultaneous degradation across:
    * Classifier error rate (0 - 40%)
    * BKT parameter misspecification (1.0x - 3.0x)
    * Concept selection noise (0 - 50% random selection)
  - Factorial sweep over all three dimensions.
  - Measure: test score gain, mastery rate, efficiency (gain per interaction).

Usage:
    cd <project_root>
    python experiments/09_end_to_end_stress/run.py

Outputs:
    experiments/09_end_to_end_stress/artifacts/results.json
    experiments/09_end_to_end_stress/artifacts/heatmap_classifier_bkt.png
    experiments/09_end_to_end_stress/artifacts/heatmap_classifier_noise.png
    experiments/09_end_to_end_stress/artifacts/degradation_surface.png
    experiments/09_end_to_end_stress/artifacts/failure_ranking.png
"""

from __future__ import annotations

import json
import random
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from knowledge_graph import KnowledgeGraph, StudentState, next_action
from simulated_student import SimulatedStudent, generate_students, load_problem_bank
from simulated_rct_v2 import (
    split_problem_bank,
    administer_test,
    adaptive_strategy_v2,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
KG_PATH = PROJECT_ROOT / "data" / "knowledge_graph.json"
SEED = 42
N_STUDENTS = 200
N_INTERACTIONS = 40

# Sweep levels
CLASSIFIER_ERROR_RATES = [0.0, 0.10, 0.20, 0.30, 0.40]
BKT_SCALES = [1.0, 1.5, 2.0, 3.0]
CONCEPT_NOISE_RATES = [0.0, 0.15, 0.30, 0.50]


# ─── Degraded Pipeline ───────────────────────────────────────────────────────

def inject_classifier_error(
    response: dict,
    kg: KnowledgeGraph,
    concept_id: str,
    error_rate: float,
) -> dict | None:
    """Simulate classifier error on what the student submitted.

    Returns the misconception ID that the (degraded) classifier reports,
    or None if it predicts "no misconception."
    """
    true_misconception = response.get("misconception_used")

    if random.random() >= error_rate:
        return true_misconception  # Correct classification

    # Error: pick a wrong misconception from this concept (or None)
    concept = kg.concepts.get(concept_id)
    if concept is None:
        return true_misconception

    all_misc = [m.id for m in concept.misconceptions]
    if true_misconception in all_misc:
        others = [m for m in all_misc if m != true_misconception]
    else:
        others = all_misc

    if not others:
        return None
    return random.choice(others)


def noisy_concept_selection(
    tutor_state: StudentState,
    kg: KnowledgeGraph,
    consecutive_same: int,
    last_concept: str | None,
    noise_rate: float,
) -> str:
    """With probability noise_rate, pick a random concept instead of the adaptive choice."""
    if random.random() < noise_rate:
        concept_ids = [c.id for c in kg.concepts_by_level()]
        return random.choice(concept_ids)
    return adaptive_strategy_v2(tutor_state, kg, consecutive_same, last_concept)


def run_degraded_pipeline(
    classifier_error: float,
    bkt_scale: float,
    concept_noise: float,
    n_students: int,
    n_interactions: int,
    seed: int,
) -> dict:
    """Run the full tutoring pipeline with specified degradation levels."""
    random.seed(seed)
    np.random.seed(seed)

    # Perturbed KG for BKT
    kg = KnowledgeGraph.from_json(KG_PATH)
    if bkt_scale != 1.0:
        for concept in kg.concepts.values():
            for p in ["p_learn", "p_guess", "p_slip"]:
                if p in concept.bkt_params:
                    original = concept.bkt_params[p]
                    concept.bkt_params[p] = min(0.99, max(0.01, original * bkt_scale))

    full_bank = load_problem_bank()
    practice_bank, test_bank = split_problem_bank(full_bank, n_test_per_concept=2, seed=seed)
    students = generate_students(n=n_students, seed=seed)
    for s in students:
        s.targeted_resolution = 0.50

    gains = []
    mastery_counts = []
    concept_ids = [c.id for c in kg.concepts_by_level()]

    for student in students:
        tutor_state = StudentState(kg)
        pre_test = administer_test(student, test_bank)
        consecutive_same = 0
        last_concept = None

        for t in range(n_interactions):
            # Concept selection (potentially noisy)
            concept_id = noisy_concept_selection(
                tutor_state, kg, consecutive_same, last_concept, concept_noise,
            )

            if concept_id == last_concept:
                consecutive_same += 1
            else:
                consecutive_same = 0
            last_concept = concept_id

            problems = practice_bank.get(concept_id, [])
            if not problems:
                continue
            problem = random.choice(problems)
            response = student.respond(problem)

            # Classifier (potentially erroneous)
            detected = inject_classifier_error(response, kg, concept_id, classifier_error)

            # BKT update (using potentially miscalibrated params)
            tutor_state.update(concept_id, correct=response["correct"], confidence=0.8)

            # Instruction (targeting whatever the classifier detected)
            student.receive_instruction(concept_id, targeted_misconception=detected)

        post_test = administer_test(student, test_bank)
        gain = post_test["aggregate"]["proportion"] - pre_test["aggregate"]["proportion"]
        gains.append(gain)

        mastered = sum(1 for c in concept_ids if tutor_state.mastery.get(c, 0) >= 0.85)
        mastery_counts.append(mastered / len(concept_ids))

    return {
        "classifier_error": classifier_error,
        "bkt_scale": bkt_scale,
        "concept_noise": concept_noise,
        "mean_gain": round(float(np.mean(gains)), 4),
        "std_gain": round(float(np.std(gains)), 4),
        "mastery_rate": round(float(np.mean(mastery_counts)), 4),
        "efficiency": round(float(np.mean(gains)) / n_interactions, 6),
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_heatmap(
    results: list[dict],
    x_key: str,
    y_key: str,
    fixed_key: str,
    fixed_val: float,
    filename: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """2D heatmap of mean_gain for a slice of the 3D parameter space."""
    filtered = [r for r in results if abs(r[fixed_key] - fixed_val) < 0.001]

    x_vals = sorted(set(r[x_key] for r in filtered))
    y_vals = sorted(set(r[y_key] for r in filtered))

    grid = np.zeros((len(y_vals), len(x_vals)))
    for r in filtered:
        xi = x_vals.index(r[x_key])
        yi = y_vals.index(r[y_key])
        grid[yi, xi] = r["mean_gain"]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", origin="lower")

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{v:.0%}" if v <= 1 else f"{v:.1f}x" for v in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v:.0%}" if v <= 1 else f"{v:.1f}x" for v in y_vals])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if grid[i, j] < 0.15 else "black")

    plt.colorbar(im, label="Mean Test Score Gain")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / filename, dpi=150)
    plt.close()


def plot_degradation_surface(results: list[dict]) -> None:
    """3D view of the degradation space: each point is a condition."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = [r["classifier_error"] for r in results]
    ys = [r["bkt_scale"] for r in results]
    zs = [r["concept_noise"] for r in results]
    colors = [r["mean_gain"] for r in results]

    p = ax.scatter(xs, ys, zs, c=colors, cmap="RdYlGn", s=80, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Classifier Error Rate")
    ax.set_ylabel("BKT Scale Factor")
    ax.set_zlabel("Concept Selection Noise")
    ax.set_title("End-to-End Degradation Surface")
    fig.colorbar(p, label="Mean Gain", shrink=0.6)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "degradation_surface.png", dpi=150)
    plt.close()


def plot_failure_ranking(results: list[dict]) -> None:
    """Rank degradation conditions by severity."""
    baseline = next(
        (r for r in results
         if r["classifier_error"] == 0 and r["bkt_scale"] == 1.0 and r["concept_noise"] == 0),
        None,
    )
    if baseline is None:
        return

    base_gain = baseline["mean_gain"]
    degradations = []
    for r in results:
        if r is baseline:
            continue
        pct_loss = (base_gain - r["mean_gain"]) / base_gain if base_gain > 0 else 0
        degradations.append({
            "label": f"Cls={r['classifier_error']:.0%} BKT={r['bkt_scale']:.1f}x Noise={r['concept_noise']:.0%}",
            "pct_loss": pct_loss,
            "gain": r["mean_gain"],
        })

    degradations.sort(key=lambda d: d["pct_loss"], reverse=True)
    top_n = min(20, len(degradations))
    top = degradations[:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors_list = ["#F44336" if d["pct_loss"] > 0.5 else "#FF9800" if d["pct_loss"] > 0.25 else "#4CAF50"
                   for d in top]
    bars = ax.barh(range(top_n), [d["pct_loss"] * 100 for d in top], color=colors_list)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([d["label"] for d in top], fontsize=8)
    ax.set_xlabel("% Gain Loss vs Baseline")
    ax.set_title(f"Top {top_n} Most Destructive Failure Combinations\n(Baseline gain: {base_gain:.4f})")
    ax.axvline(x=50, color="red", linestyle="--", alpha=0.5, label="50% loss")
    ax.axvline(x=25, color="orange", linestyle="--", alpha=0.5, label="25% loss")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "failure_ranking.png", dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 09: End-to-End Pipeline Stress Test")
    print("=" * 70)

    conditions = list(product(CLASSIFIER_ERROR_RATES, BKT_SCALES, CONCEPT_NOISE_RATES))
    total = len(conditions)
    print(f"\nSweeping {total} conditions "
          f"({len(CLASSIFIER_ERROR_RATES)} x {len(BKT_SCALES)} x {len(CONCEPT_NOISE_RATES)})")
    print(f"  Classifier error rates: {CLASSIFIER_ERROR_RATES}")
    print(f"  BKT scale factors:      {BKT_SCALES}")
    print(f"  Concept noise rates:    {CONCEPT_NOISE_RATES}")
    print(f"  Students per condition:  {N_STUDENTS}")
    print(f"  Interactions per student: {N_INTERACTIONS}")

    results = []
    for i, (cls_err, bkt_scale, noise) in enumerate(conditions, 1):
        print(f"\n[{i:3d}/{total}] cls_err={cls_err:.0%} bkt_scale={bkt_scale:.1f}x noise={noise:.0%}")
        r = run_degraded_pipeline(cls_err, bkt_scale, noise, N_STUDENTS, N_INTERACTIONS, SEED)
        print(f"         gain={r['mean_gain']:.4f} +/- {r['std_gain']:.4f}  "
              f"mastery={r['mastery_rate']:.4f}  efficiency={r['efficiency']:.6f}")
        results.append(r)

    # Save
    full_output = {
        "metadata": {
            "experiment": "09_end_to_end_stress",
            "n_students": N_STUDENTS,
            "n_interactions": N_INTERACTIONS,
            "seed": SEED,
            "classifier_error_rates": CLASSIFIER_ERROR_RATES,
            "bkt_scales": BKT_SCALES,
            "concept_noise_rates": CONCEPT_NOISE_RATES,
            "total_conditions": total,
        },
        "results": results,
    }
    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nResults saved to {ARTIFACTS / 'results.json'}")

    # Plots
    print("\nGenerating plots...")

    # Heatmap: classifier error vs BKT scale (noise=0)
    plot_heatmap(
        results,
        x_key="classifier_error", y_key="bkt_scale",
        fixed_key="concept_noise", fixed_val=0.0,
        filename="heatmap_classifier_bkt.png",
        xlabel="Classifier Error Rate", ylabel="BKT Scale Factor",
        title="Gain: Classifier Error x BKT Misspecification (no concept noise)",
    )

    # Heatmap: classifier error vs concept noise (bkt=1.0)
    plot_heatmap(
        results,
        x_key="classifier_error", y_key="concept_noise",
        fixed_key="bkt_scale", fixed_val=1.0,
        filename="heatmap_classifier_noise.png",
        xlabel="Classifier Error Rate", ylabel="Concept Selection Noise",
        title="Gain: Classifier Error x Concept Noise (correct BKT params)",
    )

    plot_degradation_surface(results)
    plot_failure_ranking(results)
    print("Plots saved.")

    # Summary analysis
    print("\n" + "=" * 70)
    print("EXPERIMENT 09 SUMMARY")
    print("=" * 70)

    baseline = next(
        (r for r in results
         if r["classifier_error"] == 0 and r["bkt_scale"] == 1.0 and r["concept_noise"] == 0),
        None,
    )
    if baseline:
        base_gain = baseline["mean_gain"]
        print(f"\n  Baseline (no degradation): gain = {base_gain:.4f}")

        worst = min(results, key=lambda r: r["mean_gain"])
        print(f"  Worst condition: cls={worst['classifier_error']:.0%} "
              f"bkt={worst['bkt_scale']}x noise={worst['concept_noise']:.0%} "
              f"gain={worst['mean_gain']:.4f} "
              f"({((base_gain - worst['mean_gain']) / base_gain * 100) if base_gain > 0 else 0:.1f}% loss)")

        # Error compounding analysis
        print("\n  Error compounding analysis:")
        for cls_err in [0.20, 0.40]:
            for bkt_s in [2.0, 3.0]:
                # Individual degradation
                cls_only = next(
                    (r for r in results
                     if r["classifier_error"] == cls_err and r["bkt_scale"] == 1.0 and r["concept_noise"] == 0),
                    None,
                )
                bkt_only = next(
                    (r for r in results
                     if r["classifier_error"] == 0 and r["bkt_scale"] == bkt_s and r["concept_noise"] == 0),
                    None,
                )
                combined = next(
                    (r for r in results
                     if r["classifier_error"] == cls_err and r["bkt_scale"] == bkt_s and r["concept_noise"] == 0),
                    None,
                )
                if cls_only and bkt_only and combined and base_gain > 0:
                    individual_sum = (base_gain - cls_only["mean_gain"]) + (base_gain - bkt_only["mean_gain"])
                    actual_loss = base_gain - combined["mean_gain"]
                    ratio = actual_loss / individual_sum if individual_sum > 0 else 0
                    mode = "multiplicative" if ratio > 1.2 else "additive" if ratio > 0.8 else "sub-additive"
                    print(f"    cls={cls_err:.0%}+bkt={bkt_s}x: "
                          f"individual_losses={individual_sum:.4f}, "
                          f"combined_loss={actual_loss:.4f}, "
                          f"ratio={ratio:.2f} ({mode})")

        # Operating envelope: find conditions where gain >= 80% of baseline
        acceptable = [r for r in results if r["mean_gain"] >= 0.8 * base_gain]
        print(f"\n  Operating envelope (>=80% of baseline gain):")
        print(f"    {len(acceptable)}/{len(results)} conditions meet threshold")
        if acceptable:
            max_cls = max(r["classifier_error"] for r in acceptable)
            max_bkt = max(r["bkt_scale"] for r in acceptable)
            max_noise = max(r["concept_noise"] for r in acceptable)
            print(f"    Max tolerable: cls_error={max_cls:.0%}, bkt_scale={max_bkt}x, noise={max_noise:.0%}")


if __name__ == "__main__":
    main()
