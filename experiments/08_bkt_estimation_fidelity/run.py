"""Experiment 08: BKT Estimation Fidelity and Downstream Impact.

Questions:
  1. How accurately does BKT track the simulated student's true p_know?
  2. How does BKT parameter misspecification (wrong p_learn, p_guess, p_slip)
     degrade concept selection and learning outcomes?
  3. Which BKT parameter is the system most sensitive to?

Design:
  - Part A: Track BKT estimate vs student's true p_know over 40 interactions,
            measure RMSE, bias, and calibration.
  - Part B: Systematically perturb each BKT parameter (+/- 50%, 100%) and
            measure the impact on concept selection accuracy and learning gains.
  - Part C: Joint perturbation - all params off simultaneously - to test
            whether errors compound multiplicatively.

Usage:
    cd <project_root>
    python experiments/08_bkt_estimation_fidelity/run.py

Outputs:
    experiments/08_bkt_estimation_fidelity/artifacts/results.json
    experiments/08_bkt_estimation_fidelity/artifacts/tracking_accuracy.png
    experiments/08_bkt_estimation_fidelity/artifacts/parameter_sensitivity.png
    experiments/08_bkt_estimation_fidelity/artifacts/concept_selection_error.png
    experiments/08_bkt_estimation_fidelity/artifacts/joint_perturbation.png
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


# ─── Part A: BKT Tracking Accuracy ───────────────────────────────────────────

def run_tracking_analysis(
    students: list[SimulatedStudent],
    kg: KnowledgeGraph,
    practice_bank: dict[str, list[dict]],
    n_interactions: int,
) -> dict:
    """Track BKT estimate vs true p_know for each student at each timestep."""
    concept_ids = [c.id for c in kg.concepts_by_level()]

    all_rmse_over_time = np.zeros(n_interactions)
    all_bias_over_time = np.zeros(n_interactions)
    all_correct_concept_selections = 0
    all_total_selections = 0
    per_concept_rmse = defaultdict(list)
    student_count = 0

    for student in students:
        tutor_state = StudentState(kg)
        consecutive_same = 0
        last_concept = None

        for t in range(n_interactions):
            # Record BKT vs truth BEFORE this interaction
            for cid in concept_ids:
                bkt_est = tutor_state.mastery.get(cid, 0.5)
                true_val = student.p_know.get(cid, 0.1)
                error = bkt_est - true_val
                per_concept_rmse[cid].append(error ** 2)

            # Mean absolute error across concepts at this timestep
            errors = [
                tutor_state.mastery.get(c, 0.5) - student.p_know.get(c, 0.1)
                for c in concept_ids
            ]
            all_rmse_over_time[t] += np.sqrt(np.mean([e**2 for e in errors]))
            all_bias_over_time[t] += np.mean(errors)

            # Concept selection
            concept_id = adaptive_strategy_v2(
                tutor_state, kg, consecutive_same, last_concept,
            )

            # "Correct" selection = would an oracle with true p_know pick the same?
            true_weakest = min(concept_ids, key=lambda c: student.p_know.get(c, 0.1))
            if concept_id == true_weakest:
                all_correct_concept_selections += 1
            all_total_selections += 1

            if concept_id == last_concept:
                consecutive_same += 1
            else:
                consecutive_same = 0
            last_concept = concept_id

            # Execute interaction
            problems = practice_bank.get(concept_id, [])
            if not problems:
                continue
            problem = random.choice(problems)
            response = student.respond(problem)

            tutor_state.update(concept_id, correct=response["correct"], confidence=0.8)
            targeted = response.get("misconception_used")
            student.receive_instruction(concept_id, targeted_misconception=targeted)

        student_count += 1

    all_rmse_over_time /= student_count
    all_bias_over_time /= student_count

    return {
        "rmse_over_time": all_rmse_over_time.tolist(),
        "bias_over_time": all_bias_over_time.tolist(),
        "concept_selection_accuracy": round(
            all_correct_concept_selections / max(all_total_selections, 1), 4
        ),
        "per_concept_rmse": {
            cid: round(float(np.sqrt(np.mean(vals))), 4)
            for cid, vals in per_concept_rmse.items()
        },
    }


# ─── Part B: Parameter Sensitivity ───────────────────────────────────────────

def run_parameter_perturbation(
    param_name: str,
    scale_factors: list[float],
    n_students: int,
    n_interactions: int,
    seed: int,
) -> list[dict]:
    """Perturb one BKT parameter and measure downstream impact."""
    kg_baseline = KnowledgeGraph.from_json(KG_PATH)
    full_bank = load_problem_bank()
    practice_bank, test_bank = split_problem_bank(full_bank, n_test_per_concept=2, seed=seed)

    results = []
    for scale in scale_factors:
        random.seed(seed)
        np.random.seed(seed)

        # Create a perturbed knowledge graph
        kg = KnowledgeGraph.from_json(KG_PATH)
        for concept in kg.concepts.values():
            if param_name in concept.bkt_params:
                original = concept.bkt_params[param_name]
                concept.bkt_params[param_name] = min(0.99, max(0.01, original * scale))

        students = generate_students(n=n_students, seed=seed)
        for s in students:
            s.targeted_resolution = 0.50

        gains = []
        bkt_rmses = []
        concept_acc = []

        for student in students:
            tutor_state = StudentState(kg)
            pre_test = administer_test(student, test_bank)
            consecutive_same = 0
            last_concept = None
            concept_ids = [c.id for c in kg.concepts_by_level()]
            correct_selections = 0
            total_selections = 0

            for t in range(n_interactions):
                concept_id = adaptive_strategy_v2(
                    tutor_state, kg, consecutive_same, last_concept,
                )
                # Oracle comparison
                true_weakest = min(concept_ids, key=lambda c: student.p_know.get(c, 0.1))
                if concept_id == true_weakest:
                    correct_selections += 1
                total_selections += 1

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
                tutor_state.update(concept_id, correct=response["correct"], confidence=0.8)
                targeted = response.get("misconception_used")
                student.receive_instruction(concept_id, targeted_misconception=targeted)

            post_test = administer_test(student, test_bank)
            gains.append(post_test["aggregate"]["proportion"] - pre_test["aggregate"]["proportion"])

            rmse = np.sqrt(np.mean([
                (tutor_state.mastery.get(c, 0.5) - student.p_know.get(c, 0.1)) ** 2
                for c in concept_ids
            ]))
            bkt_rmses.append(float(rmse))
            concept_acc.append(correct_selections / max(total_selections, 1))

        results.append({
            "param": param_name,
            "scale": scale,
            "mean_gain": round(float(np.mean(gains)), 4),
            "std_gain": round(float(np.std(gains)), 4),
            "mean_bkt_rmse": round(float(np.mean(bkt_rmses)), 4),
            "concept_selection_accuracy": round(float(np.mean(concept_acc)), 4),
        })

    return results


# ─── Part C: Joint Perturbation ──────────────────────────────────────────────

def run_joint_perturbation(
    scales: list[float],
    n_students: int,
    n_interactions: int,
    seed: int,
) -> list[dict]:
    """Perturb ALL BKT params simultaneously at the same scale."""
    full_bank = load_problem_bank()
    practice_bank, test_bank = split_problem_bank(full_bank, n_test_per_concept=2, seed=seed)

    results = []
    for scale in scales:
        random.seed(seed)
        np.random.seed(seed)

        kg = KnowledgeGraph.from_json(KG_PATH)
        for concept in kg.concepts.values():
            for p in ["p_learn", "p_guess", "p_slip"]:
                if p in concept.bkt_params:
                    original = concept.bkt_params[p]
                    concept.bkt_params[p] = min(0.99, max(0.01, original * scale))

        students = generate_students(n=n_students, seed=seed)
        for s in students:
            s.targeted_resolution = 0.50

        gains = []
        for student in students:
            tutor_state = StudentState(kg)
            pre_test = administer_test(student, test_bank)
            consecutive_same = 0
            last_concept = None

            for t in range(n_interactions):
                concept_id = adaptive_strategy_v2(
                    tutor_state, kg, consecutive_same, last_concept,
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
                tutor_state.update(concept_id, correct=response["correct"], confidence=0.8)
                targeted = response.get("misconception_used")
                student.receive_instruction(concept_id, targeted_misconception=targeted)

            post_test = administer_test(student, test_bank)
            gains.append(post_test["aggregate"]["proportion"] - pre_test["aggregate"]["proportion"])

        results.append({
            "joint_scale": scale,
            "mean_gain": round(float(np.mean(gains)), 4),
            "std_gain": round(float(np.std(gains)), 4),
        })

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_tracking_accuracy(tracking: dict) -> None:
    """BKT RMSE and bias over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ts = list(range(len(tracking["rmse_over_time"])))

    axes[0].plot(ts, tracking["rmse_over_time"], "b-", linewidth=2)
    axes[0].set_xlabel("Interaction Number")
    axes[0].set_ylabel("RMSE (BKT estimate - true p_know)")
    axes[0].set_title("BKT Tracking RMSE Over Time")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ts, tracking["bias_over_time"], "r-", linewidth=2)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Interaction Number")
    axes[1].set_ylabel("Bias (BKT estimate - true p_know)")
    axes[1].set_title("BKT Estimation Bias Over Time")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "tracking_accuracy.png", dpi=150)
    plt.close()


def plot_parameter_sensitivity(param_results: dict[str, list[dict]]) -> None:
    """Gain and concept selection accuracy vs parameter perturbation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"p_learn": "#2196F3", "p_guess": "#FF9800", "p_slip": "#F44336"}

    for param, results in param_results.items():
        scales = [r["scale"] for r in results]
        gains = [r["mean_gain"] for r in results]
        accs = [r["concept_selection_accuracy"] for r in results]

        axes[0].plot(scales, gains, "o-", label=param, color=colors[param], linewidth=2)
        axes[1].plot(scales, accs, "o-", label=param, color=colors[param], linewidth=2)

    axes[0].axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="True value")
    axes[0].set_xlabel("Parameter Scale Factor")
    axes[0].set_ylabel("Mean Test Score Gain")
    axes[0].set_title("Learning Gain vs BKT Parameter Perturbation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="True value")
    axes[1].set_xlabel("Parameter Scale Factor")
    axes[1].set_ylabel("Concept Selection Accuracy")
    axes[1].set_title("Concept Selection vs BKT Parameter Perturbation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "parameter_sensitivity.png", dpi=150)
    plt.close()


def plot_concept_selection_error(tracking: dict) -> None:
    """Per-concept BKT RMSE."""
    concepts = list(tracking["per_concept_rmse"].keys())
    rmses = [tracking["per_concept_rmse"][c] for c in concepts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(concepts)), rmses, color="#2196F3")
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=30, ha="right")
    ax.set_ylabel("RMSE (BKT estimate - true p_know)")
    ax.set_title("BKT Estimation Error by Concept")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "concept_selection_error.png", dpi=150)
    plt.close()


def plot_joint_perturbation(joint_results: list[dict], param_results: dict) -> None:
    """Compare joint perturbation vs individual param perturbation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Joint
    scales = [r["joint_scale"] for r in joint_results]
    gains = [r["mean_gain"] for r in joint_results]
    ax.plot(scales, gains, "ko-", linewidth=3, markersize=8, label="All params jointly")

    # Individual parameter lines for comparison
    colors = {"p_learn": "#2196F3", "p_guess": "#FF9800", "p_slip": "#F44336"}
    for param, results in param_results.items():
        p_scales = [r["scale"] for r in results]
        p_gains = [r["mean_gain"] for r in results]
        ax.plot(p_scales, p_gains, "o--", label=f"{param} only",
                color=colors[param], linewidth=1.5, alpha=0.6)

    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Parameter Scale Factor")
    ax.set_ylabel("Mean Test Score Gain")
    ax.set_title("Joint vs Individual BKT Parameter Perturbation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "joint_perturbation.png", dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 08: BKT Estimation Fidelity and Downstream Impact")
    print("=" * 70)

    kg = KnowledgeGraph.from_json(KG_PATH)
    full_bank = load_problem_bank()
    practice_bank, test_bank = split_problem_bank(full_bank, n_test_per_concept=2, seed=SEED)

    # Part A: Tracking accuracy
    print("\n--- Part A: BKT Tracking Accuracy ---")
    random.seed(SEED)
    np.random.seed(SEED)
    students = generate_students(n=N_STUDENTS, seed=SEED)
    for s in students:
        s.targeted_resolution = 0.50

    tracking = run_tracking_analysis(students, kg, practice_bank, N_INTERACTIONS)
    print(f"  Concept selection accuracy (vs oracle): {tracking['concept_selection_accuracy']:.4f}")
    print(f"  Per-concept RMSE:")
    for cid, rmse in tracking["per_concept_rmse"].items():
        print(f"    {cid}: {rmse:.4f}")
    print(f"  Initial RMSE: {tracking['rmse_over_time'][0]:.4f}")
    print(f"  Final RMSE:   {tracking['rmse_over_time'][-1]:.4f}")

    # Part B: Parameter sensitivity
    print("\n--- Part B: BKT Parameter Sensitivity ---")
    scale_factors = [0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0, 3.0]

    param_results = {}
    for param in ["p_learn", "p_guess", "p_slip"]:
        print(f"\n  Perturbing: {param}")
        results = run_parameter_perturbation(param, scale_factors, N_STUDENTS, N_INTERACTIONS, SEED)
        param_results[param] = results
        for r in results:
            print(f"    scale={r['scale']:.2f}  gain={r['mean_gain']:.4f}  "
                  f"bkt_rmse={r['mean_bkt_rmse']:.4f}  "
                  f"concept_acc={r['concept_selection_accuracy']:.4f}")

    # Part C: Joint perturbation
    print("\n--- Part C: Joint Perturbation ---")
    joint_scales = [0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0, 3.0]
    joint_results = run_joint_perturbation(joint_scales, N_STUDENTS, N_INTERACTIONS, SEED)
    for r in joint_results:
        print(f"  joint_scale={r['joint_scale']:.2f}  gain={r['mean_gain']:.4f} +/- {r['std_gain']:.4f}")

    # Save
    full_output = {
        "metadata": {
            "experiment": "08_bkt_estimation_fidelity",
            "n_students": N_STUDENTS,
            "n_interactions": N_INTERACTIONS,
            "seed": SEED,
        },
        "tracking": {
            "concept_selection_accuracy": tracking["concept_selection_accuracy"],
            "per_concept_rmse": tracking["per_concept_rmse"],
            "initial_rmse": tracking["rmse_over_time"][0],
            "final_rmse": tracking["rmse_over_time"][-1],
        },
        "parameter_sensitivity": param_results,
        "joint_perturbation": joint_results,
    }

    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nResults saved to {ARTIFACTS / 'results.json'}")

    # Plots
    print("\nGenerating plots...")
    plot_tracking_accuracy(tracking)
    plot_parameter_sensitivity(param_results)
    plot_concept_selection_error(tracking)
    plot_joint_perturbation(joint_results, param_results)
    print("Plots saved.")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 08 SUMMARY")
    print("=" * 70)

    # Which param is most sensitive?
    sensitivities = {}
    for param, results in param_results.items():
        baseline = next(r for r in results if r["scale"] == 1.0)
        worst = min(results, key=lambda r: r["mean_gain"])
        sensitivities[param] = {
            "baseline_gain": baseline["mean_gain"],
            "worst_gain": worst["mean_gain"],
            "worst_scale": worst["scale"],
            "degradation": round(baseline["mean_gain"] - worst["mean_gain"], 4),
        }
    most_sensitive = max(sensitivities, key=lambda p: sensitivities[p]["degradation"])
    print(f"\n  Most sensitive parameter: {most_sensitive}")
    for param, s in sensitivities.items():
        print(f"  {param}: baseline={s['baseline_gain']:.4f}, "
              f"worst={s['worst_gain']:.4f} (at {s['worst_scale']}x), "
              f"degradation={s['degradation']:.4f}")


if __name__ == "__main__":
    main()
