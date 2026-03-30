"""Experiment 05: IRT-Based vs Categorical Problem Selection.

Simulated RCT comparing four problem selection strategies:
  1. IRT-targeted: select problems near P(correct) ~ 0.70
  2. Categorical-easy: always pick the easiest unseen problem
  3. Categorical-hard: always pick the hardest unseen problem
  4. Random: pick a random unseen problem

Reuses SimulatedStudent infrastructure with held-out test evaluation.

Usage:
    cd <project_root>
    python experiments/05_irt_vs_categorical/run.py

Outputs:
    experiments/05_irt_vs_categorical/artifacts/results.json
    experiments/05_irt_vs_categorical/artifacts/learning_curves.png
    experiments/05_irt_vs_categorical/artifacts/difficulty_targeting.png
    experiments/05_irt_vs_categorical/artifacts/bank_size_scaling.png
"""

from __future__ import annotations

import json
import math
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

from simulated_student import SimulatedStudent, generate_students, load_problem_bank
from simulated_rct_v2 import split_problem_bank, administer_test

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SEED = 42
N_STUDENTS = 500
N_INTERACTIONS = 40

# IRT difficulty assignments
IRT_DIFFICULTY = {"easy": -1.5, "medium": 0.0, "hard": 1.5}
TARGET_P = 0.70


# ─── IRT Utilities ────────────────────────────────────────────────────────────

def mastery_to_theta(mastery: float) -> float:
    """Convert BKT mastery to IRT ability (log-odds)."""
    m = max(0.01, min(0.99, mastery))
    return math.log(m / (1 - m))


def irt_probability(theta: float, b: float) -> float:
    """1PL IRT: P(correct | theta, b)."""
    return 1.0 / (1.0 + math.exp(-(theta - b)))


def assign_irt_params(problem_bank: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Add irt_b to each problem based on its difficulty label."""
    enriched = {}
    for concept_id, problems in problem_bank.items():
        enriched[concept_id] = []
        for p in problems:
            p2 = dict(p)
            difficulty = p.get("difficulty", "medium")
            p2["irt_b"] = IRT_DIFFICULTY.get(difficulty, 0.0)
            enriched[concept_id].append(p2)
    return enriched


# ─── Problem Selection Strategies ─────────────────────────────────────────────

def select_irt_targeted(
    concept_id: str,
    student: SimulatedStudent,
    bank: dict[str, list[dict]],
    seen: set[str],
) -> dict | None:
    """Select the unseen problem closest to target P(correct) for this student."""
    problems = [p for p in bank.get(concept_id, []) if p["problem_id"] not in seen]
    if not problems:
        return None

    mastery = student.p_know.get(concept_id, 0.1)
    theta = mastery_to_theta(mastery)
    target_b = theta - math.log(TARGET_P / (1 - TARGET_P))

    # Pick the problem with irt_b closest to target_b
    problems.sort(key=lambda p: abs(p.get("irt_b", 0.0) - target_b))
    return problems[0]


def select_easiest(
    concept_id: str,
    bank: dict[str, list[dict]],
    seen: set[str],
) -> dict | None:
    """Select the easiest unseen problem (lowest irt_b)."""
    problems = [p for p in bank.get(concept_id, []) if p["problem_id"] not in seen]
    if not problems:
        return None
    problems.sort(key=lambda p: p.get("irt_b", 0.0))
    return problems[0]


def select_hardest(
    concept_id: str,
    bank: dict[str, list[dict]],
    seen: set[str],
) -> dict | None:
    """Select the hardest unseen problem (highest irt_b)."""
    problems = [p for p in bank.get(concept_id, []) if p["problem_id"] not in seen]
    if not problems:
        return None
    problems.sort(key=lambda p: -p.get("irt_b", 0.0))
    return problems[0]


def select_random(
    concept_id: str,
    bank: dict[str, list[dict]],
    seen: set[str],
    rng: random.Random,
) -> dict | None:
    """Select a random unseen problem."""
    problems = [p for p in bank.get(concept_id, []) if p["problem_id"] not in seen]
    if not problems:
        return None
    return rng.choice(problems)


# ─── Concept Selection (shared across conditions) ────────────────────────────

def select_concept(student: SimulatedStudent, concept_order: list[str]) -> str:
    """BKT-driven concept selection (same as adaptive in v2 RCT).
    Pick the lowest-mastery concept that isn't fully mastered.
    """
    unmastered = [
        c for c in concept_order
        if student.p_know.get(c, 0.1) < student.mastery_threshold
    ]
    if not unmastered:
        # All mastered; review weakest
        return min(concept_order, key=lambda c: student.p_know.get(c, 0.1))
    return unmastered[0]


# ─── Simulation ───────────────────────────────────────────────────────────────

def run_condition(
    condition: str,
    students: list[SimulatedStudent],
    practice_bank: dict[str, list[dict]],
    test_bank: dict[str, list[dict]],
    concept_order: list[str],
    n_interactions: int,
    seed: int,
) -> dict:
    """Run one experimental condition."""
    rng = random.Random(seed)

    pre_scores = []
    post_scores = []
    difficulty_hits = []  # fraction of problems in desirable difficulty range
    concepts_mastered_list = []

    for student in students:
        # Pre-test
        pre = administer_test(student, test_bank)
        pre_scores.append(pre["aggregate"]["proportion"])

        seen: set[str] = set()
        hit_count = 0
        total_assigned = 0

        for t in range(n_interactions):
            concept_id = select_concept(student, concept_order)

            # Select problem based on condition
            if condition == "irt_targeted":
                problem = select_irt_targeted(concept_id, student, practice_bank, seen)
            elif condition == "categorical_easy":
                problem = select_easiest(concept_id, practice_bank, seen)
            elif condition == "categorical_hard":
                problem = select_hardest(concept_id, practice_bank, seen)
            elif condition == "random":
                problem = select_random(concept_id, practice_bank, seen, rng)
            else:
                raise ValueError(f"Unknown condition: {condition}")

            if problem is None:
                # All problems seen for this concept, pick any concept
                for fallback_concept in concept_order:
                    if condition == "irt_targeted":
                        problem = select_irt_targeted(fallback_concept, student, practice_bank, seen)
                    elif condition == "categorical_easy":
                        problem = select_easiest(fallback_concept, practice_bank, seen)
                    elif condition == "categorical_hard":
                        problem = select_hardest(fallback_concept, practice_bank, seen)
                    else:
                        problem = select_random(fallback_concept, practice_bank, seen, rng)
                    if problem:
                        concept_id = fallback_concept
                        break

            if problem is None:
                # Exhausted all problems
                break

            seen.add(problem["problem_id"])

            # Student responds
            response = student.respond(problem)

            # Measure desirable difficulty: was P(correct) in [0.55, 0.85]?
            mastery = student.p_know.get(concept_id, 0.1)
            theta = mastery_to_theta(mastery)
            p_correct = irt_probability(theta, problem.get("irt_b", 0.0))
            if 0.55 <= p_correct <= 0.85:
                hit_count += 1
            total_assigned += 1

            # Instruction (always targeted if misconception detected)
            targeted = response.get("misconception_used")
            student.receive_instruction(concept_id, targeted_misconception=targeted)

        # Post-test
        post = administer_test(student, test_bank)
        post_scores.append(post["aggregate"]["proportion"])

        if total_assigned > 0:
            difficulty_hits.append(hit_count / total_assigned)
        else:
            difficulty_hits.append(0.0)

        concepts_mastered = sum(
            1 for c in concept_order
            if student.p_know.get(c, 0.1) >= student.mastery_threshold
        )
        concepts_mastered_list.append(concepts_mastered)

    pre_arr = np.array(pre_scores)
    post_arr = np.array(post_scores)
    gains = post_arr - pre_arr

    return {
        "pre_mean": round(float(pre_arr.mean()), 4),
        "post_mean": round(float(post_arr.mean()), 4),
        "gain_mean": round(float(gains.mean()), 4),
        "gain_std": round(float(gains.std()), 4),
        "difficulty_hit_rate": round(float(np.mean(difficulty_hits)), 4),
        "concepts_mastered_mean": round(float(np.mean(concepts_mastered_list)), 2),
        "efficiency": round(float(gains.mean() / n_interactions), 6),
        "gains": gains.tolist(),
    }


def compute_effect_size(gains_treatment: list[float], gains_control: list[float]) -> dict:
    """Compute Cohen's d, CI, and p-value."""
    t = np.array(gains_treatment)
    c = np.array(gains_control)
    pooled_std = np.sqrt((t.var() + c.var()) / 2)
    if pooled_std < 1e-10:
        return {"d": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_value": 1.0}
    d = float((t.mean() - c.mean()) / pooled_std)
    se = pooled_std * np.sqrt(1 / len(t) + 1 / len(c))
    ci_lo = d - 1.96 * (se / pooled_std) if pooled_std > 0 else 0.0
    ci_hi = d + 1.96 * (se / pooled_std) if pooled_std > 0 else 0.0
    _, p_val = stats.ttest_ind(t, c)
    return {
        "d": round(d, 4),
        "ci_lo": round(float(ci_lo), 4),
        "ci_hi": round(float(ci_hi), 4),
        "p_value": round(float(p_val), 6),
    }


# ─── Bank Size Scaling ───────────────────────────────────────────────────────

def generate_expanded_bank(
    base_bank: dict[str, list[dict]],
    target_per_concept: int,
    rng: random.Random,
) -> dict[str, list[dict]]:
    """Expand problem bank by generating parametric variations."""
    expanded = {}
    for concept_id, problems in base_bank.items():
        expanded[concept_id] = list(problems)
        # Generate variations of existing problems
        while len(expanded[concept_id]) < target_per_concept:
            template = rng.choice(problems)
            variant = dict(template)
            variant["problem_id"] = f"{template['problem_id']}_v{len(expanded[concept_id])}"
            # Jitter the IRT difficulty slightly
            base_b = template.get("irt_b", 0.0)
            variant["irt_b"] = base_b + rng.gauss(0, 0.3)
            expanded[concept_id].append(variant)
    return expanded


def run_bank_size_scaling(
    students_seed_fn,
    test_bank: dict[str, list[dict]],
    base_practice_bank: dict[str, list[dict]],
    concept_order: list[str],
) -> list[dict]:
    """Vary bank size and compare IRT vs categorical-easy."""
    results = []
    for problems_per_concept in [3, 5, 10, 20]:
        rng = random.Random(SEED + problems_per_concept)
        expanded = generate_expanded_bank(
            base_practice_bank, problems_per_concept, rng
        )
        expanded = assign_irt_params(expanded)  # ensure IRT params

        for condition in ["irt_targeted", "categorical_easy"]:
            students = students_seed_fn()
            res = run_condition(
                condition, students, expanded, test_bank,
                concept_order, N_INTERACTIONS, SEED + hash(condition) % 2**31,
            )
            results.append({
                "problems_per_concept": problems_per_concept,
                "condition": condition,
                "gain_mean": res["gain_mean"],
                "difficulty_hit_rate": res["difficulty_hit_rate"],
            })
            print(f"  Bank size {problems_per_concept:>3d}/concept, {condition:<20s}: "
                  f"gain={res['gain_mean']:.4f}, hit_rate={res['difficulty_hit_rate']:.4f}")

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_learning_curves(condition_results: dict) -> None:
    """Bar chart of pre/post scores and gains per condition."""
    conditions = list(condition_results.keys())
    gains = [condition_results[c]["gain_mean"] for c in conditions]
    stds = [condition_results[c]["gain_std"] for c in conditions]

    colors = {
        "irt_targeted": "#2196F3",
        "categorical_easy": "#4CAF50",
        "categorical_hard": "#F44336",
        "random": "#9E9E9E",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: gain comparison
    bars = axes[0].bar(
        range(len(conditions)), gains,
        yerr=stds, capsize=5,
        color=[colors.get(c, "#999") for c in conditions],
    )
    axes[0].set_xticks(range(len(conditions)))
    axes[0].set_xticklabels([c.replace("_", " ").title() for c in conditions], rotation=15)
    axes[0].set_ylabel("Test Score Gain (post - pre)")
    axes[0].set_title("Learning Gains by Problem Selection Strategy")
    for bar, g in zip(bars, gains):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{g:.3f}", ha="center", fontsize=9)

    # Right: efficiency (gain per interaction)
    efficiencies = [condition_results[c]["efficiency"] * 1000 for c in conditions]
    axes[1].bar(
        range(len(conditions)), efficiencies,
        color=[colors.get(c, "#999") for c in conditions],
    )
    axes[1].set_xticks(range(len(conditions)))
    axes[1].set_xticklabels([c.replace("_", " ").title() for c in conditions], rotation=15)
    axes[1].set_ylabel("Gain per 1000 Interactions")
    axes[1].set_title("Learning Efficiency")

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "learning_curves.png", dpi=150)
    plt.close()


def plot_difficulty_targeting(condition_results: dict) -> None:
    """Show difficulty hit rate and concept mastery per condition."""
    conditions = list(condition_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {
        "irt_targeted": "#2196F3",
        "categorical_easy": "#4CAF50",
        "categorical_hard": "#F44336",
        "random": "#9E9E9E",
    }

    # Hit rate
    hit_rates = [condition_results[c]["difficulty_hit_rate"] for c in conditions]
    axes[0].bar(
        range(len(conditions)), hit_rates,
        color=[colors.get(c, "#999") for c in conditions],
    )
    axes[0].set_xticks(range(len(conditions)))
    axes[0].set_xticklabels([c.replace("_", " ").title() for c in conditions], rotation=15)
    axes[0].set_ylabel("Fraction in Desirable Difficulty [0.55, 0.85]")
    axes[0].set_title("Difficulty Targeting Accuracy")
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    axes[0].legend()

    # Concepts mastered
    mastered = [condition_results[c]["concepts_mastered_mean"] for c in conditions]
    axes[1].bar(
        range(len(conditions)), mastered,
        color=[colors.get(c, "#999") for c in conditions],
    )
    axes[1].set_xticks(range(len(conditions)))
    axes[1].set_xticklabels([c.replace("_", " ").title() for c in conditions], rotation=15)
    axes[1].set_ylabel("Mean Concepts Mastered")
    axes[1].set_title("Concepts Reaching Mastery Threshold")

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "difficulty_targeting.png", dpi=150)
    plt.close()


def plot_bank_size_scaling(scaling_results: list[dict]) -> None:
    """IRT vs categorical gain as bank size grows."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"irt_targeted": "#2196F3", "categorical_easy": "#4CAF50"}
    labels = {"irt_targeted": "IRT-Targeted", "categorical_easy": "Categorical Easy"}

    for condition in ["irt_targeted", "categorical_easy"]:
        subset = [r for r in scaling_results if r["condition"] == condition]
        sizes = [r["problems_per_concept"] for r in subset]
        gains = [r["gain_mean"] for r in subset]
        ax.plot(sizes, gains, "o-", label=labels[condition],
                color=colors[condition], linewidth=2, markersize=8)

    ax.set_xlabel("Problems per Concept")
    ax.set_ylabel("Mean Test Score Gain")
    ax.set_title("IRT Advantage vs Problem Bank Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "bank_size_scaling.png", dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 05: IRT-Based vs Categorical Problem Selection")
    print("=" * 60)

    # Load and prepare data
    full_bank = load_problem_bank()
    practice_bank, test_bank = split_problem_bank(full_bank, n_test_per_concept=2, seed=SEED)
    practice_bank = assign_irt_params(practice_bank)
    test_bank = assign_irt_params(test_bank)

    # Get concept order from knowledge graph
    with open(PROJECT_ROOT / "data" / "knowledge_graph.json") as f:
        kg = json.load(f)
    concept_order = [c["id"] for c in sorted(kg["concepts"], key=lambda c: c["level"])]

    print(f"Practice bank: {sum(len(v) for v in practice_bank.values())} problems")
    print(f"Test bank: {sum(len(v) for v in test_bank.values())} problems")
    print(f"Concepts: {concept_order}")

    def make_students():
        return generate_students(N_STUDENTS, seed=SEED)

    # Run all conditions
    conditions = ["irt_targeted", "categorical_easy", "categorical_hard", "random"]
    condition_results = {}

    for cond in conditions:
        print(f"\nRunning condition: {cond}...")
        students = make_students()
        condition_results[cond] = run_condition(
            cond, students, practice_bank, test_bank,
            concept_order, N_INTERACTIONS,
            SEED + hash(cond) % 2**31,
        )
        r = condition_results[cond]
        print(f"  Pre: {r['pre_mean']:.4f}, Post: {r['post_mean']:.4f}, "
              f"Gain: {r['gain_mean']:.4f}, Hit rate: {r['difficulty_hit_rate']:.4f}")

    # Effect sizes (IRT vs each baseline)
    effect_sizes = {}
    for baseline in ["categorical_easy", "categorical_hard", "random"]:
        es = compute_effect_size(
            condition_results["irt_targeted"]["gains"],
            condition_results[baseline]["gains"],
        )
        effect_sizes[f"irt_vs_{baseline}"] = es
        print(f"\n  IRT vs {baseline}: d={es['d']:.4f} [{es['ci_lo']:.4f}, {es['ci_hi']:.4f}], p={es['p_value']:.6f}")

    # Bank size scaling
    print("\n--- Bank Size Scaling ---")
    scaling_results = run_bank_size_scaling(
        make_students, test_bank, practice_bank, concept_order,
    )

    # Assemble output
    full_output = {
        "metadata": {
            "experiment": "05_irt_vs_categorical",
            "n_students": N_STUDENTS,
            "n_interactions": N_INTERACTIONS,
            "target_p": TARGET_P,
            "seed": SEED,
        },
        "condition_results": {
            cond: {k: v for k, v in data.items() if k != "gains"}
            for cond, data in condition_results.items()
        },
        "effect_sizes": effect_sizes,
        "bank_size_scaling": scaling_results,
    }

    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nResults saved to {ARTIFACTS / 'results.json'}")

    # Plots
    print("\nGenerating plots...")
    plot_learning_curves(condition_results)
    plot_difficulty_targeting(condition_results)
    plot_bank_size_scaling(scaling_results)
    print("Plots saved.")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 05 SUMMARY")
    print("=" * 60)
    for cond, r in condition_results.items():
        print(f"\n{cond.upper().replace('_', ' ')}:")
        print(f"  Gain:              {r['gain_mean']:.4f} +/- {r['gain_std']:.4f}")
        print(f"  Difficulty hit:    {r['difficulty_hit_rate']:.4f}")
        print(f"  Concepts mastered: {r['concepts_mastered_mean']:.1f}")
        print(f"  Efficiency:        {r['efficiency']:.6f} gain/interaction")


if __name__ == "__main__":
    main()
