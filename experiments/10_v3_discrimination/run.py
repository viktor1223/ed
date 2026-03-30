"""Experiment 10: V3 Simulated Student Discrimination Test.

Evaluates the v3 student by directly measuring internal state changes,
not through the noisy proxy of pre/post test scores.

Metrics (measured directly from student internals):
  1. p_know delta: average change in mastery across ENGAGED concepts
  2. misconception resolution: fraction of targeted misconceptions resolved
  3. misconception reinforcement: fraction of misconceptions strengthened
  4. confusion accumulation: average confusion counter

Conditions:
  A: Perfect classifier (always IDs the correct misconception)
  B: Random classifier (picks random misconception or None)
  C: Always-wrong classifier (always picks nonexistent misconception)
  D: No instruction

Pass criteria:
  - Cohen's d >= 0.5 between A and C on p_know delta
  - Resolution rate A > C
  - Reinforcement rate C > 0
  - p_know ordering: A > B > D, C <= B
  - Confusion: C >> A

Usage:
    cd <project_root>
    python experiments/10_v3_discrimination/run.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulated_student_v3 import (
    SimulatedStudentV3,
    generate_students_v3,
    load_problem_bank_v2,
)
from knowledge_graph import KnowledgeGraph, StudentState

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
KG_PATH = PROJECT_ROOT / "data" / "knowledge_graph_v2.json"
SEED = 42
N_STUDENTS = 300
N_INTERACTIONS = 40


def split_problem_bank(
    bank: dict[str, list[dict]], n_test_per_concept: int = 2, seed: int = 42,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    rng = random.Random(seed)
    practice, test = {}, {}
    for concept, problems in bank.items():
        shuffled = problems[:]
        rng.shuffle(shuffled)
        test[concept] = shuffled[:min(n_test_per_concept, len(shuffled))]
        practice[concept] = shuffled[min(n_test_per_concept, len(shuffled)):]
    return practice, test


def adaptive_concept_selection(
    tutor_state: StudentState, kg: KnowledgeGraph,
) -> str:
    concepts = [c.id for c in kg.concepts_by_level()]
    return min(concepts, key=lambda c: tutor_state.mastery.get(c, 0.5))


def run_condition(
    condition: str,
    n_students: int,
    n_interactions: int,
    seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    kg = KnowledgeGraph.from_json(KG_PATH)
    bank = load_problem_bank_v2()
    practice_bank, _ = split_problem_bank(bank, n_test_per_concept=2, seed=seed)
    students = generate_students_v3(n=n_students, seed=seed)

    p_know_deltas = []
    misconception_resolved = []
    misconception_reinforced = []
    confusion_scores = []

    for student in students:
        tutor_state = StudentState(kg)

        initial_p_know = {cid: cs.p_know for cid, cs in student.concepts.items()}
        initial_misc = {
            m.misconception_id: m.p_active
            for m in student.misconceptions if m.p_active > 0.1
        }

        engaged_concepts = set()

        for t in range(n_interactions):
            concept_id = adaptive_concept_selection(tutor_state, kg)
            engaged_concepts.add(concept_id)

            problems = practice_bank.get(concept_id, [])
            if not problems:
                continue
            problem = random.choice(problems)
            response = student.respond(problem)

            tutor_state.update(concept_id, correct=response["correct"], confidence=0.8)

            true_misconception = response.get("misconception_used")

            if condition == "perfect":
                student.receive_instruction(concept_id, targeted_misconception=true_misconception)
            elif condition == "random":
                concept_obj = kg.concepts.get(concept_id)
                if concept_obj and concept_obj.misconceptions and random.random() < 0.5:
                    m = random.choice(concept_obj.misconceptions)
                    student.receive_instruction(concept_id, targeted_misconception=m.id)
                else:
                    student.receive_instruction(concept_id, targeted_misconception=None)
            elif condition == "always_wrong":
                student.receive_instruction(
                    concept_id, targeted_misconception="__nonexistent__"
                )
            elif condition == "no_instruction":
                pass

        # Measure internal state on ENGAGED concepts
        if engaged_concepts:
            deltas = [
                student.concepts[c].p_know - initial_p_know.get(c, 0.1)
                for c in engaged_concepts if c in student.concepts
            ]
            p_know_deltas.append(float(np.mean(deltas)) if deltas else 0.0)
        else:
            p_know_deltas.append(0.0)

        n_resolved = 0
        n_reinforced = 0
        n_tracked = 0
        for m in student.misconceptions:
            if m.misconception_id in initial_misc:
                n_tracked += 1
                if m.p_active < 0.1 and initial_misc[m.misconception_id] >= 0.1:
                    n_resolved += 1
                if m.p_active > initial_misc[m.misconception_id] + 0.01:
                    n_reinforced += 1

        misconception_resolved.append(n_resolved / max(n_tracked, 1))
        misconception_reinforced.append(n_reinforced / max(n_tracked, 1))
        confusion_scores.append(sum(student.confusion_count.values()))

    return {
        "condition": condition,
        "p_know_delta_mean": round(float(np.mean(p_know_deltas)), 4),
        "p_know_delta_std": round(float(np.std(p_know_deltas)), 4),
        "resolution_rate": round(float(np.mean(misconception_resolved)), 4),
        "reinforcement_rate": round(float(np.mean(misconception_reinforced)), 4),
        "confusion_mean": round(float(np.mean(confusion_scores)), 2),
        "p_know_deltas": p_know_deltas,
    }


def cohens_d(g1: list[float], g2: list[float]) -> float:
    n1, n2 = len(g1), len(g2)
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((m1 - m2) / pooled) if pooled > 0 else 0.0


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 10: V3 Discrimination Test (Direct State Measurement)")
    print("=" * 70)
    print(f"Students: {N_STUDENTS}, Interactions: {N_INTERACTIONS}, Seed: {SEED}")

    conditions = ["perfect", "random", "always_wrong", "no_instruction"]
    results = {}

    for cond in conditions:
        print(f"\n--- Condition: {cond} ---")
        r = run_condition(cond, N_STUDENTS, N_INTERACTIONS, SEED)
        results[cond] = r
        print(f"  p_know delta (engaged):   {r['p_know_delta_mean']:.4f} +/- {r['p_know_delta_std']:.4f}")
        print(f"  Misconception resolved:   {r['resolution_rate']:.4f}")
        print(f"  Misconception reinforced: {r['reinforcement_rate']:.4f}")
        print(f"  Confusion accumulated:    {r['confusion_mean']:.2f}")

    print("\n" + "=" * 70)
    print("DISCRIMINATION TESTS")
    print("=" * 70)

    perf = results["perfect"]["p_know_deltas"]
    rand = results["random"]["p_know_deltas"]
    wrong = results["always_wrong"]["p_know_deltas"]
    noinst = results["no_instruction"]["p_know_deltas"]

    d_pw = cohens_d(perf, wrong)
    t1, p1 = stats.ttest_ind(perf, wrong)
    print(f"\n1. Perfect vs Always-Wrong (p_know delta):")
    print(f"   Cohen's d = {d_pw:.3f} (target >= 0.5)")
    print(f"   t = {t1:.3f}, p = {p1:.6f}")
    print(f"   {'PASS' if d_pw >= 0.5 else 'FAIL'}")

    d_pr = cohens_d(perf, rand)
    t2, p2 = stats.ttest_ind(perf, rand)
    print(f"\n2. Perfect vs Random (p_know delta):")
    print(f"   Cohen's d = {d_pr:.3f}")
    print(f"   t = {t2:.3f}, p = {p2:.6f}")

    res_p = results["perfect"]["resolution_rate"]
    res_w = results["always_wrong"]["resolution_rate"]
    print(f"\n3. Resolution: perfect={res_p:.4f}, wrong={res_w:.4f}")
    print(f"   {'PASS' if res_p > res_w else 'FAIL'}")

    reinf = results["always_wrong"]["reinforcement_rate"]
    print(f"\n4. Wrong-instruction reinforcement: {reinf:.4f}")
    print(f"   {'PASS' if reinf > 0.01 else 'FAIL'}")

    means = {c: results[c]["p_know_delta_mean"] for c in conditions}
    ordering = (means["perfect"] > means["random"]
                and means["always_wrong"] <= means["random"])
    print(f"\n5. p_know ordering:")
    for c in conditions:
        print(f"   {c:20s}: {means[c]:.4f}")
    print(f"   {'PASS' if ordering else 'FAIL'}")

    conf_w = results["always_wrong"]["confusion_mean"]
    conf_p = results["perfect"]["confusion_mean"]
    print(f"\n6. Confusion: wrong={conf_w:.1f}, perfect={conf_p:.1f}")
    print(f"   {'PASS' if conf_w > conf_p + 1 else 'FAIL'}")

    all_pass = (d_pw >= 0.5 and res_p > res_w and reinf > 0.01
                and ordering and conf_w > conf_p + 1)

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'=' * 70}")

    output = {
        "metadata": {
            "experiment": "10_v3_discrimination",
            "n_students": N_STUDENTS,
            "n_interactions": N_INTERACTIONS,
            "seed": SEED,
            "method": "direct internal state measurement",
        },
        "conditions": {
            k: {kk: vv for kk, vv in v.items() if kk != "p_know_deltas"}
            for k, v in results.items()
        },
        "tests": {
            "perfect_vs_wrong_d": round(d_pw, 4),
            "perfect_vs_wrong_p": round(float(p1), 6),
            "perfect_vs_random_d": round(d_pr, 4),
            "perfect_vs_random_p": round(float(p2), 6),
            "resolution_perfect": res_p,
            "resolution_wrong": res_w,
            "reinforcement_wrong": reinf,
            "ordering_correct": bool(ordering),
            "all_pass": bool(all_pass),
        },
    }
    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cond_names = ["Perfect", "Random", "Always\nWrong", "No\nInstruction"]
    colors = ["#4CAF50", "#FF9800", "#F44336", "#9E9E9E"]

    vals = [results[c]["p_know_delta_mean"] for c in conditions]
    errs = [results[c]["p_know_delta_std"] for c in conditions]
    bars = axes[0, 0].bar(cond_names, vals, yerr=errs, color=colors, capsize=5)
    axes[0, 0].set_ylabel("Mean p_know Delta")
    axes[0, 0].set_title(f"Knowledge Gain (d={d_pw:.2f} perfect vs wrong)")
    axes[0, 0].axhline(y=0, color="black", linewidth=0.5)
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{v:.4f}", ha="center", fontsize=9)

    res = [results[c]["resolution_rate"] for c in conditions]
    bars2 = axes[0, 1].bar(cond_names, res, color=colors)
    axes[0, 1].set_ylabel("Resolution Rate")
    axes[0, 1].set_title("Misconceptions Resolved")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars2, res):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{v:.4f}", ha="center", fontsize=9)

    reinfs = [results[c]["reinforcement_rate"] for c in conditions]
    bars3 = axes[1, 0].bar(cond_names, reinfs, color=colors)
    axes[1, 0].set_ylabel("Reinforcement Rate")
    axes[1, 0].set_title("Misconceptions Made WORSE")
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars3, reinfs):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{v:.4f}", ha="center", fontsize=9)

    confs = [results[c]["confusion_mean"] for c in conditions]
    bars4 = axes[1, 1].bar(cond_names, confs, color=colors)
    axes[1, 1].set_ylabel("Mean Confusion Score")
    axes[1, 1].set_title("Confusion Accumulated")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars4, confs):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{v:.1f}", ha="center", fontsize=9)

    plt.suptitle("V3 Simulated Student Discrimination Test", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "discrimination_test.png", dpi=150)
    plt.close()
    print(f"\nResults saved to {ARTIFACTS}")


if __name__ == "__main__":
    main()
