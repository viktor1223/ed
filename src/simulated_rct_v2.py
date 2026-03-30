"""Simulated Classroom RCT v2: Decoupled Assessment.

Key changes from v1:
  1. Held-out test set: 2 problems per concept reserved for pre/post testing.
     Primary metric is proportion correct on the test, not internal p_know.
  2. Performance-based measurement: pre-test and post-test are independent of BKT.
     The student answers the same held-out problems before and after tutoring.
  3. Adaptive strategy with coverage floor: if a concept has received N consecutive
     interactions without mastery, advance to the next concept anyway.
  4. Tuned misconception resolution: targeted_resolution raised so resolution is
     achievable within 40 interactions.
  5. Diagnostic accuracy metric: when the student uses a misconception, how often
     does the tutor correctly identify it? Separate from mastery measurement.

Conditions:
  1. Adaptive: BKT-driven concept selection + targeted misconception remediation
  2. Random: Random concept, generic feedback
  3. Fixed sequence: Linear curriculum, generic feedback
  4. No remediation: BKT concept selection, no targeted misconception feedback

References:
  - Bloom (1984): The 2-sigma problem
  - VanLehn (2011): Step-based ITS d=0.76 matches human tutoring d=0.79
  - Kulik & Fletcher (2016): ITS median d=0.66
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_graph import KnowledgeGraph, StudentState, next_action
from simulated_student import (
    SimulatedStudent,
    generate_students,
    describe_population,
    load_problem_bank,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KG_PATH = PROJECT_ROOT / "data" / "knowledge_graph.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "simulated_rct_v2"

SEED = 42
N_STUDENTS_PER_CONDITION = 500
N_INTERACTIONS = 40

# Coverage floor: max consecutive interactions on one concept before advancing
MAX_CONSECUTIVE_SAME_CONCEPT = 6


# ─── Problem Bank Splitting ──────────────────────────────────────────────────

def split_problem_bank(
    problem_bank: dict[str, list[dict]],
    n_test_per_concept: int = 2,
    seed: int = SEED,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Split problem bank into practice and held-out test sets.

    Returns:
        (practice_bank, test_bank) - each keyed by concept ID.
    """
    rng = random.Random(seed)
    practice = {}
    test = {}

    for concept_id, problems in problem_bank.items():
        shuffled = list(problems)
        rng.shuffle(shuffled)

        n_test = min(n_test_per_concept, len(shuffled))
        test[concept_id] = shuffled[:n_test]
        practice[concept_id] = shuffled[n_test:]

    return practice, test


# ─── Held-Out Assessment ─────────────────────────────────────────────────────

def administer_test(
    student: SimulatedStudent,
    test_bank: dict[str, list[dict]],
) -> dict:
    """Give the student a fixed test without any learning/instruction.

    The student answers each test problem. No BKT update, no instruction.
    Returns per-concept and aggregate scores.

    Returns:
        dict with:
            per_concept: {concept_id: {n_correct, n_total, proportion}}
            aggregate: {n_correct, n_total, proportion}
            misconceptions_detected: list of misconception IDs triggered during test
    """
    per_concept = {}
    total_correct = 0
    total_problems = 0
    misconceptions_detected = []

    for concept_id in sorted(test_bank.keys()):
        problems = test_bank[concept_id]
        n_correct = 0
        for problem in problems:
            response = student.respond(problem)
            if response["correct"]:
                n_correct += 1
            elif response["misconception_used"]:
                misconceptions_detected.append(response["misconception_used"])
            total_problems += 1

        per_concept[concept_id] = {
            "n_correct": n_correct,
            "n_total": len(problems),
            "proportion": n_correct / len(problems) if problems else 0,
        }
        total_correct += n_correct

    return {
        "per_concept": per_concept,
        "aggregate": {
            "n_correct": total_correct,
            "n_total": total_problems,
            "proportion": total_correct / total_problems if total_problems > 0 else 0,
        },
        "misconceptions_detected": misconceptions_detected,
    }


# ─── Tutoring Strategies ─────────────────────────────────────────────────────

def adaptive_strategy_v2(
    student_state: StudentState,
    kg: KnowledgeGraph,
    consecutive_same: int,
    current_concept: str | None,
) -> str:
    """Adaptive strategy with coverage floor.

    If the same concept has been targeted MAX_CONSECUTIVE_SAME_CONCEPT times
    without mastery, advance to the next unmastered concept.
    """
    if consecutive_same >= MAX_CONSECUTIVE_SAME_CONCEPT and current_concept is not None:
        # Force advancement: find the next unmastered concept in level order
        concepts_by_level = kg.concepts_by_level()
        current_idx = next(
            (i for i, c in enumerate(concepts_by_level) if c.id == current_concept),
            0,
        )
        for offset in range(1, len(concepts_by_level)):
            candidate = concepts_by_level[(current_idx + offset) % len(concepts_by_level)]
            if not student_state.is_mastered(candidate.id):
                return candidate.id
        # All mastered: review the weakest
        return min(student_state.mastery, key=student_state.mastery.get)

    action = next_action(student_state, kg)
    return action["concept"]


def random_strategy(student_state: StudentState, kg: KnowledgeGraph) -> str:
    return random.choice(list(kg.concepts.keys()))


def fixed_strategy(student_state: StudentState, kg: KnowledgeGraph, interaction: int) -> str:
    concepts = [c.id for c in kg.concepts_by_level()]
    return concepts[interaction % len(concepts)]


def no_remediation_strategy(student_state: StudentState, kg: KnowledgeGraph) -> str:
    action = next_action(student_state, kg)
    return action["concept"]


# ─── Simulation Engine ───────────────────────────────────────────────────────

def run_student_session(
    student: SimulatedStudent,
    strategy: str,
    kg: KnowledgeGraph,
    practice_bank: dict[str, list[dict]],
    test_bank: dict[str, list[dict]],
    n_interactions: int = N_INTERACTIONS,
) -> dict:
    """Run a single student through pre-test, tutoring, and post-test.

    Returns metrics based on held-out test performance (not internal p_know).
    """
    # ── Pre-test (no learning) ──
    pre_test = administer_test(student, test_bank)

    # Record internal state for secondary analysis only
    pre_p_know = dict(student.p_know)
    pre_misconceptions = set(student.active_misconceptions())

    # ── Tutoring phase ──
    tutor_state = StudentState(kg)
    history = []
    consecutive_same_concept = 0
    last_concept = None

    # Diagnostic accuracy tracking
    true_positives = 0   # student used misconception AND tutor targeted it
    false_negatives = 0  # student used misconception BUT tutor didn't detect
    tutor_targeted = 0   # total times tutor targeted a misconception
    concept_interaction_counts = defaultdict(int)

    for i in range(n_interactions):
        # Select concept
        if strategy == "adaptive":
            concept_id = adaptive_strategy_v2(
                tutor_state, kg, consecutive_same_concept, last_concept,
            )
        elif strategy == "random":
            concept_id = random_strategy(tutor_state, kg)
        elif strategy == "fixed_sequence":
            concept_id = fixed_strategy(tutor_state, kg, i)
        elif strategy == "no_remediation":
            concept_id = no_remediation_strategy(tutor_state, kg)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Track consecutive same concept (for adaptive coverage floor)
        if concept_id == last_concept:
            consecutive_same_concept += 1
        else:
            consecutive_same_concept = 0
        last_concept = concept_id
        concept_interaction_counts[concept_id] += 1

        # Pick a practice problem
        problems = practice_bank.get(concept_id, [])
        if not problems:
            continue
        problem = random.choice(problems)

        # Student responds
        response = student.respond(problem)

        # Tutor updates belief
        tutor_state.update(
            concept_id,
            correct=response["correct"],
            confidence=0.8 if not response["correct"] else 1.0,
        )

        # Provide instruction + track diagnostic accuracy
        student_used_misconception = response["misconception_used"]

        if strategy == "no_remediation":
            student.receive_instruction(concept_id, targeted_misconception=None)
            if student_used_misconception:
                false_negatives += 1
        elif strategy == "adaptive":
            if not response["correct"] and student_used_misconception:
                # Tutor attempts targeted remediation
                student.receive_instruction(
                    concept_id,
                    targeted_misconception=student_used_misconception,
                )
                true_positives += 1
                tutor_targeted += 1
            else:
                student.receive_instruction(concept_id, targeted_misconception=None)
                if student_used_misconception:
                    false_negatives += 1
        else:
            # Random and fixed: generic feedback
            student.receive_instruction(concept_id, targeted_misconception=None)
            if student_used_misconception:
                false_negatives += 1

        history.append({
            "interaction": i,
            "concept": concept_id,
            "correct": response["correct"],
            "misconception_used": student_used_misconception,
        })

    # ── Post-test (no learning) ──
    post_test = administer_test(student, test_bank)

    # Record post internal state for secondary analysis
    post_p_know = dict(student.p_know)
    post_misconceptions = set(student.active_misconceptions())
    misconceptions_resolved = pre_misconceptions - post_misconceptions

    # ── Compute metrics ──
    concept_list = [c.id for c in kg.concepts_by_level()]

    # Primary metric: held-out test score change
    test_score_pre = pre_test["aggregate"]["proportion"]
    test_score_post = post_test["aggregate"]["proportion"]
    test_score_gain = test_score_post - test_score_pre

    # Per-concept test gains
    per_concept_test_gain = {}
    for c in concept_list:
        pre_c = pre_test["per_concept"].get(c, {}).get("proportion", 0)
        post_c = post_test["per_concept"].get(c, {}).get("proportion", 0)
        per_concept_test_gain[c] = post_c - pre_c

    # Concept coverage: how many distinct concepts did the tutor interact with?
    concepts_touched = len(concept_interaction_counts)

    # Diagnostic accuracy
    total_misconception_events = true_positives + false_negatives
    diagnostic_sensitivity = (
        true_positives / total_misconception_events
        if total_misconception_events > 0 else 0
    )

    return {
        "student_id": student.student_id,
        "strategy": strategy,
        # Primary metrics (held-out test based)
        "test_score_pre": test_score_pre,
        "test_score_post": test_score_post,
        "test_score_gain": test_score_gain,
        "per_concept_test_gain": per_concept_test_gain,
        # Secondary metrics (internal model, for analysis only)
        "internal_mastery_gain": float(np.mean([
            post_p_know.get(c, 0) - pre_p_know.get(c, 0) for c in concept_list
        ])),
        "n_misconceptions_resolved": len(misconceptions_resolved),
        "n_misconceptions_initial": len(pre_misconceptions),
        # Coverage metrics
        "concepts_touched": concepts_touched,
        "concept_interaction_distribution": dict(concept_interaction_counts),
        # Diagnostic accuracy
        "diagnostic_sensitivity": diagnostic_sensitivity,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "tutor_targeted_count": tutor_targeted,
        # Raw test results (for detailed analysis)
        "pre_test": pre_test,
        "post_test": post_test,
    }


# ─── RCT Runner ──────────────────────────────────────────────────────────────

def run_rct(
    n_per_condition: int = N_STUDENTS_PER_CONDITION,
    n_interactions: int = N_INTERACTIONS,
    seed: int = SEED,
    bkt_param_scale: float = 1.0,
    targeted_resolution: float = 0.50,
) -> dict:
    """Run the full simulated classroom RCT with decoupled assessment.

    Args:
        n_per_condition: Students per condition.
        n_interactions: Tutoring interactions per student.
        seed: Random seed.
        bkt_param_scale: Multiplier for BKT learning rates (sensitivity analysis).
        targeted_resolution: Misconception resolution probability per targeted
            intervention. Default 0.50 (raised from v1's 0.30).
    """
    random.seed(seed)
    np.random.seed(seed)

    kg = KnowledgeGraph.from_json(KG_PATH)
    full_bank = load_problem_bank()

    # Split problem bank: 2 per concept for test, rest for practice
    practice_bank, test_bank = split_problem_bank(full_bank, n_test_per_concept=2, seed=seed)

    n_practice = sum(len(v) for v in practice_bank.values())
    n_test = sum(len(v) for v in test_bank.values())

    strategies = ["adaptive", "random", "fixed_sequence", "no_remediation"]
    condition_results: dict[str, list[dict]] = {s: [] for s in strategies}

    print(f"\n{'=' * 70}")
    print(f"  SIMULATED CLASSROOM RCT v2: Decoupled Assessment")
    print(f"  N={n_per_condition} per condition, {n_interactions} interactions each")
    print(f"  Practice problems: {n_practice}, Test problems: {n_test}")
    print(f"  BKT param scale: {bkt_param_scale:.2f}")
    print(f"  Targeted resolution: {targeted_resolution:.2f}")
    print(f"{'=' * 70}")

    for strategy in strategies:
        print(f"\n--- Running condition: {strategy} ---")

        students = generate_students(n=n_per_condition, kg_path=KG_PATH, seed=seed)

        # Override misconception resolution parameters
        for s in students:
            s.targeted_resolution = targeted_resolution
            # Keep generic_resolution unchanged at 0.05

        # Apply BKT parameter scaling
        if bkt_param_scale != 1.0:
            for s in students:
                for cid in s.bkt_params:
                    if "p_learn" in s.bkt_params[cid]:
                        s.bkt_params[cid]["p_learn"] = min(
                            0.95, s.bkt_params[cid]["p_learn"] * bkt_param_scale
                        )

        for j, student in enumerate(students):
            result = run_student_session(
                student=student,
                strategy=strategy,
                kg=kg,
                practice_bank=practice_bank,
                test_bank=test_bank,
                n_interactions=n_interactions,
            )
            condition_results[strategy].append(result)

            if (j + 1) % 100 == 0:
                print(f"  Completed {j + 1}/{n_per_condition} students")

    return analyze_results(condition_results, strategies)


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze_results(
    condition_results: dict[str, list[dict]],
    strategies: list[str],
) -> dict:
    """Compute statistics using held-out test scores as the primary metric."""
    print(f"\n{'=' * 70}")
    print("  ANALYSIS (Primary metric: held-out test score)")
    print(f"{'=' * 70}")

    analysis = {"conditions": {}, "comparisons": {}, "effect_sizes": {}}

    for strategy in strategies:
        results = condition_results[strategy]
        test_gains = [r["test_score_gain"] for r in results]
        pre_scores = [r["test_score_pre"] for r in results]
        post_scores = [r["test_score_post"] for r in results]
        internal_gains = [r["internal_mastery_gain"] for r in results]
        resolved = [r["n_misconceptions_resolved"] for r in results]
        sensitivities = [r["diagnostic_sensitivity"] for r in results]
        concepts_touched = [r["concepts_touched"] for r in results]

        resolution_rate = float(np.mean([
            r["n_misconceptions_resolved"] / max(r["n_misconceptions_initial"], 1)
            for r in results
        ]))

        analysis["conditions"][strategy] = {
            "n": len(results),
            # Primary: held-out test
            "mean_test_pre": float(np.mean(pre_scores)),
            "mean_test_post": float(np.mean(post_scores)),
            "mean_test_gain": float(np.mean(test_gains)),
            "std_test_gain": float(np.std(test_gains)),
            "median_test_gain": float(np.median(test_gains)),
            # Secondary: internal model
            "mean_internal_gain": float(np.mean(internal_gains)),
            "mean_misconceptions_resolved": float(np.mean(resolved)),
            "resolution_rate": resolution_rate,
            # Coverage
            "mean_concepts_touched": float(np.mean(concepts_touched)),
            # Diagnostic accuracy (only meaningful for adaptive)
            "mean_diagnostic_sensitivity": float(np.mean(sensitivities)),
        }

        print(f"\n  {strategy}:")
        print(f"    Test score (pre):          {np.mean(pre_scores):.3f}")
        print(f"    Test score (post):         {np.mean(post_scores):.3f}")
        print(f"    Test score gain:           {np.mean(test_gains):+.3f} (SD={np.std(test_gains):.3f})")
        print(f"    Internal mastery gain:     {np.mean(internal_gains):+.4f} (secondary)")
        print(f"    Misconception resolution:  {resolution_rate:.1%}")
        print(f"    Concepts touched:          {np.mean(concepts_touched):.1f}/5")
        if strategy == "adaptive":
            print(f"    Diagnostic sensitivity:    {np.mean(sensitivities):.1%}")

    # Pairwise comparisons on test score gain
    adaptive_gains = [r["test_score_gain"] for r in condition_results["adaptive"]]

    print(f"\n--- Pairwise Comparisons (held-out test score gain) ---")

    for baseline in ["random", "fixed_sequence", "no_remediation"]:
        baseline_gains = [r["test_score_gain"] for r in condition_results[baseline]]

        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(adaptive_gains, baseline_gains, equal_var=False)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(adaptive_gains) - 1) * np.std(adaptive_gains) ** 2 +
             (len(baseline_gains) - 1) * np.std(baseline_gains) ** 2) /
            (len(adaptive_gains) + len(baseline_gains) - 2)
        )
        cohens_d = float(
            (np.mean(adaptive_gains) - np.mean(baseline_gains)) / pooled_std
            if pooled_std > 0 else 0
        )

        # Bootstrap 95% CI for Cohen's d
        boot_ds = []
        rng = np.random.default_rng(42)
        for _ in range(2000):
            boot_a = rng.choice(adaptive_gains, size=len(adaptive_gains), replace=True)
            boot_b = rng.choice(baseline_gains, size=len(baseline_gains), replace=True)
            boot_pooled = np.sqrt(
                ((len(boot_a) - 1) * np.std(boot_a) ** 2 +
                 (len(boot_b) - 1) * np.std(boot_b) ** 2) /
                (len(boot_a) + len(boot_b) - 2)
            )
            if boot_pooled > 0:
                boot_ds.append(float((np.mean(boot_a) - np.mean(boot_b)) / boot_pooled))
        ci_lower = float(np.percentile(boot_ds, 2.5)) if boot_ds else 0
        ci_upper = float(np.percentile(boot_ds, 97.5)) if boot_ds else 0

        # Bonferroni correction (3 comparisons)
        p_adjusted = min(float(p_value) * 3, 1.0)

        analysis["comparisons"][f"adaptive_vs_{baseline}"] = {
            "test_score_gain": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "p_adjusted_bonferroni": p_adjusted,
                "cohens_d": cohens_d,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "significant_at_05": p_adjusted < 0.05,
            },
        }

        sig = "***" if p_adjusted < 0.001 else ("**" if p_adjusted < 0.01 else ("*" if p_adjusted < 0.05 else "ns"))
        print(f"\n  Adaptive vs {baseline}:")
        print(f"    Cohen's d:           {cohens_d:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"    p-value (adjusted):  {p_adjusted:.4f} {sig}")

    # Effect size summary
    print(f"\n--- Effect Size Summary (held-out test) ---")
    print(f"  {'Comparison':<35s} {'Cohen d':>10s} {'95% CI':>20s} {'p (adj)':>10s}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 20} {'-' * 10}")
    for key, comp in analysis["comparisons"].items():
        mg = comp["test_score_gain"]
        ci = f"[{mg['ci_95_lower']:.3f}, {mg['ci_95_upper']:.3f}]"
        print(f"  {key:<35s} {mg['cohens_d']:>10.3f} {ci:>20s} {mg['p_adjusted_bonferroni']:>10.4f}")

    analysis["effect_sizes"] = {
        key: comp["test_score_gain"]["cohens_d"]
        for key, comp in analysis["comparisons"].items()
    }

    return analysis


# ─── Sensitivity Analysis ────────────────────────────────────────────────────

def run_sensitivity_analysis(
    n_per_condition: int = 200,
    n_interactions: int = N_INTERACTIONS,
    seed: int = SEED,
) -> dict:
    """Run with varied BKT parameters and resolution rates."""
    print(f"\n{'=' * 70}")
    print("  SENSITIVITY ANALYSIS")
    print(f"{'=' * 70}")

    # Vary BKT learning rate
    bkt_scales = [0.5, 1.0, 1.5, 2.0]
    # Vary misconception resolution
    resolution_rates = [0.30, 0.50, 0.70]

    sensitivity_results = {}

    for scale in bkt_scales:
        for res_rate in resolution_rates:
            key = f"bkt_{scale:.1f}_res_{res_rate:.2f}"
            print(f"\n  Running: BKT scale={scale:.1f}, resolution={res_rate:.2f}")
            result = run_rct(
                n_per_condition=n_per_condition,
                n_interactions=n_interactions,
                seed=seed,
                bkt_param_scale=scale,
                targeted_resolution=res_rate,
            )
            sensitivity_results[key] = {
                "bkt_scale": scale,
                "targeted_resolution": res_rate,
                "effect_sizes": result["effect_sizes"],
                "conditions": {
                    k: {
                        "mean_test_gain": v["mean_test_gain"],
                        "resolution_rate": v["resolution_rate"],
                        "mean_concepts_touched": v["mean_concepts_touched"],
                    }
                    for k, v in result["conditions"].items()
                },
            }

    # Robustness check
    print(f"\n--- Robustness Check ---")
    adaptive_always_best = True
    for key, sr in sensitivity_results.items():
        adaptive_gain = sr["conditions"]["adaptive"]["mean_test_gain"]
        for baseline in ["random", "fixed_sequence", "no_remediation"]:
            baseline_gain = sr["conditions"][baseline]["mean_test_gain"]
            if adaptive_gain <= baseline_gain:
                adaptive_always_best = False
                print(f"  WARNING: At {key}, {baseline} ({baseline_gain:.4f}) >= adaptive ({adaptive_gain:.4f})")

    if adaptive_always_best:
        print("  PASS: Adaptive outperforms all baselines across all parameter combinations")

    sensitivity_results["robust"] = adaptive_always_best
    return sensitivity_results


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_interpretation(rct_results: dict, sensitivity: dict) -> dict:
    """Generate interpretation grounded in held-out test performance."""
    summary_lines = []

    adaptive = rct_results["conditions"]["adaptive"]
    summary_lines.append(f"METRIC: Held-out test score (independent of BKT)")
    summary_lines.append(f"")
    summary_lines.append(f"Adaptive condition:")
    summary_lines.append(f"  Pre-test:  {adaptive['mean_test_pre']:.3f}")
    summary_lines.append(f"  Post-test: {adaptive['mean_test_post']:.3f}")
    summary_lines.append(f"  Gain:      {adaptive['mean_test_gain']:+.3f}")
    summary_lines.append(f"  Misconception resolution: {adaptive['resolution_rate']:.1%}")
    summary_lines.append(f"  Concepts touched: {adaptive['mean_concepts_touched']:.1f}/5")
    summary_lines.append(f"  Diagnostic sensitivity: {adaptive['mean_diagnostic_sensitivity']:.1%}")

    best_d = max(rct_results["effect_sizes"].values())
    best_comparison = max(rct_results["effect_sizes"], key=rct_results["effect_sizes"].get)
    summary_lines.append(f"")
    summary_lines.append(f"Largest effect size: d = {best_d:.3f} ({best_comparison})")

    if best_d >= 0.8:
        summary_lines.append("Matches or exceeds best ITS systems (VanLehn 2011: d=0.76)")
    elif best_d >= 0.5:
        summary_lines.append("Medium-to-large effect, competitive with established ITS (Kulik & Fletcher 2016: d=0.66)")
    elif best_d >= 0.3:
        summary_lines.append("Small-to-medium effect, typical of answer-based systems (VanLehn 2011: d=0.31)")
    else:
        summary_lines.append("Small effect; further investigation needed")

    summary_lines.append("")
    if sensitivity.get("robust"):
        summary_lines.append("ROBUSTNESS: Adaptive outperforms all baselines across all parameter variations")
    else:
        summary_lines.append("WARNING: Results NOT fully robust across parameter variation")

    summary_lines.append("")
    summary_lines.append("METHODOLOGY NOTES:")
    summary_lines.append("  - Primary metric: held-out test problems (decoupled from BKT)")
    summary_lines.append("  - Pre/post test on fixed problem set, not internal model state")
    summary_lines.append("  - Adaptive strategy includes coverage floor (max 6 consecutive on one concept)")
    summary_lines.append("  - These remain simulated results validating system behavior")
    summary_lines.append("  - A human RCT (n>=100 per group) is required for effect size claims")

    return {
        "summary": summary_lines,
        "primary_metric": "held-out test score gain",
        "bloom_2sigma_achievable": False,
        "realistic_target": "d = 0.5-0.8 on aligned assessments",
        "evidence_level": "Stage 2: Simulated evaluation with decoupled assessment",
        "next_step": "Stage 3: Wizard-of-Oz pilot with 5-10 real students",
        "methodology_improvements_over_v1": [
            "Test score measured on held-out problems, not internal BKT state",
            "Coverage floor prevents adaptive strategy from fixating on one concept",
            "Misconception resolution rate tuned to be achievable",
            "Diagnostic sensitivity tracked as independent metric",
            "Sensitivity analysis varies both BKT params and resolution rates",
        ],
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Describe population
    print("Generating student population...")
    students = generate_students(n=100, seed=SEED)
    pop_desc = describe_population(students)
    print(f"  Students: {pop_desc['n_students']}")
    print(f"  Avg active misconceptions: {pop_desc['avg_active_misconceptions']}")
    print(f"  Avg p_know: {pop_desc['avg_p_know']}")

    # Run main RCT
    rct_results = run_rct(
        n_per_condition=N_STUDENTS_PER_CONDITION,
        n_interactions=N_INTERACTIONS,
        seed=SEED,
        targeted_resolution=0.50,
    )

    # Run sensitivity analysis (smaller N)
    sensitivity = run_sensitivity_analysis(
        n_per_condition=200,
        n_interactions=N_INTERACTIONS,
        seed=SEED,
    )

    # Compile results
    full_results = {
        "metadata": {
            "version": "v2",
            "n_per_condition": N_STUDENTS_PER_CONDITION,
            "n_interactions": N_INTERACTIONS,
            "seed": SEED,
            "strategies": ["adaptive", "random", "fixed_sequence", "no_remediation"],
            "framework": "Tier 2: Decoupled Assessment with Held-Out Test Set",
            "primary_metric": "held_out_test_score_gain",
            "targeted_resolution": 0.50,
            "max_consecutive_same_concept": MAX_CONSECUTIVE_SAME_CONCEPT,
        },
        "population": pop_desc,
        "rct": rct_results,
        "sensitivity": sensitivity,
        "interpretation": generate_interpretation(rct_results, sensitivity),
    }

    output_path = RESULTS_DIR / "simulated_rct_v2_results.json"
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS SAVED to {output_path}")
    print(f"{'=' * 70}")

    interp = full_results["interpretation"]
    print(f"\n{'=' * 70}")
    print("  INTERPRETATION")
    print(f"{'=' * 70}")
    for line in interp["summary"]:
        print(f"  {line}")

    return full_results


if __name__ == "__main__":
    main()
