"""Simulated Classroom Randomized Controlled Trial.

Runs N simulated students through K interactions under multiple conditions:
  1. Adaptive: Full system (BKT + misconception detection + targeted remediation)
  2. Random: Random concept selection, generic feedback
  3. Fixed sequence: Linear curriculum, generic feedback
  4. No remediation: Adaptive concept selection but no misconception-targeted hints

Measures:
  - Pre/post mastery gains (Cohen's d between conditions)
  - Misconception resolution rate
  - Interactions to mastery per concept
  - Weak-concept targeting accuracy

Includes sensitivity analysis for BKT parameter variation.

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
RESULTS_DIR = PROJECT_ROOT / "results" / "simulated_rct"

SEED = 42
N_STUDENTS_PER_CONDITION = 500
N_INTERACTIONS = 40


# ─── Tutoring Strategies ─────────────────────────────────────────────────────

def adaptive_strategy(student_state: StudentState, kg: KnowledgeGraph) -> str:
    """Full adaptive: BKT-driven concept selection."""
    action = next_action(student_state, kg)
    return action["concept"]


def random_strategy(student_state: StudentState, kg: KnowledgeGraph) -> str:
    """Random concept selection."""
    return random.choice(list(kg.concepts.keys()))


def fixed_strategy(student_state: StudentState, kg: KnowledgeGraph, interaction: int) -> str:
    """Fixed linear sequence through concepts by level."""
    concepts = [c.id for c in kg.concepts_by_level()]
    return concepts[interaction % len(concepts)]


def no_remediation_strategy(student_state: StudentState, kg: KnowledgeGraph) -> str:
    """Adaptive concept selection but no targeted remediation."""
    action = next_action(student_state, kg)
    return action["concept"]


# ─── Simulation Engine ───────────────────────────────────────────────────────

def run_student_session(
    student: SimulatedStudent,
    strategy: str,
    kg: KnowledgeGraph,
    problem_bank: dict[str, list[dict]],
    n_interactions: int = N_INTERACTIONS,
) -> dict:
    """Run a single student through n_interactions of tutoring.

    Returns:
        dict with pre_mastery, post_mastery, history, misconceptions_resolved, etc.
    """
    # Record pre-mastery
    pre_mastery = dict(student.p_know)
    pre_misconceptions = set(student.active_misconceptions())

    # Create a tutor-side student state tracker (what the tutor believes)
    tutor_state = StudentState(kg)

    history = []
    concept_list = [c.id for c in kg.concepts_by_level()]

    for i in range(n_interactions):
        # Select concept based on strategy
        if strategy == "adaptive":
            concept_id = adaptive_strategy(tutor_state, kg)
        elif strategy == "random":
            concept_id = random_strategy(tutor_state, kg)
        elif strategy == "fixed_sequence":
            concept_id = fixed_strategy(tutor_state, kg, i)
        elif strategy == "no_remediation":
            concept_id = no_remediation_strategy(tutor_state, kg)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Pick a problem for this concept
        problems = problem_bank.get(concept_id, [])
        if not problems:
            continue
        problem = random.choice(problems)

        # Student responds
        response = student.respond(problem)

        # Tutor updates its belief about the student
        tutor_state.update(
            concept_id,
            correct=response["correct"],
            confidence=0.8 if not response["correct"] else 1.0,
        )

        # Provide instruction to the student
        if strategy == "no_remediation":
            # Generic feedback only
            student.receive_instruction(concept_id, targeted_misconception=None)
        elif strategy in ("adaptive",):
            # Targeted remediation: tutor identifies misconception
            if not response["correct"] and response["misconception_used"]:
                student.receive_instruction(
                    concept_id,
                    targeted_misconception=response["misconception_used"],
                )
            else:
                student.receive_instruction(concept_id, targeted_misconception=None)
        else:
            # Random and fixed: generic feedback
            student.receive_instruction(concept_id, targeted_misconception=None)

        history.append({
            "interaction": i,
            "concept": concept_id,
            "correct": response["correct"],
            "misconception_used": response["misconception_used"],
            "student_p_know": dict(student.p_know),
        })

    # Record post-mastery
    post_mastery = dict(student.p_know)
    post_misconceptions = set(student.active_misconceptions())
    misconceptions_resolved = pre_misconceptions - post_misconceptions

    # Compute metrics
    mastery_gain = {
        c: post_mastery.get(c, 0) - pre_mastery.get(c, 0) for c in concept_list
    }
    mean_mastery_gain = float(np.mean(list(mastery_gain.values())))

    # Count interactions per concept until mastery
    interactions_to_mastery: dict[str, int | None] = {}
    for c in concept_list:
        found = None
        for h in history:
            if h["concept"] == c and h["student_p_know"].get(c, 0) >= student.mastery_threshold:
                found = h["interaction"]
                break
        interactions_to_mastery[c] = found

    # Correct rate
    n_correct = sum(1 for h in history if h["correct"])

    return {
        "student_id": student.student_id,
        "strategy": strategy,
        "pre_mastery": pre_mastery,
        "post_mastery": post_mastery,
        "mastery_gain": mastery_gain,
        "mean_mastery_gain": mean_mastery_gain,
        "n_correct": n_correct,
        "n_interactions": len(history),
        "misconceptions_resolved": list(misconceptions_resolved),
        "n_misconceptions_resolved": len(misconceptions_resolved),
        "n_misconceptions_initial": len(pre_misconceptions),
        "interactions_to_mastery": interactions_to_mastery,
        "concepts_mastered_post": sum(
            1 for c in concept_list if post_mastery.get(c, 0) >= student.mastery_threshold
        ),
    }


# ─── RCT Runner ──────────────────────────────────────────────────────────────

def run_rct(
    n_per_condition: int = N_STUDENTS_PER_CONDITION,
    n_interactions: int = N_INTERACTIONS,
    seed: int = SEED,
    bkt_param_scale: float = 1.0,
) -> dict:
    """Run the full simulated classroom RCT.

    Args:
        n_per_condition: Number of students per condition.
        n_interactions: Interactions per student.
        seed: Random seed.
        bkt_param_scale: Multiplier for BKT learning rates (for sensitivity analysis).

    Returns:
        dict with per-condition results + statistical comparisons.
    """
    random.seed(seed)
    np.random.seed(seed)

    kg = KnowledgeGraph.from_json(KG_PATH)
    problem_bank = load_problem_bank()

    strategies = ["adaptive", "random", "fixed_sequence", "no_remediation"]
    condition_results: dict[str, list[dict]] = {s: [] for s in strategies}

    print(f"\n{'=' * 70}")
    print(f"  SIMULATED CLASSROOM RCT")
    print(f"  N={n_per_condition} per condition, {n_interactions} interactions each")
    print(f"  BKT param scale: {bkt_param_scale:.2f}")
    print(f"{'=' * 70}")

    for strategy in strategies:
        print(f"\n--- Running condition: {strategy} ---")

        # Generate a fresh set of students for each condition (same seed = same population)
        students = generate_students(
            n=n_per_condition,
            kg_path=KG_PATH,
            seed=seed,
        )

        # Apply BKT parameter scaling (for sensitivity analysis)
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
                problem_bank=problem_bank,
                n_interactions=n_interactions,
            )
            condition_results[strategy].append(result)

            if (j + 1) % 100 == 0:
                print(f"  Completed {j + 1}/{n_per_condition} students")

    return analyze_results(condition_results, strategies)


def analyze_results(
    condition_results: dict[str, list[dict]],
    strategies: list[str],
) -> dict:
    """Compute statistics and comparisons across conditions."""
    print(f"\n{'=' * 70}")
    print("  ANALYSIS")
    print(f"{'=' * 70}")

    analysis = {"conditions": {}, "comparisons": {}, "effect_sizes": {}}

    # Per-condition summaries
    for strategy in strategies:
        results = condition_results[strategy]
        gains = [r["mean_mastery_gain"] for r in results]
        resolved = [r["n_misconceptions_resolved"] for r in results]
        mastered = [r["concepts_mastered_post"] for r in results]
        correct = [r["n_correct"] for r in results]

        analysis["conditions"][strategy] = {
            "n": len(results),
            "mean_mastery_gain": float(np.mean(gains)),
            "std_mastery_gain": float(np.std(gains)),
            "median_mastery_gain": float(np.median(gains)),
            "mean_misconceptions_resolved": float(np.mean(resolved)),
            "std_misconceptions_resolved": float(np.std(resolved)),
            "mean_concepts_mastered": float(np.mean(mastered)),
            "mean_correct": float(np.mean(correct)),
            "resolution_rate": float(
                np.mean([
                    r["n_misconceptions_resolved"] / max(r["n_misconceptions_initial"], 1)
                    for r in results
                ])
            ),
        }

        print(f"\n  {strategy}:")
        print(f"    Mean mastery gain:        {np.mean(gains):.4f} (SD={np.std(gains):.4f})")
        print(f"    Misconceptions resolved:  {np.mean(resolved):.2f} / {np.mean([r['n_misconceptions_initial'] for r in results]):.1f}")
        print(f"    Resolution rate:          {analysis['conditions'][strategy]['resolution_rate']:.1%}")
        print(f"    Concepts mastered (post):  {np.mean(mastered):.2f}/5")
        print(f"    Correct responses:        {np.mean(correct):.1f}/{results[0]['n_interactions']}")

    # Pairwise comparisons: adaptive vs each baseline
    adaptive_gains = [r["mean_mastery_gain"] for r in condition_results["adaptive"]]
    adaptive_resolved = [r["n_misconceptions_resolved"] for r in condition_results["adaptive"]]

    print(f"\n--- Pairwise Comparisons (Adaptive vs. Baseline) ---")

    for baseline in ["random", "fixed_sequence", "no_remediation"]:
        baseline_gains = [r["mean_mastery_gain"] for r in condition_results[baseline]]
        baseline_resolved = [r["n_misconceptions_resolved"] for r in condition_results[baseline]]

        # Welch's t-test on mastery gains
        t_stat, p_value = stats.ttest_ind(adaptive_gains, baseline_gains, equal_var=False)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(adaptive_gains) - 1) * np.std(adaptive_gains) ** 2 +
             (len(baseline_gains) - 1) * np.std(baseline_gains) ** 2) /
            (len(adaptive_gains) + len(baseline_gains) - 2)
        )
        cohens_d = (np.mean(adaptive_gains) - np.mean(baseline_gains)) / pooled_std if pooled_std > 0 else 0

        # Bootstrap 95% CI for Cohen's d
        boot_ds = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            boot_a = rng.choice(adaptive_gains, size=len(adaptive_gains), replace=True)
            boot_b = rng.choice(baseline_gains, size=len(baseline_gains), replace=True)
            boot_pooled = np.sqrt(
                ((len(boot_a) - 1) * np.std(boot_a) ** 2 +
                 (len(boot_b) - 1) * np.std(boot_b) ** 2) /
                (len(boot_a) + len(boot_b) - 2)
            )
            if boot_pooled > 0:
                boot_ds.append((np.mean(boot_a) - np.mean(boot_b)) / boot_pooled)
        ci_lower = float(np.percentile(boot_ds, 2.5)) if boot_ds else 0
        ci_upper = float(np.percentile(boot_ds, 97.5)) if boot_ds else 0

        # Misconception resolution comparison
        t_misc, p_misc = stats.ttest_ind(adaptive_resolved, baseline_resolved, equal_var=False)
        resolution_ratio = (
            np.mean(adaptive_resolved) / np.mean(baseline_resolved)
            if np.mean(baseline_resolved) > 0 else float("inf")
        )

        # Bonferroni correction (3 comparisons)
        p_adjusted = min(p_value * 3, 1.0)

        analysis["comparisons"][f"adaptive_vs_{baseline}"] = {
            "mastery_gain": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "p_adjusted_bonferroni": float(p_adjusted),
                "cohens_d": float(cohens_d),
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "significant_at_05": p_adjusted < 0.05,
            },
            "misconception_resolution": {
                "t_statistic": float(t_misc),
                "p_value": float(p_misc),
                "resolution_ratio": float(resolution_ratio),
            },
        }

        sig = "***" if p_adjusted < 0.001 else ("**" if p_adjusted < 0.01 else ("*" if p_adjusted < 0.05 else "ns"))
        print(f"\n  Adaptive vs {baseline}:")
        print(f"    Cohen's d:           {cohens_d:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"    p-value (adjusted):  {p_adjusted:.4f} {sig}")
        print(f"    Resolution ratio:    {resolution_ratio:.2f}x")

    # Effect size summary table
    print(f"\n--- Effect Size Summary ---")
    print(f"  {'Comparison':<35s} {'Cohen d':>10s} {'95% CI':>20s} {'p (adj)':>10s}")
    print(f"  {'-'*35} {'-'*10} {'-'*20} {'-'*10}")
    for key, comp in analysis["comparisons"].items():
        mg = comp["mastery_gain"]
        ci = f"[{mg['ci_95_lower']:.3f}, {mg['ci_95_upper']:.3f}]"
        print(f"  {key:<35s} {mg['cohens_d']:>10.3f} {ci:>20s} {mg['p_adjusted_bonferroni']:>10.4f}")

    analysis["effect_sizes"] = {
        key: comp["mastery_gain"]["cohens_d"]
        for key, comp in analysis["comparisons"].items()
    }

    return analysis


# ─── Sensitivity Analysis ────────────────────────────────────────────────────

def run_sensitivity_analysis(
    n_per_condition: int = 200,
    n_interactions: int = N_INTERACTIONS,
    seed: int = SEED,
) -> dict:
    """Run the RCT with varied BKT parameters to check robustness."""
    print(f"\n{'=' * 70}")
    print("  SENSITIVITY ANALYSIS: BKT Parameter Variation")
    print(f"{'=' * 70}")

    scales = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    sensitivity_results = {}

    for scale in scales:
        print(f"\n  Scale = {scale:.1f}x")
        result = run_rct(
            n_per_condition=n_per_condition,
            n_interactions=n_interactions,
            seed=seed,
            bkt_param_scale=scale,
        )
        sensitivity_results[str(scale)] = {
            "effect_sizes": result["effect_sizes"],
            "conditions": {
                k: {"mean_mastery_gain": v["mean_mastery_gain"], "resolution_rate": v["resolution_rate"]}
                for k, v in result["conditions"].items()
            },
        }

    # Check robustness: does adaptive always win?
    print(f"\n--- Robustness Check ---")
    adaptive_always_best = True
    for scale_str, sr in sensitivity_results.items():
        adaptive_gain = sr["conditions"]["adaptive"]["mean_mastery_gain"]
        for baseline in ["random", "fixed_sequence", "no_remediation"]:
            baseline_gain = sr["conditions"][baseline]["mean_mastery_gain"]
            if adaptive_gain <= baseline_gain:
                adaptive_always_best = False
                print(f"  WARNING: At scale {scale_str}, {baseline} >= adaptive on mastery gain")

    if adaptive_always_best:
        print("  PASS: Adaptive outperforms all baselines at every parameter scale")

    sensitivity_results["robust"] = adaptive_always_best
    return sensitivity_results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Describe the student population
    print("Generating student population...")
    students = generate_students(n=100, seed=SEED)
    pop_desc = describe_population(students)
    print(f"  Students: {pop_desc['n_students']}")
    print(f"  Avg active misconceptions: {pop_desc['avg_active_misconceptions']}")
    print(f"  Avg p_know: {pop_desc['avg_p_know']}")
    print(f"  Top misconceptions: {dict(list(pop_desc['misconception_prevalence'].items())[:5])}")

    # Run main RCT
    rct_results = run_rct(
        n_per_condition=N_STUDENTS_PER_CONDITION,
        n_interactions=N_INTERACTIONS,
        seed=SEED,
    )

    # Run sensitivity analysis (smaller N for speed)
    sensitivity = run_sensitivity_analysis(
        n_per_condition=200,
        n_interactions=N_INTERACTIONS,
        seed=SEED,
    )

    # Compile final report
    full_results = {
        "metadata": {
            "n_per_condition": N_STUDENTS_PER_CONDITION,
            "n_interactions": N_INTERACTIONS,
            "seed": SEED,
            "strategies": ["adaptive", "random", "fixed_sequence", "no_remediation"],
            "framework": "Tier 2: Learning-Enabled Misconception-Aware Simulated Students",
        },
        "population": pop_desc,
        "rct": rct_results,
        "sensitivity": sensitivity,
        "interpretation": generate_interpretation(rct_results, sensitivity),
    }

    with open(RESULTS_DIR / "simulated_rct_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS SAVED to {RESULTS_DIR / 'simulated_rct_results.json'}")
    print(f"{'=' * 70}")

    # Print final interpretation
    interp = full_results["interpretation"]
    print(f"\n{'=' * 70}")
    print("  INTERPRETATION")
    print(f"{'=' * 70}")
    for line in interp["summary"]:
        print(f"  {line}")

    return full_results


def generate_interpretation(rct_results: dict, sensitivity: dict) -> dict:
    """Generate a human-readable interpretation of results."""
    summary_lines = []

    # Main finding
    adaptive = rct_results["conditions"]["adaptive"]
    summary_lines.append(f"Adaptive condition: mean mastery gain = {adaptive['mean_mastery_gain']:.4f}")
    summary_lines.append(f"Misconception resolution rate: {adaptive['resolution_rate']:.1%}")

    # Best effect size
    best_d = max(rct_results["effect_sizes"].values())
    best_comparison = max(rct_results["effect_sizes"], key=rct_results["effect_sizes"].get)
    summary_lines.append(f"Largest effect size: d = {best_d:.3f} ({best_comparison})")

    # Contextual comparison
    if best_d >= 0.8:
        summary_lines.append("This matches or exceeds the best ITS systems (VanLehn 2011: d=0.76)")
    elif best_d >= 0.5:
        summary_lines.append("This is a medium-to-large effect, competitive with established ITS (Kulik & Fletcher 2016: median d=0.66)")
    elif best_d >= 0.3:
        summary_lines.append("This is a small-to-medium effect, typical of answer-based tutoring systems (VanLehn 2011: d=0.31)")
    else:
        summary_lines.append("This is a small effect; further investigation needed")

    # Resolution comparison
    for key, comp in rct_results["comparisons"].items():
        ratio = comp["misconception_resolution"]["resolution_ratio"]
        if "random" in key:
            summary_lines.append(f"Adaptive resolves misconceptions {ratio:.1f}x faster than random")

    # Robustness
    if sensitivity.get("robust"):
        summary_lines.append("Results are ROBUST: adaptive outperforms all baselines across +/- BKT parameter variation")
    else:
        summary_lines.append("WARNING: Results are NOT fully robust across parameter variation")

    # Honest framing
    summary_lines.append("")
    summary_lines.append("IMPORTANT: These are simulated results that validate system behavior,")
    summary_lines.append("NOT evidence of real student learning outcomes.")
    summary_lines.append("A human RCT (n>=100 per group) is required for effect size claims.")
    summary_lines.append(f"Recommended benchmark: d = 0.5-0.8 on aligned assessments (Kulik & Fletcher 2016)")

    return {
        "summary": summary_lines,
        "bloom_2sigma_achievable": False,
        "realistic_target": "d = 0.5-0.8 on aligned assessments",
        "evidence_level": "Stage 2: Simulated evaluation (system validation)",
        "next_step": "Stage 3: Wizard-of-Oz pilot with 5-10 real students",
    }


if __name__ == "__main__":
    main()
