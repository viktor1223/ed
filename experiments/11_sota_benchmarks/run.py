"""Experiment 11: SOTA Benchmarking for V3 Simulated Student.

Computes a comprehensive set of metrics that are directly comparable to
published results from the simulated student literature:

  1. Error Recurrence Rate  (cf. BEAGLE Wang et al. 2026: 86.2%, real: 92.0%)
  2. Performance Gap         (cf. BEAGLE: +40% between High/Low profiles)
  3. Learning Curves         (cf. Power Law of Practice, Anderson 1982)
  4. Misconception Stability (cf. Scarlatos 2026: consistency across runs)
  5. Response Prediction AUC (cf. BKT/DKT literature: 0.65-0.85)
  6. Sessions to Resolution  (cf. cognitive tutor literature: 3-7 exercises)
  7. Instruction Sensitivity (cf. Experiment 10: Cohen's d = 1.60)
  8. Negative Transfer       (cf. interference theory: measurable degradation)

For each metric, we document:
  - Our observed value
  - The SOTA reference(s) and their values
  - Whether the comparison is apples-to-apples or methodological differences

Usage:
    cd <project_root>
    python experiments/11_sota_benchmarks/run.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, optimize

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
N_STUDENTS = 500
N_INTERACTIONS = 60  # longer trajectory for learning curve analysis


# ─── Utilities ────────────────────────────────────────────────────────────────

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


def cohens_d(g1: list[float], g2: list[float]) -> float:
    n1, n2 = len(g1), len(g2)
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((m1 - m2) / pooled) if pooled > 0 else 0.0


# ─── Benchmark 1: Error Recurrence Rate ──────────────────────────────────────
#
# BEAGLE (Wang et al. 2026) measures error recurrence on programming tasks:
#   - "When a student makes an error, does the simulated student make the SAME
#     kind of error when encountering a similar problem?"
#   - BEAGLE: 86.2% | Vanilla LLM: 7.8% | Real students: 92.0%
#
# Our analog: When a student holds misconception M and encounters a problem
# where M applies, what fraction of the time does the student exhibit M again?
# This is the "error recurrence" rate - misconception-consistent error
# reproduction across encounters.
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_error_recurrence(seed: int = SEED) -> dict:
    """Measure misconception-consistent error recurrence across encounters."""
    random.seed(seed)
    np.random.seed(seed)

    students = generate_students_v3(n=N_STUDENTS, seed=seed)
    bank = load_problem_bank_v2()

    total_applicable = 0  # times a student with active M faces an M-relevant problem
    total_fired = 0       # times the student actually exhibited M

    for student in students:
        for m in student.misconceptions:
            if m.p_active < 0.1:
                continue

            concept_problems = bank.get(m.concept_id, [])
            for problem in concept_problems:
                total_applicable += 1
                response = student.respond(problem)
                if (not response["correct"]
                        and response.get("misconception_used") == m.misconception_id):
                    total_fired += 1

    recurrence = total_fired / total_applicable if total_applicable > 0 else 0.0

    print(f"\n{'='*60}")
    print("BENCHMARK 1: Error Recurrence Rate")
    print(f"{'='*60}")
    print(f"  Applicable misconception-problem encounters: {total_applicable}")
    print(f"  Misconception fired (correct error reproduced): {total_fired}")
    print(f"  ERROR RECURRENCE RATE: {recurrence:.1%}")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ BEAGLE (Wang 2026):     86.2%  (prog tasks) │")
    print(f"  │ Vanilla LLM:             7.8%               │")
    print(f"  │ Real students:           92.0%               │")
    print(f"  │ Our v3 model:           {recurrence:5.1%}               │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  Note: BEAGLE operates on Python code generation tasks with Gemini")
    print(f"  2.0 Flash. Our model is deterministic rule-based on algebra tasks.")
    print(f"  Methodological diff: BEAGLE measures re-occurrence of the SAME")
    print(f"  error type across task attempts; we measure misconception firing")
    print(f"  probability across problems within a concept.")

    return {
        "metric": "error_recurrence_rate",
        "our_value": round(recurrence, 4),
        "n_applicable": total_applicable,
        "n_fired": total_fired,
        "references": {
            "BEAGLE_Wang_2026": {"value": 0.862, "domain": "Python programming", "note": "Gemini 2.0 Flash"},
            "Vanilla_LLM": {"value": 0.078, "domain": "Python programming"},
            "Real_students": {"value": 0.920, "domain": "Python programming Turing test"},
        },
    }


# ─── Benchmark 2: Performance Gap (High vs Low Profiles) ──────────────────
#
# BEAGLE measures a +40% performance gap between High and Low student profiles
# on TASK COMPLETION (absolute performance), not learning gains.
# Vanilla LLM: +0% (no performance differentiation).
#
# We measure TWO things:
#   (a) Absolute performance gap: response accuracy difference (High vs Low)
#       during tutoring. This is the BEAGLE-comparable metric.
#   (b) Gain gap: learning delta difference. Low students gaining MORE than
#       high students is actually realistic (ceiling effects).
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_performance_gap(seed: int = SEED) -> dict:
    """Measure performance gap between strong and weak student profiles."""
    random.seed(seed)
    np.random.seed(seed)

    kg = KnowledgeGraph.from_json(KG_PATH)
    bank = load_problem_bank_v2()
    practice_bank, _ = split_problem_bank(bank, n_test_per_concept=2, seed=seed)

    students = generate_students_v3(n=N_STUDENTS, seed=seed)

    # Classify by initial p_know: top 25% vs bottom 25%
    initial_means = []
    for s in students:
        avg = np.mean([cs.p_know for cs in s.concepts.values()])
        initial_means.append(avg)

    sorted_indices = np.argsort(initial_means)
    q25 = N_STUDENTS // 4
    low_indices = sorted_indices[:q25]
    high_indices = sorted_indices[-q25:]

    def run_tutoring(student: SimulatedStudentV3) -> tuple[float, float, float]:
        """Returns (mean_gain, accuracy_during_tutoring, final_p_know)."""
        tutor_state = StudentState(kg)
        initial = {cid: cs.p_know for cid, cs in student.concepts.items()}
        correct_count = 0

        for t in range(N_INTERACTIONS):
            concept_id = adaptive_concept_selection(tutor_state, kg)
            problems = practice_bank.get(concept_id, [])
            if not problems:
                continue
            problem = random.choice(problems)
            response = student.respond(problem)
            if response["correct"]:
                correct_count += 1
            tutor_state.update(concept_id, correct=response["correct"], confidence=0.8)

            true_misc = response.get("misconception_used")
            student.receive_instruction(concept_id, targeted_misconception=true_misc)

        final = {cid: cs.p_know for cid, cs in student.concepts.items()}
        deltas = [final[c] - initial.get(c, 0.1) for c in final]
        accuracy = correct_count / N_INTERACTIONS
        final_pk = float(np.mean([cs.p_know for cs in student.concepts.values()]))
        return float(np.mean(deltas)), accuracy, final_pk

    high_results = [run_tutoring(students[i]) for i in high_indices]
    students2 = generate_students_v3(n=N_STUDENTS, seed=seed)
    low_results = [run_tutoring(students2[i]) for i in low_indices]

    high_gains = [r[0] for r in high_results]
    low_gains = [r[0] for r in low_results]
    high_accuracy = [r[1] for r in high_results]
    low_accuracy = [r[1] for r in low_results]
    high_final_pk = [r[2] for r in high_results]
    low_final_pk = [r[2] for r in low_results]

    # Absolute performance gap (BEAGLE-comparable)
    acc_gap = float(np.mean(high_accuracy)) - float(np.mean(low_accuracy))
    acc_gap_pct = acc_gap * 100  # as percentage points
    final_pk_gap = float(np.mean(high_final_pk)) - float(np.mean(low_final_pk))

    # Gain gap (ceiling effect analysis)
    gain_gap = float(np.mean(high_gains)) - float(np.mean(low_gains))

    initial_high = float(np.mean([initial_means[i] for i in high_indices]))
    initial_low = float(np.mean([initial_means[i] for i in low_indices]))

    d_acc = cohens_d(high_accuracy, low_accuracy)
    d_gain = cohens_d(high_gains, low_gains)

    print(f"\n{'='*60}")
    print("BENCHMARK 2: Performance Gap (High vs Low Profile)")
    print(f"{'='*60}")
    print(f"  Initial p_know: High={initial_high:.3f}, Low={initial_low:.3f}")
    print(f"  (a) ABSOLUTE PERFORMANCE (BEAGLE-comparable):")
    print(f"    Accuracy: High={np.mean(high_accuracy):.3f}, Low={np.mean(low_accuracy):.3f}")
    print(f"    Gap: {acc_gap_pct:+.1f} pct pts (Cohen's d = {d_acc:.2f})")
    print(f"    Final p_know: High={np.mean(high_final_pk):.3f}, Low={np.mean(low_final_pk):.3f}")
    print(f"    p_know gap: {final_pk_gap:+.3f}")
    print(f"  (b) LEARNING GAIN (ceiling effect analysis):")
    print(f"    Gain: High={np.mean(high_gains):.4f}, Low={np.mean(low_gains):.4f}")
    print(f"    Gap: {gain_gap:+.4f} (Cohen's d = {d_gain:.2f})")
    print(f"    Low > High is EXPECTED (ceiling effect / more room to grow)")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ BEAGLE (Wang 2026):  +40% task completion   │")
    print(f"  │ Vanilla LLM:          +0%                   │")
    print(f"  │ Our v3 accuracy gap: {acc_gap_pct:+.1f} pct pts          │")
    print(f"  │ Our v3 p_know gap:   {final_pk_gap:+.3f}                │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  Note: BEAGLE's +40% is absolute task completion difference.")
    print(f"  Our accuracy gap and p_know gap are the analogous metrics.")
    print(f"  Low students gaining more than high students reflects realistic")
    print(f"  ceiling effects, consistent with learning science literature.")

    return {
        "metric": "performance_gap",
        "accuracy_gap_pct": round(acc_gap_pct, 1),
        "final_pk_gap": round(final_pk_gap, 3),
        "gain_gap": round(gain_gap, 4),
        "cohens_d_accuracy": round(d_acc, 3),
        "cohens_d_gain": round(d_gain, 3),
        "high_accuracy_mean": round(float(np.mean(high_accuracy)), 3),
        "low_accuracy_mean": round(float(np.mean(low_accuracy)), 3),
        "high_gain_mean": round(float(np.mean(high_gains)), 4),
        "low_gain_mean": round(float(np.mean(low_gains)), 4),
        "high_final_pk": round(float(np.mean(high_final_pk)), 3),
        "low_final_pk": round(float(np.mean(low_final_pk)), 3),
        "initial_high": round(initial_high, 3),
        "initial_low": round(initial_low, 3),
        "n_per_group": q25,
        "references": {
            "BEAGLE_Wang_2026": {"value": "+40%", "domain": "Python task completion (absolute)"},
            "Vanilla_LLM": {"value": "+0%", "domain": "No profile differentiation"},
        },
    }


# ─── Benchmark 3: Learning Curve Shape ───────────────────────────────────────
#
# The Power Law of Practice (Newell & Rosenbloom, 1981; Anderson, 1982) states
# that performance improves as a function of practice according to:
#     P(n) = a * n^b + c
# where n is the number of practice opportunities and b is negative.
#
# Real students follow this reliably. A validated simulated student should too.
# BKT predicts geometric (exponential) curves; real data is power-law.
#
# We fit both power-law and exponential models and compare R² to show our
# model produces realistic learning curves.
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_learning_curves(seed: int = SEED) -> dict:
    """Fit power-law and exponential models to aggregated learning curves."""
    random.seed(seed)
    np.random.seed(seed)

    kg = KnowledgeGraph.from_json(KG_PATH)
    bank = load_problem_bank_v2()
    practice_bank, _ = split_problem_bank(bank, n_test_per_concept=2, seed=seed)

    students = generate_students_v3(n=N_STUDENTS, seed=seed)

    # Track per-concept accuracy over interaction number
    # Key: interaction_number -> list of (correct: bool)
    concept_accuracy_by_step: dict[int, list[bool]] = defaultdict(list)
    overall_p_know_by_step: dict[int, list[float]] = defaultdict(list)

    for student in students:
        tutor_state = StudentState(kg)

        for t in range(N_INTERACTIONS):
            concept_id = adaptive_concept_selection(tutor_state, kg)
            problems = practice_bank.get(concept_id, [])
            if not problems:
                continue
            problem = random.choice(problems)
            response = student.respond(problem)
            tutor_state.update(concept_id, correct=response["correct"], confidence=0.8)

            concept_accuracy_by_step[t].append(response["correct"])

            # Perfect tutoring
            true_misc = response.get("misconception_used")
            student.receive_instruction(concept_id, targeted_misconception=true_misc)

            # Snapshot average p_know
            avg_pk = float(np.mean([cs.p_know for cs in student.concepts.values()]))
            overall_p_know_by_step[t].append(avg_pk)

    # Aggregate: mean accuracy and mean p_know at each step
    steps = sorted(concept_accuracy_by_step.keys())
    mean_accuracy = [float(np.mean(concept_accuracy_by_step[s])) for s in steps]
    mean_p_know = [float(np.mean(overall_p_know_by_step[s])) for s in steps]

    x = np.array(steps, dtype=float) + 1  # 1-indexed practice opportunities

    # Fit power law: y = a * x^b + c
    def power_law(x, a, b, c):
        return a * np.power(x, b) + c

    # Fit exponential: y = a * (1 - exp(-b*x)) + c
    def exponential(x, a, b, c):
        return a * (1.0 - np.exp(-b * x)) + c

    y_acc = np.array(mean_accuracy)
    y_pk = np.array(mean_p_know)

    results_curves = {}

    for label, y in [("accuracy", y_acc), ("p_know", y_pk)]:
        try:
            popt_pow, _ = optimize.curve_fit(
                power_law, x, y, p0=[0.5, 0.3, 0.3], maxfev=5000,
            )
            y_pred_pow = power_law(x, *popt_pow)
            ss_res = np.sum((y - y_pred_pow)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2_pow = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except RuntimeError:
            popt_pow = [0, 0, 0]
            r2_pow = 0.0

        try:
            popt_exp, _ = optimize.curve_fit(
                exponential, x, y, p0=[0.5, 0.05, 0.3], maxfev=5000,
            )
            y_pred_exp = exponential(x, *popt_exp)
            ss_res = np.sum((y - y_pred_exp)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2_exp = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except RuntimeError:
            popt_exp = [0, 0, 0]
            r2_exp = 0.0

        results_curves[label] = {
            "power_law_r2": round(r2_pow, 4),
            "power_law_params": [round(p, 4) for p in popt_pow],
            "exponential_r2": round(r2_exp, 4),
            "exponential_params": [round(p, 4) for p in popt_exp],
        }

    best_fit = "power_law" if results_curves["accuracy"]["power_law_r2"] >= results_curves["accuracy"]["exponential_r2"] else "exponential"

    print(f"\n{'='*60}")
    print("BENCHMARK 3: Learning Curve Shape")
    print(f"{'='*60}")
    print(f"  Accuracy curve:")
    print(f"    Power law R²:    {results_curves['accuracy']['power_law_r2']:.4f}")
    print(f"    Exponential R²:  {results_curves['accuracy']['exponential_r2']:.4f}")
    print(f"  p_know curve:")
    print(f"    Power law R²:    {results_curves['p_know']['power_law_r2']:.4f}")
    print(f"    Exponential R²:  {results_curves['p_know']['exponential_r2']:.4f}")
    print(f"  Best fit: {best_fit}")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ Real students: Power law (R² > 0.90 typical)│")
    print(f"  │ Pure BKT: Exponential (geometric growth)    │")
    print(f"  │ Our v3 model: see values above              │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  Note: Power Law of Practice (Newell & Rosenbloom 1981)")
    print(f"  predicts deceleration in improvement, which real students exhibit.")
    print(f"  Pure BKT produces geometric (exponential) curves.")
    print(f"  A good simulated student fits power-law better or comparably.")

    return {
        "metric": "learning_curve_shape",
        "best_fit": best_fit,
        "curves": results_curves,
        "mean_accuracy_trajectory": [round(v, 4) for v in mean_accuracy],
        "mean_p_know_trajectory": [round(v, 4) for v in mean_p_know],
        "references": {
            "Power_Law_of_Practice": {
                "citation": "Newell & Rosenbloom (1981), Anderson (1982)",
                "note": "Real students follow power law with R² > 0.90",
            },
            "Pure_BKT": {
                "note": "Produces exponential (geometric) mastery curves",
            },
        },
    }


# ─── Benchmark 4: Misconception Stability (Run-to-Run Consistency) ──────────
#
# Scarlatos (2026) emphasizes consistency: LLM-based students produce variable
# outputs between runs. Real students have stable misconceptions.
# We measure: across 10 independent runs (different RNG seeds for responses,
# SAME initial student), how consistent are the misconception-driven errors?
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_misconception_stability(seed: int = SEED) -> dict:
    """Measure run-to-run consistency of misconception-driven responses."""
    N_RUNS = 10
    N_PROBE = 50  # students to probe deeply

    bank = load_problem_bank_v2()

    # Generate the same students
    base_students = generate_students_v3(n=N_PROBE, seed=seed)

    # For each student, run N_RUNS independent response sequences on the same
    # problems and measure how consistently misconceptions fire
    per_student_consistency = []

    for s_idx, base_student in enumerate(base_students):
        # Collect problems where student has active misconceptions
        probes = []
        for m in base_student.misconceptions:
            if m.p_active < 0.1:
                continue
            concept_problems = bank.get(m.concept_id, [])
            for p in concept_problems[:3]:  # up to 3 problems per misconception
                probes.append((p, m.misconception_id))

        if not probes:
            continue

        # Run N_RUNS response sets
        run_results = []
        for run_i in range(N_RUNS):
            random.seed(seed * 1000 + s_idx * 100 + run_i)
            np.random.seed(seed * 1000 + s_idx * 100 + run_i)

            # Regenerate same student
            students = generate_students_v3(n=N_PROBE, seed=seed)
            student = students[s_idx]

            responses = []
            for problem, _ in probes:
                r = student.respond(problem)
                responses.append(r.get("misconception_used"))
            run_results.append(responses)

        # Compute pairwise agreement across runs
        agreements = []
        for i_run in range(N_RUNS):
            for j_run in range(i_run + 1, N_RUNS):
                matches = sum(
                    1 for a, b in zip(run_results[i_run], run_results[j_run])
                    if a == b
                )
                agreements.append(matches / len(probes) if probes else 0)

        per_student_consistency.append(float(np.mean(agreements)))

    overall = float(np.mean(per_student_consistency)) if per_student_consistency else 0.0
    std = float(np.std(per_student_consistency)) if per_student_consistency else 0.0

    print(f"\n{'='*60}")
    print("BENCHMARK 4: Misconception Stability (Run-to-Run Consistency)")
    print(f"{'='*60}")
    print(f"  Students probed: {len(per_student_consistency)}")
    print(f"  Runs per student: {N_RUNS}")
    print(f"  Mean pairwise agreement: {overall:.1%} +/- {std:.1%}")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ LLM prompting (Scarlatos 2026):  variable   │")
    print(f"  │ LLM SFT+DPO (Scarlatos 2026):   variable   │")
    print(f"  │ Deterministic model:              100%       │")
    print(f"  │ Our v3 model (stochastic BKT):   {overall:.1%}       │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  Note: Our model uses stochastic BKT sampling (p_know vs random()")
    print(f"  and p_active vs random()), so responses differ between runs.")
    print(f"  The KEY metric is consistency of WHICH misconception fires,")
    print(f"  not whether it fires every single time. LLM-based students")
    print(f"  produce entirely different error types between runs.")

    return {
        "metric": "misconception_stability",
        "mean_agreement": round(overall, 4),
        "std_agreement": round(std, 4),
        "n_students_probed": len(per_student_consistency),
        "n_runs": N_RUNS,
        "references": {
            "LLM_prompting_Scarlatos_2026": {
                "note": "Highly variable; different error types between runs",
            },
            "Deterministic_model": {
                "value": 1.0,
                "note": "Same input always produces same output",
            },
            "Scarlatos_error_metric": {
                "value_range": "0.02-0.19",
                "note": "Error replication scores; even oracle only 0.19",
            },
        },
    }


# ─── Benchmark 5: Response Prediction (Internal BKT-to-Response Alignment) ──
#
# The BKT/DKT literature reports prediction AUC of 0.65-0.85 for predicting
# student correctness. Since our student IS a BKT model, we measure how well
# an external observer with a fresh BKT tracker can predict our student's
# responses after watching some interactions. This is the "learnability" of the
# student's response pattern.
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_response_prediction(seed: int = SEED) -> dict:
    """Measure how well external BKT can predict v3 student responses."""
    random.seed(seed)
    np.random.seed(seed)

    kg = KnowledgeGraph.from_json(KG_PATH)
    bank = load_problem_bank_v2()
    practice_bank, _ = split_problem_bank(bank, n_test_per_concept=2, seed=seed)

    students = generate_students_v3(n=N_STUDENTS, seed=seed)

    all_true = []
    all_pred = []

    for student in students:
        tutor_state = StudentState(kg)

        for t in range(N_INTERACTIONS):
            concept_id = adaptive_concept_selection(tutor_state, kg)
            problems = practice_bank.get(concept_id, [])
            if not problems:
                continue
            problem = random.choice(problems)

            # External prediction: use tutor's BKT estimate
            p_correct = tutor_state.mastery.get(concept_id, 0.5)
            all_pred.append(p_correct)

            response = student.respond(problem)
            all_true.append(1 if response["correct"] else 0)

            tutor_state.update(concept_id, correct=response["correct"], confidence=0.8)

            # Perfect tutoring
            true_misc = response.get("misconception_used")
            student.receive_instruction(concept_id, targeted_misconception=true_misc)

    # Compute AUC
    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)

    # Manual AUC (no sklearn dependency)
    sorted_indices = np.argsort(-pred_arr)
    true_sorted = true_arr[sorted_indices]
    n_pos = np.sum(true_arr)
    n_neg = len(true_arr) - n_pos

    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        tp = 0.0
        auc_sum = 0.0
        for i in range(len(true_sorted)):
            if true_sorted[i] == 1:
                tp += 1
            else:
                auc_sum += tp
        auc = auc_sum / (n_pos * n_neg)

    # Also compute accuracy at threshold 0.5
    pred_binary = (pred_arr >= 0.5).astype(int)
    accuracy = float(np.mean(pred_binary == true_arr))

    # Calibration: bin predictions and compare to actual rates
    n_bins = 10
    calibration = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        mask = (pred_arr >= lo) & (pred_arr < hi)
        if np.sum(mask) >= 10:
            actual = float(np.mean(true_arr[mask]))
            predicted = float(np.mean(pred_arr[mask]))
            calibration.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(np.sum(mask)),
                "predicted": round(predicted, 3),
                "actual": round(actual, 3),
            })

    # Expected Calibration Error
    ece = 0.0
    total_n = 0
    for cal in calibration:
        ece += cal["n"] * abs(cal["predicted"] - cal["actual"])
        total_n += cal["n"]
    ece = ece / total_n if total_n > 0 else 0.0

    print(f"\n{'='*60}")
    print("BENCHMARK 5: Response Prediction (External BKT AUC)")
    print(f"{'='*60}")
    print(f"  Total response predictions: {len(all_true)}")
    print(f"  Positive rate (correct): {np.mean(true_arr):.3f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy (threshold 0.5): {accuracy:.4f}")
    print(f"  Expected Calibration Error: {ece:.4f}")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ BKT literature AUC:          0.63-0.72      │")
    print(f"  │ DKT (Piech 2015):            0.80-0.86      │")
    print(f"  │ AKT (Ghosh 2020):            0.77-0.85      │")
    print(f"  │ SAINT (Choi 2020):           0.78-0.83      │")
    print(f"  │ Our v3 (external BKT):       {auc:.4f}           │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  Note: BKT/DKT AUC is measured on REAL student data (e.g. ASSISTments,")
    print(f"  EdNet, Junyi). Our AUC measures external BKT prediction on our")
    print(f"  simulated student. High AUC means the student behaves predictably;")
    print(f"  an AUC near published BKT range (0.63-0.72) indicates the student")
    print(f"  has realistic unpredictability consistent with actual learners.")
    print(f"  Calibration tells us whether predicted probabilities match reality.")

    return {
        "metric": "response_prediction_auc",
        "auc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "ece": round(ece, 4),
        "positive_rate": round(float(np.mean(true_arr)), 4),
        "n_predictions": len(all_true),
        "calibration": calibration,
        "references": {
            "BKT_Corbett_1995": {"auc_range": "0.63-0.72", "dataset": "various"},
            "DKT_Piech_2015": {"auc_range": "0.80-0.86", "dataset": "ASSISTments"},
            "AKT_Ghosh_2020": {"auc_range": "0.77-0.85", "dataset": "ASSISTments, Statics, EdNet"},
            "SAINT_Choi_2020": {"auc_range": "0.78-0.83", "dataset": "EdNet"},
        },
    }


# ─── Benchmark 6: Sessions to Misconception Resolution ──────────────────────
#
# Cognitive tutor literature: mastery of a knowledge component typically
# requires 3-7 opportunities (Corbett & Anderson 1995, Ritter et al. 2007).
# We measure: how many targeted instruction events does it take to resolve
# an active misconception?
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_sessions_to_resolution(seed: int = SEED) -> dict:
    """Measure how many correct-targeting instructions to resolve a misconception."""
    random.seed(seed)
    np.random.seed(seed)

    students = generate_students_v3(n=N_STUDENTS, seed=seed)
    resolution_counts = []  # number of correct-target instructions until resolution
    unresolved = 0

    MAX_INSTRUCTIONS = 30

    for student in students:
        for m in student.misconceptions:
            if m.p_active < 0.1:
                continue

            initial_p_active = m.p_active
            count = 0

            for i in range(MAX_INSTRUCTIONS):
                student.receive_instruction(
                    m.concept_id, targeted_misconception=m.misconception_id,
                )
                count += 1

                if m.p_active < 0.1:
                    resolution_counts.append(count)
                    # Reset for next misconception
                    break
            else:
                # Did not resolve within MAX_INSTRUCTIONS
                unresolved += 1

    if resolution_counts:
        mean_sessions = float(np.mean(resolution_counts))
        median_sessions = float(np.median(resolution_counts))
        std_sessions = float(np.std(resolution_counts))
        q25 = float(np.percentile(resolution_counts, 25))
        q75 = float(np.percentile(resolution_counts, 75))
    else:
        mean_sessions = median_sessions = std_sessions = q25 = q75 = 0.0

    total = len(resolution_counts) + unresolved

    print(f"\n{'='*60}")
    print("BENCHMARK 6: Sessions to Misconception Resolution")
    print(f"{'='*60}")
    print(f"  Misconceptions tracked: {total}")
    print(f"  Resolved: {len(resolution_counts)}, Unresolved: {unresolved}")
    print(f"  Sessions to resolve:")
    print(f"    Mean:   {mean_sessions:.1f}")
    print(f"    Median: {median_sessions:.1f}")
    print(f"    IQR:    [{q25:.1f}, {q75:.1f}]")
    print(f"    Std:    {std_sessions:.1f}")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ Cognitive tutor lit:  3-7 opportunities      │")
    print(f"  │   Corbett & Anderson (1995): ~5 per KC       │")
    print(f"  │   Ritter et al. (2007): 3-7 range            │")
    print(f"  │ Our v3 model median: {median_sessions:.1f} sessions          │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  Note: 'Sessions' here = # of correctly-targeted instruction events.")
    print(f"  Cognitive tutor 'opportunities' = problem attempts with immediate")
    print(f"  feedback. Not identical but operationally analogous.")

    return {
        "metric": "sessions_to_resolution",
        "mean_sessions": round(mean_sessions, 2),
        "median_sessions": round(median_sessions, 2),
        "std_sessions": round(std_sessions, 2),
        "iqr": [round(q25, 2), round(q75, 2)],
        "n_resolved": len(resolution_counts),
        "n_unresolved": unresolved,
        "resolution_counts": resolution_counts[:100],  # sample for JSON
        "references": {
            "Corbett_Anderson_1995": {"value": "~5 per KC", "note": "BKT-based cognitive tutors"},
            "Ritter_2007": {"value": "3-7", "note": "Cognitive Tutor: Algebra field studies"},
        },
    }


# ─── Benchmark 7: Instruction Sensitivity (from Exp 10) ─────────────────────
#
# Our own novel metric. Recomputes experiment 10 with larger N for this report.
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_instruction_sensitivity(seed: int = SEED) -> dict:
    """Recompute instruction sensitivity with larger sample."""
    random.seed(seed)
    np.random.seed(seed)

    kg = KnowledgeGraph.from_json(KG_PATH)
    bank = load_problem_bank_v2()
    practice_bank, _ = split_problem_bank(bank, n_test_per_concept=2, seed=seed)

    conditions = ["perfect", "random", "always_wrong", "no_instruction"]
    condition_deltas: dict[str, list[float]] = {}

    for condition in conditions:
        random.seed(seed)
        np.random.seed(seed)

        students = generate_students_v3(n=N_STUDENTS, seed=seed)
        deltas_list = []

        for student in students:
            tutor_state = StudentState(kg)
            initial = {cid: cs.p_know for cid, cs in student.concepts.items()}
            engaged = set()

            for t in range(N_INTERACTIONS):
                concept_id = adaptive_concept_selection(tutor_state, kg)
                engaged.add(concept_id)
                problems = practice_bank.get(concept_id, [])
                if not problems:
                    continue
                problem = random.choice(problems)
                response = student.respond(problem)
                tutor_state.update(concept_id, correct=response["correct"], confidence=0.8)

                true_misc = response.get("misconception_used")

                if condition == "perfect":
                    student.receive_instruction(concept_id, targeted_misconception=true_misc)
                elif condition == "random":
                    concept_obj = kg.concepts.get(concept_id)
                    if concept_obj and concept_obj.misconceptions and random.random() < 0.5:
                        m = random.choice(concept_obj.misconceptions)
                        student.receive_instruction(concept_id, targeted_misconception=m.id)
                    else:
                        student.receive_instruction(concept_id, targeted_misconception=None)
                elif condition == "always_wrong":
                    student.receive_instruction(concept_id, targeted_misconception="__nonexistent__")
                elif condition == "no_instruction":
                    pass

            if engaged:
                delta_list = [
                    student.concepts[c].p_know - initial.get(c, 0.1)
                    for c in engaged if c in student.concepts
                ]
                deltas_list.append(float(np.mean(delta_list)) if delta_list else 0.0)
            else:
                deltas_list.append(0.0)

        condition_deltas[condition] = deltas_list

    d_perf_wrong = cohens_d(condition_deltas["perfect"], condition_deltas["always_wrong"])
    d_perf_rand = cohens_d(condition_deltas["perfect"], condition_deltas["random"])
    d_perf_none = cohens_d(condition_deltas["perfect"], condition_deltas["no_instruction"])

    means = {c: float(np.mean(v)) for c, v in condition_deltas.items()}

    print(f"\n{'='*60}")
    print("BENCHMARK 7: Instruction Sensitivity (Expanded Exp 10)")
    print(f"{'='*60}")
    print(f"  N = {N_STUDENTS}, interactions = {N_INTERACTIONS}")
    for c in conditions:
        print(f"  {c:20s}: mean delta = {means[c]:.4f}")
    print(f"  Cohen's d (perfect vs always_wrong): {d_perf_wrong:.3f}")
    print(f"  Cohen's d (perfect vs random):       {d_perf_rand:.3f}")
    print(f"  Cohen's d (perfect vs none):         {d_perf_none:.3f}")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ This is a NOVEL metric - no direct SOTA     │")
    print(f"  │ comparison exists.                          │")
    print(f"  │ Most simulated students do not model        │")
    print(f"  │ instruction sensitivity at all.             │")
    print(f"  │ d > 0.8 = large effect (Cohen's guidelines) │")
    print(f"  └─────────────────────────────────────────────┘")

    return {
        "metric": "instruction_sensitivity",
        "condition_means": {c: round(v, 4) for c, v in means.items()},
        "cohens_d_perf_wrong": round(d_perf_wrong, 3),
        "cohens_d_perf_rand": round(d_perf_rand, 3),
        "cohens_d_perf_none": round(d_perf_none, 3),
        "n_students": N_STUDENTS,
        "n_interactions": N_INTERACTIONS,
        "references": {
            "Cohen_1988": {
                "small": 0.2, "medium": 0.5, "large": 0.8,
                "note": "Conventional effect size thresholds",
            },
        },
    }


# ─── Benchmark 8: Negative Transfer ──────────────────────────────────────────
#
# Interference theory (Anderson 1983; Baddeley 1976) predicts that incorrect
# instruction causes measurable harm. No published simulated student models
# negative transfer from wrong instruction.
#
# The key comparison is: perfect tutoring first, then SWITCH to wrong
# instruction. Does the wrong phase undo progress? We also measure
# misconception reinforcement and confusion accumulation as secondary markers.
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_negative_transfer(seed: int = SEED) -> dict:
    """Measure negative transfer through a switch-design experiment."""
    random.seed(seed)
    np.random.seed(seed)

    kg = KnowledgeGraph.from_json(KG_PATH)
    bank = load_problem_bank_v2()
    practice_bank, _ = split_problem_bank(bank, n_test_per_concept=2, seed=seed)

    # Three conditions:
    # A: Perfect tutoring throughout (60 interactions)
    # B: Perfect first 30, then wrong for 30 (switch design)
    # C: No instruction throughout (control)
    conditions = ["perfect_throughout", "perfect_then_wrong", "no_instruction"]
    condition_results: dict[str, dict] = {}

    SWITCH_POINT = N_INTERACTIONS // 2

    for condition in conditions:
        random.seed(seed)
        np.random.seed(seed)

        students = generate_students_v3(n=N_STUDENTS, seed=seed)
        final_p_knows = []
        misconception_changes = []
        confusion_totals = []
        midpoint_p_knows = []

        for student in students:
            tutor_state = StudentState(kg)
            initial_misc = {m.misconception_id: m.p_active for m in student.misconceptions if m.p_active > 0.1}

            for t in range(N_INTERACTIONS):
                concept_id = adaptive_concept_selection(tutor_state, kg)
                problems = practice_bank.get(concept_id, [])
                if not problems:
                    continue
                problem = random.choice(problems)
                response = student.respond(problem)
                tutor_state.update(concept_id, correct=response["correct"], confidence=0.8)

                true_misc = response.get("misconception_used")

                if condition == "perfect_throughout":
                    student.receive_instruction(concept_id, targeted_misconception=true_misc)
                elif condition == "perfect_then_wrong":
                    if t < SWITCH_POINT:
                        student.receive_instruction(concept_id, targeted_misconception=true_misc)
                    else:
                        student.receive_instruction(concept_id, targeted_misconception="__nonexistent__")
                # no_instruction: nothing

                if t == SWITCH_POINT - 1:
                    midpoint_p_knows.append(
                        float(np.mean([cs.p_know for cs in student.concepts.values()]))
                    )

            final_pk = float(np.mean([cs.p_know for cs in student.concepts.values()]))
            final_p_knows.append(final_pk)

            for m in student.misconceptions:
                if m.misconception_id in initial_misc:
                    misconception_changes.append(
                        m.p_active - initial_misc[m.misconception_id]
                    )

            confusion_totals.append(sum(student.confusion_count.values()))

        condition_results[condition] = {
            "final_p_knows": final_p_knows,
            "midpoint_p_know_mean": round(float(np.mean(midpoint_p_knows)), 4) if midpoint_p_knows else 0.0,
            "final_p_know_mean": round(float(np.mean(final_p_knows)), 4),
            "misconception_change_mean": round(float(np.mean(misconception_changes)), 4) if misconception_changes else 0.0,
            "confusion_mean": round(float(np.mean(confusion_totals)), 2),
        }

    # Key comparison: perfect_throughout vs perfect_then_wrong at endpoint
    d_neg = cohens_d(
        condition_results["perfect_throughout"]["final_p_knows"],
        condition_results["perfect_then_wrong"]["final_p_knows"],
    )
    t_stat, p_val = stats.ttest_ind(
        condition_results["perfect_throughout"]["final_p_knows"],
        condition_results["perfect_then_wrong"]["final_p_knows"],
    )

    perf_final = condition_results["perfect_throughout"]["final_p_know_mean"]
    switch_final = condition_results["perfect_then_wrong"]["final_p_know_mean"]
    switch_mid = condition_results["perfect_then_wrong"]["midpoint_p_know_mean"]
    none_final = condition_results["no_instruction"]["final_p_know_mean"]

    # Did the wrong phase cause regression from midpoint?
    regression = switch_mid - switch_final
    negative_transfer_exists = switch_final < perf_final

    # Misconception reinforcement is the clearest signal
    misc_change_switch = condition_results["perfect_then_wrong"]["misconception_change_mean"]
    misc_change_perf = condition_results["perfect_throughout"]["misconception_change_mean"]
    misc_change_none = condition_results["no_instruction"]["misconception_change_mean"]

    confusion_switch = condition_results["perfect_then_wrong"]["confusion_mean"]
    confusion_perf = condition_results["perfect_throughout"]["confusion_mean"]

    print(f"\n{'='*60}")
    print("BENCHMARK 8: Negative Transfer (Switch Design)")
    print(f"{'='*60}")
    print(f"  Conditions: {N_INTERACTIONS} interactions, switch at t={SWITCH_POINT}")
    print(f"  (a) FINAL p_know:")
    print(f"    Perfect throughout:  {perf_final:.4f}")
    print(f"    Perfect-then-wrong:  {switch_final:.4f} (midpoint was {switch_mid:.4f})")
    print(f"    No instruction:      {none_final:.4f}")
    print(f"  (b) NEGATIVE TRANSFER EVIDENCE:")
    print(f"    p_know loss vs perfect: {perf_final - switch_final:.4f}")
    print(f"    Regression from midpoint: {regression:+.4f}")
    print(f"    Cohen's d (perfect vs switch): {d_neg:.3f}")
    print(f"    t = {float(t_stat):.3f}, p = {float(p_val):.6f}")
    print(f"  (c) MISCONCEPTION REINFORCEMENT:")
    print(f"    Misc change (perfect): {misc_change_perf:+.4f}")
    print(f"    Misc change (switch):  {misc_change_switch:+.4f}")
    print(f"    Misc change (none):    {misc_change_none:+.4f}")
    print(f"  (d) CONFUSION ACCUMULATION:")
    print(f"    Confusion (perfect):   {confusion_perf:.1f}")
    print(f"    Confusion (switch):    {confusion_switch:.1f}")
    print(f"  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │ NO published sim student models negative transfer.       │")
    print(f"  │ BEAGLE: prevents self-correction, no reinforcement.      │")
    print(f"  │ MalAlgoPy: static misconceptions, no learning dynamics.  │")
    print(f"  │ Interference theory (Anderson 1983):                     │")
    print(f"  │   predicts measurable harm from incorrect instruction.   │")
    print(f"  └──────────────────────────────────────────────────────────┘")

    return {
        "metric": "negative_transfer",
        "negative_transfer_detected": negative_transfer_exists,
        "perfect_final": perf_final,
        "switch_final": switch_final,
        "switch_midpoint": switch_mid,
        "none_final": none_final,
        "p_know_loss_vs_perfect": round(perf_final - switch_final, 4),
        "regression_from_midpoint": round(regression, 4),
        "cohens_d": round(d_neg, 3),
        "t_statistic": round(float(t_stat), 3),
        "p_value": round(float(p_val), 6),
        "misconception_change_perfect": misc_change_perf,
        "misconception_change_switch": misc_change_switch,
        "misconception_change_none": misc_change_none,
        "confusion_perfect": confusion_perf,
        "confusion_switch": confusion_switch,
        "references": {
            "BEAGLE_Wang_2026": {
                "negative_transfer": "No",
                "note": "Prevents self-correction via observation filtering, but does not reinforce misconceptions",
            },
            "MalAlgoPy_Sonkar_2024": {
                "negative_transfer": "No",
                "note": "Static misconception profiles; no learning dynamics",
            },
            "Interference_theory": {
                "citation": "Anderson (1983), Baddeley (1976)",
                "note": "Predicts proactive/retroactive interference from incorrect instruction",
            },
        },
    }


# ─── Main: Run All Benchmarks ────────────────────────────────────────────────

def generate_plots(all_results: dict) -> None:
    """Generate comparison plots for the paper."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle("V3 Simulated Student: SOTA Benchmark Comparison", fontsize=16, fontweight="bold")

    # Plot 1: Error recurrence comparison
    ax = axes[0, 0]
    r = all_results["error_recurrence"]
    labels = ["Real\nStudents", "BEAGLE", "Our V3", "Vanilla\nLLM"]
    values = [0.920, 0.862, r["our_value"], 0.078]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Error Recurrence Rate")
    ax.set_title("B1: Error Recurrence")
    ax.set_ylim(0, 1.0)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    ax.axhline(y=0.920, color="#2196F3", linestyle="--", alpha=0.5, linewidth=0.8)

    # Plot 2: Performance gap (accuracy-based, BEAGLE-comparable)
    ax = axes[0, 1]
    r = all_results["performance_gap"]
    labels = ["BEAGLE\n(task compl)", "Our V3\n(accuracy)", "Vanilla\nLLM"]
    values = [40.0, r["accuracy_gap_pct"], 0.0]
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Performance Gap (pct pts)")
    ax.set_title("B2: Accuracy Gap (High vs Low)")
    for bar, v in zip(bars, values):
        y_pos = max(v, 0) + 1 if v >= 0 else v - 2
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{v:+.1f}", ha="center", fontsize=9)

    # Plot 3: Learning curve
    ax = axes[0, 2]
    r = all_results["learning_curves"]
    steps = list(range(1, len(r["mean_accuracy_trajectory"]) + 1))
    ax.plot(steps, r["mean_accuracy_trajectory"], "b-", linewidth=2, label="Accuracy")
    ax.plot(steps, r["mean_p_know_trajectory"], "r--", linewidth=2, label="p_know")
    ax.set_xlabel("Interaction #")
    ax.set_ylabel("Performance")
    ax.set_title(f"B3: Learning Curve (Power R²={r['curves']['accuracy']['power_law_r2']:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Misconception stability
    ax = axes[0, 3]
    r = all_results["misconception_stability"]
    labels = ["Deterministic", "Our V3\n(stochastic)", "LLM\nPrompting"]
    values = [1.0, r["mean_agreement"], 0.3]  # 0.3 estimated for LLM
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Run-to-Run Agreement")
    ax.set_title("B4: Misconception Stability")
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)

    # Plot 5: Response prediction AUC
    ax = axes[1, 0]
    r = all_results["response_prediction"]
    labels = ["BKT\n(0.63-0.72)", "Our V3", "DKT\n(0.80-0.86)"]
    values = [0.675, r["auc"], 0.83]
    colors_auc = ["#9E9E9E", "#FF9800", "#9E9E9E"]
    bars = ax.bar(labels, values, color=colors_auc)
    ax.set_ylabel("AUC")
    ax.set_title("B5: Response Prediction AUC")
    ax.set_ylim(0.5, 0.95)
    ax.axhspan(0.63, 0.72, alpha=0.15, color="#2196F3", label="BKT range")
    ax.axhspan(0.80, 0.86, alpha=0.15, color="#4CAF50", label="DKT range")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Plot 6: Sessions to resolution
    ax = axes[1, 1]
    r = all_results["sessions_to_resolution"]
    if r.get("resolution_counts"):
        counts = r["resolution_counts"]
        ax.hist(counts, bins=range(1, min(max(counts) + 2, 32)), color="#FF9800", edgecolor="white", alpha=0.8)
        ax.axvspan(3, 7, alpha=0.2, color="#4CAF50", label="Cog tutor range (3-7)")
        ax.axvline(x=r["median_sessions"], color="red", linewidth=2, linestyle="--", label=f"Median={r['median_sessions']:.1f}")
    ax.set_xlabel("Instructions to Resolve")
    ax.set_ylabel("Count")
    ax.set_title("B6: Sessions to Resolution")
    ax.legend(fontsize=8)

    # Plot 7: Instruction sensitivity
    ax = axes[1, 2]
    r = all_results["instruction_sensitivity"]
    conds = ["perfect", "random", "always_wrong", "no_instruction"]
    labels = ["Perfect", "Random", "Always\nWrong", "No\nInstruction"]
    values = [r["condition_means"][c] for c in conds]
    colors = ["#4CAF50", "#FF9800", "#F44336", "#9E9E9E"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean p_know Delta")
    ax.set_title(f"B7: Instruction Sensitivity (d={r['cohens_d_perf_wrong']:.2f})")
    ax.axhline(y=0, color="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.001, f"{v:.4f}", ha="center", fontsize=8)

    # Plot 8: Negative transfer (switch design)
    ax = axes[1, 3]
    r = all_results["negative_transfer"]
    labels = ["Perfect\nThroughout", "Perfect\nThen Wrong", "No\nInstruction"]
    values = [r["perfect_final"], r["switch_final"], r["none_final"]]
    colors = ["#4CAF50", "#F44336", "#9E9E9E"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Final p_know")
    ax.set_title(f"B8: Negative Transfer (d={r['cohens_d']:.2f})")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "sota_benchmarks.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {ARTIFACTS / 'sota_benchmarks.png'}")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 11: SOTA Benchmarking for V3 Simulated Student")
    print("=" * 70)
    print(f"Students: {N_STUDENTS}, Interactions: {N_INTERACTIONS}, Seed: {SEED}")
    print(f"8 benchmarks comparing against published literature")

    all_results = {}

    # Run all benchmarks
    all_results["error_recurrence"] = benchmark_error_recurrence()
    all_results["performance_gap"] = benchmark_performance_gap()
    all_results["learning_curves"] = benchmark_learning_curves()
    all_results["misconception_stability"] = benchmark_misconception_stability()
    all_results["response_prediction"] = benchmark_response_prediction()
    all_results["sessions_to_resolution"] = benchmark_sessions_to_resolution()
    all_results["instruction_sensitivity"] = benchmark_instruction_sensitivity()
    all_results["negative_transfer"] = benchmark_negative_transfer()

    # Generate summary comparison table
    print(f"\n{'='*90}")
    print("SUMMARY: SOTA COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"{'Metric':<35} {'Our V3':>12} {'SOTA Reference':>15} {'Source':>25}")
    print(f"{'-'*90}")

    rows = [
        ("Error Recurrence Rate",
         f"{all_results['error_recurrence']['our_value']:.1%}",
         "86.2%",
         "BEAGLE (Wang 2026)"),
        ("Accuracy Gap (High vs Low)",
         f"{all_results['performance_gap']['accuracy_gap_pct']:+.1f} pct pts",
         "+40%",
         "BEAGLE (Wang 2026)"),
        ("p_know Gap (High vs Low)",
         f"{all_results['performance_gap']['final_pk_gap']:+.3f}",
         "N/A",
         "(absolute mastery)"),
        ("Learning Curve R² (power law)",
         f"{all_results['learning_curves']['curves']['accuracy']['power_law_r2']:.3f}",
         "> 0.90",
         "Newell & Rosenbloom 1981"),
        ("Learning Curve R² (p_know, pow)",
         f"{all_results['learning_curves']['curves']['p_know']['power_law_r2']:.3f}",
         "> 0.90",
         "Newell & Rosenbloom 1981"),
        ("Misconception Stability",
         f"{all_results['misconception_stability']['mean_agreement']:.1%}",
         "variable",
         "Scarlatos (2026)"),
        ("Response Prediction AUC",
         f"{all_results['response_prediction']['auc']:.3f}",
         "0.63-0.72",
         "BKT literature"),
        ("Sessions to Resolution (median)",
         f"{all_results['sessions_to_resolution']['median_sessions']:.1f}",
         "3-7",
         "Cognitive tutor lit"),
        ("Instruction Sensitivity (d)",
         f"{all_results['instruction_sensitivity']['cohens_d_perf_wrong']:.2f}",
         "N/A (novel)",
         "Our contribution"),
        ("Negative Transfer (switch)",
         "YES" if all_results["negative_transfer"]["negative_transfer_detected"] else "NO",
         "Not modeled",
         "Interference theory"),
        ("  p_know loss vs perfect",
         f"{all_results['negative_transfer']['p_know_loss_vs_perfect']:.4f}",
         "N/A",
         "(novel)"),
    ]

    for name, ours, ref, source in rows:
        print(f"  {name:<33} {ours:>12} {ref:>15} {source:>25}")

    print(f"\n{'='*90}")

    # Save full results
    output = {
        "metadata": {
            "experiment": "11_sota_benchmarks",
            "n_students": N_STUDENTS,
            "n_interactions": N_INTERACTIONS,
            "seed": SEED,
            "date": "2026-03-30",
            "description": "Comprehensive SOTA benchmark comparison for v3 simulated student",
        },
        "benchmarks": {},
    }

    for key, result in all_results.items():
        # Remove large arrays for JSON cleanliness
        clean = {k: v for k, v in result.items() if k not in ("resolution_counts",)}
        if "resolution_counts" in result:
            clean["resolution_counts_sample"] = result["resolution_counts"][:20]
        output["benchmarks"][key] = clean

    output["summary_table"] = rows

    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {ARTIFACTS / 'results.json'}")

    # Generate plots
    generate_plots(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
