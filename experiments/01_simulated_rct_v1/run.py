"""Experiment 01: Simulated RCT v1 (BKT-coupled assessment).

Reproducibility runner. Executes the original simulated RCT, saves results JSON
and generates summary plots in the artifacts/ folder.

Usage:
    cd <project_root>
    python experiments/01_simulated_rct_v1/run.py

Outputs:
    experiments/01_simulated_rct_v1/artifacts/results.json
    experiments/01_simulated_rct_v1/artifacts/mastery_gains.png
    experiments/01_simulated_rct_v1/artifacts/effect_sizes.png
    experiments/01_simulated_rct_v1/artifacts/sensitivity.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulated_rct import run_rct, run_sensitivity_analysis, generate_interpretation
from simulated_student import generate_students, describe_population

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SEED = 42
N_PER_CONDITION = 500
N_INTERACTIONS = 40


def save_results() -> dict:
    """Run the full v1 RCT and save results JSON."""
    from simulated_rct import N_STUDENTS_PER_CONDITION, N_INTERACTIONS as NI, SEED as S

    rct_results = run_rct(
        n_per_condition=N_PER_CONDITION,
        n_interactions=N_INTERACTIONS,
        seed=SEED,
    )

    sensitivity = run_sensitivity_analysis(
        n_per_condition=200,
        n_interactions=N_INTERACTIONS,
        seed=SEED,
    )

    students = generate_students(n=100, seed=SEED)
    pop_desc = describe_population(students)

    full_results = {
        "metadata": {
            "experiment": "01_simulated_rct_v1",
            "version": "v1",
            "n_per_condition": N_PER_CONDITION,
            "n_interactions": N_INTERACTIONS,
            "seed": SEED,
            "strategies": ["adaptive", "random", "fixed_sequence", "no_remediation"],
            "framework": "Tier 2: Learning-Enabled Misconception-Aware Simulated Students",
            "primary_metric": "internal_bkt_mastery_gain",
        },
        "population": pop_desc,
        "rct": rct_results,
        "sensitivity": sensitivity,
        "interpretation": generate_interpretation(rct_results, sensitivity),
    }

    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"Results saved to {ARTIFACTS / 'results.json'}")
    return full_results


def plot_mastery_gains(results: dict) -> None:
    """Bar chart of mean mastery gain per condition."""
    conditions = results["rct"]["conditions"]
    strategies = ["adaptive", "random", "fixed_sequence", "no_remediation"]
    labels = ["Adaptive", "Random", "Fixed\nSequence", "No\nRemediation"]
    gains = [conditions[s]["mean_mastery_gain"] for s in strategies]
    stds = [conditions[s]["std_mastery_gain"] for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2563eb", "#6b7280", "#6b7280", "#6b7280"]
    bars = ax.bar(labels, gains, yerr=stds, capsize=5, color=colors, edgecolor="white")
    ax.set_ylabel("Mean Mastery Gain (BKT p_know)")
    ax.set_title("Experiment 01: Mastery Gains by Condition (v1)")
    ax.set_ylim(0, max(gains) * 1.4)

    for bar, gain in zip(bars, gains):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{gain:.3f}", ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "mastery_gains.png", dpi=150)
    plt.close()
    print(f"Saved {ARTIFACTS / 'mastery_gains.png'}")


def plot_effect_sizes(results: dict) -> None:
    """Forest plot of Cohen's d with 95% CIs."""
    comparisons = results["rct"]["comparisons"]
    labels = []
    ds = []
    ci_lows = []
    ci_highs = []

    for key in ["adaptive_vs_random", "adaptive_vs_fixed_sequence", "adaptive_vs_no_remediation"]:
        mg = comparisons[key]["mastery_gain"]
        label = key.replace("adaptive_vs_", "vs ").replace("_", " ").title()
        labels.append(label)
        ds.append(mg["cohens_d"])
        ci_lows.append(mg["ci_95_lower"])
        ci_highs.append(mg["ci_95_upper"])

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = range(len(labels))
    errors = [[d - lo for d, lo in zip(ds, ci_lows)],
              [hi - d for d, hi in zip(ds, ci_highs)]]

    ax.barh(y_pos, ds, xerr=errors, capsize=5, color="#2563eb", height=0.5, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Experiment 01: Effect Sizes (v1, BKT-coupled)")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.axvline(x=0.2, color="gray", linewidth=0.5, linestyle="--", label="Small (0.2)")
    ax.axvline(x=0.5, color="gray", linewidth=0.5, linestyle=":", label="Medium (0.5)")
    ax.legend(loc="lower right", fontsize=8)

    for i, d in enumerate(ds):
        ax.text(d + 0.02, i, f"d={d:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "effect_sizes.png", dpi=150)
    plt.close()
    print(f"Saved {ARTIFACTS / 'effect_sizes.png'}")


def plot_sensitivity(results: dict) -> None:
    """Line plot of adaptive effect size across BKT parameter scales."""
    sensitivity = results["sensitivity"]
    scales = []
    d_vs_random = []
    d_vs_no_remed = []

    for key, val in sorted(sensitivity.items()):
        if key == "robust":
            continue
        scale = float(key)
        scales.append(scale)
        d_vs_random.append(val["effect_sizes"].get("adaptive_vs_random", 0))
        d_vs_no_remed.append(val["effect_sizes"].get("adaptive_vs_no_remediation", 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scales, d_vs_random, "o-", label="vs Random", color="#2563eb")
    ax.plot(scales, d_vs_no_remed, "s-", label="vs No Remediation", color="#dc2626")
    ax.axhline(y=0.2, color="gray", linewidth=0.5, linestyle="--", label="Small effect (0.2)")
    ax.set_xlabel("BKT Learning Rate Scale")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Experiment 01: Sensitivity Analysis (v1)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "sensitivity.png", dpi=150)
    plt.close()
    print(f"Saved {ARTIFACTS / 'sensitivity.png'}")


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  EXPERIMENT 01: Simulated RCT v1")
    print("=" * 60)

    results = save_results()
    plot_mastery_gains(results)
    plot_effect_sizes(results)
    plot_sensitivity(results)

    print("\nAll artifacts saved to experiments/01_simulated_rct_v1/artifacts/")


if __name__ == "__main__":
    main()
