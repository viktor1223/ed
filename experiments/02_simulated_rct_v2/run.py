"""Experiment 02: Simulated RCT v2 (Decoupled Assessment).

Reproducibility runner. Executes the redesigned simulated RCT with held-out
test-based evaluation, saves results JSON and generates summary plots.

Usage:
    cd <project_root>
    python experiments/02_simulated_rct_v2/run.py

Outputs:
    experiments/02_simulated_rct_v2/artifacts/results.json
    experiments/02_simulated_rct_v2/artifacts/test_score_gains.png
    experiments/02_simulated_rct_v2/artifacts/effect_sizes.png
    experiments/02_simulated_rct_v2/artifacts/sensitivity_heatmap.png
    experiments/02_simulated_rct_v2/artifacts/v1_v2_comparison.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulated_rct_v2 import run_rct, run_sensitivity_analysis, generate_interpretation
from simulated_student import generate_students, describe_population

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SEED = 42
N_PER_CONDITION = 500
N_INTERACTIONS = 40


def save_results() -> dict:
    """Run the full v2 RCT and save results JSON."""
    rct_results = run_rct(
        n_per_condition=N_PER_CONDITION,
        n_interactions=N_INTERACTIONS,
        seed=SEED,
        targeted_resolution=0.50,
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
            "experiment": "02_simulated_rct_v2",
            "version": "v2",
            "n_per_condition": N_PER_CONDITION,
            "n_interactions": N_INTERACTIONS,
            "seed": SEED,
            "strategies": ["adaptive", "random", "fixed_sequence", "no_remediation"],
            "framework": "Tier 2: Decoupled Assessment with Held-Out Test Set",
            "primary_metric": "held_out_test_score_gain",
            "targeted_resolution": 0.50,
            "max_consecutive_same_concept": 6,
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


def plot_test_score_gains(results: dict) -> None:
    """Bar chart of mean held-out test score gain per condition."""
    conditions = results["rct"]["conditions"]
    strategies = ["adaptive", "random", "fixed_sequence", "no_remediation"]
    labels = ["Adaptive", "Random", "Fixed\nSequence", "No\nRemediation"]
    gains = [conditions[s]["mean_test_gain"] for s in strategies]
    stds = [conditions[s]["std_test_gain"] for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2563eb", "#6b7280", "#6b7280", "#6b7280"]
    bars = ax.bar(labels, gains, yerr=stds, capsize=5, color=colors, edgecolor="white")
    ax.set_ylabel("Mean Test Score Gain (held-out problems)")
    ax.set_title("Experiment 02: Test Score Gains by Condition (v2)")
    ax.set_ylim(0, max(gains) * 1.4)

    for bar, gain in zip(bars, gains):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{gain:.3f}", ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "test_score_gains.png", dpi=150)
    plt.close()
    print(f"Saved {ARTIFACTS / 'test_score_gains.png'}")


def plot_effect_sizes(results: dict) -> None:
    """Forest plot of Cohen's d with 95% CIs on held-out test metric."""
    comparisons = results["rct"]["comparisons"]
    labels = []
    ds = []
    ci_lows = []
    ci_highs = []

    for key in ["adaptive_vs_random", "adaptive_vs_fixed_sequence", "adaptive_vs_no_remediation"]:
        mg = comparisons[key]["test_score_gain"]
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
    ax.set_xlabel("Cohen's d (held-out test score gain)")
    ax.set_title("Experiment 02: Effect Sizes (v2, Decoupled)")
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


def plot_sensitivity_heatmap(results: dict) -> None:
    """Heatmap of adaptive vs no_remediation effect size across parameter grid."""
    sensitivity = results["sensitivity"]

    bkt_scales = sorted(set(
        v["bkt_scale"] for k, v in sensitivity.items() if k != "robust"
    ))
    res_rates = sorted(set(
        v["targeted_resolution"] for k, v in sensitivity.items() if k != "robust"
    ))

    # Build matrix: rows = bkt_scales, cols = res_rates
    matrix = np.zeros((len(bkt_scales), len(res_rates)))
    for key, val in sensitivity.items():
        if key == "robust":
            continue
        i = bkt_scales.index(val["bkt_scale"])
        j = res_rates.index(val["targeted_resolution"])
        d = val["effect_sizes"].get("adaptive_vs_no_remediation", 0)
        matrix[i, j] = d

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.6)

    ax.set_xticks(range(len(res_rates)))
    ax.set_xticklabels([f"{r:.2f}" for r in res_rates])
    ax.set_yticks(range(len(bkt_scales)))
    ax.set_yticklabels([f"{s:.1f}x" for s in bkt_scales])
    ax.set_xlabel("Targeted Resolution Rate")
    ax.set_ylabel("BKT Learning Rate Scale")
    ax.set_title("Adaptive vs No Remediation: Cohen's d")

    for i in range(len(bkt_scales)):
        for j in range(len(res_rates)):
            color = "white" if matrix[i, j] < 0.15 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color=color)

    plt.colorbar(im, ax=ax, label="Cohen's d")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "sensitivity_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved {ARTIFACTS / 'sensitivity_heatmap.png'}")


def plot_v1_v2_comparison() -> None:
    """Side-by-side comparison of v1 and v2 effect sizes."""
    v1_path = Path(__file__).resolve().parent.parent / "01_simulated_rct_v1" / "artifacts" / "results.json"

    if not v1_path.exists():
        print(f"Skipping v1/v2 comparison (v1 results not found at {v1_path})")
        return

    with open(v1_path) as f:
        v1 = json.load(f)

    v2_path = ARTIFACTS / "results.json"
    with open(v2_path) as f:
        v2 = json.load(f)

    comparisons_list = ["adaptive_vs_random", "adaptive_vs_fixed_sequence", "adaptive_vs_no_remediation"]
    labels = ["vs Random", "vs Fixed Seq", "vs No Remed"]

    v1_ds = [v1["rct"]["comparisons"][c]["mastery_gain"]["cohens_d"] for c in comparisons_list]
    v2_ds = [v2["rct"]["comparisons"][c]["test_score_gain"]["cohens_d"] for c in comparisons_list]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, v1_ds, width, label="v1 (BKT-coupled)", color="#93c5fd")
    bars2 = ax.bar(x + width / 2, v2_ds, width, label="v2 (Held-out test)", color="#2563eb")

    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Size Comparison: v1 (inflated) vs v2 (decoupled)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.axhline(y=0.2, color="gray", linewidth=0.5, linestyle="--")
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle=":")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "v1_v2_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {ARTIFACTS / 'v1_v2_comparison.png'}")


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  EXPERIMENT 02: Simulated RCT v2 (Decoupled Assessment)")
    print("=" * 60)

    results = save_results()
    plot_test_score_gains(results)
    plot_effect_sizes(results)
    plot_sensitivity_heatmap(results)
    plot_v1_v2_comparison()

    print("\nAll artifacts saved to experiments/02_simulated_rct_v2/artifacts/")


if __name__ == "__main__":
    main()
