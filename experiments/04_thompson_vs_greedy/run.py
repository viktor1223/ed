"""Experiment 04: Thompson Sampling vs Greedy Modality Selection.

Monte Carlo simulation comparing three intervention selection policies:
  1. Thompson sampling (Beta posterior draws)
  2. Greedy (always pick highest observed success rate)
  3. Uniform random

Each simulated student has a latent modality preference drawn from
Dirichlet(1,...,1). Resolution probability when assigned modality m
equals preference_vector[m].

Usage:
    cd <project_root>
    python experiments/04_thompson_vs_greedy/run.py

Outputs:
    experiments/04_thompson_vs_greedy/artifacts/results.json
    experiments/04_thompson_vs_greedy/artifacts/resolution_curves.png
    experiments/04_thompson_vs_greedy/artifacts/regret_curves.png
    experiments/04_thompson_vs_greedy/artifacts/scalability.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SEED = 42

MODALITIES = ["visual", "concrete", "pattern", "verbal", "peer"]


# ─── Student Generator ────────────────────────────────────────────────────────

def generate_preference_vectors(
    n_students: int,
    n_modalities: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw latent modality preferences from Dirichlet(1,...,1).

    Returns shape (n_students, n_modalities) with rows summing to ~1.
    Values represent resolution probability per modality.
    Rescaled so max preference ~ 0.85 and min ~ 0.15 (realistic range).
    """
    raw = rng.dirichlet(np.ones(n_modalities), size=n_students)
    # Rescale from [0,1] to [0.15, 0.85]
    prefs = 0.15 + raw * 0.70
    return prefs


# ─── Policies ─────────────────────────────────────────────────────────────────

def thompson_select(successes: np.ndarray, failures: np.ndarray, rng: np.random.Generator) -> int:
    """Thompson sampling: draw from Beta(s+1, f+1) and pick argmax."""
    draws = rng.beta(successes + 1, failures + 1)
    return int(np.argmax(draws))


def greedy_select(successes: np.ndarray, failures: np.ndarray, rng: np.random.Generator) -> int:
    """Greedy: pick the modality with highest observed success rate (break ties randomly)."""
    total = successes + failures
    rates = np.where(total > 0, successes / total, 0.5)
    max_rate = rates.max()
    best = np.where(rates == max_rate)[0]
    return int(rng.choice(best))


def uniform_select(n_modalities: int, rng: np.random.Generator) -> int:
    """Uniform random selection."""
    return int(rng.integers(n_modalities))


def oracle_select(preferences: np.ndarray) -> int:
    """Oracle: always pick the student's true best modality."""
    return int(np.argmax(preferences))


# ─── Simulation ───────────────────────────────────────────────────────────────

def simulate_student(
    preferences: np.ndarray,
    n_interactions: int,
    policy: str,
    rng: np.random.Generator,
) -> dict:
    """Simulate one student through n_interactions.

    Returns per-interaction log and cumulative stats.
    """
    n_modalities = len(preferences)
    successes = np.zeros(n_modalities)
    failures = np.zeros(n_modalities)

    cumulative_resolved = 0
    oracle_resolved = 0
    log = []

    for t in range(n_interactions):
        if policy == "thompson":
            action = thompson_select(successes, failures, rng)
        elif policy == "greedy":
            action = greedy_select(successes, failures, rng)
        elif policy == "uniform":
            action = uniform_select(n_modalities, rng)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Resolve with probability = preference[action]
        resolved = rng.random() < preferences[action]
        if resolved:
            successes[action] += 1
            cumulative_resolved += 1
        else:
            failures[action] += 1

        # Oracle outcome
        oracle_action = oracle_select(preferences)
        oracle_resolved_this = rng.random() < preferences[oracle_action]
        if oracle_resolved_this:
            oracle_resolved += 1

        # Check convergence: does the policy's best estimate match the true best?
        total = successes + failures
        estimated_rates = np.where(total > 0, successes / total, 0.5)
        estimated_best = int(np.argmax(estimated_rates))
        true_best = int(np.argmax(preferences))
        converged = estimated_best == true_best

        log.append({
            "t": t,
            "action": action,
            "resolved": bool(resolved),
            "cumulative_resolution_rate": cumulative_resolved / (t + 1),
            "oracle_resolution_rate": oracle_resolved / (t + 1),
            "converged": converged,
        })

    return {
        "total_resolved": cumulative_resolved,
        "resolution_rate": cumulative_resolved / n_interactions,
        "oracle_rate": oracle_resolved / n_interactions,
        "regret": (oracle_resolved - cumulative_resolved) / n_interactions,
        "log": log,
    }


def run_experiment(
    n_students: int,
    n_interactions: int,
    n_modalities: int,
    seed: int,
) -> dict:
    """Run the full experiment for all three policies."""
    rng = np.random.default_rng(seed)
    preferences = generate_preference_vectors(n_students, n_modalities, rng)

    results = {}
    for policy in ["thompson", "greedy", "uniform"]:
        print(f"  Running policy: {policy} ({n_students} students, {n_modalities} modalities)...")
        policy_rng = np.random.default_rng(seed + hash(policy) % 2**31)

        student_results = []
        for i in range(n_students):
            sr = simulate_student(preferences[i], n_interactions, policy, policy_rng)
            student_results.append(sr)

        # Aggregate
        resolution_rates = [s["resolution_rate"] for s in student_results]
        regrets = [s["regret"] for s in student_results]

        # Per-timestep averages across students
        per_t_resolution = np.zeros(n_interactions)
        per_t_regret = np.zeros(n_interactions)
        convergence_counts = np.zeros(n_interactions)

        for sr in student_results:
            for entry in sr["log"]:
                t = entry["t"]
                per_t_resolution[t] += entry["cumulative_resolution_rate"]
                per_t_regret[t] += entry["oracle_resolution_rate"] - entry["cumulative_resolution_rate"]
                convergence_counts[t] += int(entry["converged"])

        per_t_resolution /= n_students
        per_t_regret /= n_students
        convergence_counts /= n_students

        # Convergence speed: median interaction where policy converges
        convergence_times = []
        for sr in student_results:
            for entry in sr["log"]:
                if entry["converged"]:
                    convergence_times.append(entry["t"])
                    break
            else:
                convergence_times.append(n_interactions)

        results[policy] = {
            "mean_resolution_rate": round(float(np.mean(resolution_rates)), 4),
            "std_resolution_rate": round(float(np.std(resolution_rates)), 4),
            "mean_regret": round(float(np.mean(regrets)), 4),
            "median_convergence": int(np.median(convergence_times)),
            "per_timestep_resolution": per_t_resolution.tolist(),
            "per_timestep_regret": per_t_regret.tolist(),
            "per_timestep_convergence_frac": convergence_counts.tolist(),
        }

    return results


# ─── Scalability ──────────────────────────────────────────────────────────────

def run_scalability(seed: int) -> list[dict]:
    """Vary number of modalities from 3 to 10."""
    results = []
    for n_mod in [3, 4, 5, 6, 8, 10]:
        print(f"\nScalability: {n_mod} modalities...")
        exp = run_experiment(
            n_students=500,
            n_interactions=50,
            n_modalities=n_mod,
            seed=seed,
        )
        for policy, data in exp.items():
            results.append({
                "n_modalities": n_mod,
                "policy": policy,
                "mean_resolution_rate": data["mean_resolution_rate"],
                "mean_regret": data["mean_regret"],
                "median_convergence": data["median_convergence"],
            })
    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_resolution_curves(results: dict, n_interactions: int) -> None:
    """Cumulative resolution rate over time for each policy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"thompson": "#2196F3", "greedy": "#FF9800", "uniform": "#9E9E9E"}
    labels = {"thompson": "Thompson Sampling", "greedy": "Greedy", "uniform": "Uniform Random"}

    for policy, data in results.items():
        ts = list(range(n_interactions))
        ax.plot(ts, data["per_timestep_resolution"], label=labels[policy],
                color=colors[policy], linewidth=2)

    ax.set_xlabel("Interaction Number")
    ax.set_ylabel("Cumulative Resolution Rate")
    ax.set_title("Modality Selection: Cumulative Resolution Rate by Policy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "resolution_curves.png", dpi=150)
    plt.close()


def plot_regret_curves(results: dict, n_interactions: int) -> None:
    """Per-timestep regret (vs oracle) for each policy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"thompson": "#2196F3", "greedy": "#FF9800", "uniform": "#9E9E9E"}
    labels = {"thompson": "Thompson Sampling", "greedy": "Greedy", "uniform": "Uniform Random"}

    for policy, data in results.items():
        ts = list(range(n_interactions))
        ax.plot(ts, data["per_timestep_regret"], label=labels[policy],
                color=colors[policy], linewidth=2)

    ax.set_xlabel("Interaction Number")
    ax.set_ylabel("Regret (Oracle Rate - Policy Rate)")
    ax.set_title("Cumulative Regret vs Oracle Policy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "regret_curves.png", dpi=150)
    plt.close()


def plot_scalability(scalability_results: list[dict]) -> None:
    """Resolution rate vs number of modalities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"thompson": "#2196F3", "greedy": "#FF9800", "uniform": "#9E9E9E"}
    labels = {"thompson": "Thompson Sampling", "greedy": "Greedy", "uniform": "Uniform Random"}

    for policy in ["thompson", "greedy", "uniform"]:
        subset = [r for r in scalability_results if r["policy"] == policy]
        n_mods = [r["n_modalities"] for r in subset]
        rates = [r["mean_resolution_rate"] for r in subset]
        regrets = [r["mean_regret"] for r in subset]

        axes[0].plot(n_mods, rates, "o-", label=labels[policy], color=colors[policy], linewidth=2)
        axes[1].plot(n_mods, regrets, "o-", label=labels[policy], color=colors[policy], linewidth=2)

    axes[0].set_xlabel("Number of Modalities")
    axes[0].set_ylabel("Mean Resolution Rate")
    axes[0].set_title("Resolution Rate vs Action Space Size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Number of Modalities")
    axes[1].set_ylabel("Mean Regret")
    axes[1].set_title("Regret vs Action Space Size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "scalability.png", dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 04: Thompson Sampling vs Greedy Modality Selection")
    print("=" * 60)

    # Main experiment: 5 modalities, 1000 students, 50 interactions
    print("\n--- Main Experiment ---")
    main_results = run_experiment(
        n_students=1000,
        n_interactions=50,
        n_modalities=5,
        seed=SEED,
    )

    # Print summary
    print("\nResults (5 modalities, 1000 students, 50 interactions):")
    print(f"  {'Policy':<20} {'Resolution Rate':>18} {'Regret':>10} {'Convergence':>14}")
    print(f"  {'-'*62}")
    for policy, data in main_results.items():
        print(f"  {policy:<20} {data['mean_resolution_rate']:>14.4f} +/- {data['std_resolution_rate']:.4f}"
              f"  {data['mean_regret']:>8.4f}  {data['median_convergence']:>10d}")

    # Scalability
    print("\n--- Scalability Experiment ---")
    scalability_results = run_scalability(SEED)

    # Assemble output
    full_output = {
        "metadata": {
            "experiment": "04_thompson_vs_greedy",
            "main_n_students": 1000,
            "main_n_interactions": 50,
            "main_n_modalities": 5,
            "seed": SEED,
        },
        "main_results": {
            policy: {k: v for k, v in data.items() if k not in ("per_timestep_resolution", "per_timestep_regret", "per_timestep_convergence_frac")}
            for policy, data in main_results.items()
        },
        "scalability": scalability_results,
    }

    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nResults saved to {ARTIFACTS / 'results.json'}")

    # Plots
    print("\nGenerating plots...")
    plot_resolution_curves(main_results, 50)
    plot_regret_curves(main_results, 50)
    plot_scalability(scalability_results)
    print("Plots saved.")

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 04 SUMMARY")
    print("=" * 60)
    for policy, data in main_results.items():
        print(f"\n{policy.upper()}:")
        print(f"  Mean resolution rate: {data['mean_resolution_rate']:.4f} +/- {data['std_resolution_rate']:.4f}")
        print(f"  Mean regret vs oracle: {data['mean_regret']:.4f}")
        print(f"  Median convergence:    {data['median_convergence']} interactions")


if __name__ == "__main__":
    main()
