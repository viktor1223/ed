"""Experiment 06: Escalation State Machine Convergence Analysis.

Two-part analysis:
  1. Analytical: model the escalation state machine as an absorbing Markov
     chain and compute absorption probabilities + expected steps.
  2. Simulation: Monte Carlo validation (10 000 episodes) confirming the
     analytical results.

Sensitivity sweeps:
  - Per-attempt resolution probability: 0.10 to 0.90
  - Max modality attempts before escalation: 2 to 8

Usage:
    cd <project_root>
    python experiments/06_escalation_convergence/run.py

Outputs:
    experiments/06_escalation_convergence/artifacts/results.json
    experiments/06_escalation_convergence/artifacts/absorption_probabilities.png
    experiments/06_escalation_convergence/artifacts/expected_steps.png
    experiments/06_escalation_convergence/artifacts/attempts_vs_resolution.png
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


# ─── Escalation State Machine ────────────────────────────────────────────────
#
# States (transient):
#   0: detected
#   1: intervention_assigned (1st modality)
#   2: modality_switched (2nd modality)
#   3: prerequisite_check
#   4: prereq_remediation
#   ... additional modality attempts depending on max_attempts
#
# Absorbing states:
#   A: resolved
#   B: teacher_conference / escalated
#
# Transition logic per modality attempt:
#   - With probability p_resolve: -> resolved (absorbing)
#   - With probability (1 - p_resolve): -> next modality or escalation
#   - At prerequisite_check with p_prereq_issue: -> prereq_remediation
#   - prereq_remediation -> back to intervention_assigned with new modality
#   - After max_attempts modalities exhausted: -> teacher_conference (absorbing)

def build_transition_matrix(
    p_resolve: float,
    max_attempts: int = 4,
    p_prereq_issue: float = 0.30,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the transition matrix for the escalation Markov chain.

    Transient states: detected, attempt_1, attempt_2, ..., attempt_{max},
                      prereq_check, prereq_remediation
    Absorbing states: resolved, teacher_conference

    Returns: (Q matrix for transient states, transient_names, absorbing_names)
    Also returns R matrix (transient -> absorbing transitions).
    """
    # State layout:
    # 0: detected
    # 1..max_attempts: attempt_i (modality i being tried)
    # max_attempts+1: prereq_check
    # max_attempts+2: prereq_remediation
    # Absorbing: resolved (A0), teacher_conference (A1)

    n_transient = max_attempts + 3  # detected + attempts + prereq_check + prereq_remediation
    transient_names = ["detected"]
    for i in range(1, max_attempts + 1):
        transient_names.append(f"attempt_{i}")
    transient_names.append("prereq_check")
    transient_names.append("prereq_remediation")
    absorbing_names = ["resolved", "teacher_conference"]

    Q = np.zeros((n_transient, n_transient))  # transient -> transient
    R = np.zeros((n_transient, 2))             # transient -> absorbing

    # detected (0) -> attempt_1 (1): always
    Q[0, 1] = 1.0

    # attempt_i: resolve with p_resolve, else move forward
    for i in range(1, max_attempts + 1):
        # Resolve
        R[i, 0] = p_resolve  # -> resolved

        if i < max_attempts:
            if i == 2:
                # After 2nd attempt, go to prerequisite check
                Q[i, max_attempts + 1] = (1 - p_resolve)  # -> prereq_check
            else:
                # Move to next attempt
                Q[i, i + 1] = (1 - p_resolve)
        else:
            # Last attempt: if not resolved, escalate
            R[i, 1] = (1 - p_resolve)  # -> teacher_conference

    # prereq_check: with p_prereq_issue -> prereq_remediation, else next attempt
    prereq_idx = max_attempts + 1
    remediation_idx = max_attempts + 2

    Q[prereq_idx, remediation_idx] = p_prereq_issue
    # If no prereq issue and we have attempts left (attempt_3)
    if max_attempts >= 3:
        Q[prereq_idx, 3] = (1 - p_prereq_issue)
    else:
        R[prereq_idx, 1] = (1 - p_prereq_issue)  # escalate

    # prereq_remediation -> back to attempt after prereq (attempt_3 or next available)
    if max_attempts >= 3:
        Q[remediation_idx, 3] = 1.0
    else:
        R[remediation_idx, 1] = 1.0  # escalate

    return Q, R, transient_names, absorbing_names


def analyze_markov_chain(
    Q: np.ndarray,
    R: np.ndarray,
) -> dict:
    """Compute absorption probabilities and expected steps.

    N = (I - Q)^{-1}  (fundamental matrix)
    B = N * R          (absorption probabilities)
    t = N * 1          (expected steps to absorption)
    """
    n = Q.shape[0]
    I = np.eye(n)
    N = np.linalg.inv(I - Q)  # fundamental matrix
    B = N @ R                  # absorption probabilities
    t = N @ np.ones(n)         # expected steps to absorption

    return {
        "fundamental_matrix": N,
        "absorption_probs": B,        # B[i, j] = P(absorb in state j | start in i)
        "expected_steps": t,           # t[i] = E[steps to absorb | start in i]
    }


# ─── Monte Carlo Simulation ──────────────────────────────────────────────────

def simulate_episode(
    p_resolve: float,
    max_attempts: int,
    p_prereq_issue: float,
    rng: np.random.Generator,
) -> dict:
    """Simulate one misconception episode through the state machine."""
    state = "detected"
    steps = 0
    modalities_tried = 0
    history = [state]

    while True:
        steps += 1

        if state == "detected":
            state = "attempt"
            modalities_tried = 1
            history.append(f"attempt_{modalities_tried}")

        elif state == "attempt":
            # Try to resolve
            if rng.random() < p_resolve:
                return {
                    "terminal": "resolved",
                    "steps": steps,
                    "modalities_tried": modalities_tried,
                    "history": history,
                }

            # Not resolved
            if modalities_tried == 2:
                # Prerequisite check
                state = "prereq_check"
                history.append("prereq_check")
            elif modalities_tried >= max_attempts:
                # Escalate
                return {
                    "terminal": "teacher_conference",
                    "steps": steps,
                    "modalities_tried": modalities_tried,
                    "history": history,
                }
            else:
                modalities_tried += 1
                history.append(f"attempt_{modalities_tried}")

        elif state == "prereq_check":
            if rng.random() < p_prereq_issue:
                state = "prereq_remediation"
                history.append("prereq_remediation")
            else:
                modalities_tried += 1
                state = "attempt"
                history.append(f"attempt_{modalities_tried}")

        elif state == "prereq_remediation":
            steps += 1  # Remediation takes a step
            modalities_tried += 1
            state = "attempt"
            history.append(f"attempt_{modalities_tried}")

        if steps > 50:  # Safety limit
            return {
                "terminal": "teacher_conference",
                "steps": steps,
                "modalities_tried": modalities_tried,
                "history": history,
            }


def run_simulation(
    n_episodes: int,
    p_resolve: float,
    max_attempts: int,
    p_prereq_issue: float,
    seed: int,
) -> dict:
    """Run Monte Carlo simulation of n episodes."""
    rng = np.random.default_rng(seed)
    terminals = {"resolved": 0, "teacher_conference": 0}
    total_steps = []
    total_modalities = []

    for _ in range(n_episodes):
        result = simulate_episode(p_resolve, max_attempts, p_prereq_issue, rng)
        terminals[result["terminal"]] += 1
        total_steps.append(result["steps"])
        total_modalities.append(result["modalities_tried"])

    return {
        "p_resolved": terminals["resolved"] / n_episodes,
        "p_teacher_conference": terminals["teacher_conference"] / n_episodes,
        "expected_steps": float(np.mean(total_steps)),
        "expected_modalities": float(np.mean(total_modalities)),
        "steps_std": float(np.std(total_steps)),
    }


# ─── Sensitivity Sweep ───────────────────────────────────────────────────────

def run_resolution_sweep(
    p_values: list[float],
    max_attempts: int = 4,
    n_episodes: int = 10_000,
    seed: int = SEED,
) -> list[dict]:
    """Sweep over resolution probabilities, compute analytical and simulated results."""
    results = []
    for p in p_values:
        # Analytical
        Q, R, t_names, a_names = build_transition_matrix(p, max_attempts)
        analysis = analyze_markov_chain(Q, R)
        # Start state is "detected" (index 0)
        analytical_p_resolved = float(analysis["absorption_probs"][0, 0])
        analytical_p_escalated = float(analysis["absorption_probs"][0, 1])
        analytical_steps = float(analysis["expected_steps"][0])

        # Simulated
        sim = run_simulation(n_episodes, p, max_attempts, 0.30, seed)

        results.append({
            "p_resolve": p,
            "max_attempts": max_attempts,
            "analytical_p_resolved": round(analytical_p_resolved, 4),
            "analytical_p_escalated": round(analytical_p_escalated, 4),
            "analytical_expected_steps": round(analytical_steps, 2),
            "simulated_p_resolved": round(sim["p_resolved"], 4),
            "simulated_p_escalated": round(sim["p_teacher_conference"], 4),
            "simulated_expected_steps": round(sim["expected_steps"], 2),
            "simulated_expected_modalities": round(sim["expected_modalities"], 2),
        })

    return results


def run_attempts_sweep(
    attempt_values: list[int],
    p_values: list[float],
    n_episodes: int = 10_000,
    seed: int = SEED,
) -> list[dict]:
    """Sweep over max attempts allowed, for several resolution probabilities."""
    results = []
    for max_att in attempt_values:
        for p in p_values:
            sim = run_simulation(n_episodes, p, max_att, 0.30, seed)
            results.append({
                "max_attempts": max_att,
                "p_resolve": p,
                "p_resolved": round(sim["p_resolved"], 4),
                "expected_steps": round(sim["expected_steps"], 2),
            })
    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_absorption_probabilities(sweep_results: list[dict]) -> None:
    """Analytical vs simulated absorption probabilities."""
    p_vals = [r["p_resolve"] for r in sweep_results]
    an_resolved = [r["analytical_p_resolved"] for r in sweep_results]
    an_escalated = [r["analytical_p_escalated"] for r in sweep_results]
    sim_resolved = [r["simulated_p_resolved"] for r in sweep_results]
    sim_escalated = [r["simulated_p_escalated"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p_vals, an_resolved, "b-", linewidth=2, label="P(resolved) - analytical")
    ax.plot(p_vals, sim_resolved, "bo", markersize=5, label="P(resolved) - simulated")
    ax.plot(p_vals, an_escalated, "r-", linewidth=2, label="P(escalated) - analytical")
    ax.plot(p_vals, sim_escalated, "ro", markersize=5, label="P(escalated) - simulated")

    ax.set_xlabel("Per-Attempt Resolution Probability")
    ax.set_ylabel("Absorption Probability")
    ax.set_title("Escalation State Machine: Absorption Probabilities")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # Annotate the p=0.50 point
    for r in sweep_results:
        if abs(r["p_resolve"] - 0.50) < 0.01:
            ax.annotate(
                f"p=0.50: {r['analytical_p_resolved']:.1%} resolve",
                xy=(0.50, r["analytical_p_resolved"]),
                xytext=(0.55, r["analytical_p_resolved"] - 0.1),
                arrowprops=dict(arrowstyle="->"),
                fontsize=9,
            )
            break

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "absorption_probabilities.png", dpi=150)
    plt.close()


def plot_expected_steps(sweep_results: list[dict]) -> None:
    """Expected steps and modalities tried vs resolution probability."""
    p_vals = [r["p_resolve"] for r in sweep_results]
    an_steps = [r["analytical_expected_steps"] for r in sweep_results]
    sim_steps = [r["simulated_expected_steps"] for r in sweep_results]
    sim_modalities = [r["simulated_expected_modalities"] for r in sweep_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(p_vals, an_steps, "b-", linewidth=2, label="Expected steps (analytical)")
    axes[0].plot(p_vals, sim_steps, "bo", markersize=5, label="Expected steps (simulated)")
    axes[0].set_xlabel("Per-Attempt Resolution Probability")
    axes[0].set_ylabel("Expected Steps to Terminal State")
    axes[0].set_title("Expected Steps to Absorption")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(p_vals, sim_modalities, "g-o", linewidth=2, markersize=5)
    axes[1].set_xlabel("Per-Attempt Resolution Probability")
    axes[1].set_ylabel("Expected Modalities Tried")
    axes[1].set_title("Expected Modalities Before Resolution/Escalation")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "expected_steps.png", dpi=150)
    plt.close()


def plot_attempts_vs_resolution(attempts_results: list[dict]) -> None:
    """Heatmap-style plot: P(resolved) vs max_attempts and p_resolve."""
    attempts = sorted(set(r["max_attempts"] for r in attempts_results))
    p_vals = sorted(set(r["p_resolve"] for r in attempts_results))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(p_vals)))

    for i, p in enumerate(p_vals):
        subset = [r for r in attempts_results if r["p_resolve"] == p]
        subset.sort(key=lambda r: r["max_attempts"])
        att_vals = [r["max_attempts"] for r in subset]
        res_vals = [r["p_resolved"] for r in subset]
        ax.plot(att_vals, res_vals, "o-", color=colors[i], label=f"p_resolve={p:.2f}", linewidth=2)

    ax.set_xlabel("Max Modality Attempts Before Escalation")
    ax.set_ylabel("P(Resolved)")
    ax.set_title("Resolution Rate vs Escalation Threshold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "attempts_vs_resolution.png", dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 06: Escalation State Machine Convergence")
    print("=" * 60)

    # Resolution probability sweep
    print("\n--- Resolution Probability Sweep (max_attempts=4) ---")
    p_values = [round(0.10 + i * 0.05, 2) for i in range(17)]  # 0.10 to 0.90
    sweep_results = run_resolution_sweep(p_values, max_attempts=4, n_episodes=10_000, seed=SEED)

    print(f"\n{'p_resolve':>10} {'An.P(res)':>10} {'Sim.P(res)':>10} {'An.Steps':>10} {'Sim.Steps':>10} {'Modalities':>10}")
    print(f"{'-'*60}")
    for r in sweep_results:
        print(f"{r['p_resolve']:>10.2f} {r['analytical_p_resolved']:>10.4f} {r['simulated_p_resolved']:>10.4f} "
              f"{r['analytical_expected_steps']:>10.2f} {r['simulated_expected_steps']:>10.2f} "
              f"{r['simulated_expected_modalities']:>10.2f}")

    # Attempts sweep
    print("\n--- Max Attempts Sweep ---")
    attempt_values = [2, 3, 4, 5, 6, 8]
    sweep_p_values = [0.20, 0.35, 0.50, 0.65, 0.80]
    attempts_results = run_attempts_sweep(attempt_values, sweep_p_values, n_episodes=10_000, seed=SEED)

    # Assemble output
    full_output = {
        "metadata": {
            "experiment": "06_escalation_convergence",
            "n_episodes": 10_000,
            "p_prereq_issue": 0.30,
            "seed": SEED,
        },
        "resolution_sweep": sweep_results,
        "attempts_sweep": attempts_results,
    }

    with open(ARTIFACTS / "results.json", "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nResults saved to {ARTIFACTS / 'results.json'}")

    # Plots
    print("\nGenerating plots...")
    plot_absorption_probabilities(sweep_results)
    plot_expected_steps(sweep_results)
    plot_attempts_vs_resolution(attempts_results)
    print("Plots saved.")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 06 SUMMARY")
    print("=" * 60)

    # Key findings at p=0.50
    p50 = next(r for r in sweep_results if abs(r["p_resolve"] - 0.50) < 0.01)
    print(f"\nAt p_resolve=0.50 (v2 RCT setting), max_attempts=4:")
    print(f"  P(resolved):              {p50['analytical_p_resolved']:.4f} (analytical)")
    print(f"  P(teacher_conference):    {p50['analytical_p_escalated']:.4f} (analytical)")
    print(f"  Expected steps:           {p50['analytical_expected_steps']:.2f}")
    print(f"  Expected modalities:      {p50['simulated_expected_modalities']:.2f}")

    # Analytical-simulation agreement
    max_diff = max(
        abs(r["analytical_p_resolved"] - r["simulated_p_resolved"])
        for r in sweep_results
    )
    print(f"\n  Max analytical-simulation P(resolved) discrepancy: {max_diff:.4f}")

    p20 = next(r for r in sweep_results if abs(r["p_resolve"] - 0.20) < 0.01)
    print(f"\nAt p_resolve=0.20 (low resolution):")
    print(f"  P(resolved):              {p20['analytical_p_resolved']:.4f}")
    print(f"  P(teacher_conference):    {p20['analytical_p_escalated']:.4f}")
    print(f"  -> Frequent escalation validates the safety net.")


if __name__ == "__main__":
    main()
