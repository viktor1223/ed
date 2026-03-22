"""Interactive CLI for the algebra tutor."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

print("Loading libraries (first run may take ~30s)...", flush=True)
from tutor_session import TutorSession

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KG_PATH = PROJECT_ROOT / "data" / "knowledge_graph.json"
MODEL_DIR = PROJECT_ROOT / "models" / "classifier" / "best"
PROBLEM_BANK = PROJECT_ROOT / "data" / "problem_bank.json"


def print_mastery_bar(name: str, mastery: float, mastered: bool) -> None:
    bar_len = 30
    filled = int(mastery * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    status = " ✓" if mastered else ""
    print(f"  {name:<35} [{bar}] {mastery:.0%}{status}")


def main():
    print("=" * 60)
    print("  Algebra Misconception Tutor")
    print("  Type your answer, 'summary' for progress, 'quit' to exit")
    print("=" * 60)
    print()

    print("Loading model...")
    session = TutorSession(
        kg_path=KG_PATH,
        model_dir=MODEL_DIR,
        problem_bank_path=PROBLEM_BANK,
    )
    print("Ready!\n")

    while True:
        # Present a problem
        problem = session.present_problem()
        if "error" in problem:
            print(f"Error: {problem['error']}")
            break

        print(f"─── {problem['concept_name']} ({problem['action']}) ───")
        print(f"  {problem['problem_text']}")
        print()

        # Get student answer
        try:
            answer = input("Your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if answer.lower() == "quit":
            break

        if answer.lower() == "summary":
            _print_summary(session)
            continue

        if not answer:
            print("  Please enter an answer.\n")
            continue

        # Evaluate
        result = session.evaluate_response(answer)

        if result["correct"]:
            print(f"  ✓ Correct! (mastery: {result['mastery_after']:.0%})\n")
        else:
            print(f"  ✗ Not quite. The correct answer is: {result['correct_answer']}")
            if result["predicted_misconception"]:
                print(f"    Detected misconception: {result['predicted_misconception']}")
            if result["hint"]:
                print(f"    Hint: {result['hint']}")
            print(f"    (mastery: {result['mastery_after']:.0%})\n")

    _print_summary(session)


def _print_summary(session: TutorSession) -> None:
    summary = session.session_summary()
    print()
    print("═" * 60)
    print(f"  Session Summary: {summary['correct_answers']}/{summary['total_problems']} correct ({summary['accuracy']:.0%})")
    print("═" * 60)
    for cid, info in summary["concepts"].items():
        print_mastery_bar(info["name"], info["mastery"], info["mastered"])
    print()


if __name__ == "__main__":
    main()
