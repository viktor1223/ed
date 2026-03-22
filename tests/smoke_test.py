"""Smoke test for the full tutoring loop."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tutor_session import TutorSession

root = Path(__file__).resolve().parent.parent

session = TutorSession(
    kg_path=root / "data" / "knowledge_graph.json",
    model_dir=root / "models" / "classifier" / "best",
    problem_bank_path=root / "data" / "problem_bank.json",
)

print("=== Smoke Test ===")
for i in range(5):
    p = session.present_problem()
    print(f"Problem: {p['problem_text']} ({p['concept_name']}, {p['action']})")

    if i < 2:
        result = session.evaluate_response("I think the answer is 999")
        print(f"  correct={result['correct']}, misconception={result.get('predicted_misconception')}, mastery={result['mastery_after']}")
    else:
        result = session.evaluate_response("-10")
        print(f"  correct={result['correct']}, mastery={result['mastery_after']}")
    print()

summary = session.session_summary()
print(f"Summary: {summary['correct_answers']}/{summary['total_problems']} correct")
for cid, info in summary["concepts"].items():
    print(f"  {info['name']}: {info['mastery']:.2%} (attempts: {info['attempts']})")
