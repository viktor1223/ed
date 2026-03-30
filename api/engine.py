"""Diagnostic engine: wraps the existing classifier, knowledge graph, and BKT modules.

This is the bridge between the API layer and the src/ intelligence.
The API never imports from src/ directly - it goes through this module.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Add src/ to path so we can import the existing modules
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "classifier" / "best"

from knowledge_graph import KnowledgeGraph, StudentState

# Pre-built hints keyed by misconception ID (from tutor_session.py)
HINTS: dict[str, str] = {
    "sign_sum_negatives": "When you add two negative numbers, the result is more negative, not positive.",
    "sign_neg_times_neg": "Negative times negative gives a positive result.",
    "sign_sub_negative": "Subtracting a negative is the same as adding a positive.",
    "sign_always_subtract_smaller": "When subtracting a larger number from a smaller one, the result is negative.",
    "oo_left_to_right": "Multiplication and division come before addition and subtraction.",
    "oo_exponent_after_add": "Exponents come before addition. Calculate the power first.",
    "oo_parentheses_ignored": "Always calculate what's inside the parentheses first.",
    "dist_first_term_only": "Multiply the factor by EVERY term inside the parentheses.",
    "dist_square_over_addition": "(a + b)² = a² + 2ab + b². Don't forget the middle term.",
    "dist_sign_error_negative": "A negative times a negative gives a positive.",
    "dist_drop_parens": "Multiply the factor by every term inside the parentheses.",
    "clt_combine_unlike": "You can only combine terms with the same variable.",
    "clt_multiply_variables": "When combining like terms, add coefficients. Keep the variable the same.",
    "clt_constant_as_variable": "Constants and variable terms are not like terms.",
    "clt_add_exponents": "When combining like terms, the exponent stays the same.",
    "leq_reverse_operation": "Use the OPPOSITE operation to isolate the variable.",
    "leq_divide_wrong_direction": "Divide both sides by the coefficient to isolate the variable.",
    "leq_subtract_wrong_side": "Perform the same operation on BOTH sides of the equation.",
    "leq_move_without_sign_change": "When moving a term across the equals sign, change its sign.",
}

INTERVENTIONS: dict[str, str] = {
    "sign_sum_negatives": "Small-group: use a number line to show adding negatives moves further left.",
    "sign_neg_times_neg": "Use pattern recognition: -3×3=-9, -3×2=-6, -3×1=-3, -3×0=0, -3×(-1)=?",
    "sign_sub_negative": "Use chip/counter model: removing negative chips is like adding positives.",
    "sign_always_subtract_smaller": "Number line walk: start at first number, walk left for subtraction.",
    "oo_left_to_right": "Circle all × and ÷ first, solve those, then do + and - left to right.",
    "oo_exponent_after_add": "Write PEMDAS steps: underline exponents first, then multiplication, then addition.",
    "oo_parentheses_ignored": "Use colored parentheses: highlight them and solve inside-out.",
    "dist_first_term_only": "Area model: draw a rectangle with width = factor and length = (a + b).",
    "dist_square_over_addition": "Expand (a+b)² as (a+b)(a+b) and use FOIL.",
    "dist_sign_error_negative": "Two-step: distribute the magnitude first, then fix all signs.",
    "dist_drop_parens": "Arrow method: draw arrows from the factor to each term inside parentheses.",
    "clt_combine_unlike": "Sort terms into groups by variable using colored highlighting.",
    "clt_multiply_variables": "Apples analogy: 2 apples + 3 apples = 5 apples, not 5 apples².",
    "clt_constant_as_variable": "Separate: draw a line between variable terms and constants.",
    "clt_add_exponents": "x² means x·x. Two groups of x·x is still x·x (just more of them).",
    "leq_reverse_operation": "Inverse operations chart: + ↔ -, × ↔ ÷. Always use the opposite.",
    "leq_divide_wrong_direction": "Cover the variable: 3x = 12. What times 3 is 12?",
    "leq_subtract_wrong_side": "Balance scale visual: whatever you do to one side, do to the other.",
    "leq_move_without_sign_change": "Rewrite as inverse operation instead of 'moving' terms.",
}


class DiagnosticEngine:
    """Shared engine instance for the API. Loads once, used by all requests."""

    def __init__(self):
        self.kg = KnowledgeGraph.from_json(DATA_DIR / "knowledge_graph.json")
        self.problem_bank = self._load_problem_bank()
        self.problems_by_id: dict[str, dict] = {
            p["problem_id"]: p for p in self._all_problems()
        }
        self._classifier = None  # Lazily loaded

    def _load_problem_bank(self) -> dict[str, list[dict]]:
        with open(DATA_DIR / "problem_bank.json") as f:
            problems = json.load(f)
        bank: dict[str, list[dict]] = {}
        for p in problems:
            bank.setdefault(p["concept"], []).append(p)
        return bank

    def _all_problems(self) -> list[dict]:
        result = []
        for problems in self.problem_bank.values():
            result.extend(problems)
        return result

    @property
    def classifier(self):
        """Lazy-load classifier to avoid slow startup when not needed."""
        if self._classifier is None:
            from classifier import MisconceptionClassifier
            self._classifier = MisconceptionClassifier(MODEL_DIR)
        return self._classifier

    def classify_response(self, problem_text: str, student_text: str) -> dict:
        """Classify a student's response against a problem."""
        result = self.classifier.predict(problem_text, student_text)
        return {
            "label": result["label"],
            "confidence": result["confidence"],
        }

    def get_problem(self, problem_id: str) -> dict | None:
        return self.problems_by_id.get(problem_id)

    def get_problems_for_concept(self, concept_id: str, difficulty: str | None = None) -> list[dict]:
        problems = self.problem_bank.get(concept_id, [])
        if difficulty:
            problems = [p for p in problems if p.get("difficulty") == difficulty]
        return problems

    def mastery_status(self, level: float) -> str:
        if level >= 0.85:
            return "mastered"
        elif level >= 0.60:
            return "progressing"
        elif level >= 0.35:
            return "struggling"
        else:
            return "critical"

    def bkt_update(self, concept_id: str, current_mastery: float, correct: bool, confidence: float = 1.0) -> float:
        """Run a single BKT update step and return new mastery."""
        params = self.kg.concepts[concept_id].bkt_params
        p_L = current_mastery
        p_G = params.get("p_guess", 0.10)
        p_S = params.get("p_slip", 0.10)
        p_T = params.get("p_learn", 0.15)

        if correct:
            p_correct = p_L * (1 - p_S) + (1 - p_L) * p_G
            p_L_given_obs = (p_L * (1 - p_S)) / p_correct if p_correct > 0 else p_L
        else:
            p_incorrect = p_L * p_S + (1 - p_L) * (1 - p_G)
            p_L_given_obs = (p_L * p_S) / p_incorrect if p_incorrect > 0 else p_L

        p_new = p_L_given_obs + (1 - p_L_given_obs) * p_T

        if not correct and confidence > 0.5:
            penalty = 0.05 * confidence
            p_new = max(0.01, p_new - penalty)

        return round(p_new, 4)

    def check_correct(self, student_text: str, correct_answer: str) -> bool:
        """Fuzzy-check if a student response matches the correct answer."""
        def normalize(s: str) -> str:
            return (
                s.lower()
                .replace(" ", "")
                .replace("×", "*")
                .replace("÷", "/")
                .replace("−", "-")
                .strip()
            )
        return normalize(student_text) == normalize(correct_answer)

    def get_intervention(self, misconception_id: str) -> str:
        return INTERVENTIONS.get(misconception_id, "Review the concept with the student individually.")

    def get_hint(self, misconception_id: str) -> str | None:
        return HINTS.get(misconception_id)

    def downstream_concepts(self, concept_id: str) -> list[str]:
        """Return all concepts that directly or transitively depend on concept_id."""
        downstream = []
        for c in self.kg.concepts_by_level():
            if concept_id in self._all_prerequisites(c.id) and c.id != concept_id:
                downstream.append(c.id)
        return downstream

    def _all_prerequisites(self, concept_id: str) -> set[str]:
        """Recursively collect all prerequisites."""
        result: set[str] = set()
        stack = list(self.kg.prerequisites_of(concept_id))
        while stack:
            prereq = stack.pop()
            if prereq not in result:
                result.add(prereq)
                stack.extend(self.kg.prerequisites_of(prereq))
        return result

    def recommend_problems(self, concept_id: str, count: int = 3) -> list[str]:
        """Pick problem IDs targeting a concept."""
        problems = self.get_problems_for_concept(concept_id)
        # Prefer easier problems first for remediation
        problems.sort(key=lambda p: {"easy": 0, "medium": 1, "hard": 2}.get(p.get("difficulty", "medium"), 1))
        return [p["problem_id"] for p in problems[:count]]


# Singleton
engine = DiagnosticEngine()
