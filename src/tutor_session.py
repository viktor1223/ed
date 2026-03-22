"""Adaptive tutoring session integrating classifier, knowledge graph, and mastery model."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from classifier import MisconceptionClassifier
from knowledge_graph import KnowledgeGraph, StudentState, next_action


# Pre-built hints keyed by misconception ID
HINTS: dict[str, str] = {
    "sign_sum_negatives": "When you add two negative numbers, the result is more negative, not positive. Think of it like owing money: if you owe $6 and then owe $3 more, you owe $9 total, not positive $9.",
    "sign_neg_times_neg": "Remember the sign rule for multiplication: negative times negative gives a positive result. Think of it as 'two wrongs make a right' for multiplication.",
    "sign_sub_negative": "Subtracting a negative is the same as adding a positive. For example, 5 - (-3) becomes 5 + 3 = 8. The two negatives cancel out.",
    "sign_always_subtract_smaller": "Be careful with the sign! When you subtract a larger number from a smaller one, the result is negative. 3 - 7 = -4, not 4.",
    "oo_left_to_right": "Remember the order of operations: multiplication and division come before addition and subtraction. Don't just go left to right - check for × and ÷ first.",
    "oo_exponent_after_add": "Exponents come before addition in the order of operations. Calculate the power first, then add. For 3 + 4², first compute 4² = 16, then 3 + 16 = 19.",
    "oo_parentheses_ignored": "Parentheses have the highest priority! Always calculate what's inside the parentheses first before doing anything else.",
    "dist_first_term_only": "When distributing, multiply the factor by EVERY term inside the parentheses, not just the first one. In 2(x + 3), both x and 3 get multiplied by 2.",
    "dist_square_over_addition": "Squaring a sum is not the same as squaring each part! (a + b)² = a² + 2ab + b². Don't forget the middle term (2ab).",
    "dist_sign_error_negative": "When distributing a negative number, be careful with signs. A negative times a negative gives a positive. So -3(x - 4) = -3x + 12, not -3x - 12.",
    "dist_drop_parens": "You need to multiply the factor by every term inside the parentheses. Don't just remove the parentheses - actually distribute the multiplication.",
    "clt_combine_unlike": "You can only combine terms that have the same variable. 3x + 2y cannot be simplified to 5xy - they are unlike terms.",
    "clt_multiply_variables": "When combining like terms, add the coefficients but keep the variable the same. 2x + 3x = 5x (not 5x²). You're adding, not multiplying.",
    "clt_constant_as_variable": "Constants (plain numbers) and variable terms are not like terms. 5x + 3 cannot be simplified further - keep them separate.",
    "clt_add_exponents": "When combining like terms, the exponent stays the same. 2x² + 3x² = 5x² (not 5x⁴). You add coefficients, not exponents.",
    "leq_reverse_operation": "To isolate the variable, use the OPPOSITE operation. If the equation has addition, subtract. If it has subtraction, add. This 'undoes' the operation.",
    "leq_divide_wrong_direction": "When a variable is multiplied by a number, divide both sides by that number to isolate it. For 3x = 12, divide by 3: x = 4.",
    "leq_subtract_wrong_side": "Make sure you perform the same operation on BOTH sides of the equation. If you subtract from one side, subtract the same amount from the other side too.",
    "leq_move_without_sign_change": "When moving a term to the other side of the equation, change its sign. If it was subtracted on one side, it becomes added on the other side.",
}


class TutorSession:
    """Ties together classifier, knowledge graph, and student state for tutoring."""

    def __init__(
        self,
        kg_path: str | Path,
        model_dir: str | Path,
        problem_bank_path: str | Path,
    ):
        self.kg = KnowledgeGraph.from_json(kg_path)
        self.state = StudentState(self.kg)
        self.classifier = MisconceptionClassifier(model_dir)
        self.problem_bank = self._load_problem_bank(problem_bank_path)
        self.current_problem: dict | None = None
        self.history: list[dict] = []

    @staticmethod
    def _load_problem_bank(path: str | Path) -> dict[str, list[dict]]:
        with open(path) as f:
            problems = json.load(f)
        bank: dict[str, list[dict]] = {}
        for p in problems:
            bank.setdefault(p["concept"], []).append(p)
        return bank

    def present_problem(self) -> dict:
        """Select and return the next problem based on the adaptive engine."""
        action = next_action(self.state, self.kg)
        concept_id = action["concept"]

        problems = self.problem_bank.get(concept_id, [])
        if not problems:
            return {"error": f"No problems available for {concept_id}"}

        # Prefer problems not recently shown
        recent_ids = {h["problem_id"] for h in self.history[-5:]}
        available = [p for p in problems if p["problem_id"] not in recent_ids]
        if not available:
            available = problems

        problem = random.choice(available)
        self.current_problem = {**problem, "action": action}
        return {
            "problem_text": problem["problem_text"],
            "concept": concept_id,
            "concept_name": self.kg.concepts[concept_id].name,
            "action": action["action"],
            "reason": action["reason"],
        }

    def evaluate_response(self, student_text: str) -> dict[str, Any]:
        """Classify the student's response and update mastery."""
        if self.current_problem is None:
            return {"error": "No current problem. Call present_problem() first."}

        problem = self.current_problem
        concept_id = problem["concept"]
        correct_answer = problem["correct_answer"]

        # Run classifier
        prediction = self.classifier.predict(
            question=problem["problem_text"],
            student_response=student_text,
        )

        misconception_id = prediction["label"]
        confidence = prediction["confidence"]

        # Check if the answer is correct (simple heuristic: extract the answer part)
        is_correct = self._check_correct(student_text, correct_answer)

        if is_correct:
            misconception_id = "correct"
            self.state.update(concept_id, correct=True)
        else:
            self.state.update(concept_id, correct=False, confidence=confidence)

        # Get hint if incorrect
        hint = None
        if not is_correct and misconception_id in HINTS:
            hint = HINTS[misconception_id]

        result = {
            "correct": is_correct,
            "correct_answer": correct_answer,
            "predicted_misconception": misconception_id if not is_correct else None,
            "confidence": round(confidence, 3),
            "hint": hint,
            "mastery_after": round(self.state.mastery[concept_id], 4),
        }

        # Record history
        self.history.append({
            "problem_id": problem["problem_id"],
            "concept": concept_id,
            "student_response": student_text,
            "correct": is_correct,
            "misconception": misconception_id,
        })

        self.current_problem = None
        return result

    @staticmethod
    def _check_correct(student_text: str, correct_answer: str) -> bool:
        """Fuzzy check if the student's response matches the correct answer."""
        import re

        def normalize(s: str) -> str:
            return (
                s.lower()
                .replace(" ", "")
                .replace("×", "*")
                .replace("÷", "/")
                .replace("²", "^2")
                .replace("³", "^3")
                .replace("⁴", "^4")
            )

        def extract_value(s: str) -> str:
            """Extract the numeric/expression value, stripping 'x=' prefix and prose."""
            s = normalize(s)
            # Remove common prose
            for phrase in ["ithinktheansweris", "myanswer is", "igot", "theansweris", "ithink"]:
                s = s.replace(phrase, "")
            # Strip variable assignment like 'x=' or 'm='
            s = re.sub(r"^[a-z]=", "", s)
            return s.strip()

        student_val = extract_value(student_text)
        correct_val = extract_value(correct_answer)

        # Direct match after extraction
        if student_val == correct_val:
            return True

        # Check if the correct value appears anywhere in the student text
        # Use word-boundary-like check to avoid "-4" matching "4"
        student_norm = normalize(student_text)
        if correct_val and correct_val in student_norm:
            # Make sure it's not a substring of a larger number (e.g. "4" in "-4")
            idx = student_norm.find(correct_val)
            before = student_norm[idx - 1] if idx > 0 else ""
            if before not in "0123456789.-":
                return True

        # Check if the extracted student value appears in the correct answer
        # Only if same length (avoid "4" matching "-4")
        if student_val and student_val == correct_val:
            return True

        return False

    def get_hint(self, misconception_id: str) -> str:
        """Get a targeted hint for a specific misconception."""
        return HINTS.get(misconception_id, "Try reviewing this concept and attempt the problem again.")

    def session_summary(self) -> dict[str, Any]:
        """Return current mastery state across all concepts."""
        summary = self.state.summary()
        total = len(self.history)
        correct = sum(1 for h in self.history if h["correct"])
        return {
            "total_problems": total,
            "correct_answers": correct,
            "accuracy": round(correct / total, 3) if total > 0 else 0,
            "concepts": {
                cid: {
                    **info,
                    "name": self.kg.concepts[cid].name,
                }
                for cid, info in summary.items()
            },
        }
