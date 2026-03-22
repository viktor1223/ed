"""Knowledge graph and BKT-based student mastery model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Misconception:
    id: str
    label: str
    description: str
    examples: list[dict[str, str]]


@dataclass
class Concept:
    id: str
    name: str
    description: str
    level: int
    prerequisites: list[str]
    mae_ids: list[str]
    bkt_params: dict[str, float]
    misconceptions: list[Misconception]


@dataclass
class KnowledgeGraph:
    """Loads and queries the algebra concept prerequisite graph."""

    concepts: dict[str, Concept] = field(default_factory=dict)
    edges: list[dict[str, str]] = field(default_factory=list)
    mastery_threshold: float = 0.85
    mastery_initial: float = 0.5

    @classmethod
    def from_json(cls, path: str | Path) -> KnowledgeGraph:
        with open(path) as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        kg = cls(
            mastery_threshold=meta.get("mastery_threshold", 0.85),
            mastery_initial=meta.get("mastery_initial", 0.5),
            edges=data.get("edges", []),
        )

        for c in data.get("concepts", []):
            misconceptions = [
                Misconception(
                    id=m["id"],
                    label=m["label"],
                    description=m["description"],
                    examples=m.get("examples", []),
                )
                for m in c.get("misconceptions", [])
            ]
            kg.concepts[c["id"]] = Concept(
                id=c["id"],
                name=c["name"],
                description=c["description"],
                level=c["level"],
                prerequisites=c.get("prerequisites", []),
                mae_ids=c.get("mae_ids", []),
                bkt_params=c.get("bkt_params", {}),
                misconceptions=misconceptions,
            )
        return kg

    def get_concept(self, concept_id: str) -> Concept:
        return self.concepts[concept_id]

    def prerequisites_of(self, concept_id: str) -> list[str]:
        return self.concepts[concept_id].prerequisites

    def concepts_by_level(self) -> list[Concept]:
        return sorted(self.concepts.values(), key=lambda c: c.level)

    def misconception_to_concept(self, misconception_id: str) -> str | None:
        for concept in self.concepts.values():
            for m in concept.misconceptions:
                if m.id == misconception_id:
                    return concept.id
        return None

    def all_misconception_ids(self) -> list[str]:
        ids = []
        for concept in self.concepts_by_level():
            for m in concept.misconceptions:
                ids.append(m.id)
        return ids

    def label_list(self) -> list[str]:
        """Return sorted list of all classification labels (misconception IDs + 'correct')."""
        return sorted(self.all_misconception_ids() + ["correct"])


class StudentState:
    """Per-student mastery tracking using Bayesian Knowledge Tracing."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.mastery: dict[str, float] = {
            cid: kg.concepts[cid].bkt_params.get("p_init", kg.mastery_initial)
            for cid in kg.concepts
        }
        self.attempts: dict[str, int] = {cid: 0 for cid in kg.concepts}

    def update(self, concept_id: str, correct: bool, confidence: float = 1.0) -> float:
        """Update mastery for a concept after an observation.

        Args:
            concept_id: Which concept was tested.
            correct: Whether the student answered correctly.
            confidence: Classifier confidence (0-1). Only used for incorrect answers
                        to scale the mastery decrease.

        Returns:
            Updated mastery probability.
        """
        params = self.kg.concepts[concept_id].bkt_params
        p_L = self.mastery[concept_id]
        p_G = params.get("p_guess", 0.10)
        p_S = params.get("p_slip", 0.10)
        p_T = params.get("p_learn", 0.15)

        # Standard BKT posterior update
        if correct:
            # P(L_n | correct) = P(correct | L_n) * P(L_n) / P(correct)
            p_correct = p_L * (1 - p_S) + (1 - p_L) * p_G
            p_L_given_obs = (p_L * (1 - p_S)) / p_correct if p_correct > 0 else p_L
        else:
            # P(L_n | incorrect) = P(incorrect | L_n) * P(L_n) / P(incorrect)
            p_incorrect = p_L * p_S + (1 - p_L) * (1 - p_G)
            p_L_given_obs = (p_L * p_S) / p_incorrect if p_incorrect > 0 else p_L

        # Learning transition: P(L_{n+1}) = P(L_n | obs) + (1 - P(L_n | obs)) * P(T)
        p_new = p_L_given_obs + (1 - p_L_given_obs) * p_T

        # For incorrect answers with high classifier confidence, apply additional penalty
        if not correct and confidence > 0.5:
            penalty = 0.05 * confidence
            p_new = max(0.01, p_new - penalty)

        self.mastery[concept_id] = p_new
        self.attempts[concept_id] += 1
        return p_new

    def is_mastered(self, concept_id: str) -> bool:
        return self.mastery[concept_id] >= self.kg.mastery_threshold

    def prerequisites_met(self, concept_id: str) -> bool:
        """Check if all prerequisites for a concept are mastered."""
        for prereq in self.kg.prerequisites_of(concept_id):
            if not self.is_mastered(prereq):
                return False
        return True

    def summary(self) -> dict[str, Any]:
        return {
            cid: {
                "mastery": round(self.mastery[cid], 4),
                "mastered": self.is_mastered(cid),
                "attempts": self.attempts[cid],
            }
            for cid in self.kg.concepts
        }


def next_action(state: StudentState, kg: KnowledgeGraph) -> dict[str, str]:
    """Decide the next tutoring action based on student state.

    Returns a dict with:
        action: "remediate" | "progress" | "review" | "start"
        concept: the target concept ID
        reason: human-readable explanation
    """
    # 1. If no concepts attempted, start with the lowest-level concept
    if all(a == 0 for a in state.attempts.values()):
        first = kg.concepts_by_level()[0]
        return {
            "action": "start",
            "concept": first.id,
            "reason": f"Begin with {first.name} (Level {first.level})",
        }

    # 2. Check for concepts that need remediation (attempted + below threshold)
    for concept in kg.concepts_by_level():
        if state.attempts[concept.id] > 0 and not state.is_mastered(concept.id):
            return {
                "action": "remediate",
                "concept": concept.id,
                "reason": f"{concept.name} mastery is {state.mastery[concept.id]:.2f}, below threshold {kg.mastery_threshold}",
            }

    # 3. Check for progression (prerequisites met, not yet attempted or not mastered)
    for concept in kg.concepts_by_level():
        if state.attempts[concept.id] == 0 and state.prerequisites_met(concept.id):
            return {
                "action": "progress",
                "concept": concept.id,
                "reason": f"Prerequisites met - advance to {concept.name} (Level {concept.level})",
            }

    # 4. If all attempted and mastered, find the lowest-mastered for review
    borderline = sorted(
        [(cid, m) for cid, m in state.mastery.items()],
        key=lambda x: x[1],
    )
    if borderline:
        cid, m = borderline[0]
        return {
            "action": "review",
            "concept": cid,
            "reason": f"All concepts mastered - review {kg.concepts[cid].name} (mastery {m:.2f})",
        }

    # Should never reach here with a non-empty graph
    return {"action": "review", "concept": kg.concepts_by_level()[0].id, "reason": "Fallback"}
