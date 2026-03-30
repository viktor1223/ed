"""Misconception-aware simulated students with BKT-driven learning.

Implements a 3-tier simulated student model:
  Tier 0: Static probability profiles (legacy, in evaluate.py)
  Tier 1: Rule-based misconception-aware responses
  Tier 2: Learning-enabled with misconception resolution

Each student has:
  - A set of active misconceptions with activation probabilities
  - BKT-driven internal knowledge state per concept
  - Misconception resolution after targeted remediation
  - Prerequisite-gated learning rates

References:
  - Brown & Burton (1978): BUGGY procedural bug model
  - Sonkar et al. (2024): MalAlgoPy cognitive student models
  - Scarlatos et al. (2026): Simulated student evaluation framework
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KG_PATH = PROJECT_ROOT / "data" / "knowledge_graph.json"
PROBLEM_BANK_PATH = PROJECT_ROOT / "data" / "problem_bank.json"


@dataclass
class MisconceptionState:
    """Tracks a single misconception's activation state within a student."""

    misconception_id: str
    concept_id: str
    p_active: float  # probability this misconception fires when applicable
    wrong_answer_templates: list[dict]  # [{problem_pattern, wrong_answer}]


@dataclass
class SimulatedStudent:
    """A simulated student with misconceptions and learning capability.

    Internal state:
        p_know: per-concept probability of having learned the concept (BKT L_n)
        misconceptions: list of active misconceptions with probabilities
        bkt_params: per-concept BKT parameters (from knowledge graph)
    """

    student_id: str
    p_know: dict[str, float] = field(default_factory=dict)
    misconceptions: list[MisconceptionState] = field(default_factory=list)
    bkt_params: dict[str, dict[str, float]] = field(default_factory=dict)
    prereqs: dict[str, list[str]] = field(default_factory=dict)
    mastery_threshold: float = 0.85

    # Tunable learning parameters
    remediation_bonus: float = 2.0      # targeted remediation accelerates learning
    generic_resolution: float = 0.05    # generic feedback barely resolves misconceptions
    targeted_resolution: float = 0.30   # targeted feedback resolves misconceptions
    prereq_penalty: float = 0.2         # learning rate multiplier when prereqs unmet

    def respond(self, problem: dict) -> dict:
        """Generate a response to a problem.

        Returns:
            dict with keys: student_response, correct, misconception_used (or None)
        """
        concept_id = problem["concept"]
        correct_answer = problem["correct_answer"]

        params = self.bkt_params.get(concept_id, {})
        p_L = self.p_know.get(concept_id, 0.1)
        p_slip = params.get("p_slip", 0.10)
        p_guess = params.get("p_guess", 0.10)

        # Does the student "know" this concept right now?
        knows = random.random() < p_L

        if knows:
            # Known concept: correct unless slip
            if random.random() < p_slip:
                # Slip: pick a random misconception for this concept if available
                concept_misconceptions = [
                    m for m in self.misconceptions
                    if m.concept_id == concept_id and m.p_active > 0.1
                ]
                if concept_misconceptions:
                    m = random.choice(concept_misconceptions)
                    wrong = self._apply_misconception(m, problem)
                    if wrong:
                        return {
                            "student_response": wrong,
                            "correct": False,
                            "misconception_used": m.misconception_id,
                        }
                # Generic slip - return a slightly wrong answer
                return {
                    "student_response": correct_answer,
                    "correct": True,
                    "misconception_used": None,
                }
            else:
                return {
                    "student_response": correct_answer,
                    "correct": True,
                    "misconception_used": None,
                }
        else:
            # Does not know: check for applicable misconceptions
            applicable = [
                m for m in self.misconceptions
                if m.concept_id == concept_id
            ]

            # Try each applicable misconception
            for m in applicable:
                if random.random() < m.p_active:
                    wrong = self._apply_misconception(m, problem)
                    if wrong:
                        return {
                            "student_response": wrong,
                            "correct": False,
                            "misconception_used": m.misconception_id,
                        }

            # No misconception triggered - guess
            if random.random() < p_guess:
                return {
                    "student_response": correct_answer,
                    "correct": True,
                    "misconception_used": None,
                }
            else:
                # Wrong answer but no specific misconception
                return {
                    "student_response": "I don't know",
                    "correct": False,
                    "misconception_used": None,
                }

    def _apply_misconception(self, m: MisconceptionState, problem: dict) -> str | None:
        """Try to produce a misconception-consistent wrong answer."""
        # Check if any template matches this problem
        for template in m.wrong_answer_templates:
            if template.get("problem_id") == problem.get("problem_id"):
                return template["wrong_answer"]

        # Fallback: check if problem text partially matches a template
        for template in m.wrong_answer_templates:
            if template.get("problem_text") and template["problem_text"] in problem.get("problem_text", ""):
                return template["wrong_answer"]

        # No specific template - use generic wrong answer from misconception examples
        if m.wrong_answer_templates:
            return random.choice(m.wrong_answer_templates)["wrong_answer"]

        return None

    def receive_instruction(
        self,
        concept_id: str,
        targeted_misconception: str | None = None,
    ) -> None:
        """Update internal state after receiving instruction.

        Args:
            concept_id: The concept being taught.
            targeted_misconception: If the tutor identified a specific misconception
                and provided targeted feedback. None for generic feedback.
        """
        params = self.bkt_params.get(concept_id, {})
        p_T = params.get("p_learn", 0.15)

        # Prerequisite gating
        prereqs_met = all(
            self.p_know.get(p, 0) >= self.mastery_threshold
            for p in self.prereqs.get(concept_id, [])
        )
        if not prereqs_met:
            p_T *= self.prereq_penalty

        # Learning transition
        p_L = self.p_know.get(concept_id, 0.1)
        if targeted_misconception:
            # Targeted remediation boosts learning
            p_new = p_L + (1 - p_L) * p_T * self.remediation_bonus
        else:
            p_new = p_L + (1 - p_L) * p_T

        self.p_know[concept_id] = min(1.0, p_new)

        # Misconception resolution
        if targeted_misconception:
            for m in self.misconceptions:
                if m.misconception_id == targeted_misconception:
                    m.p_active *= (1 - self.targeted_resolution)
                    break
        else:
            # Generic feedback slightly reduces all misconceptions for this concept
            for m in self.misconceptions:
                if m.concept_id == concept_id:
                    m.p_active *= (1 - self.generic_resolution)

    def is_mastered(self, concept_id: str) -> bool:
        return self.p_know.get(concept_id, 0) >= self.mastery_threshold

    def active_misconceptions(self, concept_id: str | None = None) -> list[str]:
        """Return misconception IDs that are still active (p_active > 0.1)."""
        result = []
        for m in self.misconceptions:
            if m.p_active > 0.1:
                if concept_id is None or m.concept_id == concept_id:
                    result.append(m.misconception_id)
        return result

    def summary(self) -> dict:
        return {
            "student_id": self.student_id,
            "p_know": {k: round(v, 4) for k, v in self.p_know.items()},
            "active_misconceptions": [
                {"id": m.misconception_id, "p_active": round(m.p_active, 3)}
                for m in self.misconceptions if m.p_active > 0.1
            ],
        }


# ─── Student Profile Generator ───────────────────────────────────────────────

# Archetype definitions: (name, description, concept_p_know_ranges, n_misconceptions)
ARCHETYPES = [
    {
        "name": "strong_overall",
        "weight": 0.15,
        "p_know_range": (0.60, 0.85),
        "n_misconceptions": (1, 2),
        "description": "Strong student with 1-2 residual misconceptions",
    },
    {
        "name": "strong_arithmetic_weak_algebra",
        "weight": 0.25,
        "p_know_by_level": {1: (0.60, 0.80), 2: (0.50, 0.70), 3: (0.10, 0.30), 4: (0.05, 0.20), 5: (0.05, 0.15)},
        "n_misconceptions": (3, 5),
        "description": "Strong on basics, struggles with algebra",
    },
    {
        "name": "specific_gap",
        "weight": 0.20,
        "p_know_range": (0.50, 0.75),
        "gap_concepts": 1,  # 1 concept is very weak
        "gap_p_know": (0.05, 0.15),
        "n_misconceptions": (2, 4),
        "description": "Generally competent but has a specific conceptual gap",
    },
    {
        "name": "weak_overall",
        "weight": 0.20,
        "p_know_range": (0.05, 0.25),
        "n_misconceptions": (4, 6),
        "description": "Weak across the board, many misconceptions",
    },
    {
        "name": "random_mixed",
        "weight": 0.20,
        "p_know_range": (0.10, 0.70),
        "n_misconceptions": (2, 5),
        "description": "Mixed abilities, unpredictable pattern",
    },
]


def load_misconception_templates(kg_path: str | Path = KG_PATH) -> dict[str, list[dict]]:
    """Build misconception response templates from the knowledge graph examples."""
    with open(kg_path) as f:
        data = json.load(f)

    templates: dict[str, list[dict]] = {}
    for concept in data["concepts"]:
        for m in concept.get("misconceptions", []):
            m_id = m["id"]
            templates[m_id] = []
            for ex in m.get("examples", []):
                templates[m_id].append({
                    "problem_text": ex["problem"],
                    "wrong_answer": ex["wrong"],
                    "correct_answer": ex["correct"],
                })
    return templates


def load_problem_bank(path: str | Path = PROBLEM_BANK_PATH) -> dict[str, list[dict]]:
    """Load problem bank grouped by concept."""
    with open(path) as f:
        problems = json.load(f)
    bank: dict[str, list[dict]] = {}
    for p in problems:
        bank.setdefault(p["concept"], []).append(p)
    return bank


def generate_students(
    n: int,
    kg_path: str | Path = KG_PATH,
    seed: int = 42,
) -> list[SimulatedStudent]:
    """Generate n diverse simulated students with misconception profiles.

    Students are sampled from archetypes weighted by classroom distribution.
    Each student gets a subset of misconceptions appropriate to their profile.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    with open(kg_path) as f:
        kg_data = json.load(f)

    concepts = kg_data["concepts"]
    concept_ids = [c["id"] for c in concepts]
    concept_levels = {c["id"]: c["level"] for c in concepts}
    concept_prereqs = {c["id"]: c.get("prerequisites", []) for c in concepts}
    concept_bkt = {c["id"]: c.get("bkt_params", {}) for c in concepts}

    # Build misconception pool per concept
    misconception_pool: dict[str, list[dict]] = {}
    for c in concepts:
        misconception_pool[c["id"]] = [
            {"id": m["id"], "concept_id": c["id"], "examples": m.get("examples", [])}
            for m in c.get("misconceptions", [])
        ]

    templates = load_misconception_templates(kg_path)

    # Deep-copy BKT params so each student has independent dicts
    import copy

    # Archetype weights
    archetype_weights = [a["weight"] for a in ARCHETYPES]
    total = sum(archetype_weights)
    archetype_probs = [w / total for w in archetype_weights]

    students: list[SimulatedStudent] = []

    for i in range(n):
        # Select archetype
        archetype = np_rng.choice(ARCHETYPES, p=archetype_probs)

        # Generate p_know per concept
        p_know: dict[str, float] = {}
        if "p_know_by_level" in archetype:
            for cid in concept_ids:
                level = concept_levels[cid]
                lo, hi = archetype["p_know_by_level"][level]
                p_know[cid] = rng.uniform(lo, hi)
        else:
            lo, hi = archetype["p_know_range"]
            for cid in concept_ids:
                p_know[cid] = rng.uniform(lo, hi)

        # Apply gap if archetype has one
        if archetype.get("gap_concepts"):
            gap_count = archetype["gap_concepts"]
            gap_concepts = rng.sample(concept_ids, min(gap_count, len(concept_ids)))
            gap_lo, gap_hi = archetype["gap_p_know"]
            for gc in gap_concepts:
                p_know[gc] = rng.uniform(gap_lo, gap_hi)

        # Assign misconceptions
        n_misc_lo, n_misc_hi = archetype["n_misconceptions"]
        n_misc = rng.randint(n_misc_lo, n_misc_hi)

        # Prefer misconceptions in weaker concepts
        weak_concepts = sorted(concept_ids, key=lambda c: p_know[c])
        available_misconceptions = []
        for cid in weak_concepts:
            available_misconceptions.extend(misconception_pool.get(cid, []))

        # Also add some from stronger concepts (residual misconceptions)
        for cid in concept_ids:
            if p_know[cid] > 0.5:
                for m_info in misconception_pool.get(cid, []):
                    if m_info not in available_misconceptions:
                        available_misconceptions.append(m_info)

        # Sample misconceptions (favor weaker concepts by list order)
        selected = []
        for m_info in available_misconceptions:
            if len(selected) >= n_misc:
                break
            if rng.random() < 0.6:  # 60% chance to pick each in order (biased to weak)
                selected.append(m_info)

        # Ensure we have at least 1 misconception
        if not selected and available_misconceptions:
            selected = [rng.choice(available_misconceptions)]

        misconceptions = []
        for m_info in selected:
            # Activation probability inversely correlated with p_know
            concept_know = p_know[m_info["concept_id"]]
            p_active = max(0.2, min(0.95, 1.0 - concept_know + rng.uniform(-0.1, 0.1)))

            misconceptions.append(MisconceptionState(
                misconception_id=m_info["id"],
                concept_id=m_info["concept_id"],
                p_active=p_active,
                wrong_answer_templates=templates.get(m_info["id"], []),
            ))

        student = SimulatedStudent(
            student_id=f"sim_{i:04d}",
            p_know=p_know,
            misconceptions=misconceptions,
            bkt_params=copy.deepcopy(concept_bkt),
            prereqs=concept_prereqs,
        )
        students.append(student)

    return students


# ─── Convenience ──────────────────────────────────────────────────────────────

def describe_population(students: list[SimulatedStudent]) -> dict:
    """Summary statistics for a generated student population."""
    n = len(students)
    all_concepts = list(students[0].p_know.keys()) if students else []

    avg_p_know = {
        c: np.mean([s.p_know[c] for s in students]) for c in all_concepts
    }
    avg_n_misconceptions = np.mean([len(s.active_misconceptions()) for s in students])

    misconception_counts: dict[str, int] = {}
    for s in students:
        for m in s.misconceptions:
            if m.p_active > 0.1:
                misconception_counts[m.misconception_id] = misconception_counts.get(m.misconception_id, 0) + 1

    return {
        "n_students": n,
        "avg_p_know": {k: round(v, 3) for k, v in avg_p_know.items()},
        "avg_active_misconceptions": round(float(avg_n_misconceptions), 1),
        "misconception_prevalence": {
            k: round(v / n, 3) for k, v in sorted(misconception_counts.items(), key=lambda x: -x[1])
        },
    }
