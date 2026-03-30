"""V3 Misconception-aware simulated student with conditional learning.

Key differences from v2 (simulated_student.py):
  - Learning is CONDITIONAL on instruction quality:
    * Correct targeting: p_know increases, misconception resolves
    * Wrong targeting: p_know stays flat, confusion accumulates
    * Generic (no targeting): minimal learning, barely touches misconceptions
  - Negative transfer: wrong instruction can strengthen misconceptions
  - Confusion model: repeated wrong instruction degrades learning rate
  - Compatible API: same respond() and receive_instruction() signatures

Grounded in:
  - BKT (Corbett & Anderson, 1995) for mastery tracking
  - Procedural bug theory (Brown & Burton, 1978) for stable misconceptions
  - Interference theory for negative transfer
  - BEAGLE (Wang et al., 2026) architectural principles (knowledge gating,
    conditional learning, decoupled evaluation)
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KG_V2_PATH = PROJECT_ROOT / "data" / "knowledge_graph_v2.json"
PROBLEM_BANK_V2_PATH = PROJECT_ROOT / "data" / "problem_bank_v2.json"


@dataclass
class MisconceptionState:
    """Tracks a single misconception's activation state."""

    misconception_id: str
    concept_id: str
    p_active: float
    strength: float  # resistance to resolution (0=fragile, 1=entrenched)
    wrong_answer_templates: list[dict]


@dataclass
class ConceptState:
    """Per-concept knowledge state with consolidation tracking."""

    concept_id: str
    p_know: float  # BKT mastery probability
    p_know_stable: float  # consolidated mastery (resistant to interference)
    exposure_count: int = 0  # total instruction events


@dataclass
class SimulatedStudentV3:
    """Simulated student with conditional learning and negative transfer.

    API-compatible with SimulatedStudent (v2):
      - respond(problem) -> dict
      - receive_instruction(concept_id, targeted_misconception) -> None
    """

    student_id: str
    concepts: dict[str, ConceptState] = field(default_factory=dict)
    misconceptions: list[MisconceptionState] = field(default_factory=list)
    bkt_params: dict[str, dict[str, float]] = field(default_factory=dict)
    prereqs: dict[str, list[str]] = field(default_factory=dict)
    mastery_threshold: float = 0.85

    # --- Learning parameters (the key changes from v2) ---

    # Correct targeting: substantial learning + misconception resolution
    correct_target_learning_bonus: float = 2.5
    correct_target_resolution: float = 0.50

    # Wrong targeting: NO learning, misconception REINFORCEMENT
    wrong_target_reinforcement: float = 0.20  # strengthens the actual misconception
    wrong_target_confusion_increment: float = 1.0

    # Generic (no targeting): minimal learning
    generic_learning_multiplier: float = 0.2
    generic_resolution: float = 0.01

    # Confusion model: wrong instruction degrades future learning
    confusion_count: dict[str, float] = field(default_factory=dict)
    confusion_threshold: float = 2.0  # after this many wrong instructions, learning degrades
    confusion_penalty: float = 0.2  # learning rate multiplier when confused

    # Prerequisite gating
    prereq_penalty: float = 0.15

    # Consolidation: how fast p_know_stable catches up to p_know
    consolidation_rate: float = 0.15

    # --- Backward-compatible properties ---

    @property
    def p_know(self) -> dict[str, float]:
        """Backward-compatible access to p_know dict."""
        return {cid: cs.p_know for cid, cs in self.concepts.items()}

    def respond(self, problem: dict) -> dict:
        """Generate a response to a problem.

        Returns:
            dict with keys: student_response, correct, misconception_used (or None)
        """
        concept_id = problem["concept"]
        correct_answer = problem["correct_answer"]

        params = self.bkt_params.get(concept_id, {})
        cs = self.concepts.get(concept_id)
        p_L = cs.p_know if cs else 0.1
        p_slip = params.get("p_slip", 0.10)
        p_guess = params.get("p_guess", 0.10)

        knows = random.random() < p_L

        if knows:
            if random.random() < p_slip:
                # Slip: use a misconception if one is active
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

            for m in applicable:
                if random.random() < m.p_active:
                    wrong = self._apply_misconception(m, problem)
                    if wrong:
                        return {
                            "student_response": wrong,
                            "correct": False,
                            "misconception_used": m.misconception_id,
                        }

            if random.random() < p_guess:
                return {
                    "student_response": correct_answer,
                    "correct": True,
                    "misconception_used": None,
                }
            else:
                return {
                    "student_response": "I don't know",
                    "correct": False,
                    "misconception_used": None,
                }

    def _apply_misconception(self, m: MisconceptionState, problem: dict) -> str | None:
        """Produce a misconception-consistent wrong answer."""
        for template in m.wrong_answer_templates:
            if template.get("problem_id") == problem.get("problem_id"):
                return template["wrong_answer"]

        for template in m.wrong_answer_templates:
            pt = template.get("problem_text", "")
            if pt and pt in problem.get("problem_text", ""):
                return template["wrong_answer"]

        if m.wrong_answer_templates:
            return random.choice(m.wrong_answer_templates)["wrong_answer"]

        return None

    def receive_instruction(
        self,
        concept_id: str,
        targeted_misconception: str | None = None,
    ) -> None:
        """Update internal state after receiving instruction.

        This is where v3 fundamentally differs from v2:
        - v2: any targeted_misconception (even wrong) triggers 2x bonus
        - v3: checks if targeted_misconception matches an ACTIVE misconception
        """
        params = self.bkt_params.get(concept_id, {})
        p_T = params.get("p_learn", 0.15)

        cs = self.concepts.get(concept_id)
        if cs is None:
            cs = ConceptState(concept_id=concept_id, p_know=0.1, p_know_stable=0.1)
            self.concepts[concept_id] = cs

        cs.exposure_count += 1

        # Prerequisite gating
        prereqs_met = all(
            self.concepts.get(p, ConceptState(p, 0.1, 0.1)).p_know >= self.mastery_threshold
            for p in self.prereqs.get(concept_id, [])
        )
        if not prereqs_met:
            p_T *= self.prereq_penalty

        # Confusion penalty: past wrong instructions degrade learning
        confusion = self.confusion_count.get(concept_id, 0.0)
        if confusion >= self.confusion_threshold:
            p_T *= self.confusion_penalty

        # Determine instruction quality
        if targeted_misconception is None:
            # GENERIC instruction: minimal learning, barely touches misconceptions
            p_new = cs.p_know + (1 - cs.p_know) * p_T * self.generic_learning_multiplier
            cs.p_know = min(1.0, p_new)

            for m in self.misconceptions:
                if m.concept_id == concept_id:
                    m.p_active *= (1 - self.generic_resolution)
        else:
            # TARGETED instruction: check if it matches an active misconception
            active_ids = {
                m.misconception_id
                for m in self.misconceptions
                if m.concept_id == concept_id and m.p_active > 0.1
            }

            if targeted_misconception in active_ids:
                # CORRECT targeting: substantial learning + misconception resolution
                p_new = cs.p_know + (1 - cs.p_know) * p_T * self.correct_target_learning_bonus
                cs.p_know = min(1.0, p_new)

                for m in self.misconceptions:
                    if m.misconception_id == targeted_misconception:
                        resolution = self.correct_target_resolution * (1 - m.strength * 0.5)
                        m.p_active *= (1 - resolution)
                        m.strength = max(0, m.strength - 0.1)
                        break

                # Consolidate knowledge
                cs.p_know_stable += (cs.p_know - cs.p_know_stable) * self.consolidation_rate
            else:
                # WRONG targeting: no learning, misconception REINFORCEMENT
                # p_know does NOT increase

                # Reinforce the actual active misconceptions (student gets confused)
                for m in self.misconceptions:
                    if m.misconception_id in active_ids:
                        m.p_active = min(0.95, m.p_active * (1 + self.wrong_target_reinforcement))
                        m.strength = min(1.0, m.strength + 0.05)

                # Accumulate confusion
                self.confusion_count[concept_id] = confusion + self.wrong_target_confusion_increment

    def is_mastered(self, concept_id: str) -> bool:
        cs = self.concepts.get(concept_id)
        return cs is not None and cs.p_know >= self.mastery_threshold

    def active_misconceptions(self, concept_id: str | None = None) -> list[str]:
        result = []
        for m in self.misconceptions:
            if m.p_active > 0.1:
                if concept_id is None or m.concept_id == concept_id:
                    result.append(m.misconception_id)
        return result

    def summary(self) -> dict:
        return {
            "student_id": self.student_id,
            "p_know": {cid: round(cs.p_know, 4) for cid, cs in self.concepts.items()},
            "p_know_stable": {cid: round(cs.p_know_stable, 4) for cid, cs in self.concepts.items()},
            "confusion": {k: round(v, 1) for k, v in self.confusion_count.items() if v > 0},
            "active_misconceptions": [
                {"id": m.misconception_id, "p_active": round(m.p_active, 3), "strength": round(m.strength, 2)}
                for m in self.misconceptions if m.p_active > 0.1
            ],
        }


# ─── Student Generation ──────────────────────────────────────────────────────

ARCHETYPES = [
    {
        "name": "strong_overall",
        "weight": 0.15,
        "p_know_range": (0.55, 0.80),
        "n_misconceptions": (1, 2),
    },
    {
        "name": "strong_arith_weak_algebra",
        "weight": 0.25,
        "p_know_by_level": {
            1: (0.55, 0.75), 2: (0.45, 0.65), 3: (0.15, 0.35),
            4: (0.08, 0.20), 5: (0.05, 0.15), 6: (0.03, 0.10), 7: (0.02, 0.08),
        },
        "n_misconceptions": (3, 6),
    },
    {
        "name": "specific_gap",
        "weight": 0.20,
        "p_know_range": (0.40, 0.70),
        "gap_concepts": 2,
        "gap_p_know": (0.05, 0.15),
        "n_misconceptions": (2, 5),
    },
    {
        "name": "weak_overall",
        "weight": 0.20,
        "p_know_range": (0.05, 0.25),
        "n_misconceptions": (5, 8),
    },
    {
        "name": "random_mixed",
        "weight": 0.20,
        "p_know_range": (0.10, 0.65),
        "n_misconceptions": (3, 6),
    },
]


def load_misconception_templates_v2(kg_path: str | Path = KG_V2_PATH) -> dict[str, list[dict]]:
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


def load_problem_bank_v2(path: str | Path = PROBLEM_BANK_V2_PATH) -> dict[str, list[dict]]:
    with open(path) as f:
        problems = json.load(f)
    bank: dict[str, list[dict]] = {}
    for p in problems:
        bank.setdefault(p["concept"], []).append(p)
    return bank


def generate_students_v3(
    n: int,
    kg_path: str | Path = KG_V2_PATH,
    seed: int = 42,
) -> list[SimulatedStudentV3]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    with open(kg_path) as f:
        kg_data = json.load(f)

    concepts = kg_data["concepts"]
    concept_ids = [c["id"] for c in concepts]
    concept_levels = {c["id"]: c["level"] for c in concepts}
    concept_prereqs = {c["id"]: c.get("prerequisites", []) for c in concepts}
    concept_bkt = {c["id"]: c.get("bkt_params", {}) for c in concepts}

    misconception_pool: dict[str, list[dict]] = {}
    for c in concepts:
        misconception_pool[c["id"]] = [
            {"id": m["id"], "concept_id": c["id"], "examples": m.get("examples", [])}
            for m in c.get("misconceptions", [])
        ]

    templates = load_misconception_templates_v2(kg_path)

    archetype_weights = [a["weight"] for a in ARCHETYPES]
    total = sum(archetype_weights)
    archetype_probs = [w / total for w in archetype_weights]

    students: list[SimulatedStudentV3] = []

    for i in range(n):
        archetype = np_rng.choice(ARCHETYPES, p=archetype_probs)

        # Generate p_know per concept
        concept_states: dict[str, ConceptState] = {}
        if "p_know_by_level" in archetype:
            for cid in concept_ids:
                level = concept_levels[cid]
                lo, hi = archetype["p_know_by_level"].get(level, (0.1, 0.3))
                pk = rng.uniform(lo, hi)
                concept_states[cid] = ConceptState(concept_id=cid, p_know=pk, p_know_stable=pk * 0.8)
        else:
            lo, hi = archetype["p_know_range"]
            for cid in concept_ids:
                pk = rng.uniform(lo, hi)
                concept_states[cid] = ConceptState(concept_id=cid, p_know=pk, p_know_stable=pk * 0.8)

        # Apply gaps
        if archetype.get("gap_concepts"):
            gap_count = archetype["gap_concepts"]
            gap_cids = rng.sample(concept_ids, min(gap_count, len(concept_ids)))
            gap_lo, gap_hi = archetype["gap_p_know"]
            for gc in gap_cids:
                pk = rng.uniform(gap_lo, gap_hi)
                concept_states[gc] = ConceptState(concept_id=gc, p_know=pk, p_know_stable=pk * 0.5)

        # Assign misconceptions (favor weaker concepts)
        n_misc_lo, n_misc_hi = archetype["n_misconceptions"]
        n_misc = rng.randint(n_misc_lo, n_misc_hi)

        weak_concepts = sorted(concept_ids, key=lambda c: concept_states[c].p_know)
        available = []
        for cid in weak_concepts:
            available.extend(misconception_pool.get(cid, []))

        selected = []
        for m_info in available:
            if len(selected) >= n_misc:
                break
            if rng.random() < 0.5:
                selected.append(m_info)

        if not selected and available:
            selected = [rng.choice(available)]

        misconceptions = []
        for m_info in selected:
            concept_know = concept_states[m_info["concept_id"]].p_know
            p_active = max(0.2, min(0.95, 1.0 - concept_know + rng.uniform(-0.1, 0.1)))
            strength = max(0.0, min(1.0, rng.uniform(0.1, 0.6) + (1.0 - concept_know) * 0.3))

            misconceptions.append(MisconceptionState(
                misconception_id=m_info["id"],
                concept_id=m_info["concept_id"],
                p_active=p_active,
                strength=strength,
                wrong_answer_templates=templates.get(m_info["id"], []),
            ))

        student = SimulatedStudentV3(
            student_id=f"v3_{i:04d}",
            concepts=concept_states,
            misconceptions=misconceptions,
            bkt_params=copy.deepcopy(concept_bkt),
            prereqs=concept_prereqs,
        )
        students.append(student)

    return students


def describe_population_v3(students: list[SimulatedStudentV3]) -> dict:
    n = len(students)
    if not students:
        return {"n_students": 0}

    all_concepts = list(students[0].concepts.keys())

    avg_p_know = {
        c: np.mean([s.concepts[c].p_know for s in students]) for c in all_concepts
    }
    avg_n_misc = np.mean([len(s.active_misconceptions()) for s in students])

    misc_counts: dict[str, int] = {}
    for s in students:
        for m in s.misconceptions:
            if m.p_active > 0.1:
                misc_counts[m.misconception_id] = misc_counts.get(m.misconception_id, 0) + 1

    return {
        "n_students": n,
        "avg_p_know": {k: round(v, 3) for k, v in avg_p_know.items()},
        "avg_active_misconceptions": round(float(avg_n_misc), 1),
        "misconception_prevalence": {
            k: round(v / n, 3)
            for k, v in sorted(misc_counts.items(), key=lambda x: -x[1])[:20]
        },
    }
