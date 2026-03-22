"""Unit tests for KnowledgeGraph, StudentState, and next_action."""

import json
import tempfile
from pathlib import Path

import pytest

from knowledge_graph import KnowledgeGraph, StudentState, next_action


@pytest.fixture
def kg_path():
    return Path(__file__).resolve().parent.parent / "data" / "knowledge_graph.json"


@pytest.fixture
def kg(kg_path):
    return KnowledgeGraph.from_json(kg_path)


@pytest.fixture
def state(kg):
    return StudentState(kg)


# ─── KnowledgeGraph loading ──────────────────────────────────────────────────

class TestKnowledgeGraph:
    def test_loads_all_concepts(self, kg):
        assert len(kg.concepts) == 5

    def test_concept_names(self, kg):
        expected = {
            "integer_sign_ops",
            "order_of_operations",
            "distributive_property",
            "combining_like_terms",
            "solving_linear_equations",
        }
        assert set(kg.concepts.keys()) == expected

    def test_prerequisite_chain(self, kg):
        assert kg.prerequisites_of("integer_sign_ops") == []
        assert kg.prerequisites_of("order_of_operations") == ["integer_sign_ops"]
        assert kg.prerequisites_of("solving_linear_equations") == ["combining_like_terms"]

    def test_concepts_by_level(self, kg):
        ordered = kg.concepts_by_level()
        levels = [c.level for c in ordered]
        assert levels == sorted(levels)
        assert ordered[0].id == "integer_sign_ops"
        assert ordered[-1].id == "solving_linear_equations"

    def test_misconception_count(self, kg):
        all_ids = kg.all_misconception_ids()
        assert len(all_ids) == 19

    def test_misconception_to_concept(self, kg):
        assert kg.misconception_to_concept("sign_sum_negatives") == "integer_sign_ops"
        assert kg.misconception_to_concept("oo_left_to_right") == "order_of_operations"
        assert kg.misconception_to_concept("nonexistent") is None

    def test_label_list(self, kg):
        labels = kg.label_list()
        assert "correct" in labels
        assert len(labels) == 20  # 19 misconceptions + correct
        assert labels == sorted(labels)

    def test_edges(self, kg):
        assert len(kg.edges) == 4

    def test_each_concept_has_bkt_params(self, kg):
        for concept in kg.concepts.values():
            assert "p_init" in concept.bkt_params
            assert "p_learn" in concept.bkt_params
            assert "p_guess" in concept.bkt_params
            assert "p_slip" in concept.bkt_params

    def test_each_concept_has_misconceptions(self, kg):
        for concept in kg.concepts.values():
            assert len(concept.misconceptions) >= 3


# ─── StudentState BKT ────────────────────────────────────────────────────────

class TestStudentState:
    def test_initial_mastery(self, state, kg):
        """Initial mastery should match p_init from BKT params."""
        for cid, concept in kg.concepts.items():
            assert state.mastery[cid] == concept.bkt_params["p_init"]

    def test_correct_increases_mastery(self, state):
        initial = state.mastery["integer_sign_ops"]
        state.update("integer_sign_ops", correct=True)
        assert state.mastery["integer_sign_ops"] > initial

    def test_incorrect_does_not_increase_mastery(self, state):
        initial = state.mastery["integer_sign_ops"]
        state.update("integer_sign_ops", correct=False, confidence=0.9)
        assert state.mastery["integer_sign_ops"] <= initial

    def test_three_correct_pushes_above_threshold(self, state, kg):
        """3 correct answers in a row should push mastery above 0.8 (PLAYBOOK requirement)."""
        cid = "integer_sign_ops"
        for _ in range(3):
            state.update(cid, correct=True)
        assert state.mastery[cid] > 0.80

    def test_many_correct_reaches_mastery(self, state, kg):
        """Enough correct answers should reach mastery threshold."""
        cid = "integer_sign_ops"
        for _ in range(10):
            state.update(cid, correct=True)
        assert state.is_mastered(cid)

    def test_attempts_tracked(self, state):
        assert state.attempts["integer_sign_ops"] == 0
        state.update("integer_sign_ops", correct=True)
        assert state.attempts["integer_sign_ops"] == 1
        state.update("integer_sign_ops", correct=False)
        assert state.attempts["integer_sign_ops"] == 2

    def test_mastery_stays_bounded(self, state):
        cid = "integer_sign_ops"
        # Many wrong answers should not push below 0
        for _ in range(50):
            state.update(cid, correct=False, confidence=1.0)
        assert state.mastery[cid] >= 0.01

        # Many correct answers should not push above 1
        for _ in range(100):
            state.update(cid, correct=True)
        assert state.mastery[cid] <= 1.0

    def test_confidence_scaling(self, state):
        """Low confidence incorrect should penalize less than high confidence."""
        cid_lo = "integer_sign_ops"
        cid_hi = "order_of_operations"
        state.update(cid_lo, correct=False, confidence=0.3)
        state.update(cid_hi, correct=False, confidence=0.95)
        # High confidence wrong should result in lower mastery
        assert state.mastery[cid_hi] <= state.mastery[cid_lo]

    def test_summary(self, state):
        state.update("integer_sign_ops", correct=True)
        s = state.summary()
        assert "integer_sign_ops" in s
        assert s["integer_sign_ops"]["attempts"] == 1
        assert isinstance(s["integer_sign_ops"]["mastery"], float)
        assert isinstance(s["integer_sign_ops"]["mastered"], bool)


# ─── Prerequisites ────────────────────────────────────────────────────────────

class TestPrerequisites:
    def test_first_concept_prerequisites_met(self, state):
        assert state.prerequisites_met("integer_sign_ops") is True

    def test_second_concept_prerequisites_not_met(self, state):
        assert state.prerequisites_met("order_of_operations") is False

    def test_prerequisites_met_after_mastery(self, state):
        # Master integer_sign_ops
        for _ in range(10):
            state.update("integer_sign_ops", correct=True)
        assert state.prerequisites_met("order_of_operations") is True

    def test_deep_prerequisite_chain(self, state):
        """solving_linear_equations needs combining_like_terms mastered,
        which needs distributive_property, etc."""
        assert state.prerequisites_met("solving_linear_equations") is False
        # Master the whole chain
        for cid in ["integer_sign_ops", "order_of_operations",
                     "distributive_property", "combining_like_terms"]:
            for _ in range(10):
                state.update(cid, correct=True)
        assert state.prerequisites_met("solving_linear_equations") is True


# ─── next_action ──────────────────────────────────────────────────────────────

class TestNextAction:
    def test_no_attempts_returns_start(self, state, kg):
        result = next_action(state, kg)
        assert result["action"] == "start"
        assert result["concept"] == "integer_sign_ops"

    def test_after_wrong_answer_returns_remediate(self, state, kg):
        state.update("integer_sign_ops", correct=False)
        result = next_action(state, kg)
        assert result["action"] == "remediate"
        assert result["concept"] == "integer_sign_ops"

    def test_mastered_concept_progresses(self, state, kg):
        for _ in range(10):
            state.update("integer_sign_ops", correct=True)
        result = next_action(state, kg)
        assert result["action"] == "progress"
        assert result["concept"] == "order_of_operations"

    def test_prerequisite_gating(self, state, kg):
        """Cannot progress to level 3 without mastering level 2."""
        for _ in range(10):
            state.update("integer_sign_ops", correct=True)
        # order_of_operations not mastered yet, but int_sign_ops is
        result = next_action(state, kg)
        assert result["concept"] == "order_of_operations"
        # Even if we attempt distrib directly, order_of_ops should be remediated first
        state.update("order_of_operations", correct=False)
        result = next_action(state, kg)
        assert result["action"] == "remediate"
        assert result["concept"] == "order_of_operations"

    def test_all_mastered_returns_review(self, state, kg):
        for cid in kg.concepts:
            for _ in range(10):
                state.update(cid, correct=True)
        result = next_action(state, kg)
        assert result["action"] == "review"

    def test_remediation_prioritizes_lowest_level(self, state, kg):
        """If multiple concepts need remediation, pick the lowest level."""
        # Give both concepts wrong answers so both are below threshold
        state.update("integer_sign_ops", correct=False)
        state.update("order_of_operations", correct=False)
        result = next_action(state, kg)
        # Should remediate integer_sign_ops (level 1) before OoO (level 2)
        assert result["concept"] == "integer_sign_ops"


# ─── Mini KG edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def _mini_kg(self) -> KnowledgeGraph:
        data = {
            "metadata": {"mastery_threshold": 0.85, "mastery_initial": 0.5},
            "concepts": [
                {
                    "id": "only_concept",
                    "name": "Only Concept",
                    "description": "Test",
                    "level": 1,
                    "prerequisites": [],
                    "mae_ids": [],
                    "bkt_params": {"p_init": 0.15, "p_learn": 0.15, "p_guess": 0.10, "p_slip": 0.10},
                    "misconceptions": [
                        {"id": "m1", "label": "Test", "description": "Test", "examples": []},
                    ],
                }
            ],
            "edges": [],
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, tmp)
        tmp.close()
        return KnowledgeGraph.from_json(tmp.name)

    def test_single_concept_start(self):
        kg = self._mini_kg()
        state = StudentState(kg)
        result = next_action(state, kg)
        assert result["action"] == "start"
        assert result["concept"] == "only_concept"

    def test_single_concept_mastered(self):
        kg = self._mini_kg()
        state = StudentState(kg)
        for _ in range(10):
            state.update("only_concept", correct=True)
        result = next_action(state, kg)
        assert result["action"] == "review"
        assert result["concept"] == "only_concept"
