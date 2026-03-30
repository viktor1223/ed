"""Seed the database with demo data for development.

Run with: python -m api.seed
Login with: demo@school.edu / demo123
"""

from __future__ import annotations

import json
import random

from api.auth import hash_password
from api.database import get_db, init_db
from api.engine import engine

random.seed(42)  # Reproducible demo data

STUDENT_NAMES = [
    "Maria G", "James T", "Aiden K", "Sofia R", "Liam P",
    "Emma W", "Noah B", "Olivia M", "Lucas H", "Ava C",
    "Ethan D", "Isabella F", "Mason J", "Mia L", "Logan N",
    "Charlotte S", "Alexander V", "Amelia Z", "Benjamin A", "Harper E",
    "Elijah Q", "Evelyn U", "William I", "Abigail O", "Henry Y",
    "Emily X", "Sebastian R", "Ella T",
]

ARCHETYPES = [
    # (index_range, mastery_range, error_rate)
    (range(0, 8), (0.75, 0.95), 0.10),   # strong
    (range(8, 18), (0.50, 0.80), 0.30),   # average
    (range(18, 24), (0.25, 0.55), 0.50),  # struggling
    (range(24, 28), (0.15, 0.35), 0.70),  # foundation_gap
]


def _arch_for(i: int):
    for rng, mastery_range, error_rate in ARCHETYPES:
        if i in rng:
            return mastery_range, error_rate
    return (0.50, 0.80), 0.30


def seed():
    init_db()

    with get_db() as conn:
        if conn.execute("SELECT COUNT(*) FROM teachers").fetchone()[0] > 0:
            print("Database already has data. Skipping seed.")
            return

        # Teacher
        pw_hash = hash_password("demo123")
        conn.execute(
            "INSERT INTO teachers (name, email, password_hash) VALUES (?, ?, ?)",
            ("Ms. Johnson", "demo@school.edu", pw_hash),
        )
        teacher_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Classroom
        conn.execute(
            """INSERT INTO classrooms (teacher_id, name, grade_level, join_code,
                                      current_concept, next_concept)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (teacher_id, "Period 2 - Algebra I", "8th Grade", "ABC123",
             "distributive_property", "combining_like_terms"),
        )
        classroom_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        concept_order = [c.id for c in engine.kg.concepts_by_level()]
        misconception_ids_by_concept = {
            cid: [m.id for m in engine.kg.concepts[cid].misconceptions]
            for cid in concept_order
        }

        # Students + mastery
        student_ids = []
        student_mastery = {}  # sid -> {concept -> level}
        for i, name in enumerate(STUDENT_NAMES):
            conn.execute(
                "INSERT INTO students (classroom_id, name) VALUES (?, ?)",
                (classroom_id, name),
            )
            sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            student_ids.append(sid)
            mastery_range, _ = _arch_for(i)
            sm = {}
            for j, cid in enumerate(concept_order):
                level_penalty = j * 0.12
                low = max(0.05, mastery_range[0] - level_penalty)
                high = max(0.10, mastery_range[1] - level_penalty)
                level = round(random.uniform(low, high), 4)
                attempts = random.randint(4, 15) if j < 3 else random.randint(0, 5)
                conn.execute(
                    """INSERT INTO student_mastery (student_id, concept_id, mastery_level, attempts)
                       VALUES (?, ?, ?, ?)""",
                    (sid, cid, level, attempts),
                )
                sm[cid] = level
            student_mastery[sid] = sm

        # Assignment with problems from first 3 concepts
        problem_ids = []
        for cid in concept_order[:3]:
            for p in engine.get_problems_for_concept(cid)[:2]:
                problem_ids.append(p["problem_id"])
        conn.execute(
            "INSERT INTO assignments (classroom_id, title, problem_ids, status) VALUES (?, ?, ?, ?)",
            (classroom_id, "Diagnostic Check #1", json.dumps(problem_ids), "active"),
        )
        assignment_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Responses
        for i, sid in enumerate(student_ids):
            _, error_rate = _arch_for(i)
            sm = student_mastery[sid]
            for pid in problem_ids:
                prob = engine.get_problem(pid)
                if not prob:
                    continue
                cid = prob["concept"]
                mastery = sm.get(cid, 0.5)
                is_correct = random.random() > error_rate
                misconception = None
                confidence = 0.0
                if is_correct:
                    student_text = prob["correct_answer"]
                else:
                    miscs = misconception_ids_by_concept.get(cid, [])
                    misconception = random.choice(miscs) if miscs else None
                    confidence = round(random.uniform(0.6, 0.95), 3)
                    student_text = f"wrong_{pid}"
                conn.execute(
                    """INSERT INTO responses
                       (assignment_id, student_id, problem_id, student_text, correct,
                        classified_misconception, confidence, concept_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (assignment_id, sid, pid, student_text, int(is_correct),
                     misconception, confidence, cid),
                )

        # Alerts: cascade risk for foundation_gap students
        for sid in student_ids[24:]:
            name_row = conn.execute("SELECT name FROM students WHERE id = ?", (sid,)).fetchone()
            m_row = conn.execute(
                "SELECT mastery_level FROM student_mastery WHERE student_id = ? AND concept_id = 'integer_sign_ops'",
                (sid,),
            ).fetchone()
            if m_row and m_row["mastery_level"] < 0.35:
                downstream = engine.downstream_concepts("integer_sign_ops")
                names = [engine.kg.concepts[d].name for d in downstream]
                conn.execute(
                    """INSERT INTO alerts (student_id, classroom_id, alert_type, severity,
                                          message, recommendation, concept_id)
                       VALUES (?, ?, 'cascade_risk', 'critical', ?, ?, 'integer_sign_ops')""",
                    (sid, classroom_id,
                     f"{name_row['name']}: Integer & Sign Ops mastery is {m_row['mastery_level']:.2f}. Blocks: {', '.join(names)}.",
                     "Assign targeted sign operation problems with number line scaffolding."),
                )

        # Alerts: persistent misconceptions for struggling students
        for sid in student_ids[18:24]:
            row = conn.execute(
                """SELECT classified_misconception, COUNT(*) as cnt
                   FROM responses WHERE student_id = ? AND correct = 0
                     AND classified_misconception IS NOT NULL
                   GROUP BY classified_misconception
                   HAVING cnt >= 2
                   ORDER BY cnt DESC LIMIT 1""",
                (sid,),
            ).fetchone()
            if row:
                mid = row["classified_misconception"]
                conn.execute(
                    """INSERT INTO alerts (student_id, classroom_id, alert_type, severity,
                                          message, recommendation, misconception_id)
                       VALUES (?, ?, 'persistent_misconception', 'warning', ?, ?, ?)""",
                    (sid, classroom_id,
                     f"Has exhibited '{mid}' {row['cnt']} times.",
                     engine.get_intervention(mid),
                     mid),
                )

    print(f"Seeded: 1 teacher (demo@school.edu / demo123), "
          f"1 classroom (code: ABC123), {len(STUDENT_NAMES)} students, "
          f"{len(problem_ids)} problems assigned")


if __name__ == "__main__":
    seed()
