"""FastAPI application - Teacher Misconception Diagnostic API.

Runs independently from the frontend. Communicates only via JSON over HTTP + SSE.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sqlite3
from collections import Counter
from datetime import datetime
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.auth import create_token, decode_token, hash_password, verify_password
from api.database import get_db, init_db
from api.engine import engine
from api.schemas import (
    AlertOut,
    AssignmentCreate,
    AssignmentOut,
    ClassOverview,
    ClassroomCreate,
    ClassroomOut,
    ClassroomScheduleUpdate,
    EvidenceItem,
    LoginRequest,
    MasteryCell,
    MisconceptionGroup,
    ReadinessCheck,
    ReadinessStudent,
    ResponseOut,
    ResponseSubmit,
    StudentCard,
    StudentCreate,
    StudentJoin,
    StudentOut,
    StudentOverviewRow,
    TeacherOut,
    TokenResponse,
    TrajectoryPoint,
)

app = FastAPI(title="Ed Diagnostic API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SSE event bus: classroom_id -> list of asyncio.Queue
_sse_subscribers: dict[int, list[asyncio.Queue]] = {}


@app.on_event("startup")
def startup():
    init_db()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def get_current_teacher(authorization: str = Header(...)) -> int:
    """Extract teacher_id from Bearer token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header")
    token = authorization[7:]
    teacher_id = decode_token(token)
    if teacher_id is None:
        raise HTTPException(401, "Invalid or expired token")
    return teacher_id


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth/register", response_model=TokenResponse)
def register(req: LoginRequest):
    with get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO teachers (name, email, password_hash) VALUES (?, ?, ?)",
                (req.email.split("@")[0], req.email, hash_password(req.password)),
            )
            teacher_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Email already registered")
    return TokenResponse(access_token=create_token(teacher_id))


@app.post("/api/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM teachers WHERE email = ?",
            (req.email,),
        ).fetchone()
    if not row or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    return TokenResponse(access_token=create_token(row["id"]))


@app.get("/api/auth/me", response_model=TeacherOut)
def me(teacher_id: int = Depends(get_current_teacher)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    return TeacherOut(id=row["id"], name=row["name"], email=row["email"])


# ---------------------------------------------------------------------------
# Classrooms
# ---------------------------------------------------------------------------

@app.get("/api/classrooms", response_model=list[ClassroomOut])
def list_classrooms(teacher_id: int = Depends(get_current_teacher)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM classrooms WHERE teacher_id = ?", (teacher_id,)
        ).fetchall()
        result = []
        for r in rows:
            sc = conn.execute(
                "SELECT COUNT(*) FROM students WHERE classroom_id = ?", (r["id"],)
            ).fetchone()[0]
            ac = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE classroom_id = ? AND resolved = 0",
                (r["id"],),
            ).fetchone()[0]
            result.append(ClassroomOut(
                id=r["id"], name=r["name"], grade_level=r["grade_level"],
                join_code=r["join_code"],
                current_concept=r["current_concept"],
                next_concept=r["next_concept"],
                student_count=sc, alert_count=ac,
            ))
    return result


@app.post("/api/classrooms", response_model=ClassroomOut)
def create_classroom(req: ClassroomCreate, teacher_id: int = Depends(get_current_teacher)):
    join_code = secrets.token_hex(3).upper()  # 6-char hex
    with get_db() as conn:
        conn.execute(
            "INSERT INTO classrooms (teacher_id, name, grade_level, join_code) VALUES (?, ?, ?, ?)",
            (teacher_id, req.name, req.grade_level, join_code),
        )
        cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return ClassroomOut(
        id=cid, name=req.name, grade_level=req.grade_level,
        join_code=join_code, current_concept=None, next_concept=None,
    )


@app.patch("/api/classrooms/{classroom_id}/schedule")
def update_schedule(
    classroom_id: int,
    req: ClassroomScheduleUpdate,
    teacher_id: int = Depends(get_current_teacher),
):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM classrooms WHERE id = ? AND teacher_id = ?",
            (classroom_id, teacher_id),
        ).fetchone()
        if not row:
            raise HTTPException(404)
        if req.current_concept is not None:
            conn.execute(
                "UPDATE classrooms SET current_concept = ? WHERE id = ?",
                (req.current_concept, classroom_id),
            )
        if req.next_concept is not None:
            conn.execute(
                "UPDATE classrooms SET next_concept = ? WHERE id = ?",
                (req.next_concept, classroom_id),
            )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

@app.get("/api/classrooms/{classroom_id}/students", response_model=list[StudentOut])
def list_students(classroom_id: int, teacher_id: int = Depends(get_current_teacher)):
    with get_db() as conn:
        _verify_classroom_owner(conn, classroom_id, teacher_id)
        rows = conn.execute(
            "SELECT * FROM students WHERE classroom_id = ? ORDER BY name",
            (classroom_id,),
        ).fetchall()
    return [StudentOut(id=r["id"], classroom_id=r["classroom_id"], name=r["name"]) for r in rows]


@app.post("/api/classrooms/{classroom_id}/students", response_model=StudentOut)
def add_student(classroom_id: int, req: StudentCreate, teacher_id: int = Depends(get_current_teacher)):
    with get_db() as conn:
        _verify_classroom_owner(conn, classroom_id, teacher_id)
        try:
            conn.execute(
                "INSERT INTO students (classroom_id, name) VALUES (?, ?)",
                (classroom_id, req.name),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Student name already exists in this class")
        sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        # Initialize mastery for all concepts
        for cid in engine.kg.concepts:
            p_init = engine.kg.concepts[cid].bkt_params.get("p_init", 0.5)
            conn.execute(
                "INSERT OR IGNORE INTO student_mastery (student_id, concept_id, mastery_level) VALUES (?, ?, ?)",
                (sid, cid, p_init),
            )
    return StudentOut(id=sid, classroom_id=classroom_id, name=req.name)


@app.post("/api/join", response_model=StudentOut)
def student_join(req: StudentJoin):
    """Student joins a class via code (no auth required)."""
    with get_db() as conn:
        classroom = conn.execute(
            "SELECT id FROM classrooms WHERE join_code = ?", (req.join_code,)
        ).fetchone()
        if not classroom:
            raise HTTPException(404, "Invalid class code")
        cid = classroom["id"]
        # Upsert: if student name exists, return existing
        existing = conn.execute(
            "SELECT * FROM students WHERE classroom_id = ? AND name = ?",
            (cid, req.name),
        ).fetchone()
        if existing:
            return StudentOut(id=existing["id"], classroom_id=cid, name=existing["name"])
        conn.execute("INSERT INTO students (classroom_id, name) VALUES (?, ?)", (cid, req.name))
        sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for concept_id in engine.kg.concepts:
            p_init = engine.kg.concepts[concept_id].bkt_params.get("p_init", 0.5)
            conn.execute(
                "INSERT OR IGNORE INTO student_mastery (student_id, concept_id, mastery_level) VALUES (?, ?, ?)",
                (sid, concept_id, p_init),
            )
    return StudentOut(id=sid, classroom_id=cid, name=req.name)


# ---------------------------------------------------------------------------
# Assignments
# ---------------------------------------------------------------------------

@app.get("/api/classrooms/{classroom_id}/assignments", response_model=list[AssignmentOut])
def list_assignments(classroom_id: int, teacher_id: int = Depends(get_current_teacher)):
    with get_db() as conn:
        _verify_classroom_owner(conn, classroom_id, teacher_id)
        rows = conn.execute(
            "SELECT * FROM assignments WHERE classroom_id = ? ORDER BY assigned_at DESC",
            (classroom_id,),
        ).fetchall()
        sc = conn.execute(
            "SELECT COUNT(*) FROM students WHERE classroom_id = ?", (classroom_id,)
        ).fetchone()[0]
        result = []
        for r in rows:
            sub_count = conn.execute(
                "SELECT COUNT(DISTINCT student_id) FROM responses WHERE assignment_id = ?",
                (r["id"],),
            ).fetchone()[0]
            result.append(AssignmentOut(
                id=r["id"], classroom_id=classroom_id, title=r["title"],
                problem_ids=json.loads(r["problem_ids"]),
                status=r["status"], assigned_at=r["assigned_at"],
                due_at=r["due_at"], submission_count=sub_count, student_count=sc,
            ))
    return result


@app.post("/api/classrooms/{classroom_id}/assignments", response_model=AssignmentOut)
def create_assignment(
    classroom_id: int,
    req: AssignmentCreate,
    teacher_id: int = Depends(get_current_teacher),
):
    with get_db() as conn:
        _verify_classroom_owner(conn, classroom_id, teacher_id)
        conn.execute(
            "INSERT INTO assignments (classroom_id, title, problem_ids, due_at) VALUES (?, ?, ?, ?)",
            (classroom_id, req.title, json.dumps(req.problem_ids), req.due_at),
        )
        aid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        sc = conn.execute(
            "SELECT COUNT(*) FROM students WHERE classroom_id = ?", (classroom_id,)
        ).fetchone()[0]
    return AssignmentOut(
        id=aid, classroom_id=classroom_id, title=req.title,
        problem_ids=req.problem_ids, status="active",
        assigned_at=datetime.utcnow().isoformat(), due_at=req.due_at,
        submission_count=0, student_count=sc,
    )


@app.get("/api/assignments/{assignment_id}/problems")
def get_assignment_problems(assignment_id: int):
    """Student-facing: get the problem list for an assignment (no auth)."""
    with get_db() as conn:
        row = conn.execute("SELECT problem_ids FROM assignments WHERE id = ?", (assignment_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    problem_ids = json.loads(row["problem_ids"])
    problems = []
    for pid in problem_ids:
        p = engine.get_problem(pid)
        if p:
            problems.append({"problem_id": p["problem_id"], "problem_text": p["problem_text"], "concept": p["concept"]})
    return problems


# ---------------------------------------------------------------------------
# Response submission (the core action)
# ---------------------------------------------------------------------------

@app.post("/api/assignments/{assignment_id}/responses", response_model=ResponseOut)
def submit_response(assignment_id: int, req: ResponseSubmit):
    """Student submits an answer. No auth (students join via code)."""
    problem = engine.get_problem(req.problem_id)
    if not problem:
        raise HTTPException(404, "Problem not found")

    # Classify
    is_correct = engine.check_correct(req.student_text, problem["correct_answer"])
    misconception_id = None
    confidence = 0.0

    if not is_correct:
        try:
            result = engine.classify_response(problem["problem_text"], req.student_text)
            misconception_id = result["label"]
            confidence = result["confidence"]
            if misconception_id == "correct":
                misconception_id = None
        except Exception:
            # Classifier not loaded or failed - still record the response
            pass

    concept_id = problem["concept"]

    with get_db() as conn:
        # Verify student exists
        student = conn.execute("SELECT * FROM students WHERE id = ?", (req.student_id,)).fetchone()
        if not student:
            raise HTTPException(404, "Student not found")

        # Insert response
        conn.execute(
            """INSERT INTO responses
               (assignment_id, student_id, problem_id, student_text, correct,
                classified_misconception, confidence, concept_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (assignment_id, req.student_id, req.problem_id, req.student_text,
             int(is_correct), misconception_id, confidence, concept_id),
        )
        response_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Update BKT mastery
        mastery_row = conn.execute(
            "SELECT mastery_level FROM student_mastery WHERE student_id = ? AND concept_id = ?",
            (req.student_id, concept_id),
        ).fetchone()
        current_mastery = mastery_row["mastery_level"] if mastery_row else 0.5
        new_mastery = engine.bkt_update(concept_id, current_mastery, is_correct, confidence)
        conn.execute(
            """INSERT INTO student_mastery (student_id, concept_id, mastery_level, attempts, updated_at)
               VALUES (?, ?, ?, 1, datetime('now'))
               ON CONFLICT(student_id, concept_id) DO UPDATE SET
               mastery_level = ?, attempts = attempts + 1, updated_at = datetime('now')""",
            (req.student_id, concept_id, new_mastery, new_mastery),
        )

        # Check and create alerts
        _check_alerts(conn, req.student_id, student["classroom_id"], concept_id, misconception_id, new_mastery)

        # Get response with student name for return
        student_name = student["name"]

    # Publish SSE event
    _publish_sse(student["classroom_id"], {
        "type": "student_response",
        "student_id": req.student_id,
        "student_name": student_name,
        "problem_id": req.problem_id,
        "problem_text": problem["problem_text"],
        "student_text": req.student_text,
        "correct": is_correct,
        "misconception": misconception_id,
        "concept_id": concept_id,
        "new_mastery": new_mastery,
    })

    return ResponseOut(
        id=response_id,
        student_id=req.student_id,
        student_name=student_name,
        problem_id=req.problem_id,
        problem_text=problem["problem_text"],
        student_text=req.student_text,
        correct=is_correct,
        classified_misconception=misconception_id,
        confidence=confidence,
        concept_id=concept_id,
        created_at=datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# Diagnostic views
# ---------------------------------------------------------------------------

@app.get("/api/classrooms/{classroom_id}/overview", response_model=ClassOverview)
def class_overview(classroom_id: int, teacher_id: int = Depends(get_current_teacher)):
    with get_db() as conn:
        classroom = _verify_classroom_owner(conn, classroom_id, teacher_id)

        students = conn.execute(
            "SELECT * FROM students WHERE classroom_id = ? ORDER BY name", (classroom_id,)
        ).fetchall()

        alert_count = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE classroom_id = ? AND resolved = 0",
            (classroom_id,),
        ).fetchone()[0]

        concept_order = [c.id for c in engine.kg.concepts_by_level()]
        student_rows = []
        on_track = 0
        misconception_counter: Counter = Counter()

        for s in students:
            mastery_rows = conn.execute(
                "SELECT concept_id, mastery_level FROM student_mastery WHERE student_id = ?",
                (s["id"],),
            ).fetchall()
            mastery_map = {r["concept_id"]: r["mastery_level"] for r in mastery_rows}

            # Latest misconception per concept
            misc_map: dict[str, tuple[str, str]] = {}
            for cid in concept_order:
                last_wrong = conn.execute(
                    """SELECT classified_misconception FROM responses
                       WHERE student_id = ? AND concept_id = ? AND correct = 0
                       AND classified_misconception IS NOT NULL
                       ORDER BY created_at DESC LIMIT 1""",
                    (s["id"], cid),
                ).fetchone()
                if last_wrong and last_wrong["classified_misconception"]:
                    mid = last_wrong["classified_misconception"]
                    misc_map[cid] = (mid, _misconception_label(mid))
                    misconception_counter[mid] += 1

            has_alert = bool(conn.execute(
                "SELECT 1 FROM alerts WHERE student_id = ? AND resolved = 0 LIMIT 1",
                (s["id"],),
            ).fetchone())

            cells = []
            student_on_track = True
            for cid in concept_order:
                level = mastery_map.get(cid, engine.kg.concepts[cid].bkt_params.get("p_init", 0.5))
                status = engine.mastery_status(level)
                if status in ("struggling", "critical"):
                    student_on_track = False
                mid, mlabel = misc_map.get(cid, (None, None))
                cells.append(MasteryCell(
                    concept_id=cid, mastery_level=round(level, 4),
                    status=status, misconception_id=mid, misconception_label=mlabel,
                ))

            if student_on_track:
                on_track += 1

            student_rows.append(StudentOverviewRow(
                student_id=s["id"], student_name=s["name"],
                mastery=cells, has_alert=has_alert,
            ))

        # Sort: students with alerts/critical first
        student_rows.sort(key=lambda r: (
            not r.has_alert,
            min((c.mastery_level for c in r.mastery), default=1.0),
        ))

        total = len(students)
        on_track_pct = round(on_track / total * 100, 1) if total > 0 else 0

        top_misc = misconception_counter.most_common(1)
        top_mid = top_misc[0][0] if top_misc else None
        top_mlabel = _misconception_label(top_mid) if top_mid else None
        top_mcount = top_misc[0][1] if top_misc else 0

        # Readiness for next concept
        next_concept = classroom["next_concept"]
        ready_count = 0
        if next_concept and next_concept in engine.kg.concepts:
            prereqs = engine.kg.prerequisites_of(next_concept)
            for s in students:
                all_met = True
                for prereq in prereqs:
                    m = conn.execute(
                        "SELECT mastery_level FROM student_mastery WHERE student_id = ? AND concept_id = ?",
                        (s["id"], prereq),
                    ).fetchone()
                    if not m or m["mastery_level"] < engine.kg.mastery_threshold:
                        all_met = False
                        break
                if all_met:
                    ready_count += 1

    return ClassOverview(
        classroom_id=classroom_id,
        classroom_name=classroom["name"],
        student_count=total,
        on_track_pct=on_track_pct,
        alert_count=alert_count,
        top_misconception=top_mid,
        top_misconception_label=top_mlabel,
        top_misconception_count=top_mcount,
        readiness_ready=ready_count,
        readiness_total=total,
        next_concept=next_concept,
        students=student_rows,
    )


@app.get("/api/classrooms/{classroom_id}/groups", response_model=list[MisconceptionGroup])
def misconception_groups(classroom_id: int, teacher_id: int = Depends(get_current_teacher)):
    with get_db() as conn:
        _verify_classroom_owner(conn, classroom_id, teacher_id)

        # Get recent wrong responses grouped by misconception
        rows = conn.execute(
            """SELECT r.classified_misconception, r.concept_id, r.problem_id,
                      r.student_text, r.created_at, s.name as student_name, s.id as student_id
               FROM responses r
               JOIN students s ON r.student_id = s.id
               WHERE s.classroom_id = ? AND r.correct = 0
                 AND r.classified_misconception IS NOT NULL
               ORDER BY r.created_at DESC""",
            (classroom_id,),
        ).fetchall()

        groups: dict[str, dict] = {}
        for r in rows:
            mid = r["classified_misconception"]
            if mid not in groups:
                concept = engine.kg.concepts.get(r["concept_id"])
                groups[mid] = {
                    "misconception_id": mid,
                    "misconception_label": _misconception_label(mid),
                    "concept_id": r["concept_id"],
                    "concept_name": concept.name if concept else r["concept_id"],
                    "students": set(),
                    "evidence": [],
                }
            groups[mid]["students"].add(r["student_name"])
            if len(groups[mid]["evidence"]) < 5:  # Cap evidence
                problem = engine.get_problem(r["problem_id"])
                groups[mid]["evidence"].append(EvidenceItem(
                    student_name=r["student_name"],
                    problem_text=problem["problem_text"] if problem else r["problem_id"],
                    student_text=r["student_text"],
                    correct_answer=problem["correct_answer"] if problem else "",
                    created_at=r["created_at"],
                ))

    result = []
    for g in groups.values():
        students = sorted(g["students"])
        result.append(MisconceptionGroup(
            misconception_id=g["misconception_id"],
            misconception_label=g["misconception_label"],
            concept_id=g["concept_id"],
            concept_name=g["concept_name"],
            student_count=len(students),
            students=students,
            evidence=g["evidence"],
            suggested_intervention=engine.get_intervention(g["misconception_id"]),
        ))
    result.sort(key=lambda g: -g.student_count)
    return result


@app.get("/api/classrooms/{classroom_id}/readiness", response_model=ReadinessCheck)
def readiness_check(
    classroom_id: int,
    target_concept: str | None = Query(None),
    teacher_id: int = Depends(get_current_teacher),
):
    with get_db() as conn:
        classroom = _verify_classroom_owner(conn, classroom_id, teacher_id)

        # Default to next_concept from schedule, or infer next in chain
        target = target_concept or classroom["next_concept"]
        if not target:
            # Pick the first concept not yet mastered by majority
            for c in engine.kg.concepts_by_level():
                target = c.id
                break
        if target not in engine.kg.concepts:
            raise HTTPException(400, f"Unknown concept: {target}")

        concept = engine.kg.concepts[target]
        prereqs = engine.kg.prerequisites_of(target)
        prereq_id = prereqs[0] if prereqs else None

        students = conn.execute(
            "SELECT * FROM students WHERE classroom_id = ?", (classroom_id,)
        ).fetchall()

        readiness_students = []
        ready_count = 0

        for s in students:
            if not prereqs:
                readiness_students.append(ReadinessStudent(
                    student_id=s["id"], student_name=s["name"], ready=True,
                    prerequisite_mastery=1.0,
                ))
                ready_count += 1
                continue

            all_met = True
            worst_prereq = None
            worst_mastery = 1.0
            for prereq in prereqs:
                m = conn.execute(
                    "SELECT mastery_level FROM student_mastery WHERE student_id = ? AND concept_id = ?",
                    (s["id"], prereq),
                ).fetchone()
                level = m["mastery_level"] if m else engine.kg.concepts[prereq].bkt_params.get("p_init", 0.5)
                if level < engine.kg.mastery_threshold:
                    all_met = False
                    if level < worst_mastery:
                        worst_mastery = level
                        worst_prereq = prereq

            if all_met:
                readiness_students.append(ReadinessStudent(
                    student_id=s["id"], student_name=s["name"], ready=True,
                    prerequisite_mastery=round(worst_mastery if worst_prereq else 1.0, 4),
                ))
                ready_count += 1
            else:
                # Find blocking misconception
                last_misc = conn.execute(
                    """SELECT classified_misconception FROM responses
                       WHERE student_id = ? AND concept_id = ? AND correct = 0
                       AND classified_misconception IS NOT NULL
                       ORDER BY created_at DESC LIMIT 1""",
                    (s["id"], worst_prereq),
                ).fetchone()
                blocker_misc = last_misc["classified_misconception"] if last_misc else None
                readiness_students.append(ReadinessStudent(
                    student_id=s["id"], student_name=s["name"], ready=False,
                    prerequisite_mastery=round(worst_mastery, 4),
                    blocker_concept=worst_prereq,
                    blocker_misconception=blocker_misc,
                    suggested_action=engine.get_intervention(blocker_misc) if blocker_misc else "Review prerequisite concepts.",
                ))

        total = len(students)
        not_ready = total - ready_count
        rec = f"{ready_count}/{total} students are ready for {concept.name}."
        if not_ready > 0:
            rec += f" {not_ready} students need prerequisite remediation."

    return ReadinessCheck(
        target_concept=target,
        target_concept_name=concept.name,
        prerequisite_concept=prereq_id,
        ready_count=ready_count,
        not_ready_count=not_ready,
        total=total,
        students=readiness_students,
        recommendation=rec,
    )


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@app.get("/api/classrooms/{classroom_id}/alerts", response_model=list[AlertOut])
def list_alerts(
    classroom_id: int,
    severity: str | None = Query(None),
    teacher_id: int = Depends(get_current_teacher),
):
    with get_db() as conn:
        _verify_classroom_owner(conn, classroom_id, teacher_id)
        query = """SELECT a.*, s.name as student_name
                   FROM alerts a JOIN students s ON a.student_id = s.id
                   WHERE a.classroom_id = ? AND a.resolved = 0"""
        params: list = [classroom_id]
        if severity:
            query += " AND a.severity = ?"
            params.append(severity)
        query += " ORDER BY CASE a.severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END, a.created_at DESC"
        rows = conn.execute(query, params).fetchall()
    return [AlertOut(
        id=r["id"], student_id=r["student_id"], student_name=r["student_name"],
        classroom_id=r["classroom_id"], alert_type=r["alert_type"],
        severity=r["severity"], message=r["message"],
        recommendation=r["recommendation"], concept_id=r["concept_id"],
        misconception_id=r["misconception_id"],
        acknowledged=bool(r["acknowledged"]), resolved=bool(r["resolved"]),
        created_at=r["created_at"],
    ) for r in rows]


@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, teacher_id: int = Depends(get_current_teacher)):
    with get_db() as conn:
        conn.execute(
            "UPDATE alerts SET acknowledged = 1, acknowledged_at = datetime('now') WHERE id = ?",
            (alert_id,),
        )
    return {"ok": True}


@app.post("/api/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int, teacher_id: int = Depends(get_current_teacher)):
    with get_db() as conn:
        conn.execute(
            "UPDATE alerts SET resolved = 1, resolved_at = datetime('now') WHERE id = ?",
            (alert_id,),
        )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Student detail
# ---------------------------------------------------------------------------

@app.get("/api/students/{student_id}/card", response_model=StudentCard)
def student_card(student_id: int, teacher_id: int = Depends(get_current_teacher)):
    with get_db() as conn:
        student = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
        if not student:
            raise HTTPException(404)
        _verify_classroom_owner(conn, student["classroom_id"], teacher_id)

        concept_order = [c.id for c in engine.kg.concepts_by_level()]

        mastery_rows = conn.execute(
            "SELECT concept_id, mastery_level, attempts FROM student_mastery WHERE student_id = ?",
            (student_id,),
        ).fetchall()
        mastery_map = {r["concept_id"]: r for r in mastery_rows}

        cells = []
        worst_concept = None
        worst_level = 1.0
        for cid in concept_order:
            m = mastery_map.get(cid)
            level = m["mastery_level"] if m else engine.kg.concepts[cid].bkt_params.get("p_init", 0.5)
            status = engine.mastery_status(level)
            cells.append(MasteryCell(concept_id=cid, mastery_level=round(level, 4), status=status))
            if level < worst_level and (m and m["attempts"] > 0):
                worst_level = level
                worst_concept = cid

        # Active misconceptions with evidence
        active_misconceptions = []
        for cid in concept_order:
            miscs = conn.execute(
                """SELECT classified_misconception, COUNT(*) as cnt,
                          MAX(created_at) as last_seen
                   FROM responses
                   WHERE student_id = ? AND concept_id = ? AND correct = 0
                     AND classified_misconception IS NOT NULL
                   GROUP BY classified_misconception
                   ORDER BY cnt DESC""",
                (student_id, cid),
            ).fetchall()
            for m in miscs:
                evidence = conn.execute(
                    """SELECT r.problem_id, r.student_text, r.created_at
                       FROM responses r
                       WHERE r.student_id = ? AND r.classified_misconception = ?
                       ORDER BY r.created_at DESC LIMIT 5""",
                    (student_id, m["classified_misconception"]),
                ).fetchall()
                ev_list = []
                for e in evidence:
                    prob = engine.get_problem(e["problem_id"])
                    ev_list.append({
                        "problem_text": prob["problem_text"] if prob else e["problem_id"],
                        "student_text": e["student_text"],
                        "correct_answer": prob["correct_answer"] if prob else "",
                        "date": e["created_at"],
                    })
                active_misconceptions.append({
                    "misconception_id": m["classified_misconception"],
                    "label": _misconception_label(m["classified_misconception"]),
                    "concept_id": cid,
                    "times_observed": m["cnt"],
                    "last_seen": m["last_seen"],
                    "evidence": ev_list,
                })

        # Bottleneck
        bottleneck = None
        if worst_concept:
            downstream = engine.downstream_concepts(worst_concept)
            if downstream:
                names = [engine.kg.concepts[d].name for d in downstream]
                bottleneck = f"{engine.kg.concepts[worst_concept].name} is blocking {', '.join(names)}"

        # Recommendation
        if worst_concept:
            rec_action = f"Focus on {engine.kg.concepts[worst_concept].name}."
            if active_misconceptions:
                top = active_misconceptions[0]
                rec_action += f" Address: {top['label']}."
        else:
            rec_action = "Student is on track. Continue to next concept."

        rec_problems = engine.recommend_problems(worst_concept) if worst_concept else []

    return StudentCard(
        student_id=student_id,
        student_name=student["name"],
        mastery=cells,
        active_misconceptions=active_misconceptions,
        bottleneck=bottleneck,
        recommended_action=rec_action,
        recommended_problems=rec_problems,
    )


@app.get("/api/students/{student_id}/trajectory", response_model=list[TrajectoryPoint])
def student_trajectory(student_id: int, teacher_id: int = Depends(get_current_teacher)):
    """Mastery over time for charting."""
    with get_db() as conn:
        student = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
        if not student:
            raise HTTPException(404)
        _verify_classroom_owner(conn, student["classroom_id"], teacher_id)

        # Reconstruct trajectory from responses
        responses = conn.execute(
            "SELECT * FROM responses WHERE student_id = ? ORDER BY created_at",
            (student_id,),
        ).fetchall()

        # Replay BKT
        mastery_state: dict[str, float] = {
            cid: engine.kg.concepts[cid].bkt_params.get("p_init", 0.5)
            for cid in engine.kg.concepts
        }
        points = []
        for r in responses:
            cid = r["concept_id"]
            mastery_state[cid] = engine.bkt_update(
                cid, mastery_state[cid], bool(r["correct"]),
                r["confidence"] or 0.0,
            )
            points.append(TrajectoryPoint(
                date=r["created_at"],
                concept_id=cid,
                mastery_level=mastery_state[cid],
            ))
    return points


@app.get("/api/students/{student_id}/responses", response_model=list[ResponseOut])
def student_responses(
    student_id: int,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    teacher_id: int = Depends(get_current_teacher),
):
    with get_db() as conn:
        student = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
        if not student:
            raise HTTPException(404)
        _verify_classroom_owner(conn, student["classroom_id"], teacher_id)
        rows = conn.execute(
            """SELECT * FROM responses WHERE student_id = ?
               ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (student_id, limit, offset),
        ).fetchall()
    result = []
    for r in rows:
        prob = engine.get_problem(r["problem_id"])
        result.append(ResponseOut(
            id=r["id"], student_id=student_id, student_name=student["name"],
            problem_id=r["problem_id"],
            problem_text=prob["problem_text"] if prob else r["problem_id"],
            student_text=r["student_text"],
            correct=bool(r["correct"]),
            classified_misconception=r["classified_misconception"],
            confidence=r["confidence"],
            concept_id=r["concept_id"],
            created_at=r["created_at"],
        ))
    return result


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------

@app.get("/api/problems")
def list_problems(concept: str | None = None, difficulty: str | None = None):
    if concept:
        problems = engine.get_problems_for_concept(concept, difficulty)
    else:
        problems = []
        for c in engine.kg.concepts:
            problems.extend(engine.get_problems_for_concept(c, difficulty))
    return problems


# ---------------------------------------------------------------------------
# SSE live feed
# ---------------------------------------------------------------------------

@app.get("/api/classrooms/{classroom_id}/live")
async def live_feed(classroom_id: int, request: Request):
    """SSE endpoint for real-time classroom updates."""
    queue: asyncio.Queue = asyncio.Queue()
    if classroom_id not in _sse_subscribers:
        _sse_subscribers[classroom_id] = []
    _sse_subscribers[classroom_id].append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _sse_subscribers[classroom_id].remove(queue)
            if not _sse_subscribers[classroom_id]:
                del _sse_subscribers[classroom_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _publish_sse(classroom_id: int, data: dict):
    """Push an event to all SSE subscribers for a classroom."""
    if classroom_id in _sse_subscribers:
        for queue in _sse_subscribers[classroom_id]:
            queue.put_nowait(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_classroom_owner(conn, classroom_id: int, teacher_id: int) -> dict:
    row = conn.execute(
        "SELECT * FROM classrooms WHERE id = ? AND teacher_id = ?",
        (classroom_id, teacher_id),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Classroom not found")
    return row


def _misconception_label(misconception_id: str | None) -> str | None:
    if not misconception_id:
        return None
    for concept in engine.kg.concepts.values():
        for m in concept.misconceptions:
            if m.id == misconception_id:
                return m.label
    return misconception_id


def _check_alerts(conn, student_id: int, classroom_id: int, concept_id: str, misconception_id: str | None, mastery: float):
    """Create or auto-resolve alerts after a response."""
    # Count active alerts for this classroom
    active_count = conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE classroom_id = ? AND resolved = 0",
        (classroom_id,),
    ).fetchone()[0]

    # C3: Cascade risk - Level 1 concept below threshold
    concept = engine.kg.concepts[concept_id]
    if concept.level == 1 and mastery < 0.35:
        downstream = engine.downstream_concepts(concept_id)
        if downstream and active_count < 5:
            existing = conn.execute(
                "SELECT 1 FROM alerts WHERE student_id = ? AND alert_type = 'cascade_risk' AND concept_id = ? AND resolved = 0",
                (student_id, concept_id),
            ).fetchone()
            if not existing:
                names = [engine.kg.concepts[d].name for d in downstream]
                conn.execute(
                    """INSERT INTO alerts (student_id, classroom_id, alert_type, severity, message, recommendation, concept_id)
                       VALUES (?, ?, 'cascade_risk', 'critical', ?, ?, ?)""",
                    (student_id, classroom_id,
                     f"{concept.name} mastery is {mastery:.2f}. Blocks: {', '.join(names)}.",
                     engine.get_intervention(misconception_id) if misconception_id else "Review foundation concepts.",
                     concept_id),
                )

    # C2: Persistent misconception - 3+ occurrences
    if misconception_id:
        count = conn.execute(
            "SELECT COUNT(*) FROM responses WHERE student_id = ? AND classified_misconception = ?",
            (student_id, misconception_id),
        ).fetchone()[0]
        if count >= 3 and active_count < 5:
            existing = conn.execute(
                "SELECT 1 FROM alerts WHERE student_id = ? AND alert_type = 'persistent_misconception' AND misconception_id = ? AND resolved = 0",
                (student_id, misconception_id),
            ).fetchone()
            if not existing:
                label = _misconception_label(misconception_id) or misconception_id
                conn.execute(
                    """INSERT INTO alerts (student_id, classroom_id, alert_type, severity, message, recommendation, concept_id, misconception_id)
                       VALUES (?, ?, 'persistent_misconception', 'warning', ?, ?, ?, ?)""",
                    (student_id, classroom_id,
                     f"Has exhibited '{label}' {count} times.",
                     engine.get_intervention(misconception_id),
                     concept_id, misconception_id),
                )

    # Auto-resolve: prerequisite gap alerts if mastery now above threshold
    if mastery >= engine.kg.mastery_threshold:
        conn.execute(
            """UPDATE alerts SET resolved = 1, resolved_at = datetime('now')
               WHERE student_id = ? AND concept_id = ? AND alert_type = 'prerequisite_gap' AND resolved = 0""",
            (student_id, concept_id),
        )
        # Also auto-resolve persistent misconception if student got 2 correct in a row
        recent = conn.execute(
            """SELECT correct FROM responses WHERE student_id = ? AND concept_id = ?
               ORDER BY created_at DESC LIMIT 2""",
            (student_id, concept_id),
        ).fetchall()
        if len(recent) >= 2 and all(r["correct"] for r in recent):
            conn.execute(
                """UPDATE alerts SET resolved = 1, resolved_at = datetime('now')
                   WHERE student_id = ? AND concept_id = ? AND alert_type = 'persistent_misconception' AND resolved = 0""",
                (student_id, concept_id),
            )
