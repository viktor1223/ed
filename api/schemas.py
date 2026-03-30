"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, EmailStr


# --- Auth ---

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TeacherOut(BaseModel):
    id: int
    name: str
    email: str


# --- Classrooms ---

class ClassroomCreate(BaseModel):
    name: str
    grade_level: str | None = None


class ClassroomOut(BaseModel):
    id: int
    name: str
    grade_level: str | None
    join_code: str
    current_concept: str | None
    next_concept: str | None
    student_count: int = 0
    alert_count: int = 0


class ClassroomScheduleUpdate(BaseModel):
    current_concept: str | None = None
    next_concept: str | None = None


# --- Students ---

class StudentCreate(BaseModel):
    name: str


class StudentOut(BaseModel):
    id: int
    classroom_id: int
    name: str


class StudentJoin(BaseModel):
    join_code: str
    name: str


# --- Assignments ---

class AssignmentCreate(BaseModel):
    title: str
    problem_ids: list[str]
    due_at: str | None = None


class AssignmentOut(BaseModel):
    id: int
    classroom_id: int
    title: str
    problem_ids: list[str]
    status: str
    assigned_at: str
    due_at: str | None
    submission_count: int = 0
    student_count: int = 0


# --- Responses ---

class ResponseSubmit(BaseModel):
    student_id: int
    problem_id: str
    student_text: str


class ResponseOut(BaseModel):
    id: int
    student_id: int
    student_name: str
    problem_id: str
    problem_text: str
    student_text: str
    correct: bool
    classified_misconception: str | None
    confidence: float | None
    concept_id: str
    created_at: str


# --- Diagnostic views ---

class MasteryCell(BaseModel):
    concept_id: str
    mastery_level: float
    status: str  # mastered, progressing, struggling, critical
    misconception_id: str | None = None
    misconception_label: str | None = None


class StudentOverviewRow(BaseModel):
    student_id: int
    student_name: str
    mastery: list[MasteryCell]
    has_alert: bool


class ClassOverview(BaseModel):
    classroom_id: int
    classroom_name: str
    student_count: int
    on_track_pct: float
    alert_count: int
    top_misconception: str | None
    top_misconception_label: str | None
    top_misconception_count: int
    readiness_ready: int
    readiness_total: int
    next_concept: str | None
    students: list[StudentOverviewRow]


class EvidenceItem(BaseModel):
    student_name: str
    problem_text: str
    student_text: str
    correct_answer: str
    created_at: str


class MisconceptionGroup(BaseModel):
    misconception_id: str
    misconception_label: str
    concept_id: str
    concept_name: str
    student_count: int
    students: list[str]
    evidence: list[EvidenceItem]
    suggested_intervention: str


class ReadinessStudent(BaseModel):
    student_id: int
    student_name: str
    ready: bool
    prerequisite_mastery: float | None = None
    blocker_concept: str | None = None
    blocker_misconception: str | None = None
    suggested_action: str | None = None


class ReadinessCheck(BaseModel):
    target_concept: str
    target_concept_name: str
    prerequisite_concept: str | None
    ready_count: int
    not_ready_count: int
    total: int
    students: list[ReadinessStudent]
    recommendation: str


class AlertOut(BaseModel):
    id: int
    student_id: int
    student_name: str
    classroom_id: int
    alert_type: str
    severity: str
    message: str
    recommendation: str | None
    concept_id: str | None
    misconception_id: str | None
    acknowledged: bool
    resolved: bool
    created_at: str


class StudentCard(BaseModel):
    student_id: int
    student_name: str
    mastery: list[MasteryCell]
    active_misconceptions: list[dict]
    bottleneck: str | None
    recommended_action: str
    recommended_problems: list[str]


class TrajectoryPoint(BaseModel):
    date: str
    concept_id: str
    mastery_level: float
