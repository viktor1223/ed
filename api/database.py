"""SQLite database setup and schema."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ed.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS teachers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS classrooms (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                teacher_id  INTEGER NOT NULL REFERENCES teachers(id),
                name        TEXT NOT NULL,
                grade_level TEXT,
                join_code   TEXT UNIQUE NOT NULL,
                current_concept TEXT,
                next_concept    TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS students (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                classroom_id INTEGER NOT NULL REFERENCES classrooms(id),
                name        TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now')),
                UNIQUE(classroom_id, name)
            );

            CREATE TABLE IF NOT EXISTS assignments (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                classroom_id INTEGER NOT NULL REFERENCES classrooms(id),
                title       TEXT NOT NULL,
                problem_ids TEXT NOT NULL,
                status      TEXT DEFAULT 'active',
                assigned_at TEXT DEFAULT (datetime('now')),
                due_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS responses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                assignment_id INTEGER NOT NULL REFERENCES assignments(id),
                student_id  INTEGER NOT NULL REFERENCES students(id),
                problem_id  TEXT NOT NULL,
                student_text TEXT NOT NULL,
                correct     INTEGER NOT NULL,
                classified_misconception TEXT,
                confidence  REAL,
                concept_id  TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS student_mastery (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  INTEGER NOT NULL REFERENCES students(id),
                concept_id  TEXT NOT NULL,
                mastery_level REAL NOT NULL DEFAULT 0.5,
                attempts    INTEGER NOT NULL DEFAULT 0,
                updated_at  TEXT DEFAULT (datetime('now')),
                UNIQUE(student_id, concept_id)
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  INTEGER NOT NULL REFERENCES students(id),
                classroom_id INTEGER NOT NULL REFERENCES classrooms(id),
                alert_type  TEXT NOT NULL,
                severity    TEXT NOT NULL,
                message     TEXT NOT NULL,
                recommendation TEXT,
                concept_id  TEXT,
                misconception_id TEXT,
                acknowledged INTEGER DEFAULT 0,
                resolved    INTEGER DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now')),
                acknowledged_at TEXT,
                resolved_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_responses_student
                ON responses(student_id);
            CREATE INDEX IF NOT EXISTS idx_responses_assignment
                ON responses(assignment_id);
            CREATE INDEX IF NOT EXISTS idx_student_mastery_student
                ON student_mastery(student_id);
            CREATE INDEX IF NOT EXISTS idx_alerts_classroom
                ON alerts(classroom_id, resolved);
            CREATE INDEX IF NOT EXISTS idx_alerts_student
                ON alerts(student_id, resolved);
        """)
