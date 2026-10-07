"""SQLite storage for patients and per-eye examinations (standard library sqlite3 only)."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id    TEXT PRIMARY KEY,
    age           INTEGER,
    sex           TEXT CHECK (sex IN ('M','F','O','U')),
    created_date  TEXT NOT NULL,          -- date the patient was first registered
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS examinations (
    examination_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id                 TEXT NOT NULL,     -- groups the left + right eye of one upload
    patient_id               TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    exam_date                TEXT NOT NULL,     -- YYYY-MM-DD (date the photo was taken)
    eye                      TEXT NOT NULL CHECK (eye IN ('L','R')),
    age_at_exam              INTEGER,
    image_path               TEXT NOT NULL,
    heatmap_path             TEXT,
    heatmap_class            TEXT,
    disease_probs            TEXT,              -- JSON {code: probability}
    disease_uncertainty      TEXT,              -- JSON {code: MC-dropout std}
    disease_thresholds       TEXT,              -- JSON {code: threshold}
    disease_confidence       REAL,
    predicted_se             REAL,              -- estimated spherical equivalent (D)
    se_uncertainty           REAL,              -- predictive std (D)
    se_interval_low          REAL,
    se_interval_high         REAL,
    se_confidence            REAL,
    quality                  TEXT,              -- JSON quality report
    disease_model_version    TEXT,
    refractive_model_version TEXT,
    disease_is_demo          INTEGER DEFAULT 0,
    refractive_is_demo       INTEGER DEFAULT 0,
    timestamp                TEXT NOT NULL      -- when the analysis ran
);
CREATE INDEX IF NOT EXISTS idx_exam_patient ON examinations(patient_id, exam_date);
CREATE INDEX IF NOT EXISTS idx_exam_visit ON examinations(visit_id);
"""

JSON_COLS = ("disease_probs", "disease_uncertainty", "disease_thresholds", "quality")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    @staticmethod
    def _row(r):
        d = dict(r)
        for k in JSON_COLS:
            if d.get(k):
                d[k] = json.loads(d[k])
        return d

    # ------------------------------------------------------------------ patients
    def upsert_patient(self, patient_id, age, sex):
        now = dt.datetime.now().isoformat(timespec="seconds")
        with self.conn() as c:
            c.execute("""INSERT INTO patients(patient_id, age, sex, created_date, updated_at)
                         VALUES (?,?,?,?,?)
                         ON CONFLICT(patient_id) DO UPDATE SET
                           age=COALESCE(excluded.age, patients.age),
                           sex=COALESCE(excluded.sex, patients.sex),
                           updated_at=excluded.updated_at""",
                      (patient_id, age, sex, dt.date.today().isoformat(), now))

    def get_patient(self, patient_id):
        with self.conn() as c:
            r = c.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
            return dict(r) if r else None

    def list_patients(self):
        with self.conn() as c:
            rows = c.execute("""
                SELECT p.*, COUNT(DISTINCT e.visit_id) AS n_visits, MAX(e.exam_date) AS last_exam
                FROM patients p LEFT JOIN examinations e ON e.patient_id = p.patient_id
                GROUP BY p.patient_id ORDER BY last_exam DESC, p.patient_id""").fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------- examinations
    def add_exam(self, rec: dict) -> int:
        rec = dict(rec)
        for k in JSON_COLS:
            if k in rec and rec[k] is not None:
                rec[k] = json.dumps(rec[k])
        rec.setdefault("timestamp", dt.datetime.now().isoformat(timespec="seconds"))
        cols = ",".join(rec)
        with self.conn() as c:
            cur = c.execute(f"INSERT INTO examinations({cols}) VALUES ({','.join('?' * len(rec))})",
                            tuple(rec.values()))
            return cur.lastrowid

    def exams_for_patient(self, patient_id):
        with self.conn() as c:
            rows = c.execute("""SELECT * FROM examinations WHERE patient_id=?
                                ORDER BY exam_date, eye""", (patient_id,)).fetchall()
            return [self._row(r) for r in rows]

    def exams_for_visit(self, visit_id):
        with self.conn() as c:
            rows = c.execute("SELECT * FROM examinations WHERE visit_id=? ORDER BY eye DESC",
                             (visit_id,)).fetchall()
            return [self._row(r) for r in rows]

    def delete_visit(self, visit_id):
        with self.conn() as c:
            rows = c.execute("SELECT image_path, heatmap_path FROM examinations WHERE visit_id=?",
                             (visit_id,)).fetchall()
            c.execute("DELETE FROM examinations WHERE visit_id=?", (visit_id,))
            return [dict(r) for r in rows]

    def recent_visits(self, limit=8):
        with self.conn() as c:
            rows = c.execute("""SELECT visit_id, patient_id, exam_date, MAX(timestamp) AS ts
                                FROM examinations GROUP BY visit_id
                                ORDER BY exam_date DESC, ts DESC LIMIT ?""", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def latest_per_patient_eye(self):
        with self.conn() as c:
            rows = c.execute("""
                SELECT e.* FROM examinations e
                JOIN (SELECT patient_id, eye, MAX(exam_date || timestamp) AS k
                      FROM examinations GROUP BY patient_id, eye) last
                  ON last.patient_id = e.patient_id AND last.eye = e.eye
                 AND (e.exam_date || e.timestamp) = last.k""").fetchall()
            return [self._row(r) for r in rows]

    def counts(self):
        with self.conn() as c:
            return {
                "patients": c.execute("SELECT COUNT(*) FROM patients").fetchone()[0],
                "visits": c.execute("SELECT COUNT(DISTINCT visit_id) FROM examinations").fetchone()[0],
                "eye_examinations": c.execute("SELECT COUNT(*) FROM examinations").fetchone()[0],
            }
