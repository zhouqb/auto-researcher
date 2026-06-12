"""Experience store (design §5): success AND failure trajectories, cross-project.

SQLite + FTS5 keyword retrieval (top-k; LLM-side reranking happens naturally
because the planner reads the hits). Guardrails against self-reinforcing
error: confidence scores, supersede links, and recency ordering — a single
failure never blacklists an approach without re-test evidence.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
  experience_id  TEXT PRIMARY KEY,
  project_id     TEXT NOT NULL,
  iteration      INTEGER,
  branch         TEXT,
  hypothesis     TEXT NOT NULL,
  method         TEXT,                -- JSON {dataset, model/approach, key_params, code_artifact_ref}
  result         TEXT,                -- JSON {metric, value, baseline, ...}
  outcome        TEXT NOT NULL,       -- success | failure | inconclusive | aborted
  failure_mode   TEXT,                -- OOM | data_leakage | no_improvement | bug | divergence | null
  lessons        TEXT NOT NULL,
  codex_thread_id TEXT,
  tokens_used    INTEGER,
  wallclock_s    REAL,
  confidence     REAL DEFAULT 0.7,
  supersedes     TEXT,                -- JSON list of experience_ids
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts
  USING fts5(experience_id UNINDEXED, hypothesis, method, lessons, failure_mode);
"""

VALID_OUTCOMES = {"success", "failure", "inconclusive", "aborted"}


@dataclass
class Experience:
    experience_id: str
    project_id: str
    hypothesis: str
    outcome: str
    lessons: str
    iteration: Optional[int] = None
    branch: Optional[str] = None
    method: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    failure_mode: Optional[str] = None
    codex_thread_id: Optional[str] = None
    tokens_used: Optional[int] = None
    wallclock_s: Optional[float] = None
    confidence: float = 0.7
    supersedes: list[str] = field(default_factory=list)
    created_at: Optional[str] = None

    def brief(self) -> dict[str, Any]:
        """Compact form injected into agent context."""
        return {
            "experience_id": self.experience_id,
            "outcome": self.outcome,
            "failure_mode": self.failure_mode,
            "hypothesis": self.hypothesis,
            "method": self.method,
            "result": self.result,
            "lessons": self.lessons,
            "confidence": self.confidence,
            "superseded": False,
            "created_at": self.created_at,
        }


def _row_to_experience(row: sqlite3.Row) -> Experience:
    return Experience(
        experience_id=row["experience_id"],
        project_id=row["project_id"],
        iteration=row["iteration"],
        branch=row["branch"],
        hypothesis=row["hypothesis"],
        method=json.loads(row["method"]) if row["method"] else {},
        result=json.loads(row["result"]) if row["result"] else {},
        outcome=row["outcome"],
        failure_mode=row["failure_mode"],
        lessons=row["lessons"],
        codex_thread_id=row["codex_thread_id"],
        tokens_used=row["tokens_used"],
        wallclock_s=row["wallclock_s"],
        confidence=row["confidence"] if row["confidence"] is not None else 0.7,
        supersedes=json.loads(row["supersedes"]) if row["supersedes"] else [],
        created_at=row["created_at"],
    )


class ExperienceStore:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def record(
        self,
        *,
        project_id: str,
        hypothesis: str,
        outcome: str,
        lessons: str,
        method: Optional[dict[str, Any]] = None,
        result: Optional[dict[str, Any]] = None,
        failure_mode: Optional[str] = None,
        iteration: Optional[int] = None,
        branch: Optional[str] = None,
        codex_thread_id: Optional[str] = None,
        tokens_used: Optional[int] = None,
        wallclock_s: Optional[float] = None,
        confidence: float = 0.7,
        supersedes: Optional[list[str]] = None,
    ) -> Experience:
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(VALID_OUTCOMES)}")
        exp_id = "exp_" + uuid.uuid4().hex[:12]
        method_json = json.dumps(method or {})
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO experiences
                   (experience_id, project_id, iteration, branch, hypothesis,
                    method, result, outcome, failure_mode, lessons,
                    codex_thread_id, tokens_used, wallclock_s, confidence, supersedes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    exp_id, project_id, iteration, branch, hypothesis,
                    method_json, json.dumps(result or {}), outcome,
                    failure_mode, lessons, codex_thread_id, tokens_used,
                    wallclock_s, confidence, json.dumps(supersedes or []),
                ),
            )
            conn.execute(
                "INSERT INTO experiences_fts (experience_id, hypothesis, method, lessons, failure_mode)"
                " VALUES (?,?,?,?,?)",
                (exp_id, hypothesis, method_json, lessons, failure_mode or ""),
            )
        return self.get(exp_id)

    def get(self, experience_id: str) -> Optional[Experience]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiences WHERE experience_id = ?", (experience_id,)
            ).fetchone()
        return _row_to_experience(row) if row else None

    def superseded_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT supersedes FROM experiences WHERE supersedes != '[]'"
            ).fetchall()
        out: set[str] = set()
        for r in rows:
            out.update(json.loads(r["supersedes"]))
        return out

    def search(self, query: str, *, k: int = 5) -> list[Experience]:
        """FTS5 keyword search; superseded records are excluded.

        The query is sanitized to bare keywords (FTS5 operators stripped) and
        OR-joined so partial matches still hit.
        """
        terms = [t for t in "".join(
            c if c.isalnum() else " " for c in query
        ).split() if len(t) > 1]
        if not terms:
            return []
        fts_query = " OR ".join(terms[:24])
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT e.* FROM experiences_fts f
                   JOIN experiences e ON e.experience_id = f.experience_id
                   WHERE experiences_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (fts_query, max(k * 4, 20)),
            ).fetchall()
        dead = self.superseded_ids()
        hits = [_row_to_experience(r) for r in rows]
        return [h for h in hits if h.experience_id not in dead][:k]

    def recent(self, *, project_id: Optional[str] = None, k: int = 10) -> list[Experience]:
        q = "SELECT * FROM experiences"
        params: list[Any] = []
        if project_id:
            q += " WHERE project_id = ?"
            params.append(project_id)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(k)
        with self._connect() as conn:
            rows = conn.execute(q, params).fetchall()
        return [_row_to_experience(r) for r in rows]
