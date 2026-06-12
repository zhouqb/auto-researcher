"""Jobs table (design §9/§10): tracks Codex experiment runs across processes.

The agent process registers a job (with pid/pgid) when it launches Codex; the
UI process reads the table to display live branches and can kill a branch's
process group (design §11.2 kill-branch) without touching siblings.
"""

from __future__ import annotations

import os
import signal
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id      TEXT PRIMARY KEY,        -- '<project>:<run_id>'
  project_id  TEXT NOT NULL,
  branch      TEXT NOT NULL,
  run_id      TEXT NOT NULL,
  status      TEXT NOT NULL,           -- running | completed | failed | timeout | killed
  pid         INTEGER,
  pgid        INTEGER,
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

TERMINAL = {"completed", "failed", "timeout", "killed"}


@dataclass
class Job:
    job_id: str
    project_id: str
    branch: str
    run_id: str
    status: str
    pid: Optional[int]
    pgid: Optional[int]


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        job_id=row["job_id"], project_id=row["project_id"], branch=row["branch"],
        run_id=row["run_id"], status=row["status"], pid=row["pid"], pgid=row["pgid"],
    )


class JobsStore:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # sqlite3's own `with conn` only manages the transaction — it never
        # closes, and connections left to GC leak fds under steady polling.
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def start(self, *, project_id: str, branch: str, run_id: str,
              pid: Optional[int] = None, pgid: Optional[int] = None) -> Job:
        job_id = f"{project_id}:{run_id}"
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs (job_id, project_id, branch, run_id, status, pid, pgid)
                   VALUES (?,?,?,?,'running',?,?)
                   ON CONFLICT(job_id) DO UPDATE SET
                     status='running', pid=excluded.pid, pgid=excluded.pgid,
                     updated_at=CURRENT_TIMESTAMP""",
                (job_id, project_id, branch, run_id, pid, pgid),
            )
        return self.get(job_id)

    def finish(self, job_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
                (status, job_id),
            )

    def get(self, job_id: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def for_project(self, project_id: str) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def delete_project(self, project_id: str) -> int:
        """Drop a project's job rows (kill running ones first via kill())."""
        with self._connect() as conn:
            return conn.execute(
                "DELETE FROM jobs WHERE project_id = ?", (project_id,)
            ).rowcount

    def kill(self, job_id: str) -> bool:
        """SIGTERM the job's process group (kill-branch). Returns True if signaled."""
        job = self.get(job_id)
        if job is None or job.status in TERMINAL:
            return False
        signaled = False
        if job.pgid:
            try:
                os.killpg(job.pgid, signal.SIGTERM)
                signaled = True
            except (ProcessLookupError, PermissionError):
                pass
        elif job.pid:
            try:
                os.kill(job.pid, signal.SIGTERM)
                signaled = True
            except (ProcessLookupError, PermissionError):
                pass
        self.finish(job_id, "killed")
        return signaled
