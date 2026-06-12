"""SQLite artifact catalog (design §7.2).

Every artifact version is one row in ``artifacts``; lineage edges live in
``artifact_lineage``; ``artifact_fts`` powers keyword search. The catalog
shares the SQLite file with ADK sessions but owns its own tables.

Connections are short-lived (open per operation, WAL mode) — catalog traffic
is low-volume metadata; bulk data goes to files on disk, never the DB.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL,
  kind          TEXT NOT NULL,
  path          TEXT NOT NULL,
  version       INTEGER NOT NULL,
  content_hash  TEXT,
  iteration     INTEGER,
  branch        TEXT,
  run_id        TEXT,
  title         TEXT,
  summary       TEXT,
  meta          TEXT,
  created_by    TEXT,
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_project_path_version
  ON artifacts (project_id, path, version);

CREATE TABLE IF NOT EXISTS artifact_lineage (
  child_id  TEXT NOT NULL,
  parent_id TEXT NOT NULL,
  relation  TEXT NOT NULL,
  PRIMARY KEY (child_id, parent_id, relation)
);

CREATE VIRTUAL TABLE IF NOT EXISTS artifact_fts
  USING fts5(artifact_id UNINDEXED, title, summary, body);
"""

# Path-prefix → kind fallback when the producer does not declare one.
_KIND_BY_PREFIX = {
    "brief": "brief",
    "design": "design",
    "plan": "plan",
    "decisions": "decision",
    "budget": "budget",
    "checkpoints": "checkpoint",
    "lit": "lit_notes",
    "reports": "report",
}

_FTS_BODY_CAP = 50_000


def guess_kind(path: str) -> str:
    prefix = path.split("/", 1)[0]
    return _KIND_BY_PREFIX.get(prefix, "other")


@dataclass
class ArtifactRecord:
    id: str
    project_id: str
    kind: str
    path: str
    version: int
    content_hash: Optional[str] = None
    iteration: Optional[int] = None
    branch: Optional[str] = None
    run_id: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)
    created_by: Optional[str] = None
    created_at: Optional[str] = None


def _row_to_record(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        id=row["id"],
        project_id=row["project_id"],
        kind=row["kind"],
        path=row["path"],
        version=row["version"],
        content_hash=row["content_hash"],
        iteration=row["iteration"],
        branch=row["branch"],
        run_id=row["run_id"],
        title=row["title"],
        summary=row["summary"],
        meta=json.loads(row["meta"]) if row["meta"] else {},
        created_by=row["created_by"],
        created_at=row["created_at"],
    )


class ArtifactCatalog:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def register(
        self,
        *,
        project_id: str,
        path: str,
        version: int,
        kind: Optional[str] = None,
        content_hash: Optional[str] = None,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        body_text: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
        iteration: Optional[int] = None,
        branch: Optional[str] = None,
        run_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> ArtifactRecord:
        """Insert one artifact version; links `supersedes` to the prior version."""
        art_id = "art_" + uuid.uuid4().hex[:12]
        kind = kind or guess_kind(path)
        prior = self.get(project_id=project_id, path=path)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO artifacts
                   (id, project_id, kind, path, version, content_hash, iteration,
                    branch, run_id, title, summary, meta, created_by)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    art_id, project_id, kind, path, version, content_hash,
                    iteration, branch, run_id, title, summary,
                    json.dumps(meta or {}), created_by,
                ),
            )
            conn.execute(
                "INSERT INTO artifact_fts (artifact_id, title, summary, body) VALUES (?,?,?,?)",
                (art_id, title or "", summary or "", (body_text or "")[:_FTS_BODY_CAP]),
            )
            if prior is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO artifact_lineage (child_id, parent_id, relation) VALUES (?,?,?)",
                    (art_id, prior.id, "supersedes"),
                )
        return self.get_by_id(art_id)  # round-trip for created_at

    def add_lineage(self, child_id: str, parent_id: str, relation: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO artifact_lineage (child_id, parent_id, relation) VALUES (?,?,?)",
                (child_id, parent_id, relation),
            )

    def get_by_id(self, art_id: str) -> Optional[ArtifactRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (art_id,)).fetchone()
        return _row_to_record(row) if row else None

    def get(
        self, *, project_id: str, path: str, version: Optional[int] = None
    ) -> Optional[ArtifactRecord]:
        """Latest version of an artifact path, or a specific version."""
        query = "SELECT * FROM artifacts WHERE project_id = ? AND path = ?"
        params: list[Any] = [project_id, path]
        if version is not None:
            query += " AND version = ?"
            params.append(version)
        query += " ORDER BY version DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return _row_to_record(row) if row else None

    def versions(self, *, project_id: str, path: str) -> list[ArtifactRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE project_id = ? AND path = ? ORDER BY version",
                (project_id, path),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def list_paths(self, *, project_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT path FROM artifacts WHERE project_id = ? ORDER BY path",
                (project_id,),
            ).fetchall()
        return [r["path"] for r in rows]

    def list_latest(self, *, project_id: str) -> list[ArtifactRecord]:
        """Latest version of every artifact path in a project."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT a.* FROM artifacts a
                   JOIN (SELECT path, MAX(version) AS v FROM artifacts
                         WHERE project_id = ? GROUP BY path) m
                     ON a.path = m.path AND a.version = m.v
                   WHERE a.project_id = ?
                   ORDER BY a.path""",
                (project_id, project_id),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def delete_path(self, *, project_id: str, path: str) -> None:
        with self._connect() as conn:
            ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM artifacts WHERE project_id = ? AND path = ?",
                    (project_id, path),
                ).fetchall()
            ]
            conn.execute(
                "DELETE FROM artifacts WHERE project_id = ? AND path = ?",
                (project_id, path),
            )
            for art_id in ids:
                conn.execute("DELETE FROM artifact_fts WHERE artifact_id = ?", (art_id,))
                conn.execute(
                    "DELETE FROM artifact_lineage WHERE child_id = ? OR parent_id = ?",
                    (art_id, art_id),
                )

    def search(self, query: str, *, project_id: Optional[str] = None, limit: int = 20) -> list[ArtifactRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT a.* FROM artifact_fts f
                   JOIN artifacts a ON a.id = f.artifact_id
                   WHERE artifact_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
        records = [_row_to_record(r) for r in rows]
        if project_id is not None:
            records = [r for r in records if r.project_id == project_id]
        return records

    def lineage(self, art_id: str) -> list[tuple[str, str, str]]:
        """All (child, parent, relation) edges touching an artifact."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT child_id, parent_id, relation FROM artifact_lineage WHERE child_id = ? OR parent_id = ?",
                (art_id, art_id),
            ).fetchall()
        return [(r["child_id"], r["parent_id"], r["relation"]) for r in rows]
