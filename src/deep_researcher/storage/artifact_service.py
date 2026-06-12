"""LocalArtifactService: filesystem + SQLite-catalog artifact backend (design §7).

Layout under ``data_root``::

    projects/<project_id>/<filename>                  # latest version, human-browsable
    projects/<project_id>/.versions/<filename>/<N>    # every version, raw bytes

``filename`` may contain slashes ("plan/plan.md"), giving the semantic tree
from design §7.2. Phase 0 maps project_id == ADK session_id; files saved with
the ADK ``user:`` namespace live under ``users/<user_id>/`` instead.

Versions are 0-based, matching ADK convention. Each save also registers a
catalog row; ``custom_metadata`` keys (kind, title, summary, iteration,
branch, run_id, created_by) flow into catalog columns so agents can register
artifacts through the plain ``tool_context.save_artifact`` API.
"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Union

from google.adk.artifacts import BaseArtifactService
from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.genai import types
from typing_extensions import override

from .catalog import ArtifactCatalog

_CATALOG_KEYS = ("kind", "title", "summary", "iteration", "branch", "run_id", "created_by")

_TEXT_MIMES = {"application/json", "application/x-ndjson"}


def _is_text_mime(mime: str) -> bool:
    return mime.startswith("text/") or mime in _TEXT_MIMES


def _guess_mime(filename: str) -> str:
    if filename.endswith(".md"):
        return "text/markdown"
    return mimetypes.guess_type(filename)[0] or "text/plain"


def _safe_relpath(filename: str) -> PurePosixPath:
    rel = PurePosixPath(filename)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe artifact filename: {filename!r}")
    return rel


class LocalArtifactService(BaseArtifactService):
    def __init__(self, root: Path, catalog: ArtifactCatalog):
        self._root = root
        self._catalog = catalog

    # -- scoping ---------------------------------------------------------

    @staticmethod
    def _has_user_namespace(filename: str) -> bool:
        return filename.startswith("user:")

    def _scope(
        self, user_id: str, session_id: Optional[str], filename: str
    ) -> tuple[str, Path, str]:
        """Returns (project_id, base_dir, relative filename)."""
        if self._has_user_namespace(filename):
            rel = filename[len("user:"):]
            return f"user:{user_id}", self._root / "users" / user_id, rel
        if not session_id:
            raise ValueError("session_id is required for session-scoped artifacts")
        return session_id, self._root / "projects" / session_id, filename

    def _version_file(self, base: Path, rel: str, version: int) -> Path:
        return base / ".versions" / rel / str(version)

    # -- BaseArtifactService ----------------------------------------------

    @override
    async def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        artifact: Union[types.Part, dict[str, Any]],
        session_id: Optional[str] = None,
        custom_metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        if isinstance(artifact, dict):
            artifact = types.Part.model_validate(artifact)
        project_id, base, rel = self._scope(user_id, session_id, filename)
        _safe_relpath(rel)

        if artifact.text is not None:
            data = artifact.text.encode("utf-8")
            mime = _guess_mime(rel)
        elif artifact.inline_data is not None and artifact.inline_data.data is not None:
            data = artifact.inline_data.data
            mime = artifact.inline_data.mime_type or _guess_mime(rel)
        else:
            raise ValueError("Only text and inline-bytes artifacts are supported")

        version = len(self._catalog.versions(project_id=project_id, path=filename))

        archive = self._version_file(base, rel, version)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(data)
        latest = base / rel
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_bytes(data)

        meta = dict(custom_metadata or {})
        catalog_fields = {k: meta.pop(k, None) for k in _CATALOG_KEYS}
        meta["mime_type"] = mime
        body_text = data.decode("utf-8", errors="ignore") if _is_text_mime(mime) else None
        self._catalog.register(
            project_id=project_id,
            path=filename,
            version=version,
            kind=catalog_fields["kind"],
            content_hash="sha256:" + hashlib.sha256(data).hexdigest(),
            title=catalog_fields["title"],
            summary=catalog_fields["summary"],
            body_text=body_text,
            meta=meta,
            iteration=catalog_fields["iteration"],
            branch=catalog_fields["branch"],
            run_id=catalog_fields["run_id"],
            created_by=catalog_fields["created_by"],
        )
        return version

    @override
    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Optional[types.Part]:
        project_id, base, rel = self._scope(user_id, session_id, filename)
        record = self._catalog.get(project_id=project_id, path=filename, version=version)
        if record is None:
            return None
        path = self._version_file(base, rel, record.version)
        if not path.exists():
            return None
        data = path.read_bytes()
        mime = record.meta.get("mime_type", "text/plain")
        if _is_text_mime(mime):
            return types.Part(text=data.decode("utf-8", errors="replace"))
        return types.Part.from_bytes(data=data, mime_type=mime)

    @override
    async def list_artifact_keys(
        self, *, app_name: str, user_id: str, session_id: Optional[str] = None
    ) -> list[str]:
        keys: list[str] = []
        if session_id:
            keys.extend(self._catalog.list_paths(project_id=session_id))
        keys.extend(
            f"user:{p}" if not p.startswith("user:") else p
            for p in self._catalog.list_paths(project_id=f"user:{user_id}")
        )
        return sorted(set(keys))

    @override
    async def list_versions(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: Optional[str] = None,
    ) -> list[int]:
        project_id, _, _ = self._scope(user_id, session_id, filename)
        return [r.version for r in self._catalog.versions(project_id=project_id, path=filename)]

    @override
    async def list_artifact_versions(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: Optional[str] = None,
    ) -> list[ArtifactVersion]:
        project_id, base, rel = self._scope(user_id, session_id, filename)
        return [
            self._to_artifact_version(base, rel, record)
            for record in self._catalog.versions(project_id=project_id, path=filename)
        ]

    @override
    async def get_artifact_version(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Optional[ArtifactVersion]:
        project_id, base, rel = self._scope(user_id, session_id, filename)
        record = self._catalog.get(project_id=project_id, path=filename, version=version)
        if record is None:
            return None
        return self._to_artifact_version(base, rel, record)

    @override
    async def delete_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: Optional[str] = None,
    ) -> None:
        import shutil

        project_id, base, rel = self._scope(user_id, session_id, filename)
        self._catalog.delete_path(project_id=project_id, path=filename)
        (base / rel).unlink(missing_ok=True)
        shutil.rmtree(base / ".versions" / rel, ignore_errors=True)

    # -- helpers -----------------------------------------------------------

    def _to_artifact_version(self, base: Path, rel: str, record) -> ArtifactVersion:
        custom = {
            k: getattr(record, k)
            for k in ("kind", "title", "summary", "iteration", "branch", "run_id", "created_by")
            if getattr(record, k) is not None
        }
        custom.update({k: v for k, v in record.meta.items() if k != "mime_type"})
        custom["artifact_id"] = record.id
        return ArtifactVersion(
            version=record.version,
            canonical_uri=self._version_file(base, rel, record.version).as_uri(),
            custom_metadata=custom,
            mime_type=record.meta.get("mime_type"),
        )
