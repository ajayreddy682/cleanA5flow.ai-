"""Local cleaning-history storage.

Only job metadata is stored: filename, counts, operations, and status. Uploaded
file contents, cell values, and generated files are never written to history.
The JSON file lives in ``.cleanflow/history.json`` (git-ignored) and is capped
at 50 entries.

For a public multi-user deployment, this single-process file must be replaced
with per-user storage (for example SQLite keyed by an authenticated user ID).
See README.md -> Deployment caveats before exposing the app beyond localhost.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("cleanflow.history")
_LOCK = threading.Lock()
MAX_JOBS = 50
MAX_ERROR_LENGTH = 300


def history_path() -> Path:
    override = os.environ.get("CLEANFLOW_HISTORY_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1] / ".cleanflow" / "history.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_jobs() -> list[dict[str, Any]]:
    """Load valid metadata records, newest first. Corrupt files are ignored."""

    path = history_path()
    if not path.exists():
        return []
    try:
        with _LOCK:
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
            return []
        jobs = [job for job in payload["jobs"] if isinstance(job, dict) and job.get("id")]
        jobs.sort(key=lambda job: str(job.get("timestamp", "")), reverse=True)
        return jobs[:MAX_JOBS]
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("CleanFlow history could not be read; starting with an empty list.")
        return []


def save_jobs(jobs: list[dict[str, Any]]) -> None:
    """Atomically persist the newest ``MAX_JOBS`` records."""

    path = history_path()
    _ensure_parent(path)
    trimmed = jobs[:MAX_JOBS]
    payload = {
        "version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jobs": trimmed,
    }
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with _LOCK:
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            os.replace(temp_path, path)
        except OSError:
            LOGGER.exception("Could not save CleanFlow history")
            try:
                temp_path.unlink()
            except OSError:
                pass


def append_job(job: dict[str, Any]) -> dict[str, Any]:
    record = dict(job)
    record.setdefault("id", uuid.uuid4().hex)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    record["filename"] = str(record.get("filename", "upload"))[:120]
    record["status"] = record.get("status", "completed")
    record["error"] = str(record.get("error", ""))[:MAX_ERROR_LENGTH]
    record["operations"] = [str(item)[:80] for item in record.get("operations", [])][:20]
    record["tables"] = [str(item)[:80] for item in record.get("tables", [])][:20]
    record["outputs"] = [str(item)[:80] for item in record.get("outputs", [])][:12]
    jobs = load_jobs()
    jobs = [existing for existing in jobs if existing.get("id") != record["id"]]
    jobs.insert(0, record)
    save_jobs(jobs)
    return record


def create_job(
    *,
    filename: str,
    file_type: str,
    size_bytes: int,
    results: dict[str, Any] | None = None,
    status: str = "completed",
    error: str = "",
    duration_ms: int = 0,
) -> dict[str, Any]:
    """Build a metadata-only history record."""

    results = results or {}
    rows_before = sum(result.summary.rows_before for result in results.values())
    rows_after = sum(result.summary.rows_after for result in results.values())
    duplicates = sum(result.summary.duplicates_removed for result in results.values())
    missing_before = sum(result.summary.missing_cells_before for result in results.values())
    missing_after = sum(result.summary.missing_cells_after for result in results.values())
    operations: list[str] = []
    for result in results.values():
        for entry in result.audit:
            if entry.action not in operations:
                operations.append(entry.action)
    job = {
        "filename": filename,
        "file_type": file_type,
        "size_bytes": int(size_bytes),
        "tables": list(results.keys()),
        "table_count": len(results),
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "duplicates_removed": int(duplicates),
        "missing_before": int(missing_before),
        "missing_after": int(missing_after),
        "operation_count": sum(len(result.audit) for result in results.values()),
        "operations": operations,
        "status": status,
        "error": error,
        "duration_ms": int(duration_ms),
        "outputs": (
            [
                "Cleaned Excel",
                "Cleaned CSV",
                "Data quality report",
                "Activity log",
                "Power BI package (available)",
            ]
            if results
            else []
        ),
    }
    return append_job(job)


def clear_jobs() -> None:
    save_jobs([])


def history_stats(jobs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    jobs = jobs if jobs is not None else load_jobs()
    completed = [job for job in jobs if job.get("status") == "completed"]
    total_rows = sum(int(job.get("rows_after") or 0) for job in completed)
    total_duplicates = sum(int(job.get("duplicates_removed") or 0) for job in completed)
    last_run = jobs[0].get("timestamp") if jobs else None
    return {
        "total_jobs": len(jobs),
        "completed_jobs": len(completed),
        "failed_jobs": len(jobs) - len(completed),
        "total_rows_cleaned": total_rows,
        "duplicates_removed": total_duplicates,
        "last_run": last_run,
        "files": len({job.get("filename") for job in jobs}),
    }


def history_frame(jobs: list[dict[str, Any]] | None = None) -> Any:
    import pandas as pd

    jobs = jobs if jobs is not None else load_jobs()
    rows = []
    for job in jobs:
        rows.append(
            {
                "When (UTC)": job.get("timestamp", ""),
                "File": job.get("filename", ""),
                "Type": str(job.get("file_type", "")).upper(),
                "Tables": job.get("table_count", 0),
                "Rows in": job.get("rows_before", 0),
                "Rows out": job.get("rows_after", 0),
                "Duplicates removed": job.get("duplicates_removed", 0),
                "Operations": job.get("operation_count", 0),
                "Status": job.get("status", "").title(),
                "Duration": f"{int(job.get('duration_ms') or 0) / 1000:.1f}s",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "When (UTC)", "File", "Type", "Tables", "Rows in", "Rows out",
                "Duplicates removed", "Operations", "Status", "Duration",
            ]
        )
    return pd.DataFrame(rows)


def export_history_csv() -> bytes:
    import pandas as pd

    jobs = load_jobs()
    frame = pd.DataFrame(
        [
            {
                "timestamp_utc": job.get("timestamp"),
                "filename": job.get("filename"),
                "file_type": job.get("file_type"),
                "size_bytes": job.get("size_bytes"),
                "tables": "; ".join(job.get("tables", [])),
                "rows_before": job.get("rows_before"),
                "rows_after": job.get("rows_after"),
                "duplicates_removed": job.get("duplicates_removed"),
                "operations": "; ".join(job.get("operations", [])),
                "status": job.get("status"),
                "duration_ms": job.get("duration_ms"),
                "error": job.get("error"),
            }
            for job in jobs
        ]
    )
    csv_text = frame.to_csv(index=False)
    return ("\ufeff" + csv_text).encode("utf-8")
