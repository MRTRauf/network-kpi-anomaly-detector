from __future__ import annotations

from datetime import datetime
from pathlib import Path


ART_DIR = Path("artifacts")


def make_run_dir(tag: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = ART_DIR / f"run_{tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _list_runs(tag: str) -> list[Path]:
    if not ART_DIR.exists():
        return []
    return sorted([p for p in ART_DIR.glob(f"run_{tag}_*") if p.is_dir()])


def latest_run_dir(tag: str) -> Path | None:
    runs = _list_runs(tag)
    return runs[-1] if runs else None
