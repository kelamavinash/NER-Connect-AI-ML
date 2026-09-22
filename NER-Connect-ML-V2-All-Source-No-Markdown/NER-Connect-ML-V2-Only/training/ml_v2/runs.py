"""Immutable experiment run directories and JSON evidence files."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunDirectory:
    path: Path

    @classmethod
    def create(cls, runs_root: Path, run_id: str | None = None) -> "RunDirectory":
        identity = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = runs_root / identity
        path.mkdir(parents=True, exist_ok=False)
        return cls(path)

    def write_json(self, name: str, value: object) -> Path:
        if not name.endswith(".json") or Path(name).name != name:
            raise ValueError("Run evidence must be a top-level JSON file")
        target = self.path / name
        with target.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False, sort_keys=True, default=str)
        return target
