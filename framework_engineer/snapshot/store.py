"""Filesystem storage for Phase 1 snapshots."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .models import SnapshotCase


class SnapshotStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.selected_dir = self.root / "selected"
        self.manifest_path = self.root / "manifest.json"

    def ensure(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.selected_dir.mkdir(parents=True, exist_ok=True)

    def raw_case_dir(self, call_id: str) -> Path:
        return self.raw_dir / call_id

    def selected_case_dir(self, case_id: str) -> Path:
        return self.selected_dir / case_id

    def write_case_meta(self, case_dir: Path, case: SnapshotCase) -> None:
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "meta.json").write_text(json.dumps(case.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def read_case_meta(self, case_dir: Path) -> SnapshotCase:
        data = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
        return SnapshotCase.from_dict(data)

    def list_raw_cases(self) -> list[SnapshotCase]:
        return self._list_cases(self.raw_dir)

    def list_selected_cases(self) -> list[SnapshotCase]:
        return self._list_cases(self.selected_dir)

    def _list_cases(self, root: Path) -> list[SnapshotCase]:
        if not root.exists():
            return []
        cases = []
        for meta in sorted(root.glob("*/meta.json")):
            cases.append(SnapshotCase.from_dict(json.loads(meta.read_text(encoding="utf-8"))))
        return cases

    def copy_raw_to_selected(self, raw_call_id: str, selected_case_id: str, updated_case: SnapshotCase) -> Path:
        src = self.raw_case_dir(raw_call_id)
        dst = self.selected_case_dir(selected_case_id)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        self.write_case_meta(dst, updated_case)
        return dst

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

