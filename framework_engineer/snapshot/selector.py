"""Select representative snapshot cases from raw captures."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import SCHEMA_VERSION, SnapshotCase
from .store import SnapshotStore


class SnapshotSelector:
    def __init__(self, store: SnapshotStore):
        self.store = store

    def select(self, *, max_cases: int | None = None) -> dict[str, Any]:
        raw_cases = self.store.list_raw_cases()
        groups: dict[str, list[SnapshotCase]] = defaultdict(list)
        for case in raw_cases:
            groups[case.hashes["semantic_hash"]].append(case)

        ranked_groups = sorted(groups.values(), key=lambda items: (-len(items), items[0].case_id))
        if max_cases is not None:
            ranked_groups = ranked_groups[:max_cases]

        selected_cases: list[SnapshotCase] = []
        for idx, group in enumerate(ranked_groups, start=1):
            representative = group[0]
            short_key = representative.hashes["case_key"][:8]
            case_id = f"case_{idx:04d}_{short_key}"
            updated = SnapshotCase(
                task_id=representative.task_id,
                case_id=case_id,
                raw_call_ids=[case.case_id for case in group],
                target=representative.target,
                interface=representative.interface,
                files=representative.files,
                mutation=representative.mutation,
                hashes=representative.hashes,
                selection={
                    "call_count": len(group),
                    "priority": "required",
                    "reason": "top_frequency" if idx == 1 else "semantic_group",
                },
                tolerance=representative.tolerance,
                schema_version=representative.schema_version,
            )
            self.store.copy_raw_to_selected(representative.case_id, case_id, updated)
            selected_cases.append(updated)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "selection_policy": "group_by_semantic_hash_keep_first_rank_by_frequency",
            "raw_case_count": len(raw_cases),
            "selected_case_count": len(selected_cases),
            "cases": [case.to_dict() for case in selected_cases],
        }
        self.store.write_manifest(manifest)
        return manifest


def write_shape_list_summary(task_pack: Path, manifest: dict[str, Any]) -> None:
    shape_cases = []
    for case in manifest.get("cases", []):
        tensors = []
        args_tree = case.get("interface", {}).get("args_tree")
        kwargs_tree = case.get("interface", {}).get("kwargs_tree")
        for tree_name, tree in (("args", args_tree), ("kwargs", kwargs_tree)):
            tensors.extend(_collect_tensors(tree, tree_name))
        shape_cases.append(
            {
                "case_id": case["case_id"],
                "priority": case.get("selection", {}).get("priority", "required"),
                "call_count": case.get("selection", {}).get("call_count", 1),
                "semantic_hash": case.get("hashes", {}).get("semantic_hash"),
                "shape_hash": case.get("hashes", {}).get("shape_hash"),
                "snapshot_dir": f"snapshots/selected/{case['case_id']}",
                "tensors": tensors,
            }
        )
    payload = {
        "schema_version": "phase1.shape_summary.v1",
        "source": "snapshots/manifest.json",
        "note": "Selected snapshots are the replay source; this file is only an index/summary.",
        "shape_cases": shape_cases,
    }
    (task_pack / "shape_list.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _collect_tensors(tree: Any, prefix: str) -> list[dict[str, Any]]:
    if not tree:
        return []
    kind = tree.get("kind")
    if kind == "tensor":
        meta = dict(tree["meta"])
        meta["path"] = prefix if not meta.get("path") else meta["path"]
        return [meta]
    if kind in ("tuple", "list"):
        out = []
        for idx, item in enumerate(tree.get("items", [])):
            out.extend(_collect_tensors(item, f"{prefix}.{idx}"))
        return out
    if kind == "dict":
        out = []
        for key, item in tree.get("items", {}).items():
            out.extend(_collect_tensors(item, f"{prefix}.{key}"))
        return out
    return []

