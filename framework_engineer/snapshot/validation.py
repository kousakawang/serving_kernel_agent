"""Validation helpers for snapshot task packs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


REQUIRED_TASK_PACK_FILES = [
    "README.md",
    "task.yaml",
    "shape_list.json",
    "env_manifest.yaml",
    "snapshot_runtime.py",
    "snapshots/manifest.json",
    "original_impl.py",
    "reference_impl.py",
    "candidate_impl.py",
    "correctness_test.py",
    "benchmark.py",
    "scripts/run_correctness.sh",
    "scripts/run_benchmark.sh",
    "scripts/run_ncu.sh",
]


def validate_files(task_pack: Path) -> list[str]:
    errors = []
    for rel in REQUIRED_TASK_PACK_FILES:
        if not (task_pack / rel).exists():
            errors.append(f"missing required file: {rel}")
    manifest = task_pack / "snapshots" / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not data.get("case_groups"):
            errors.append("snapshots/manifest.json has no selected case_groups")
        for group in data.get("case_groups", []):
            group_dir = task_pack / "snapshots" / "selected" / group["group_id"]
            if not (group_dir / "group_meta.json").exists():
                errors.append(f"missing group_meta.json for {group['group_id']}")
            for sample in group.get("samples", []):
                sample_dir = group_dir / "samples" / sample["sample_id"]
                for rel in ("meta.json", "pre_inputs.pt", "post_inputs.pt", "outputs.pt"):
                    if not (sample_dir / rel).exists():
                        errors.append(f"missing snapshot file for {group['group_id']}/{sample['sample_id']}: {rel}")
    return errors


def run_smoke(task_pack: Path, *, correctness: bool, benchmark: bool, timeout: int) -> list[dict[str, Any]]:
    results = []
    commands = []
    if correctness:
        commands.append(["bash", "scripts/run_correctness.sh"])
    if benchmark:
        commands.append(["bash", "scripts/run_benchmark.sh"])
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            cwd=task_pack,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        results.append(
            {
                "command": " ".join(cmd),
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        )
    return results
