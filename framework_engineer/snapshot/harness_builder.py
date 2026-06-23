"""Generate standalone task-pack harness files from selected snapshot groups."""

from __future__ import annotations

import shutil
from pathlib import Path

from .selector import write_shape_list_summary
from .store import SnapshotStore


class SnapshotHarnessBuilder:
    def __init__(self, task_pack: Path | str):
        self.task_pack = Path(task_pack)
        self.store = SnapshotStore(self.task_pack / "snapshots")

    def generate(self, *, candidate_function: str = "candidate") -> None:
        manifest = self.store.read_manifest()
        self._write_runtime()
        self._write_reference_impl(candidate_function)
        self._write_candidate_impl(candidate_function)
        self._write_correctness(candidate_function)
        self._write_benchmark(candidate_function)
        self._write_scripts()
        write_shape_list_summary(self.task_pack, manifest)

    def _write_runtime(self) -> None:
        (self.task_pack / "snapshot_runtime.py").write_text(SNAPSHOT_RUNTIME, encoding="utf-8")

    def _write_reference_impl(self, candidate_function: str) -> None:
        (self.task_pack / "reference_impl.py").write_text(
            REFERENCE_IMPL.replace("__CANDIDATE_FUNCTION__", candidate_function),
            encoding="utf-8",
        )

    def _write_candidate_impl(self, candidate_function: str) -> None:
        (self.task_pack / "candidate_impl.py").write_text(
            CANDIDATE_IMPL.replace("__CANDIDATE_FUNCTION__", candidate_function),
            encoding="utf-8",
        )

    def _write_correctness(self, candidate_function: str) -> None:
        (self.task_pack / "correctness_test.py").write_text(
            CORRECTNESS_TEST.replace("__CANDIDATE_FUNCTION__", candidate_function),
            encoding="utf-8",
        )

    def _write_benchmark(self, candidate_function: str) -> None:
        (self.task_pack / "benchmark.py").write_text(
            BENCHMARK.replace("__CANDIDATE_FUNCTION__", candidate_function),
            encoding="utf-8",
        )

    def _write_scripts(self) -> None:
        scripts = self.task_pack / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        files = {
            "run_correctness.sh": RUN_CORRECTNESS,
            "run_benchmark.sh": RUN_BENCHMARK,
            "run_ncu.sh": RUN_NCU,
        }
        for name, text in files.items():
            path = scripts / name
            path.write_text(text, encoding="utf-8")
            path.chmod(0o755)


def copy_probe_templates(template_dir: Path, task_pack: Path) -> None:
    env_probe = task_pack / "env_probe"
    env_probe.mkdir(parents=True, exist_ok=True)
    for name in ("probe_triton.py", "probe_cutedsl.py", "probe_cuda_extension.py", "probe_ncu.sh"):
        src = template_dir / name
        if src.exists():
            dst = env_probe / name
            shutil.copy2(src, dst)
            if dst.suffix == ".sh":
                dst.chmod(0o755)


SNAPSHOT_RUNTIME = r'''"""Standalone snapshot replay runtime copied into task packs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def _torch():
    try:
        import torch
    except Exception:
        return None
    return torch


CURRENT_SAMPLE = None


def load_manifest(root: Path = Path("snapshots")) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text())


def list_groups(root: Path = Path("snapshots"), priority: str | None = "required") -> list[dict[str, Any]]:
    manifest = load_manifest(root)
    groups = manifest.get("case_groups", [])
    if priority is None:
        return groups
    return [group for group in groups if group.get("selection", {}).get("priority") == priority]


def list_samples(
    root: Path = Path("snapshots"),
    *,
    group_id: str | None = None,
    sample_id: str | None = None,
    priority: str | None = "required",
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for group in list_groups(root, priority=priority):
        if group_id is not None and group["group_id"] != group_id:
            continue
        for sample in group.get("samples", []):
            if sample_id is not None and sample["sample_id"] != sample_id:
                continue
            out.append((group, sample))
    return out


def load_sample(group_id: str, sample_id: str, root: Path = Path("snapshots"), device: str = "cuda") -> dict[str, Any]:
    group_dir = root / "selected" / group_id
    sample_dir = group_dir / "samples" / sample_id
    group_meta = json.loads((group_dir / "group_meta.json").read_text())
    sample_meta = json.loads((sample_dir / "meta.json").read_text())
    pre_inputs = _load_payload(sample_dir / sample_meta["files"]["pre_inputs"])
    post_inputs = _load_payload(sample_dir / sample_meta["files"]["post_inputs"])
    outputs = _load_payload(sample_dir / sample_meta["files"]["outputs"])
    return {
        "group": group_meta,
        "sample_meta": sample_meta,
        "pre_inputs": tree_to_device(pre_inputs, device),
        "post_inputs": tree_to_device(post_inputs, device),
        "outputs": tree_to_device(outputs, device),
    }


def _load_payload(path: Path) -> Any:
    torch = _torch()
    if torch is not None:
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            pass
    with path.open("rb") as f:
        return pickle.load(f)


def set_current_sample(sample: dict[str, Any]) -> None:
    global CURRENT_SAMPLE
    CURRENT_SAMPLE = sample


def get_current_sample() -> dict[str, Any]:
    if CURRENT_SAMPLE is None:
        raise RuntimeError("No current snapshot sample is set.")
    return CURRENT_SAMPLE


def tree_clone(value: Any) -> Any:
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.clone()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return tuple(tree_clone(v) for v in value)
    if isinstance(value, list):
        return [tree_clone(v) for v in value]
    if isinstance(value, dict):
        return {k: tree_clone(v) for k, v in value.items()}
    raise TypeError(f"Unsupported snapshot value for clone: {type(value)!r}")


def tree_to_device(value: Any, device: str) -> Any:
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.to(device)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return tuple(tree_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [tree_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: tree_to_device(v, device) for k, v in value.items()}
    raise TypeError(f"Unsupported snapshot value for device transfer: {type(value)!r}")


def get_path(tree: Any, path: str) -> Any:
    cur = tree
    for part in [p for p in path.split(".") if p]:
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot descend into {type(cur)!r} at {part!r}")
    return cur


def set_path(tree: Any, path: str, value: Any) -> None:
    parts = [p for p in path.split(".") if p]
    cur = tree
    for part in parts[:-1]:
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot descend into {type(cur)!r} at {part!r}")
    last = parts[-1]
    torch = _torch()
    if isinstance(cur, dict):
        old = cur.get(last)
        if torch is not None and isinstance(old, torch.Tensor) and isinstance(value, torch.Tensor):
            old.copy_(value)
        else:
            cur[last] = value
    elif isinstance(cur, list):
        index = int(last)
        old = cur[index]
        if torch is not None and isinstance(old, torch.Tensor) and isinstance(value, torch.Tensor):
            old.copy_(value)
        else:
            cur[index] = value
    elif isinstance(cur, tuple):
        index = int(last)
        old = cur[index]
        if torch is not None and isinstance(old, torch.Tensor) and isinstance(value, torch.Tensor):
            old.copy_(value)
        else:
            raise TypeError(f"Cannot assign non-tensor value into tuple path {path!r}")
    else:
        raise KeyError(f"Cannot set path {path!r}")


def apply_snapshot_mutations(call_tree: dict[str, Any], sample: dict[str, Any]) -> None:
    sample_meta = sample["sample_meta"]
    post_inputs = sample["post_inputs"]
    for path in sample_meta.get("mutation", {}).get("mutable_arg_paths", []):
        set_path(call_tree, path, tree_clone(get_path(post_inputs, path)))


def assert_tree_close(actual: Any, expected: Any, *, atol: float, rtol: float, path: str = "") -> None:
    torch = _torch()
    if torch is not None and isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{path}: actual is not a tensor")
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    if expected is None or isinstance(expected, (str, int, float, bool)):
        if actual != expected:
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(actual) != len(expected):
            raise AssertionError(f"{path}: tuple mismatch")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_tree_close(a, e, atol=atol, rtol=rtol, path=f"{path}.{i}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list mismatch")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_tree_close(a, e, atol=atol, rtol=rtol, path=f"{path}.{i}")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{path}: dict key mismatch")
        for key in expected:
            assert_tree_close(actual[key], expected[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
        return
    raise TypeError(f"Unsupported snapshot value for comparison: {type(expected)!r}")
'''


REFERENCE_IMPL = '''"""Snapshot-golden reference implementation generated by Framework Engineer."""

from __future__ import annotations

import snapshot_runtime


def reference(*args, **kwargs):
    sample = snapshot_runtime.get_current_sample()
    call_tree = {"args": list(args), "kwargs": kwargs}
    snapshot_runtime.apply_snapshot_mutations(call_tree, sample)
    return snapshot_runtime.tree_clone(sample["outputs"])


if "__CANDIDATE_FUNCTION__" != "reference":
    globals()["reference___CANDIDATE_FUNCTION__"] = reference
'''


CANDIDATE_IMPL = '''"""Candidate implementation generated for Kernel Engineer.

The initial candidate delegates to snapshot-golden reference so the task pack
starts with passing correctness. Kernel Engineer should replace candidate().
"""

from __future__ import annotations

import reference_impl


def candidate(*args, **kwargs):
    return reference_impl.reference(*args, **kwargs)


if "__CANDIDATE_FUNCTION__" != "candidate":
    globals()["__CANDIDATE_FUNCTION__"] = candidate
'''


CORRECTNESS_TEST = '''"""Snapshot replay correctness harness."""

from __future__ import annotations

import argparse
import json

import candidate_impl
import reference_impl
import snapshot_runtime


CANDIDATE_FUNCTION = "__CANDIDATE_FUNCTION__"


def _call(fn, call_tree):
    return fn(*call_tree.get("args", ()), **call_tree.get("kwargs", {}))


def run_sample(group_meta, sample_meta, *, device: str, mode: str) -> dict:
    sample = snapshot_runtime.load_sample(group_meta["group_id"], sample_meta["sample_id"], device=device)
    snapshot_runtime.set_current_sample(sample)
    tol = sample["sample_meta"].get("tolerance", {})
    atol = float(tol.get("atol", 2e-2))
    rtol = float(tol.get("rtol", 2e-2))

    ref_tree = snapshot_runtime.tree_clone(sample["pre_inputs"])
    cand_tree = snapshot_runtime.tree_clone(sample["pre_inputs"])

    if mode == "reference-replay":
        expected = _call(reference_impl.reference, ref_tree)
        expected_mut_tree = ref_tree
    else:
        expected = snapshot_runtime.tree_clone(sample["outputs"])
        expected_mut_tree = sample["post_inputs"]

    candidate = getattr(candidate_impl, CANDIDATE_FUNCTION)
    actual = _call(candidate, cand_tree)
    snapshot_runtime.assert_tree_close(actual, expected, atol=atol, rtol=rtol)

    for path in sample["sample_meta"].get("mutation", {}).get("mutable_arg_paths", []):
        snapshot_runtime.assert_tree_close(
            snapshot_runtime.get_path(cand_tree, path),
            snapshot_runtime.get_path(expected_mut_tree, path),
            atol=atol,
            rtol=rtol,
            path=path,
        )

    return {
        "group_id": group_meta["group_id"],
        "sample_id": sample_meta["sample_id"],
        "status": "PASS",
        "mode": mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["reference-replay", "snapshot-golden"], default="snapshot-golden")
    parser.add_argument("--all-priorities", action="store_true")
    args = parser.parse_args()

    priority = None if args.all_priorities else "required"
    selected = snapshot_runtime.list_samples(group_id=args.group_id, sample_id=args.sample_id, priority=priority)
    for group, sample in selected:
        print(json.dumps(run_sample(group, sample, device=args.device, mode=args.mode), sort_keys=True))


if __name__ == "__main__":
    main()
'''


BENCHMARK = '''"""Snapshot replay benchmark harness."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict

import candidate_impl
import reference_impl
import snapshot_runtime


CANDIDATE_FUNCTION = "__CANDIDATE_FUNCTION__"


def _torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def sync() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _call(fn, call_tree):
    return fn(*call_tree.get("args", ()), **call_tree.get("kwargs", {}))


def elapsed_us(fn, make_inputs, *, warmup: int, repeat: int, use_cuda_events: bool) -> dict:
    torch = _torch()
    for _ in range(warmup):
        _call(fn, make_inputs())
    sync()

    values = []
    for _ in range(repeat):
        inputs = make_inputs()
        sync()
        if use_cuda_events and torch is not None and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _call(fn, inputs)
            end.record()
            sync()
            values.append(float(start.elapsed_time(end) * 1000.0))
        else:
            start_t = time.perf_counter()
            _call(fn, inputs)
            sync()
            values.append((time.perf_counter() - start_t) * 1_000_000.0)
    return {
        "median_us": statistics.median(values),
        "mean_us": statistics.mean(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def benchmark_sample(group_meta, sample_meta, *, device: str, target: str, warmup: int, repeat: int) -> dict:
    sample = snapshot_runtime.load_sample(group_meta["group_id"], sample_meta["sample_id"], device=device)
    snapshot_runtime.set_current_sample(sample)

    def make_inputs():
        return snapshot_runtime.tree_clone(sample["pre_inputs"])

    result = {
        "record_type": "sample",
        "group_id": group_meta["group_id"],
        "sample_id": sample_meta["sample_id"],
        "warmup": warmup,
        "repeat": repeat,
    }
    use_events = device.startswith("cuda")
    candidate = getattr(candidate_impl, CANDIDATE_FUNCTION)
    if target in ("reference", "both"):
        result["reference"] = elapsed_us(reference_impl.reference, make_inputs, warmup=warmup, repeat=repeat, use_cuda_events=use_events)
    if target in ("candidate", "both"):
        result["candidate"] = elapsed_us(candidate, make_inputs, warmup=warmup, repeat=repeat, use_cuda_events=use_events)
    if "reference" in result and "candidate" in result and result["candidate"]["median_us"] > 0:
        result["speedup_median"] = result["reference"]["median_us"] / result["candidate"]["median_us"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", choices=["reference", "candidate", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--all-priorities", action="store_true")
    args = parser.parse_args()

    priority = None if args.all_priorities else "required"
    selected = snapshot_runtime.list_samples(group_id=args.group_id, sample_id=args.sample_id, priority=priority)
    by_group = defaultdict(list)
    for group, sample in selected:
        result = benchmark_sample(group, sample, device=args.device, target=args.target, warmup=args.warmup, repeat=args.repeat)
        by_group[group["group_id"]].append(result)
        print(json.dumps(result, sort_keys=True))

    for group_id, rows in sorted(by_group.items()):
        summary = {"record_type": "group_summary", "group_id": group_id, "sample_count": len(rows)}
        for target_name in ("reference", "candidate"):
            medians = [row[target_name]["median_us"] for row in rows if target_name in row]
            if medians:
                summary[target_name] = {
                    "median_of_sample_medians_us": statistics.median(medians),
                    "mean_of_sample_medians_us": statistics.mean(medians),
                    "min_sample_median_us": min(medians),
                    "max_sample_median_us": max(medians),
                }
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
'''


RUN_CORRECTNESS = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

"${PYTHON:-python3}" correctness_test.py --device "${DEVICE:-cuda}" --mode "${CORRECTNESS_MODE:-snapshot-golden}"
'''


RUN_BENCHMARK = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

"${PYTHON:-python3}" benchmark.py --device "${DEVICE:-cuda}" --target "${TARGET:-both}" --warmup "${WARMUP:-20}" --repeat "${REPEAT:-100}"
'''


RUN_NCU = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GROUP_ID="${1:-}"
SAMPLE_ID="${2:-}"
if [ -z "$GROUP_ID" ]; then
  echo "usage: bash scripts/run_ncu.sh <group_id> [sample_id]" >&2
  exit 2
fi

args=(benchmark.py --group-id "$GROUP_ID" --device "${DEVICE:-cuda}" --target "${TARGET:-candidate}" --warmup "${WARMUP:-5}" --repeat "${REPEAT:-20}")
if [ -n "$SAMPLE_ID" ]; then
  args+=(--sample-id "$SAMPLE_ID")
fi

ncu --set full --target-processes all "${PYTHON:-python3}" "${args[@]}"
'''
