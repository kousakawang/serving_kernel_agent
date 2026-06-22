"""Generate standalone task-pack harness files from selected snapshots."""

from __future__ import annotations

import shutil
from pathlib import Path

from .selector import write_shape_list_summary
from .store import SnapshotStore


class SnapshotHarnessBuilder:
    def __init__(self, task_pack: Path | str):
        self.task_pack = Path(task_pack)
        self.store = SnapshotStore(self.task_pack / "snapshots")

    def generate(self, *, candidate_function: str = "candidate_extend") -> None:
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
        suffix = _suffix(candidate_function)
        (self.task_pack / "reference_impl.py").write_text(REFERENCE_IMPL.format(suffix=suffix), encoding="utf-8")

    def _write_candidate_impl(self, candidate_function: str) -> None:
        suffix = _suffix(candidate_function)
        (self.task_pack / "candidate_impl.py").write_text(CANDIDATE_IMPL.format(suffix=suffix), encoding="utf-8")

    def _write_correctness(self, candidate_function: str) -> None:
        suffix = _suffix(candidate_function)
        (self.task_pack / "correctness_test.py").write_text(
            CORRECTNESS_TEST.format(candidate_function=candidate_function, suffix=suffix),
            encoding="utf-8",
        )

    def _write_benchmark(self, candidate_function: str) -> None:
        suffix = _suffix(candidate_function)
        (self.task_pack / "benchmark.py").write_text(
            BENCHMARK.format(candidate_function=candidate_function, suffix=suffix),
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


def _suffix(candidate_function: str) -> str:
    if candidate_function.startswith("candidate_"):
        return candidate_function.removeprefix("candidate_")
    return candidate_function


SNAPSHOT_RUNTIME = r'''"""Standalone snapshot replay runtime copied into task packs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

CURRENT_CASE = None


def load_manifest(root: Path = Path("snapshots")) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text())


def list_cases(root: Path = Path("snapshots"), priority: str | None = "required") -> list[dict[str, Any]]:
    manifest = load_manifest(root)
    cases = manifest.get("cases", [])
    if priority is None:
        return cases
    return [case for case in cases if case.get("selection", {}).get("priority") == priority]


def load_case(case_id: str, root: Path = Path("snapshots"), device: str = "cuda") -> dict[str, Any]:
    case_dir = root / "selected" / case_id
    meta = json.loads((case_dir / "meta.json").read_text())
    pre_inputs = torch.load(case_dir / meta["files"]["pre_inputs"], map_location="cpu")
    post_inputs = torch.load(case_dir / meta["files"]["post_inputs"], map_location="cpu")
    outputs = torch.load(case_dir / meta["files"]["outputs"], map_location="cpu")
    return {
        "meta": meta,
        "pre_inputs": tree_to_device(pre_inputs, device),
        "post_inputs": tree_to_device(post_inputs, device),
        "outputs": tree_to_device(outputs, device),
    }


def set_current_case(case: dict[str, Any]) -> None:
    global CURRENT_CASE
    CURRENT_CASE = case


def get_current_case() -> dict[str, Any]:
    if CURRENT_CASE is None:
        raise RuntimeError("No current snapshot case is set.")
    return CURRENT_CASE


def tree_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
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
    if isinstance(value, torch.Tensor):
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
        elif isinstance(cur, list):
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot descend into {type(cur)!r} at {part!r}")
    last = parts[-1]
    if isinstance(cur, dict):
        if last in cur and isinstance(cur[last], torch.Tensor) and isinstance(value, torch.Tensor):
            cur[last].copy_(value)
        else:
            cur[last] = value
    elif isinstance(cur, list):
        index = int(last)
        if isinstance(cur[index], torch.Tensor) and isinstance(value, torch.Tensor):
            cur[index].copy_(value)
        else:
            cur[index] = value
    else:
        raise KeyError(f"Cannot set path {path!r}")


def call_tree_to_extend_inputs(call_tree: dict[str, Any]) -> dict[str, Any]:
    args = list(call_tree.get("args", ()))
    kwargs = dict(call_tree.get("kwargs", {}))
    names = ["q", "k", "v", "g", "beta"]
    out = {name: args[idx] for idx, name in enumerate(names) if idx < len(args)}
    out.update(kwargs)
    return out


def extend_inputs_to_call_tree(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc):
    return {
        "args": [q, k, v, g, beta],
        "kwargs": {
            "ssm_states": ssm_states,
            "cache_indices": cache_indices,
            "query_start_loc": query_start_loc,
        },
    }


def apply_snapshot_mutations(call_tree: dict[str, Any], case: dict[str, Any]) -> None:
    meta = case["meta"]
    post_inputs = case["post_inputs"]
    for path in meta.get("mutation", {}).get("mutable_arg_paths", []):
        set_path(call_tree, path, tree_clone(get_path(post_inputs, path)))


def assert_tree_close(actual: Any, expected: Any, *, atol: float, rtol: float, path: str = "") -> None:
    if isinstance(expected, torch.Tensor):
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


REFERENCE_IMPL = '''"""Reference implementation generated by Framework Engineer.

Default mode is snapshot-golden: it returns captured outputs and applies captured
mutable post-state. Framework Engineer may replace this with the current reliable
SGLang implementation for reference-replay benchmarking.
"""

from __future__ import annotations

import snapshot_runtime


def reference_{suffix}(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc):
    case = snapshot_runtime.get_current_case()
    call_tree = snapshot_runtime.extend_inputs_to_call_tree(
        q, k, v, g, beta,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )
    snapshot_runtime.apply_snapshot_mutations(call_tree, case)
    return snapshot_runtime.tree_clone(case["outputs"])
'''


CANDIDATE_IMPL = '''"""Candidate implementation generated for Kernel Engineer.

Initial candidate delegates to reference so the task pack starts with passing
correctness. Kernel Engineer should replace this implementation only.
"""

from __future__ import annotations

import reference_impl


def candidate_{suffix}(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc):
    return reference_impl.reference_{suffix}(
        q, k, v, g, beta,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )
'''


CORRECTNESS_TEST = '''"""Snapshot replay correctness harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import candidate_impl
import reference_impl
import snapshot_runtime


def run_case(case_meta, *, device: str, mode: str) -> dict:
    case = snapshot_runtime.load_case(case_meta["case_id"], device=device)
    snapshot_runtime.set_current_case(case)
    tol = case["meta"].get("tolerance", {{}})
    atol = float(tol.get("atol", 2e-2))
    rtol = float(tol.get("rtol", 2e-2))

    ref_tree = snapshot_runtime.tree_clone(case["pre_inputs"])
    cand_tree = snapshot_runtime.tree_clone(case["pre_inputs"])
    ref_inputs = snapshot_runtime.call_tree_to_extend_inputs(ref_tree)
    cand_inputs = snapshot_runtime.call_tree_to_extend_inputs(cand_tree)

    if mode == "reference-replay":
        expected = reference_impl.reference_{suffix}(**ref_inputs)
        expected_mut_tree = ref_tree
    else:
        expected = snapshot_runtime.tree_clone(case["outputs"])
        expected_mut_tree = case["post_inputs"]

    actual = candidate_impl.{candidate_function}(**cand_inputs)
    snapshot_runtime.assert_tree_close(actual, expected, atol=atol, rtol=rtol)

    for path in case["meta"].get("mutation", {{}}).get("mutable_arg_paths", []):
        snapshot_runtime.assert_tree_close(
            snapshot_runtime.get_path(cand_tree, path),
            snapshot_runtime.get_path(expected_mut_tree, path),
            atol=atol,
            rtol=rtol,
            path=path,
        )

    return {{"case_id": case_meta["case_id"], "status": "PASS", "mode": mode}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["reference-replay", "snapshot-golden"], default="snapshot-golden")
    parser.add_argument("--all-priorities", action="store_true")
    args = parser.parse_args()

    priority = None if args.all_priorities else "required"
    cases = snapshot_runtime.list_cases(priority=priority)
    for case in cases:
        if args.case_id is not None and case["case_id"] != args.case_id:
            continue
        print(json.dumps(run_case(case, device=args.device, mode=args.mode), sort_keys=True))


if __name__ == "__main__":
    main()
'''


BENCHMARK = '''"""Snapshot replay benchmark harness."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

import candidate_impl
import reference_impl
import snapshot_runtime


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_reference(inputs):
    return reference_impl.reference_{suffix}(**inputs)


def run_candidate(inputs):
    return candidate_impl.{candidate_function}(**inputs)


def elapsed_us(fn, make_inputs, *, warmup: int, repeat: int, use_cuda_events: bool) -> dict:
    for _ in range(warmup):
        fn(make_inputs())
    sync()

    values = []
    for _ in range(repeat):
        inputs = make_inputs()
        sync()
        if use_cuda_events and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(inputs)
            end.record()
            sync()
            values.append(float(start.elapsed_time(end) * 1000.0))
        else:
            start_t = time.perf_counter()
            fn(inputs)
            sync()
            values.append((time.perf_counter() - start_t) * 1_000_000.0)
    return {{
        "median_us": statistics.median(values),
        "mean_us": statistics.mean(values),
        "min_us": min(values),
        "max_us": max(values),
    }}


def benchmark_case(case_meta, *, device: str, target: str, warmup: int, repeat: int) -> None:
    case = snapshot_runtime.load_case(case_meta["case_id"], device=device)
    snapshot_runtime.set_current_case(case)

    def make_inputs():
        tree = snapshot_runtime.tree_clone(case["pre_inputs"])
        return snapshot_runtime.call_tree_to_extend_inputs(tree)

    result = {{"case_id": case_meta["case_id"], "warmup": warmup, "repeat": repeat}}
    use_events = device.startswith("cuda")
    if target in ("reference", "both"):
        result["reference"] = elapsed_us(run_reference, make_inputs, warmup=warmup, repeat=repeat, use_cuda_events=use_events)
    if target in ("candidate", "both"):
        result["candidate"] = elapsed_us(run_candidate, make_inputs, warmup=warmup, repeat=repeat, use_cuda_events=use_events)
    if "reference" in result and "candidate" in result and result["candidate"]["median_us"] > 0:
        result["speedup_median"] = result["reference"]["median_us"] / result["candidate"]["median_us"]
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", choices=["reference", "candidate", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--all-priorities", action="store_true")
    args = parser.parse_args()

    priority = None if args.all_priorities else "required"
    cases = snapshot_runtime.list_cases(priority=priority)
    for case in cases:
        if args.case_id is not None and case["case_id"] != args.case_id:
            continue
        benchmark_case(case, device=args.device, target=args.target, warmup=args.warmup, repeat=args.repeat)


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

CASE_ID="${1:-}"
if [ -z "$CASE_ID" ]; then
  echo "usage: bash scripts/run_ncu.sh <case_id>" >&2
  exit 2
fi

ncu --set full --target-processes all "${PYTHON:-python3}" benchmark.py --case-id "$CASE_ID" --device "${DEVICE:-cuda}" --target "${TARGET:-candidate}" --warmup "${WARMUP:-5}" --repeat "${REPEAT:-20}"
'''
