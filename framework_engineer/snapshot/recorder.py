"""Runtime helpers for probing calls and capturing raw snapshots."""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any, Callable

from . import hashing
from .models import SCHEMA_VERSION, SnapshotCase
from .store import SnapshotStore
from .tree import tree_meta, tree_to_cpu


def _torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def _sync_cuda() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def make_probe_decorator(log_path: str | Path, target_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that logs calls without saving tensor payloads."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            record = {
                "target": target_name,
                "qualified_name": f"{fn.__module__}.{fn.__qualname__}",
                "time": time.time(),
                "arg_count": len(args),
                "kwarg_keys": sorted(kwargs),
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            return fn(*args, **kwargs)

        return wrapper

    return decorate


class SnapshotRecorder:
    """Capture pre inputs, outputs, and post inputs for a Python-callable target."""

    def __init__(
        self,
        store: SnapshotStore,
        *,
        task_id: str,
        target: dict[str, Any],
        signature: str,
        mutable_arg_paths: list[str] | None = None,
        tolerance: dict[str, float] | None = None,
        drop_first_arg: bool = False,
        max_raw_cases: int | None = None,
    ):
        self.store = store
        self.store.ensure()
        self.task_id = task_id
        self.target = target
        self.signature = signature
        self.mutable_arg_paths = mutable_arg_paths or []
        self.tolerance = tolerance or {"atol": 2e-2, "rtol": 2e-2}
        self.drop_first_arg = drop_first_arg
        self.max_raw_cases = max_raw_cases
        self.call_index = len(self.store.list_raw_cases())

    def decorate(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.max_raw_cases is not None and self.call_index >= self.max_raw_cases:
                return fn(*args, **kwargs)
            capture_args = args[1:] if self.drop_first_arg else args
            pre_inputs = {"args": tree_to_cpu(tuple(capture_args)), "kwargs": tree_to_cpu(dict(kwargs))}
            _sync_cuda()
            outputs = fn(*args, **kwargs)
            _sync_cuda()
            post_inputs = {"args": tree_to_cpu(tuple(capture_args)), "kwargs": tree_to_cpu(dict(kwargs))}
            saved_outputs = tree_to_cpu(outputs)
            self.save_call(pre_inputs, post_inputs, saved_outputs)
            return outputs

        return wrapper

    def save_call(self, pre_inputs: dict[str, Any], post_inputs: dict[str, Any], outputs: Any) -> SnapshotCase:
        torch = _torch()
        if torch is None:
            raise RuntimeError("Snapshot capture requires torch to save payloads.")

        self.call_index += 1
        call_id = f"call_{self.call_index:06d}"
        call_dir = self.store.raw_case_dir(call_id)
        call_dir.mkdir(parents=True, exist_ok=True)

        torch.save(pre_inputs, call_dir / "pre_inputs.pt")
        torch.save(post_inputs, call_dir / "post_inputs.pt")
        torch.save(outputs, call_dir / "outputs.pt")

        input_meta = tree_meta(pre_inputs)
        output_meta = tree_meta(outputs, "outputs")
        post_input_meta = tree_meta(post_inputs)
        shape_digest = hashing.shape_hash(self.target, input_meta)
        semantic_digest = hashing.semantic_hash(shape_digest, pre_inputs, self.target)
        value_digest = hashing.value_hash({"inputs": pre_inputs, "outputs": outputs})
        key = hashing.case_key(SCHEMA_VERSION, self.target, semantic_digest)

        case = SnapshotCase(
            task_id=self.task_id,
            case_id=call_id,
            raw_call_ids=[call_id],
            target=self.target,
            interface={
                "signature": self.signature,
                "args_tree": input_meta.get("items", {}).get("args"),
                "kwargs_tree": input_meta.get("items", {}).get("kwargs"),
                "output_tree": output_meta,
                "post_input_tree": post_input_meta,
            },
            files={
                "pre_inputs": "pre_inputs.pt",
                "post_inputs": "post_inputs.pt",
                "outputs": "outputs.pt",
            },
            mutation={
                "mutable_arg_paths": list(self.mutable_arg_paths),
                "compare_mutations": bool(self.mutable_arg_paths),
            },
            hashes={
                "shape_hash": shape_digest,
                "semantic_hash": semantic_digest,
                "value_hash": value_digest,
                "case_key": key,
            },
            selection={"call_count": 1, "priority": "raw", "reason": "captured_call"},
            tolerance=self.tolerance,
        )
        self.store.write_case_meta(call_dir, case)
        return case


def make_snapshot_decorator(
    snapshot_root: str | Path,
    task_id: str,
    target_name: str,
    signature: str,
    mutable_arg_paths: str = "",
    mode: str = "",
    backend: str = "",
    layer_id: str = "",
    drop_first_arg: bool = False,
    max_raw_cases: int | str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    paths = [p.strip() for p in mutable_arg_paths.split(",") if p.strip()]
    target = {
        "qualified_name": target_name,
        "logical_name": target_name.split(".")[-1],
        "mode": mode or None,
        "backend": backend or None,
        "layer_id": int(layer_id) if str(layer_id).isdigit() else None,
    }
    recorder = SnapshotRecorder(
        SnapshotStore(Path(snapshot_root)),
        task_id=task_id,
        target=target,
        signature=signature,
        mutable_arg_paths=paths,
        drop_first_arg=drop_first_arg,
        max_raw_cases=int(max_raw_cases) if max_raw_cases not in (None, "") else None,
    )
    return recorder.decorate
