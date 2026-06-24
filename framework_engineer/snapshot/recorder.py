"""Runtime helpers for probing calls and capturing grouped snapshots."""

from __future__ import annotations

import contextvars
import functools
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from . import hashing
from .models import SCHEMA_VERSION, SnapshotSample
from .store import SnapshotStore
from .tree import get_path, tree_meta, tree_to_cpu


_CURRENT_FORWARD_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "kernel_agent_current_forward_id",
    default=None,
)
_FORWARD_SEQ: defaultdict[str, int] = defaultdict(int)


def _torch():
    try:
        import torch
    except Exception:  # pragma: no cover - torch may be absent on local hosts.
        return None
    return torch


def _sync_cuda() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def current_forward_id() -> str | None:
    return _CURRENT_FORWARD_ID.get()


def make_forward_boundary_decorator(boundary_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that marks a high-level forward window."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            qualified = boundary_name or f"{fn.__module__}.{fn.__qualname__}"
            seq = _FORWARD_SEQ[qualified]
            _FORWARD_SEQ[qualified] += 1
            forward_id = f"{os.getpid()}:{qualified}:{seq:06d}"
            token = _CURRENT_FORWARD_ID.set(forward_id)
            try:
                return fn(*args, **kwargs)
            finally:
                _CURRENT_FORWARD_ID.reset(token)

        return wrapper

    return decorate


def make_probe_decorator(
    log_path: str | Path,
    target_name: str,
    drop_first_arg: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that logs calls without saving tensor payloads."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            captured_args = args[1:] if drop_first_arg else args
            record = {
                "target": target_name,
                "qualified_name": f"{fn.__module__}.{fn.__qualname__}",
                "time": time.time(),
                "forward_id": current_forward_id(),
                "positional_arg_count": len(args),
                "kwarg_count": len(kwargs),
                "kwarg_keys": sorted(kwargs),
                "drop_first_arg": bool(drop_first_arg),
                "captured_positional_arg_count": len(captured_args),
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            return fn(*args, **kwargs)

        return wrapper

    return decorate


class SnapshotRecorder:
    """Capture grouped pre inputs, outputs, and post inputs for a target."""

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
        calls_per_forward: int | None = None,
        max_capture_groups: int = 64,
        max_samples_per_group: int = 8,
        max_samples_per_forward_per_group: int = 3,
    ):
        self.store = store
        self.store.ensure()
        self.task_id = task_id
        self.target = target
        self.signature = signature
        self.mutable_arg_paths = mutable_arg_paths or []
        self.tolerance = tolerance or {"atol": 2e-2, "rtol": 2e-2}
        self.drop_first_arg = drop_first_arg
        self.calls_per_forward = calls_per_forward
        self.max_capture_groups = max_capture_groups
        self.max_samples_per_group = max_samples_per_group
        self.max_samples_per_forward_per_group = max_samples_per_forward_per_group
        index = self.store.read_raw_index()
        self.call_index_global = int(index.get("total_hit_count", 0))
        self.forward_call_counts: dict[str, int] = {}

    def decorate(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_index_global = self.call_index_global
            self.call_index_global += 1
            forward_id = self._resolve_forward_id(call_index_global)
            call_index_in_forward = self.forward_call_counts.get(forward_id, 0)
            self.forward_call_counts[forward_id] = call_index_in_forward + 1

            capture_args = args[1:] if self.drop_first_arg else args
            pre_inputs = {"args": tree_to_cpu(tuple(capture_args)), "kwargs": tree_to_cpu(dict(kwargs))}
            pre_mutable_paths, pre_mutation_warnings = self._resolve_mutable_paths(pre_inputs, "pre_inputs")
            input_meta = tree_meta(pre_inputs)
            shape_digest = hashing.shape_hash(self.target, input_meta)
            group_digest = hashing.group_key(self.target, input_meta)

            should_track, should_save, sample_id, reason = self._plan_capture(
                group_digest,
                shape_digest,
                input_meta,
                forward_id,
            )

            _sync_cuda()
            outputs = fn(*args, **kwargs)
            _sync_cuda()

            if not should_track:
                self._record_dropped_hit(group_digest)
                return outputs

            if not should_save:
                self._record_group_hit(group_digest, forward_id, call_index_global, call_index_in_forward)
                return outputs

            post_inputs = {"args": tree_to_cpu(tuple(capture_args)), "kwargs": tree_to_cpu(dict(kwargs))}
            post_mutable_paths, post_mutation_warnings = self._resolve_mutable_paths(post_inputs, "post_inputs")
            effective_mutable_paths = [path for path in pre_mutable_paths if path in set(post_mutable_paths)]
            mutation_warnings = pre_mutation_warnings + post_mutation_warnings
            skipped_post_paths = sorted(set(pre_mutable_paths) - set(post_mutable_paths))
            for path in skipped_post_paths:
                mutation_warnings.append(
                    f"mutable_arg_path {path!r} exists in pre_inputs but not in post_inputs; mutation comparison disabled for this path"
                )
            saved_outputs = tree_to_cpu(outputs)
            self.save_sample(
                group_key=group_digest,
                shape_hash=shape_digest,
                sample_id=sample_id,
                reason=reason,
                pre_inputs=pre_inputs,
                post_inputs=post_inputs,
                outputs=saved_outputs,
                forward_id=forward_id,
                call_index_global=call_index_global,
                call_index_in_forward=call_index_in_forward,
                effective_mutable_paths=effective_mutable_paths,
                mutation_warnings=mutation_warnings,
            )
            return outputs

        return wrapper

    def save_sample(
        self,
        *,
        group_key: str,
        shape_hash: str,
        sample_id: str,
        reason: str,
        pre_inputs: dict[str, Any],
        post_inputs: dict[str, Any],
        outputs: Any,
        forward_id: str,
        call_index_global: int,
        call_index_in_forward: int,
        effective_mutable_paths: list[str] | None = None,
        mutation_warnings: list[str] | None = None,
    ) -> SnapshotSample:
        input_meta = tree_meta(pre_inputs)
        output_meta = tree_meta(outputs, "outputs")
        post_input_meta = tree_meta(post_inputs)
        group_id = self._group_id(group_key)
        sample_dir = self.store.raw_sample_dir(group_id, sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        serializer = self.store.save_payload(pre_inputs, sample_dir / "pre_inputs.pt")
        self.store.save_payload(post_inputs, sample_dir / "post_inputs.pt")
        self.store.save_payload(outputs, sample_dir / "outputs.pt")

        value_digest = hashing.value_hash({"inputs": pre_inputs, "outputs": outputs})
        effective_mutable_paths = effective_mutable_paths or []
        mutation_warnings = mutation_warnings or []
        sample = SnapshotSample(
            task_id=self.task_id,
            group_id=group_id,
            sample_id=sample_id,
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
                "serializer": serializer,
            },
            mutation={
                "requested_mutable_arg_paths": list(self.mutable_arg_paths),
                "mutable_arg_paths": list(effective_mutable_paths),
                "ignored_mutable_arg_paths": [
                    path for path in self.mutable_arg_paths if path not in set(effective_mutable_paths)
                ],
                "mutation_warnings": list(mutation_warnings),
                "compare_mutations": bool(effective_mutable_paths),
            },
            hashes={
                "shape_hash": shape_hash,
                "group_key": group_key,
                "value_hash": value_digest,
                "sample_key": hashing.short_hash(
                    {"schema_version": SCHEMA_VERSION, "group_key": group_key, "value_hash": value_digest},
                    16,
                ),
            },
            capture={
                "forward_id": forward_id,
                "call_index_global": call_index_global,
                "call_index_in_forward": call_index_in_forward,
                "time": time.time(),
                "reason": reason,
                "mutation_warning_count": len(mutation_warnings),
            },
            tolerance=self.tolerance,
        )
        self.store.write_sample_meta(sample_dir, sample)
        self._record_group_hit(
            group_key,
            forward_id,
            call_index_global,
            call_index_in_forward,
            saved_sample=sample,
        )
        return sample

    def _resolve_forward_id(self, call_index_global: int) -> str:
        explicit = current_forward_id()
        if explicit:
            return explicit
        if self.calls_per_forward and self.calls_per_forward > 0:
            return f"{os.getpid()}:calls_per_forward:{call_index_global // self.calls_per_forward:06d}"
        return "unknown"

    def _plan_capture(
        self,
        group_key: str,
        shape_hash: str,
        input_meta: dict[str, Any],
        forward_id: str,
    ) -> tuple[bool, bool, str, str]:
        index = self.store.read_raw_index()
        groups = index.setdefault("groups", {})
        group_id = self._group_id(group_key)
        group = groups.get(group_key)
        if group is None and len(groups) >= self.max_capture_groups:
            return False, False, "", "max_capture_groups_reached"

        if group is None:
            return True, True, "sample_0001", "first_group_sample"

        if int(group.get("sample_count", 0)) >= self.max_samples_per_group:
            return True, False, "", "max_samples_per_group_reached"
        per_forward = group.get("samples_per_forward", {})
        if int(per_forward.get(forward_id, 0)) >= self.max_samples_per_forward_per_group:
            return True, False, "", "max_samples_per_forward_per_group_reached"
        sample_id = f"sample_{int(group.get('sample_count', 0)) + 1:04d}"
        reason = "cross_forward_sample" if forward_id not in set(group.get("forward_ids", [])) else "same_forward_sample"
        return True, True, sample_id, reason

    def _record_group_hit(
        self,
        group_key: str,
        forward_id: str,
        call_index_global: int,
        call_index_in_forward: int,
        *,
        saved_sample: SnapshotSample | None = None,
    ) -> None:
        index = self.store.read_raw_index()
        index.setdefault("schema_version", SCHEMA_VERSION)
        index.setdefault("index_type", "raw_group_index")
        groups = index.setdefault("groups", {})
        group_id = self._group_id(group_key)
        if group_key not in groups:
            if saved_sample is None:
                return
            groups[group_key] = {
                "task_id": self.task_id,
                "group_id": group_id,
                "group_key": group_key,
                "shape_hash": saved_sample.hashes["shape_hash"],
                "target": saved_sample.target,
                "interface": saved_sample.interface,
                "mutation": saved_sample.mutation,
                "tolerance": saved_sample.tolerance,
                "total_hit_count": 0,
                "forward_ids": [],
                "forward_hit_count": 0,
                "sample_count": 0,
                "samples": [],
                "samples_per_forward": {},
                "first_seen": None,
                "last_seen": None,
            }
        group = groups[group_key]
        group["total_hit_count"] = int(group.get("total_hit_count", 0)) + 1
        forward_ids = list(group.get("forward_ids", []))
        if forward_id not in forward_ids:
            forward_ids.append(forward_id)
        group["forward_ids"] = forward_ids
        group["forward_hit_count"] = len(forward_ids)
        group["last_seen"] = {
            "forward_id": forward_id,
            "call_index_global": call_index_global,
            "call_index_in_forward": call_index_in_forward,
        }
        if group.get("first_seen") is None:
            group["first_seen"] = dict(group["last_seen"])
        if saved_sample is not None:
            samples = list(group.get("samples", []))
            samples.append(
                {
                    "sample_id": saved_sample.sample_id,
                    "group_id": saved_sample.group_id,
                    "sample_dir": f"raw/{saved_sample.group_id}/{saved_sample.sample_id}",
                    "hashes": saved_sample.hashes,
                    "capture": saved_sample.capture,
                    "files": saved_sample.files,
                }
            )
            group["samples"] = samples
            group["sample_count"] = len(samples)
            per_forward = dict(group.get("samples_per_forward", {}))
            per_forward[forward_id] = int(per_forward.get(forward_id, 0)) + 1
            group["samples_per_forward"] = per_forward
        groups[group_key] = group
        self.store.write_raw_index(index)

    def _record_dropped_hit(self, group_key: str) -> None:
        index = self.store.read_raw_index()
        index["dropped_hit_count"] = int(index.get("dropped_hit_count", 0)) + 1
        dropped_groups = index.setdefault("dropped_group_keys", [])
        if group_key not in dropped_groups:
            dropped_groups.append(group_key)
        self.store.write_raw_index(index)

    def _resolve_mutable_paths(self, tree: dict[str, Any], tree_name: str) -> tuple[list[str], list[str]]:
        valid_paths = []
        warnings = []
        for path in self.mutable_arg_paths:
            try:
                get_path(tree, path)
            except Exception as exc:
                warnings.append(
                    f"mutable_arg_path {path!r} not found in captured {tree_name}; mutation comparison disabled for this path"
                )
            else:
                valid_paths.append(path)
        return valid_paths, warnings

    @staticmethod
    def _group_id(group_key: str) -> str:
        return f"group_{group_key[:12]}"


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
    source_info: dict[str, Any] | None = None,
    calls_per_forward: int | str | None = None,
    max_capture_groups: int | str = 64,
    max_samples_per_group: int | str = 8,
    max_samples_per_forward_per_group: int | str = 3,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    paths = [p.strip() for p in mutable_arg_paths.split(",") if p.strip()]
    target = {
        "qualified_name": target_name,
        "logical_name": target_name.split(".")[-1],
        "mode": mode or None,
        "backend": backend or None,
        "layer_id": int(layer_id) if str(layer_id).isdigit() else None,
        "source": source_info or {},
    }
    recorder = SnapshotRecorder(
        SnapshotStore(Path(snapshot_root)),
        task_id=task_id,
        target=target,
        signature=signature,
        mutable_arg_paths=paths,
        drop_first_arg=drop_first_arg,
        calls_per_forward=int(calls_per_forward) if calls_per_forward not in (None, "") else None,
        max_capture_groups=int(max_capture_groups),
        max_samples_per_group=int(max_samples_per_group),
        max_samples_per_forward_per_group=int(max_samples_per_forward_per_group),
    )
    return recorder.decorate
