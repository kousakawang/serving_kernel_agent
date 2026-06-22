"""Snapshot replay benchmark harness template."""

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
    return reference_impl.reference_extend(**inputs)


def run_candidate(inputs):
    return candidate_impl.candidate_extend(**inputs)


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
    return {
        "median_us": statistics.median(values),
        "mean_us": statistics.mean(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def benchmark_case(case_meta, *, device: str, target: str, warmup: int, repeat: int) -> None:
    case = snapshot_runtime.load_case(case_meta["case_id"], device=device)
    snapshot_runtime.set_current_case(case)

    def make_inputs():
        tree = snapshot_runtime.tree_clone(case["pre_inputs"])
        return snapshot_runtime.call_tree_to_extend_inputs(tree)

    result = {"case_id": case_meta["case_id"], "warmup": warmup, "repeat": repeat}
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
    for case in snapshot_runtime.list_cases(priority=priority):
        if args.case_id is not None and case["case_id"] != args.case_id:
            continue
        benchmark_case(case, device=args.device, target=args.target, warmup=args.warmup, repeat=args.repeat)


if __name__ == "__main__":
    main()

