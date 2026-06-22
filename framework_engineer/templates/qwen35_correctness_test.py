"""Snapshot replay correctness harness template."""

from __future__ import annotations

import argparse
import json

import candidate_impl
import reference_impl
import snapshot_runtime


def run_case(case_meta, *, device: str, mode: str) -> dict:
    case = snapshot_runtime.load_case(case_meta["case_id"], device=device)
    snapshot_runtime.set_current_case(case)
    tol = case["meta"].get("tolerance", {})
    atol = float(tol.get("atol", 2e-2))
    rtol = float(tol.get("rtol", 2e-2))

    ref_tree = snapshot_runtime.tree_clone(case["pre_inputs"])
    cand_tree = snapshot_runtime.tree_clone(case["pre_inputs"])
    ref_inputs = snapshot_runtime.call_tree_to_extend_inputs(ref_tree)
    cand_inputs = snapshot_runtime.call_tree_to_extend_inputs(cand_tree)

    if mode == "reference-replay":
        expected = reference_impl.reference_extend(**ref_inputs)
        expected_mut_tree = ref_tree
    else:
        expected = snapshot_runtime.tree_clone(case["outputs"])
        expected_mut_tree = case["post_inputs"]

    actual = candidate_impl.candidate_extend(**cand_inputs)
    snapshot_runtime.assert_tree_close(actual, expected, atol=atol, rtol=rtol)

    for path in case["meta"].get("mutation", {}).get("mutable_arg_paths", []):
        snapshot_runtime.assert_tree_close(
            snapshot_runtime.get_path(cand_tree, path),
            snapshot_runtime.get_path(expected_mut_tree, path),
            atol=atol,
            rtol=rtol,
            path=path,
        )

    return {"case_id": case_meta["case_id"], "status": "PASS", "mode": mode}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["reference-replay", "snapshot-golden"], default="snapshot-golden")
    parser.add_argument("--all-priorities", action="store_true")
    args = parser.parse_args()

    priority = None if args.all_priorities else "required"
    for case in snapshot_runtime.list_cases(priority=priority):
        if args.case_id is not None and case["case_id"] != args.case_id:
            continue
        print(json.dumps(run_case(case, device=args.device, mode=args.mode), sort_keys=True))


if __name__ == "__main__":
    main()

