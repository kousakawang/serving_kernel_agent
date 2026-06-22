"""Reference implementation entry for Qwen3.5 GDN linear attention core.

Default template uses snapshot-golden replay: it returns captured outputs and
applies captured mutable post-state. Framework Engineer may replace this with a
current reliable SGLang implementation for reference-replay benchmarking.
"""

from __future__ import annotations

import snapshot_runtime


def reference_extend(
    q,
    k,
    v,
    g,
    beta,
    *,
    ssm_states,
    cache_indices,
    query_start_loc,
):
    """Reference for GDN prefill/extend core."""
    case = snapshot_runtime.get_current_case()
    call_tree = snapshot_runtime.extend_inputs_to_call_tree(
        q, k, v, g, beta,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )
    snapshot_runtime.apply_snapshot_mutations(call_tree, case)
    return snapshot_runtime.tree_clone(case["outputs"])


def reference_decode(
    q,
    k,
    v,
    a,
    b,
    *,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    query_start_loc,
):
    """Optional reference for GDN decode core."""
    raise NotImplementedError("Optional: provide reference_decode if task.yaml requires it.")
