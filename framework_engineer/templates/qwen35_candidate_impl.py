"""Candidate implementation entry for Qwen3.5 GDN linear attention core.

Initial template delegates to reference so the task pack starts with passing
correctness. Kernel Engineer should replace this implementation only.
"""

from __future__ import annotations

import reference_impl


def candidate_extend(
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
    """Candidate for GDN prefill/extend core."""
    return reference_impl.reference_extend(
        q, k, v, g, beta,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )


def candidate_decode(
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
    """Optional candidate for GDN decode core."""
    raise NotImplementedError("Optional: implement candidate_decode if task.yaml requires it.")
