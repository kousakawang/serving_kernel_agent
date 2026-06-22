from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from kernel_agent.framework_engineer.snapshot.recorder import SnapshotRecorder
from kernel_agent.framework_engineer.snapshot.selector import SnapshotSelector, write_shape_list_summary
from kernel_agent.framework_engineer.snapshot.harness_builder import SnapshotHarnessBuilder
from kernel_agent.framework_engineer.snapshot.store import SnapshotStore
from kernel_agent.framework_engineer.snapshot.tree import tree_meta


@unittest.skipIf(torch is None, "torch is required for snapshot tests")
class SnapshotTests(unittest.TestCase):
    def _save_case(self, store: SnapshotStore, query_start_loc, *, task_id: str = "task_pack") -> None:
        q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        k = q + 1
        v = q + 2
        g = q + 3
        beta = q + 4
        ssm_pre = torch.zeros(2, 2)
        ssm_post = torch.ones(2, 2)
        cache_indices = torch.tensor([0, 1], dtype=torch.int64)
        pre = {
            "args": (q, k, v, g, beta),
            "kwargs": {
                "ssm_states": ssm_pre,
                "cache_indices": cache_indices,
                "query_start_loc": torch.tensor(query_start_loc, dtype=torch.int64),
            },
        }
        post = {
            "args": (q, k, v, g, beta),
            "kwargs": {
                "ssm_states": ssm_post,
                "cache_indices": cache_indices,
                "query_start_loc": torch.tensor(query_start_loc, dtype=torch.int64),
            },
        }
        recorder = SnapshotRecorder(
            store,
            task_id=task_id,
            target={"qualified_name": "test.extend", "logical_name": "gdn_extend_core_v1", "mode": "extend", "backend": "test"},
            signature="candidate_extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc)",
            mutable_arg_paths=["kwargs.ssm_states"],
        )
        recorder.save_call(pre, post, q + k)

    def test_tensor_meta_records_layout(self) -> None:
        tensor = torch.zeros(2, 3).t()
        meta = tree_meta({"x": tensor})["items"]["x"]["meta"]
        self.assertEqual(meta["shape"], [3, 2])
        self.assertEqual(meta["stride"], [1, 3])
        self.assertFalse(meta["is_contiguous"])
        self.assertEqual(meta["storage_offset"], 0)

    def test_semantic_hash_separates_query_start_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp) / "snapshots")
            self._save_case(store, [0, 2, 4])
            self._save_case(store, [0, 1, 4])
            cases = store.list_raw_cases()
            self.assertEqual(len(cases), 2)
            self.assertNotEqual(cases[0].hashes["semantic_hash"], cases[1].hashes["semantic_hash"])

    def test_select_and_generated_harness_passes_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_pack = Path(tmp) / "task_pack"
            (task_pack / "snapshots" / "raw").mkdir(parents=True)
            (task_pack / "snapshots" / "selected").mkdir(parents=True)
            store = SnapshotStore(task_pack / "snapshots")
            self._save_case(store, [0, 2, 4], task_id=task_pack.name)
            manifest = SnapshotSelector(store).select()
            write_shape_list_summary(task_pack, manifest)
            SnapshotHarnessBuilder(task_pack).generate()
            proc = subprocess.run(
                [sys.executable, "correctness_test.py", "--device", "cpu"],
                cwd=task_pack,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('"status": "PASS"', proc.stdout)


if __name__ == "__main__":
    unittest.main()

