from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from kernel_agent.framework_engineer.snapshot.harness_builder import SnapshotHarnessBuilder
from kernel_agent.framework_engineer.snapshot.recorder import SnapshotRecorder, make_forward_boundary_decorator
from kernel_agent.framework_engineer.snapshot.selector import SnapshotSelector, write_shape_list_summary
from kernel_agent.framework_engineer.snapshot.store import SnapshotStore
from kernel_agent.framework_engineer.snapshot.tree import tree_meta


class SnapshotTests(unittest.TestCase):
    def _capture_primitive_calls(self, store: SnapshotStore, *, task_id: str = "task_pack") -> None:
        recorder = SnapshotRecorder(
            store,
            task_id=task_id,
            target={"qualified_name": "toy.extend", "logical_name": "extend", "mode": "extend", "backend": "test"},
            signature="candidate(*args, **kwargs)",
            mutable_arg_paths=["kwargs.state.total"],
            max_capture_groups=8,
            max_samples_per_group=4,
            max_samples_per_forward_per_group=2,
        )

        @recorder.decorate
        def target(*, values, state):
            state["total"] += sum(values)
            return {"out": [v + 1 for v in values]}

        @make_forward_boundary_decorator("toy.forward")
        def forward(values):
            target(values=values, state={"total": 0})
            target(values=values, state={"total": 0})
            target(values=values, state={"total": 0})

        forward([1, 2, 3])
        forward([1, 2, 3])

    @unittest.skipIf(torch is None, "torch is required for tensor metadata test")
    def test_tensor_meta_records_layout(self) -> None:
        tensor = torch.zeros(2, 3).t()
        meta = tree_meta({"x": tensor})["items"]["x"]["meta"]
        self.assertEqual(meta["shape"], [3, 2])
        self.assertEqual(meta["stride"], [1, 3])
        self.assertFalse(meta["is_contiguous"])
        self.assertEqual(meta["storage_offset"], 0)

    def test_group_hit_count_and_sample_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp) / "snapshots")
            self._capture_primitive_calls(store)
            index = store.read_raw_index()
            self.assertEqual(index["raw_group_count"], 1)
            self.assertEqual(index["total_hit_count"], 6)
            group = next(iter(index["groups"].values()))
            self.assertEqual(group["total_hit_count"], 6)
            self.assertEqual(group["forward_hit_count"], 2)
            self.assertEqual(group["sample_count"], 4)
            for count in group["samples_per_forward"].values():
                self.assertLessEqual(count, 2)

    def test_select_and_generated_harness_passes_without_torch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_pack = Path(tmp) / "task_pack"
            (task_pack / "snapshots" / "raw").mkdir(parents=True)
            (task_pack / "snapshots" / "selected").mkdir(parents=True)
            store = SnapshotStore(task_pack / "snapshots")
            self._capture_primitive_calls(store, task_id=task_pack.name)
            manifest = SnapshotSelector(store).select(max_groups=1, max_samples_per_group=4)
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
