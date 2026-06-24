from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Phase1CliEndToEndTests(unittest.TestCase):
    """Exercise the Framework Engineer CLI on a tiny pure-Python target."""

    def test_cli_probe_capture_select_generate_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_pack = tmp_path / "task_pack"
            target_file = tmp_path / "toy_kernel.py"
            workload_file = tmp_path / "workload.py"
            service_file = tmp_path / "service.py"

            self._write_target(target_file)
            self._write_workload(workload_file, tmp_path)
            service_file.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")

            service_cmd = f"{sys.executable} {service_file}"
            workload_cmd = f"{sys.executable} {workload_file}"
            target_line = self._line_for(target_file, "def extend(*, values, state):")
            boundary_line = self._line_for(target_file, "def forward_window(self, values):")

            self._run_cli("scaffold-task-pack", "--task-id", "toy_extend", "--out", str(task_pack))

            baseline = self._run_cli(
                "run-baseline",
                "--task-pack",
                str(task_pack),
                "--service-cmd",
                service_cmd,
                "--workload-cmd",
                workload_cmd,
                "--startup-timeout",
                "1",
            )
            self.assertEqual(baseline["status"], "ok")

            probe = self._run_cli(
                "probe-target-calls",
                "--task-pack",
                str(task_pack),
                "--service-cmd",
                service_cmd,
                "--workload-cmd",
                workload_cmd,
                "--target-file",
                str(target_file),
                "--target-line",
                str(target_line),
                "--forward-boundary-file",
                str(target_file),
                "--forward-boundary-line",
                str(boundary_line),
                "--startup-timeout",
                "1",
            )
            self.assertEqual(probe["call_count"], 6)
            self.assertEqual(probe["target_interface"]["function_name"], "extend")
            self.assertEqual(probe["target_interface"]["qualified_name"], "toy_kernel.extend")
            self.assertEqual(probe["forward_boundary_interface"]["qualified_name"], "toy_kernel.Worker.forward_window")
            probe_rows = [
                json.loads(line)
                for line in (task_pack / "docs" / "target_call_probe.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(probe_rows[0]["positional_arg_count"], 0)
            self.assertEqual(probe_rows[0]["captured_positional_arg_count"], 0)
            self.assertEqual(probe_rows[0]["kwarg_count"], 2)
            self.assertIsNotNone(probe_rows[0]["forward_id"])
            self.assertIn("--disable-cuda-graph", probe["service_cmd"])

            capture = self._run_cli(
                "capture-snapshots",
                "--task-pack",
                str(task_pack),
                "--service-cmd",
                service_cmd + " --disable-cuda-graph --disable-cuda-graph",
                "--workload-cmd",
                workload_cmd,
                "--target-file",
                str(target_file),
                "--target-line",
                str(target_line),
                "--signature",
                "candidate(*args, **kwargs)",
                "--mutable-arg-path",
                "kwargs.state.total",
                "--mutable-arg-path",
                "kwargs.ssm_states",
                "--forward-boundary-file",
                str(target_file),
                "--forward-boundary-line",
                str(boundary_line),
                "--max-capture-groups",
                "8",
                "--max-samples-per-group",
                "4",
                "--max-samples-per-forward-per-group",
                "2",
                "--startup-timeout",
                "1",
            )
            self.assertEqual(capture["raw_group_count"], 1)
            self.assertEqual(capture["raw_sample_count"], 4)
            self.assertEqual(capture["total_hit_count"], 6)
            self.assertGreater(capture["mutation_warning_count"], 0)
            self.assertEqual(capture["service_cmd"].count("--disable-cuda-graph"), 1)
            self.assertNotIn("@__import__", target_file.read_text(encoding="utf-8"))

            selected = self._run_cli(
                "select-snapshots",
                "--task-pack",
                str(task_pack),
                "--max-groups",
                "1",
                "--max-selected-samples-per-group",
                "4",
            )
            self.assertEqual(selected["selected_group_count"], 1)
            self.assertEqual(selected["selected_sample_count"], 4)

            self._run_cli("generate-harness", "--task-pack", str(task_pack))

            validate = self._run_cli(
                "validate-task-pack",
                "--task-pack",
                str(task_pack),
                "--skip-env-check",
                "--run-correctness",
                "--run-benchmark",
                extra_env={
                    "DEVICE": "cpu",
                    "WARMUP": "1",
                    "REPEAT": "2",
                    "PYTHON": sys.executable,
                },
            )
            self.assertTrue(validate["valid"], validate)
            self.assertFalse(validate["errors"], validate)

            manifest = json.loads((task_pack / "snapshots" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_group_count"], 1)
            self.assertEqual(manifest["selected_sample_count"], 4)
            first_sample = manifest["case_groups"][0]["samples"][0]
            sample_meta_path = (
                task_pack
                / "snapshots"
                / "selected"
                / manifest["case_groups"][0]["group_id"]
                / "samples"
                / first_sample["sample_id"]
                / "meta.json"
            )
            sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
            self.assertEqual(sample_meta["mutation"]["mutable_arg_paths"], ["kwargs.state.total"])
            self.assertIn("kwargs.ssm_states", sample_meta["mutation"]["ignored_mutable_arg_paths"])

            shape_list = json.loads((task_pack / "shape_list.json").read_text(encoding="utf-8"))
            self.assertEqual(shape_list["source"], "snapshots/manifest.json")
            self.assertEqual(len(shape_list["shape_groups"]), 1)

            resolved = self._run_cli(
                "resolve-interface",
                "--file",
                str(target_file),
                "--line",
                str(target_line),
            )
            self.assertEqual(resolved["function_name"], "extend")
            self.assertEqual(resolved["target_name"], "toy_kernel.extend")

    def _run_cli(self, *args: str, extra_env: dict[str, str] | None = None) -> dict:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run(
            [sys.executable, "-m", "kernel_agent.framework_engineer.cli", *args],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, f"args={args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        self.assertTrue(lines, "CLI produced no stdout")
        return json.loads(lines[-1])

    @staticmethod
    def _write_target(path: Path) -> None:
        path.write_text(
            textwrap.dedent(
                """
                def extend(*, values, state):
                    state["total"] += sum(values)
                    return {"out": [v + 1 for v in values], "total": state["total"]}


                class Other:
                    def forward_window(self):
                        return "not the boundary"


                class Worker:
                    def forward_window(self, values):
                        extend(values=values, state={"total": 0})
                        extend(values=values, state={"total": 0})
                        extend(values=values, state={"total": 0})


                _WORKER = Worker()


                def run(values):
                    _WORKER.forward_window(values)
                """
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_workload(path: Path, module_dir: Path) -> None:
        path.write_text(
            textwrap.dedent(
                f"""
                import sys

                sys.path.insert(0, {str(module_dir)!r})

                import toy_kernel

                toy_kernel.run([1, 2, 3])
                toy_kernel.run([1, 2, 3])
                """
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _line_for(path: Path, needle: str) -> int:
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if needle in line:
                return idx
        raise AssertionError(f"missing line containing {needle!r}")


if __name__ == "__main__":
    unittest.main()
