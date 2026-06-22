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
    """Exercise the real Framework Engineer CLI on a tiny target function.

    This test intentionally uses subprocesses and temporary source rewriting so
    it covers the same path used for SGLang target probing/capture, without
    requiring a real SGLang or GPU environment.
    """

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

            self._run_cli(
                "scaffold-task-pack",
                "--task-id",
                "toy_gdn_extend",
                "--out",
                str(task_pack),
            )

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
            self.assertTrue((task_pack / "docs" / "baseline_run_report.md").exists())

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
                "--function-name",
                "extend",
                "--target-name",
                "toy_kernel.extend",
                "--startup-timeout",
                "1",
            )
            self.assertEqual(probe["call_count"], 3)

            capture = self._run_cli(
                "capture-snapshots",
                "--task-pack",
                str(task_pack),
                "--service-cmd",
                service_cmd,
                "--workload-cmd",
                workload_cmd,
                "--target-file",
                str(target_file),
                "--function-name",
                "extend",
                "--target-name",
                "toy_kernel.extend",
                "--mutable-arg-path",
                "kwargs.ssm_states",
                "--startup-timeout",
                "1",
            )
            self.assertEqual(capture["raw_snapshot_count"], 3)
            self.assertNotIn("@__import__", target_file.read_text(encoding="utf-8"))

            selected = self._run_cli("select-snapshots", "--task-pack", str(task_pack))
            self.assertEqual(selected["selected_case_count"], 2)
            self.assertTrue((task_pack / "snapshots" / "manifest.json").exists())

            self._run_cli("generate-harness", "--task-pack", str(task_pack))
            self.assertTrue((task_pack / "snapshot_runtime.py").exists())
            self.assertTrue((task_pack / "correctness_test.py").exists())
            self.assertTrue((task_pack / "benchmark.py").exists())

            validate = self._run_cli(
                "validate-task-pack",
                "--task-pack",
                str(task_pack),
                "--skip-env-check",
                "--run-correctness",
                "--run-benchmark",
                extra_env={
                    "DEVICE": os.environ.get("KA_TEST_DEVICE", "cpu"),
                    "WARMUP": "1",
                    "REPEAT": "2",
                    "PYTHON": sys.executable,
                },
            )
            self.assertTrue(validate["valid"], validate)
            self.assertFalse(validate["errors"], validate)

            manifest = json.loads((task_pack / "snapshots" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["raw_case_count"], 3)
            self.assertEqual(manifest["selected_case_count"], 2)

            shape_list = json.loads((task_pack / "shape_list.json").read_text(encoding="utf-8"))
            self.assertEqual(shape_list["source"], "snapshots/manifest.json")
            self.assertEqual(len(shape_list["shape_cases"]), 2)

    def _run_cli(self, *args: str, extra_env: dict[str, str] | None = None) -> dict:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
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
        self.assertEqual(proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        self.assertTrue(lines, "CLI produced no stdout")
        return json.loads(lines[-1])

    @staticmethod
    def _write_target(path: Path) -> None:
        path.write_text(
            textwrap.dedent(
                """
                import torch


                def extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc):
                    # Mutate state to exercise pre/post snapshot correctness.
                    update = q.sum(dim=-1, keepdim=True).to(ssm_states.dtype)
                    ssm_states.copy_(ssm_states + update)
                    return q + k + v + g * beta
                """
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_workload(path: Path, module_dir: Path) -> None:
        path.write_text(
            textwrap.dedent(
                f"""
                import os
                import sys
                from pathlib import Path

                sys.path.insert(0, {str(module_dir)!r})

                import torch
                import toy_kernel


                device = os.environ.get("KA_TEST_DEVICE", "cpu")
                if device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("KA_TEST_DEVICE=cuda but torch.cuda.is_available() is false")


                def run_case(query_start_loc):
                    q = torch.arange(6, dtype=torch.float32, device=device).reshape(3, 2)
                    k = q + 1
                    v = q + 2
                    g = torch.full_like(q, 0.5)
                    beta = torch.full_like(q, 2.0)
                    ssm_states = torch.zeros(3, 1, dtype=torch.float32, device=device)
                    cache_indices = torch.arange(3, dtype=torch.int64, device=device)
                    query_start_loc = torch.tensor(query_start_loc, dtype=torch.int64, device=device)
                    out = toy_kernel.extend(
                        q, k, v, g, beta,
                        ssm_states=ssm_states,
                        cache_indices=cache_indices,
                        query_start_loc=query_start_loc,
                    )
                    torch.cuda.synchronize() if device == "cuda" else None
                    print(float(out.sum().detach().cpu()), float(ssm_states.sum().detach().cpu()))


                run_case([0, 1, 3, 3])
                run_case([0, 1, 3, 3])
                run_case([0, 2, 3, 3])
                """
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()

