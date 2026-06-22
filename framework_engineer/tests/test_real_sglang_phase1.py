from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _enabled() -> bool:
    return os.environ.get("KA_REAL_SGLANG") == "1"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise unittest.SkipTest(f"{name} is required when KA_REAL_SGLANG=1")
    return value


@unittest.skipUnless(_enabled(), "Set KA_REAL_SGLANG=1 to run real SGLang integration tests")
class RealSGLangPhase1CliTests(unittest.TestCase):
    """Run the Framework Engineer CLI against a real SGLang/workload target.

    This test is intentionally environment-driven. It does not know the user's
    SGLang checkout or model path; the caller provides commands and target
    source file through environment variables.
    """

    def test_real_sglang_snapshot_task_pack_flow(self) -> None:
        service_cmd = _required_env("KA_SERVICE_CMD")
        workload_cmd = _required_env("KA_WORKLOAD_CMD")
        target_file = _required_env("KA_TARGET_FILE")
        function_name = _required_env("KA_FUNCTION_NAME")
        target_name = _required_env("KA_TARGET_NAME")

        keep = os.environ.get("KA_KEEP_TASK_PACK") == "1"
        provided_task_pack = os.environ.get("KA_TASK_PACK")
        tmpdir: str | None = None
        if provided_task_pack:
            task_pack = Path(provided_task_pack).resolve()
        else:
            tmpdir = tempfile.mkdtemp(prefix="ka_real_sglang_")
            task_pack = Path(tmpdir) / "task_pack"

        print(f"[ka-real] task_pack={task_pack}")
        try:
            self._run_cli(
                "scaffold-task-pack",
                "--task-id",
                os.environ.get("KA_TASK_ID", "real_sglang_phase1"),
                "--out",
                str(task_pack),
                "--force",
            )

            if os.environ.get("KA_SKIP_BASELINE") != "1":
                self._run_cli(
                    "run-baseline",
                    "--task-pack",
                    str(task_pack),
                    "--service-cmd",
                    service_cmd,
                    "--workload-cmd",
                    workload_cmd,
                    *self._service_args(),
                )

            probe = self._run_cli(
                "probe-target-calls",
                "--task-pack",
                str(task_pack),
                "--service-cmd",
                self._non_cudagraph_service_cmd(service_cmd),
                "--workload-cmd",
                workload_cmd,
                "--target-file",
                target_file,
                "--function-name",
                function_name,
                "--target-name",
                target_name,
                *self._drop_first_arg(),
                *self._service_args(),
            )
            self.assertGreater(probe["call_count"], 0, probe)

            capture = self._run_cli(
                "capture-snapshots",
                "--task-pack",
                str(task_pack),
                "--service-cmd",
                self._non_cudagraph_service_cmd(service_cmd),
                "--workload-cmd",
                workload_cmd,
                "--target-file",
                target_file,
                "--function-name",
                function_name,
                "--target-name",
                target_name,
                "--signature",
                os.environ.get(
                    "KA_SIGNATURE",
                    "candidate_extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc)",
                ),
                "--mode",
                os.environ.get("KA_TARGET_MODE", ""),
                "--backend",
                os.environ.get("KA_TARGET_BACKEND", ""),
                "--layer-id",
                os.environ.get("KA_TARGET_LAYER_ID", ""),
                "--max-raw-cases",
                os.environ.get("KA_MAX_RAW_CASES", "32"),
                *self._mutable_arg_args(),
                *self._drop_first_arg(),
                *self._service_args(),
            )
            self.assertGreater(capture["raw_snapshot_count"], 0, capture)

            selected = self._run_cli(
                "select-snapshots",
                "--task-pack",
                str(task_pack),
                *self._max_cases_args(),
            )
            self.assertGreater(selected["selected_case_count"], 0, selected)

            self._run_cli("generate-harness", "--task-pack", str(task_pack))

            if os.environ.get("KA_RUN_PROBE_ENV") == "1":
                self._run_cli("probe-env", "--task-pack", str(task_pack))

            validate_args = [
                "validate-task-pack",
                "--task-pack",
                str(task_pack),
                "--run-correctness",
            ]
            if os.environ.get("KA_RUN_BENCHMARK") == "1":
                validate_args.append("--run-benchmark")
            if os.environ.get("KA_RUN_PROBE_ENV") != "1" or os.environ.get("KA_SKIP_ENV_CHECK") == "1":
                validate_args.append("--skip-env-check")

            validate = self._run_cli(
                *validate_args,
                extra_env={
                    "DEVICE": os.environ.get("KA_VALIDATE_DEVICE", "cuda"),
                    "WARMUP": os.environ.get("KA_VALIDATE_WARMUP", "3"),
                    "REPEAT": os.environ.get("KA_VALIDATE_REPEAT", "5"),
                    "PYTHON": sys.executable,
                },
            )
            self.assertTrue(validate["valid"], validate)

            manifest = json.loads((task_pack / "snapshots" / "manifest.json").read_text(encoding="utf-8"))
            self.assertGreater(manifest["selected_case_count"], 0)
        finally:
            if tmpdir and not keep:
                shutil.rmtree(tmpdir, ignore_errors=True)
            elif keep:
                print(f"[ka-real] kept task_pack={task_pack}")

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
            timeout=int(os.environ.get("KA_TEST_TIMEOUT", "3600")),
        )
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        self.assertEqual(proc.returncode, 0, f"args={args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        self.assertTrue(lines, f"no stdout for args={args}")
        return json.loads(lines[-1])

    def _service_args(self) -> list[str]:
        args: list[str] = [
            "--startup-timeout",
            os.environ.get("KA_STARTUP_TIMEOUT", "240"),
            "--workload-timeout",
            os.environ.get("KA_WORKLOAD_TIMEOUT", "1200"),
        ]
        if os.environ.get("KA_HEALTH_URL"):
            args.extend(["--health-url", os.environ["KA_HEALTH_URL"]])
        return args

    def _drop_first_arg(self) -> list[str]:
        if os.environ.get("KA_DROP_FIRST_ARG", "1") == "1":
            return ["--drop-first-arg"]
        return []

    def _mutable_arg_args(self) -> list[str]:
        paths = [p.strip() for p in os.environ.get("KA_MUTABLE_ARG_PATHS", "kwargs.ssm_states").split(",") if p.strip()]
        args: list[str] = []
        for path in paths:
            args.extend(["--mutable-arg-path", path])
        return args

    def _max_cases_args(self) -> list[str]:
        if os.environ.get("KA_MAX_SELECTED_CASES"):
            return ["--max-cases", os.environ["KA_MAX_SELECTED_CASES"]]
        return []

    def _non_cudagraph_service_cmd(self, service_cmd: str) -> str:
        if os.environ.get("KA_NON_CUDAGRAPH_SERVICE_CMD"):
            return os.environ["KA_NON_CUDAGRAPH_SERVICE_CMD"]
        if "--disable-cuda-graph" in service_cmd:
            return service_cmd
        return service_cmd + " --disable-cuda-graph"


if __name__ == "__main__":
    unittest.main()

