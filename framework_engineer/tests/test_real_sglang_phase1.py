from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class RealSGLangConfig:
    service_cmd: str
    workload_cmd: str
    target_file: str
    function_name: str
    target_name: str

    task_id: str = "real_sglang_phase1"
    task_pack: str | None = None
    keep_task_pack: bool = True
    skip_baseline: bool = False

    non_cudagraph_service_cmd: str | None = None
    health_url: str | None = None
    startup_timeout: int = 240
    workload_timeout: int = 1200
    test_timeout: int = 3600

    signature: str = "candidate_extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc)"
    target_mode: str = ""
    target_backend: str = ""
    target_layer_id: str = ""
    drop_first_arg: bool = True
    mutable_arg_paths: list[str] = field(default_factory=lambda: ["kwargs.ssm_states"])

    max_raw_cases: int = 32
    max_selected_cases: int | None = 8

    run_probe_env: bool = False
    skip_env_check: bool = True
    run_benchmark: bool = False
    validate_device: str = "cuda"
    validate_warmup: int = 3
    validate_repeat: int = 5

    extra_env: dict[str, str] = field(default_factory=dict)


def _load_config() -> RealSGLangConfig:
    config_path = os.environ.get("KA_REAL_SGLANG_CONFIG")
    if not config_path:
        raise unittest.SkipTest("Set KA_REAL_SGLANG_CONFIG=/path/to/real_sglang_phase1_config.py")
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise unittest.SkipTest(f"KA_REAL_SGLANG_CONFIG does not exist: {path}")

    spec = importlib.util.spec_from_file_location("ka_real_sglang_config", path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"Cannot import config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "CONFIG"):
        raw = getattr(module, "CONFIG")
        if not isinstance(raw, dict):
            raise TypeError("CONFIG must be a dict when provided")
    else:
        raw = {
            key: getattr(module, key)
            for key in dir(module)
            if key.islower() and not key.startswith("_")
        }

    required = ["service_cmd", "workload_cmd", "target_file", "function_name", "target_name"]
    missing = [name for name in required if not raw.get(name)]
    if missing:
        raise unittest.SkipTest(f"Missing required config fields: {', '.join(missing)}")
    return RealSGLangConfig(**raw)


class RealSGLangPhase1CliTests(unittest.TestCase):
    """Run the Framework Engineer CLI against a real SGLang/workload target."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = _load_config()

    def test_real_sglang_snapshot_task_pack_flow(self) -> None:
        cfg: RealSGLangConfig = self.cfg

        tmpdir: str | None = None
        if cfg.task_pack:
            task_pack = Path(cfg.task_pack).expanduser().resolve()
        else:
            tmpdir = tempfile.mkdtemp(prefix="ka_real_sglang_")
            task_pack = Path(tmpdir) / "task_pack"

        print(f"[ka-real] task_pack={task_pack}")
        try:
            self._run_cli(
                "scaffold-task-pack",
                "--task-id",
                cfg.task_id,
                "--out",
                str(task_pack),
                "--force",
            )

            if not cfg.skip_baseline:
                self._run_cli(
                    "run-baseline",
                    "--task-pack",
                    str(task_pack),
                    "--service-cmd",
                    cfg.service_cmd,
                    "--workload-cmd",
                    cfg.workload_cmd,
                    *self._service_args(),
                )

            probe = self._run_cli(
                "probe-target-calls",
                "--task-pack",
                str(task_pack),
                "--service-cmd",
                self._non_cudagraph_service_cmd(cfg.service_cmd),
                "--workload-cmd",
                cfg.workload_cmd,
                "--target-file",
                cfg.target_file,
                "--function-name",
                cfg.function_name,
                "--target-name",
                cfg.target_name,
                *self._drop_first_arg(),
                *self._service_args(),
            )
            self.assertGreater(probe["call_count"], 0, probe)

            capture = self._run_cli(
                "capture-snapshots",
                "--task-pack",
                str(task_pack),
                "--service-cmd",
                self._non_cudagraph_service_cmd(cfg.service_cmd),
                "--workload-cmd",
                cfg.workload_cmd,
                "--target-file",
                cfg.target_file,
                "--function-name",
                cfg.function_name,
                "--target-name",
                cfg.target_name,
                "--signature",
                cfg.signature,
                "--mode",
                cfg.target_mode,
                "--backend",
                cfg.target_backend,
                "--layer-id",
                cfg.target_layer_id,
                "--max-raw-cases",
                str(cfg.max_raw_cases),
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

            if cfg.run_probe_env:
                self._run_cli("probe-env", "--task-pack", str(task_pack))

            validate_args = [
                "validate-task-pack",
                "--task-pack",
                str(task_pack),
                "--run-correctness",
            ]
            if cfg.run_benchmark:
                validate_args.append("--run-benchmark")
            if not cfg.run_probe_env or cfg.skip_env_check:
                validate_args.append("--skip-env-check")

            validate = self._run_cli(
                *validate_args,
                extra_env={
                    "DEVICE": cfg.validate_device,
                    "WARMUP": str(cfg.validate_warmup),
                    "REPEAT": str(cfg.validate_repeat),
                    "PYTHON": sys.executable,
                },
            )
            self.assertTrue(validate["valid"], validate)

            manifest = json.loads((task_pack / "snapshots" / "manifest.json").read_text(encoding="utf-8"))
            self.assertGreater(manifest["selected_case_count"], 0)
        finally:
            if tmpdir and not cfg.keep_task_pack:
                shutil.rmtree(tmpdir, ignore_errors=True)
            elif cfg.keep_task_pack:
                print(f"[ka-real] kept task_pack={task_pack}")

    def _run_cli(self, *args: str, extra_env: dict[str, str] | None = None) -> dict:
        cfg: RealSGLangConfig = self.cfg
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env.update(cfg.extra_env)
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
            timeout=cfg.test_timeout,
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
        cfg: RealSGLangConfig = self.cfg
        args: list[str] = [
            "--startup-timeout",
            str(cfg.startup_timeout),
            "--workload-timeout",
            str(cfg.workload_timeout),
        ]
        if cfg.health_url:
            args.extend(["--health-url", cfg.health_url])
        return args

    def _drop_first_arg(self) -> list[str]:
        return ["--drop-first-arg"] if self.cfg.drop_first_arg else []

    def _mutable_arg_args(self) -> list[str]:
        args: list[str] = []
        for path in self.cfg.mutable_arg_paths:
            args.extend(["--mutable-arg-path", path])
        return args

    def _max_cases_args(self) -> list[str]:
        if self.cfg.max_selected_cases is not None:
            return ["--max-cases", str(self.cfg.max_selected_cases)]
        return []

    def _non_cudagraph_service_cmd(self, service_cmd: str) -> str:
        if self.cfg.non_cudagraph_service_cmd:
            return self.cfg.non_cudagraph_service_cmd
        if "--disable-cuda-graph" in service_cmd:
            return service_cmd
        return service_cmd + " --disable-cuda-graph"


if __name__ == "__main__":
    unittest.main()

