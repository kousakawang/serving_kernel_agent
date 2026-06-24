"""Framework Engineer Phase 1 CLI.

Run as:

    python -m kernel_agent.framework_engineer.cli <subcommand>
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .snapshot.harness_builder import SnapshotHarnessBuilder, copy_probe_templates
from .snapshot.selector import SnapshotSelector, write_shape_list_summary
from .snapshot.store import SnapshotStore
from .snapshot.validation import run_smoke, validate_files


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "framework_engineer" / "templates"


@dataclass(frozen=True)
class SourceInterface:
    file: Path
    function_name: str
    qualified_name: str
    line: int
    end_line: int | None
    class_path: list[str]
    module_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": str(self.file),
            "function_name": self.function_name,
            "qualified_name": self.qualified_name,
            "line": self.line,
            "end_line": self.end_line,
            "class_path": self.class_path,
            "module_name": self.module_name,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="framework-engineer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scaffold-task-pack")
    p.add_argument("--task-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_scaffold_task_pack)

    p = sub.add_parser("run-baseline")
    p.add_argument("--task-pack", type=Path, required=True)
    p.add_argument("--service-cmd", required=True)
    p.add_argument("--workload-cmd", required=True)
    p.add_argument("--health-url", default=None)
    p.add_argument("--startup-timeout", type=int, default=120)
    p.add_argument("--workload-timeout", type=int, default=600)
    p.set_defaults(func=cmd_run_baseline)

    p = sub.add_parser("resolve-interface")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--line", type=int, required=True)
    p.set_defaults(func=cmd_resolve_interface)

    p = sub.add_parser("probe-target-calls")
    _add_run_and_instrument_args(p)
    p.set_defaults(func=cmd_probe_target_calls)

    p = sub.add_parser("capture-snapshots")
    _add_run_and_instrument_args(p)
    p.add_argument("--signature", default="candidate(*args, **kwargs)")
    p.add_argument("--mutable-arg-path", action="append", default=[])
    p.add_argument("--mode", default="")
    p.add_argument("--backend", default="")
    p.add_argument("--layer-id", default="")
    p.add_argument("--calls-per-forward", type=int, default=None)
    p.add_argument("--max-capture-groups", type=int, default=64)
    p.add_argument("--max-samples-per-group", type=int, default=8)
    p.add_argument("--max-samples-per-forward-per-group", type=int, default=3)
    p.add_argument("--max-raw-cases", type=int, default=None, help="Deprecated alias for --max-capture-groups.")
    p.set_defaults(func=cmd_capture_snapshots)

    p = sub.add_parser("select-snapshots")
    p.add_argument("--task-pack", type=Path, required=True)
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--max-selected-samples-per-group", type=int, default=8)
    p.add_argument("--max-cases", type=int, default=None, help="Deprecated alias for --max-groups.")
    p.set_defaults(func=cmd_select_snapshots)

    p = sub.add_parser("generate-harness")
    p.add_argument("--task-pack", type=Path, required=True)
    p.add_argument("--candidate-function", default="candidate")
    p.set_defaults(func=cmd_generate_harness)

    p = sub.add_parser("probe-env")
    p.add_argument("--task-pack", type=Path, required=True)
    p.set_defaults(func=cmd_probe_env)

    p = sub.add_parser("validate-task-pack")
    p.add_argument("--task-pack", type=Path, required=True)
    p.add_argument("--run-correctness", action="store_true")
    p.add_argument("--run-benchmark", action="store_true")
    p.add_argument("--skip-env-check", action="store_true")
    p.add_argument("--timeout", type=int, default=300)
    p.set_defaults(func=cmd_validate_task_pack)

    args = parser.parse_args(argv)
    return int(args.func(args))


def _add_run_and_instrument_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-pack", type=Path, required=True)
    parser.add_argument("--service-cmd", required=True)
    parser.add_argument("--non-cudagraph-service-cmd", default=None)
    parser.add_argument("--workload-cmd", required=True)
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--target-line", type=int, default=None)
    parser.add_argument("--function-name", default=None)
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--drop-first-arg", action="store_true")
    parser.add_argument("--forward-boundary-file", type=Path, default=None)
    parser.add_argument("--forward-boundary-line", type=int, default=None)
    parser.add_argument("--forward-boundary-function", default=None)
    parser.add_argument("--forward-boundary-name", default=None)
    parser.add_argument("--health-url", default=None)
    parser.add_argument("--startup-timeout", type=int, default=120)
    parser.add_argument("--workload-timeout", type=int, default=600)


def cmd_scaffold_task_pack(args: argparse.Namespace) -> int:
    out: Path = args.out
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} already exists and is not empty; pass --force to overwrite scaffold files.")
    out.mkdir(parents=True, exist_ok=True)
    for rel in ("docs", "scripts", "snapshots/raw", "snapshots/selected", "env_probe", "kernel_sources"):
        (out / rel).mkdir(parents=True, exist_ok=True)

    _copy_template("task_pack_README.md", out / "README.md")
    _copy_template("task_pack_manifest.yaml", out / "task.yaml")
    _copy_template("env_manifest.yaml", out / "env_manifest.yaml")
    (out / "snapshots" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "phase1.snapshot.v1",
                "selection_policy": "not_selected_yet",
                "raw_group_count": 0,
                "raw_sample_count": 0,
                "selected_group_count": 0,
                "selected_sample_count": 0,
                "case_groups": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out / "shape_list.json").write_text(
        json.dumps(
            {
                "schema_version": "phase1.shape_summary.v1",
                "source": "snapshots/manifest.json",
                "note": "Selected snapshot samples are the replay source; this file is only a group index/summary.",
                "shape_groups": [],
                "shape_cases": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out / "snapshots" / "raw_index.json").write_text(
        json.dumps(
            {
                "schema_version": "phase1.snapshot.v1",
                "index_type": "raw_group_index",
                "raw_group_count": 0,
                "raw_sample_count": 0,
                "total_hit_count": 0,
                "groups": {},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    copy_probe_templates(TEMPLATE_DIR, out)
    _write_json(out / "docs" / "scaffold_result.json", {"task_id": args.task_id, "task_pack": str(out)})
    print(json.dumps({"status": "ok", "task_pack": str(out)}, sort_keys=True))
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    result = _run_service_and_workload(
        service_cmd=args.service_cmd,
        workload_cmd=args.workload_cmd,
        health_url=args.health_url,
        startup_timeout=args.startup_timeout,
        workload_timeout=args.workload_timeout,
    )
    docs = args.task_pack / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    _write_json(docs / "baseline_result.json", result)
    _write_baseline_report(docs / "baseline_run_report.md", args, result)
    print(json.dumps({"status": "ok" if result["workload"]["returncode"] == 0 else "failed", "report": str(docs / "baseline_run_report.md")}, sort_keys=True))
    return 0 if result["workload"]["returncode"] == 0 else 1


def cmd_resolve_interface(args: argparse.Namespace) -> int:
    interface = _resolve_source_interface(
        file=args.file,
        line=args.line,
        function_name=None,
        qualified_name=None,
        role="interface",
    )
    payload = {
        **interface.to_dict(),
        "target_file": str(interface.file),
        "function_name": interface.function_name,
        "target_name": interface.qualified_name,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def cmd_probe_target_calls(args: argparse.Namespace) -> int:
    docs = args.task_pack / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    log_path = docs / "target_call_probe.jsonl"
    target = _resolve_target_interface(args)
    boundary = _resolve_forward_boundary_interface(args)
    decorator_expr = (
        "__import__('kernel_agent.framework_engineer.snapshot.recorder', "
        "fromlist=['make_probe_decorator']).make_probe_decorator("
        f"{str(log_path)!r}, {target.qualified_name!r}, drop_first_arg={bool(args.drop_first_arg)!r})"
    )
    service_cmd = _resolve_non_cudagraph_service_cmd(args.service_cmd, args.non_cudagraph_service_cmd)
    with _instrumentation_context(target, boundary, decorator_expr):
        result = _run_service_and_workload(
            service_cmd=service_cmd,
            workload_cmd=args.workload_cmd,
            health_url=args.health_url,
            startup_timeout=args.startup_timeout,
            workload_timeout=args.workload_timeout,
        )
    calls = _read_jsonl(log_path)
    report = {
        "target_name": target.qualified_name,
        "target_interface": target.to_dict(),
        "forward_boundary_interface": boundary.to_dict() if boundary else None,
        "call_count": len(calls),
        "workload_returncode": result["workload"]["returncode"],
        "log_path": str(log_path),
        "service_cmd": service_cmd,
        "workload_cmd": args.workload_cmd,
    }
    _write_json(docs / "target_call_probe_report.json", report)
    (docs / "target_call_probe_report.md").write_text(
        f"# Target Call Probe Report\n\n- target: `{args.target_name}`\n- call_count: {len(calls)}\n- workload_returncode: {result['workload']['returncode']}\n- log: `{log_path}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if result["workload"]["returncode"] == 0 and calls else 1


def cmd_capture_snapshots(args: argparse.Namespace) -> int:
    snapshot_root = args.task_pack / "snapshots"
    mutable_paths = ",".join(args.mutable_arg_path)
    max_capture_groups = args.max_raw_cases if args.max_raw_cases is not None else args.max_capture_groups
    target = _resolve_target_interface(args)
    boundary = _resolve_forward_boundary_interface(args)
    decorator_expr = (
        "__import__('kernel_agent.framework_engineer.snapshot.recorder', "
        "fromlist=['make_snapshot_decorator']).make_snapshot_decorator("
        f"{str(snapshot_root)!r}, {args.task_pack.name!r}, {target.qualified_name!r}, {args.signature!r}, "
        f"mutable_arg_paths={mutable_paths!r}, mode={args.mode!r}, backend={args.backend!r}, "
        f"layer_id={args.layer_id!r}, drop_first_arg={bool(args.drop_first_arg)!r}, "
        f"source_info={target.to_dict()!r}, "
        f"calls_per_forward={args.calls_per_forward!r}, max_capture_groups={max_capture_groups!r}, "
        f"max_samples_per_group={args.max_samples_per_group!r}, "
        f"max_samples_per_forward_per_group={args.max_samples_per_forward_per_group!r})"
    )
    service_cmd = _resolve_non_cudagraph_service_cmd(args.service_cmd, args.non_cudagraph_service_cmd)
    with _instrumentation_context(target, boundary, decorator_expr):
        result = _run_service_and_workload(
            service_cmd=service_cmd,
            workload_cmd=args.workload_cmd,
            health_url=args.health_url,
            startup_timeout=args.startup_timeout,
            workload_timeout=args.workload_timeout,
        )
    raw_index = SnapshotStore(snapshot_root).read_raw_index()
    report = {
        "target_name": target.qualified_name,
        "target_interface": target.to_dict(),
        "forward_boundary_interface": boundary.to_dict() if boundary else None,
        "windowing_mode": _windowing_mode(args),
        "raw_group_count": raw_index.get("raw_group_count", 0),
        "raw_sample_count": raw_index.get("raw_sample_count", 0),
        "raw_snapshot_count": raw_index.get("raw_sample_count", 0),
        "total_hit_count": raw_index.get("total_hit_count", 0),
        "dropped_hit_count": raw_index.get("dropped_hit_count", 0),
        "mutation_warning_count": _mutation_warning_count(raw_index),
        "workload_returncode": result["workload"]["returncode"],
        "service_cmd": service_cmd,
        "workload_cmd": args.workload_cmd,
        "max_raw_cases_deprecated_alias_used": args.max_raw_cases is not None,
        "workload_stdout_tail": result["workload"].get("stdout", "")[-2000:],
        "workload_stderr_tail": result["workload"].get("stderr", "")[-2000:],
    }
    docs = args.task_pack / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    _write_json(docs / "snapshot_capture_report.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if result["workload"]["returncode"] == 0 and raw_index.get("raw_sample_count", 0) else 1


def cmd_select_snapshots(args: argparse.Namespace) -> int:
    store = SnapshotStore(args.task_pack / "snapshots")
    max_groups = args.max_groups if args.max_groups is not None else args.max_cases
    manifest = SnapshotSelector(store).select(
        max_groups=max_groups,
        max_samples_per_group=args.max_selected_samples_per_group,
    )
    write_shape_list_summary(args.task_pack, manifest)
    docs = args.task_pack / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    _write_json(docs / "snapshot_selection_report.json", manifest)
    (docs / "snapshot_selection_report.md").write_text(
        "# Snapshot Selection Report\n\n"
        f"- raw_group_count: {manifest['raw_group_count']}\n"
        f"- raw_sample_count: {manifest['raw_sample_count']}\n"
        f"- selected_group_count: {manifest['selected_group_count']}\n"
        f"- selected_sample_count: {manifest['selected_sample_count']}\n"
        f"- policy: `{manifest['selection_policy']}`\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_group_count": manifest["selected_group_count"],
                "selected_sample_count": manifest["selected_sample_count"],
                "selected_case_count": manifest["selected_group_count"],
                "manifest": str(store.manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


def cmd_generate_harness(args: argparse.Namespace) -> int:
    SnapshotHarnessBuilder(args.task_pack).generate(candidate_function=args.candidate_function)
    print(json.dumps({"status": "ok", "task_pack": str(args.task_pack)}, sort_keys=True))
    return 0


def cmd_probe_env(args: argparse.Namespace) -> int:
    result = probe_environment(args.task_pack)
    (args.task_pack / "env_manifest.yaml").write_text(_env_to_yaml(result), encoding="utf-8")
    _write_json(args.task_pack / "docs" / "env_probe_result.json", result)
    print(json.dumps({"status": "ok", "env_manifest": str(args.task_pack / "env_manifest.yaml")}, sort_keys=True))
    return 0


def cmd_validate_task_pack(args: argparse.Namespace) -> int:
    errors = validate_files(args.task_pack)
    env_check: dict[str, Any] | None = None
    if not args.skip_env_check:
        expected_path = args.task_pack / "docs" / "env_probe_result.json"
        if not expected_path.exists():
            errors.append("missing docs/env_probe_result.json; run probe-env before validate-task-pack or pass --skip-env-check")
        else:
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            current = probe_environment(args.task_pack)
            mismatches = _compare_availability(expected, current)
            env_check = {"mismatches": mismatches}
            errors.extend(f"env availability mismatch: {item}" for item in mismatches)
    smoke = run_smoke(
        args.task_pack,
        correctness=args.run_correctness,
        benchmark=args.run_benchmark,
        timeout=args.timeout,
    )
    for item in smoke:
        if item["returncode"] != 0:
            errors.append(f"command failed: {item['command']}")
    report = {"valid": not errors, "errors": errors, "env_check": env_check, "smoke": smoke}
    docs = args.task_pack / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    _write_json(docs / "task_pack_validation_report.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if not errors else 1


def probe_environment(task_pack: Path) -> dict[str, Any]:
    env_probe = task_pack / "env_probe"
    env_probe.mkdir(parents=True, exist_ok=True)
    copy_probe_templates(TEMPLATE_DIR, task_pack)
    return {
        "task_id": task_pack.name,
        "generated_by": "framework_engineer.cli probe-env",
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "pytorch": _run_python_probe("import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"),
        "gpu": _run_python_probe(
            "import torch\n"
            "print(torch.cuda.device_count())\n"
            "p=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None\n"
            "print('' if p is None else p.name)\n"
            "print('' if p is None else f'{p.major}.{p.minor}')\n"
            "print('' if p is None else p.total_memory)"
        ),
        "triton": _run_command([sys.executable, "env_probe/probe_triton.py"], cwd=task_pack),
        "cutedsl": _run_command([sys.executable, "env_probe/probe_cutedsl.py"], cwd=task_pack),
        "cuda_extension": _run_command([sys.executable, "env_probe/probe_cuda_extension.py"], cwd=task_pack),
        "ncu": _run_command(["bash", "env_probe/probe_ncu.sh"], cwd=task_pack),
        "dependency_policy": {
            "kernel_engineer_may_install_packages": False,
            "allowed_paths_only": True,
        },
    }


def _copy_template(name: str, dst: Path) -> None:
    src = TEMPLATE_DIR / name
    if src.exists():
        shutil.copy2(src, dst)


def _run_python_probe(code: str) -> dict[str, Any]:
    return _run_command([sys.executable, "-c", code], cwd=Path.cwd())


def _run_command(cmd: list[str], *, cwd: Path, timeout: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        return {
            "available": proc.returncode == 0,
            "returncode": proc.returncode,
            "command": " ".join(cmd),
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except FileNotFoundError as exc:
        return {"available": False, "returncode": 127, "command": " ".join(cmd), "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"available": False, "returncode": 124, "command": " ".join(cmd), "stdout": exc.stdout or "", "stderr": exc.stderr or "timeout"}


def _run_service_and_workload(
    *,
    service_cmd: str,
    workload_cmd: str,
    health_url: str | None,
    startup_timeout: int,
    workload_timeout: int,
) -> dict[str, Any]:
    started = time.time()
    env = _subprocess_env()
    service = subprocess.Popen(
        service_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    try:
        health = _wait_for_service(service, health_url=health_url, timeout=startup_timeout)
        workload_start = time.time()
        workload = subprocess.run(
            workload_cmd,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=workload_timeout,
            check=False,
        )
        workload_elapsed = time.time() - workload_start
        return {
            "service_cmd": service_cmd,
            "workload_cmd": workload_cmd,
            "health": health,
            "startup_elapsed_sec": time.time() - started,
            "workload": {
                "returncode": workload.returncode,
                "elapsed_sec": workload_elapsed,
                "stdout": workload.stdout[-8000:],
                "stderr": workload.stderr[-8000:],
            },
        }
    finally:
        _terminate_process(service)


def _wait_for_service(proc: subprocess.Popen, *, health_url: str | None, timeout: int) -> dict[str, Any]:
    deadline = time.time() + timeout
    if health_url is None:
        time.sleep(min(10, timeout))
        return {"mode": "sleep", "ready": proc.poll() is None}
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            return {"mode": "http", "ready": False, "error": f"service exited with {proc.returncode}"}
        try:
            with urlopen(health_url, timeout=5) as response:
                return {"mode": "http", "ready": 200 <= response.status < 500, "status": response.status}
        except Exception as exc:
            last_error = str(exc)
            time.sleep(2)
    return {"mode": "http", "ready": False, "error": last_error or "timeout"}


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    project_root = str(ROOT.parent)
    current = env.get("PYTHONPATH", "")
    parts = [project_root]
    if current:
        parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _resolve_target_interface(args: argparse.Namespace) -> SourceInterface:
    return _resolve_source_interface(
        file=args.target_file,
        line=args.target_line,
        function_name=args.function_name,
        qualified_name=args.target_name,
        role="target",
    )


def _resolve_forward_boundary_interface(args: argparse.Namespace) -> SourceInterface | None:
    if args.forward_boundary_file is None:
        return None
    return _resolve_source_interface(
        file=args.forward_boundary_file,
        line=args.forward_boundary_line,
        function_name=args.forward_boundary_function,
        qualified_name=args.forward_boundary_name,
        role="forward boundary",
    )


def _resolve_source_interface(
    *,
    file: Path,
    line: int | None,
    function_name: str | None,
    qualified_name: str | None,
    role: str,
) -> SourceInterface:
    file = file.expanduser().resolve()
    if not file.exists():
        raise SystemExit(f"{role}: file does not exist: {file}")
    source = file.read_text(encoding="utf-8")
    module_name = _infer_module_name(file)
    if line is not None:
        resolved = _resolve_interface_by_line(file, source, line, module_name)
        if function_name is not None and function_name != resolved.function_name:
            raise SystemExit(
                f"{role}: --function-name {function_name!r} does not match function at line {line}: "
                f"{resolved.function_name!r}"
            )
        if qualified_name is not None:
            return SourceInterface(
                file=resolved.file,
                function_name=resolved.function_name,
                qualified_name=qualified_name,
                line=resolved.line,
                end_line=resolved.end_line,
                class_path=resolved.class_path,
                module_name=resolved.module_name,
            )
        return resolved
    if function_name is None:
        raise SystemExit(f"{role}: provide either --{_role_prefix(role)}-line or --function-name")
    resolved = _resolve_interface_by_function_name(file, source, function_name, module_name)
    if qualified_name is not None:
        return SourceInterface(
            file=resolved.file,
            function_name=resolved.function_name,
            qualified_name=qualified_name,
            line=resolved.line,
            end_line=resolved.end_line,
            class_path=resolved.class_path,
            module_name=resolved.module_name,
        )
    return resolved


def _role_prefix(role: str) -> str:
    return "forward-boundary" if role == "forward boundary" else "target"


def _resolve_interface_by_line(file: Path, source: str, line: int, module_name: str) -> SourceInterface:
    candidates = _function_candidates(file, source, module_name)
    matching = [
        candidate
        for candidate in candidates
        if candidate.line <= line <= (candidate.end_line or candidate.line)
    ]
    if not matching:
        raise SystemExit(f"No function definition in {file} contains line {line}")
    return min(matching, key=lambda item: ((item.end_line or item.line) - item.line, -len(item.class_path)))


def _resolve_interface_by_function_name(file: Path, source: str, function_name: str, module_name: str) -> SourceInterface:
    matches = [candidate for candidate in _function_candidates(file, source, module_name) if candidate.function_name == function_name]
    if not matches:
        raise SystemExit(f"Could not find function definition {function_name!r} in {file}")
    return sorted(matches, key=lambda item: item.line)[0]


def _function_candidates(file: Path, source: str, module_name: str) -> list[SourceInterface]:
    tree = ast.parse(source, filename=str(file))
    out: list[SourceInterface] = []

    def visit(node: ast.AST, class_path: list[str]) -> None:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                visit(child, [*class_path, node.name])
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts = [module_name, *class_path, node.name]
            out.append(
                SourceInterface(
                    file=file,
                    function_name=node.name,
                    qualified_name=".".join(part for part in parts if part),
                    line=int(node.lineno),
                    end_line=getattr(node, "end_lineno", None),
                    class_path=list(class_path),
                    module_name=module_name,
                )
            )
            for child in node.body:
                visit(child, class_path)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, class_path)

    visit(tree, [])
    return out


def _infer_module_name(file: Path) -> str:
    parts = list(file.with_suffix("").parts)
    if "python" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("python")
        module_parts = parts[idx + 1 :]
        if module_parts:
            return ".".join(module_parts)
    return file.stem


def _instrumentation_context(
    target: SourceInterface,
    boundary: SourceInterface | None,
    target_decorator_expr: str,
):
    stack = contextlib.ExitStack()
    entries: list[tuple[SourceInterface, str]] = []
    if boundary is not None:
        boundary_name = boundary.qualified_name
        boundary_expr = (
            "__import__('kernel_agent.framework_engineer.snapshot.recorder', "
            "fromlist=['make_forward_boundary_decorator']).make_forward_boundary_decorator("
            f"{boundary_name!r})"
        )
        entries.append((boundary, boundary_expr))
    entries.append((target, target_decorator_expr))

    by_file: dict[Path, list[tuple[SourceInterface, str]]] = {}
    for interface, expr in entries:
        by_file.setdefault(interface.file, []).append((interface, expr))
    for file, file_entries in by_file.items():
        if len(file_entries) == 1:
            interface, expr = file_entries[0]
            stack.enter_context(_temporary_decorator(file, interface.function_name, expr, line=interface.line))
        else:
            stack.enter_context(_temporary_decorators(file, file_entries))
    return stack


def _resolve_non_cudagraph_service_cmd(service_cmd: str, explicit_cmd: str | None) -> str:
    if explicit_cmd:
        return _dedupe_disable_cuda_graph(explicit_cmd)
    cmd = _dedupe_disable_cuda_graph(service_cmd)
    if "--disable-cuda-graph" not in cmd.split():
        cmd = cmd.rstrip() + " --disable-cuda-graph"
    return cmd


def _dedupe_disable_cuda_graph(cmd: str) -> str:
    parts = cmd.split()
    seen = False
    changed = False
    out = []
    for part in parts:
        if part == "--disable-cuda-graph":
            if seen:
                changed = True
                continue
            seen = True
        out.append(part)
    return " ".join(out) if changed else cmd


def _windowing_mode(args: argparse.Namespace) -> str:
    if args.forward_boundary_file and (args.forward_boundary_function or args.forward_boundary_line):
        return "forward_boundary"
    if getattr(args, "calls_per_forward", None):
        return "calls_per_forward"
    return "unknown_forward"


def _mutation_warning_count(raw_index: dict[str, Any]) -> int:
    count = 0
    for group in raw_index.get("groups", {}).values():
        for sample in group.get("samples", []):
            count += int(sample.get("capture", {}).get("mutation_warning_count", 0))
    return count


class _temporary_decorator:
    def __init__(self, target_file: Path, function_name: str, decorator_expr: str, *, line: int | None = None):
        self.target_file = target_file
        self.function_name = function_name
        self.decorator_expr = decorator_expr
        self.line = line
        self.original = ""

    def __enter__(self):
        self.original = self.target_file.read_text(encoding="utf-8")
        self.target_file.write_text(
            _insert_decorator(self.original, self.function_name, self.decorator_expr, line=self.line),
            encoding="utf-8",
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.target_file.write_text(self.original, encoding="utf-8")
        return False


class _temporary_decorators:
    def __init__(self, target_file: Path, entries: list[tuple[SourceInterface, str]]):
        self.target_file = target_file
        self.entries = entries
        self.original = ""

    def __enter__(self):
        self.original = self.target_file.read_text(encoding="utf-8")
        source = self.original
        for interface, expr in sorted(self.entries, key=lambda item: item[0].line, reverse=True):
            source = _insert_decorator(source, interface.function_name, expr, line=interface.line)
        self.target_file.write_text(source, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.target_file.write_text(self.original, encoding="utf-8")
        return False


def _insert_decorator(source: str, function_name: str, decorator_expr: str, *, line: int | None = None) -> str:
    lines = source.splitlines()
    needle = f"def {function_name}("
    async_needle = f"async def {function_name}("
    target_line_idx = line - 1 if line is not None else None
    for idx, source_line in enumerate(lines):
        stripped = source_line.lstrip()
        if stripped.startswith(needle) or stripped.startswith(async_needle):
            if target_line_idx is not None and idx != target_line_idx:
                continue
            indent = source_line[: len(source_line) - len(stripped)]
            lines.insert(idx, f"{indent}@{decorator_expr}")
            return "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    if target_line_idx is not None:
        raise ValueError(f"Could not find function definition {function_name!r} at line {line}")
    raise ValueError(f"Could not find function definition {function_name!r}")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_baseline_report(path: Path, args: argparse.Namespace, result: dict[str, Any]) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            # Baseline Run Report

            - service_cmd: `{args.service_cmd}`
            - workload_cmd: `{args.workload_cmd}`
            - health_url: `{args.health_url}`
            - workload_returncode: `{result['workload']['returncode']}`
            - workload_elapsed_sec: `{result['workload']['elapsed_sec']:.3f}`

            ## Workload Stdout Tail

            ```text
            {result['workload']['stdout']}
            ```

            ## Workload Stderr Tail

            ```text
            {result['workload']['stderr']}
            ```
            """
        ),
        encoding="utf-8",
    )


def _env_to_yaml(data: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit(value: Any, indent: int, key: str | None = None) -> None:
        prefix = " " * indent
        if isinstance(value, dict):
            if key is not None:
                lines.append(f"{prefix}{key}:")
                indent += 2
                prefix = " " * indent
            for k, v in value.items():
                emit(v, indent, str(k))
        elif isinstance(value, list):
            if key is not None:
                lines.append(f"{prefix}{key}:")
                indent += 2
                prefix = " " * indent
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    emit(item, indent + 2)
                else:
                    lines.append(f"{prefix}- {json.dumps(item)}")
        else:
            scalar = json.dumps(value)
            if key is None:
                lines.append(f"{prefix}{scalar}")
            else:
                lines.append(f"{prefix}{key}: {scalar}")

    emit(data, 0)
    return "\n".join(lines) + "\n"


def _compare_availability(expected: dict[str, Any], current: dict[str, Any]) -> list[str]:
    expected_map = _availability_map(expected)
    current_map = _availability_map(current)
    mismatches = []
    for key, expected_value in sorted(expected_map.items()):
        current_value = current_map.get(key)
        if current_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, got {current_value}")
    return mismatches


def _availability_map(data: Any, prefix: str = "") -> dict[str, bool]:
    out: dict[str, bool] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key == "available" and isinstance(value, bool):
                out[prefix] = value
            else:
                out.update(_availability_map(value, path))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            out.update(_availability_map(item, f"{prefix}.{idx}"))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
