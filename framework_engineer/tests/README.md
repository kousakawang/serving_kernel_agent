# Framework Engineer Phase 1 Tests

这些测试用于在真实 GPU 服务器上先验证 CLI 和 snapshot 基础链路，再接入真实 SGLang 目标。

## 1. 本地/CPU 快速测试

在仓库根目录运行：

```bash
python3 -m unittest \
  kernel_agent.framework_engineer.tests.test_snapshot \
  kernel_agent.framework_engineer.tests.test_cli_phase1
```

这组测试不需要 SGLang，也不需要 GPU。它会：

- 创建临时 task pack。
- 创建一个 toy kernel 接口和 workload。
- 通过 CLI 临时插桩目标函数。
- 跑 `probe-target-calls`。
- 跑 `capture-snapshots`。
- 跑 `select-snapshots`。
- 跑 `generate-harness`。
- 跑 `validate-task-pack --skip-env-check --run-correctness --run-benchmark`。

## 2. GPU smoke 测试

如果服务器有 CUDA，可以运行：

```bash
KA_TEST_DEVICE=cuda python3 -m unittest \
  kernel_agent.framework_engineer.tests.test_cli_phase1
```

这仍然使用 toy kernel，但 tensor 会放在 CUDA 上，能验证 snapshot capture 的 GPU tensor 保存、CPU copy、replay、mutable post-state correctness。

## 3. 只测试 snapshot 数据结构

```bash
python3 -m unittest kernel_agent.framework_engineer.tests.test_snapshot
```

覆盖：

- tensor meta 记录 dtype/shape/stride/storage_offset/contiguous。
- tree serializer 支持 tensor/primitive/list/tuple/dict/None。
- `query_start_loc` diff 不同会产生不同 `semantic_hash`。
- selected snapshot 生成 harness 后 correctness 初始可通过。

## 4. 真实 SGLang 接入前建议

先确认 CLI 可用：

```bash
python3 -m kernel_agent.framework_engineer.cli --help
```

如果没有完整环境或不想跑环境探测，验证 task pack 时可以先跳过环境一致性：

```bash
python3 -m kernel_agent.framework_engineer.cli validate-task-pack \
  --task-pack <task_pack> \
  --skip-env-check \
  --run-correctness
```

真实 GPU/SGLang 环境可用后，再运行：

```bash
python3 -m kernel_agent.framework_engineer.cli probe-env \
  --task-pack <task_pack>

python3 -m kernel_agent.framework_engineer.cli validate-task-pack \
  --task-pack <task_pack> \
  --run-correctness \
  --run-benchmark
```

## 5. 真实 SGLang 集成测试

这组测试默认跳过。你只需要提供一个 Python 配置文件：

```bash
cp kernel_agent/framework_engineer/tests/real_sglang_phase1_config.example.py \
  /tmp/real_sglang_phase1_config.py

# 编辑 /tmp/real_sglang_phase1_config.py

KA_REAL_SGLANG_CONFIG=/tmp/real_sglang_phase1_config.py \
python3 -m unittest kernel_agent.framework_engineer.tests.test_real_sglang_phase1
```

配置文件是普通 Python 文件，必填字段：

```python
service_cmd = "python -m sglang.launch_server ..."
workload_cmd = "python run_your_workload.py ..."
target_file = "/path/to/gdn_triton.py"
function_name = "extend"
target_name = "sglang...TritonGDNKernel.extend"
```

完整模板见：

- `kernel_agent/framework_engineer/tests/real_sglang_phase1_config.example.py`

可以用 `cli_tests` 字典控制每个 CLI 是否执行。key 是 CLI subcommand 名，value 是 bool；没写到字典里的 CLI 默认执行：

```python
cli_tests = {
    "scaffold-task-pack": True,
    "run-baseline": False,
    "probe-target-calls": True,
    "capture-snapshots": True,
    "select-snapshots": True,
    "generate-harness": True,
    "probe-env": False,
    "validate-task-pack": True,
}
```

跳过有前后依赖的步骤时，测试不会自动补齐依赖。例如跳过 `capture-snapshots` 但保留 `select-snapshots`，需要你确保 task pack 里已经有可用的 raw snapshots。

测试流程：

1. `scaffold-task-pack`
2. `run-baseline`
3. `probe-target-calls`
4. `capture-snapshots`
5. `select-snapshots`
6. `generate-harness`
7. 可选 `probe-env`
8. `validate-task-pack --run-correctness`

如果当前只想测试 snapshot/harness，不想跑 baseline：

```python
skip_baseline = True
```

如果 `--disable-cuda-graph` 追加策略不适合你的 SGLang 版本，务必使用 `KA_NON_CUDAGRAPH_SERVICE_CMD` 覆盖。

```python
non_cudagraph_service_cmd = "python -m sglang.launch_server ... --disable-cuda-graph"
```
