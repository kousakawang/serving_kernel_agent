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

本地如果没有 PyTorch，也可以运行这组测试；toy workload 使用纯 Python
输入来验证 CLI、group/sample snapshot、selector 和 generic harness。

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
- forward boundary 能给 target call 标记 `forward_id`。
- 同一 group 的 hit count 和 sample 上限统计正确。
- selected group/sample 生成 harness 后 correctness 初始可通过。

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
target_line = 123
```

也可以继续使用旧写法：

```python
function_name = "extend"
target_name = "sglang...TritonGDNKernel.extend"
```

推荐先用 CLI 验证行号解析结果：

```bash
python3 -m kernel_agent.framework_engineer.cli resolve-interface \
  --file /path/to/sglang/python/sglang/srt/models/qwen3_5.py \
  --line 123
```

输出会包含 `target_file`、`function_name`、`target_name`，并能识别 class method，例如
`sglang.srt.models.qwen3_5.Qwen3_5ForCausalLM.forward`。

当 `target_line` 已设置时，真实测试只会把 `target_file + target_line` 传给 CLI；
`function_name` 和 `target_name` 不需要手动填写。

可选但推荐配置 forward boundary，让 capture 能区分模型 forward window：

```python
forward_boundary_file = "/path/to/backend_or_model.py"
forward_boundary_line = 456
```

如果需要 override 自动推导的名字，可以额外设置：

```python
forward_boundary_function = "forward"
forward_boundary_name = "sglang...Qwen3_5ForCausalLM.forward"
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

`generate-harness` 会生成 `original_impl.py` 作为 benchmark reference，默认调用
capture 时的原始 target。`snapshot-golden` 只用于 correctness fallback，不能作为
性能 baseline。

如果当前只想测试 snapshot/harness，不想跑 baseline：

```python
skip_baseline = True
```

如果 `--disable-cuda-graph` 追加策略不适合你的 SGLang 版本，务必使用 `KA_NON_CUDAGRAPH_SERVICE_CMD` 覆盖。

```python
non_cudagraph_service_cmd = "python -m sglang.launch_server ... --disable-cuda-graph"
```

FLA chunk 这类 free function 一般需要：

```python
drop_first_arg = False
signature = "candidate(*args, **kwargs)"
mutable_arg_paths = []
```

`mutable_arg_paths` 只用于目标接口会原地修改输入的情况，例如某些 state/cache
更新。普通算子只返回 outputs 时保持空列表即可；如果误填了不存在的路径，
capture 会继续运行，并在 report/meta 中记录 warning。
