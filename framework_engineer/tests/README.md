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

这组测试默认跳过。你需要在真实 GPU/SGLang 环境里显式开启：

```bash
export KA_REAL_SGLANG=1
export KA_SERVICE_CMD='python -m sglang.launch_server ...'
export KA_WORKLOAD_CMD='python run_your_workload.py ...'
export KA_TARGET_FILE='/path/to/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py'
export KA_FUNCTION_NAME='extend'
export KA_TARGET_NAME='sglang.srt.layers.attention.linear.kernels.gdn_triton.TritonGDNKernel.extend'

python3 -m unittest kernel_agent.framework_engineer.tests.test_real_sglang_phase1
```

常用可选项：

```bash
# 如果目标是 instance method，默认就是 1；普通函数可设为 0
export KA_DROP_FIRST_ARG=1

# 多个 mutable path 用逗号分隔
export KA_MUTABLE_ARG_PATHS='kwargs.ssm_states'

# 捕获 raw snapshot 上限，避免 workload 太大时写爆磁盘
export KA_MAX_RAW_CASES=32

# selected case 上限
export KA_MAX_SELECTED_CASES=8

# 如果默认追加 --disable-cuda-graph 不适合你的启动命令，显式提供 non-cudagraph 命令
export KA_NON_CUDAGRAPH_SERVICE_CMD='python -m sglang.launch_server ... --disable-cuda-graph'

# 保留临时 task pack，便于手动检查
export KA_KEEP_TASK_PACK=1

# 指定输出目录；如果设置了这个目录，测试不会自动删除
export KA_TASK_PACK=/tmp/qwen35_gdn_extend_task_pack

# 跑环境探测和 benchmark
export KA_RUN_PROBE_ENV=1
export KA_RUN_BENCHMARK=1

# validate 使用的设备，默认 cuda
export KA_VALIDATE_DEVICE=cuda
```

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

```bash
export KA_SKIP_BASELINE=1
```

如果 `--disable-cuda-graph` 追加策略不适合你的 SGLang 版本，务必使用 `KA_NON_CUDAGRAPH_SERVICE_CMD` 覆盖。
