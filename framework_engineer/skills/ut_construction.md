# Skill: Snapshot Harness Construction

Snapshot harness 用来把真实框架接口调用变成可独立验证的 kernel 任务。Phase 1 不再默认从 shape 随机生成输入；UT 和 benchmark 必须 replay selected snapshots。

## 输入

- `snapshots/manifest.json`
- `snapshots/selected/<group_id>/group_meta.json`
- `snapshots/selected/<group_id>/samples/<sample_id>/meta.json`
- `snapshots/selected/<group_id>/samples/<sample_id>/pre_inputs.pt`
- `snapshots/selected/<group_id>/samples/<sample_id>/post_inputs.pt`
- `snapshots/selected/<group_id>/samples/<sample_id>/outputs.pt`
- 目标候选接口签名默认为：

```python
candidate(*args, **kwargs)
```

## Correctness 规则

1. 从 selected snapshot 加载 `pre_inputs`。
2. 为 reference 和 candidate 各 clone 一份输入。
3. 运行 reference 或使用 snapshot-golden fallback。
4. 运行 candidate。
5. 比较 outputs。
6. 对 `mutable_arg_paths` 中的输入，比较运行后的 post-state。

`mutable_arg_paths` 不是必填项。只有当目标接口会原地修改输入，并且该
post-state 是语义的一部分时才需要配置。不存在于 captured inputs 的路径会被
记录为 warning，并从当前 sample 的 mutation comparison 中忽略。

如果 reference 不能脱离 SGLang 独立 import，允许使用 snapshot-golden fallback。此时 reference 返回 captured outputs，并把 mutable inputs 更新到 captured post-state。

## Benchmark 规则

- reference 和 candidate 使用同一批 selected snapshots。
- 每轮 timed run 前恢复 mutable inputs 到 pre-state。
- reset/clone/copy 不计入 timed region。
- CUDA event timing 优先；CPU timer 只作为 fallback。
- 输出 JSONL，包含 group_id、sample_id、target、warmup、repeat、median_us、mean_us、min_us、max_us，并输出 group summary。

## Candidate 初始状态

`candidate_impl.py` 初始版本应该直接调用 `reference_impl.py`，确保 task pack 初始 correctness pass。Kernel Engineer 接手后只替换 candidate 实现。

## 交付给 Kernel Agent 的内容

- `task.yaml`
- `shape_list.json`
- `snapshot_runtime.py`
- `snapshots/manifest.json`
- `snapshots/selected/<group_id>/samples/<sample_id>`
- `reference_impl.py`
- `candidate_impl.py`
- `correctness_test.py`
- `benchmark.py`
- `scripts/run_correctness.sh`
- `scripts/run_benchmark.sh`
- `scripts/run_ncu.sh`
- `env_manifest.yaml`
