# Binary Fallback 调优算法

## 简介

Binary Fallback（二分回退算法）是一种仅通过**二分搜索**确定最小量化回退配置的自动调优策略。与 Standing High 不同，本策略**不**遍历离群值抑制策略，也**不**执行摸高减层；离群值抑制等处理器固定写在 `template` 中，算法只调整 `rollback_path` 指向的 list 型回退字段。

当前仅支持 `apiversion: modelslim_v1` 的量化配置。

## 原理

1. **零回退评估**：将 `rollback_path` 对应字段设为 `[]` 并评估；若精度达标则结束。
2. **候选列表**：
   - 配置了 `rollback_candidates` 时，直接使用用户给定的有序列表，**跳过**敏感层分析。
   - 未配置时，运行敏感层分析（`quant_modules=["*"]`），按敏感度降序得到层名列表。
3. **二分搜索**：在候选前缀 `candidates[:k]` 中搜索满足精度的**最小** `k`。
4. **终止**：
   - 找到最小 `k` → 输出对应 PracticeConfig。
   - 即使 `k = len(candidates)` 仍不达标 → 抛出 `SpecError`。

## 使用说明

在自动调优 YAML 的 `strategy` 字段将 `type` 设为 `binary_fallback`。完整命令见 [自动调优使用说明](../../feature_guide/auto_precision_tuning/usage.md)。

### YAML 配置示例

```yaml
strategy:
  type: binary_fallback
  rollback_path: spec.process.1.exclude
  rollback_candidates: []   # 省略或空 → 跑敏感层分析
  analysis_dataset: mix_calib.jsonl  # 可选；默认 template.spec.dataset
  template:
    apiversion: modelslim_v1
    metadata:
      config_id: qwen3-32b-w8a8-tune
      label:
        w_bit: 8
        a_bit: 8
        is_sparse: false
        kv_cache: false
    spec:
      runner: auto
      process:
        - type: iter_smooth
          alpha: 0.5
        - type: linear_quant
          qconfig:
            act:
              scope: per_tensor
              dtype: int8
              symmetric: false
              method: minmax
            weight:
              scope: per_channel
              dtype: int8
              symmetric: true
              method: minmax
          include: ["*"]
          exclude: []
      save:
        - type: ascendv1_saver
          part_file_size: 4
      dataset: mix_calib.jsonl
```

用户指定回退候选（跳过敏感层分析）：

```yaml
rollback_candidates:
  - "*model.layers.2.mlp.down_proj.*"
  - "*model.layers.5.mlp.down_proj.*"
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `type` | 固定为 `binary_fallback` |
| `template` | 完整最佳实践 PracticeConfig（含 `apiversion`、`metadata`、`spec`）；`apiversion` 必须为 `modelslim_v1` |
| `rollback_path` | 点分路径，指向 template 内**必须为 list** 的字段，如 `spec.process.1.exclude` |
| `rollback_candidates` | 可选有序回退候选；非空则跳过敏感层分析 |
| `analysis_dataset` | 可选；敏感层分析校准集名称，默认 `template.spec.dataset` |

### 与 Standing High 对比

| 项 | Standing High | Binary Fallback |
|----|---------------|-----------------|
| 离群值抑制 | 遍历 `anti_outlier_strategies` | 固定在 `template.spec.process` |
| 回退搜索 | 二分 + 摸高减层 | 仅二分 |
| template | 仅 spec 片段 + 外层 metadata | 完整 PracticeConfig |
| 回退写入 | 硬编码 `linear_quant.exclude` | `rollback_path` 指定 |
| 最大回退仍不达标 | 返回当前最佳 | `SpecError` |

## 适用要求

**推理引擎**：与 Standing High 相同，推理引擎需支持配置中的任意层回退（如 vLLM-Ascend 单算子模式）。

**模型适配**：

| 场景 | 要求 |
|------|------|
| 已配置非空 `rollback_candidates` | 无额外模型协议要求（跳过敏感层分析） |
| 未配置或为空 `rollback_candidates` | 须实现 **`ModelSlimPipelineInterfaceV1`**（即敏感层分析服务 `PipelineAnalysisService` 的模型协议，与 `PipelineInterface` 相同）；校准集由调优层注入的 `DatasetLoaderInfra` 经分析服务加载 |

与 [Standing High](standing_high.md#适用要求) 的自动敏感层分析要求一致。模型接入见 [LLM 大模型接入指南 — 自动调优与敏感层分析](../../../development_guide/integrating_models.md#自动调优与敏感层分析)。
