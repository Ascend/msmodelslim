<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.practice.interface.PracticeConfig -->
# PracticeConfig 配置说明

## 1. 配置概述

完整最佳实践量化任务配置：metadata + task（task ≡ apiversion + spec）。

| 项目 | 内容 |
|------|------|
| 配置类 | `PracticeConfig` |
| 源码 | [interface.py](../../../../../msmodelslim/core/practice/interface.py) |

## 2. 参数列表

<h3 id="2-1-practice-config">2.1 PracticeConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `metadata` | `object` | 可选 | 见嵌套配置默认值 | — | 量化配置元数据（config_id/score/label/verified_*） | 本页 <a href="#2-2-metadata">§2.2</a> |
| `task` | `object` | 可选 | 见嵌套配置默认值 | — | 量化任务描述（apiversion + spec） | 本页 <a href="#2-3-basequantconfig">§2.3</a> |

**配置约束**

- 预处理：旧格式 {apiversion, metadata, spec, ...} → {metadata, task:{apiversion, spec, ...}}。
- 剥除聚合后 task. 前缀，使错误路径与 yaml 的 spec. 一致。

<h3 id="2-2-metadata">2.2 Metadata</h3>

量化配置元数据：标识配置的 ID、评分、标签与已验证的模型/场景。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `config_id` | `string` | 可选 | `Unknown` | — | 量化配置 ID，例如 'Qwen3-32B W8A8' | 无 |
| `score` | `float` | 可选 | `100.0` | — | 量化配置评分，用于排序，必须 >= 0 | 无 |
| `label` | `object` | 可选 | `{}` | — | 量化配置标签，用于过滤，例如 {'w_bit': 8, 'a_bit': 8, 'is_sparse': True, 'kv_cache': True} | 无 |
| `verified_model_types` | `list[string]` | 可选 | `[]` | — | 已验证的模型类型列表，例如 ['LLaMa3.1-70B', 'Qwen2.5-72B'] | 无 |
| `verified_tags` | `object` | 可选 | `{}` | — | 已验证场景标签：键为模型类型，值为场景标签列表（每个场景是一组标签，如 ['MindIE','Atlas_A2_Inference']） | 无 |

**配置约束**

- 无。

<h3 id="2-3-basequantconfig">2.3 BaseQuantConfig</h3>

**派生类**

| 配置类 | 说明 | 文档 |
|--------|------|------|
| `ModelslimV1QuantConfig` | `modelslim_v1` 量化任务配置，位于 YAML 根节点。 | 《[modelslim_v1 配置说明](modelslim_v1.md)》 |
| `ModelslimConvertQuantConfig` | `modelslim_convert` 量化（权重转换）任务配置，位于 YAML 根节点。 | 《[modelslim_convert 配置说明](modelslim_convert.md)》 |

## 3. 完整配置参考

```yaml
metadata:
  config_id: Qwen3-32B W8A8
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: false
    kv_cache: false
  verified_model_types:
  - Qwen3-32B
task:
  apiversion: modelslim_v1
  spec:
    process:
    - type: linear_quant
      qconfig:
        act:
          dtype: int8
          scope: per_tensor
          symmetric: false
          method: minmax
        weight:
          dtype: int8
          scope: per_channel
          symmetric: true
          method: minmax
      include:
      - '*'
      exclude: []
    save:
    - type: ascendv1_saver
      part_file_size: 4
    dataset: mix_calib.jsonl
```
