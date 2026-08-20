<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.practice.interface.PracticeConfig -->
# PracticeConfig 配置说明

## 1. 配置概述

完整最佳实践量化任务配置：apiversion + spec + metadata。

| 项目 | 内容 |
|------|------|
| 配置类 | `PracticeConfig` |
| 源码 | [interface.py](../../../../../msmodelslim/core/practice/interface.py) |

## 2. 参数列表

<h3 id="2-1-practice-config">2.1 PracticeConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `apiversion` | `string` | 可选 | Unknown（代码占位；YAML 中须按任务类型显式指定） | `modelslim_v1`、`multimodal_vlm_modelslim_v1`、`multimodal_sd_modelslim_v1`、`modelslim_convert` | API 版本（任务类型），决定 spec 的结构：`modelslim_v1`、`multimodal_vlm_modelslim_v1`、`multimodal_sd_modelslim_v1`、`modelslim_convert`；YAML 中必须显式指定，默认值 `Unknown` 仅为代码内部占位，不可直接使用。 | 无 |
| `spec` | `any` | 可选 | `{}` | — | 任务规格，结构随 apiversion 而定。 | 无 |
| `metadata` | `object` | 可选 | 见嵌套配置默认值 | — | 量化配置元数据（config_id/score/label/verified_*） | 本页 <a href="#2-2-metadata">§2.2</a> |

**配置约束**

- 无。

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
