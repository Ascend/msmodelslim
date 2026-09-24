# 配置说明导航

本目录是 msModelSlim 配置（YAML）的字段级参考文档，按先选协议、再查 `type`的方式组织。使用流程与端到端操作步骤见《[一键量化完整指南](../../user_guide/usage_quick_quantization.md)》；任务协议总览见《[量化任务配置说明导航](quant/README.md)》；自动调优的协议总览见《[自动调优配置说明导航](tuning/README.md)》。

## 1. 怎么用这套文档

1. **确定协议**：YAML 根节点的 `apiversion` 决定 `spec` 的结构，按任务场景选择协议页（总览见《[量化任务配置说明导航](quant/README.md)》）。各协议的字段取值与可用的 `process` / `save` 子类型并不相同，以对应任务页为准。
2. **查处理器**：`spec.process[]` 的每一项由 `type` 分派，按 `type` 到下方[处理器](#3-处理器specprocess-按-type-分派)表查对应页面；`spec.prior[].process` 同样引用这些处理器。
3. **查保存格式**：`spec.save[]` 的每一项由 `type` 分派，按下方[保存格式](#4-保存格式specsave-按-type-分派)表查对应页面；VLM / 多模态生成协议只接受其中的 AscendV1 与 MindIE 两类。
4. **查调优**：自动调优的顶层结构见《[自动调优配置说明导航](tuning/README.md)》，`strategy` 与 `evaluation` 的字段见下方[自动调优](#5-自动调优)表。

> 本页、《[量化任务配置说明导航](quant/README.md)》与《[自动调优配置说明导航](tuning/README.md)》为手写导航；其余配置页由 `skills/docs-management/scripts/gen_config_api_docs.py` 从源码 Pydantic 注解生成。修改字段说明请改源码 `Field(description=)` 等注解后重新生成，不要直接编辑带 `generated-by` 标记的页面。

## 2. 任务配置（按 `apiversion` 选择）

按协议查阅字段见《[量化任务配置说明导航](quant/README.md)》，下表为快速入口：

| 配置 | `apiversion` | 说明 |
|------|--------------|------|
| [modelslim_v1](quant/modelslim_v1.md) | `modelslim_v1` | 大语言模型量化与权重转换的默认协议，含 `runner`/`process`/`save`/`dataset`。 |
| [multimodal_vlm_modelslim_v1](quant/multimodal_vlm_modelslim_v1.md) | `multimodal_vlm_modelslim_v1` | 多模态理解（VLM）模型量化协议。 |
| [multimodal_sd_modelslim_v1](quant/multimodal_sd_modelslim_v1.md) | `multimodal_sd_modelslim_v1` | 多模态生成（DiT/SD）模型量化协议，含 dump 与推理参数。 |
| [modelslim_convert](quant/modelslim_convert.md) | `modelslim_convert` | 纯权重转换协议，用于权重重命名、结构替换与格式转换。 |
| [practice_config](quant/practice_config.md) | `modelslim_v1`（在 `task` 内） | 最佳实践配置：`metadata` + `task`，被自动调优策略引用。 |

## 3. 处理器（`spec.process[]` 按 `type` 分派）

| `type` | 配置 | 作用 |
|--------|------|------|
| `adapt_rotation` | [adapt_rotation](processor/adapt_rotation.md) | 自适应旋转，抑制离群值。 |
| `autoround_quant` | [autoround_quant](processor/autoround_quant.md) | AutoRound 量化训练。 |
| `awq` | [awq](processor/awq.md) | AWQ 激活感知权重量化。 |
| `binary_analysis` | [binary_analysis](processor/binary_analysis.md) | 二值（有/无量化）敏感度分析。 |
| `binary_operator_layer_wise` | [binary_operator_layer_wise](processor/binary_operator_layer_wise.md) | 逐层敏感度分析。 |
| `binary_operator_model_wise` | [binary_operator_model_wise](processor/binary_operator_model_wise.md) | 模型级敏感度分析。 |
| `dynamic_cache` | [dynamic_cache](processor/dynamic_cache.md) | KV cache 动态量化。 |
| `fa3_quant` | [fa3_quant](processor/fa3_quant.md) | FA3 注意力量化。 |
| `flatquant` | [flatquant](processor/flatquant.md) | FlatQuant 量化训练。 |
| `flex_awq_ssz` | [flex_awq_ssz](processor/flex_awq_ssz.md) | FlexAWQSSZ 平滑量化。 |
| `flex_smooth_quant` | [flex_smooth_quant](processor/flex_smooth_quant.md) | FlexSmoothQuant 平滑量化。 |
| `float_sparse` | [float_sparse](processor/float_sparse.md) | 浮点稀疏。 |
| `group` | [group](processor/group.md) | 组合处理器，共享推理过程。 |
| `iter_smooth` | [iter_smooth](processor/iter_smooth.md) | 迭代平滑。 |
| `kv_smooth` | [kv_smooth](processor/kv_smooth.md) | KV cache 平滑。 |
| `linear_quant` | [linear_quant](processor/linear_quant.md) | 线性层量化，含 `qconfig`（激活/权重）。 |
| `load` | [load](processor/load.md) | 模块加载/卸载。 |
| `oasq` | [oasq](processor/oasq.md) | OASQ 离群值感知平滑量化。 |
| `online_quarot` | [online_quarot](processor/online_quarot.md) | 在线 QuaRot 旋转。 |
| `quarot` | [quarot](processor/quarot.md) | 离线 QuaRot 旋转。 |
| `saver` | [saver](processor/saver.md) | 统一保存处理器。 |
| `smooth_quant` | [smooth_quant](processor/smooth_quant.md) | SmoothQuant 平滑量化。 |
| `svd_res` | [svd_res](processor/svd_res.md) | SVD 低秩残差补偿。 |
| `trainable_linear_quant` | [trainable_linear_quant](processor/trainable_linear_quant.md) | 可训练线性量化（TLQ）。 |
| `unary_analysis` | [unary_analysis](processor/unary_analysis.md) | 一元（无量化）敏感度分析。 |

## 4. 保存格式（`spec.save[]` 按 `type` 分派）

| `type` | 配置 | 适用协议 |
|--------|------|----------|
| `ascendv1_saver` | [ascendv1_saver](format/ascendv1_saver.md) | 全部协议（`modelslim_v1`、`multimodal_*`、`modelslim_convert`）。 |
| `compressed_tensors` | [compressed_tensors](format/compressed_tensors.md) | 仅 `modelslim_v1` 与 `modelslim_convert`；VLM / 多模态生成协议不接受。 |
| `mindie_format_saver` | [mindie_format_saver](format/mindie_format_saver.md) | 全部协议。 |

## 5. 自动调优

顶层协议与字段总览见《[自动调优配置说明导航](tuning/README.md)》，以下为各策略与评估服务的字段参考：

| 配置 | 说明 |
|------|------|
| [tuning_plan](tuning/tuning_plan.md) | 调优计划根配置（`strategy` + `evaluation`）。 |
| [strategy_standing_high](tuning/strategy_standing_high.md) | `standing_high` 策略。 |
| [strategy_standing_high_with_experience](tuning/strategy_standing_high_with_experience.md) | `standing_high_with_experience` 策略。 |
| [strategy_binary_fallback](tuning/strategy_binary_fallback.md) | `binary_fallback` 策略。 |
| [evaluation_service_oriented](tuning/evaluation_service_oriented.md) | AISBench 评测服务与预检查配置。 |
