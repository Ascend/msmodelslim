# 自动调优配置说明导航

本目录是 msModelSlim 自动调优配置（YAML）的字段级参考文档，按先写调优计划、再按 `type` 查策略与评估服务的方式组织。使用流程与端到端操作步骤见《[自动调优使用说明](../../../user_guide/usage_auto_precision_tuning.md)》，命令行参数见《[msmodelslim tune 命令行 API 文档](../../cli/msmodelslim_tune.md)》，策略选型见《[自动调优策略总览](../../../knowledge_base/tuning_strategies/README.md)》。量化任务 YAML 的字段见《[量化任务配置说明导航](../quant/README.md)》；处理器与保存格式总表见《[配置说明导航](../README.md)》。

## 1. 怎么用这套文档

1. **确定顶层结构**：调优 YAML 根节点不是 `apiversion`，而是《[tuning_plan](tuning_plan.md)》的两个必选字段 `strategy` 与 `evaluation`。
2. **查策略**：`strategy` 由 `type` 分派，按下方[调优策略](#3-调优策略strategy-按-type-分派)表查对应页面。
3. **查评估服务**：`evaluation` 由 `type` 分派，按下方[评估服务](#4-评估服务evaluation-按-type-分派)表查对应页面。当前仅 `service_oriented`。
4. **查内嵌量化配置**：`standing_high` 与 `binary_fallback` 的 `template` 引用最佳实践或 `modelslim_v1` 任务配置，协议页见《[量化任务配置说明导航](../quant/README.md)》，处理器 `type` 见《[配置说明导航](../README.md)》。`standing_high_with_experience` 不要求手写完整量化 YAML，由 `quant_type` 与 `structure_configs` 生成。

> 本目录除本页外，均由 `skills/docs-management/scripts/gen_config_api_docs.py` 从源码 Pydantic 注解生成。修改字段说明请改源码 `Field(description=)` 等注解后重新生成，不要直接编辑带 `generated-by` 标记的页面。

## 2. 调优计划

| 配置 | 说明 |
|------|------|
| [tuning_plan](tuning_plan.md) | 调优计划根配置，只含 `strategy` 与 `evaluation`。 |

## 3. 调优策略（`strategy` 按 `type` 分派）

| `type` | 配置 | 作用 |
|--------|------|------|
| `standing_high` | [standing_high](strategy_standing_high.md) | 摸高：先做敏感层分析，再逐步回退并尝试不同离群值抑制链；需提供量化模板。 |
| `standing_high_with_experience` | [standing_high_with_experience](strategy_standing_high_with_experience.md) | 基于专家经验的摸高：由量化类型与结构配置生成量化方案，无需手写完整模板。 |
| `binary_fallback` | [binary_fallback](strategy_binary_fallback.md) | 二分回退：在完整 `PracticeConfig` 模板上搜索最小回退前缀。 |

`standing_high.template` 指向《[modelslim_v1 配置说明](../quant/modelslim_v1.md)》。`binary_fallback.template` 指向《[PracticeConfig 配置说明](../quant/practice_config.md)》，且 `apiversion` 须为 `modelslim_v1`。

## 4. 评估服务（`evaluation` 按 `type` 分派）

| `type` | 配置 | 作用 |
|--------|------|------|
| `service_oriented` | [evaluation_service_oriented](evaluation_service_oriented.md) | 面向服务的评估：精度期望、AISBench 评测与 vLLM-Ascend 推理引擎。 |
