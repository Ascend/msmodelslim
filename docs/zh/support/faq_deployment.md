---
title: FAQ - 部署与推理
description: 部署与推理 FAQ；MindIE / vLLM-Ascend 部署问题请查阅《主流模型量化部署》，并覆盖 FP8/INT4→BF16、BF16→MXFP8 权重转换。
keywords: [部署, 推理, MindIE, vLLM-Ascend, 上线, 权重转换, FP8, INT4, BF16, MXFP8, AscendV1, convert]
---

# 部署与推理 FAQ

> 本节用于沉淀量化模型在 MindIE、vLLM-Ascend 等推理引擎部署与推理阶段的常见问题。

已获得量化权重、需要部署上线的场景，请参见：

- 《[主流模型量化部署](../user_guide/process_mainstream_model_deployment.md)》

新增条目请按 `1.`、`2.` 顺序在文末追加，并保持“问题现象 / 问题原因 / 解决方法 / 关联文档”的四段结构（段内小节编号依次为 `x.1`~`x.4`）。

## 1. 为什么 FP8/INT4 权重转 BF16 时找不到可执行脚本或命令？

### 1.1 问题现象

拿到 FP8 或 INT4 packed 权重后想转成 BF16 继续量化/部署，直接执行 `convert_fp8_to_bf16.py` 时无参数入口或报错；不清楚应该用哪条命令。

### 1.2 问题原因

权重转换入口是 `msmodelslim quant --config`，使用 `modelslim_convert` 配置；`example/*/convert_fp8_to_bf16.py` 是库/工具函数，不是命令行入口。转换工具会根据 checkpoint 自动识别 `FP8_BLOCK`、`INT4_PACKED`、`FLOAT` 源格式。

### 1.3 解决方法

1. 新建 `modelslim_convert` 配置，将需要转换的 linear 设置为 `target: FLOAT`。
2. 优先复用现成示例：`qwen3_8b_fp8_to_bf16.yaml`（FP8→BF16）、`kimi_k2_5_int4_per_group_to_bf16.yaml`（INT4→BF16）。
3. 执行转换：

   ```bash
   msmodelslim quant \
     --model_path "${MODEL_PATH}" \
     --save_path "${SAVE_PATH}" \
     --config "${CONFIG_PATH}"
   ```

4. DeepSeek 官方 FP8 权重可直接作为量化输入，无需先转 BF16；需要多次量化时，建议先持久化 BF16 权重以复用。
5. 转换后校验 `config.json`、权重分片与 scale 字段；不同模型的 scale 字段名可能不同（如 `.scale`、`.weight_scale_inv`），以对应模型转换脚本为准。

> 若目标模型/格式未在《权重转换使用指南》或模型 README 中列出，先确认支持范围，不要直接套用其他模型配置。

### 1.4 关联文档

- 转换配置、操作步骤与示例请参见《[权重转换使用指南](../knowledge_base/ptq/convert/usage_weight_conversion.md)》。
- DeepSeek 官方 FP8 权重的量化与 BF16 持久化说明请参见《[DeepSeek 模型量化示例](../../../example/DeepSeek/README.md)》。

## 2. 为什么 BF16 权重转换后没有生成 MXFP8 产物？

### 2.1 问题现象

希望得到 W8A8 MXFP8 权重用于昇腾部署，但按普通量化流程执行后，产物不是 MXFP8 格式。

### 2.2 问题原因

MXFP8 需要显式指定目标格式 `target: W8A8_MXFP8`，并使用昇腾格式保存；未指定目标格式或保存格式时，产物不会自动变成 MXFP8。不同模型和入口（`--config` / `--quant_type`）的支持范围不同，需按对应文档确认。

### 2.3 解决方法

1. 使用 `modelslim_convert` 配置，将目标 linear 设置为 `target: W8A8_MXFP8`。
2. 保存格式设置为 `save.type: ascend_v1`，参考 `qwen3_8b_bf16_to_mxfp8.yaml`。
3. 若源权重是 FP8，使用 `route: [FP8_BLOCK, FLOAT, W8A8_MXFP8]` 显式指定“先反量化再量化”，或让工具自动选择最短路径。
4. 执行 `msmodelslim quant --config` 后，用产物中的 `quant_model_description.json` 校验格式。

### 2.4 关联文档

- 转换配置与示例请参见《[权重转换使用指南](../knowledge_base/ptq/convert/usage_weight_conversion.md)》。
- 量化入口与 `--quant_type` 支持范围请参见《[模型量化使用指南](../user_guide/usage_weight_quantization.md)》。
- MXFP8 的概念与数据格式请参见《[量化模式总览](../knowledge_base/quantization_mode/README.md)》与《[MXFP8 数据格式](../knowledge_base/dtype_ops/term_mxfp8.md)》。
- 部署侧的格式与引擎匹配请参见《[主流模型量化部署](../user_guide/process_mainstream_model_deployment.md)》。
