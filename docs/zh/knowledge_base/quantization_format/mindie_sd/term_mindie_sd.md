# MindIE-SD

> **词条类别**：量化数据格式
>
> **英文名称**：MindIE-SD Quantization Format
>
> **英文缩写**：MindIE-SD
>
> **中文别名**：MindIE 多模态生成保存格式
>
> **应用领域**：多模态生成模型量化压缩、MindIE 部署
>
> **msModelSlim 实现**：`msmodelslim/format/mindie_format/`、`MindIEQuantFormatConfig`

## 1. 概述

MindIE-SD 是 msModelSlim 面向 **多模态生成** 场景、供 MindIE-SD 消费的[量化格式](../README.md)。一键量化通过保存器类型 `mindie_format_saver` 启用。它解决扩散 / DiT 等多模态生成模型量化权重与 MindIE-SD 加载约定对齐的问题；核心特征是与 `multimodal_sd_modelslim_v1` 配置协议配合，并以 `quant_model_description.json` + `quant_model_weight.safetensors` 落盘。

配置、执行与部署步骤见《[MindIE-SD 使用指南](mindie_sd_usage.md)》。各量化模式的原理与公式见《[量化模式](../../quantization_mode/term_quantization_mode.md)》及下文支持表中的词条链接。

## 2. 词条介绍

### 2.1 原理

#### 2.1.1 核心思想

MindIE-SD 本质上是一套**面向多模态生成、供 MindIE-SD 推理引擎消费的量化模型落盘约定**：它不执行校准或伪量化计算，而是在 `multimodal_sd_modelslim_v1` 量化流水线末尾，将已量化完成的模块转换为 MindIE-SD 推理引擎可直接加载的两类交付件——**量化描述文件** `quant_model_description*.json` 与 **量化权重文件** `quant_model_weight*.safetensors`（注意权重文件名为单数 `weight`，与 AscendV1 的 `quant_model_weights` 不同）。前者作为张量级索引，供 MindIE-SD 推理引擎识别各张量名称及其量化模式；后者按相同键名存放对应量化参数。推理侧先读取描述文件，再按键名从 safetensors 取数。

因此，用户可将 MindIE-SD 理解为：**多模态生成量化结果的 MindIE-SD 推理引擎标准导出包**——算法与多模态适配器负责量化与校准编排，MindIE-SD 负责与 MindIE-SD 推理引擎加载路径对齐的落盘与描述。

#### 2.1.2 关键性质

- 面向多模态生成（如 Wan2.2），而非通用 LLM AscendV1 默认路径。
- YAML 中 `type` 固定为 `mindie_format_saver`。
- 支持按 `part_file_size` 分片；`0` 表示不分片。
- 常与 `multimodal_sd_config`（dump / inference_config）一同出现。
- 描述文件中的枚举值表达对多种量化模式的承载能力；未实现 handler 的模式需改用《[AscendV1](../ascendv1/term_ascendv1.md)》。

### 2.2 <span id="export-artifacts">导出产物（交付件）</span>

#### 目录与文件说明

执行一键量化（`mindie_format_saver`）后，典型交付件位于 `save_path`（分布式场景下可能落在 `rank_*` 子目录）：

```bash
├── quant_model_description.json              # 量化权重描述（也可带量化类型后缀）
├── quant_model_weight.safetensors            # 量化权重（可分片；也可带量化类型后缀）
├── quant_model_weight.safetensors.index.json # 分片时生成的索引（可选）
├── *.json / *.py                             # 自源模型复制的配置与代码（不含 index.json）
└── （可选）calib_data_*.pth                  # dump 校准数据（enable_dump 时，目录由 dump_data_dir 决定）
```

| 文件名                                                                               | 说明                                            |
| --------------------------------------------------------------------------------- | --------------------------------------------- |
| `quant_model_description.json`（或 `quant_model_description_{quant_type}.json`）     | **量化权重描述文件**，记录张量量化类型与元数据                     |
| `quant_model_weight.safetensors`（或 `quant_model_weight_{quant_type}.safetensors`） | **量化权重文件**；较大时可分片，并通过 index.json 索引           |
| `*.json` / `*.py`                                                                 | 自源模型复制的配置与代码文件；**不复制** `index.json`；权限按工具约定设置 |
| `calib_data_*.pth`                                                                | **可选**：校准 dump 数据，见下文可选导出                     |

> 注意：MindIE-SD 权重文件名为 `quant_model_weight`（单数），与 AscendV1 的 `quant_model_weights`（复数）不同。

`quant_model_description.json` 中，每个张量键对应一个量化类型标识；同一 Linear 层的相关参数通常共享相同类型标识。

#### quant_model_description.json

##### 文件结构示例

```json
{
  "model_quant_type": "W8A8",
  "group_size": 32,
  "model.layers.0.self_attn.q_proj.weight": "W8A8",
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8",
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8",
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8",
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8",
  "model.layers.0.self_attn.q_proj.bias": "FLOAT"
}
```

> 张量键名由模型适配器决定；启用 FA3 等时还可出现 `fa_quant_type`、层级 `quant_type` 等字段。

##### <span id="global-metadata">全局元数据字段</span>

| 字段名                | 类型     | 说明                                               |
| ------------------ | ------ | ------------------------------------------------ |
| `model_quant_type` | string | 模型整体量化类型（由 handler 写入，如 `"W8A8"`、`"W8A8_MXFP8"`） |
| `group_size`       | int    | 分组 / block 相关大小；MXFP 等场景常见默认 `32`                |
| `fa_quant_type`    | string | Flash Attention 相关量化类型（启用 FA3 等时写入）              |
| `{张量名}`            | string | 各张量所属量化类型枚举                                      |

完整多模态配置协议见《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md#53-multimodal_sd_modelslim_v1-配置详解)》。

#### <span id="optional-dump">可选导出：校准 dump 数据</span>

当 `multimodal_sd_config.dump_config.enable_dump` 为 true 时，可在 `dump_data_dir`（为空则使用 `save_path`）写出校准 pth，例如：

```bash
calib_data_<task_config>_low_noise_model.pth
calib_data_<task_config>_high_noise_model.pth
```

字段含义见《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md#dump_config---校准数据捕获配置)》。该目录属于量化过程辅助数据，**不一定**随 MindIE 部署目录一并交付。

### 2.3 <span id="engine-support">推理引擎支持情况</span>

MindIE-SD **均可落盘**下表中的格式枚举；下表描述的是产物能否被 **MindIE 多模态生成**路径加载。口径依据《[大模型支持矩阵](../../model/README.md)》与 lab_practice 验证标签；具体模型 × 模式 × MindIE-SD 版本以支持矩阵与官方最佳实践为准。

| 格式枚举值 | MindIE-SD | 说明 |
| --- | --- | --- |
| `FLOAT` | √ | 未量化张量，随模型一并加载 |
| `W8A8` | √ | 静态 W8A8 |
| `W8A8_DYNAMIC` | √ | 动态 W8A8 |
| `W8A8_MXFP8` | √ | MXFP8；常见于生成最佳实践 |
| `W4A4_MXFP4` | √ | MXFP4 |
| `W4A4_MXFP4_DUALSCALE` | √ | MXFP4 双 Scale |
| `FAQuant` | √ | FA3 等；常与线性层量化组合 |

> **图例**：`√` 表示该引擎存在可加载路径或已有验证实践。通用 LLM 的 vLLM Ascend / SGLang / MindIE 路径请改用《[AscendV1](../ascendv1/term_ascendv1.md)》；选型后再在下文「[量化模式支持情况](#mode-support)」核对交付件字段。

### 2.4 <span id="mode-support">量化模式支持情况</span>

> **交付件说明**：「交付件：量化描述 JSON」→ `quant_model_description*.json`；「交付件：量化 safetensors」→ `quant_model_weight*.safetensors`。模式原理见《[量化模式](../../quantization_mode/term_quantization_mode.md)》词条。具体模型与 `quant_type` 组合以《[大模型支持矩阵](../../model/README.md)》及 lab_practice / example 为准。

| 格式枚举值                  | MindIE-SD 是否支持落盘 | 量化模式词条 | 交付件：量化描述 JSON                                    | 交付件：量化 safetensors                              |
| ---------------------- | ---------------- | ------ | ------------------------------------------------ | ----------------------------------------------- |
| `FLOAT`                | 支持               | [量化模式总览](../../quantization_mode/term_quantization_mode.md) | [FLOAT 描述键](#desc-float)                         | [FLOAT 权重张量](#st-float)                         |
| `W8A8`                 | 支持               | [W8A8 静态量化](../../quantization_mode/linear_layer_quantization/term_w8a8_static.md) | [W8A8 描述键](#desc-w8a8)                           | [W8A8 权重张量](#st-w8a8)                           |
| `W8A8_DYNAMIC`         | 支持               | [W8A8 动态量化](../../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md) | [W8A8_DYNAMIC 描述键](#desc-w8a8-dynamic)           | [W8A8_DYNAMIC 权重张量](#st-w8a8-dynamic)           |
| `W8A8_MXFP8`           | 支持               | [W8A8 MX 动态量化](../../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md) | [W8A8_MXFP8 描述键](#desc-w8a8-mxfp8)               | [W8A8_MXFP8 权重张量](#st-w8a8-mxfp8)               |
| `W4A4_MXFP4`           | 支持               | [W4A4 MX 动态量化](../../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md) | [W4A4_MXFP4 描述键](#desc-w4a4-mxfp4)               | [W4A4_MXFP4 权重张量](#st-w4a4-mxfp4)               |
| `W4A4_MXFP4_DUALSCALE` | 支持               | [W4A4 MX 双 Scale](../../quantization_mode/linear_layer_quantization/term_w4a4_mx_dualscale.md) | [W4A4_MXFP4_DUALSCALE 描述键](#desc-mxfp-dualscale) | [W4A4_MXFP4_DUALSCALE 权重张量](#st-mxfp-dualscale) |
| `FAQuant`              | 支持（FA3 等）        | [FA PerHead](../../quantization_mode/fa_quantization/term_fa_perhead.md) | [FAQuant 描述键](#desc-faquant)                     | [FAQuant 权重张量](#st-faquant)                     |

### 2.5 各量化模式交付件格式

约定：

- `{prefix}` 为模块前缀（由多模态适配器命名决定）。
- **描述 JSON**：文件 `quant_model_description*.json`；键为张量全名，值为量化类型字符串；全局字段见上文「[全局元数据字段](#global-metadata)」。
- **safetensors**：文件 `quant_model_weight*.safetensors`（可分片）；键为 `{prefix}.<param>`，存实际数值张量。

#### FLOAT

##### <span id="desc-float">quant_model_description.json</span>

| 描述键                    | 取值        | 说明                                              |
| ---------------------- | --------- | ----------------------------------------------- |
| `{prefix}.weight` 等参数名 | `"FLOAT"` | 未量化参数；`on_float_module` 按 `named_parameters` 写出 |

##### <span id="st-float">quant_model_weight*.safetensors</span>

| 张量名          | 数据类型   | 说明                                  |
| ------------ | ------ | ----------------------------------- |
| `{prefix}.*` | 与源参数一致 | 浮点参数原样落盘；在线旋转矩阵等也可能以 `"FLOAT"` 标签写入 |

#### W8A8

##### <span id="desc-w8a8">quant_model_description.json</span>

| 描述键                     | 取值        | 说明            |
| ----------------------- | --------- | ------------- |
| `{prefix}.weight`       | `"W8A8"`  | 量化权重          |
| `{prefix}.quant_bias`   | `"W8A8"`  | 量化偏置          |
| `{prefix}.input_scale`  | `"W8A8"`  | 激活 scale      |
| `{prefix}.input_offset` | `"W8A8"`  | 激活 zero-point |
| `{prefix}.deq_scale`    | `"W8A8"`  | 综合反量化 scale   |
| `{prefix}.bias`         | `"FLOAT"` | 原始浮点偏置（可选）    |

##### <span id="st-w8a8">quant_model_weight*.safetensors</span>

| 张量名                     | 数据类型    | 说明              |
| ----------------------- | ------- | --------------- |
| `{prefix}.weight`       | int8    | 量化权重            |
| `{prefix}.quant_bias`   | int32   | 量化偏置            |
| `{prefix}.input_scale`  | float32 | 激活量化 scale      |
| `{prefix}.input_offset` | float32 | 激活量化 zero-point |
| `{prefix}.deq_scale`    | float32 | 综合反量化 scale     |
| `{prefix}.bias`         | float32 | 原始浮点偏置（可选）      |

#### W8A8_DYNAMIC

##### <span id="desc-w8a8-dynamic">quant_model_description.json</span>

| 描述键                      | 取值               | 说明              |
| ------------------------ | ---------------- | --------------- |
| `{prefix}.weight`        | `"W8A8_DYNAMIC"` | 量化权重            |
| `{prefix}.weight_scale`  | `"W8A8_DYNAMIC"` | 权重量化 scale      |
| `{prefix}.weight_offset` | `"W8A8_DYNAMIC"` | 权重量化 zero-point |
| `{prefix}.bias`          | `"FLOAT"`        | 偏置（可选）          |

激活动态参数不落盘。

##### <span id="st-w8a8-dynamic">quant_model_weight*.safetensors</span>

| 张量名                      | 数据类型    | 说明                      |
| ------------------------ | ------- | ----------------------- |
| `{prefix}.weight`        | int8    | 量化权重                    |
| `{prefix}.weight_scale`  | float32 | 权重量化 scale              |
| `{prefix}.weight_offset` | float32 | 权重量化 zero-point（对称时为 0） |
| `{prefix}.bias`          | float32 | 原始浮点偏置（可选）              |

#### W8A8_MXFP8

##### <span id="desc-w8a8-mxfp8">quant_model_description.json</span>

| 描述键                     | 取值             | 说明               |
| ----------------------- | -------------- | ---------------- |
| `{prefix}.weight`       | `"W8A8_MXFP8"` | MXFP8 权重         |
| `{prefix}.weight_scale` | `"W8A8_MXFP8"` | block-wise scale |
| `{prefix}.bias`         | `"FLOAT"`      | 偏置（可选）           |

##### <span id="st-w8a8-mxfp8">quant_model_weight*.safetensors</span>

| 张量名                     | 数据类型          | 说明                                     |
| ----------------------- | ------------- | -------------------------------------- |
| `{prefix}.weight`       | float8_e4m3fn | MXFP8 权重                               |
| `{prefix}.weight_scale` | uint8         | block-wise scale（导出时常见 **+127 偏移**后存储） |
| `{prefix}.bias`         | float32       | 原始浮点偏置（可选）                             |

#### W4A4_MXFP4

##### <span id="desc-w4a4-mxfp4">quant_model_description.json</span>

| 描述键                     | 取值             | 说明               |
| ----------------------- | -------------- | ---------------- |
| `{prefix}.weight`       | `"W4A4_MXFP4"` | 打包 FP4 权重        |
| `{prefix}.weight_scale` | `"W4A4_MXFP4"` | block-wise scale |
| `{prefix}.bias`         | `"FLOAT"`      | 偏置（可选）           |

##### <span id="st-w4a4-mxfp4">quant_model_weight*.safetensors</span>

| 张量名                     | 数据类型    | 说明                               |
| ----------------------- | ------- | -------------------------------- |
| `{prefix}.weight`       | uint8   | FP4 打包存储                         |
| `{prefix}.weight_scale` | uint8   | block-wise scale（常见 **+127 偏移**） |
| `{prefix}.bias`         | float32 | 原始浮点偏置（可选）                       |

#### W4A4_MXFP4_DUALSCALE

##### <span id="desc-mxfp-dualscale">quant_model_description.json</span>

在 [W4A4_MXFP4 描述键](#desc-w4a4-mxfp4) 基础上，取值均为 `"W4A4_MXFP4_DUALSCALE"`，并增加：

| 描述键                          | 取值                       | 说明        |
| ---------------------------- | ------------------------ | --------- |
| `{prefix}.weight_dual_scale` | `"W4A4_MXFP4_DUALSCALE"` | 第二路 scale |

##### <span id="st-mxfp-dualscale">quant_model_weight*.safetensors</span>

| 张量名                          | 数据类型    | 说明                        |
| ---------------------------- | ------- | ------------------------- |
| `{prefix}.weight`            | uint8   | FP4 打包存储                  |
| `{prefix}.weight_scale`      | uint8   | 第一路 scale（常见 **+127 偏移**） |
| `{prefix}.weight_dual_scale` | float32 | 第二路 scale                 |
| `{prefix}.bias`              | float32 | 原始浮点偏置（可选）                |

#### FAQuant

##### <span id="desc-faquant">quant_model_description.json</span>

| 描述键 / 全局字段                        | 取值          | 说明                 |
| --------------------------------- | ----------- | ------------------ |
| `{prefix}.scale`                  | `"FAQuant"` | FA per-head scale  |
| `{prefix}.offset`                 | `"FAQuant"` | FA per-head offset |
| `fa_quant_type` / 层级 `quant_type` | 由 FA3 策略拼装  | 启用 FA3 等时写入        |

##### <span id="st-faquant">quant_model_weight*.safetensors</span>

| 张量名               | 数据类型           | 说明                         |
| ----------------- | -------------- | -------------------------- |
| `{prefix}.scale`  | float32        | FA 量化 scale                |
| `{prefix}.offset` | int8 或 float32 | 随 INT8 / FP8 per-head 路径而定 |

### 2.6 适用场景与限制

#### 2.6.1 适用场景

- Wan2.2 等已接入的多模态生成模型量化并交付 MindIE。
- 需要与 `multimodal_sd_modelslim_v1` 的 `inference_config` / dump 配置一并使用的导出场景。

#### 2.6.2 使用限制

- 不替代 AscendV1 作为通用 LLM 默认导出格式；未实现 handler 的模式会提示改用 AscendV1。
- 产物文件命名与字段随模型适配器演进，部署前须按目标 MindIE 版本核对。
- 本词条不展开量化模式原理、反量化公式与 NPU 算子说明（见《[量化模式](../../quantization_mode/term_quantization_mode.md)》）。

## 3. 关联流程

| 流程                                                                    | 说明                 |
| --------------------------------------------------------------------- | ------------------ |
| 《[MindIE-SD 使用指南](mindie_sd_usage.md)》                                | 确认模式支持、配置与执行 |
| 《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》         | multimodal_sd 配置详解 |
| 《[多模态生成模型接入](../../model/integrating_multimodal_generation_model.md)》 | 模型接入与示例            |
| 《[量化格式接入指南](../iformat_integration_guide.md)》                         | 新格式开发对照            |

## 4. 关联词条

- [量化格式](../README.md)：上位概念，本词条所属目录。
- [AscendV1](../ascendv1/term_ascendv1.md)：其他，同属量化格式的并列落盘协议；未实现 handler 的模式可改用 AscendV1。
- [compressed-tensors](../compressed_tensors/term_compressed_tensors.md)：其他，同属量化格式的并列落盘协议。
- [量化模式](../../quantization_mode/term_quantization_mode.md)：配套术语，本格式交付件枚举对应各量化模式；详见本页「[量化模式支持情况](#mode-support)」。
