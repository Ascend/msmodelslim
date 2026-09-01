# AscendV1 量化格式 量化术语百科词条

> **词条类别**：[量化格式](../README.md)<br>
> **英文名称**：AscendV1 Quantization Format<br>
> **应用领域**：昇腾 NPU 推理、大语言模型量化压缩、多模态理解模型量化压缩<br>
> **msModelSlim 实现**：[`msmodelslim/format/ascendV1_format/`](../../../../../msmodelslim/format/ascendV1_format/)

---

## 1. 概述

AscendV1 是 msModelSlim 面向昇腾 NPU 推理的[量化格式 量化术语百科词条](../README.md)。推理框架（vLLM Ascend、SGLang、MindIE）通过 `quant_model_description.json` 识别各张量的量化类型，并从 `quant_model_weights*.safetensors` 加载对应参数。它解决昇腾侧量化权重的统一落盘与加载问题。核心特征是覆盖多种量化模式枚举，并与 vLLM Ascend、SGLang、MindIE 加载路径对齐。

配置、执行与部署步骤见《[AscendV1 使用指南](ascendv1_usage.md)》。各量化模式的原理与公式见《[量化模式](../../quantization_mode/README.md)》及下文支持表中的词条链接。

---

## 2. 词条介绍

### 2.1 原理

**核心思想**

AscendV1 本质上是一套**昇腾推理侧的量化模型落盘约定**：它不执行校准或伪量化计算，而是在一键量化流水线末尾，将已量化完成的 QIR 模块转换为推理框架可直接加载的两类交付件——**量化描述文件** `quant_model_description.json` 与 **量化权重文件** `quant_model_weights*.safetensors`。前者作为张量级索引，供 vLLM Ascend、SGLang、MindIE 识别各张量名称及其量化模式（如 `W8A8`、`W4A4_MXFP4`）；后者按相同键名存放对应的 int8 / FP8 / scale / zero-point 等数值。推理侧先读取描述文件以确定加载与算子路径，再按键名从 safetensors 取数，二者键名一一对应，缺一不可。

因此，用户可将 AscendV1 理解为：**msModelSlim 量化结果的昇腾标准导出包**——算法负责量化计算与参数生成，AscendV1 负责权重落盘、元数据描述以及与 vLLM Ascend、SGLang、MindIE 加载路径的对齐。

**关键性质**

- 支持分布式导出与权重分片（`part_file_size`）。
- 可选导出 QuaRot 旋转矩阵。
- 通过描述文件中的枚举值表达对多种量化模式的承载能力。

---

## 3. <span id="export-artifacts">导出产物（交付件）</span>

### 3.1 目录与文件说明

执行一键量化并指定 AscendV1 落盘格式（保存器 `type` 为 `ascendv1_saver`）后，在指定的 `save_path` 目录下生成以下文件：

```bash
├── config.json                            # 原始模型配置文件
├── generation_config.json                 # 原始生成配置文件
├── quant_model_description.json           # 量化权重描述文件
├── quant_model_weights*.safetensors       # 量化权重：不分片为 quant_model_weights.safetensors；分片如 quant_model_weights-00001-of-00003.safetensors … + quant_model_weights.safetensors.index.json
├── tokenizer_config.json                  # 原始分词器配置文件
├── tokenizer.json                         # 原始分词器词汇表
├── {model_type}_best_practice.yaml        # 量化配置协议
├── vocab.json                             # 原始词汇映射文件（部分模型）
└── optional/                              # 可选导出目录（部分算法启用时生成）
    └── quarot.safetensors                 # QuaRot 全局旋转矩阵（启用 export_extra_info 时生成）
```

| 文件名 | 说明 |
|--------|------|
| `config.json` | 原始模型的配置文件，包含模型架构、层数、隐藏维度等关键参数 |
| `generation_config.json` | 原始模型的生成配置文件，包含采样策略、最大生成长度等推理相关参数 |
| `quant_model_description.json` | **量化权重描述文件**，记录每个权重张量的量化类型和元数据；每个张量键对应一个量化类型标识，同一 Linear 层的所有参数（weight、scale 等）共享相同的类型标识 |
| `quant_model_weights*.safetensors` | **量化权重文件**；不分片为 `quant_model_weights.safetensors`；分片为 `quant_model_weights-00001-of-0000N.safetensors` 等形式，并由 `quant_model_weights.safetensors.index.json` 索引 |
| `tokenizer_config.json` | 原始分词器的配置文件，包含特殊 token、词表大小等信息 |
| `tokenizer.json` | 原始分词器的词汇表文件，定义 token 与 ID 的映射关系 |
| `{model_type}_best_practice.yaml` | **量化配置协议文件**，记录本次量化所使用的完整配置信息，参考《[量化配置协议详解](../../../user_guide/usage_quick_quantization.md#5-量化配置协议详解)》 |
| `vocab.json` | 原始词汇映射文件，部分模型（如 GPT 风格模型）会包含此文件 |
| `optional/quarot.safetensors` | **可选导出**：QuaRot 全局旋转矩阵（仅在使用 QuaRot 且 `export_extra_info: true` 时生成），见下文可选导出 |

### 3.2 quant_model_description.json

**文件结构示例**

```json
{
  "model_quant_type": "W8A8",
  "version": "1.0.0",
  "group_size": 128,
  "kv_quant_type": "KV8",
  "model.layers.0.self_attn.qkv_proj.weight": "W8A8",
  "model.layers.0.self_attn.o_proj.weight": "W8A8",
  "model.layers.0.mlp.gate_proj.weight": "W8A8",
  "model.layers.0.mlp.up_proj.weight": "W8A8",
  "model.layers.0.mlp.down_proj.weight": "W8A8",
  "metadata": {},
  "optional": {}
}
```

> 上例中 `*.weight` 字段名称由模型本身决定。

**<span id="global-metadata">全局元数据字段</span>**

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `model_quant_type` | string | 模型整体量化类型（混合量化时取优先级最高者，见 [混合量化优先级](#quant-type-priority)） |
| `version` | string | 格式版本，当前 `"1.0.0"` |
| `group_size` | int | 分组量化时的 group 大小 |
| `kv_quant_type` / `kv_cache_type` | string | KV Cache 量化类型 |
| `fa_quant_type` | string | Flash Attention 量化类型 |
| `reduce_quant_type` | string | 通信量化类型 |
| `metadata` | object | 扩展元数据（如 QuaRot 在线旋转描述） |
| `optional` | object | 可选导出件（如 QuaRot 全局旋转矩阵路径） |

其余键值对为 `{张量名}: {量化类型}`，例如 `"model.layers.0.self_attn.q_proj.weight": "W8A8"`。

**<span id="quant-type-priority">混合量化时 `model_quant_type` 优先级</span>**

列表越靠后优先级越高（低比特优先；同比特内 `W8A8` 优先于 `W8A8_DYNAMIC` / `W8A8_MIX`）。`W4A8_DYNAMIC` 不参与选取。优先级由低到高：

1. `FLOAT`
2. `W16A16S`
3. `W8A16`
4. `W8A8_DYNAMIC`
5. `W8A8_MIX`
6. `W8A8`
7. `WFP8AFP8_DYNAMIC`
8. `W8A8_MXFP8`
9. `W4A8_MXFP`
10. `W4A4_DYNAMIC`
11. `W4A4_MXFP4`
12. `W4A4_MXFP4_DUALSCALE`
13. `W4A4_MXFP4_SVD`

### 3.3 <span id="optional-quarot">可选导出：QuaRot 相关文件</span>

当流水线启用 QuaRot 且配置 `export_extra_info: true` 时，AscendV1 可额外写出旋转矩阵文件，并在描述文件中登记路径。算法本身见对应算法词条；此处仅说明**格式侧**落盘约定。

```bash
optional/
└── quarot.safetensors
```

| 键名 | 数据类型 | 说明 |
|------|----------|------|
| `global_rotation` | float32 | 全局旋转矩阵 |

启用 `online` 时，`quant_model_description.json` 包含 `metadata.quarot`（层列表与在线旋转描述），示例如下：

```json
{
  "metadata": {
    "quarot": {
      "heads_rotation": {
        "layers": ["model.layers.0.self_attn.o_proj", "model.layers.1.self_attn.o_proj"]
      },
      "kronecker_rotation": {
        "layers": ["model.layers.0.mlp.down_proj"]
      }
    }
  }
}
```

启用 `export_extra_info` 时，包含：

```jsonc
{
  "optional": {
    "quarot": {
      "rotation_map": {
        "global_rotation": "optional/quarot.safetensors"
      }
    }
  }
}
```

推理框架按 `optional.quarot.rotation_map` 加载矩阵文件。

---

## 4. <span id="engine-support">推理引擎支持情况</span>

下表中的格式枚举**均可被 AscendV1 落盘**；下表描述的是产物能否被目标推理引擎加载部署。口径依据《[大模型支持矩阵](../../model/README.md)》注释与 lab_practice 验证标签；具体模型 × 模式 × 引擎组合以支持矩阵与官方最佳实践为准，并受 CANN / 引擎版本与硬件代际约束。

| 格式枚举值 | vLLM Ascend | SGLang | MindIE | 说明 |
| --- | --- | --- | --- | --- |
| `FLOAT` | √ | √ | √ | 未量化张量，随模型一并加载 |
| `W8A8` | √ | √ | √ | 昇腾A2系列产品 / 昇腾A3系列产品 / 昇腾950PR&950DT系列产品通用入口之一 |
| `W8A8_DYNAMIC` | √ | √ | √ | 昇腾A2系列产品 / 昇腾A3系列产品 / 昇腾950PR&950DT系列产品通用入口之一 |
| `W4A8_DYNAMIC` | √ | √ | √ | 显存更紧时常用 |
| `W4A4_DYNAMIC` | √ | √ | √ | 更低比特 INT 路径 |
| `W8A8_MXFP8` | √ | √ | √ | **推荐昇腾950PR&950DT系列产品**；昇腾A2系列产品 / 昇腾A3系列产品通常不可用 |
| `W4A8_MXFP` | √ | √ | √ | **推荐昇腾950PR&950DT系列产品** |
| `W4A4_MXFP4` | √ | √ | √ | **推荐昇腾950PR&950DT系列产品** |
| `W4A4_MXFP4_DUALSCALE` | √ | √ | √ | **推荐昇腾950PR&950DT系列产品** |
| `WFP8AFP8_DYNAMIC` | √ | — | — | 仅 vLLM Ascend；**推荐昇腾950PR&950DT系列产品** |
| `W8A16` | — | — | √ | 仅 MindIE |
| `W8A8_MIX` | — | — | √ | PD-Mix；仅 MindIE |
| `W16A16S` | — | — | √ | 稀疏量化；仅 MindIE |
| `C8` | — | — | √ | KV Cache；仅 MindIE（含 w8a8c8 / w4a8c8 等组合） |
| `FAQuant` | — | — | √ | FA3 等；仅 MindIE |

> **图例**：`√` 表示该引擎存在可加载路径或已有验证实践；`—` 表示当前不作为该引擎推荐部署路径。选型时先按引擎缩小候选枚举，再在下文 [量化模式支持情况](#mode-support) 核对 AscendV1 交付件字段。

---

## 5. <span id="mode-support">量化模式支持情况</span>

> **交付件说明**：表中交付件：量化描述 JSON对应 `quant_model_description.json`；交付件：量化 safetensors对应 `quant_model_weights*.safetensors`。模式原理见《[量化模式](../../quantization_mode/README.md)》词条，本表不展开反量化公式与 NPU 算子。本词条交付件分两列说明：描述 JSON 键值与 safetensors 张量字段。

| 格式枚举值 | AscendV1 是否支持落盘 | 量化模式词条 | 交付件：量化描述 JSON | 交付件：量化 safetensors |
| --- | --- | --- | --- | --- |
| `FLOAT` | 支持 | [量化模式总览](../../quantization_mode/README.md) | [FLOAT 描述键](#desc-float) | [FLOAT 权重张量](#st-float) |
| `W16A16S` | 支持 | [W16A16S](../../quantization_mode/linear_layer_quantization/term_w16a16s.md) | [W16A16S 描述键](#desc-w16a16s) | [W16A16S 权重张量](#st-w16a16s) |
| `W8A8` | 支持 | [W8A8 静态量化](../../quantization_mode/linear_layer_quantization/term_w8a8_static.md) | [W8A8 描述键](#desc-w8a8) | [W8A8 权重张量](#st-w8a8) |
| `W8A8_DYNAMIC` | 支持 | [W8A8 动态量化](../../quantization_mode/linear_layer_quantization/term_w8a8_dynamic.md) | [W8A8_DYNAMIC 描述键](#desc-w8a8-dynamic) | [W8A8_DYNAMIC 权重张量](#st-w8a8-dynamic) |
| `W8A8_MIX` | 支持 | [W8A8 PD-Mix](../../quantization_mode/linear_layer_quantization/term_w8a8_pdmix.md) | [W8A8_MIX 描述键](#desc-w8a8-mix) | [W8A8_MIX 权重张量](#st-w8a8-mix) |
| `W8A16` | 支持 | [W8A16 静态量化](../../quantization_mode/linear_layer_quantization/term_w8a16_static.md) | [W8A16 描述键](#desc-w8a16) | [W8A16 权重张量](#st-w8a16) |
| `W4A4_DYNAMIC` | 支持 | [W4A4 动态量化](../../quantization_mode/linear_layer_quantization/term_w4a4_dynamic.md) | [W4A4_DYNAMIC 描述键](#desc-w4a4-dynamic) | [W4A4_DYNAMIC 权重张量](#st-w4a4-dynamic) |
| `W4A8_DYNAMIC` | 支持 | [W4A8 动态量化](../../quantization_mode/linear_layer_quantization/term_w4a8_dynamic.md) | [W4A8_DYNAMIC 描述键](#desc-w4a8-dynamic) | [W4A8_DYNAMIC 权重张量](#st-w4a8-dynamic) |
| `WFP8AFP8_DYNAMIC` | 支持 | [W8A8 FP8 动态量化](../../quantization_mode/linear_layer_quantization/term_w8a8_fp8_dynamic.md) | [WFP8AFP8_DYNAMIC 描述键](#desc-wfp8afp8-dynamic) | [WFP8AFP8_DYNAMIC 权重张量](#st-wfp8afp8-dynamic) |
| `W8A8_MXFP8` | 支持 | [W8A8 MX 动态量化](../../quantization_mode/linear_layer_quantization/term_w8a8_mx_dynamic.md) | [W8A8_MXFP8 描述键](#desc-mxfp) | [W8A8_MXFP8 权重张量](#st-mxfp) |
| `W4A8_MXFP` | 支持 | [W4A8 MX 动态量化](../../quantization_mode/linear_layer_quantization/term_w4a8_mx_dynamic.md) | [W4A8_MXFP 描述键](#desc-mxfp) | [W4A8_MXFP 权重张量](#st-mxfp) |
| `W4A4_MXFP4` | 支持 | [W4A4 MX 动态量化](../../quantization_mode/linear_layer_quantization/term_w4a4_mx_dynamic.md) | [W4A4_MXFP4 描述键](#desc-mxfp) | [W4A4_MXFP4 权重张量](#st-mxfp) |
| `W4A4_MXFP4_DUALSCALE` | 支持 | [W4A4 MX 双 Scale](../../quantization_mode/linear_layer_quantization/term_w4a4_mx_dualscale.md) | [W4A4_MXFP4_DUALSCALE 描述键](#desc-mxfp-dualscale) | [W4A4_MXFP4_DUALSCALE 权重张量](#st-mxfp-dualscale) |
| `C8` | 支持 | [KVCache-PerChannel](../../quantization_mode/kv_cache_quantization/term_kv_cache_perchannel.md) | [C8 描述键](#desc-c8) | [C8 权重张量](#st-c8) |
| `FAQuant` | 支持 | [FA 量化](../../quantization_mode/fa_quantization/README.md) | [FAQuant 描述键](#desc-faquant) | [FAQuant 权重张量](#st-faquant) |

---

## 6. 各量化模式交付件格式

约定：

- `{prefix}` 为模块前缀（例如 `model.layers.0.self_attn.q_proj`）。
- **描述 JSON**：文件 `quant_model_description.json`；键为张量全名，值为量化类型字符串（与枚举一致）；全局字段见上文[全局元数据字段](#global-metadata)。
- **safetensors**：文件 `quant_model_weights*.safetensors`（可分片）；键为 `{prefix}.<param>`，存实际数值张量。

### 6.1 FLOAT
**<span id="desc-float">quant_model_description.json</span>**

| 描述键 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | `"FLOAT"` | 未量化权重 |
| `{prefix}.bias` | `"FLOAT"` | 偏置（若存在） |

整体 `model_quant_type` 在仅含 FLOAT 时可为 `"FLOAT"`（混合量化时按 [优先级](#quant-type-priority) 选取）。

**<span id="st-float">`quant_model_weights*.safetensors`</span>**

| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | float16 / bfloat16 | 原始浮点权重 |
| `{prefix}.bias` | float16 / bfloat16 | 偏置（可选） |

### 6.2 W16A16S
**<span id="desc-w16a16s">quant_model_description.json</span>**


| 描述键 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | `"W16A16S"` | 稀疏权重 |
| `{prefix}.scale` | `"W16A16S"` | 稀疏缩放因子 |

**<span id="st-w16a16s">`quant_model_weights*.safetensors`</span>**


| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | float16 / bfloat16 | 稀疏处理后的权重 |
| `{prefix}.scale` | float16 / bfloat16 | 缩放因子 |

### 6.3 W8A8
**<span id="desc-w8a8">quant_model_description.json</span>**

同一 Linear 下下列键共享类型 `"W8A8"`（`bias` 若保留浮点则可标 `"FLOAT"`）：

| 描述键 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | `"W8A8"` | 量化权重 |
| `{prefix}.quant_bias` | `"W8A8"` | 量化偏置 |
| `{prefix}.input_scale` | `"W8A8"` | 激活 scale |
| `{prefix}.input_offset` | `"W8A8"` | 激活 zero-point |
| `{prefix}.deq_scale` | `"W8A8"` | 综合反量化 scale |
| `{prefix}.bias` | `"FLOAT"` 或 `"W8A8"` | 原始浮点偏置（可选） |

**<span id="st-w8a8">`quant_model_weights*.safetensors`</span>**


| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | int8 | 量化权重 |
| `{prefix}.quant_bias` | int32 | 量化偏置 |
| `{prefix}.input_scale` | float32 | 激活量化 scale |
| `{prefix}.input_offset` | float32 | 激活量化 zero-point |
| `{prefix}.deq_scale` | int64 / float32 | 综合反量化 scale（bfloat16 模型多为 float32，否则常按算子约定以 int64 位型存储） |
| `{prefix}.bias` | float32 | 原始浮点偏置（可选） |

### 6.4 W8A8_DYNAMIC
**<span id="desc-w8a8-dynamic">quant_model_description.json</span>**


| 描述键 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | `"W8A8_DYNAMIC"` | 量化权重 |
| `{prefix}.weight_scale` | `"W8A8_DYNAMIC"` | 权重量化 scale |
| `{prefix}.weight_offset` | `"W8A8_DYNAMIC"` | 权重量化 zero-point |
| `{prefix}.bias` | `"FLOAT"` 或 `"W8A8_DYNAMIC"` | 偏置（可选） |

激活动态参数不落盘，故 description 中无 `input_scale` / `input_offset` 等激活静态键。

**<span id="st-w8a8-dynamic">`quant_model_weights*.safetensors`</span>**


| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | int8 | 量化权重 |
| `{prefix}.weight_scale` | float32 | 权重量化 scale |
| `{prefix}.weight_offset` | float32 | 权重量化 zero-point（对称时为 0） |
| `{prefix}.bias` | float32 | 原始浮点偏置（可选） |

激活量化参数在推理时动态计算，**不写入**权重文件。

### 6.5 W8A8_MIX
**<span id="desc-w8a8-mix">quant_model_description.json</span>**


| 描述键 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | `"W8A8_MIX"` | 量化权重 |
| `{prefix}.quant_bias` | `"W8A8_MIX"` | 量化偏置 |
| `{prefix}.input_scale` | `"W8A8_MIX"` | 激活 scale |
| `{prefix}.input_offset` | `"W8A8_MIX"` | 激活 zero-point |
| `{prefix}.deq_scale` | `"W8A8_MIX"` | 综合反量化 scale |
| `{prefix}.weight_scale` | `"W8A8_MIX"` | 权重量化 scale |
| `{prefix}.weight_offset` | `"W8A8_MIX"` | 权重量化 zero-point |
| `{prefix}.bias` | `"FLOAT"` 或 `"W8A8_MIX"` | 偏置（可选） |

**<span id="st-w8a8-mix">`quant_model_weights*.safetensors`</span>**

W8A8 静态激活相关字段与 W8A8_DYNAMIC 权重量化字段的并集：

| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | int8 | 量化权重 |
| `{prefix}.quant_bias` | int32 | 量化偏置 |
| `{prefix}.input_scale` | float32 | 激活量化 scale |
| `{prefix}.input_offset` | float32 | 激活量化 zero-point |
| `{prefix}.deq_scale` | int64 / float32 | 综合反量化 scale |
| `{prefix}.weight_scale` | float32 | 权重量化 scale |
| `{prefix}.weight_offset` | float32 | 权重量化 zero-point |
| `{prefix}.bias` | float32 | 原始浮点偏置（可选） |

### 6.6 W8A16
**<span id="desc-w8a16">quant_model_description.json</span>**


| 描述键 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | `"W8A16"` | 量化权重 |
| `{prefix}.weight_scale` | `"W8A16"` | 权重量化 scale |
| `{prefix}.weight_offset` | `"W8A16"` | 权重量化 zero-point |
| `{prefix}.bias` | `"FLOAT"` 或 `"W8A16"` | 偏置（可选） |

**<span id="st-w8a16">`quant_model_weights*.safetensors`</span>**


| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | int8 | 量化权重 |
| `{prefix}.weight_scale` | float32 | 权重量化 scale |
| `{prefix}.weight_offset` | float32 | 权重量化 zero-point |
| `{prefix}.bias` | float32 | 原始浮点偏置（可选） |

### 6.7 W4A4_DYNAMIC
**<span id="desc-w4a4-dynamic">quant_model_description.json</span>**


| 描述键 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | `"W4A4_DYNAMIC"` | 量化权重（打包） |
| `{prefix}.weight_scale` | `"W4A4_DYNAMIC"` | 权重量化 scale |
| `{prefix}.weight_offset` | `"W4A4_DYNAMIC"` | 权重量化 zero-point |
| `{prefix}.bias` | `"FLOAT"` 或 `"W4A4_DYNAMIC"` | 偏置（可选） |

**<span id="st-w4a4-dynamic">`quant_model_weights*.safetensors`</span>**


| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | int8 | int4 打包存储 |
| `{prefix}.weight_scale` | float32 | 权重量化 scale |
| `{prefix}.weight_offset` | float32 | 权重量化 zero-point |
| `{prefix}.bias` | float32 | 原始浮点偏置（可选） |

激活量化参数推理时动态计算，不写入权重文件。

### 6.8 W4A8_DYNAMIC
**<span id="desc-w4a8-dynamic">quant_model_description.json</span>**


| 描述键 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | `"W4A8_DYNAMIC"` | 量化权重（打包） |
| `{prefix}.weight_scale` | `"W4A8_DYNAMIC"` | 权重量化 scale |
| `{prefix}.weight_offset` | `"W4A8_DYNAMIC"` | 权重量化 zero-point |
| `{prefix}.scale_bias` | `"W4A8_DYNAMIC"` | 反量化额外调整因子 |
| `{prefix}.bias` | `"FLOAT"` 或 `"W4A8_DYNAMIC"` | 偏置（可选） |

**<span id="st-w4a8-dynamic">`quant_model_weights*.safetensors`</span>**


| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | int8 | int4 打包存储 |
| `{prefix}.weight_scale` | float32 | 权重量化 scale |
| `{prefix}.weight_offset` | float32 | 权重量化 zero-point |
| `{prefix}.scale_bias` | float32 | 反量化额外调整因子 |
| `{prefix}.bias` | float32 | 原始浮点偏置（可选） |

### 6.9 WFP8AFP8_DYNAMIC
**<span id="desc-wfp8afp8-dynamic">quant_model_description.json</span>**


| 描述键 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | `"WFP8AFP8_DYNAMIC"` | FP8 权重 |
| `{prefix}.weight_scale` | `"WFP8AFP8_DYNAMIC"` | 权重量化 scale |
| `{prefix}.weight_offset` | `"WFP8AFP8_DYNAMIC"` | 权重量化 zero-point |
| `{prefix}.bias` | `"FLOAT"` 或 `"WFP8AFP8_DYNAMIC"` | 偏置（可选） |

**<span id="st-wfp8afp8-dynamic">`quant_model_weights*.safetensors`</span>**


| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | float8_e4m3fn | FP8 权重 |
| `{prefix}.weight_scale` | float32 | 权重量化 scale |
| `{prefix}.weight_offset` | float32 | 权重量化 zero-point |
| `{prefix}.bias` | float32 | 原始浮点偏置（可选） |

### 6.10 W8A8_MXFP8 / W4A8_MXFP / W4A4_MXFP4
**<span id="desc-mxfp">quant_model_description.json</span>**

描述键取值分别为 `"W8A8_MXFP8"` / `"W4A8_MXFP"` / `"W4A4_MXFP4"`（与具体枚举一致）：

| 描述键 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | 对应 MXFP 枚举 | 量化权重 |
| `{prefix}.weight_scale` | 对应 MXFP 枚举 | block-wise scale |
| `{prefix}.bias` | `"FLOAT"` 或对应枚举 | 偏置（可选） |

**<span id="st-mxfp">`quant_model_weights*.safetensors`</span>**


| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.weight` | float8_e4m3fn 或 uint8（packed fp4） | 量化权重 |
| `{prefix}.weight_scale` | uint8 | block-wise scale（导出时常见 **+127 偏移**后存储，范围 0~255） |
| `{prefix}.bias` | float32 | 原始浮点偏置（可选） |

### 6.11 W4A4_MXFP4_DUALSCALE
**<span id="desc-mxfp-dualscale">quant_model_description.json</span>**

在 [MXFP 描述字段](#desc-mxfp)基础上，取值均为 `"W4A4_MXFP4_DUALSCALE"`，并增加：

| 描述键 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.weight_dual_scale` | `"W4A4_MXFP4_DUALSCALE"` | 第二路 scale |

**<span id="st-mxfp-dualscale">`quant_model_weights*.safetensors`</span>**

在 [MXFP 权重字段](#st-mxfp)基础上额外包含：

| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.weight_dual_scale` | float32 | 第二路 scale |

### 6.12 C8
**<span id="desc-c8">quant_model_description.json</span>**


| 描述键 / 全局字段 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.kv_cache_scale` | `"C8"` | KV Cache scale |
| `{prefix}.kv_cache_offset` | `"C8"` | KV Cache zero-point |
| `kv_quant_type` / `kv_cache_type` | 如 `"C8"` / `"KV8"` | 全局 KV 量化类型（与导出实现一致时写入） |

**<span id="st-c8">`quant_model_weights*.safetensors`</span>**


| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.kv_cache_scale` | float32 / float16 | KV Cache 量化 scale |
| `{prefix}.kv_cache_offset` | float32 / float16 | KV Cache 量化 zero-point |

具体 `{prefix}` 随注意力 KV 相关模块命名而定。

### 6.13 FAQuant
> **命名约定**：本节 `{prefix}` 指**注意力模块**全名（如 `model.layers.0.self_attn`），与上文 Linear 的 `{prefix}`（如 `...self_attn.q_proj`）不同。AscendV1 经 `export_fa_quant_params` 按 Q / K / V **分别**写出 `fa_q` / `fa_k` / `fa_v` 子键，而非单一的 `{prefix}.scale` / `{prefix}.offset`。

**<span id="desc-faquant">quant_model_description.json</span>**


| 描述键 / 全局字段 | 取值 | 说明 |
| --- | --- | --- |
| `{prefix}.fa_q.scale` | `"FAQuant"` | Q 激活量化 scale（仅静态 per-head 路径写出） |
| `{prefix}.fa_q.offset` | `"FAQuant"` | Q 激活量化 zero-point（仅静态 per-head 路径写出） |
| `{prefix}.fa_k.scale` | `"FAQuant"` | K 激活量化 scale（仅静态 per-head 路径写出） |
| `{prefix}.fa_k.offset` | `"FAQuant"` | K 激活量化 zero-point（仅静态 per-head 路径写出） |
| `{prefix}.fa_v.scale` | `"FAQuant"` | V 激活量化 scale（仅静态 per-head 路径写出） |
| `{prefix}.fa_v.offset` | `"FAQuant"` | V 激活量化 zero-point（仅静态 per-head 路径写出） |
| `{prefix}.quant_type` | string | 该注意力层 FA 策略串，由 AscendV1 按 Q/K/V 的 dtype 与 STATIC/DYNAMIC 拼装（如 `INT8`、`FP8_DYNAMIC`） |
| `fa_quant_type` | string | 全局 FA 量化类型（启用 FA 量化时写入，如 `"FAQuant"`） |

动态量化路径（激活 `scope` 为 `per_token` / `per_block`）下，对应激活**不落盘** `scale` / `offset`，仅更新上述 `{prefix}.quant_type`；静态 per-head 路径通常 Q/K/V 六键齐全。该落盘差异由 AscendV1 导出逻辑决定，与《[FA3 Quant](../../quantization_algorithms/fa3_quant/term_fa3_quant.md)》算法配置（`qconfig` / `details`）配合使用。

**<span id="st-faquant">`quant_model_weights*.safetensors`</span>**


| 张量名 | 数据类型 | 说明 |
| --- | --- | --- |
| `{prefix}.fa_q.scale` | float16 / bfloat16 | Q 的 per-head scale |
| `{prefix}.fa_q.offset` | int8 | Q 的 per-head zero-point（导出时转为 int8） |
| `{prefix}.fa_k.scale` | float16 / bfloat16 | K 的 per-head scale |
| `{prefix}.fa_k.offset` | int8 | K 的 per-head zero-point |
| `{prefix}.fa_v.scale` | float16 / bfloat16 | V 的 per-head scale |
| `{prefix}.fa_v.offset` | int8 | V 的 per-head zero-point |

---

## 7. 适用场景与限制

### 7.1 适用场景

- 昇腾侧 vLLM Ascend、SGLang、MindIE 部署 LLM / 多模态理解量化权重。
- 需要在同一套描述文件中承载多种量化类型枚举的落盘。

### 7.2 使用限制

- 不适用于仅面向 HF `quantization_config` / compressed-tensors 的通用 vLLM 路径。
- 具体量化模式是否可用取决于 CANN、推理框架版本与模型最佳实践。

---

## 8. 关联流程

- 《[AscendV1 使用指南](ascendv1_usage.md)》：确认模式支持、配置与执行。
- 《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》：命令与配置协议。
- 《[量化格式接入指南](../iformat_integration_guide.md)》：新格式开发对照。

---

## 9. 关联词条

- [量化格式 量化术语百科词条](../README.md)：上位概念，本词条所属目录。
- [compressed-tensors 量化格式 量化术语百科词条](../compressed_tensors/term_compressed_tensors.md)：其他，同属量化格式的并列落盘协议。
- [MindIE-SD 量化格式 量化术语百科词条](../mindie_sd/term_mindie_sd.md)：其他，同属量化格式的并列落盘协议。
- [量化模式](../../quantization_mode/README.md)：配套术语，本格式交付件枚举对应各量化模式。
