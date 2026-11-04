# 权重转换量化术语百科词条

> **词条类别**：量化工具接口<br>
> **英文名称**：Weight Conversion<br>
> **应用领域**：已有量化权重的格式/精度变换、离线部署

本文档介绍权重转换（Weight Conversion）这一离线、data-free 量化工具能力，涵盖 IR 抽象、自动路由、虚拟模块树与 7 阶段流水线等核心机制，并给出适用的转换场景与使用限制，适用于需要对已有量化权重做格式或精度变换的开发者与部署人员。

---

## 1. 概述

权重转换（Weight Conversion）是 msModelSlim 中的离线、data-free 量化工具能力，能在不加载模型代码、不依赖校准数据集的前提下，对已有的 checkpoint 做格式或精度变换。与需要校准数据的常规[训练后量化（PTQ）](../term_ptq.md)不同，权重转换直接操作张量级 IR（Intermediate Representation），通过自动路由机制选择最短转换路径，适用于 FP8 block 反量化、BF16 转 MXFP8、INT4 分组反量化等场景。权重转换通常作为[LLM 量化](../llm/term_large_language_model_quantization.md)的后续步骤，用于调整已有量化权重的落盘格式以适配不同推理框架。

## 2. 词条介绍

在实际部署中，常遇到已有量化权重（如 FP8 block、INT4 packed 格式）需要转换为其他精度或落盘格式的场景。例如：FP8 权重需反量化至 BF16 以在 HuggingFace 生态推理；BF16 浮点权重需离线量化为 W8A8 MXFP8 以部署到昇腾 NPU。常规一键量化需要加载模型代码和校准数据集，流程繁琐且可能引入额外精度损失。权重转换提供了一种轻量级、无模型代码依赖的纯数据变换路径，大幅简化了格式适配工作。

权重转换基于 IR（Intermediate Representation）概念，将 checkpoint 中的张量抽象为 IR 节点，通过注册的 IR 图自动选择最短转换路径，核心包括 IR 抽象、自动路由、虚拟模块树和 7 阶段流水线等机制。

## 3. 核心原理

### 3.1 数学描述

权重转换涉及以下三种核心数学映射：

#### 3.1.1 FP8 block 反量化（无损）

$$ x_{\text{bf16}} = x_{\text{fp8}} \cdot s_{\text{scale\_inv}} $$

其中 $x_{\text{fp8}}$ 为 FP8 block 格式的权重，$s_{\text{scale\_inv}}$ 为 block 级缩放因子的倒数。

#### 3.1.2 INT4 packed 反量化（无损）

$$ x_{\text{bf16}} = (x_{\text{int4}} - z) \cdot \Delta $$

其中 $x_{\text{int4}}$ 为分组 INT4 packed 包装格式的权重，$\Delta$ 为分组缩放因子，$z$ 为分组零点。

#### 3.1.3 FLOAT 到 MXFP8 量化（有损）

$$ \hat{x} = \text{round}\left(\frac{x}{\Delta_{\text{block}}}\right) $$

其中 $\Delta_{\text{block}}$ 为 block 级缩放因子，MXFP8 格式按 block 粒度独立统计。

### 3.2 关键性质

- **Data-free 特性**：无需校准数据集，无需加载模型代码，纯权重离线计算。
- **无损转换**：`FP8_BLOCK → FLOAT`、`INT4_PACKED → FLOAT`、`MXFP4_PACKED → FLOAT`，以及满足条件时的 `FP8_BLOCK → W8A8_MXFP8` 直转。
- **有损转换**：`FLOAT → W8A8_MXFP8` 与 `FLOAT → W4A4_MXFP4` 为有损转换（重新量化引入量化误差）。不满足直转条件的 `FP8_BLOCK → W8A8_MXFP8` 先反量化再量化，后半段有损。
- **自动路由**：`route: auto` 时自动推断源 IR 并选择转换路径，支持多步串联。
- **7 阶段流水线**：从读 catalog 到落盘，全流程无需模型 forward。
- **多种保存格式**：`ascend_v1`（MXFP8 部署）、`huggingface`（HF 生态）。
- **NPU / CPU**：NPU 用 `--device npu --device_id`；CPU 用 `parallel.cpu_workers`。

## 4. 流程示意

> 详细算法步骤与代码级解析，请参阅《[权重转换使用指南](./usage_weight_conversion.md)》。

### 4.1 处理流程

```mermaid
flowchart LR
    A[源权重 Checkpoint] --> B[读 catalog 与预处理]
    B --> C[虚拟树构建与 IR 路由]
    C --> D[IR 转换执行]
    D --> E[保存为指定格式]
```

- **读 catalog**：扫描 checkpoint 文件，获取所有 tensor key 与 dtype 信息。
- **预处理**：可选，对 checkpoint key 做结构性变换（如 fused gate/up 拆分）。
- **建虚拟模块树**：按模块层级构建 `ModelFreeModule` 树。
- **建 IR 任务**：为每个匹配的线性层创建 IR 转换任务。
- **路由**：按 `route` 配置自动选择转换路径。
- **执行**：调用对应的 IR 转换处理器（`Fp8BlockToFloatProcessor`、`Fp8BlockToMxFp8Processor`、`Int4PackedToFloatProcessor`、`MxFp4PackedToFloatProcessor`、`MxFp8QuantProcessor`、`MxFp4QuantProcessor`）。
- **保存**：按配置的 `save.type` 落盘（`ascend_v1` 或 `huggingface`）。

### 4.2 配置示例

以下为最小可用的 YAML 配置片段（以 FP8 block 转 BF16 为例）。各字段的详细含义与取值范围，请参阅《[权重转换使用指南](./usage_weight_conversion.md)》。

```yaml
apiversion: modelslim_convert
spec:
  linears:
    - match:
        - "model.layers.*.self_attn.q_proj"
        - "model.layers.*.self_attn.k_proj"
        - "model.layers.*.self_attn.v_proj"
        - "model.layers.*.self_attn.o_proj"
        - "model.layers.*.mlp.gate_proj"
        - "model.layers.*.mlp.up_proj"
        - "model.layers.*.mlp.down_proj"
      target: FLOAT
      route: auto
  save:
    - type: huggingface
      part_file_size: 4
  parallel:
    cpu_workers: 8
```

## 5. 适用场景与限制

### 5.1 适用场景

- **FP8 block 反量化**：将 FP8 量化权重反量化为 BF16，以在 HuggingFace 生态推理。
- **BF16 离线 MXFP8 量化**：将 BF16 浮点权重离线量化为 W8A8 MXFP8，供昇腾 NPU 推理。
- **FP8 转 MXFP8**：`route: auto` 在 E4M3 + E8M0 且 FP8 block 的列数是 32 的整数倍时走无损直转 `FP8_BLOCK → W8A8_MXFP8`，否则串联 `FP8_BLOCK → FLOAT → W8A8_MXFP8`。
- **INT4 packed 反量化**：将 INT4 分组量化权重反量化为 BF16。

### 5.2 使用限制

- **仅转换线性层**：仅 `linears.match` 中匹配的模块参与转换，未匹配权重原样保留。
- **不能替代常规量化**：无法进行需要校准的量化（如 W8A8 INT8 需激活统计）。
- **MXFP8 格式唯一**：MXFP8 必须用 `ascend_v1` 落盘，不支持 HuggingFace 格式。
- **无模型代码加载**：无法处理需要模型结构理解的结构变换。

## 6. 关联流程

- 《[权重转换使用指南](./usage_weight_conversion.md)》：权重转换完整的操作流程、命令说明与详细配置协议。
- 《[msmodelslim quant 命令行](../../../api_reference/cli/msmodelslim_quant.md)》：含可复制的权重转换命令。
- 《[msModelSlim 一键量化使用指南](../../../user_guide/usage_one_click_quantization.md)》：常规一键量化（含校准）的完整流程。

## 7. 关联词条

- 《[大语言模型量化](../llm/term_large_language_model_quantization.md)》：前置术语，权重转换通常是对 LLM 量化权重的后续处理。
- 《[视觉语言模型量化](../vlm/term_vision_transformer_quantization.md)》：应用对象，VLM 量化权重也可通过转换调整格式。
- 《[Diffusion Transformer 量化](../dit/term_diffusion_transformer_quantization.md)》：应用对象，DiT 量化权重也可通过转换调整格式。
- 《[训练后量化（PTQ）](../term_ptq.md)》：上位概念，权重转换属于 PTQ 生态中的工具链环节。

## 8. 参考文档

1. 《[权重转换使用指南](./usage_weight_conversion.md)》：msModelSlim 权重转换详细文档。
2. 《[msModelSlim 一键量化使用指南](../../../user_guide/usage_one_click_quantization.md)》：一键量化总体流程。
3. 《[格式支持矩阵](../../quantization_format/README.md)》：量化格式与存储格式说明。
