# 权重转换术语百科词条

> **词条类别**：[量化工具接口](../README.md)
> **英文名称**：Weight Conversion
> **英文缩写**：Convert
> **应用领域**：已有量化权重的格式/精度变换、离线部署
> **msModelSlim 实现**：`msmodelslim/core/quant_service/modelslim_convert/`

---

## 1. 概述

权重转换（Convert）是 msModelSlim 中的离线、data-free 量化工具能力，能在不加载模型代码、不依赖校准数据集的前提下，对已有的 checkpoint 做格式或精度变换。与需要校准数据的常规[训练后量化（PTQ）](../README.md)不同，权重转换直接操作张量级 IR（Intermediate Representation），通过自动路由机制选择最短转换路径，适用于 FP8 block 反量化、BF16 转 MXFP8、INT4 分组反量化等场景。权重转换通常作为[LLM 量化](../llm/term_large_language_model_quantization.md)的后续步骤，用于调整已有量化权重的落盘格式以适配不同推理框架。

---

## 2. 背景与动机

在实际部署中，常遇到已有量化权重（如 FP8 block、INT4 packed 格式）需要转换为其他精度或落盘格式的场景。例如：FP8 权重需反量化至 BF16 以在 HuggingFace 生态推理；BF16 浮点权重需离线量化为 W8A8 MXFP8 以部署到昇腾 NPU。常规一键量化需要加载模型代码和校准数据集，流程繁琐且可能引入额外精度损失。权重转换提供了一种轻量级、无模型代码依赖的纯数据变换路径，大幅简化了格式适配工作。

---

## 3. 原理

### 3.1 核心思想

权重转换基于 IR（Intermediate Representation）概念，将 checkpoint 中的张量抽象为 IR 节点，通过注册的 IR 图自动选择最短转换路径。其核心包括：

- **IR 抽象**：每种张量格式（如 FP8_BLOCK、FLOAT、W8A8_MXFP8、INT4_PACKED）对应一个 IRKind，通过有向边（TransformEdge）定义转换关系。
- **自动路由**：`route: auto` 时，工具根据 checkpoint 中张量的 dtype 与元数据自动推断源 IR，并在 IR 图上执行 Dijkstra 最短路径路由。
- **虚拟模块树**：使用 `ModelFreeModule` 构建虚拟模块树，管理 checkpoint 的模块结构，无需加载模型代码。
- **7 阶段流水线**：读 catalog → 预处理 → 建虚拟树 → 建 IR 任务 → 路由 → 执行 → 保存。

### 3.2 数学描述

权重转换涉及以下三种核心数学映射：

#### 3.2.1 FP8 block 反量化（无损）

$$ x_{\text{bf16}} = x_{\text{fp8}} \cdot s_{\text{scale\_inv}} $$

其中 $x_{\text{fp8}}$ 为 FP8 block 格式的权重，$s_{\text{scale\_inv}}$ 为 block 级缩放因子的倒数。

#### 3.2.2 INT4 packed 反量化（无损）

$$ x_{\text{bf16}} = (x_{\text{int4}} - z) \cdot \Delta $$

其中 $x_{\text{int4}}$ 为分组 INT4 包装格式的权重，$\Delta$ 为分组缩放因子，$z$ 为分组零点。

#### 3.2.3 FLOAT 到 MXFP8 量化（有损）

$$ \hat{x} = \text{round}\left(\frac{x}{\Delta_{\text{block}}}\right) $$

其中 $\Delta_{\text{block}}$ 为 block 级缩放因子，MXFP8 格式按 block 粒度独立统计。

### 3.3 关键性质

- **Data-free 特性**：无需校准数据集，无需加载模型代码，纯权重离线计算。
- **无损转换**：`FP8_BLOCK → FLOAT` 和 `INT4_PACKED → FLOAT` 为无损转换（仅反量化）。
- **有损转换**：`FLOAT → W8A8_MXFP8` 为有损转换（重新量化引入量化误差）。
- **自动路由**：`route: auto` 时自动推断源 IR 并选择最短转换路径，支持多步串联。
- **7 阶段流水线**：从读 catalog 到落盘，全流程无需模型 forward。
- **多种保存格式**：`ascend_v1`（MXFP8 部署）、`huggingface` / `compressed_tensors`（HF 生态）。

---

## 4. 流程示意

> 详细算法步骤与代码级解析，请参阅 [《权重转换完整指南》](./usage_weight_conversion.md)。

```mermaid
flowchart LR
    A[源权重 Checkpoint] --> B[读 catalog 与预处理]
    B --> C[虚拟树构建与 IR 路由]
    C --> D[IR 转换执行]
    D --> E[保存为指定格式]
```

---

## 5. 在 msModelSlim 中的实现

> 关于该能力在 msModelSlim 中的完整实现细节与代码架构解析，请参阅 [《权重转换完整指南》](./usage_weight_conversion.md)。

### 5.1 实现位置

权重转换在 msModelSlim 中由独立的量化服务实现，位于 `msmodelslim/core/quant_service/modelslim_convert/` 目录，对应 `apiversion: modelslim_convert`。IR 转换处理器位于 `msmodelslim/processor/convert/` 目录。

### 5.2 处理流程

- **读 catalog**：扫描 checkpoint 文件，获取所有 tensor key 与 dtype 信息。
- **预处理**：可选，对 checkpoint key 做结构性变换（如 fused gate/up 拆分）。
- **建虚拟模块树**：按 tick-tock 模式组织模块层次，构建 `ModelFreeModule` 树。
- **建 IR 任务**：为每个匹配的线性层创建 IR 转换任务。
- **路由**：按 `route` 配置自动选择转换路径。
- **执行**：调用对应的 IR 转换处理器（`DequantToFloatProcessor`、`Int4PackedToFloatProcessor`、`MxFp8QuantProcessor`）。
- **保存**：按配置的 `save.type` 落盘（`ascend_v1` 或 `huggingface`）。

### 5.3 配置示例

以下为最小可用的 YAML 配置片段（以 FP8 block 转 BF16 为例）。各字段的详细含义与取值范围，请参阅 [《权重转换完整指南》](./usage_weight_conversion.md)。

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
    workers: 8
```

---

## 6. 适用场景与限制

### 6.1 适用场景

- **FP8 block 反量化**：将 FP8 量化权重反量化为 BF16，以在 HuggingFace 生态推理。
- **BF16 离线 MXFP8 量化**：将 BF16 浮点权重离线量化为 W8A8 MXFP8，供昇腾 NPU 推理。
- **FP8 直接转 MXFP8**：自动路由串联 `FP8_BLOCK → FLOAT → W8A8_MXFP8`，一步完成。
- **INT4 packed 反量化**：将 INT4 分组量化权重反量化为 BF16。

### 6.2 使用限制

- **仅转换线性层**：仅 `linears.match` 中匹配的模块参与转换，未匹配权重原样保留。
- **不能替代常规量化**：无法进行需要校准的量化（如 W8A8 INT8 需激活统计）。
- **MXFP8 格式唯一**：MXFP8 必须用 `ascend_v1` 落盘，不支持 HuggingFace 格式。
- **无模型代码加载**：无法处理需要模型结构理解的结构变换。

---

## 7. 关联流程

- [《权重转换使用指南》](./usage_weight_conversion.md)：权重转换完整的操作流程与命令说明。
- [《权重转换完整指南》](./usage_weight_conversion.md)：权重转换的详细配置协议与使用示例。
- [《一键量化完整指南》](../../../user_guide/usage_quick_quantization.md)：常规一键量化（含校准）的完整流程。

---

## 8. 关联词条

- [LLM 量化](../llm/term_large_language_model_quantization.md)：前置术语，权重转换通常是对 LLM 量化权重的后续处理。
- [VLM 量化](../vlm/term_vision_transformer_quantization.md)：应用对象，VLM 量化权重也可通过转换调整格式。
- [DiT 量化](../dit/term_diffusion_transformer_quantization.md)：应用对象，DiT 量化权重也可通过转换调整格式。
- [PTQ](../README.md)：上位概念，权重转换属于 PTQ 生态中的工具链环节。

---

## 9. 参考资料

1. [《权重转换完整指南》](./usage_weight_conversion.md)：msModelSlim 权重转换详细文档。
2. [《msModelSlim 一键量化完整指南》](../../../user_guide/usage_quick_quantization.md)：一键量化总体流程。
3. [《格式支持矩阵》](../../quantization_format/README.md)：量化格式与存储格式说明。
