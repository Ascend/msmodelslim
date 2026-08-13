# Diffusion Transformer（DiT）量化术语百科词条

> **词条类别**：[量化基础概念](../README.md)
> **英文名称**：Diffusion Transformer Quantization
> **英文缩写**：DiT Quantization
> **应用领域**：多模态生成模型（文生图/文生视频）推理加速
> **msModelSlim 实现**：`msmodelslim/core/quant_service/multimodal_sd_v1/`

---

## 1. 概述

Diffusion Transformer（DiT）量化是指对以 Diffusion Transformer 为主干的文生图 / 文生视频模型（如 FLUX.1、HunyuanVideo、Wan2.2 等）进行[训练后量化（PTQ）](../README.md)的过程。与[LLM 量化](../llm/term_large_language_model_quantization.md)依赖文本校准数据集不同，DiT 量化通过运行完整的浮点去噪推理管线生成校准数据，并针对多专家（multi-expert）模型按专家分别量化。DiT 量化的核心挑战在于去噪过程的多次前向传播带来校准数据生成的高成本，以及不同去噪步长下激活分布的非平稳性。

---

## 2. 背景与动机

DiT 模型在文生图（FLUX.1、SD3）和文生视频（HunyuanVideo、Wan2.2）等领域取得了显著进展，但参数量大、推理延迟高，亟需量化压缩。与自回归的 LLM 不同，DiT 的去噪过程需要多次迭代前向传播，量化收益更加显著。此外，部分 DiT 模型（如 Wan2.2）采用多专家架构（low_noise 和 high_noise 专家），各专家需分别量化，进一步增加了量化方案的复杂度。因此，针对 DiT 架构特性设计专门的量化方案具有重要的工程价值。

---

## 3. 原理

### 3.1 核心思想

DiT 量化的核心思想是利用完整去噪推理管线产生的激活 dump 作为校准数据，针对 DiT 架构的独特处理器链（online_quarot + fa3_quant + linear_quant）完成量化。其关键路径包括：

- **去噪管线校准**：运行浮点推理管线，捕获各层在不同去噪步长下的输入输出激活，作为校准数据。
- **多专家量化**：对多专家 DiT 模型（如 Wan2.2），每个专家独立 dump 校准数据、独立量化、独立保存。
- **Timestep 感知**：校准数据覆盖不同去噪步长（timestep），确保量化参数适应分布变化。
- **处理器链**：`online_quarot`（在线 QuaRot 旋转）→ `fa3_quant`（FA3 激活量化）→ `linear_quant`（线性层 MXFP8 量化）。

### 3.2 数学描述

DiT 量化中线性层的量化映射与标准均匀量化一致：

$$ \hat{x} = \text{round}\left(\frac{x}{\Delta}\right) + z $$

其中：

- $x$：原始浮点值
- $\hat{x}$：量化后的整数值
- $\Delta$：缩放因子
- $z$：零点偏移

DiT 特有的 per_block 量化（MXFP8）将权重按 block 粒度分组，每组独立计算缩放因子。对于多专家量化，每个专家 $e$ 独立统计其激活分布，得到各自的量化参数 $(\Delta_e, z_e)$。

此外，DiT 的 FA3 激活量化采用 per_token 粒度的 FP8（fp8_e4m3）格式，对注意力模块的激活值按 token 独立计算缩放因子，以适配去噪过程中激活分布的非平稳性。

### 3.3 关键性质

- **去噪管线校准**：校准数据通过运行完整浮点推理管线生成，而非使用文本数据集，耗时较长。
- **多专家量化**：多专家模型按专家名分别 dump 校准数据和量化，各专家输出独立子目录。
- **Timestep 感知**：校准数据覆盖不同去噪步长，适应激活分布随去噪过程的非平稳性。
- **处理器链特异**：`online_quarot` + `fa3_quant` + `linear_quant` 串联，无 KV cache 量化。
- **per_block 量化粒度**：线性层采用 MXFP8 per_block 量化，权重与激活均按 block 分组独立统计缩放因子。
- **仅支持单卡 LAYER_WISE**：DiT 量化不支持分布式 DP 或 MODEL_WISE runner。
- **输出格式唯一**：量化结果仅支持 MindIE-SD 格式，面向昇腾 NPU 推理栈。

---

## 4. 流程示意

> 详细算法步骤与代码级解析，请参阅 [《多模态生成模型接入》](../../model/integrating_multimodal_generation_model.md)。

```mermaid
flowchart LR
    A[浮点推理配置] --> B[运行去噪管线]
    B --> C[Dump 校准激活]
    C --> D[多专家分别量化]
    D --> E[处理器链: QuaRot + FA3 + LinearQuant]
    E --> F[MindIE-SD 格式落盘]
```

---

## 5. 在 msModelSlim 中的实现

> 关于该能力在 msModelSlim 中的完整实现细节与代码架构解析，请参阅 [《多模态生成模型接入》](../../model/integrating_multimodal_generation_model.md)。

### 5.1 实现位置

DiT 量化在 msModelSlim 中由独立的量化服务实现，位于 `msmodelslim/core/quant_service/multimodal_sd_v1/` 目录，对应 `apiversion: multimodal_sd_modelslim_v1`。

### 5.2 处理流程

- 加载推理配置（`inference_config` 或 `model_config`），确定推理参数。
- 运行浮点推理管线，dump 校准数据（按专家命名为 pth 文件）。
- 按专家初始化模型，逐专家执行量化处理器链。
- 保存为 MindIE-SD 格式的量化权重。

### 5.3 配置示例

以下为最小可用的 YAML 配置片段（以 HunyuanVideo W8A8 MXFP8 为例）。各字段的详细含义与取值范围，请参阅[《量化配置协议详解》](../../../user_guide/usage_quick_quantization.md#53-multimodal_sd_modelslim_v1-配置详解)。

```yaml
apiversion: multimodal_sd_modelslim_v1
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp8"
        weight:
          scope: "per_block"
          dtype: "mxfp8"
      include:
        - "*"
    - type: "online_quarot"
      include: ["*"]
    - type: "fa3_quant"
      qconfig:
        dtype: "fp8_e4m3"
        scope: "per_token"
      include:
        - "*"
  dataset: hunyuanvideo
  multimodal_sd_config:
    dump_config:
      enable_dump: False
      capture_mode: "args"
    inference_config:
      model_resolution: "720p"
      video_size: [720, 1280]
      infer_steps: 50
```

---

## 6. 适用场景与限制

### 6.1 适用场景

- **文生图模型量化部署**：FLUX.1、SD3 等文生图模型的推理加速。
- **文生视频模型量化部署**：HunyuanVideo、Wan2.2 等文生视频模型的推理加速。
- **图生视频 / 图像编辑模型量化**：Wan2.2 I2V、Qwen-Image-Edit 等模型的量化部署。

### 6.2 使用限制

- **校准数据生成耗时**：需运行完整浮点推理管线，校准数据生成时间长。
- **仅支持单卡量化**：不支持分布式 DP 或 MODEL_WISE 量化。
- **多专家校准数据量大**：各专家需分别 dump 校准数据，存储和计算开销大。
- **输出格式唯一**：仅支持 MindIE-SD 格式，不支持 HuggingFace 格式落盘。

---

## 7. 关联流程

- [《DiT 量化使用指南》](./usage_diffusion_transformer_quantization.md)：DiT 量化完整的操作流程与命令说明，使用本词条所述的量化方法。
- [《DiT 新模型接入案例》](./integration_guide_diffusion_transformer_quantization.md)：将新多模态生成模型接入量化流程的完整开发指南。
- [《一键量化完整指南》](../../../user_guide/usage_quick_quantization.md)：涵盖多模态生成模型的一键量化流程，默认集成 DiT 量化方案。

---

## 8. 关联词条

- [LLM 量化](../llm/term_large_language_model_quantization.md)：对比算法，自回归语言模型的量化，与 DiT 的去噪管线校准策略不同。
- [VLM 量化](../vlm/term_vision_transformer_quantization.md)：同类概念，多模态理解模型的量化，同样涉及视觉 Transformer 编码器。
- [ViT 量化](../vlm/term_vision_transformer_quantization.md)：配套术语，DiT 与 VLM 的视觉编码器均含视觉 Transformer 结构。
- [权重转换](../convert/term_weight_conversion.md)：配套术语，对已有量化权重做格式/精度变换。
- [PTQ](../README.md)：上位概念，DiT 量化属于训练后量化的一种具体应用。

---

## 9. 参考资料

1. [《多模态生成模型接入》](../../model/integrating_multimodal_generation_model.md)：DiT 模型接入与量化架构说明。
2. [《msModelSlim 一键量化完整指南》](../../../user_guide/usage_quick_quantization.md)：一键量化总体流程。
3. Peebles W, Xie S. Scalable Diffusion Models with Transformers. ICCV 2023. https://arxiv.org/abs/2212.09748
