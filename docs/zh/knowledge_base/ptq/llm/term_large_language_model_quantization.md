# 大语言模型（LLM）量化术语百科词条

> **词条类别**：[量化基础概念](../README.md)
> **英文名称**：Large Language Model Quantization
> **英文缩写**：LLM Quantization
> **应用领域**：大语言模型推理加速、模型压缩
> **msModelSlim 实现**：`msmodelslim/core/quant_service/modelslim_v1/`

---

## 1. 概述

大语言模型（LLM）量化是指对 Decoder-only 架构的自回归语言模型（如 LLaMA、Qwen、GLM、DeepSeek 等）进行[训练后量化（PTQ）](../README.md)的过程。它将模型权重和激活值从浮点表示映射到低比特整数（如 INT8、INT4）或低精度浮点格式（如 FP8、MXFP8），从而显著降低模型部署时的显存占用和推理延迟。LLM 量化的核心挑战在于自回归推理中激活值离群值的抑制，以及如何在低比特下保持模型原有的生成质量。

---

## 2. 背景与动机

随着 GPT-3、LLaMA 等大语言模型参数规模从数十亿扩展到数千亿，模型部署对显存和算力的需求急剧增长。以 70B 参数模型为例，FP16 权重约需 140GB 显存，远超单张加速卡的容量。通过量化将权重压缩至 INT4 可减少约4倍存储，同时利用硬件低比特计算单元加速推理。LLM 量化因此成为大模型服务化部署的关键技术。

---

## 3. 原理

### 3.1 核心思想

LLM 量化的核心思想是利用校准数据集统计模型权重和激活值的分布特征，确定缩放因子（scale）和零点（zero point），将浮点张量映射到低比特表示。量化过程通常分为两个维度：

- **权重量化**：对模型线性层的权重矩阵做逐通道（per_channel）或逐分组（per_group）量化，压缩模型体积。
- **激活量化**：对前向传播中的激活值做逐张量（per_tensor）或逐 token（per_token）量化，利用整数运算加速推理。

### 3.2 数学描述

标准均匀量化的数学映射为：

$$ \hat{x} = \text{round}\left(\frac{x}{\Delta}\right) + z $$

其中：

- $x$：原始浮点值
- $\hat{x}$：量化后的整数值
- $\Delta$：缩放因子，$\Delta = \frac{x_{\text{max}} - x_{\text{min}}}{2^{b} - 1}$
- $z$：零点偏移
- $b$：量化比特数

反量化过程为：

$$ x' = (\hat{x} - z) \cdot \Delta $$

### 3.3 关键性质

- **压缩比显著**：W8A8（8比特权重和激活）相比 FP16 可减少约2倍存储，W4A16 可减少约4倍权重存储。
- **精度-效率权衡**：量化比特数越低，压缩率越高，但精度损失风险也越大，需要配合离群值抑制等算法。
- **校准数据依赖**：LLM 量化依赖少量（通常 128~512条）校准样本来统计激活分布，校准集质量直接影响量化精度。
- **硬件加速友好**：低比特整数运算在现代 NPU/GPU 上具有更高的吞吐和更低的能耗。

---

## 4. 流程示意

> 详细算法步骤与代码级解析，请参阅 [《LLM 大模型接入指南》](../../model/integrating_models.md)。

```mermaid
flowchart LR
    A[校准数据集] --> B[加载模型]
    B --> C[逐层加载 DecoderLayer]
    C --> D[处理器链: Smooth/Quant/Rotation]
    D --> E[量化权重落盘]
```

---

## 5. 在 msModelSlim 中的实现

> 关于该能力在 msModelSlim 中的完整实现细节与代码架构解析，请参阅 [《LLM 大模型接入指南》](../../model/integrating_models.md)。

### 5.1 实现位置

LLM 量化在 msModelSlim 中由独立的量化服务实现，位于 `msmodelslim/core/quant_service/modelslim_v1/` 目录，对应 `apiversion: modelslim_v1`。处理器链中的各算法模块位于 `msmodelslim/processor/` 目录下，按功能划分为 `anti_outlier`（离群值抑制）、`quant`（线性层量化）、`quarot`（旋转量化）等子模块。

### 5.2 处理流程

- 加载校准数据集（文本 JSONL 格式，128~512条样本）。
- 选择 runner：`MODEL_WISE`、`LAYER_WISE` 或 `DP_LAYER_WISE`。
- 注册处理器链（如 smooth_quant、linear_quant、QuaRot 等）。
- 逐层量化并保存量化权重。

### 5.3 配置示例

以下为最小可用的 YAML 配置片段（以 Qwen3-32B W8A8 动态量化为例）。各字段的详细含义与取值范围，请参阅[《量化配置协议详解》](../../../user_guide/usage_quick_quantization.md#52-modelslim_v1-配置详解)。

```yaml
apiversion: modelslim_v1
spec:
  dataset:
    data_dir: calib_data
  process:
    - type: linear_quant
      ...
```

---

## 6. 适用场景与限制

### 6.1 适用场景

- **大模型推理服务部署**：将大语言模型量化后部署到在线服务，降低单卡显存需求，提升吞吐。
- **边缘设备推理**：在资源受限的设备上运行量化后的轻量级语言模型。
- **多模型共存**：在同一硬件上同时部署多个量化模型，提高资源利用率。

### 6.2 使用限制

- **离群值敏感**：LLM 激活值中存在显著离群值，直接量化可能导致较大精度损失，需要配合离群值抑制算法。
- **低比特精度风险**：INT4 及更低比特量化在部分任务上可能出现明显的质量下降。
- **校准集分布偏移**：校准集与推理数据分布差异较大时，量化效果可能不理想。

---

## 7. 关联流程

- [《LLM 量化使用指南》](./usage_large_language_model_quantization.md)：LLM 量化完整的操作流程与命令说明，使用本词条所述的量化方法。
- [《LLM 新模型接入案例》](./integration_guide_large_language_model_quantization.md)：将新 Decoder-only 模型接入量化流程的完整开发指南。
- [《一键量化完整指南》](../../../user_guide/usage_quick_quantization.md)：涵盖 LLM 与多模态模型的一键量化流程，默认集成 LLM 量化方案。
- [《量化精度调优指南》](../../../user_guide/process_quantization_precision_tuning.md)：当量化后精度不达标时，通过本流程进行调优。

---

## 8. 关联词条

- [VLM 量化](../vlm/term_vision_transformer_quantization.md)：同类概念，多模态理解模型的量化，与 LLM 量化共享相同的量化数学基础。
- [DiT 量化](../dit/term_diffusion_transformer_quantization.md)：同类概念，多模态生成模型的量化，与 LLM 量化在架构和校准策略上存在差异。
- [权重转换](../convert/term_weight_conversion.md)：配套术语，对已有 LLM 量化权重做格式/精度变换，通常作为量化的后续步骤。
- [PTQ](../README.md)：上位概念，LLM 量化属于训练后量化的一种具体应用。

---

## 9. 参考资料

1. [《LLM 大模型接入指南》](../../model/integrating_models.md)：LLM 模型接入与量化架构说明。
2. [《msModelSlim 一键量化完整指南》](../../../user_guide/usage_quick_quantization.md)：一键量化总体流程。
3. Xiao G et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arXiv:2211.10438, 2022. https://arxiv.org/abs/2211.10438
4. Frantar E et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arXiv:2210.17323, 2022. https://arxiv.org/abs/2210.17323
