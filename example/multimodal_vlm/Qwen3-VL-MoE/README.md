# Qwen3-VL-MoE 量化使用说明

## 模型介绍

Qwen3-VL-MoE 是阿里云 Qwen 团队推出的大规模多模态视觉语言 Mixture-of-Experts (MoE) 模型，具备以下特点：

- **稀疏 MoE 架构**: 采用稀疏激活的 MoE 结构，在保持高性能的同时显著降低计算成本
- **多模态理解能力**: 支持图像和文本的联合理解，可执行图像描述、视觉问答等多种任务
- **大规模参数**: 提供 30B-A3B 和 235B-A22B 两种规格，其中 "A" 代表激活参数量
- **3D 融合专家权重**: 专家层权重以 3D 张量形式融合存储，需要特殊的量化处理

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../docs/zh/install_guide/install_guide.md)。
- 注意：由于高版本transformers的特殊性，PyTorch及torch_npu需要配置安装为2.7.1版本
- 针对 Qwen3-VL-MoE，transformers 版本需要 4.57.1：

  ```bash
  pip install transformers==4.57.1
  ```

- 还需要安装 flax 依赖：

  ```bash
  pip install flax
  ```

## Qwen3-VL-MoE 模型当前已验证的量化方法

| 模型 | 原始浮点权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|-------------|---------|----------------|---------|
| Qwen3-VL-235B-A22B | [Qwen3-VL-235B-A22B](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/tree/main) | W8A8 混合量化 | MindIE 待支持<br>vLLM Ascend 支持中 | [W8A8 混合量化](#qwen3-vl-moe-w8a8-混合量化) |
| Qwen3-VL-235B-A22B / Qwen3-VL-30B-A3B | [Qwen3-VL-235B-A22B](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/tree/main) / [Qwen3-VL-30B-A3B](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main) | W8A8 + MXFP8 静态量化 | MindIE 待支持<br>vLLM Ascend 支持中 | [W8A8 + MXFP8 静态量化](#qwen3-vl-moe-w8a8-mxfp8) |

注：[Qwen3-VL-30B-A3B](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main) 尚未验证过量化精度，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。

**说明：**

- 点击量化命令列中的链接可跳转到对应的具体量化命令
- W8A8 混合量化：Attention 和常规 MLP 层使用静态量化，MoE experts 使用动态量化

## 量化特性

### MoE 专家层自动转换

- **3D 权重拆分**: 自动将融合的 3D 专家权重 `(num_experts, hidden_size, expert_dim)` 拆分为独立的 `nn.Linear` 层
- **逐层处理**: 结合 v1 框架的逐层加载机制，在加载每一层时自动完成 MoE 转换
- **内存友好**: 转换过程采用 in-place 策略，及时释放原始 3D 权重，大幅降低内存占用

### 异常值抑制 (Iterative Smooth)

- **QuaRot 算法**: 使用基于旋转的离群值抑制算法显著平滑数据的分布，有效降低量化误差
- **iter_smooth 算法**: 使用迭代平滑算法抑制激活值异常点，提升量化精度
- **多种子图类型**: 支持 norm-linear、ov 等多种子图融合
- **自适应配置**: 自动识别 MoE 层结构，为不同层类型应用合适的平滑策略

### 混合量化策略

- **Attention 层**: W8A8 静态量化 (激活per_tensor)，适合激活分布稳定的层
- **MoE Experts**: W8A8 动态量化 (激活per_token)，适应不同 token 的激活差异，保持精度
- **语言部分 MLP Gate 层**: 不进行量化，保持浮点精度，确保专家路由准确性
- **视觉部分 linear_fc2 层**: 精度敏感，不进行量化，保持浮点精度
- **视觉部分 merger、deepstack 层**: 精度敏感，不进行量化，保持浮点精度

### 逐层量化

- **内存优化**: 支持逐层加载、量化、offload 的流程，显著降低显存占用
- **单卡支持**: 结合逐层量化特性，可在 Atlas 800I A2 (64G) 设备上完成大模型量化

## 生成量化权重

## 使用示例

### <span id="qwen3-vl-moe-w8a8-混合量化">Qwen3-VL-235B-A22B W8A8 混合量化</span>

该模型的量化已经集成至《一键量化》文档中的"[参数说明](../../../docs/zh/user_guide/feature_guide/quick_quantization_v1/usage.md#42-参数说明)"章节内容，具体量化命令参考如下：

```shell
msmodelslim quant \
    --model_path /path/to/qwen3_vl_moe_float_weights \
    --save_path /path/to/qwen3_vl_moe_quantized_weights \
    --device npu \
    --model_type Qwen3-VL-235B-A22B \
    --quant_type w8a8 \
    --trust_remote_code True
```

### <span id="qwen3-vl-moe-w8a8-mxfp8">Qwen3-VL-MoE W8A8 + MXFP8 静态量化</span>

该模型的量化已经集成至《一键量化》文档中的"[参数说明](../../../docs/zh/user_guide/feature_guide/quick_quantization_v1/usage.md#42-参数说明)"章节内容，具体量化命令参考如下：

```shell
msmodelslim quant \
    --model_path /path/to/qwen3_vl_moe_float_weights \
    --save_path /path/to/qwen3_vl_moe_quantized_weights \
    --device npu \
    --model_type Qwen3-VL-235B-A22B \
    --quant_type w8a8 \
    --tag vLLM_Ascend Atlas_350 \
    --trust_remote_code True
```

## FAQ

### Q1: 为什么 MoE experts 使用动态量化？

**A**: MoE experts 的激活分布在不同 token 间差异较大：

- **静态量化** (per_tensor): 所有 token 共享一个 scale → 精度损失大
- **动态量化** (per_token): 每个 token 独立 scale → 精度更高

这是 MoE 模型的标准做法，参考 DeepSeek-V3 等模型的最佳实践。

### Q2: 如何自定义校准数据集？

**A**: 有以下几种方式：

1. 使用 `lab_calib/calibImages/` 目录，并统一自定义所有图像的文本prompt：通过yaml配置文件中default_text字段配置文本prompt；
2. 使用 `lab_calib/calibImages/` 目录，并自定义每个图像的文本prompt：在图像目录添加一个JSON/JSONL文件，具体示例可参考[dataset - 校准数据路径配置](../../../docs/zh/user_guide/feature_guide/quick_quantization_v1/usage.md#dataset---校准数据路径配置)；
3. 使用自定义图像目录，并统一自定义所有图像的文本prompt：在yaml配置文件中修改 `dataset` 字段为自定义图像目录，并通过yaml配置文件中default_text字段配置文本prompt；
4. 使用自定义图像目录，并自定义每个图像的文本prompt：在yaml配置文件中修改 `dataset` 字段为自定义图像目录，在图像目录添加一个JSON/JSONL文件，具体示例可参考[dataset - 校准数据路径配置](../../../docs/zh/user_guide/feature_guide/quick_quantization_v1/usage.md#dataset---校准数据路径配置)。

建议使用与实际应用场景相似的图像作为校准集，数量一般不超过30张。

## 附录

### 相关资源

- [multimodal_vlm_modelslim_v1 配置详解](../../../docs/zh/user_guide/feature_guide/quick_quantization_v1/usage.md#62-modelslim_v1-配置详解)
- [一键量化配置协议说明](../../../docs/zh/user_guide/feature_guide/quick_quantization_v1/usage.md#6-量化配置协议详解)
- [逐层量化特性说明](../../../docs/zh/user_guide/feature_guide/quick_quantization_v1/usage.md#51-逐层量化及分布式逐层量化)
- [QuaRot 算法说明](../../../docs/zh/user_guide/quantization_algorithms/outlier_suppression_algorithms/quarot.md)
- [Iterative Smooth 算法说明](../../../docs/zh/user_guide/quantization_algorithms/outlier_suppression_algorithms/iterative_smooth.md)
- [LinearQuantProcess 线性层量化处理器说明](../../../docs/zh/user_guide/quantization_algorithms/quantization_algorithms/linear_quant.md)
