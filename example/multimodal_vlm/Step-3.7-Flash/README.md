# Step-3.7-Flash 量化使用说明

## 模型介绍

[Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash) 是 StepFun AI 开源的视觉语言模型，采用稀疏专家混合（MoE）架构，总参数量约 198B，由 196B 语言骨干与 1.8B 视觉编码器组成，原生支持图像理解。模型针对高频生产场景设计，每个 token 仅激活约 11B 参数，最高吞吐量可达 400 tokens/s，支持 256k 上下文窗口，并提供低、中、高三档可调推理级别，便于在速度、成本与认知深度之间灵活平衡。

Step-3.7-Flash 主要面向需要把感知、检索与推理融合到同一代理工作流的开发者，可一次性解析大体量财报、跨步搜索并交叉验证、或在并发编码代理流水线中保持高吞吐。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../docs/zh/install_guide/install_guide.md)》。
- 针对 Step-3.7-Flash，transformers 版本需 `>=4.57.1,<5.0.0`：

  ```bash
  pip install "transformers>=4.57.1,<5.0.0"
  ```

## Step-3.7-Flash 模型当前已验证的量化方法

| 模型 | 原始浮点权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|-------------|---------|----------------|---------|
| Step-3.7-Flash | [Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash) | W8A8 MXFP8 混合量化（MoE experts） | vLLM Ascend 支持 | [W8A8 MXFP8 混合量化](#step-37-flash-w8a8-mxfp8-混合量化) |

## 生成量化权重

### <span id="step-37-flash-w8a8-mxfp8-混合量化">Step-3.7-Flash W8A8 MXFP8 混合量化</span>

该模型的量化已集成至一键量化，示例参数详见文档《一键量化完整指南》中的"[参数说明](../../../docs/zh/user_guide/usage_quick_quantization.md#32-参数说明)"章节。实践配置见[step3_7_flash_w8a8_mxfp.yaml](../../../lab_practice/step_3_7_flash/step3_7_flash_w8a8_mxfp.yaml)。

```shell
msmodelslim quant \
    --model_path /path/to/Step-3.7-Flash \
    --save_path /path/to/step3_7_flash_w8a8_mxfp \
    --device npu \
    --model_type Step-3.7-Flash \
    --quant_type w8a8 \
    --trust_remote_code True
```

## 附录

### 相关资源

- [一键量化配置协议说明](../../../docs/zh/user_guide/usage_quick_quantization.md#5-量化配置协议详解)。
- [multimodal_vlm_modelslim_v1 量化服务配置详解](../../../docs/zh/user_guide/usage_quick_quantization.md#54-multimodal_vlm_modelslim_v1-配置详解)。
