# InternVL 3.5 MoE 量化说明

## 模型介绍

- [InternVL 3.5](https://github.com/OpenGVLab/InternVL) 是由上海人工智能实验室推出的多模态大模型书生·万象系列的最新版本。InternVL 3.5 通过级联强化学习（Cascade RL）、视觉分辨率路由器（ViR）与解耦视觉语言部署（DvD）等创新，显著提升了推理能力、部署效率与通用能力。其中 InternVL3.5-241B-A28B 采用稀疏 MoE 架构，总参数量 241B，激活参数量 28B，在开源多模态大模型中取得了领先的通用多模态、推理、文本和智能任务成绩。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../docs/zh/install_guide/install_guide.md)》。

- 需安装依赖包：

  ```bash
  pip install timm==1.0.27 transformers==4.57.6
  ```

## InternVL 3.5 MoE 模型当前已验证的量化方法

| 模型 | 原始浮点权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|-------------|---------|----------------|---------|
| InternVL3_5-241B-A28B | [InternVL3_5-241B-A28B](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B) | W8A8 混合量化 | vLLM Ascend | [W8A8 混合量化](#internvl3_5-241b-a28b-w8a8混合量化) |

> [!NOTE]
>
> - 点击量化命令列中的链接可跳转到对应的具体量化命令。
> - W8A8 混合量化：Attention 和常规 MLP 层使用静态量化，MoE experts 使用动态量化。

## 使用示例

### <span id="internvl3_5-241b-a28b-w8a8混合量化">InternVL3_5-241B-A28B W8A8 混合量化</span>

该模型的量化已集成至一键量化，示例参数详见文档《一键量化完整指南》中的“[参数说明](../../../docs/zh/user_guide/usage_quick_quantization.md#32-参数说明)”章节。

```shell
msmodelslim quant \
    --model_path {model_path} \
    --save_path {save_path} \
    --device npu \
    --model_type InternVL3_5-241B-A28B \
    --quant_type w8a8 \
    --trust_remote_code true
```

## 附录

### 相关资源

- [一键量化配置协议说明](../../../docs/zh/user_guide/usage_quick_quantization.md#5-量化配置协议详解)
- [multimodal_vlm_modelslim_v1 量化服务配置详解](../../../docs/zh/user_guide/usage_quick_quantization.md#54-multimodal_vlm_modelslim_v1-配置详解)
