# GLM-4.6V 量化说明

## 模型介绍

[GLM-4.6V](https://z.ai/blog/glm-4.6v) 是智谱多模态大语言模型的最新迭代版本。GLM-4.6V（106B），一款专为云及高性能集群场景设计的基础模型。
GLM-4.6V在训练中将上下文窗口扩展到128k个词元，并在相似参数规模的模型中，在视觉理解和推理方面达到了最先进的性能。其集成了原生函数调用能力。这有效地弥合了“视觉感知”和“可执行动作”之间的差距，为现实世界业务场景中的多模态智能体提供了统一的技术基础。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../docs/zh/install_guide/install_guide.md)。
- 针对 GLM-4.6V，transformers 版本需为 5.0.0rc0：

  ```bash
  pip install transformers==5.0.0rc0
  ```

## GLM-4.6V 模型当前已验证的量化方法

| 模型 | 原始浮点权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|-------------|---------|----------------|---------|
| GLM-4.6V | [GLM-4.6V](https://huggingface.co/zai-org/GLM-4.6V) | W8A8 混合量化（MoE专家动态量化） | MindIE 待支持<br>vLLM Ascend 支持中 | [W8A8 混合量化](#glm-4.6v-w8a8-混合量化) |

**说明：** 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 校准数据说明

校准数据支持的方式，详见 [dataset 配置说明](../../../docs/zh/user_guide/usage_quick_quantization.md#dataset---校准数据路径配置)：

对 GLM-4.6V，校准时每条样本需要提供文本提示词 `text` 和对应的图像`image`，当前缺项的样本不支持。

## 生成量化权重

### <span id="glm-4.6v-w8a8-混合量化">GLM-4.6V W8A8 混合量化</span>

该模型的量化已经集成至[一键量化](../../../docs/zh/user_guide/usage_quick_quantization.md#32-参数说明)。

```shell
msmodelslim quant \
    --model_path /path/to/GLM-4.6V_float_weights \
    --save_path /path/to/GLM-4.6V_quantized_weights \
    --device npu \
    --model_type GLM-4.6V \
    --quant_type w8a8 \
    --trust_remote_code true
```

## 附录

- [multimodal_vlm_modelslim_v1 量化服务配置详解](../../../docs/zh/user_guide/usage_quick_quantization.md#54-multimodal_vlm_modelslim_v1-配置详解)
