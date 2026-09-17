# Qwen3-Omni 量化使用说明

## 模型介绍

Qwen3-Omni 是阿里云 Qwen 团队推出的多模态 Omni 模型，支持语音、图像、视频与文本的多模态理解与生成。当前支持以下规格的 W8A8 量化：

- **Qwen3-Omni-30B-A3B-Thinking**：具备思考链能力，30B 总参数、3B 激活参数 MoE 规格
- **Qwen3-Omni-30B-A3B-Instruct**：指令微调版本，30B 总参数、3B 激活参数 MoE 规格

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../docs/zh/install_guide/install_guide.md)。
- 注意：由于高版本 transformers 的特殊性，PyTorch 及 TorchNPU 需按安装指南配置为兼容版本。
- 针对 Qwen3-Omni，transformers 版本需为 **4.57.3**：

  ```bash
  pip install transformers==4.57.3
  ```

- 需安装 `qwen_omni_utils`（用于多模态数据预处理）：

  ```bash
  pip install qwen_omni_utils
  ```

- 需在环境中**额外安装 ffmpeg**（用于音视频预处理）：

  ```bash
    # Ubuntu
    sudo apt-get update && sudo apt install -y ffmpeg

    # CentOS
    sudo yum install -y ffmpeg

    # 验证ffmpeg安装成功
    ffmpeg -version
  ```

## Qwen3-Omni 模型当前已验证的量化方法

| 模型 | 原始浮点权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|-------------|---------|----------------|---------|
| Qwen3-Omni-30B-A3B-Thinking | [Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) | W8A8 量化 | vLLM Ascend | [W8A8 量化](#qwen3-omni-w8a8) |
| Qwen3-Omni-30B-A3B-Instruct | [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | W8A8 量化 | vLLM Ascend | [W8A8 量化](#qwen3-omni-w8a8) |

**说明：** 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 使用示例

### <span id="qwen3-omni-w8a8">Qwen3-Omni-30B-A3B-Thinking / Qwen3-Omni-30B-A3B-Instruct W8A8 量化</span>

该系列模型的量化已集成至[一键量化](../../../docs/zh/user_guide/usage_quick_quantization.md#32-参数说明)。将 `--model_type` 设为对应模型名称、`--quant_type` 设为 `w8a8` 即可。

**Qwen3-Omni-30B-A3B-Thinking：**

```shell
msmodelslim quant \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --device npu \
    --model_type Qwen3-Omni-30B-A3B-Thinking \
    --quant_type w8a8 \
    --trust_remote_code true
```

**Qwen3-Omni-30B-A3B-Instruct：**

```shell
msmodelslim quant \
    --model_path ${model_path}  \
    --save_path ${save_path} \
    --device npu \
    --model_type Qwen3-Omni-30B-A3B-Instruct \
    --quant_type w8a8 \
    --trust_remote_code true
```

## 附录

### 相关资源

- [一键量化配置协议说明](../../../docs/zh/user_guide/usage_quick_quantization.md#5-量化配置协议详解)
- [multimodal_vlm_modelslim_v1 量化服务配置详解](../../../docs/zh/user_guide/usage_quick_quantization.md#54-multimodal_vlm_modelslim_v1-配置详解)
