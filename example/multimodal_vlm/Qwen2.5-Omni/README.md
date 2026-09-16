# Qwen2.5-Omni 量化说明

## 模型介绍

[Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) 是端到端多模态模型，可同时感知文本、图像、音频与视频，并以流式方式生成文本与自然语音。主要特点如下：

- **Omni 与 Thinker-Talker 架构**：端到端多模态模型，支持文本、图像、音频、视频的联合感知，并同时以流式生成文本与自然语音；采用 TMRoPE（Time-aligned Multimodal RoPE）位置编码，对齐视频与音频时间戳。
- **实时语音与视频对话**：支持分块输入与即时输出的全实时交互。
- **自然鲁棒的语音生成**：在流式与非流式场景下均表现优异，语音自然度与鲁棒性突出。
- **多模态能力均衡**：在同等规模单模态基准上表现优异，音频能力优于同规模 Qwen2-Audio，视觉能力与 Qwen2.5-VL-7B 相当。
- **端到端语音指令遵循**：在 MMLU、GSM8K 等基准上，语音指令遵循效果与文本输入相当。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../docs/zh/install_guide/install_guide.md)。
- 针对 Qwen2.5-Omni，transformers 版本需为 4.57.3：

  ```bash
  pip install transformers==4.57.3
  ```

- 需要安装 qwen_omni_utils 依赖用于模型本身数据预处理：

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

## Qwen2.5-Omni 模型当前已验证的量化方法

| 模型 | 原始浮点权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|-------------|---------|----------------|---------|
| Qwen2.5-Omni-7B | [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | W8A8 动态量化 | MindIE 待支持<br>vLLM Ascend 支持中 | [W8A8 动态量化](#qwen25-omni-7b-w8a8-量化) |

**说明：** 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 校准数据说明

校准数据支持的方式，详见 [dataset 配置说明](../../../docs/zh/user_guide/usage_quick_quantization.md#dataset---校准数据路径配置)：

对 Qwen2.5-Omni，推荐使用 index.json 或 index.jsonl（文件路径或仅含一个 index.json 或 index.jsonl 的目录），支持多模态字段。校准时每条样本提供 `text` 及与推理场景一致的多模态组合（`image`、`audio`、`video`），当前缺项的样本会被跳过。

`dataset` 可配置为短名称（在 `lab_calib` 等目录下查找）、绝对路径或相对路径。配置示例见 [qwen2_5_omni_thinker_w8a8.yaml](../../../lab_practice/qwen2_5_omni_thinker/qwen2_5_omni_thinker_w8a8.yaml)：`dataset` 指定校准数据集，`default_text` 可设为如 "What are the elements can you see and hear in this media." 等多模态描述 prompt。

## 生成量化权重

### <span id="qwen25-omni-7b-w8a8-量化">Qwen2.5-Omni-7B W8A8 动态量化</span>

该模型的量化已经集成至[一键量化](../../../docs/zh/user_guide/usage_quick_quantization.md#32-参数说明)。

```shell
msmodelslim quant \
    --model_path /path/to/qwen2_5_omni_float_weights \
    --save_path /path/to/qwen2_5_omni_quantized_weights \
    --device npu \
    --model_type Qwen2.5-Omni-7B \
    --quant_type w8a8 \
    --trust_remote_code true
```

## 附录

- [multimodal_vlm_modelslim_v1 量化服务配置详解](../../../docs/zh/user_guide/usage_quick_quantization.md#54-multimodal_vlm_modelslim_v1-配置详解)
