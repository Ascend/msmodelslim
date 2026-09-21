# Qwen3-Omni 量化使用说明

## 模型介绍

Qwen3-Omni 是阿里云 Qwen 团队推出的多模态 Omni 模型，支持语音、图像、视频与文本的多模态理解与生成。当前支持以下规格的 W8A8 量化：

- **Qwen3-Omni-30B-A3B-Thinking**：具备思考链能力，30B 总参数、3B 激活参数 MoE 规格
- **Qwen3-Omni-30B-A3B-Instruct**：指令微调版本，30B 总参数、3B 激活参数 MoE 规格

## 校准模态支持

量化校准集（推荐 `index.jsonl`，见[一键量化 dataset 配置](../../../docs/zh/user_guide/usage_quick_quantization.md#dataset---校准数据路径配置)）中，**每条样本必须包含非空 `text`**；`image` / `audio` / `video` 为可选路径字段。

### 已支持的模态组合

在**同一次量化任务**内，允许且仅允许下列之一（同任务内样本须同质，见下节）：

| 有效模态组合 | index.jsonl 字段示意 | 说明 |
|-------------|----------------------|------|
| 纯文本 | `{"text":"..."}` | 仅文本校准 |
| 文本 + 图像 | `text` + `image` | |
| 文本 + 音频 | `text` + `audio` | |
| 文本 + 视频 | `text` + `video` | 视频是否含音轨会影响「有效音频」判定 |
| 文本 + 图像 + 音频 | `text` + `image` + `audio` | |
| 文本 + 图像 + 视频 | `text` + `image` + `video` | |
| 文本 + 音频 + 视频 | `text` + `audio` + `video` | |
| 文本 + 图像 + 音频 + 视频 | 四者均有 | |

> **有效音频**：显式提供 `audio` 路径，**或** `video` 文件本身含音轨（与推理侧 `use_audio_in_video` 一致）。仅有无声视频、无独立 `audio` 时，有效模态为「文本 + 视频」，不含音频。

### 同质约束（重要）

- 同一 YAML / 同一量化任务内，**所有校准样本的有效模态签名必须一致**（是否含图像、是否含有效音频、是否含视频）。
- **不可混用**异构样本，例如同一任务中既有纯文本又有图文。
- **不可混用**「有音轨视频」与「无声视频」：二者有效音频标志不同，会被拒绝并报 `InvalidDatasetError`。
- 需要覆盖多种模态时：为每种组合分别准备校准集并分次量化，或使用官方默认校准集（已同质）。

### 与 visit / forward 路径的关系

- **visit（data-free）** 配方：按固定顺序遍历 audio / visual / decoder tower（与校准样本模态无关）。
- **forward（calibration）** 配方：按当前样本实际存在的 `input_features` / `pixel_values` / `pixel_values_videos` 决定是否前向对应 tower。

## 校准数据说明

校准数据支持的方式，详见 [dataset 配置说明](../../../docs/zh/user_guide/usage_quick_quantization.md#dataset---校准数据路径配置)：

对 Qwen3-Omni，推荐使用 index.json 或 index.jsonl（文件路径或仅含一个 index.json 或 index.jsonl 的目录），支持多模态字段。校准时每条样本提供非空 `text` 及与推理场景一致的多模态组合（`image`、`audio`、`video`），`text` 缺省时使用配置中的 `default_text`。同一任务内所有样本的模态组合需保持一致（同质约束），否则会报 `InvalidDatasetError`，详见上文「校准模态支持」。

`dataset` 可配置为短名称（在 `lab_calib` 等目录下查找）、绝对路径或相对路径。配置示例见 [qwen3-omni-moe-w8a8.yaml](../../../lab_practice/qwen3_omni_moe/qwen3-omni-moe-w8a8.yaml)：`dataset` 指定校准数据集（如 `calibVideos`），`default_text` 可设为如 "What are the elements can you see and hear in these media." 等多模态描述 prompt。

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
