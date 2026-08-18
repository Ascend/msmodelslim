# Diffusion Transformer（DiT）量化使用指南

## 1. 适用范围

本流程适用于对 **Diffusion Transformer 架构的多模态生成模型**（如 FLUX.1、HunyuanVideo、Wan2.2、Open-Sora-Plan v1.2、Qwen-Image-Edit 等）执行训练后量化（PTQ）操作。

**适用角色**：大模型量化算法工程师、模型部署工程师

**典型场景**：

- 文生图模型（FLUX.1、SD3）的 W8A8 MXFP8 量化
- 文生视频模型（HunyuanVideo、Wan2.2）的量化部署
- 图像编辑模型（Qwen-Image-Edit）的量化部署

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，进入量化环节。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[安装指南](../../../install_guide/install_guide.md)》）
- 已获取目标 DiT 模型浮点权重
- 已确认目标模型在 [支持矩阵](../../model/README.md) 中
- 已准备好推理配置参数（分辨率、步数、任务类型等）

**后续操作**：量化结果精度验证 → 部署到 MindIE-SD 推理框架

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点 DiT 权重目录 | 模型下载或本地路径 | 含模型权重文件及配置文件 | 可被推理管线加载 |
| 输入 | 推理配置 | YAML 中 `multimodal_sd_config.inference_config` | 含 size/frame_num/sample_steps/task 等 | 与 model_type 匹配 |
| 输入 | 校准样本 | `dataset` 字段指定 | 由适配器 `handle_dataset` 处理 | 至少包含 1条样本 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | MindIE-SD 格式，多专家含各专家子目录 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
  A[准备 DiT 模型与推理配置] --> B[选择量化方案]
  B --> C[运行浮点推理管线生成校准数据]
  C --> D[执行多专家量化]
  D --> E[验证量化结果]
  E --> F[MindIE-SD 部署]
```

## 5. 操作步骤

### 步骤 1：安装 msModelSlim 工具

**目标**：完成 msModelSlim 工具的安装与验证。

**操作**：

```bash
# 安装完成后验证工具可用
msmodelslim --help
```

> 详细的安装方式（在线安装、离线安装、源码安装）请参阅《[msModelSlim 安装指南](../../../install_guide/install_guide.md)》。

**输出**：msModelSlim 工具安装成功，可正常执行 `msmodelslim` 命令。

### 步骤 2：确认模型与量化方案

**目标**：确认 `model_type`、task 类型与量化方案匹配。

**操作**：

1. 查阅 [大模型支持矩阵](../../model/README.md)，确认目标 DiT 模型已支持的量化模式。
2. 根据模型类型和 task 选择对应的 YAML 配置（如 Wan2.2-T2V-A14B w8a8f8_mxfp）。
3. DiT 量化必须使用 `config_path` 指定 YAML 配置，不支持 `quant_type` 方式。

**输出**：确定的模型名称（`${MODEL_TYPE}`）和配置路径（`${CONFIG_PATH}`）。

### 步骤 3：执行 DiT 量化

**目标**：对浮点 DiT 模型执行量化，生成量化权重。

**操作**：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config ${CONFIG_PATH}
```

**参数说明**：

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `model_path` | 必选 | 浮点 DiT 模型权重路径 |
| `save_path` | 必选 | 量化权重保存路径 |
| `device` | 可选 | 量化设备，默认 `npu` |
| `model_type` | 必选 | 模型名称，与支持矩阵一致 |
| `config_path` | 必选 | 指向 `lab_practice/` 下 DiT 的 YAML 配置 |

**注意事项**：

- DiT 量化会首先运行浮点推理管线（可能耗时较长），生成校准数据后自动进入量化阶段。
- 对于多专家模型（如 Wan2.2），校准数据会按专家名分别 dump 为 pth 文件。
- 可设置 `dump_config.enable_dump: False` 跳过浮点推理（仅适用于纯动态量化场景）。
- 不支持 `--quant_type` 方式，必须通过 `--config` 指定完整配置。

**参考配置示例**：

| 模型 | 量化方案 | 配置路径 |
|------|---------|---------|
| HunyuanVideo | W8A8 MXFP8 | `lab_practice/hunyuan_video/hunyuan_video_w8a8f8_mxfp.yaml` |
| Wan2.2 T2V | W8A8 MXFP8 | `lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_t2v.yaml` |
| Wan2.2 T2V | W4A4 MXFP4 | `lab_practice/wan2_2/wan2_2_w4a4f4_mxfp_t2v.yaml` |
| Wan2.2 I2V | W8A8 MXFP8 | `lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_i2v.yaml` |
| FLUX.1 | W8A8 MXFP8 | `lab_practice/flux1/flux1_w8a8f8_mxfp.yaml` |
| Qwen-Image-Edit | W8A8 MXFP8 | `lab_practice/qwen_image_edit/qwen-image-edit-w8a8f8-mxfp.yaml` |
| Wan2.1 | W8A8 dynamic | `lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml` |

**输出**：量化权重目录 `${SAVE_PATH}`。多专家模型下含 `low_noise_model/`、`high_noise_model/` 等子目录。

### 步骤 4：验证量化结果

**目标**：确认量化权重完整且可加载。验证的核心是**权重完整性**——生成的量化权重应完整，即输出目录包含权重描述信息与全部权重分片，无缺失或损坏。

**操作**：

1. 核对输出目录是否包含 MindIE-SD 格式的量化权重文件，并确认其中的权重分片数量齐全、无缺失。
2. 多专家模型核对各专家子目录是否均有完整权重文件。

**输出**：量化权重验证通过。

### 步骤 5：精度测评

**目标**：对量化后 DiT 模型进行精度测评，确认量化精度满足生成质量要求。

**操作**：

方式一（使用 curl 请求推理服务进行抽样验证）：

```bash
curl -X POST ${INFERENCE_ENDPOINT} \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "一只在雪地里奔跑的哈士奇",
    "num_inference_steps": 50
  }'
```

通过对比量化前后模型对同一组 prompt 的生成图像质量，评估质量损失。

方式二（使用标准评测工具进行系统评估）：

推荐使用 [AISbench](https://github.com/AISBench/benchmark) 评测工具，对量化前后模型的生成质量进行对比评估（支持 FID、CLIP Score 等图像/视频生成质量指标）。

> 详细精度调优方法请参阅《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》。

**输出**：量化精度评估报告，确认量化后生成质量在可接受范围内。

## 6. 验收条件

- 量化权重目录中多专家模型各子目录均有完整的权重文件。
- 量化后 DiT 推理生成质量在可接受范围内（建议与浮点基线对比）。
- 量化后模型可被 MindIE-SD 推理框架成功加载。

## 7. 异常处置

- **量化失败**：检查日志中的错误信息，确认模型路径、`model_type` 与参数是否配置正确；若使用了多卡分布式逐层量化，需确认所用处理器（Processor）支持分布式执行——框架在启动时会统一校验并抛出不支持错误。
- **浮点推理失败**：检查 `multimodal_sd_config.inference_config` 参数是否正确（如分辨率、步数、任务类型等）。
- **校准数据缺失**：检查 `enable_dump` 是否为 True，`dump_data_dir` 是否有写入权限。
- **精度不达标**：尝试调整推理配置参数，增加校准样本数量，或参考《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》进行调优。
- **量化速度较慢**：增加 `--device` 指定多卡，框架启用数据并行（DP）逐层量化，借助多卡并行显著提升量化效率。多卡 DP 的开启方式与配置详见[《多卡分布式量化示例》](../../../user_guide/usage_quick_quantization.md#333-示例3多卡分布式量化)。
- **显存不足**：若模型参数过大，单卡无法放置一层的权重，可在模型适配器中适配自动按卡切分专家（EP）功能，将模型切分后按卡加载，避免单卡显存溢出。EP 相关适配详见[《专家并行机制使用指南》](../../parallel/expert_parallelism/expert_parallelism_guide.md)。此外，若仍遇到显存不足，请确认 `--device npu --device_id 0` 指定的 NPU 未被其他任务占用。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| DiT 量化 | Diffusion Transformer 多模态生成模型训练后量化 | [DiT 量化词条](./term_diffusion_transformer_quantization.md) |
| 多专家量化 | 多专家 DiT 按专家分别校准和量化 | [DiT 量化词条 - 原理](./term_diffusion_transformer_quantization.md#3-原理) |
| PTQ | 训练后量化 | [PTQ 总览](../README.md) |
| MindIE-SD 格式 | 昇腾多模态生成模型推理格式 | [一键量化生成结果](../../quantization_format/ascendv1/ascendv1_usage.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim quant` | 一键量化 CLI 入口 | [一键量化完整指南](../../../user_guide/usage_quick_quantization.md) |
| 多模态生成模型接入 | DiT 模型接入与适配器开发 | [多模态生成模型接入](../../model/integrating_multimodal_generation_model.md) |
