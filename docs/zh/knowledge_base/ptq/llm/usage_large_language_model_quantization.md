# 大语言模型（LLM）量化使用指南

## 1. 适用范围

本流程适用于对 **Decoder-only 架构的大语言模型**（如 LLaMA、Qwen、GLM、DeepSeek 等）执行训练后量化（PTQ）操作。用户在完成模型选择和校准数据准备后，可通过本流程快速完成量化。

**适用角色**：大模型量化算法工程师、模型部署工程师

**典型场景**：

- 将浮点大语言模型量化为 W8A8 / W4A16 / W4A8 等低比特格式
- 为在线推理服务准备量化权重
- 在资源受限环境下部署大语言模型

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，进入量化环节。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[安装指南](../../../install_guide/install_guide.md)》）
- 已获取目标模型的浮点权重（HuggingFace 格式）
- 已确认目标模型在 [支持矩阵](../../model/README.md) 中
- 已准备好校准数据集（可选，工具提供默认校准集）

**后续操作**：量化结果精度验证 → 部署上线或进入调优

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt | 至少包含 128条样本 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors` | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
  A[准备模型与校准集] --> B[选择量化方案]
  B --> C[执行量化命令]
  C --> D[验证量化结果]
  D --> E[部署或调优]
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

**目标**：确认 `model_type`、`quant_type` 与目标推理场景匹配。

**操作**：

1. 查阅 [大模型支持矩阵](../../model/README.md)，确认目标模型已支持的量化模式。
2. 根据推理场景选择合适的量化类型（如 `w8a8`、`w4a16`、`w4a8` 等）。

**输出**：确定的模型名称（`${MODEL_TYPE}`）和量化类型（`${QUANT_TYPE}`）。

### 步骤 3：执行一键量化

**目标**：对浮点模型执行量化，生成量化权重。

**操作**：

方式一（推荐，使用 `quant_type` 参数）：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --quant_type ${QUANT_TYPE} \
  --trust_remote_code True
```

方式二（使用自定义配置）：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config ${CONFIG_PATH} \
  --trust_remote_code True
```

**参数说明**：

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `model_path` | 必选 | 浮点模型权重路径 |
| `save_path` | 必选 | 量化权重保存路径 |
| `device` | 可选 | 量化设备，默认 `npu` |
| `model_type` | 必选 | 模型名称，与支持矩阵一致 |
| `quant_type` | 与 `config_path` 二选一 | 量化类型，如 `w8a8` |
| `config_path` | 与 `quant_type` 二选一 | 自定义 YAML 配置路径 |
| `trust_remote_code` | 可选 | 是否信任远程代码，默认 `False` |

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 4：验证量化结果

**目标**：确认量化权重完整且可加载。验证的核心是**权重完整性**——生成的量化权重应完整，即输出目录包含量化描述文件与全部权重分片，无缺失或损坏。

**操作**：

1. 核对输出目录是否包含 `quant_model_description.json` 文件，并确认其中的权重分片（如 `*.safetensors`）数量齐全、无缺失。

**输出**：量化权重验证通过。

### 步骤 5：精度测评

**目标**：对量化后模型进行精度测评，确认量化精度满足上线要求。

**操作**：

方式一（使用 curl 请求推理服务进行抽样验证）：

```bash
curl -X POST ${INFERENCE_ENDPOINT} \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请用一句话解释什么是量子计算。",
    "max_tokens": 128
  }'
```

通过对比量化前后模型对同一组测试 prompt 的生成结果，评估质量损失。

方式二（使用标准评测工具进行系统评估）：

推荐使用 [AISbench](https://github.com/AISBench/benchmark) 评测工具，在量化前后模型上运行相同的评测任务（如 MMLU、CEval、GSM8K 等），对比精度指标。

> 详细精度调优方法请参阅《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》。

**输出**：量化精度评估报告，确认量化后精度在可接受范围内。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 量化后模型推理精度在可接受范围内（建议与浮点基线对比）。
- 量化后模型可被目标推理框架成功加载。

## 7. 异常处置

- **量化失败**：检查日志中的错误信息，确认模型路径、`model_type` 与参数是否配置正确；若使用了多卡分布式逐层量化，需确认所用处理器（Processor）支持分布式执行——框架在启动时会统一校验并抛出不支持错误。
- **精度不达标**：参考《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》进行调优。
- **量化速度较慢**：增加 `--device` 指定多卡，框架启用数据并行（DP）逐层量化，借助多卡并行显著提升量化效率。多卡 DP 的开启方式与配置详见[《多卡分布式量化示例》](../../../user_guide/usage_quick_quantization.md#333-示例3多卡分布式量化)。
- **显存不足**：若模型参数过大，单卡无法放置一层的权重，可在模型适配器中适配自动按卡切分专家（EP）功能，将模型切分后按卡加载，避免单卡显存溢出。EP 相关适配详见[《专家并行机制使用指南》](../../parallel/expert_parallelism/expert_parallelism_guide.md)。此外，若仍遇到显存不足，请确认 `--device npu --device_id 0` 指定的 NPU 未被其他任务占用。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| LLM 量化 | 大语言模型训练后量化 | [LLM 量化词条](./term_large_language_model_quantization.md) |
| PTQ | 训练后量化 | [PTQ 总览](../README.md) |
| 逐层量化 | 逐层处理模型以降低内存占用 | [逐层量化说明](../../../user_guide/usage_quick_quantization.md#41-逐层量化及分布式逐层量化) |
