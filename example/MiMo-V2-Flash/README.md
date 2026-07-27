# MiMo-V2-Flash 量化说明

## 模型介绍

- [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) 是小米 MiMo 团队开源的混合专家（Mixture-of-Experts，MoE）语言模型，总参数量为 309B，每个 token 激活约 15B 参数，支持最长 256K 上下文。
- MiMo-V2-Flash 采用滑动窗口注意力与全局注意力交替组成的混合注意力架构，并支持多 Token 预测（Multi-Token Prediction，MTP），面向推理及 Agent 场景进行优化。
- msModelSlim 已支持 MiMo-V2-Flash 模型的 W8A8 动态量化。量化过程中使用逐层加载方式处理 Decoder Layer，以降低模型加载和量化过程中的内存峰值。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../docs/zh/install_guide/install_guide.md)》。
- MiMo-V2-Flash 适配器依赖 `transformers==4.57.6`，请执行：

  ```shell
  pip install transformers==4.57.6
  ```

- 请准备 Hugging Face 格式的 MiMo-V2-Flash BF16 权重。模型目录中应包含 `config.json`、Safetensors 权重文件及对应的权重索引文件。
- MiMo-V2-Flash 使用自定义模型代码，量化命令中需设置 `--trust_remote_code True`。启用该选项前，请确认模型代码来源可信。
- MiMo-V2-Flash 参数量较大，请在内存和显存充足的 NPU 环境中执行量化。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | Hugging Face链接 | W8A8 | W8A16 | W4A8 | W4A16 | W4A4 | 稀疏量化 | KV Cache | 量化命令 |
|---------|---------|------------------|------|-------|------|-------|------|---------|----------|---------|
| **MiMo** | MiMo-V2-Flash | [XiaomiMiMo/MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) | ✅ | | | | | | | [W8A8](#mimo-v2-flash-w8a8量化) |

**说明：**

- ✅ 表示该量化策略已通过 msModelSlim 验证，建议优先采用。
- 空白表示该量化策略暂未通过 msModelSlim 验证，量化效果和功能稳定性不作保证。
- 点击“量化命令”列中的链接，可跳转到对应的量化命令。

## 量化权重生成

MiMo-V2-Flash W8A8 配置已接入一键量化。指定 `--model_type MiMo-V2-Flash` 和 `--quant_type w8a8` 后，工具会自动推荐并使用 [mimo-v2-flash-w8a8.yaml](../../lab_practice/mimo_v2/mimo-v2-flash-w8a8.yaml)。

### <span id="mimo-v2-flash-w8a8量化">MiMo-V2-Flash W8A8量化</span>

请将 `${model_path}` 和 `${save_path}` 替换为实际路径。

推荐使用以下一键量化命令：

```shell
msmodelslim quant \
  --device npu \
  --model_path ${model_path} \
  --save_path ${save_path} \
  --model_type MiMo-V2-Flash \
  --quant_type w8a8 \
  --trust_remote_code True
```

使用实际目录时，命令示例如下：

```shell
msmodelslim quant \
  --device npu \
  --model_path ./MiMo-V2-Flash-BF16/ \
  --save_path ./MiMo-V2-Flash-w8a8/ \
  --model_type MiMo-V2-Flash \
  --quant_type w8a8 \
  --trust_remote_code True
```

若需调试或验证指定配置文件，也可以显式传入配置路径：

```shell
msmodelslim quant \
  --device npu \
  --model_path ./MiMo-V2-Flash-BF16/ \
  --save_path ./MiMo-V2-Flash-w8a8/ \
  --model_type MiMo-V2-Flash \
  --config_path lab_practice/mimo_v2/mimo-v2-flash-w8a8.yaml \
  --trust_remote_code True
```

其中：

- `--model_path`：MiMo-V2-Flash BF16 浮点权重目录。
- `--save_path`：量化权重保存目录。
- `--model_type`：需填写 `MiMo-V2-Flash`，区分大小写。
- `--quant_type w8a8`：启用 W8A8 一键量化，并自动推荐 MiMo-V2-Flash 最佳实践配置。
- `--trust_remote_code True`：允许加载模型目录中的自定义模型代码，请确保代码来源可信。

## 量化策略说明

| 配置项 | 说明 |
|--------|------|
| 激活量化 | per-token、INT8、对称 MinMax 动态量化 |
| 权重量化 | per-channel、INT8、对称 MinMax 量化 |
| 量化范围 | `include: ["*"]`，对模型中的 Linear 层执行 W8A8 量化 |
| 保存格式 | `ascendv1_saver` |
| 权重分片 | `part_file_size: 4`，单个 Safetensors 文件最大 4GB |

## FAQ

### 指定 `--quant_type w8a8` 后没有自动匹配 MiMo-V2-Flash 配置怎么办？

请依次检查：

1. `--model_type` 是否精确填写为 `MiMo-V2-Flash`。
2. `lab_practice/mimo_v2/mimo-v2-flash-w8a8.yaml` 是否存在，并包含 `metadata`。
3. 适配器 `get_model_pedigree()` 的返回值是否为 `mimo_v2`，与 `lab_practice/mimo_v2` 目录名一致。
4. 是否安装了包含本模型适配和最佳实践配置的最新 msModelSlim 版本。

### 如何确认自动推荐成功？

运行带有 `--quant_type w8a8` 的一键量化命令，日志中应出现使用 `mimo-v2-flash-w8a8` 最佳实践配置的提示，且不应回退到 `default-w8a8`。

### transformers 版本不匹配怎么办？

请安装适配器要求的版本：

```shell
pip install transformers==4.57.6
```
