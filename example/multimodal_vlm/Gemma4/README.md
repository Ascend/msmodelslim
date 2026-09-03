# Gemma4 量化说明

## 模型介绍

Gemma4 是多模态视觉语言模型，支持图像与文本联合输入。msModelSlim 当前支持 Gemma4 dense 结构的 W8A8 与 MXFP8/MXFP4 混合量化最佳实践。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../docs/zh/install_guide/install_guide.md)。
- 针对 Gemma4，transformers 版本需要 5.5.3：

  ```bash
  pip install "transformers==5.5.3"
  ```

- 若使用 `--trust_remote_code True`，请确保加载的自定义代码文件来源可靠，避免潜在的安全风险。

## Gemma4 模型当前已验证的量化方法

| 模型 | 原始浮点权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|-------------|---------|----------------|---------|
| gemma-4-31B-it | [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | W8A8量化 | vLLM Ascend | [W8A8量化](#gemma4-31b-dense-w8a8量化) |
| gemma-4-31B-it | [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | MXFP8 / MXFP4 混合量化 | vLLM Ascend（优先 Ascend 950） | [MXFP8/MXFP4 混合量化](#gemma4-31b-dense-mxfp8mxfp4-混合量化) |

**说明：** 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 使用示例

### <span id="gemma4-31b-dense-w8a8量化">gemma-4-31B-it W8A8量化</span>

该模型的量化已集成至[一键量化](../../../docs/zh/user_guide/usage_quick_quantization.md#32-参数说明)。使用 `model_type=gemma-4-31B-it`、`quant_type=w8a8` 即可。

```shell
msmodelslim quant \
    --model_path /path/to/gemma4_float_weights \
    --save_path /path/to/gemma4_w8a8_weights \
    --device npu \
    --model_type gemma-4-31B-it \
    --quant_type w8a8 \
    --trust_remote_code True
```

若需指定自定义配置文件，可通过 `config_path` 指定 [gemma4_w8a8.yaml](../../../lab_practice/gemma4/gemma4_w8a8.yaml)。

```shell
msmodelslim quant \
    --model_path /path/to/gemma4_float_weights \
    --save_path /path/to/gemma4_w8a8_weights \
    --device npu \
    --model_type gemma-4-31B-it \
    --config lab_practice/gemma4/gemma4_w8a8.yaml \
    --trust_remote_code True
```

### <span id="gemma4-31b-dense-mxfp8mxfp4-混合量化">gemma-4-31B-it MXFP8/MXFP4 混合量化</span>

实践配置见 [`gemma4_mxfp8_mxfp4.yaml`](../../../lab_practice/gemma4/gemma4_mxfp8_mxfp4.yaml)。可通过 `--quant_type w4a4`（匹配 `label.w_bit/a_bit=4`）。

```shell
msmodelslim quant \
    --model_path /path/to/gemma4_float_weights \
    --save_path /path/to/gemma4_mxfp8_mxfp4_weights \
    --device npu \
    --model_type gemma-4-31B-it \
    --quant_type w4a4 \
    --trust_remote_code True
```

> [!NOTE]
>
> MXFP4/MXFP8 算子路径优先在 Ascend 950 + vLLM Ascend 验证；其他硬件请以实际 CANN / 推理框架版本为准。

## 附录

- [一键量化配置协议说明](../../../docs/zh/user_guide/usage_quick_quantization.md#5-量化配置协议详解)
- [multimodal_vlm_modelslim_v1 量化服务配置详解](../../../docs/zh/user_guide/usage_quick_quantization.md#54-multimodal_vlm_modelslim_v1-配置详解)
