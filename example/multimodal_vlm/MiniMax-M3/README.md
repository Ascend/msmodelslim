# MiniMax-M3 量化案例

## 模型介绍

[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3-preview) 是 MiniMax 开源的视觉-语言模型，采用 60 层混合专家（MoE）架构，支持多模态理解和长上下文推理，具备优秀的文本理解和视觉感知能力。

msModelSlim 已适配 MiniMax-M3 的 W8A8 一键量化实践，量化结果可用于 vLLM Ascend 推理。

## 环境配置

- 基础环境配置请参考[安装指南](../../../docs/zh/install_guide/install_guide.md)。
- 针对 MiniMax-M3，transformers 版本需要 5.12.0：

  ```bash
  pip install transformers==5.12.0
  ```

## MiniMax-M3 模型当前已验证的量化方法

| 模型 | 原始浮点权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|-------------|---------|----------------|---------|
| MiniMax-M3-preview | [MiniMax-M3-preview](https://huggingface.co/MiniMaxAI/MiniMax-M3-preview) | W8A8 量化 | vLLM Ascend 支持 | [W8A8量化](#minimax-m3-w8a8量化) |

## 量化权重生成

### 使用示例

请将 `${MODEL_PATH}` 替换为用户实际浮点权重路径，`${SAVE_PATH}` 替换为量化权重保存路径。

- 如果需要使用 NPU 多卡量化，请先配置多卡环境变量：

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

- 若加载自定义模型，调用 `from_pretrained` 函数时要指定 `trust_remote_code=True`。

#### <span id="minimax-m3-w8a8量化">MiniMax-M3 W8A8量化</span>

MiniMax-M3 的量化已集成至[一键量化](../../../docs/zh/user_guide/usage_quick_quantization.md)。使用 `model_type=MiniMax-M3`、`quant_type=w8a8` 即可。若需使用自定义配置，可通过 `config_path` 指定 [minimax_m3_w8a8.yaml](../../../lab_practice/minimax_m3/minimax_m3_w8a8.yaml)。

```shell
msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --device npu \
    --model_type MiniMax-M3 \
    --quant_type w8a8 \
    --trust_remote_code True
```

使用自定义配置文件时：

```shell
msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --device npu \
    --model_type MiniMax-M3 \
    --quant_type w8a8 \
    --config_path lab_practice/minimax_m3/minimax_m3_w8a8.yaml \
    --trust_remote_code True
```

**说明：**

- MiniMax-M3 默认精度为 `bfloat16`。
- 该量化命令匹配使用的量化配置文件为 [minimax_m3_w8a8.yaml](../../../lab_practice/minimax_m3/minimax_m3_w8a8.yaml)，可以在其中查看具体的量化策略。
