# MiniMax-M2 量化说明

## 模型介绍

[MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) 是 MiniMax 开源的大语言模型。msModelSlim 已适配 MiniMax-M2.7 的 W8A8、W8A8C8 与 W8A8 MXFP8 一键量化实践，量化结果可用于 vLLM Ascend 推理。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../docs/zh/install_guide/install_guide.md)》。
- 由于模型量化对显存要求较高，请确保在满足模型加载和校准数据处理需求的环境下执行。

## 昇腾AI处理器支持情况

- 支持 Atlas A3 训练、推理产品。
- 支持 Atlas A5 训练、推理产品。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接 | W8A8 | W8A8 MXFP8 | W8A16 | W4A8 | W8A8C8 | W4A8C8 | 稀疏量化 | KV Cache | Attention | 量化命令 |
|---------|---------|----------------|------|------------|-------|------|--------|--------|---------|----------|-----------|----------|
| **MiniMax-M2** | MiniMax-M2.7 | [MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) | ✅ | ✅ |  |  | ✅ |  |  | ✅ |  | [W8A8](#minimax-m27-w8a8量化) / [W8A8 MXFP8](#minimax-m27-w8a8-mxfp8量化) / [W8A8C8](#minimax-m27-w8a8c8量化) |

>[!NOTE]
>
> - ✅ 表示该量化策略已通过 msModelSlim 官方验证，功能完整且性能稳定，建议优先采用。
> - 空格表示该量化策略暂未通过 msModelSlim 官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
> - W8A8C8 表示在 W8A8 量化基础上开启 KV Cache INT8 量化。

## 量化权重生成

### 使用示例

请将`${MODEL_PATH}`替换为用户实际浮点权重路径，`${SAVE_PATH}`替换为量化权重保存路径。

#### <span id="minimax-m27-w8a8量化">MiniMax-M2.7 W8A8量化</span>

生成 MiniMax-M2.7 模型 W8A8 量化权重：

```shell
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type MiniMax-M2.7 \
  --quant_type w8a8 \
  --trust_remote_code True
```

该一键量化命令匹配使用的量化配置文件为[minimax_m27_w8a8.yaml](../../lab_practice/minimax_m2/minimax_m27_w8a8.yaml)，可以在其中查看具体的量化策略。

#### <span id="minimax-m27-w8a8c8量化">MiniMax-M2.7 W8A8C8量化</span>

生成 MiniMax-M2.7 模型 W8A8C8 量化权重：

```shell
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type MiniMax-M2.7 \
  --quant_type w8a8c8 \
  --trust_remote_code True
```

该一键量化命令匹配使用的量化配置文件为[minimax_m27_w8a8c8.yaml](../../lab_practice/minimax_m2/minimax_m27_w8a8c8.yaml)，可以在其中查看具体的量化策略。

#### <span id="minimax-m27-w8a8-mxfp8量化">MiniMax-M2.7 W8A8 MXFP8量化</span>

生成 MiniMax-M2.7 模型 W8A8 MXFP8 量化权重：

```shell
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type MiniMax-M2.7 \
  --quant_type w8a8 \
  --tag Atlas_A5_Interface \
  --trust_remote_code True
```

该一键量化命令匹配使用的量化配置文件为[minimax_m27_w8a8_mxfp8.yaml](../../lab_practice/minimax_m2/minimax_m27_w8a8_mxfp8.yaml)，可以在其中查看具体的量化策略。
