# Step3.5-Flash 量化说明

## 模型介绍

[Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) 是 StepFun AI 开源的大语言模型，采用 MoE 架构并包含 MTP（Multi-Token Prediction）模块。msModelSlim 已适配 Step-3.5-Flash 的 W8A8 一键量化实践，当前量化配置主要覆盖 MoE experts 线性层动态量化，量化结果可用于昇腾推理场景。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../docs/zh/install_guide/install_guide.md)》。
- 需要安装 transformers 4.57.1 及以上、5.0.0 以下版本：

  ```shell
  pip install "transformers>=4.57.1,<5.0.0"
  ```

- 由于模型量化对显存要求较高，请确保在满足模型加载和校准数据处理需求的环境下执行。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接 | W8A8 | W8A16 | W4A8 | W8A8C8 | W4A8C8 | 稀疏量化 | KV Cache | Attention | 量化命令 |
|---------|---------|----------------|------|-------|------|--------|--------|---------|----------|-----------|----------|
| **Step3.5** | Step-3.5-Flash | [Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) | ✅ |  |  |  |  |  |  |  | [W8A8](#step-35-flash-w8a8量化) |

>[!NOTE]
>
> - ✅ 表示该量化策略已通过 msModelSlim 官方验证，功能完整且性能稳定，建议优先采用。
> - 空格表示该量化策略暂未通过 msModelSlim 官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
> - W8A8C8 表示在 W8A8 量化基础上开启 KV Cache INT8 量化。

## 量化权重生成

### 使用示例

请将`${MODEL_PATH}`替换为用户实际浮点权重路径，`${SAVE_PATH}`替换为量化权重保存路径。

#### <span id="step-35-flash-w8a8量化">Step-3.5-Flash W8A8量化</span>

生成 Step-3.5-Flash 模型 W8A8 量化权重：

```shell
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type Step-3.5-Flash \
  --quant_type w8a8 \
  --trust_remote_code true
```

该一键量化命令匹配使用的量化配置文件为[step3_5_moe_w8a8.yaml](../../lab_practice/step_3_5_flash/step3_5_moe_w8a8.yaml)，可以在其中查看具体的量化策略。
