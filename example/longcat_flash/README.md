# LongCat-Flash 量化说明

## 模型介绍

[LongCat-Flash-Chat](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat) 是 meituan-longcat 开源的大语言模型。msModelSlim 已适配 LongCat-Flash-Chat 的 W4A4 MXFP4 一键量化实践，量化结果可用于 SGLang Ascend 推理。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../docs/zh/install_guide/install_guide.md)》。
- 由于模型量化对显存要求较高，请确保在满足模型加载和校准数据处理需求的环境下执行。

## 昇腾AI处理器支持情况

- 支持 Atlas A5 推理产品。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接 | W4A4 MXFP4 | W8A8 MXFP8 | 稀疏量化 | KV Cache | 量化命令 |
|---------|---------|----------------|------------|------------|---------|----------|----------|
| **LongCat-Flash** | LongCat-Flash-Chat | [LongCat-Flash-Chat](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat) | ✅ |  |  |  | [W4A4 MXFP4](#longcat-flash-chat-w4a4-mxfp4量化) |

>[!NOTE]
>
> - ✅ 表示该量化策略已通过 msModelSlim 官方验证，功能完整且性能稳定，建议优先采用。
> - 空格表示该量化策略暂未通过 msModelSlim 官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。

## 量化权重生成

### 使用示例

请将`${MODEL_PATH}`替换为用户实际浮点权重路径，`${SAVE_PATH}`替换为量化权重保存路径。

#### <span id="longcat-flash-chat-w4a4-mxfp4量化">LongCat-Flash-Chat W4A4 MXFP4量化</span>

生成 LongCat-Flash-Chat 模型 W4A4 MXFP4 量化权重：

```shell
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type longcat_flash \
  --quant_type w4a4 \
  --tag Atlas_A5_Interface \
  --trust_remote_code True
```

该一键量化命令匹配使用的量化配置文件为[longcat_flash_w4a4_mxfp4.yaml](../../lab_practice/longcat_flash/longcat_flash_w4a4_mxfp4.yaml)，可以在其中查看具体的量化策略。
