# Qwen3-Next 量化说明

## 模型介绍

- [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) 代表了Qwen团队推出的下一代基础模型，专为极端上下文长度和大规模参数效率进行了优化。其引入了一系列架构创新，在最大化性能的同时最小化计算成本。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../docs/zh/getting_started/install_guide.md)。
- 需要安装 transformers 4.57.0 或更高版本：
    - pip install transformers==4.57.1

## 昇腾AI处理器支持情况

- 支持Atlas A2训练、推理产品，Atlas A3训练、推理产品

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接                                                 | W8A8 | W8A16 | W4A8 | W4A16 | W4A4  | 稀疏量化 | KV Cache | Attention | 量化命令                                          |
|---------|---------|---------------------------------------------------------------|-----|-----|-----|--------|------|---------|----------|-----------|-----------------------------------------------|
| **Qwen3-Next** | Qwen3-Next-80B-A3B-Instruct | [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)   | ✅ |  |    |        |   |  |   |   | [W8A8](#qwen3-next-80b-a3b-instruct)|

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令

## 量化权重生成

### 使用示例

- 请将`${MODEL_PATH}`替换为用户实际浮点权重路径,`${SAVE_PATH}`替换为量化权重保存路径。

#### Qwen3-Next-80B-A3B-Instruct

##### <span id="qwen3-next-80b-a3b-instruct-w8a8量">Qwen3-Next-80B-A3B-Instruct W8A8量化</span>

该模型的量化已经集成至[一键量化](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3-Next-80B-A3B-Instruct --quant_type w8a8 --trust_remote_code True
  ```
