# Qwen3.5 量化说明

## 模型介绍

**Qwen3.5** 是 Qwen 系列最新的旗舰**多模态**模型，采用 **MoE (Mixture of Experts)** 架构，在保持极强模型能力的同时显著降低推理成本。核心架构特点包括：原生多模态能力（Vision Encoder + 图文融合）、混合注意力机制（常规 Attention 与 Linear-Attention 交替）、MTP 多 Token 预测分支、以及高性能 MoE 专家路由与共享专家机制。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../docs/zh/getting_started/install_guide.md)。

- transformers 版本需要配置安装 5.2.0 版本。
  - pip install transformers==5.2.0

## 昇腾AI处理器支持情况

- 支持 Atlas A2 训练、推理产品，Atlas A3 训练、推理产品

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接                                                 | W8A8 | W8A16 | W4A8 | W4A16 | W4A4  | 稀疏量化 | KV Cache | Attention | 量化命令                                          |
|---------|---------|---------------------------------------------------------------|-----|-----|-----|--------|------|---------|----------|-----------|-----------------------------------------------|
| **Qwen3.5-MoE** | Qwen3.5-397B-A17B | [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)   | ✅ |  | ✅ |      |   |  |   |   | [W8A8](#Qwen3.5-397B-A17B-w8a8)/[W4A8](#Qwen3.5-397B-A17B-w4a8)|
| **Qwen3.5-MoE** | Qwen3.5-122B-A10B | [Qwen3.5-122B-A10B](https://modelscope.cn/models/Qwen/Qwen3.5-122B-A10B)   | ✅ |  |    |        |   |  |   |   | [W8A8](#Qwen3.5-122B-A10B-w8a8)|
| **Qwen3.5-MoE** | Qwen3.5-35B-A3B | [Qwen3.5-35B-A3B](https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B)   | ✅ |  |    |        |   |  |   |   | [W8A8](#Qwen3.5-35B-A3B-w8a8)|
| **Qwen3.5-Dense** | Qwen3.5-27B | [Qwen3.5-27B](https://modelscope.cn/models/Qwen/Qwen3.5-27B)   | ✅ |  |    |        |   |  |   |   | [W8A8](#Qwen3.5-27B-w8a8)|
| **Qwen3.5-Dense** | Qwen3.6-27B | [Qwen3.6-27B](https://modelscope.cn/models/Qwen/Qwen3.6-27B)   | ✅ |  |    |        |   |  |   |   | [W8A8](#Qwen3.6-27B-w8a8)|

**说明：**

- ✅ 表示该量化策略已通过 msModelSlim 官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过 msModelSlim 官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 量化权重生成

### 使用示例

- 请将{MODEL_PATH}替换为用户实际浮点权重路径，{SAVE_PATH}替换为量化权重保存路径。

#### 1. Qwen3.5-397B-A17B

##### <span id="Qwen3.5-397B-A17B-w8a8">Qwen3.5-397B-A17B W8A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-397B-A17B --quant_type w8a8 --trust_remote_code True
  ```

##### <span id="Qwen3.5-397B-A17B-w4a8">Qwen3.5-397B-A17B W4A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-397B-A17B --quant_type w4a8 --trust_remote_code True
  ```

#### 2. Qwen3.5-122B-A10B

##### <span id="Qwen3.5-122B-A10B-w8a8">Qwen3.5-122B-A10B W8A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-122B-A10B --quant_type w8a8 --trust_remote_code True
  ```

#### 3. Qwen3.5-35B-A3B

##### <span id="Qwen3.5-35B-A3B-w8a8">Qwen3.5-35B-A3B W8A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-35B-A3B --quant_type w8a8 --trust_remote_code True
  ```

#### 4. Qwen3.5-27B

##### <span id="Qwen3.5-27B-w8a8">Qwen3.5-27B W8A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-27B --quant_type w8a8 --trust_remote_code True
  ```

#### 4. Qwen3.6-27B

##### <span id="Qwen3.6-27B-w8a8">Qwen3.6-27B W8A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.6-27B --quant_type w8a8 --trust_remote_code True
  ```
