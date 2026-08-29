# GLM5-MOE 量化说明

## 模型介绍

[GLM-5](https://huggingface.co/zai-org/GLM-5)是智谱 AI 于 2026 年 2 月 1 日发布的开源旗舰大语言模型，与 GLM-4.5 相比，GLM-5 的参数规模从 355B 扩展至 744B ，预训练数据量也从 23 万亿个 token 增加到 28.5 万亿个 token。GLM-5 还集成了 DeepSeek 稀疏注意力机制 (DSA)，在大幅降低部署成本的同时，保持了长时域上下文处理能力。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../docs/zh/install_guide/install_guide.md)。
- GLM-5、GLM-5.1 需要配置安装 transformers 5.4.0 版本：

  ```bash
  pip install transformers==5.4.0
  ```

- GLM-5.2 需要配置安装 transformers 5.12.0 版本：

  ```bash
  pip install transformers==5.12.0
  ```

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接                                                 | W8A8 | W8A8C8 | W8A16 | W4A8 | W4A8C8 | W4A16 | W4A4C8  | 稀疏量化 | KV Cache | Attention | 量化命令                                          |
|---------|---------|---------------------------------------------------------------|-----|-----|-----|-----|-----|--------|------|---------|----------|-----------|-----------------------------------------------|
| **GLM5-MOE** | GLM-5 | <https://huggingface.co/zai-org/GLM-5> | ✅ |  |  | ✅ |  |        |   |  |   |   | [W8A8](#glm-5-w8a8量化) / [W4A8](#glm-5-w4a8量化) |
| **GLM5-MOE** | GLM-5.1 | <https://huggingface.co/zai-org/GLM-5.1> | ✅ | ✅ |  | ✅ | ✅ |        | ✅ |  |   |   | [W8A8](#glm-51-w8a8量化) / [W4A8](#glm-51-w4a8量化) / [W8A8C8](#glm-51-w8a8c8量化) / [W4A8C8](#glm-51-w4a8c8量化) / [W4A4C8](#glm-51-w4a4c8-mxfp4量化) |
| **GLM5-MOE** | GLM-5.2 | <https://huggingface.co/zai-org/GLM-5.2> | ✅ | ✅ |  |  |  |        |   |  |   |   | [W8A8](#glm-52-w8a8量化) / [W8A8C8](#glm-52-w8a8c8量化) |
| **GLM5-MOE** | GLM-5.3-Flash | <https://huggingface.co/zai-org/GLM-5.3-Flash> |  |  |  |  |  |        |   |  |   |   | [W8A8（免校准）](#glm-53-flash-w8a8免校准量化) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 一键量化生成量化权重

一键量化命令参考《[一键量化使用指南](../../docs/zh/user_guide/usage_quick_quantization.md)》。

### GLM-5 一键量化命令示例

#### GLM-5 W8A8量化

``` bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type GLM-5 \
  --quant_type w8a8 \
  --trust_remote_code True
```

- 其中`MODEL_PATH`为GLM-5模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[glm_5_w8a8.yaml](../../lab_practice/glm_5/glm_5_w8a8.yaml)，可以在其中查看具体的量化策略。

#### GLM-5 W4A8量化

``` bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type GLM-5 \
  --quant_type w4a8 \
  --trust_remote_code True
```

- 其中`MODEL_PATH`为GLM-5模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[glm_5_w4a8.yaml](../../lab_practice/glm_5/glm_5_w4a8.yaml)，可以在其中查看具体的量化策略。

### GLM-5.1 一键量化命令示例

#### GLM-5.1 W8A8量化

``` bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type GLM-5.1 \
  --quant_type w8a8 \
  --trust_remote_code True
```

- 其中`MODEL_PATH`为GLM-5.1模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[glm_5_1_w8a8.yaml](../../lab_practice/glm_5/glm_5_1_w8a8.yaml)，可以在其中查看具体的量化策略。

#### GLM-5.1 W4A8量化

``` bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type GLM-5.1 \
  --quant_type w4a8 \
  --trust_remote_code True
```

- 其中`MODEL_PATH`为GLM-5.1模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[glm_5_1_w4a8.yaml](../../lab_practice/glm_5/glm_5_1_w4a8.yaml)，可以在其中查看具体的量化策略。

#### GLM-5.1 W8A8C8量化

``` bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type GLM-5.1 \
  --quant_type w8a8c8 \
  --trust_remote_code True
```

- 其中`MODEL_PATH`为GLM-5.1模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[glm_5_1_w8a8c8.yaml](../../lab_practice/glm_5/glm_5_1_w8a8c8.yaml)，可以在其中查看具体的量化策略。

#### GLM-5.1 W4A8C8量化

``` bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type GLM-5.1 \
  --quant_type w4a8c8 \
  --trust_remote_code True
```

- 其中`MODEL_PATH`为GLM-5.1模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[glm_5_1_w4a8c8.yaml](../../lab_practice/glm_5/glm_5_1_w4a8c8.yaml)，可以在其中查看具体的量化策略。

#### GLM-5.1 W4A4C8 (mxfp4)量化

``` bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type GLM-5.1 \
  --quant_type w4a4c8 \
  --trust_remote_code True \
  --tags vLLM_Ascend Ascend_950
```

- 其中`MODEL_PATH`为GLM-5.1模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[glm_5_1_w4a4c8_mxfp4.yaml](../../lab_practice/glm_5/glm_5_1_w4a4c8_mxfp4.yaml)，可以在其中查看具体的量化策略。

### GLM-5.2 一键量化命令示例

#### GLM-5.2 W8A8量化

``` bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type GLM-5.2 \
  --quant_type w8a8 \
  --trust_remote_code True
```

- 其中`MODEL_PATH`为GLM-5.2模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[glm_5_2_w8a8.yaml](../../lab_practice/glm_5_2/glm_5_2_w8a8.yaml)，可以在其中查看具体的量化策略。

#### GLM-5.2 W8A8C8量化

``` bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type GLM-5.2 \
  --quant_type w8a8c8 \
  --trust_remote_code True
```

- 其中`MODEL_PATH`为GLM-5.2模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[glm_5_2_w8a8c8.yaml](../../lab_practice/glm_5_2/glm_5_2_w8a8c8.yaml)，可以在其中查看具体的量化策略。

### GLM-5.3-Flash 量化命令示例

GLM-5.3-Flash（`model_type=glm5_next`）为 45 层混合架构模型：34 层 KDA 线性注意力 + 11 层 DeepSeek 稀疏注意力（含 Indexer），并引入 MHC 超连接、288 专家 MoE 与 1 层 MTP 投机草稿层。当前提供**免校准 W8A8_DYNAMIC** 转换：仅对 FFN 侧投影（路由专家、共享专家、稠密 FFN 的 gate/up/down_proj）做 per-channel 对称 INT8 权重量化，激活值在推理时按 token 动态量化，无需校准数据集；KDA 注意力投影、稀疏注意力投影与 Indexer、MHC 超连接模块、MoE 路由、各层 Norm、视觉塔与 MTP 草稿层均保留 BF16，理由详见 [w8a8_dynamic.py](../../msmodelslim/model/glm5_next/w8a8_dynamic.py)。

#### GLM-5.3-Flash W8A8（免校准）量化

``` bash
python -c "
from msmodelslim.model.glm5_next.w8a8_dynamic import convert_to_w8a8_dynamic

convert_to_w8a8_dynamic('${MODEL_PATH}', '${SAVE_PATH}')
"
```

- 其中`MODEL_PATH`为GLM-5.3-Flash BF16模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该转换只读取`config.json`与 safetensors 权重，不加载模型权重之外的网络结构，因此不依赖特定 transformers 版本；后续带校准的量化需要 transformers 5.16.0 及以上版本才能构建`glm5_next`模型。
- 转换产物为`quant_model_description.json`、`quant_model_weights-xxxxx.safetensors`（及索引文件），可直接加载到 vLLM-Ascend 等推理框架。
- 带校准的 W8A8 一键量化暂未支持，量化配置参考[glm5_next_w8a8.yaml](../../lab_practice/glm5_next/glm5_next_w8a8.yaml)。
