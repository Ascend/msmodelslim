# GLM4-MOE 量化说明

## 模型介绍

[GLM-4.7](https://huggingface.co/zai-org/GLM-4.7)是智谱 AI 于 2025 年 12 月 23 日发布的开源旗舰大语言模型，主打Agentic Coding（智能体编程）、复杂推理与工具协同，在编码、长程任务与前端生成上表现突出。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/getting_started/install_guide.md)。
- transformers版本需要配置安装4.57.3版本：

  ```bash
  pip install transformers==4.57.3
  ```

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接                                                 | W8A8 | W8A16 | W4A8 | W4A16 | W4A4  | 稀疏量化 | KV Cache | Attention | 量化命令                                          |
|---------|---------|---------------------------------------------------------------|-----|-----|-----|--------|------|---------|----------|-----------|-----------------------------------------------|
| **GLM4-MOE** | GLM-4.7 | https://huggingface.co/zai-org/GLM-4.7 | ✅ |  |  |        |   |  |   |   | [W8A8](#glm-47-w8a8量化)       |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 一键量化生成量化权重

一键量化命令参考[《一键量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/feature_guide/quick_quantization_v1/usage.md)。

### <span id="glm-47-w8a8量化">GLM-4.7 一键量化命令示例</span>

```python
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu:0,1,2,3,4,5,6,7 \
  --model_type GLM-4.7 \
  --quant_type w8a8 \
  --trust_remote_code True
```

- 其中`MODEL_PATH`为GLM-4.7模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[glm4_7_moe-w8a8-v1.yaml](../../lab_practice/glm4_moe/glm4_7_moe-w8a8-v1.yaml)，可以在其中查看具体的量化策略。
