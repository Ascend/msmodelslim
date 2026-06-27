# GLM5-MOE 量化说明

## 模型介绍

[GLM-5](https://huggingface.co/zai-org/GLM-5)是智谱 AI 于 2026 年 2 月 1 日发布的开源旗舰大语言模型，与 GLM-4.5 相比，GLM-5 的参数规模从 355B 扩展至 744B ，预训练数据量也从 23 万亿个 token 增加到 28.5 万亿个 token。GLM-5 还集成了 DeepSeek 稀疏注意力机制 (DSA)，在大幅降低部署成本的同时，保持了长时域上下文处理能力。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../docs/zh/install_guide/install_guide.md)。
- transformers版本需要配置安装 5.4.0 版本：

  ```bash
  pip install transformers==5.4.0
  ```

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接                                                 | W8A8 | W8A8C8 | W8A16 | W4A8 | W4A8C8 | W4A16 | W4A4  | 稀疏量化 | KV Cache | Attention | 量化命令                                          |
|---------|---------|---------------------------------------------------------------|-----|-----|-----|-----|-----|--------|------|---------|----------|-----------|-----------------------------------------------|
| **GLM5-MOE** | GLM-5 | <https://huggingface.co/zai-org/GLM-5> | ✅ |  |  | ✅ |  |        |   |  |   |   ||
| **GLM5-MOE** | GLM-5.1 | <https://huggingface.co/zai-org/GLM-5.1> | ✅ | ✅ |  | ✅ | ✅ |        |   |  |   |   ||

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 一键量化生成量化权重

一键量化命令参考[《一键量化使用指南》](../../docs/zh/user_guide/feature_guide/quick_quantization_v1/usage.md)。

### GLM-5 一键量化命令示例

``` bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu:0 \
  --model_type GLM-5 \
  --quant_type w8a8 \
  --trust_remote_code True
```

- 其中`MODEL_PATH`为GLM-5模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[glm_5_w8a8.yaml](../../lab_practice/glm_5/glm_5_w8a8.yaml)，可以在其中查看具体的量化策略。

### GLM-5.1 一键量化命令示例

``` bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu:0 \
  --model_type GLM-5.1 \
  --quant_type w8a8 \
  --trust_remote_code True
```

- 其中`MODEL_PATH`为GLM-5.1模型的路径，`SAVE_PATH`为量化后的权重保存路径。
- 如需使用 W4A8 量化，将 `--quant_type` 替换为 `w4a8` 即可。
- 该一键量化命令匹配使用的量化配置文件为[glm_5_1_w8a8.yaml](../../lab_practice/glm_5/glm_5_1_w8a8.yaml)，可以在其中查看具体的量化策略。
