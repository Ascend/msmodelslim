# Qwen3-VL-Embedding & Reranker 量化案例

## 模型介绍

 - [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) 是基于 Qwen3-VL打造的开源多模态向量化模型，可把文本、图片、图文文档、视频转为同空间向量，主打跨模态检索，MMEB-V2 评测登顶 SOTA，支持 MRL 动态降维与 32K 超长输入。
 - [Qwen3-VL-Reranker](https://github.com/QwenLM/Qwen3-VL-Embedding) 依托 Qwen3-VL 视觉语言底座，对图文候选结果做细粒度相关性打分重排，大幅优化多模态检索与多模态 RAG 的召回准确率。

## 环境配置

- 基础环境配置请参考《[安装指南](../../../docs/zh/getting_started/install_guide.md)》，注意：由于高版本transformers的特殊性，PyTorch及torch_npu需要配置安装为≥2.2版本。
- 针对 Qwen3-VL-Embedding & Reranker模型transformers 版本需要 4.57.1：

  ```bash
  pip install transformers==4.57.1
  ```

## Qwen3-VL-Embedding & Reranker模型当前已验证的量化方法

| 模型                    | 原始浮点权重                                                                               | 量化方式   | 推理框架支持情况                   |
|-----------------------|--------------------------------------------------------------------------------------|--------|----------------------------|
| Qwen3-VL-Embedding-2B | [Qwen3-VL-Embedding-2B](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Embedding-2B) | W8A8量化 | vLLM Ascend v0.19.0及之后版本支持 |
| Qwen3-VL-Embedding-8B | [Qwen3-VL-Embedding-8B](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Embedding-8B) | W8A8量化 | vLLM Ascend v0.19.0及之后版本支持 |
| Qwen3-VL-Reranker-2B  | [Qwen3-VL-Reranker-2B](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Reranker-2B)   | W8A8量化 | vLLM Ascend v0.19.0及之后版本支持 |
| Qwen3-VL-Reranker-8B  | [Qwen3-VL-Reranker-8B](https://www.modelscope.cn/models/Qwen/Qwen3-VL-Reranker-8B)   | W8A8量化 | vLLM Ascend v0.19.0及之后版本支持 |


### 使用案例

- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`，让修改后的自定义代码文件能够正确地被加载（请确保加载的自定义代码文件的安全性）。

#### 1. Qwen3-VL-Embedding

该示例在NPU上生成Qwen3-VL-Embedding-2B模型的量化权重。

一键量化命令如下：

```shell
msmodelslim quant --model_path /path/to/Qwen3-VL-Embedding-2B \
--save_path /path/to/Qwen3-VL-Embedding-2B-quantied \
--device npu \
--model_type Qwen3-VL-Embedding-2B \
--quant_type w8a8
  ```
#### 2. Qwen3-VL-Reranker

该示例在NPU上生成Qwen3-VL-Reranker-2B模型的量化权重。

一键量化命令如下：

```shell
msmodelslim quant --model_path /path/to/Qwen3-VL-Reranker-2B \
--save_path /path/to/Qwen3-VL-Reranker-2B-quantied \
--device npu \
--model_type Qwen3-VL-Reranker-2B \
--quant_type w8a8
  ```

- 更多参数配置要求，请参考《[参数说明](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md)》。


> [!Note] 说明
>
> - Qwen3-VL-Embedding & Qwen3-VL-Reranker 默认精度为`bfloat16`，建议在支持bfloat16精度的设备上量化权重，若修改模型权重路径下`config.json`中的`torch_dtype`为`float16`进行量化，可能会导致模型精度异常。
> - 若硬件只支持float16精度推理（例如Atlas 300I/300T系列），建议采用默认精度`bfloat16`量化后将模型权重路径下`config.json`中的`torch_dtype`修改为`float16`进行推理。
