# 多模态理解模型量化

多模态理解模型，也称为视觉语言模型（Vision-Language Models, VLM），具备强大的功能。可以处理图像、视频和文本等多种数据类型，并能执行多种下游任务，如理解图像内容并生成相应的自然语言描述。

## 使用前准备

- 请选用8.2.RC1及之后的配套CANN版本。
- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../docs/zh/getting_started/install_guide.md)。
- **不同的多模态理解模型依赖的transformers版本、第三方库有所差异，请务必参考各个多模态理解模型的量化说明进行配置。**

## 已验证量化模型

| 模型 | 支持量化 | 权重链接 | 量化部署支持 | 量化推荐实践 |
|-----------|-----------|---------|---------|---------|
| LLaVA | W8A8静态量化 | [LLaVA-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/a272c74b2481d8aff3aa6fc2c4bf891fe57334fb) | MindIE当前不支持<br>vLLM Ascend当前不支持 | 《[LLaVA 量化使用说明](./LLaVA/README.md)》 |
| Qwen-VL | W8A8静态量化 | [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL/tree/main) | MindIE当前不支持<br>vLLM Ascend当前不支持 | 《[Qwen-VL 量化使用说明](./Qwen-VL/README.md)》 |
| InternVL2 | W8A8静态量化 | [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main)<br>[InternVL2-40B](https://huggingface.co/OpenGVLab/InternVL2-40B/tree/main) | MindIE当前不支持<br>vLLM Ascend当前不支持 | 《[InternVL2 量化使用说明](./InternVL2/README.md)》 |
| Qwen2-VL | W8A8静态量化 | [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/tree/main)<br>[Qwen2-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct/tree/main) | MindIE 2.1.RC1及之后版本支持<br>vLLM Ascend当前不支持 | 《[Qwen2-VL 量化使用说明](./Qwen2-VL/README.md)》 |
| Qwen2.5-VL | W8A8静态量化，W4A8动态量化【部署暂不支持】 | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)<br>[Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/tree/main) | MindIE 2.2.RC1及之后版本支持<br>vLLM Ascend v0.10.2rc2及之后版本支持 | 《[Qwen2.5-VL 量化使用说明](./Qwen2.5-VL/README.md)》 |
| Qwen2.5-Omni | W8A8动态量化 | [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | MindIE 待支持<br>vLLM Ascend 支持中 | 《[Qwen2.5-Omni 量化使用说明](./Qwen2.5-Omni/README.md)》 |
| Qwen3-VL | W8A8量化，W8A8SC量化【vLLM Ascend暂不支持】 | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)<br>[Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | MindIE 预计3.0.RC1版本支持<br>vLLM Ascend v0.13.0及之后版本支持 | 《[Qwen3-VL 量化使用说明](./Qwen3-VL/README.md)》 |
| Qwen3-VL-MoE | W8A8混合量化（MoE专家动态量化） | [Qwen3-VL-30B-A3B](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main)<br>[Qwen3-VL-235B-A22B](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/tree/main) | MindIE 待支持<br>vLLM Ascend 支持中 | 《[Qwen3-VL-MoE 量化使用说明](./Qwen3-VL-MoE/README.md)》 |
| Qwen3-Omni | W8A8量化 | [Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)<br>[Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | vLLM Ascend | 《[Qwen3-Omni 量化使用说明](./Qwen3-Omni/README.md)》 |
| GLM-4.1V | W8A8SC量化 | [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking/tree/main) | MindIE 预计3.0.RC1版本支持<br>vLLM Ascend 当前不支持 | 《[GLM-4.1V 量化使用说明](./GLM-4.1V/README.md)》 |
| GLM-4.6V | W8A8混合量化（MoE专家动态量化） | [GLM-4.6V](https://huggingface.co/zai-org/GLM-4.6V) | MindIE 待支持<br>vLLM Ascend 支持中 | 《[GLM-4.6V 量化使用说明](./GLM-4.6V/README.md)》 |
| Kimi-K2.5 | W4A8混合量化（MoE专家动态量化） | [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) | MindIE 待支持<br>vLLM Ascend 支持中 | 《[Kimi-K2.5 量化使用说明](./Kimi-K2.5/README.md)》 |
| Kimi-K2.6 | W4A8混合量化（MoE专家动态量化） | [Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) | MindIE 待支持<br>vLLM Ascend 支持中 | 《[Kimi-K2.6 量化使用说明](./Kimi-K2.6/README.md)》 |
| InternVL3.5 | W8A8量化 | [InternVL3_5-38B](https://huggingface.co/OpenGVLab/InternVL3_5-38B) | MindIE 待支持<br>vLLM Ascend | 《[InternVL3.5 量化使用说明](./InternVL3_5/README.md)》 |
| InternVL3.5-MoE | W8A8混合量化（MoE专家动态量化） | [InternVL3_5-241B-A28B](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B) | MindIE 待支持<br>vLLM Ascend | 《[InternVL3.5-MoE 量化使用说明](./InternVL3_5-MoE/README.md)》 |
