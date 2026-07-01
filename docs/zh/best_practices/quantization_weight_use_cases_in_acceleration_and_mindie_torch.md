# 加速库场景下W8A8量化权重的使用案例

## 概述

量化工具与MindIE工具关系：msModelSlim作为量化工具提供量化能力，MindIE加速库可以调用量化权重进行推理。

## 前期准备

请参考《[MindIE安装指南](https://www.hiascend.com/document/detail/zh/mindie/10RC3/envdeployment/instg/mindie_instg_0001.html)》安装MindIE，并参考《MindIE安装指南》中“配置MindIE > [配置MindIE LLM](https://www.hiascend.com/document/detail/zh/mindie/10RC3/envdeployment/instg/mindie_instg_0028.html)”章节配置MindIE LLM。

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../install_guide/install_guide.md)。

## 量化权重生成

  此处以Llama2-13b-hf为例

  （1）准备一份模型权重数据。

```tex
├── config.json
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json
├── pytorch_model-00001-of-00006.bin
├── ...
├── pytorch_model-00006-of-00006.bin
├── pytorch_model.bin.index.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── tokenizer.model
```

  （2）使用指令生成W8A8量化权重。

```shell
# 进入加速库路径下
cd ${ATB_SPEED_HOME_PATH}
# 运行脚本生成量化权重
bash examples/models/llama/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重保存路径} -type llama2_13b_w8a8

# 以上指令展示了生成Llama2-13b-hf W8A8权重，不同模型的参数配置不同，请参考模型Readme文件。
# W8A8量化权重的config.json中应包含quantize字段，其值为"w8a8"。
# MindIE量化脚本除了调用msModelSlim量化工具生成权重及权重描述外，还复制了tokenizer文件以及复制并修改了config.json到量化权重保存路径中。
```

  （3）量化后权重目录结构：

```tex
├─ config.json
├─ quant_model_weight_w8a8.safetensors
├─ quant_model_description_w8a8.json
├─ tokenizer_config.json
├─ tokenizer.json
└─ tokenizer.model
```

量化输出包含：权重文件quant_model_weight_w8a8.safetensors和权重描述文件quant_model_description_w8a8.json。

目录中的其余文件为推理时所需的配置文件，不同模型略有差异。

在 safetensors 中，数据以字典格式存储，包含两部分内容：量化后的权重，以及未经过量化处理的浮点权重。其中，量化权重的键（key）命名规则是：各层 Linear 的名称加上其对应权重的名称。

以下展示了量化后权重文件quant_model_weight_w8a8.safetensors中的部分内容：

```tex
{
  "model.embed_tokens.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_scale": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_offset": tensor([...]),
  "model.layers.0.self_attn.q_proj.quant_bias": tensor([...]),
  "model.layers.0.self_attn.q_proj.deq_scale": tensor([...]),
  "model.layers.0.self_attn.k_proj.weight": tensor([...]),
 ...
}
```

以下展示了量化后权重描述文件quant_model_description_w8a8.json中的部分内容：

```tex
{
  "model_quant_type": "W8A8",                               # 整体量化类型为W8A8量化
  "model.embed_tokens.weight": "FLOAT",                     # 来自原始浮点模型的embed_tokens权重
  "model.layers.0.self_attn.q_proj.weight": "W8A8",         # 量化新增的第0层self_attn.q_proj的quant_weight
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8",    # 量化新增的第0层self_attn.q_proj的input_scale
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8",   # 量化新增的第0层self_attn.q_proj的input_offset
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8",     # 量化新增的第0层self_attn.q_proj的quant_bias
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8",      # 量化新增的第0层self_attn.q_proj的deq_scale
  "model.layers.0.self_attn.k_proj.weight": "W8A8",         # 量化新增的第0层self_attn.k_proj的quant_weight
 ...
}
```

  （4）运行推理
以Llama2-13b-hf为例，您可以使用以下指令执行对话测试，推理内容为"What's deep learning"。详细可参考昇腾社区开发指南参考链接：[MindIE LLM 开发指南](https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindiellm/llmdev/mindie_llm0281.html)。

```shell
# 进入加速库路径下
cd ${ATB_SPEED_HOME_PATH}
# 运行推理脚本
bash examples/models/llama/run_pa.sh {W8A8量化权重路径} 20
# 参数20为推理结果最大长度
```

预期推理结果：

```coldfusion
Question: "What's deep learning"
Answer: Deep learning is a subset of machine learning that uses artificial neural networks to learn from data.
```
