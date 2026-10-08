# msmodelslim量化权重格式

msmodelslim llm-ptq工具生成的safetensors量化权重文件包含两个文件，quant_model_weight.safetensors权重文件和quant_model_description.json权重描述文件

msmodelslim量化类型说明：
[W8A16](#2-w8a16量化): Linear权重int8量化，激活值不量化
[W8A8](#3-w8a8-w8a8s量化): Linear权重int8量化，激活值int8量化
[W8A8S](#3-w8a8-w8a8s量化): Linear权重int8稀疏量化，激活值int8量化

**注意** msmodelslim工具生成的量化权重均为signed场景，即int8数据分布范围为-128到127。开源权重若为unsigned场景，对于int8可以考虑将weight和offset权重减去128

脚本convert_example.py提供了将开源ChatGLM2-6B转换成msmodelslim量化权重的示例，权重获取链接见[开源模型README](https://github.com/zai-org/ChatGLM2-6B?tab=readme-ov-file#%E4%BD%8E%E6%88%90%E6%9C%AC%E9%83%A8%E7%BD%B2) 低成本部署章节，使用前请修改224行和225行的输入输出路径。使用方式`python convert_example.py`

## 1. 量化权重、描述文件格式

### 1.1 safetensors权重格式

权重保存为safetensors格式，内部格式为python的字典 dict，包含量化权重和量化不修改的浮点权重，字典的key值为权重名称，value为具体权重的数值
以ChatGLM2-6B为例：'transformer.embedding.word_embeddings.weight'为浮点模型中word_embedding层的权重，名称和权重均未修改，对应描述文件量化类型为'FLOAT'；'transformer.encoder.layers.0.self_attention.dense.weight'为原始模型第0层layer的dense层linear的权重，经过量化修改，数据类型为int8，对应描述文件量化类型为'W8A16'；'transformer.encoder.layers.0.self_attention.dense.weight_scale'为原始模型第0层layer的dense层linear量化后新增的量化参数weight_scale，对应描述文件量化类型为'W8A16'

示例 ChatGLM2-6B W8A16量化权重：

```text
{
    'transformer.embedding.word_embeddings.weight': tensor([...]),
    'transformer.encoder.final_layernorm.weight': tensor([...]),
    'transformer.encoder.layers.0.input_layernorm.weight': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_4h_to_h.weight': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_4h_to_h.weight_scale': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_4h_to_h.weight_offset': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_h_to_4h.weight': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_h_to_4h.weight_scale': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_h_to_4h.weight_offset': tensor([...]),
    'transformer.encoder.layers.0.post_attention_layernorm.weight': tensor([...]),
    'transformer.encoder.layers.0.self_attention.dense.weight': tensor([...]),
    'transformer.encoder.layers.0.self_attention.dense.weight_scale': tensor([...]),
    'transformer.encoder.layers.0.self_attention.dense.weight_offset': tensor([...]),
    'transformer.encoder.layers.0.self_attention.query_key_value.weight': tensor([...]),
    'transformer.encoder.layers.0.self_attention.query_key_value.weight_scale': tensor([...]),
    'transformer.encoder.layers.0.self_attention.query_key_value.weight_offset': tensor([...]),
    ...
    剩下几层以此类推
    ...
    'transformer.output_layer.weight': tensor([...]),
    'transformer.rotary_pos_emb.inv_freq': tensor([...])
}
```

### 1.2 json描述文件格式

json描述文件内部存储格式为python的字典 dict，字典的key值为权重名称，value为权重对应的量化类型。"model_quant_type"描述整体的量化类型，"kv_cache_type"表示kv_cache是否量化，其余为各个权重的类型，"FLOAT"表示来自浮点权重，"W8A8"表示来自W8A8量化，"W8A16"表示来自W8A16量化，"W8A8S"表示来自稀疏量化
示例 ChatGLM2-6B W8A16量化权重的描述文件：
描述文件字典内容排序不影响实际使用

```text
{
    "model_quant_type": "W8A16",
    "kv_cache_type": "C8", # 使用kv cache量化后会生成该行
    "transformer.embedding.word_embeddings.weight": "FLOAT",
    "transformer.rotary_pos_emb.inv_freq": "FLOAT",
    "transformer.encoder.layers.0.input_layernorm.weight": "W8A16",
    "transformer.encoder.layers.0.self_attention.query_key_value.weight": "W8A16",
    # 使用kv cache量化后生成如下4行
    "transformer.encoder.layers.0.self_attention.query_key_value.k_proj.kv_cache_scale": "W8A16",
    "transformer.encoder.layers.0.self_attention.query_key_value.k_proj.kv_cache_offset": "W8A16",
    "transformer.encoder.layers.0.self_attention.query_key_value.v_proj.kv_cache_scale": "W8A16",
    "transformer.encoder.layers.0.self_attention.query_key_value.v_proj.kv_cache_offset": "W8A16",
    "transformer.encoder.layers.0.self_attention.query_key_value.weight_scale": "W8A16",
    "transformer.encoder.layers.0.self_attention.query_key_value.weight_offset": "W8A16",
    "transformer.encoder.layers.0.post_attention_layernorm.weight": "FLOAT",
    "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight": "W8A16",
    "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight_scale": "W8A16",
    "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight_offset": "W8A16",
    ...
    剩下几层以此类推
    ...
    "transformer.encoder.final_layernorm.weight": "FLOAT",
    "transformer.output_layer.weight": "FLOAT"
}
```

## 2. W8A16量化

量化工具对于每个量化的Linear生成3个参数，参数名称为：`weight`、`weight_scale`、`weight_offset`，在safetensors权重文件中，完整的权重名称为Linear层的名称+参数名称，例如ChatGLM2-6B量化权重中，`"transformer.encoder.layers.0.self_attention.query_key_value.weight_scale"`，`"transformer.encoder.layers.0.self_attention.query_key_value"`为Linear层的名称，`"weight_scale"`为参数的名称

### 2.1 权重说明

`weight`为量化后的int8的权重，数据类型为torch.Tensor，dtype为torch.int8，shape和原始浮点的shape一致，记为n, k = weight.shape，k为hidden_size
`weight_scale`为量化的缩放系数，数据类型为torch.Tensor，dtype为torch.float32，在per_channel场景下，shape为[n]，在per_group场景下，shape为[n, k / group_size]
`weight_offset`为量化的偏移系数，数据类型为torch.Tensor，dtype和shape和weight_scale一致。对称量化场景下需要构造全0的weight_offset

### 2.2 反量化计算公式

per_channel 场景：

```python
deq_weight = (weight - weight_offset) * weight_scale
```

per_group 场景：

```python
weight = weight.reshape((-1, group_size))
weight_offset = weight_offset.reshape((n * k / group_size, 1))
weight_scale = weight_scale.reshape((n * k / group_size, 1))
deq_weight = ((weight - weight_offset) * weight_scale).reshape((n, k))
```

**注意** NPU量化算子计算时实际的逻辑为(weight + weight_offset) * weight_scale，昇腾推理框架在加载量化权重时进行了取负操作

代码实现可以参考demo样例MSModelSlimWeightProcessor.weight_process，请根据开源权重的反量化公式和msmodelslim工具的反量化公式进行相应修改

## 3. W8A8, W8A8S量化

msmodelslim量化工具对于每个量化的Linear生成5个参数，参数名称为：weight, input_scale, input_offset, deq_scale, quant_bias
在safetensors权重文件中，完整的权重名称为Linear层的名称+参数名称，与W8A16类似

### 3.1 权重说明

`weight`为量化后的int8的权重，数据类型为torch.Tensor，dtype为torch.int8，shape和原始浮点的shape一致，记为n, k = weight.shape，k为hidden_size
`input_scale`为激活值量化的缩放系数，数据类型为torch.Tensor，dtype为torch.float16或torch.bfloat16，shape为[1]
`input_offset`为激活值量化的偏移系数，数据类型为torch.Tensor，dtype和shape和input_scale一致。
`deq_scale`为反量化缩放系数，数据类型为torch.Tensor，dtype为torch.int64或torch.float32，shape为[n]。注意为了亲和昇腾量化算子，开源量化若基于fp16，则deq_scale的数据在传给量化算子前需要进行数据类型转换，可以参考示例代码120行进行处理，若开源量化权重为bf16，则不需要数据类型转换
`quant_bias`为反量化的偏移系数，数据类型为torch.Tensor，dtype为torch.int32，shape为[n]

### 3.2 量化、反量化计算公式

```python
input_quant = input_fp / input_scale + input_offset
output_quant = input_quant * weight + quant_bias
output_dequant = output_quant * deq_scale
```

代码实现可以参考demo样例MSModelSlimWeightProcessor.weight_activation_process，请根据开源权重的计算公式和msmodelslim工具的计算公式进行相应修改

## 4. smooth quant

msmodelslim量化工具使用smooth quant后，对于每个norm层，生成2个参数，module.weight和module.bias。完整的权重名称为norm层的名称+参数名称，例如ChatGLM2-6B量化权重中，`"transformer.encoder.layers.0.input_layernorm.module.weight"`，`"transformer.encoder.layers.0.input_layernorm"`为norm层的名称，`"module.weight"`为量化参数名称

msmodelslim量化工具集成的smooth quant算法针对norm层后的Linear层进行smooth平滑操作，而不是所有Linear层。采取这种量化方案的优势在于可以将原本乘在激活值上的scale等价转移到原始浮点模型norm层的权重`norm.weight`上，从而避免额外引入算子带来的性能开销

`module.weight`为scale后的norm.weight，数据类型、dtype、shape和norm.weight一致
`module.bias`为引入module.weight后带来的偏移系数，数据类型、dtype、shape和norm.weight一致

为了适配几种特殊的回退情况，msmodelslim生成的smooth quant权重中还包含原始浮点权重norm层的权重，norm.weight。如果开源量化权重不涉及回退场景，设置为None即可

代码实现可以参考demo样例MSModelSlimWeightProcessor.anti_outlier_process，请根据开源权重的设计方案和msmodelslim工具的设计方案进行相应修改

## 5. KV Cache量化

msmodelslim工具提供的KV Cache量化采用int8量化。对于每个attention层，生成4个参数，`k_proj.kv_cache_scale`, `k_proj.kv_cache_offset`, `v_proj.kv_cache_scale`, `v_proj.kv_cache_offset`。对于qkv合并或kv合并的场景，完整的四个参数的名称为合并的Linear名称+参数名；对于qkv分离场景，k_proj的scale、offset完整的参数名称为k对应Linear名称+参数名称，v_proj的scale、offset完整的参数名称为v对应Linear名称+参数名称。例如`"transformer.encoder.layers.0.query_key_value.k_proj.kv_cache_scale"`，`"transformer.encoder.layers.0.query_key_value"`为qkv合并的Linear层名称，`"k_proj.kv_cache_scale"`为参数名称

### 5.1 权重说明

`kv_cache_scale`为kvcache量化scale的缩放系数，数据类型为torch.Tensor，dtype为torch.float32或torch.float16，shape为kv channel的size，如果是qkv分开场景，则为k或v层linear的n维（见上文**w8a16量化** 章节 **权重说明** weight的shape说明）
`kv_cache_offset`为kvcache量化scale的偏移系数，数据类型、dtype、shape和scale一致

计算公式：
量化

```python
cache_int = cache_fp / cache_scale + cache_offset
```

反量化

```python
cache_deq = (cache_int - cache_offset) * cache_scale
```

代码实现可以参考demo样例MSModelSlimWeightProcessor.kv_cache_process，请根据开源权重的计算公式和msmodelslim工具的计算公式进行相应修改
