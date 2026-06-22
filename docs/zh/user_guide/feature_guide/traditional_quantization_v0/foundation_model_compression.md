# 压缩与结构优化（大模型为主）

## 1. 大模型量化

大模型量化工具将高位浮点数转为低位的定点数，例如16bit降低到8bit，直接减少模型权重的体积，生成量化参数和权重文件。在无需训练成本的前提下，完成大模型的训练后压缩并最大程度保障其精度。

### 1.1 使用前准备

- 仅支持在以下产品中使用。
    - Atlas 推理系列产品（Atlas 300I Duo 推理卡）。
    - Atlas 训练系列产品。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件。

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。
- 大模型量化工具须执行命令安装如下依赖。
  以下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install onnx --user。针对某些开发环境可能存在依赖不完全匹配的情况，请根据界面报错提示自行修改依赖版本。

```bash
pip3 install numpy==1.25.2
pip3 install transformers        #需大于等于4.29.1版本，LLaMA模型需指定安装4.29.1版本
pip3 install accelerate==0.21.0  #若需要使用NPU多卡并行方式对模型进行量化，需大于等于0.28.0版本
pip3 install tqdm==4.66.1
```

- （可选）如果需要在大模型量化工具中使用NPU多卡并行的方式对模型进行量化，需关闭NPU设备中的虚拟内存，并手动配置量化将会执行的设备序列环境。

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False # 关闭NPU的虚拟内存
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 #配置量化将会执行的设备序列环境
```

说明
仅Atlas 训练系列产品和Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件支持此功能。

- （可选）如果需要在大模型量化工具中使用NF的权重量化方式，
说明
仅Atlas 训练系列产品和Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件支持此功能。
- （可选）如果需要在大模型量化工具中使用W4A4_FLATQUANT_DYNAMIC量化方式，仅Atlas 训练系列产品和Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件支持此功能。

### 1.2 功能介绍

参考[已验证模型列表](../../model_support/foundation_model_support_matrix.md)

### 1.3 说明

大模型压缩技术主要针对常规大语言模型进行量化压缩，但在量化拥有特殊结构的模型时，msModelSlim工具可能存在以下限制：

MOE模型支持W8A8_per-token量化场景、W8A16 per-channel量化场景和W8A16 per-group量化场景，不支持lowbit稀疏量化场景。
多模态模型仅支持W8A16量化场景，不支持W8A8量化场景和lowbit算法稀疏量化场景。

### 1.4 功能实现流程

图1 量化接口调用流程

![量化接口调用流程](./figures/[pytorch]quantization_api_calling.png)

关键步骤说明如下：

1. 用户准备原始模型和校准数据。

2. 可选：使用离群值抑制功能对LLM模型进行离群值抑制，可参考精度保持策略选择是否启用。
    - 使用AntiOutlierConfig生成离群值抑制配置。
    - 调用AntiOutlier接口，将模型、校准数据等传入，生成抑制器。
    - 调用抑制器的process()方法对原始模型进行离群值抑制。

3. 使用QuantConfig生成量化配置。

4. 根据原始模型、量化配置和校准数据，调用Calibrator接口构建量化校准对象。

5. 调用生成的量化校准对象的run()方法对原始模型进行量化。

6. 调用生成的量化校准对象的save()接口保存量化后的模型，包括模型量化权重和模型相关参数，用于后续量化模型的部署任务，具体请参见MindIE的[“加速库支持模型列表”](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0001.html)章节中已适配量化的模型。

### 1.5 量化步骤（以ChatGLM2-6B为例）

1.用户自行准备模型、权重文件和校准数据，本样例以ChatGLM2-6B为例，目录示例如下：

```text
├── config.json
├── configuration_chatglm.py
├── modeling_chatglm.py
├── pytorch_model-00001-of-00007.bin
├── pytorch_model-00002-of-00007.bin
├── pytorch_model-00003-of-00007.bin
├── pytorch_model-00004-of-00007.bin
├── pytorch_model-00005-of-00007.bin
├── pytorch_model-00006-of-00007.bin
├── pytorch_model-00007-of-00007.bin
├── pytorch_model.bin.index.json
├── quantization.py
├── README.md
├── tokenization_chatglm.py
├── tokenizer.model
├── tokenizer_config.json
```

2.ChatGLM2-6B模型进行量化前请执行如下命令安装所需依赖，若运行量化工具过程中提示缺失某个依赖，请根据提示安装。

```bash
pip3 install protobuf==4.24.1
pip3 install sentencepiece==0.1.99
pip3 install sympy==1.11.1
pip3 install transformers==4.43.0 # 参考ChatGLM2-6B仓chatglm2-6b/config.json里的相关版本要求
```

3.新建模型的quant.py量化脚本，编辑quant.py文件，根据实际的量化场景导入样例代码，并根据实际情况进行修改。

    - W8A8 per_channel量化场景导入的样例代码如下，kvcache、lowbit算法以及per_token算法量化场景导入的代码样例请参考[w8a8量化场景](quantization_and_sparse_quantization_scenario_import_code_examples.md)。

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在cpu上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True) # 若存在自定义代码，需要配置参数trust_remote_code=True，请确保加载的modeling文件的安全性。
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
  ).npu()    # 若在npu上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto',创建model时需去掉.npu()；若在cpu上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu();  若存在自定义代码，需要配置参数trust_remote_code=True，请确保加载的modeling文件的安全性。
# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'],
    dev_id=model.device.index,
    dev_type='npu',   # 在cpu进行量化时，需配置参数dev_type='cpu'，并取消dev_id=model.device.index参数的配置
    act_method=3,
    pr=0.5,
    mm_tensor=False
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径
print('Save quant weight success!')
```

W8A16或W4A16 per_channel量化场景导入的样例代码如下，MinMax算法、HQQ算法、GPTQ算法、AWQ算法以及w4a16 per-group量化场景导入的代码样例请参考w8a16或w4a16量化场景。

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在cpu上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
    ).npu()    # 若在npu上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在cpu上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改，W8A16 Label-Free模式下请忽略此步骤
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    w_bit=8,     # W4A16场景下，w_bit值需配置为4。在W4A16的per_group场景下，需参考W4A16的per_group量化场景参数进行设置。
    a_bit=16,
    disable_names=[],
    dev_id=model.device.index,
    dev_type='npu',   # 在cpu进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=False,
    mm_tensor=False
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径
print('Save quant weight success!')
```

4.启动模型量化任务，并在指定的输出目录获取模型量化参数，量化后权重文件的介绍请参见量化后权重文件，若使用MindIE进行后续的推理部署任务，请保存为safetensors格式，具体请参见MindIE的[“MindIE支持模型列表”](https://www.hiascend.com/software/mindie/modellist)章节中已适配量化的模型。

```bash
python3 quant.py
```

量化任务完成后，可能会存在模型精度下降的情况，可以参考精度保持策略进行配置优化减少精度损耗。

### 1.6 量化及稀疏量化场景导入代码样例

其他场景样例可参考[此处](quantization_and_sparse_quantization_scenario_import_code_examples.md)

### 1.7 量化后权重文件

- npy格式
当[save_type](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_save().md)设置为['numpy']或不设置时，量化权重会保存为npy文件，npy储存格式为字典，其中key值为各层Linear的名字，例如ChatGLM2-6B模型的transformer.encoder.layers.0.self_attention.query_key_value，value值为第0层query_key_value的Linear权重。

> 注意：w4a8_dynamic 量化类型不支持 ['numpy'] 格式保存。因此，当 save_type 设置为['numpy']时，会有报错提醒。当 save_type 设置为 ['numpy', 'safe_tensor']时，会保存 `safe_tensor` 格式数据；而对于 `numpy` 格式数据，会跳过保存，并会在日志中输出一个 error 提示。

```text
├── anti_fp_norm.npy   #LLaMA模型且已启用离群抑制功能，具体操作请参见使用离群值抑制功能，将会生成此文件。antioutlier算法生成浮点权重中的norm层权重文件，用于量化层的input和post norm的权重适配
├── deq_scale.npy      #W8A8量化和稀疏量化的量化参数权重文件，Tensor数据类型为int64，deq_scale已针对量化算子进行数据类型转换，可直接适配算子。在量化BF16模型情况下，数据类型不会转换为int64，仍然为float32
├── input_offset.npy   #W8A8量化和稀疏量化的激活值量化偏移值权重文件，Tensor数据类型为float32
├── input_scale.npy    #W8A8量化和稀疏量化的激活值量化缩放因子权重文件，Tensor数据类型为float32
├── kv_cache_offset.npy    #kv cache量化参数文件，kv linear激活值量化偏移值权重文件，Tensor数据类型为float32
├── kv_cache_scale.npy   #kv cache量化参数文件，kv linear激活值量化缩放因子权重文件，Tensor数据类型为float32
├── quant_bias.npy     #W8A8量化和稀疏量化的量化参数权重文件，Tensor数据类型为int32，quant_bias已考虑原始浮点模型linear层的bias值
├── quant_weight.npy   #量化权重文件，Tensor数据类型为int8
├── weight_offset.npy  #w8a16和w4a16权重量化参数文件，Tensor数据类型为float32
├── weight_scale.npy   #w8a16和w4a16权重量化参数文件，Tensor数据类型为float32
```

推理部署时读取上述文件的示例代码：quant_param_dict = np.load("xxx.npy", allow_pickle=True).item()。

- safetensors格式
当[save_type](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_save().md)设置为['safe_tensor']时，量化权重会保存为safetensors文件和json描述文件。
**说明**：当用户设置的[part_file_size](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_save().md)值大于0时，会使能PyTorch框架的分片保存功能。msModelSlim工具会统计遍历到的权重文件的大小，若权重文件的大小大于part_file_size值，则将统计到的权重作为一个part，然后重新进行统计。统计完成后，将各个权重分片保存，并生成权重索引文件（xxx.safetensors.index.json）。权重和索引的名称可参照开源模型的权重，例如xxx-0000x-of-0000x.safetensors，当part数大于99999时，权重和索引的名称将会被命名为xxx-x-of-x.safetensors。

    - safetensors中储存格式为字典，包含量化权重和量化不修改的浮点权重。其中量化权重的key值为各层Linear的名字加上对应权重的名字，module.weight和module.bias对应anti_fp_norm.npy，weight对应quant_weight.npy，quant_bias对应quant_bias.npy等以此类推。例如ChatGLM2-6B模型的transformer.encoder.layers.0.self_attention.query_key_value.deq_scale对应npy格式权重中deq_scale.npy中的transformer.encoder.layers.0.self_attention.query_key_value。

```text
# llama模型稀疏量化生成的权重文件部分内容
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

> 注意：当使用 w4a8_dynamic 量化类型时，safe_tensor 的内容会多生成一个 weight_scale_second 和 weight_offset_second 的 key 和对应的 tensor 值。

- json描述文件中储存的量化权重的总体类型model_quant_type，是否启用kvcache量化kv_cache_type，和其中各个权重的类型，来自原始浮点权重则为FLOAT，来自W8A8量化则为W8A8，来自稀疏量化则为W8A8S，来自压缩则为W8A8SC，来自NF4量化则为NF4, 来自W4A4 Flatquant量化方式则为W4A4_FLATQUANT_DYNAMIC。

```text
# llama模型稀疏量化生成的json描述文件部分内容
{
  "model_quant_type": "W8A8S",                               # 整体量化类型为稀疏量化
  "model.embed_tokens.weight": "FLOAT",                      # 来自原始浮点模型的embed_tokens权重
  "model.layers.0.self_attn.q_proj.weight": "W8A8S",         # 量化新增的第0层self_attn.q_proj的quant_weight
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8S",    # 量化新增的第0层self_attn.q_proj的input_scale
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8S",   # 量化新增的第0层self_attn.q_proj的input_offset
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8S",     # 量化新增的第0层self_attn.q_proj的quant_bias
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8S",      # 量化新增的第0层self_attn.q_proj的deq_scale
  "model.layers.0.self_attn.k_proj.weight": "W8A8S",         # 量化新增的第0层self_attn.k_proj的quant_weight
 ...
}
```

> 注意：当使用 w4a8_dynamic 量化类型时，json 描述文件中会多生成一个 model_quant_type_second 和 kv_cache_type_second 的 key 和对应的量化类型 W4A8_DYNAMIC。

- ascendV1格式
<br>当[save_type](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_save().md)['ascendV1']时，量化权重会保存为safetensors文件和json描述文件。其中，safetensors文件会跟设置的格式不同而有所区别，当设置为['ascendV1']并开启异常值抑制时不会导出*norm.module.bias/weight，而设置为['safe_tensor']时则会导出。json描述文件名从quant_model_description_{quant_type}.json改为quant_model_description.json，并新增了一个version参数。

> 注意：当使用w4a16量化类型时，默认会对权重进行pack。

```text
# llama模型稀疏量化生成的json描述文件部分内容
{
  "version": "1.0.0",                                        # 标注现有的权重格式保存版本
  "model_quant_type": "W8A8S",                               # 整体量化类型为稀疏量化
  "kv_cache_type": "C8",                                     # 对KVCache的量化
  "fa_quant_type": "FAQuant",                                # FA3量化类型，其他模型FA3量化都为FAQUant，但是DeepSeek为FAKQuant
  "group_size": "256",                                       # per_group量化时的分组数量
  "kv_quant_type": "C8",                                     # 对KV的量化类型
  "reduce_quant_type": "per_channel",                        # 通信量化类型
  "model.embed_tokens.weight": "FLOAT",                      # 来自原始浮点模型的embed_tokens权重
  "model.layers.0.self_attn.q_proj.weight": "W8A8S",         # 量化新增的第0层self_attn.q_proj的quant_weight
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8S",    # 量化新增的第0层self_attn.q_proj的input_scale
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8S",   # 量化新增的第0层self_attn.q_proj的input_offset
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8S",     # 量化新增的第0层self_attn.q_proj的quant_bias
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8S",      # 量化新增的第0层self_attn.q_proj的deq_scale
  "model.layers.0.self_attn.k_proj.weight": "W8A8S",         # 量化新增的第0层self_attn.k_proj的quant_weight
 ...
}
```

### 1.8 精度保持策略

在量化权重生成后，可以使用伪量化模型进行推理，检验伪量化精度是否正常。伪量化是指通过torch，通过浮点运算完成量化模型运算逻辑，运算过程中的数据和真实量化的数据差异只在算子精度上，同时可以规避接入推理框架时引入的精度误差。如果伪量化精度不满足预期，真实量化结果也将无法满足预期。在调用Calibrator.run()方法后，构建Calibrator时传入的model会被替换为伪量化模型，可以直接调用进行前向推理，用来测试对话效果。如果伪量化结果不理想，可先使用[精度定位方法](#181-精度定位方法)进行定位，再可以参考以下手段进行调优。一般来说，W8A16的精度调优较为容易，W8A8和稀疏量化的精度调优相对复杂。

#### 1.8.1 精度定位方法

（1）将safetensors文件和json描述文件上传至[FakeQuantizeCalibrator接口](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/FakeQuantizeCalibrator.md)，构建FakeQuantizeCalibrator时传入的model会被替换为伪量化模型，可以直接调用进行前向推理，测试对话效果，并调用精度测试接口测试量化后模型权重的精度情况。

- 说明：支持W8A8（per_channel）、W8A16 per-channel（MinMax、GPTQ、HQQ）场景。

（2）调用[Calibrator.run()](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_run().md)，构建Calibrator时传入的model会被替换为伪量化模型，可以直接调用进行前向推理，用来测试对话效果。

#### 1.8.2 数据集精度掉点严重，对话乱码或胡言乱语

1. 对于label free校准场景，确认浮点模型使用torch npu推理是否正常。量化校准依赖浮点模型推理，如果浮点推理异常，量化校准时获取到的数据分布信息不对，校准结果自然不对。

2. 适当增加回退层。某些模型中的部分Linear层对精度的影响比较显著。例如ChatGlm2-6B模型W8A8量化时的layers.0.mlp.dense_4h_to_h层，依据调优经验以及相关论文数据，模型靠前和靠后的decoder layer、各个decoder layer的mlp down层对精度的影响一般较大，可以优先考虑回退这些层。如果回退效果不理想的话，可以尝试较为激进的回退策略，例如回退掉1/4或者1/2的Linear层，直到完全回退成浮点模型，模型的精度也完全回退成浮点模型的精度。回退越多，精度越高，性能越差。

3. 使用混合量化功能。在一些场景下，如果对一些层的精度要求没有那么高，同时希望提高性能，那么对一些模型中的部分敏感Linear层、如Qwen模型里的down层，可以不回退到浮点，而是用混合量化的方式将其量化为更高精度的数据类型，例如w8a16或w8a8_dynamic。这样做可以在尽量保持整体INT8性能的同时，降低对话出现乱码或胡言乱语的风险。

#### 1.8.3 数据集精度部分掉点，对话正常

1. 调整量化参数。例如W8A8量化调整act_method，W8A16量化更换使用的w_method。act_method默认为1。该参数可以选 ‘1’ ‘2’ ‘3’：1代表min-max量化方式；2代表histogram量化方式；3代表min-max和histogram自动混合量化方式。LLM大模型场景下推荐使用3。（稀疏量化的情况下只支持1和2的方式）

2. 稀疏量化可以调整fraction，该参数的含义为限制异常值的保护范围，建议在0.01~0.1之间将相应的值调大来增加精度。Lowbit场景下，除了上述参数微调，还可以使用sigma调整sigma_factor，该参数的含义也为限制异常值的保护范围，建议在3.0~4.0之间将相应的值调小来增加精度。

3. 使用异常值抑制算法，将do_smooth设置为True。W8A8量化使用anti_method=m1或m2，W8A16量化使用m3，通过抑制量化过程中的异常值，从而提高量化模型的精度。Lowbit场景下只需要开启即可，无需设置方法类型。

4. 调整校准数据。校准数据的数量一般为20-40条。在选取时需要考虑模型部署时的具体推理场景，例如中文模型需要使用中文输入作为校准集；英文模型需要使用英文输入；代码生成类模型则使用代码生成类任务；中英文兼顾的模型考虑使用中英文混合的校准集合。正常情况下，可以增加数据得到精度提升，但是到一定数据后，提高数据对精度影响有限。有些场景下，减少数据反而能得到精度提升。（例如长数据场景）<br>
    获取混合校准集可以使用[CalibrationData模块](foundation_model_quantization_and_calibration.md#2-混合校准数据集使用方法说明)

5. 增加回退层，可以使用disable_level自动回退功能按照一定的标准自动回退对精度影响比较大的Linear层，或者按照一定的经验，通过disable_name手动设置回退层。

6. 使用混合量化功能。在一些场景下，若发现只有少量关键层需保留更高精度，则可以先尝试只针对这些层做混合量化，再观察指标及对话质量。若精度不够，可进一步扩大范围到更多层；若性能损失过大，也可缩小范围到更精简的关键层集合。

## 2. 大模型稀疏量化和权重压缩

### 2.1 简介

大模型压缩工具共分为稀疏、量化和权重压缩三大环节，用户需连续运行方可实现大模型压缩。

### 2.2 使用前准备

稀疏和量化工具支持在以下产品中使用：

Atlas 推理系列产品。

Atlas 训练系列产品。

Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件。

权重压缩工具仅支持在Atlas 推理系列产品上使用。

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

大模型稀疏和压缩前须执行命令安装如下依赖。

如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install onnx --user。

```bash
pip3 install numpy==1.25.2
pip3 install transformers       #需大于等于4.29.1版本，LLaMA模型需指定安装4.29.1版本
pip3 install torch==2.1.0       #安装PyTorch 2.1.0版本
pip3 install accelerate==0.21.0 #若需要使用NPU多卡并行方式对模型进行量化，需大于等于0.28.0版本
pip3 install tqdm==4.66.1
pip3 install tensorboard      #需大于等于2.11.2版本
pip3 install typepy           #需大于等于1.3.1版本
pip3 install sacrebleu        #需大于等于2.3.1版本
pip3 install datasets         #需大于等于2.13.1版本
pip3 install sqlitedict       #需大于等于2.1.0版本
pip3 install omegaconf        #需大于等于2.3.0版本
pip3 install pycountry        #需大于等于22.3.5版本
pip3 install rouge_score      #需大于等于0.1.2版本
pip3 install peft             #需大于等于0.5.0版本
```

（可选）如果需要在大模型量化工具中使用NPU多卡并行的方式对模型进行量化，需关闭NPU设备中的虚拟内存，并手动配置量化将会执行的设备序列环境。

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False # 关闭NPU的虚拟内存
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 #配置量化将会执行的设备序列环境
```

说明

仅Atlas 训练系列产品和Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件支持此功能。

### 2.3 功能介绍

#### 2.3.1 Pytorch

目前支持对表1中的大模型进行量化（包括但不限于）。

表1 已验证模型列表

| 模型名称 | 框架 |
|----------|-------|
| ChatGLM2-6B | PyTorch |
| LLaMA2-7B | PyTorch |
| LLaMA-13B | PyTorch |

#### 2.3.2 大模型稀疏量化工具关键步骤说明如下

图1 稀疏量化接口调用流程

![稀疏量化接口调用流程](./figures/[pytorch]sparse_quantization_api_calling.png)

用户准备原始模型和校准数据。

使用QuantConfig生成稀疏量化配置。

根据原始模型、稀疏量化配置和校准数据，调用Calibrator接口构建稀疏和量化校准对象。

调用生成的稀疏和量化校准对象的run()方法，对原始模型进行稀疏和量化处理。

调用生成的稀疏和量化校准对象的save()接口，保存量化后的模型，包括模型稀疏量化的权重和模型相关参数。其中，模型权重文件quant_weight.npy将会用于权重压缩。

权重压缩工具关键步骤说明如下：

图2 量化接口调用流程

![权重压缩接口调用流程](./figures/[pytorch]weight_compression_api_calling.png)

使用CompressConfig生成权重压缩配置。

用户准备经过稀疏和量化处理后的模型权重文件quant_weight.npy。

根据模型权重文件和压缩配置，调用Compressor接口构建权重压缩对象。

调用权重压缩对象的run()方法，对模型权重进行压缩处理。

调用权重压缩对象的export()方法，保存压缩结果，用于后续模型的推理部署任务。

说明

权重压缩工具在加载输入的权重文件时，存在一定的反序列化攻击安全风险。权重压缩工具通过界面提示操作存在反序列化攻击的安全风险，在加载前用户交互确认加载的权重文件无风险后，才开始进行对文件的操作。

#### 2.3.3 稀疏量化步骤（以ChatGLM2-6B为例）

用户自行准备模型、权重文件和校准数据，本样例以ChatGLM2-6B为例，单击Link自行下载权重文件，并上传至服务器文件夹内，如上传至“chatglm2”文件夹，目录示例如下：

```text
├── config.json
├── configuration chatglm.py
├── modeling_chatglm.py
├── pytorch_model-00001-of-00007.bin
├── pytorch_model-00002-of-00007.bin
├── pytorch_model-00003-of-00007.bin
├── pytorch_model-00004-of-00007.bin
├── pytorch_model-00005-of-00007.bin
├── pytorch_model-00006-of-00007.bin
├── pytorch_model-00007-of-00007.bin
├── pytorch_model.bin.index.json
├── quantization.py
├── README.md
├── tokenization_chatglm.py
├── tokenizer.model
├── tokenizer_config.json
```

说明

大模型量化工具建议在大模型下游任务评估流程打通的前提下使用，请自行调试源码后进行以下量化配置。

ChatGLM2-6B为模型进行量化前请执行如下命令安装所需依赖，若运行量化工具过程中提示缺失某个依赖，请根据提示安装。

```bash
pip3 install protobuf==4.24.1
pip3 install sentencepiece==0.1.99
pip3 install sympy==1.11.1
```

新建模型的稀疏量化脚本sparse_quant.py，编辑sparse_quant.py文件。

稀疏量化场景导入样例代码如下，lowbit算法稀疏量化场景导入的代码样例请参考[lowbit算法稀疏量化场景](quantization_and_sparse_quantization_scenario_import_code_examples.md)，请参考信息提示，根据实际情况进行修改。

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在cpu上进行量化，可忽略此步骤
import torch.utils.data
from transformers import AutoTokenizer, AutoModel
# for local path
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='./chatglm2',
    local_files_only=True
)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2',
    torch_dtype=torch.float16,
    local_files_only=True
  ).npu()    # 如果需要在npu上进行多卡量化，需要先参考前提条件进行配置，并配置以下参数device_map='auto', torch_dtype为当前使用模型的默认数据类型；在npu上进行量化时，单卡校准需将模型移到npu上model = model.npu()，多卡校准时不需要
# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset
dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 稀疏量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入稀疏量化配置接口
# 使用QuantConfig接口，配置稀疏量化参数，并返回配置实例
quant_config = QuantConfig(
    w_bit=4,
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'],
    act_method=3,
    dev_type='npu',  # 在cpu进行量化时，需要配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    dev_id=model.device.index,
    pr=2.0,
    fraction=0.011,
    nonuniform=False,
    mm_tensor=False,
    co_sparse=True
 )
#使用Calibrator接口，输入加载的原模型、稀疏量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight')      #使用save()保存模型量化参数，请根据实际情况修改路径
print('Save quant weight success!')
```

说明

因为在存储量化参数过程中存在反序列化风险，所以已通过在存储过程中，将保存的量化结果文件夹权限设置为750，量化权重文件权限设置为400，量化权重描述文件设为600来消减风险。

启动模型稀疏量化任务，并在指定的输出目录获取模型量化参数。

```bash
python3 sparse_quant.py
```

说明

生成的模型权重文件quant_weight.npy将会用于压缩步骤（以ChatGLM2-6B为例）的步骤2。

#### 2.3.4 压缩步骤（以ChatGLM2-6B为例）

编译压缩函数。

已参考前提条件，完成开发环境配置。

请参见[安装指南](../../../install_guide/install_guide.md#23-源码安装)中有关稀疏量化内容，完成**稀疏量化压缩场景**的相关编译步骤。

编译结束后，在当前路径下生成build目录，执行如下命令查看编译结果compress_executor。

```bash
cd build
```

用户参考稀疏量化步骤对ChatGLM2-6B进行稀疏量化之后，在指定的输出目录“chatglm2”文件夹得到模型稀疏量化后的参数，目录示例如下：

```text
├── deq_scale.npy
├── input_offset.npy
├── input_scale.npy
├── quant_bias.npy
├── quant_weight.npy
```

说明

权重压缩工具仅需要对步骤4生成的权重文件quant_weight.npy进行压缩。

新建压缩脚本compress.py，编辑compress.py文件，导入如下样例代码，并根据实际情况进行修改。

```python
# 导入相关依赖
import sys
import os
import numpy as np
# 导入权重压缩接口
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o750, exist_ok=True)
    return path

# 准备待压缩权重文件和相关压缩配置，请根据实际情况进行修改
weight_path = "./chatglm2/quant_weight.npy"       # 待压缩权重文件的路径
save_path = "./compress"                          # 压缩后权重文件保存的路径
index_root = make_dir(os.path.join(save_path, 'index'))
weight_root = make_dir(os.path.join(save_path, 'weight'))
info_root = make_dir(os.path.join(save_path, 'info'))

# 使用CompressConfig接口，配置压缩参数，并返回配置实例
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=save_path, multiprocess_num=8)

#使用Compressor接口，输入加载的压缩配置和待压缩权重文件
compressor = Compressor(compress_config, weight_path)
compress_weight, compress_index, compress_info = compressor.run()

#使用export()接口，保存压缩后的结果文件
compressor.export(compress_weight, weight_root)
compressor.export(compress_index, index_root)
compressor.export(compress_info, info_root, dtype=np.int64)
```

运行压缩脚本，并在指定的输出目录获取压缩后的权重文件，用于后续的推理部署任务，具体请参见MindIE的[“MindIE支持模型列表”](https://www.hiascend.com/document/detail/zh/mindie/10RC3/whatismindie/mindie_what_0003.html)章节中已适配量化的模型。

```bash
python3 compress.py
```

### 2.3.5 MindSpore

#### 2.3.5.1 前提条件

稀疏和量化工具支持在以下产品中使用：

Atlas 推理系列产品。

Atlas 训练系列产品。

Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件。

权重压缩工具仅支持在Atlas 推理系列产品上使用。

已参考[安装指南](../../../install_guide/install_guide.md)完成开发环境配置。

大模型稀疏和压缩前须执行命令安装如下依赖。

如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install onnx --user。

```bash
pip3 install numpy==1.25.2
pip3 install tqdm==4.66.1
pip3 install typepy           #需大于等于1.3.1版本
pip3 install sacrebleu        #需大于等于2.3.1版本
pip3 install datasets         #需大于等于2.13.1版本
pip3 install sqlitedict       #需大于等于2.1.0版本
pip3 install omegaconf        #需大于等于2.3.0版本
pip3 install pycountry        #需大于等于22.3.5版本
pip3 install rouge_score      #需大于等于0.1.2版本
pip3 install peft             #需大于等于0.5.0版本
```

#### 2.3.5.2 功能实现流程

大模型压缩工具共分为稀疏、量化和权重压缩三大环节，用户需连续运行方可实现大模型压缩。

大模型稀疏量化工具关键步骤说明如下：

图3 稀疏量化接口调用流程

![稀疏量化接口调用流程](./figures/[pytorch]sparse_quantization_api_calling.png)

用户准备原始模型和校准数据。

使用QuantConfig生成稀疏量化配置。

根据原始模型、稀疏量化配置和校准数据，调用Calibrator接口构建稀疏和量化校准对象。

调用生成的稀疏和量化校准对象的run()方法，对原始模型进行稀疏和量化处理。

调用生成的稀疏和量化校准对象的save()接口，保存量化后的模型，包括模型稀疏量化的权重和模型相关参数。其中，模型权重文件ckpt将会用于权重压缩。

权重压缩工具关键步骤说明如下：

图4 量化接口调用流程

![权重压缩接口调用流程](./figures/[pytorch]weight_compression_api_calling.png)

使用CompressConfig生成权重压缩配置。

用户准备经过稀疏和量化处理后的ckpt文件。

根据模型权重文件和压缩配置，调用Compressor接口构建权重压缩对象。

调用权重压缩对象的run()方法，对模型权重进行压缩处理。

调用权重压缩对象的export()方法，保存压缩结果，用于后续模型的推理部署任务。

说明

权重压缩工具在加载输入的权重文件时，存在一定的反序列化攻击安全风险。权重压缩工具通过界面提示操作存在反序列化攻击的安全风险，在加载前用户交互确认加载的权重文件无风险后，才开始进行对文件的操作。

#### 2.3.5.3 稀疏量化步骤

说明

本示例旨在说明基于MindSpore框架的大模型稀疏量化的操作步骤。

用户自行准备模型、权重文件和校准数据，权重格式为ckpt。

说明

大模型量化工具建议在大模型下游任务评估流程打通的前提下使用，请自行调试源码后进行以下量化配置。

新建模型的稀疏量化脚本sparse_quant.py，编辑sparse_quant.py文件。

稀疏量化场景导入样例代码如下：

```python
# 导入相关依赖
import mindspore as ms
model, tokenizer = load_model_and_tokenizer()    # 用户根据实际使用场景自行加载MindSpore框架下的组件

# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]

#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='np', padding=True)
        calib_dataset.append([inputs.data['input_ids']])
    return calib_dataset
dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 稀疏量化配置，请根据实际情况修改
from msmodelslim.mindspore.llm_ptq import Calibrator, QuantConfig    # 导入稀疏量化配置接口
# 使用QuantConfig接口，配置稀疏量化参数，并返回配置实例
quant_config = QuantConfig(
    disable_names=[],      # 回退层按照实际需求进行配置
    fraction=0.03,         # fraction参数按照实际需求配置
)

#使用Calibrator接口，输入加载的原模型、稀疏量化配置和校准数据，定义校准
calibrator = Calibrator(
    cfg=quant_config,
    model=model,
    model_ckpt="./model.ckpt",
    calib_data=dataset_calib
)
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight.ckpt')      #使用save()保存模型量化参数，请根据实际情况修改路径
print('Save quant weight success!')
```

说明

因为在存储量化参数过程中存在反序列化风险，所以已通过在存储过程中，将保存的量化结果文件夹权限设置为750，量化权重文件权限设置为400，量化权重描述文件设为600来消减风险。

启动模型稀疏量化任务，并在指定的输出目录获取模型量化参数，用于后续的推理部署任务

```bash
python3 sparse_quant.py
```

#### 2.3.5.4 压缩步骤

编译压缩函数。

已参考前提条件，完成开发环境配置。

请参见[安装指南](../../../install_guide/install_guide.md#23-源码安装)中有关稀疏量化内容，完成**稀疏量化压缩场景**的相关编译步骤。

编译结束后，在当前路径下生成build目录，执行如下命令查看编译结果compress_executor。

```bash
cd build
```

用户参考稀疏量化步骤对指定模型进行稀疏量化之后，在指定的权重保存路径（./quant_weight.ckpt）下得到ckpt文件，并对待压缩的权重部分进行压缩操作。

```python
import re
import mindspore as ms
import numpy as np

from ascend_utils.common.security import SafeWriteUmask #请根据实际情况导入对应框架的库文件

linear_weight_pattern = r"^(?=.{1,100}$)model\.layers\.\d+\.(attention[^_]|feed_forward|augs_attn\d+).*\.weight$" #根据实际情况进行权重键值的筛选
reg = re.compile(linear_weight_pattern)
sparse_ckpt = ms.load_checkpoint(f"./quant_weight.ckpt")  #./quant_weight.ckpt为稀疏量化结果文件的保存路径
compressed_weight_dict = {}
disable_names = [] # 回退层列表
for k, v in sparse_ckpt.items():
   if reg.search(k):
        if k in disable_names:
            continue
        compressed_weight_dict[k] = v.numpy()
with SafeWriteUmask():
    np.save(f"quant_weight.npy", compressed_weight_dict)
```

说明

权重压缩工具仅需要对权重文件ckpt进行压缩操作。

新建压缩脚本compress.py，编辑compress.py文件，导入如下样例代码，并根据实际情况进行修改。

```python
# 导入相关依赖
import sys
import os
import numpy as np
# 导入权重压缩接口
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o750, exist_ok=True)
    return path

# 准备待压缩权重文件和相关压缩配置，请根据实际情况进行修改
weight_path = "./quant_weight.npy"       # 待压缩权重文件的路径
save_path = "./compress"                          # 压缩后权重文件保存的路径
index_root = make_dir(os.path.join(save_path, 'index'))
weight_root = make_dir(os.path.join(save_path, 'weight'))
info_root = make_dir(os.path.join(save_path, 'info'))

# 使用CompressConfig接口，配置压缩参数，并返回配置实例
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=save_path, multiprocess_num=8)

#使用Compressor接口，输入加载的压缩配置和待压缩权重文件
compressor = Compressor(compress_config, weight_path)
compress_weight, compress_index, compress_info = compressor.run()

#使用export()接口，保存压缩后的结果文件
compressor.export(compress_weight, weight_root)
compressor.export(compress_index, index_root)
compressor.export(compress_info, info_root, dtype=np.int64)
```

运行压缩脚本，并在指定的输出目录获取压缩后的权重文件，用于后续的推理部署任务。

```bash
python3 compress.py
```

说明

用户需要将压缩后得到的权重文件（compress_weight、compress_index和compress_info）自行整合到ckpt文件中，才能用于后续的推理部署任务。

## 3. 长序列压缩

### 3.1 Alibi编码类型简介

Alibi编码是一种位置编码方法，与RazorAttention结合使用，通过Alibi编码来识别哪些注意力头对位置信息更为敏感，从而决定哪些头可以被压缩。Alibi编码并不直接在网络中加入显式的位置编码，而是通过在query-key注意力分数上施加一个与距离成比例的偏置实现位置信息的建模。

KV Cache的管理需考虑batch, seq_len, num_heads和head_size这四个维度，其中seq_len维度通常是压缩的重点，因为随着序列长度的增加，KV Cache的内存占用会迅速增长。传统的压缩方法可能会忽略不同注意力头（heads）之间的差异，而RazorAttention加速技术则提供了一种更细粒度的内存压缩方法，针对使用Alibi编码的模型进行优化，可以更有效地识别哪些注意力头对于位置信息更为敏感，并据此调整压缩策略。RazorAttention加速技术支持全量加速和增量加速：

全量加速：压缩后的KV Cache可直接用于模型推理，实现全量加速。

增量加速：支持只更新和压缩新token对应的KV Cache部分。

目前支持对表1中Alibi编码的大模型进行长序列压缩（包括但不限于）。

已验证模型列表：

|模型名称|框架|
|----|-----|
|baichuan2-13b|PyTorch|

### 3.2 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。
执行命令安装如下依赖。
以下命令若使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install numpy==1.26.4 --user。

```bash
pip3 install numpy==1.26.4
pip3 install transformers==4.43.1
pip3 install torch==2.1.0   # 安装CPU版本的PyTorch 2.1.0（不依赖torch_npu）
```

### 3.3 功能介绍

图1 压缩接口调用流程
![Alibi压缩接口调用流程](./figures/[pytorch]alibi_compress_api_calling.png)

关键步骤说明如下：

用户准备原始模型。
调用RACompressConfig接口生成压缩配置，新建模型的压缩脚本run.py。

执行压缩算法RACompressor启动长序列压缩任务，进行长序列压缩。

调用get_alibi_windows接口导出压缩窗口，并在指定路径中获取.pt文件，具体请参见MindIE的[“加速库支持模型列表”](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0001.html)章节中已适配量化的模型。

#### 3.3.1 压缩步骤（以baichuan2-13b为例）

用户准备原始模型。

用户需要自行准备模型、权重文件。本样例以baichuan2-13b为例，从该网站下载权重文件，并上传至服务器的“baichuan2-13b”文件夹，目录示例如下：

```text
config.json
configuration_baichuan.py
cut_utils.py
generation_config.json
generation_utils.py
handler.py
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
modeling_baichuan_cut.py
modeling_baichuan.py
model.safetensors.index.json
pytorch_model.bin.index.json
quantizer.py
special_tokens_map.json
tokenization_baichuan.py
tokenizer_config.json
tokenizer.model
```

新建模型的压缩脚本run.py，将如下样例代码导入run.py文件，并执行。

```python
from msmodelslim.pytorch.ra_compression import RACompressConfig, RACompressor
from transformers import AutoTokenizer, AutoModelForCausalLM
config = RACompressConfig(theta=0.00001, alpha=100)   # 压缩类的配置，需根据实际情况进行修改
input_model_path = "/data1/models/baichuan/baichuan2-13b/float_path/"    # 模型权重文件的保存路径，需根据实际情况进行修改
save_path = "./win.pt"   # 生成压缩窗口的路径，需根据实际情况进行修改
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=input_model_path,
    local_files_only=True
    )
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=input_model_path,
    local_files_only=True
    ).float().cpu()   # 不支持使用npu方式进行加载
ra = RACompressor(model, config)
ra.get_alibi_windows(save_path)
```

执行以下命令，启动长序列压缩任务，并在“baichuan2-13b”文件夹的路径下获取.pt文件。
python3 run.py
.pt文件可用于后续的推理部署任务，具体请参见MindIE的[“加速库支持模型列表”](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0001.html)章节中已适配量化的模型。

## 4. RoPE编码类型简介

RoPE（Rotary Position Embedding）编码是一种高效的位置编码方式，有以下特点：

旋转编码：通过旋转操作将位置信息编码到每个token的嵌入向量中。这种旋转操作确保了模型能够捕捉到序列中元素的相对位置信息，而不依赖于绝对位置。
维度保持：旋转操作在每个维度上独立进行，有助于模型在不同的特征维度上捕获位置信息。
计算效率：不需要额外的参数来编码位置信息，而是通过数学旋转操作来实现，计算效率较高。
通过RoPE编码与RazorAttention结合，可分析注意力头对位置编码的依赖性，来决定哪些头可以被压缩，以优化模型的存储、传输和计算效率，提高模型在实际应用中的可部署性和实用性。

利用RoPE编码的位置信息：由于RoPE编码已经有效地将位置信息编码到每个token中，RA算法可以利用这一点来更好地识别哪些注意力头对于位置信息更为敏感，从而更有针对性地进行压缩。
优化压缩策略：通过结合RoPE编码和RA算法，可以在保持模型性能的同时，针对不同的注意力头实施更精细的压缩策略。例如，对于依赖位置信息的Retrieval Head，可以保持其KV Cache的完整性，而对于不依赖位置信息的Non-retrieval Head，则进行压缩。
目前支持对表1中RoPE编码的大模型进行长序列压缩（包括但不限于）。

表1 已验证模型列表

| 模型名称             | 框架    |
|----------------------|--------|
| Qwen2-72b-instruct   | PyTorch|
| llama3.1-70b         | PyTorch|

### 4.1 使用前准备

已参考[安装指南](../../../install_guide/install_guide.md)完成开发环境配置。
执行命令安装如下依赖。
以下命令若使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install numpy==1.26.4 --user。

> **说明**：torch_npu 的 xxx 需要根据实际情况进行选择，具体请参考[安装指南](../../../install_guide/install_guide.md)中 torch_npu 版本。

```bash
pip3 install numpy==1.26.4
pip3 install transformers==4.43.1
pip3 install torch==2.1.0       # 安装CPU版本的PyTorch 2.1.0（依赖torch_npu）
pip3 install torch_npu-2.1.0.xxx.whl
```

### 4.2 功能介绍

图2 压缩接口调用流程

![RoPE压缩接口调用流程](./figures/[pytorch]rope_compress_api_calling.png)

关键步骤说明如下：

用户准备原始模型。

调用RARopeCompressConfig接口生成压缩配置，并新建模型的压缩脚本run.py。

调用RARopeCompressor启动长序列压缩任务，进行长序列压缩。

调用get_compress_heads接口导出需保留的Head信息，并在指定路径获取.pt文件。

用户可根据.pt文件进行压缩

压缩后的文件可用于后续的推理部署，具体请参见MindIE的[“加速库支持模型列表”](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0001.html)章节中已适配量化的模型。

### 4.3 压缩步骤（以Qwen2-72b-instruct为例）

用户准备原始模型。

用户需要自行准备模型、权重文件。本样例以Qwen2-72b-instruct为例，从该网站下载权重文件，并上传至服务器的“Qwen2-72b-instruct”文件夹内，目录示例如下：

```text
config.json
generation_config.json
merges.txt
model-00001-of-00037.safetensors
......
model-00037-of-00037.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
vocab.json
```

新建模型的量化脚本run.py，并将如下样例代码导入run.py文件，并执行以下命令。

```python
import torch
from msmodelslim.pytorch.ra_compression import RARopeCompressConfig, RARopeCompressor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_npu
torch.npu.set_compile_mode(jit_compile=False)
config = RARopeCompressConfig(induction_head_ratio=0.14, echo_head_ratio=0.01)
save_path = "./win.pt"
model_path = "./Qwen2-72B-Instruct/"

model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    ).eval()

tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        local_files_only=True
    )
ra = RARopeCompressor(model, tokenizer, config)
ra.get_compress_heads(save_path)
```

启动长序列压缩任务，并在“Qwen2-72b-instruct”文件夹的路径下获取需要保留KV Cache的Head信息的.pt文件。

用户可根据.pt文件进行压缩。

压缩后的文件可用于后续的推理部署，具体请参见MindIE的[“MindIE支持模型列表”](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0001.html)章节中已适配量化的模型。

```bash
python3 run.py
```

## 5. 权重压缩基本使用流程

### 5.1 编译压缩函数

- 进入python环境下的site-packages包管理路径，以下是以/usr/local/为用户所在目录、Python版本为3.11.10为例

```bash
cd /usr/local/lib/python3.11/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/
```

- 编译weight_compression组件 `sudo bash build.sh {CANN包安装路径}/ascend-toolkit/latest`
- 上一步编译操作会得到build文件夹，给build文件夹相关权限 `chmod -R 550 build`

### 5.2 导入工具

from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor

### 5.3 设置压缩工具配置

由于压缩工具调用的压缩函数已将大部分配置参数固定，因此在工具层面无需设置很多参数。

```python
class CompressConfig:
    """ The configuration for LLM weight compression """
    def __init__(self,
        do_pseudo_sparse=False,  # whether to do pseudo sparse before compression
        sparse_ratio=1,  # percentile of non-zero values after pseudo sparse
        is_debug=False,  # print the compression ratio for each weight if is_debug is True
        compress_disable_layers=[],  # the layers in compress_disable_layers will not be compressed and directly saved in compress_output
        record_detail_root='./',  # the save path for the temporary data
        multiprocess_num=1) -> object:  # multiprocess num

save_path = "./compress"
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=save_path, multiprocess_num=8)
```

### 5.4 定义压缩任务类，启动

```python
compressor = Compressor(compress_config, path_save)
compress_weight, compress_index, compress_info = compressor.run()
```

说明：

1. `compressor.run()`有一个参数`bool: weight_transpose`，默认为`False`，即是否开启权重转置。目前已知chatGLM2-6B无需开启权重转置
2. 开启多进程权重压缩模式时，需要手动设置当前环境下最大可打开文件数，可参考以下命令：

    ```bash
    #check current limit
    ulimit -n

    #raise limit to 65535
    ulimit -n 65535
    ```

### 5.5 启动之后对每个层的权重进行 转numpy->转Nz->压缩->保存

调用C编译后的压缩函数，通过文件的形式进行交互，完成权重压缩过程。

### 5.6 导出压缩处理结果

```python
compressor.export(compress_weight, weight_root)
compressor.export(compress_index, index_root)
compressor.export(compress_info, info_root, dtype=np.int64) # info数据为int64格式需要特别声明，否则默认将会保存为int8的格式
```

说明：权重压缩工具在加载输入的权重文件时，存在一定的反序列化攻击安全风险。权重压缩工具通过界面提示操作存在反序列化攻击的安全风险，在加载前用户交互确认加载的权重文件无风险后，再进行后续操作。

## 6. 模型低秩分解

### 6.1 简介

深度学习运算，尤其是CV（计算机视觉）和NLP（自然语言处理）类任务运算，包含大量的矩阵运算，而低秩分解通过将大矩阵分解为若干个低秩矩阵的乘积，从而降低存储空间和计算量，降低推理开销。

### 6.2 使用前准备

当前支持在训练服务器上对MindSpore和PyTorch框架下的模型进行低秩分解。
安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

### 6.3 功能介绍

1. 用户需自行准备模型、训练脚本和数据集，本样例以PyTorch框架的ResNet50和数据集ImageNet为例。

2. 编辑模型的训练脚本pytorch_resnet50_apex.py文件，导入低秩分解的库文件。

```python
from msmodelslim.pytorch import low_rank_decompose
# 请根据实际情况导入对应框架的库文件
from ascend_utils.common.utils import count_parameters
from ascend_utils.common.security import SafeWriteUmask
```

- 说明
MindSpore框架下库文件的路径为msmodelslim.mindspore.low_rank_decompose，PyTorch框架下库文件的路径为 msmodelslim.pytorch.low_rank_decompose。

3.（可选）调整日志输出等级，启动训练任务后，将日志输出等级设置为相应级别，以便在训练时显示对应的日志信息。[日志级别说明](../../../python_api_v0/common_apis.md#参数说明)

```python
from msmodelslim import set_logger_level
set_logger_level("info")        #根据实际情况配置
```

4.（可选）在模型创建后，使用count_parameters接口，配置日志打屏显示内容。启动调优任务后，将打屏显示原模型的参数量信息。请参考count_parameters配置。

```python
print("Original model parameters:", count_parameters(model))
```

5.在模型创建后，使用Decompose类接口，配置低秩分解的方式，请参考Decompose进行配置。

```python
decomposer = low_rank_decompose.Decompose(model).from_ratio(0.5)
```

6.使用Decompose类的decompose_network接口，实际执行低秩分解，并返回分解后的模型，请参考decompose_network进行配置。

```python
model = decomposer.decompose_network()    #使用分解后的模型替换原模型
```

7.（可选）配置日志打屏显示内容，启动调优任务后，将打屏显示分解后模型的参数量信息。

```python
print("Decomposed model parameters:", count_parameters(model))
```

8.多卡训练时，需要先在单卡训练下保存模型权重（单卡训练时无需执行本步骤，直接启动调优任务即可）。

```python
state_dict_file = "/home/xxx/decompose_state_dict.ckpt"   #请根据实际情况配置模型权重文件保存的路径及名称
with SafeWriteUmask():
    torch.save(model.state_dict(),state_dict_file)
```

9.多卡训练时，需要在多卡下指定 do_decompose_weight=False，只转换模型结构为低秩分解后的模型，不分解模型权重。然后加载单卡训练下保存的权重（单卡训练时无需执行本步骤，直接启动调优任务即可）。

```python
from ascend_utils.common.security.pytorch import safe_torch_load
model = decomposer.decompose_network(do_decompose_weight=False)
model.load_state_dict(state_dict = safe_torch_load(state_dict_file, map_location="cpu"))
```

10.启动训练任务，根据单卡或多卡调用不同的执行脚本，并指定data_path为数据集路径。

- 单卡训练时，执行命令启动训练任务。

```bash
bash ./test/train_full_1p.sh --data_path=./datasets/imagenet  #请根据实际情况配置数据集路径
```

- 多卡训练时，执行命令启动训练任务，会在步骤8指定路径下生成模型权重文件。以下示例为8卡训练，请根据实际情况替换启动脚本。

```bash
bash ./test/train_full_8p.sh --data_path=./datasets/imagenet   #请根据实际情况配置数据集路径
```

11.查看结果。
训练完成后输出模型训练精度和性能信息。
