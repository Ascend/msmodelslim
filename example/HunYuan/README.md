# Hunyuan 量化说明

## 模型介绍

- [Tencent-Hunyuan-Large](https://github.com/Tencent/Tencent-Hunyuan-Large) 目前业界开源的基于 Transformer 的最大 MoE 模型，拥有 3890 亿个参数、520 亿个活跃参数，且其具备高质量合成数据增强训练、KV 缓存压缩、专家特定学习率缩放、长上下文处理能力强（预训练模型支持 256K 文本序列，Instruct 模型支持 128K）。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/getting_started/install_guide.md)。
- transformers版本需要配置>=4.48.2

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接                                                 | W8A8 | W8A16 | W4A16 | W4A4  | 稀疏量化 | KV Cache | Attention | 量化命令                                                                                                                                                                                 |
|---------|---------|---------------------------------------------------------------|-----|-------|-------|------|---------|----------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Hunyuan** | Tencent-Hunyuan-Large | [Tencent-Hunyuan-Large](https://huggingface.co/tencent/Tencent-Hunyuan-Large/tree/main/Hunyuan-A52B-Instruct) | ✅ |   |   |   |  |   |   | [W8A8混合量化](#hunyuan-large-w8a8-混合量化-experts层-w8a8-dynamic量化其余层-w8a8量化)                                                                                                                                                            |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令

## 量化权重生成

- 量化权重可使用 [quant_hunyuan.py](./quant_hunyuan.py) 脚本生成，以下提供Hunyuan模型量化权重生成快速启动命令。

### quant_hunyuan.py 量化参数说明

| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入Hunyuan权重目录路径。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| part_file_size | 生成量化权重文件大小，单位是GB | 5 | 可选参数；<br>生成量化权重文件大小，默认5GB。 |
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 |
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 |
| disable_names | 手动回退的量化层名称 | 默认回退所有mlp.gate.wg层 | 用户可根据精度要求手动设置，默认回退隐藏层的降维投影层。|
| device_type | device类型 | npu | 可选值：['cpu', 'npu']。 |
| fraction | 模型权重稀疏量化过程中被保护的异常值占比  |0.01| 取值范围[0.01,0.1]。|
| act_method | 激活值量化方法 | 1 |(1) 1代表Label-Free场景的min-max量化方式。 <br>(2) 2代表Label-Free场景的histogram量化方式。 <br>(3) 3代表Label-Free场景的自动混合量化方式，LLM大模型场景下推荐使用。|
| co_sparse | 是否开启稀疏量化功能 | False | True: 使用稀疏量化功能；<br>False: 不使用稀疏量化功能。 |
| anti_method | 离群值抑制参数 | 无默认值 |'m1': SmoothQuant算法。<br>'m2': SmoothQuant加强版算法，推荐使用。<br>'m3': AWQ算法。<br>'m4': smooth优化算法 。<br>'m5': CBQ量化算法。|
| disable_level | L自动回退等级 | L0 | 配置示例如下：<br>'L0'：默认值，不执行回退。<br>'L1'：回退1层。<br>'L2'：回退2层。<br>'L3'：回退3层。<br>'L4'：回退4层。<br>'L5'：回退5层。|
| do_smooth | 是否启动smooth量化功能 | False | True: 开启smooth量化功能；<br>False: 不开启smooth量化功能。 |
| use_sigma | 是否启动sigma功能 | False|True: 开启sigma功能；<br>False: 不开启sigma功能。 |
| use_reduce_quant | 权重量化是否是lccl all reduce量化 | False | 用于MindIE推理的标识。 |
| sigma_factor | sigma功能中sigma的系数 | 3.0 | 数据类型为float，默认值为3.0，取值范围为[1.0, 3.0]。<br>说明：仅当use_sigma为True时生效。 |
| is_lowbit | 是否开启lowbit量化功能 | False|(1) 当w_bit=4，a_bit=8时，为大模型稀疏量化场景，表示开启lowbit稀疏量化功能。<br>(2) 其他场景为大模型量化场景，会开启量化自动精度调优功能。<br>当前量化自动精度调优框架支持W8A8，W8A16量化。|
| mm_tensor | 是否开启mm_tensor量化功能 | True | True: 开启mm_tensor量化功能；<br>False: 不开启mm_tensor量化功能。 |
| w_sym | 是否开启w_sym量化功能 | True | True: 开启w_sym量化功能；<br>False: 不开启w_sym量化功能。 |
| use_kvcache_quant | 是否使用kvcache量化功能 | False | True: 使用kvcache量化功能；<br>False: 不使用kvcache量化功能。|
| use_fa_quant | 是否使用FA3量化 | False | True: 使用FA3量化类型；<br>False: 不使用FA3量化类型。|
| fa_amp | FA3量化场景下的自动回退的layer数量 | 0 | 数据类型为int，默认值为0。数据取值范围是大于等于0，并且小于等于模型layer数量，如果超出模型的layer数量将会取模型的最大layer数量为回退层数。 |
| open_outlier | 是否开启离群值抑制功能 | True | True: 开启离群值抑制功能；<br>False: 不开启离群值抑制功能。 |
| group_size | 离群值抑制参数 | 64 | 数据类型为int，默认值为64。数据取值范围是大于等于1，并且小于等于1024。 |
| is_dynamic | 是否使用per-token动态量化功能 | False | True: 使用per-token动态量化；<br>False: 不使用per-token动态量化。 |
| input_ids_name | 指定分词结果中输入 ID 对应的键名 | input_ids | 无 |
| attention_mask_name | 指定分词结果中注意力掩码对应的键名 | attention_mask | 无 |
| tokenizer_args | 加载自定义tokenizer时传入的自定义参数 | 无 | 以字典方式传入。 |
| disable_last_linear | 是否回退最后linear层 | True | True：回退最后linear层。<br>False：不回退最后linear层。 |
| model_name | 模型名称，可选参数 | None | 用于控制异常值抑制参数。 |
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载(请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险)。 |
| mindie_format | 非多模态模型量化后的权重配置文件是否兼容MindIE现有版本 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE 2.1.RC1及之前的版本。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig/)
  以及量化参数配置类 [Calibrator](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator/)

## 使用示例

- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用NPU多卡量化，请先配置环境变量，支持多卡量化：

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)

### Hunyuan-Large

#### 运行前必检

Hunyuan-Large模型较大，且存在需要手动适配的点，为了避免浪费时间，还请在运行脚本前，请根据以下必检项对相关内容进行更改。

- 1、需安装更新transformers版本（>=4.48.2）

#### <span id="hunyuan-large-w8a8-混合量化-experts层-w8a8-dynamic量化其余层-w8a8量化">Hunyuan-Large W8A8 混合量化 (experts层: W8A8 Dynamic量化，其余层: W8A8量化)</span>

注：需进入当前脚本目录下执行命令行

- 生成Hunyuan-Large模型 W8A8 混合量化权重

  ```shell
  python3 quant_hunyuan.py --model_path {浮点权重路径} --save_directory {量化权重路径} --anti_method m4 --trust_remote_code True
  ```

#### Hunyuan-Large量化QA

- Q：modeling_utils.py报错 if metadata.get("format") not in ["pt", "tf", "flax", "mix"]: AttributeError: "NoneType" object has no attribute 'get';
- A：说明输入的权重中缺少metadata字段，需安装更新transformers版本（>=4.48.2）
