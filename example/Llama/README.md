# LLaMA量化说明

## 模型介绍

- [LLaMA（Large Language Model Meta AI）](https://github.com/facebookresearch/llama/tree/llama_v1)、 [LLaMA2（Large Language Model Meta AI 2）](https://github.com/facebookresearch/llama)和[LLaMA3（Large Language Model Meta AI 3）](https://github.com/meta-llama/llama3)，是由 Meta AI 发布的一个开放且高效的大型基础语言模型，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../docs/zh/install_guide/install_guide.md)。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接                                                                          | W8A8 | W8A16 | W4A8 | 稀疏量化 | KV Cache | Attention | W4A8_DYNAMIC | 量化命令 |
|---------|---------|----------------------------------------------------------------------------------------|-----|-------|------|---------|----------|-----------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **LLaMA** | LLaMA-33B | [LLaMA-33B](https://github.com/facebookresearch/llama/tree/llama_v1)                   |   |   |   | ✅ |   |   |   | [稀疏](#llama-33b-稀疏量化配置) |
| | LLaMA-65B | [LLaMA-65B](https://github.com/facebookresearch/llama/tree/llama_v1)                                                                          |   | ✅ |   |   |   |   |   | [W8A16](#llama-65b-w8a16量化) |
| **LLaMA2** | LLaMA2-7B | [LLaMA2-7B](https://github.com/facebookresearch/llama/tree/v2)                 | ✅ |   |   | ✅ |   |   |   | [W8A8](#llama2-7b13b-w8a8量化) / [稀疏](#llama2-7b13b-稀疏量化配置) |
| | LLaMA2-13B | [LLaMA2-13B](https://github.com/facebookresearch/llama/tree/v2)               | ✅ |   |   | ✅ |   |   |   | [W8A8](#llama2-7b13b-w8a8量化) / [稀疏](#llama2-7b13b-稀疏量化配置) |
| | LLaMA2-70B | [LLaMA2-70B](https://github.com/facebookresearch/llama/tree/v2)               | ✅ | ✅ |   |   |   |   |   | [W8A8](#llama2-70b-npu多卡w8a8量化) / [W8A16](#llama2-70b-w8a16量化) |
| **LLaMA3** | LLaMA3-70B | [LLaMA3-70B](https://github.com/meta-llama/llama3)                  |   | ✅ |   |   |   |   |   | [W8A16](#llama3-70b-w8a16量化) |
| **LLaMA3.1** | LLaMA3.1-8B | [LLaMA3.1-8B](https://github.com/meta-llama/llama3)                | ✅ |   |   |   |   |   |   | [W8A8](#llama31-8b-w8a8量化) |
| | LLaMA3.1-70B | [LLaMA3.1-70B](https://github.com/meta-llama/llama3)              | ✅ |   |   | ✅ | ✅ | ✅ |   | [W8A8](#llama31-70b-w8a8量化) / [KV Cache](#llama31-70b-w8a8量化搭配-kv-cache-int8量化) / [Attention](#llama31-70b-w8a8量化搭配attention量化) / [PDMix+KV Cache](#llama31-70b-w8a8-pdmix量化prefill阶段-w8a8动态量化-decode阶段-w8a8量化-搭配-kv-cache-int8量化) |
| | LLaMA3.1-8B-Instruct | [LLaMA3.1-8B-Instruct](https://github.com/meta-llama/llama3) |   |   |   |   |   |   | ✅ | [W4A8_DYNAMIC](#llama31-8b-instruct-w4a8_dynamic量化) |
| | LLaMA3.1-70B-Instruct | [LLaMA3.1-70B-Instruct](https://github.com/meta-llama/llama3) |   |   |   |   |   |   | ✅ | [W4A8_DYNAMIC](#llama31-70b-instruct-w4a8_dynamic量化) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 量化权重生成

- 量化权重统一使用[quant_llama.py](./quant_llama.py)脚本生成，以下提供LLaMA模型量化权重生成快速启动命令

### 量化参数说明

| 参数名 | 含义 | 默认值 | 使用方法 |
| ------ | ---- | --- | -------- |
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入LLaMA权重目录路径。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| part_file_size | 生成量化权重文件大小，单位是GB | 无默认值 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的最大限制。 |
| calib_texts | 量化校准数据 | 无默认值 | 可选参数；<br>校准数据集。 |
| calib_file | 量化校准数据 | teacher_qualification.jsonl | 可选参数；<br>存放校准数据的json文件。 |
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 |
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 |
| disable_names | 手动回退的量化层名称 | w8a8量化默认回退所有down_proj层 <br> LLaMA3 w8a16量化默认回退前5层 | 用户可根据精度要求手动设置，默认回退隐藏层的降维投影层。 |
| device_type | device类型 | cpu | 可选值：['cpu', 'npu']。 |
| fraction | 模型权重稀疏量化过程中被保护的异常值占比  |0.01| 取值范围[0.01,0.1]。|
| act_method | 激活值量化方法 | 1 |(1) 1代表Label-Free场景的min-max量化方式。 <br>(2) 2代表Label-Free场景的histogram量化方式。 <br>(3) 3代表Label-Free场景的自动混合量化方式，LLM大模型场景下推荐使用。|
|co_sparse|是否开启稀疏量化功能|False|True:使用稀疏量化功能；<br>False: 不使用稀疏量化功能。|
| anti_method | 离群值抑制参数 | 无默认值 |'m1': SmoothQuant算法。<br>'m2': SmoothQuant加强版算法。<br>'m3': AWQ算法。<br>'m4': smooth优化算法 。<br>'m5': CBQ量化算法。<br>'m6': Flex smooth量化算法。|
| disable_level | L自动回退等级 | L0 | 配置示例如下：<br>'L0'：默认值，不执行回退。<br>'L1'：回退1层。<br>'L2'：回退2层。<br>'L3'：回退3层。<br>'L4'：回退4层。<br>'L5'：回退5层。|
| do_smooth | 是否启动smooth量化功能 | False | True: 开启smooth量化功能；<br>False: 不开启smooth量化功能。 |
| use_sigma | 是否启动sigma功能 | False|True: 开启sigma功能；<br>False: 不开启sigma功能。 |
| use_reduce_quant | 权重量化是否是lccl all reduce量化 | False | 用于MindIE推理的标识。 |
| tp_size | 模拟多卡量化时的卡数 | 1 | 数据取值范围为[1,2,4,8,16]，默认值为1，不启用模拟多卡量化。<br>设置为2、4、8、16时，对于通信层的linear会进行模拟多卡，每张卡使用不同的scale和offset进行量化。 |
| sigma_factor | sigma功能中sigma的系数 | 3.0 | 数据类型为float，默认值为3.0，取值范围为[1.0, 3.0]。<br>说明：仅当use_sigma为True时生效。 |
| is_lowbit | 是否开启lowbit量化功能 | False|(1) 当w_bit=4，a_bit=8时，为大模型稀疏量化场景，表示开启lowbit稀疏量化功能。<br>(2) 其他场景为大模型量化场景，会开启量化自动精度调优功能。<br>当前量化自动精度调优框架支持W8A8，W8A16量化。|
| mm_tensor | 是否开启mm_tensor量化功能 | True | True: 开启mm_tensor量化功能；<br>False: 不开启mm_tensor量化功能。 |
| w_sym | 是否开启w_sym量化功能 | True | True: 开启w_sym量化功能；<br>False: 不开启w_sym量化功能。 |
| use_kvcache_quant | 是否使用kvcache量化功能 | False | True: 使用kvcache量化功能；<br>False: 不使用kvcache量化功能。|
| use_fa_quant | 是否使用FA3量化 | False | True: 使用FA3量化类型；<br>False: 不使用FA3量化类型。|
| fa_amp | FA3量化场景下的自动回退的layer数量 | 0 | 数据类型为int，默认值为0。数据取值范围是大于等于0，并且小于等于模型layer数量，如果超出模型的layer数量将会取模型的最大layer数量为回退层数。 |
| open_outlier | 是否开启权重异常值划分 | True | True：开启权重异常值划分。<br>False：关闭权重异常值划分。<br>说明：(1)仅在lowbit设置为True时生效。(2)per_group量化场景下，需协同设置is_lowbit为True，open_outlier为False。|
| group_size | per_group量化中group的大小 | 64 | 默认值为64，支持配置为32，64，128。<br>说明:仅适用于per_group量化场景，需协同设置is_lowbit为True，open_outlier为False。|
| is_dynamic | 是否使用per-token动态量化功能 | False | True: 使用per-token动态量化；<br>False: 不使用per-token动态量化。 |
| input_ids_name | 指定分词结果中输入 ID 对应的键名 | input_ids | 无 |
| attention_mask_name | 指定分词结果中注意力掩码对应的键名 | attention_mask | 无 |
| tokenizer_args | 加载自定义tokenizer时传入的自定义参数 | 无 | 以字典方式传入。 |
| disable_last_linear | 是否回退最后linear层 | True | True：回退最后linear层。<br>False：不回退最后linear层。 |
| model_name | 模型名称，可选参数 | None | 用于控制异常值抑制参数。 |
| model_type | llama模型类型 | llama2 | 若使用 LLaMA3 模型，请配置为 `llama3`；若使用 LLaMA3.1 基础模型（8B/70B），请配置为 `llama3.1_fp`；若使用 LLaMA3.1 Instruct 模型，请配置为 `llama3.1_instruct`。 |
| anti_calib_file | 离群值抑制校准数据文件 | None | 用于离群值抑制的校准数据文件路径(.json或.jsonl格式)。|
| disable_threshold | 自动选择回退层的阈值 | 0 | 当值大于0时，会根据该阈值自动选择需要回退的层。值越大，回退的层越多。|
| pdmix | 是否使用PDMix量化类型 | False | True: 使用PDMix量化类型；<br>False: 不使用PDMix量化类型。|
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载(请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险)。 |
| mindie_format | 非多模态模型量化后的权重配置文件是否兼容MindIE现有版本 | False | True：开启`mindie_format`时保存的量化权重格式能够兼容MindIE 2.1.RC1及之前的版本。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)

## 使用示例

- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用NPU多卡量化，请先配置环境变量，支持多卡量化：

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载。(请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险)

### LLaMA

#### <span id="llama-65b-w8a16量化">LLaMA-65B W8A16量化</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/teacher_qualification.jsonl --w_bit 8 --a_bit 16 --act_method 3 --trust_remote_code True
  ```

#### <span id="llama-33b-稀疏量化配置">LLaMA 33B 稀疏量化配置</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/boolq.jsonl --act_method 2 --do_smooth True --use_sigma True --is_lowbit True --co_sparse True --w_bit 4 --trust_remote_code True
  ```

### LLaMA2

#### <span id="llama2-7b13b-w8a8量化">LLaMA2-7B/13B W8A8量化</span>

- 生成llama2-7b量化权重，antioutlier使用m1算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在CPU上进行运算

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --device_type cpu --disable_level L0 --anti_method m1 --act_method 1 --trust_remote_code True
  ```

- 生成llama2-13b量化权重，antioutlier使用m2算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在CPU上进行运算

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type cpu --disable_level L0 --anti_method m2 --act_method 1 --trust_remote_code True
  ```

#### <span id="llama2-7b13b-稀疏量化配置">LLaMA2 7B/13B 稀疏量化配置</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W4A8S量化权重路径} --calib_file ../common/teacher_qualification.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --trust_remote_code True
  ```

#### <span id="llama2-70b-npu多卡w8a8量化">LLaMA2-70B NPU多卡W8A8量化</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --trust_remote_code True
  ```

#### <span id="llama2-70b-w8a16量化">LLaMA2-70B W8A16量化</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/teacher_qualification.jsonl --w_bit 8 --a_bit 16 --act_method 3 --trust_remote_code True
  ```

### LLaMA3

#### <span id="llama3-70b-w8a16量化">LLaMA3-70B W8A16量化</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --a_bit 16 --w_sym False --mm_tensor False --anti_method m3 --act_method 3 --model_type llama3 --trust_remote_code True
  ```

### LLaMA3.1

#### <span id="llama31-8b-w8a8量化">LLaMA3.1-8B W8A8量化：</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L0 --anti_method m1 --act_method 1 --model_type llama3.1_fp --trust_remote_code True
  ```

#### <span id="llama31-70b-w8a8量化">LLaMA3.1-70B W8A8量化：</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --anti_method m3 --act_method 3 --model_type llama3.1_fp --trust_remote_code True
  ```

#### <span id="llama31-70b-w8a8量化搭配-kv-cache-int8量化">LLaMA3.1-70B W8A8量化搭配 KV cache int8量化</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --anti_method m3 --act_method 3 --model_type llama3.1_fp --use_kvcache_quant True --trust_remote_code True
  ```

#### <span id="llama31-70b-w8a8量化搭配attention量化">LLaMA3.1-70B W8A8量化搭配Attention量化</span>

- 当前仅支持基于BF16权重生成量化权重
- 需修改`modeling_llama.py`文件和`config.json`文件，配置方法参考[FA量化使用说明](../../docs/zh/user_guide/quantization_algorithms/quantization_algorithms/fa3_quant.md)。
- 相比于W8A8量化，需额外设置`use_fa_quant`参数为True

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --disable_level L5 --anti_method m4 --act_method 3 --model_type llama3.1_fp --use_fa_quant True --trust_remote_code True
  ```

#### <span id="llama31-70b-w8a8-pdmix量化prefill阶段-w8a8动态量化-decode阶段-w8a8量化-搭配-kv-cache-int8量化">LLaMA3.1-70B W8A8-pdmix量化(prefill阶段 W8A8动态量化, decode阶段 W8A8量化) 搭配 KV cache int8量化</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} \
  --save_directory {W8A8-pdmix量化权重路径} \
  --calib_file ../common/llama_calib_prompt.jsonl  \
  --anti_calib_file ../common/llama_anti_prompt.jsonl \
  --device_type npu \
  --anti_method m6 \
  --act_method 3 \
  --model_type llama3.1_fp \
  --use_kvcache_quant True \
  --disable_threshold 1 \
  --pdmix True \
  --trust_remote_code True
  ```

#### <span id="llama31-8b-instruct-w4a8_dynamic量化">LLaMA3.1-8B-Instruct W4A8_DYNAMIC量化</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {量化权重路径} --w_bit 4 --device_type npu --anti_method m3 --act_method 1 --model_type llama3.1_instruct --is_lowbit True --mm_tensor False --open_outlier False --group_size 32 --is_dynamic True --anti_calib_file ../common/llama_anti_prompt.json --trust_remote_code True
  ```

#### <span id="llama31-70b-instruct-w4a8_dynamic量化">LLaMA3.1-70B-Instruct W4A8_DYNAMIC量化</span>

  ```shell
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {量化权重路径} --w_bit 4 --device_type npu --anti_method m3 --act_method 1 --model_type llama3.1_instruct --is_lowbit True --mm_tensor False --open_outlier False --group_size 32 --is_dynamic True --anti_calib_file ../common/llama_anti_prompt.json --trust_remote_code True
  ```
