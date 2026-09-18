# QuantConfig

## 功能说明

量化参数配置类，保存量化过程中配置的参数。

## 函数原型

```python
QuantConfig(a_bit=8, w_bit=8, disable_names=None, dev_type='cpu', dev_id=None, act_method=1, pr=1.0, w_sym=True, mm_tensor=True, w_method='MinMax', co_sparse=False, fraction=0.01, nonuniform=False, is_lowbit =False, do_smooth=False, use_sigma=False, sigma_factor=3, disable_last_linear: bool=True, use_kvcache_quant=False, is_dynamic=False, open_outlier=True, group_size=64, percdamp=0.01, pdmix=False)
```

## 参数说明

| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| a_bit | 输入 | 激活值量化bit。| 可选。<br>数据类型：int。<br>可选值为4，8和16，默认为8。<br>大模型量化场景下，可配置为4，8或16。per-group的场景下需配置为8或16（a_bit=8 当前仅在 w4a8 量化中使用）。is_dynamic参数配置为True，使用per-token动态量化场景下，需配置为4或8。<br>大模型稀疏量化场景下，需配置为8。 <br>w4a4场景仅支持per-token动态量化，仅支持配置is_dynamic为True，其他参数不适用，该场景目前仅支持Qwen3系列稠密模型，并且不建议使用异常值抑制功能。|
| w_bit | 输入 | 权重量化bit。| 可选。<br>数据类型：int。<br> 可选值为4和8，默认为8。<br>大模型量化场景下，可配置为4或8。is_dynamic参数配置为True，使用per-token动态量化场景下，需配置为4或8。<br>大模型稀疏量化场景下，需配置为4。 |
| disable_names | 输入 | 需排除量化的节点名称，即手动回退的量化层名称。<br>如精度太差，推荐回退量化敏感层，如分类层、输入层、检测head层等。| 可选。<br>数据类型：object。 |
| dev_type | 输入 | device类型。| 可选。<br>数据类型：object。<br>可选值：['cpu', 'npu']，默认为'cpu'。 |
| dev_id | 输入 | Device ID。| 可选。<br>数据类型：int。<br>默认值为None。<br>仅在“dev_type”配置为“npu”时生效。“dev_id”指定的Device ID优先级高于环境变量配置的Device ID。 |
| act_method | 输入 | 激活值量化方法。| 可选。<br>数据类型：int。可选值如下所示，默认为1。<br>(1) 1代表Label-Free场景的min-max量化方式。<br>(2) 2代表Label-Free场景的histogram量化方式。<br>(3) 3代表Label-Free场景的自动混合量化方式，LLM大模型场景下推荐使用。<br>说明：开启lowbit稀疏量化功能时，不支持选择值3。|
| pr | 输入 | 量化选择概率。| 可选。<br>数据类型：float。取值范围：[0,1]。<br>默认值：1.0，建议取值1.0。|
| w_sym | 输入 | 权重量化是否为对称量化。| 可选。<br>数据类型：bool。默认为True。<br>W8A8和W4A8_DYNAMIC场景仅支持配置为True。<br>不适用于w4a4场景。|
| mm_tensor | 输入 | 选择进行per-channel量化或per-tensor量化。| 可选。<br>数据类型：bool。默认为True。<br>True: per-tensor量化。<br>False: per-channel量化，建议选择该量化方式。|
| w_method | 输入 | 选择权重量化策略。| 可选。<br>数据类型：str。默认为'MinMax'，可选值：'MinMax','GPTQ','HQQ','NF'。<br>MinMax、HQQ和NF支持Data-Free。<br>GPTQ不支持Data-Free。<br>NF不支持per-tensor,per-channel,per-group,lowbit,通信量化，kvcache量化，离群值抑制混合使用。|
| co_sparse | 输入 | 是否开启稀疏量化功能。| 可选。<br>数据类型：bool。默认值：False，不开启稀疏量化。<br>大模型稀疏量化场景下，优先使用lowbit稀疏量化功能，开启lowbit稀疏量化后，co_sparse参数自动失效。|
| fraction | 输入 | 模型权重稀疏量化过程中被保护的异常值占比。| 可选。<br>数据类型：float。取值范围[0.01,0.1]。默认值为0.01。|
| nonuniform | 输入 | 是否在稀疏量化中采用非均匀量化。| 可选。<br>数据类型：bool。默认值为False。|
| is_lowbit | 输入 | 是否开启lowbit量化功能。| 可选。<br>数据类型：bool默认为False，不开启lowbit量化功能。<br>设置为True之后，有以下两种情况：<br>(1) 当w_bit=4，a_bit=8时，为大模型稀疏量化场景，表示开启lowbit稀疏量化功能。大模型稀疏量化场景下，优先使用lowbit稀疏量化功能，开启lowbit稀疏量化后，co_sparse参数自动失效。`注意：若要开启 W4A8_DYNAMIC 量化类型，需要同时开启per-group和per-token`<br>(2) 其他场景为大模型量化场景，会开启量化自动精度调优功能。当前量化自动精度调优框架支持W8A8，W8A16、W4A8_DYNAMIC量化。<br>说明：per-group量化场景下，需协同设置is_lowbit为True，open_outlier为False。 |
| do_smooth | 输入 | 是否开启smooth功能。<br>启用do_smooth功能后，平滑激活值。|可选。<br>数据类型：bool。默认为False，不开启smooth功能。 |
| use_sigma | 输入 | 是否启动sigma功能。<br>启用use_sigma功能后，可根据正态分布数值特点进行异常值保护。|可选。<br>数据类型：bool。默认为False，不开启sigma功能。 |
| sigma_factor | 输入 | 启用sigma功能后sigma_factor的值，用于限制异常值的保护范围。|可选。<br>数据类型：float。<br>默认为3，取值范围为[3, 4]。 |
| disable_last_linear | 输入 | 是否自动回退至最后线性层linear。<br>当该参数为True时，会自动回退最后一个线性层linear。例如LLaMA2-13B模型的[lm_head]层，ChatGLM2-6B的[transformer.output_layer]层。|可选。<br>数据类型：bool。<br>默认为True。<br>True：自动回退最后一个线性层linear。False：不会回退最后一个线性层linear。 |
| use_kvcache_quant | 输入 | 是否使用kvcache量化功能。|可选。<br>数据类型：bool。<br>默认为False。<br>True：使用kvcache量化功能。False：不使用kvcache量化功能。<br>说明：将此参数设置为true并配置表1 量化配置表里的kv_quant参数后，方可使用kvcache量化功能。 |
| is_dynamic | 输入 | 是否使用per-token动态量化功能。|可选。<br>数据类型：bool。<br>默认为False。<br>True：使用per-token动态量化。False：不使用per-token动态量化。|
| open_outlier | 输入 | 是否开启权重异常值划分。|可选。<br>数据类型：bool。<br>默认为True。<br>True：开启权重异常值划分。False：关闭权重异常值划分。<br>说明：(1)仅在lowbit设置为True时生效。(2)per-group量化场景下，需协同设置is_lowbit为True，open_outlier为False。|
| group_size | 输入 | per-group量化中group的大小。|可选。<br>数据类型：int。<br>默认值为64，支持配置为32，64，128，256。<br>说明：仅适用于per-group量化场景，需协同设置is_lowbit为True，open_outlier为False。<br>不适用于w4a4场景。|
| percdamp | 输入 | GPTQ算法的矩阵正定偏置系数，用于保障计算过程的稳定性。当GPTQ运行出现非正定矩阵导致的报错时，可以适当增大该参数。|可选。<br>数据类型：float。<br>取值范围为[0,1]，默认值为0.01。<br>说明：仅适用于w_method为GPTQ算法的情况。|
| pdmix | 输入 | 是否同时提供动态量化参数和静态量化参数。|可选。<br>数据类型：bool。<br>默认为False。<br>True：同时生成动态量化参数和静态量化参数。False：仅生成单一类型的量化参数。<br>说明：设置为True时，会同时提供动态量化参数和静态量化参数，便于在推理时根据实际需求选择使用哪种参数类型。设置is_dynamic=True时，不支持此功能。|

## 调用示例一

根据实际需求，在QuantConfig初始化中完成所有参数的配置。

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(pr=1.0, mm_tensor=False)
```

## 调用示例二

在QuantConfig初始化中完成核心参数（w_bit，a_bit，disable_names，disable_last_linear，dev_type，dev_id）的配置后，再根据不同量化场景，配置表中的参数。目前支持的量化场景有稀疏量化、权重量化、kvcache量化、权重激活量化和模拟多卡量化，具体参数配置情况和调用示例请参考下表。

| 量化类型 | 需配置参数列表 | 调用示例 |
| --- | --- | --- |
| weight_quant<br>权重量化的参数初始化，即 w8a16。<br>说明：使用多模态模型时，会自动将conv层回退，不进行量化处理。| w_method，mm_tensor，w_sym，group_size  | quant_config = QuantConfig(w_bit=8,disable_names=disable_names,dev_type='npu',dev_id=0).weight_quant (w_sym=False) |
| weight_activation_quant<br>权重激活量化的参数初始化， 即w8a8。| act_method，pr，is_dynamic  | quant_config = QuantConfig(w_bit=8,disable_names=disable_names,dev_type='npu',dev_id=0).weight_activation_quant (act_method=2) |
| sparse_quant<br>稀疏量化的参数初始化。| act_method，fraction，nonuniform，is_lowbit，do_smooth，use_sigma，sigma_factor  | quant_config = QuantConfig(w_bit=4,disable_names=disable_names,dev_type='npu',dev_id=0).sparse_quant(is_lowbit=True) |
| kv_quant<br>kvcache对称量化的参数初始化。<br>说明：调用本函数后，会自动将use_kvcache_quant设置为True。| kv_sym用于是否使用kvcache对称量化功能，为可选参数，数据类型为bool，默认值为True。<br>True：使用kvcache对称量化功能。False：使用kvcache非对称量化功能。  | quant_config = QuantConfig(w_bit=8,disable_names=disable_names,dev_type='npu',dev_id=0).kv_quant(kv_sym=True) |
| simulate_tp<br>模拟多卡量化的参数初始化。<br>说明：默认会对通信层的linear进行模拟多卡量化，且每张卡会使用不同的量化参数。再通过使能enable_communication_quant参数启用模拟多卡通信量化的功能。| tp_size用于模拟多卡量化时的卡数，为必选参数，数据类型为int，数据取值范围为[2,4,8,16]。<br>enable_communication_quant用于是否使用模拟多卡通信量化的功能，为可选参数，数据类型为bool，默认为True。<br>True：使用模拟多卡通信量化场景。False：不使用模拟多卡通信量化场景。<br>enable_per_device_quant用于配置模拟多卡通信量化的方式。为可选参数，数据类型为bool，默认值为True。<br>True：每张卡使用不同的reduce scale。False：每张卡使用相同的reduce scale。| quant_config=QuantConfig(w_bit=4,disable_names=disable_names,dev_type='npu',dev_id=0).simulate_tp(tp_size=4, enable_communication_quant=True) |
| NF4<br>Normal Float 4bit量化的参数初始化。| w_method用于选择不同的权重量化类型，此处选择'NF'量化。<br>block_size用于设置一个block内的元素的个数，类似per-group量化，越小量化精度越高。为可选参数，数据类型为int，默认值为64，数据取值范围为[64, 128, 256, 512, 1024, 2048, 4096]。  | quant_config = QuantConfig(w_bit=4,a_bit=16,dev_type='npu',dev_id=0).weight_quant(w_method='NF', block_size=64) |
| fa_quant<br>FA3量化的参数初始化。|fa_amp用于配置自动精度回退，根据想要回退的layer的数量来设置。<br>数据类型为int，默认值为0。数据取值范围是大于等于0，并且小于等于模型layer数量，如果超出模型的layer数量将会取模型的最大layer数量为回退层数。 | quant_config=QuantConfig(w_bit=8,  a_bit=8, disable_names=disable_names,dev_type='npu',dev_id=0).fa_quant(fa_amp=5) |
