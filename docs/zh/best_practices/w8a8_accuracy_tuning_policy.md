# W8A8量化精度调优策略

## 概述

W8A8量化精度调优策略是结合msModelSlim量化工具和精度测试工具precision tool进行精度验证和调优开展。大模型经过W8A8量化后精度损失大，可以参考本文的精度调优策略进行调优。

注：
1、针对模型的特殊配置建议：若使用 ChatGLM 模型（如 ChatGLM2-6B），建议手动设置线程数，可提升运行效率；其他模型无需额外配置线程数。

2、Transformers 版本适配说明：ChatGLM2-6B 模型需依赖 4.40.2 版本的 Transformers 库，若运行时出现 Transformers 相关报错，可尝试将库版本降至 4.40.2 以解决兼容性问题。

```python
# 设置线程数
export OMP_NUM_THREADS=48
```

## 前期准备

参考以下两篇文档完成工具使用前准备工作
安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../install_guide/install_guide.md)
以及[大模型量化工具依赖安装](../user_guide/feature_guide/traditional_quantization_v0/foundation_model_compression.md#11-使用前准备)  。

## 代码示例

W8A8量化及伪量化测精度过程示例（NPU）：

```python
"""
1、导入相关依赖
"""
import os
import json
import torch
import torch_npu # 如果需要使用npu进行量化
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from precision_tool.precision_tool import PrecisionTest # precision_tool用于伪量化测精度

SEQ_LEN_OUT = 100
device_id = 0
batch_size = 5

# 如果使用npu进行量化需开启二进制编译，避免在线编译算子
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)


"""
2、导入相关模型
"""
fp16_path = '/data/chatglm2-6b' # 原始浮点模型路径

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=fp16_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=fp16_path,
    local_files_only=True,
    device_map="auto",
    torch_dtype="auto"
).eval()

"""
数据集测原始模型浮点精度（此示例中选择的是boolq）
"""
precision_test = PrecisionTest(model, tokenizer, "boolq", batch_size, "npu")
precision_test.test()


"""
3、获取校准数据
"""
# 一般数据都在cpu上，用npu进行量化的时候都需要指定数据到npu设备上
def build_prompt(title, text, passage):
    prompt = f"{title} -- {passage}\nQuestion:{text}?\nAnswer:"
    return prompt

def get_calib_dataset(tokenizer, calib_list, device=f"npu:{device_id}"):
    calib_dataset = []
    for calib_data in calib_list:
        title = calib_data["title"]
        text = calib_data["question"]
        passage = calib_data["passage"]
        queries = build_prompt(title, text, passage)
        inputs = tokenizer(queries, return_tensors='pt')
        calib_dataset.append([inputs.data['input_ids'].to(device), inputs.data['attention_mask'].to(device)])
    return calib_dataset

entry = "/path/to/calib_dataset" # 此示例中校准数据选取"precision_tool/dataset/boolq/dev.jsonl"
calib_set = []
i = 0
with open(entry, encoding="utf-8") as file:
    for line in file:
        data = json.loads(line) # 将字符串转换为字典
        while i < 50: # 获取50条校准数据
            calib_set.append(data)
            i += 1

dataset_calib = get_calib_dataset(tokenizer, calib_set)


"""
4、离群值抑制AntiOutlier(W8A8)
"""
anti_config = AntiOutlierConfig(anti_method="m2", dev_type="npu", dev_id=device_id)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()


"""
5、回退层设置
"""
"""
因为一些量化后的网络层对精度影响太大了，所以需要让这些网络层使用浮点权重进行计算， disable_names中为需要进行回退的网络层。
"""
disable_names = []


"""
6、执行PTQ量化校准 + 存储量化参数用于部署
"""
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id,
    act_method=3,
    pr=1.0,
    w_sym=True,
    mm_tensor=False
)

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  # disable_level: 自动回退n个linear
calibrator.run()  # 执行PTQ量化校准
calibrator.save('/save/path', save_type=["safe_tensor", "numpy"]) # "safe_tensor"对应safetensors格式权重，"numpy"对应npy格式权重


"""
数据集测伪量化模型精度（此示例中选择的是boolq）
"""
precision_test = PrecisionTest(model, tokenizer, "boolq", batch_size, "npu")
precision_test.test()


"""
7、伪量化验证一轮推理（可选）
"""
print("testing quantized weights...")
test_prompt = "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:"
test_input = tokenizer(test_prompt, return_tensors="pt")
print("model is inferring...")
model = model.to(f"npu:{device_id}")
model.eval()
generate_ids = model.generate(
    test_input.input_ids.to(f"npu:{device_id}"),
    attention_mask=test_input.attention_mask.to(f"npu:{device_id}"),
    max_new_tokens=SEQ_LEN_OUT
)

res = tokenizer.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
for idx, item in enumerate(res):
    print(item)

```

在调用Calibrator.run()方法后，构建Calibrator时传入的model会被替换为伪量化模型，可以直接调用进行前向推理，用来测试对话效果。
如果伪量化结果不理想，可以参考以下方法进行调优:

## W8A8量化模型的调参配置步骤

### 1 调整离群值抑制

原因：通过抑制量化过程中的异常值，使能后续更好的量化。

```python
anti_config = AntiOutlierConfig(
    anti_method="m2",
    dev_type="npu",
    dev_id=device_id
)
```

可优化参数——anti_method

m1：SmoothQuant算法
m2：SmoothQuant升级版
m3：AWQ算法（适用于W8A16/W4A16）
m4：SmoothQuant优化算法
m5：CBQ算法
m6：Flex smooth量化算法

w8a8适用m1、m2、m4、m5、m6
建议从m1尝试到m6，因为不同模型对不同离群抑制算法表现不一样，当前m2已适配qwen-vl和llava-v1.5-7b多模态模型

### 2 量化参数选择

```python
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id,
    act_method=3,
    pr=1.0,
    w_sym=True,
    mm_tensor=False
)

calibrator = Calibrator(
    model,
    quant_config,
    calib_data=dataset_calib,
    disable_level='L0'
)
```

可优化参数——disable_names、disable_level、act_method
【增加回退层（建议最后进行调整），可以按照一定经验，通过disable_names手动设置回退层，或使用disable_level自动回退功能按照一定的标准自动回退对精度影响比较大的Linear层】

disable_names: 手动指定回退层（根据理论经验和日志信息）
disable_level='L0': 自动回退

act_method：激活值量化方法
    act_method默认值为1，该参数可选1、2、3
    1代表min-max量化方式；
    2代表histogram量化方式；
    3代表min-max和histogram混合的量化的方式。
    LLM大模型场景下建议使用3。

### 3 校准集调整

1.当算法层面无法提升精度时，可以增大校准数据集(10~50条)。
（正常情况下，可以增加数据得到精度提升，但是到一定数据后，提高数据对精度影响有限。有些场景下，减少数据反而得到精度提升。（例如长数据场景））
2.针对特定场景切换成应用场景的数据作为校准集。
（在选取时需要考虑模型部署时的具体推理场景，例如中文模型需要使用中文输入作为校准集；英文模型使用英文输入；代码生成类模型则使用代码生成类任务；中英文兼顾的模型考虑使用中英文混合的校准集）。
3.剔除量化前后模型输出变化较大的数据作为校准集。
4.注意校准集格式：
在下述示例中，get_calib_dataset的作用是调整校准集格式，以boolq数据集作为校准集为例，boolq数据集格式为 dict={"question":str, "title":str, "answer":bool, "passage":str}，而tokenizer中需要传入的数据格式为："str"（单个提示词）、"List[str]"（批量或单个提示词）或 "List[List[str]]"（批量提示词）。

```python
def get_calib_dataset(tokenizer, calib_list, device=f"npu:{device_id}"):
    calib_dataset = []
    for calib_data in calib_list:
        title = calib_data["title"]
        text = calib_data["question"]
        passage = calib_data["passage"]
        queries = build_prompt(title, text, passage)
        inputs = tokenizer(queries, return_tensors='pt')
        calib_dataset.append([inputs.data['input_ids'].to(device), inputs.data['attention_mask'].to(device)])
    return calib_dataset
```

注： 需要将msmodelslim文件夹下的[precision_tool文件夹](https://gitcode.com/Ascend/msmodelslim/tree/master/precision_tool)和[security文件夹](https://gitcode.com/Ascend/msmodelslim/tree/master/security)复制一份出来，和量化脚本放置于同一目录下，再将待测试数据集放入precision_tool文件夹中，具体操作见：[Precision Tool 使用方法说明及数据集下载链接](../user_guide/feature_guide/traditional_quantization_v0/fake_quantization_accuracy_testing_tool.md)

### 4 量化回退

大模型需要量化的原因：模型量化可以降低模型大小，减少计算量，降低内存占用，提升推理速度。

大模型量化线性层的原因：大模型中的线性层层数多、权重数量庞大且存在矩阵相乘（计算量大），通过量化线性层的权重和激活值，可以达到降低模型大小，减少计算量，降低内存占用，提升推理速度。

量化回退的原因：某些线性层对于量化比较敏感，量化后会带来一定的精度损失，这些层是不太适合量化的，应该使用浮点数进行计算，这个过程称之为回退，可以通过设置disable_names控制哪些线性层应该被回退。

怎么判定敏感：终端的打印日志中会显示每一层算子激活量化输入的range_parm数值，range_parm数值越大越敏感。
终端打印日志示例：

```python
时间戳 - msmodelslim-logger - INFO - use histogram observer:transformer.encoder.layers.27.mlp.dense_h_to_4h.quant_input, range_parm:41.21875
```

此示例中的 range_parm:41.21875 数值就很大（和日志中其他层的range_parm相比），说明该层敏感，需要回退。

量化回退的方法：分为手动回退和自动回退两个方法（可叠加使用），建议先手动回退，如果不清楚该回退哪些模型层或者手动回退精度不好的话，再自动回退。

注：量化回退会造成一定的性能损失。

#### 手动回退——disable_names

disable_names=[]: []手动回退层名称，如果不添加则不回退

按以下顺序进行回退：
1、回退down_proj层（精度敏感）：mlp的采样层，（如果没有标识出down_proj就看out_features, 数值小的就是下采样层)。
2、回退o_proj层（通常精度敏感）：是self_attention中调用的最后一个线性层，(model中打出来的只是初始化时的顺序，要去模型代码里看实际调用顺序) 。
3、根据理论经验或终端打印日志中的range_parm数值大小找出量化敏感层进行回退。

如下示例为手动回退chatglm2-6b的所有down_proj层：

```python
disable_names=[
    'transformer.encoder.layers.0.mlp.dense_4h_to_h',
    'transformer.encoder.layers.1.mlp.dense_4h_to_h',
    'transformer.encoder.layers.2.mlp.dense_4h_to_h',
    'transformer.encoder.layers.3.mlp.dense_4h_to_h',
    'transformer.encoder.layers.4.mlp.dense_4h_to_h',
    'transformer.encoder.layers.5.mlp.dense_4h_to_h',
    'transformer.encoder.layers.6.mlp.dense_4h_to_h',
    'transformer.encoder.layers.7.mlp.dense_4h_to_h',
    'transformer.encoder.layers.8.mlp.dense_4h_to_h',
    'transformer.encoder.layers.9.mlp.dense_4h_to_h',
    'transformer.encoder.layers.10.mlp.dense_4h_to_h',
    'transformer.encoder.layers.11.mlp.dense_4h_to_h',
    'transformer.encoder.layers.12.mlp.dense_4h_to_h',
    'transformer.encoder.layers.13.mlp.dense_4h_to_h',
    'transformer.encoder.layers.14.mlp.dense_4h_to_h',
    'transformer.encoder.layers.15.mlp.dense_4h_to_h',
    'transformer.encoder.layers.16.mlp.dense_4h_to_h',
    'transformer.encoder.layers.17.mlp.dense_4h_to_h',
    'transformer.encoder.layers.18.mlp.dense_4h_to_h',
    'transformer.encoder.layers.19.mlp.dense_4h_to_h',
    'transformer.encoder.layers.20.mlp.dense_4h_to_h',
    'transformer.encoder.layers.21.mlp.dense_4h_to_h',
    'transformer.encoder.layers.22.mlp.dense_4h_to_h',
    'transformer.encoder.layers.23.mlp.dense_4h_to_h',
    'transformer.encoder.layers.24.mlp.dense_4h_to_h',
    'transformer.encoder.layers.25.mlp.dense_4h_to_h',
    'transformer.encoder.layers.26.mlp.dense_4h_to_h',
    'transformer.encoder.layers.27.mlp.dense_4h_to_h',
]

```

#### 自动回退——disable_level

自动回退会根据range_parm参数由大到小排序回退对精度影响比较大的Linear层。
设置disable_level='Lx'，x为自动回退的linear层数量，会在终端显示回退的层名称，disable_level='L0'即为不进行回退，x设置的数量超过模型层数就是全部回退，并且也不报错。

### 5 KV Cache int8量化

可在QuantConfig后调用kv_quant函数来开启KV Cache int8量化。

```python
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id
).kv_quant()
```

长序列场景下KV Cache占用显存空间较大，通过KV Cache量化可以节约显存占用，增加并发数。
调用kv_quant函数会自动将QuantConfig中use_kvcache_quant设置为True。
use_kvcache_quant=True启用KV Cache量化，支持与W8A8、W8A16和稀疏量化同时使用。

### 6 FA3量化

[FA量化使用说明](../user_guide/feature_guide/traditional_quantization_v0/foundation_model_quantization_and_calibration.md#3-fa3量化)

### 7 以chatglm2-6b为例，逐步进行调优后的精度改变

表格中精度为按下述步骤依次进行操作所得（仅作参考）

  | 步骤                 | 参数更改描述                                                                                                                                                                | 精度 |
  | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
  | 原浮点模型精度           |     无                     | 0.794   |
  | 添加QuantConfig量化参数          | disable_names = [] ，act_method = 3， disable_level = "L0"， dataset_calib 数据量为2条                                                                                                           | 0.519   |
  | 添加离群值抑制             | anti_method = "m2"|  0.497 |
  | 增加boolq校准数据            | 从2条增加到50条|  0.505 |
  | 增加量化回退（手动回退）           | 手动回退所有layer中的mlp.down_proj                          | 0.793   |
  | 增加量化回退（自动回退）          |        disable_names = []，设置 disable_level = "L28"                                                                                             | 0.791  |
  | 增加量化回退（手动回退+自动回退）             | disable_names手动回退10层，disable_level = "L28"|  0.795 |

### 8 模型量化后的部署推理

模型量化后，可基于MindIE、vLLM等进行部署推理。
