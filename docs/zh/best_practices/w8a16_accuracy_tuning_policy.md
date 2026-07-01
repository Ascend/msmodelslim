# W8A16精度调优策略

## 概述

W8A16量化是仅对权重进行量化，适用于对精度要求较高的场景。
W8A16量化精度调优策略是结合msModelSlim量化工具和精度测试工具precision tool进行精度验证和调优开展。因为W8A16本身量化后精度就很高，大部分模型量化后精度都能满足需求，所以就不结合具体的案例展开精度调试过程，如果发现有W8A16量化后精度不合格，可以参考本文的精度调优策略进行调试。另外，本文是精度调优策略，本文仅对涉及精度调优的参数以及方法进行说明。

注：
Transformers 版本适配说明：ChatGLM2-6B 模型需依赖 4.40.2 版本的 Transformers 库，若运行时出现 Transformers 相关报错，可尝试将库版本降至 4.40.2 以解决兼容性问题。

## 前期准备

参考以下两篇文档完成工具使用前准备工作
安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../install_guide/install_guide.md)
以及[大模型量化工具依赖安装](../user_guide/feature_guide/traditional_quantization_v0/foundation_model_compression.md#11-使用前准备)。

## 代码示例

W8A16量化及伪量化测精度过程示例（NPU）：

```python
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
4、离群值抑制AntiOutlier(W8A16)
"""
anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", dev_id=device_id)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

"""
5、回退层设置
"""
"""
因为一些量化后的网络层对精度影响太大了，所以需要让这些网络层使用浮点权重进行计算， disable_names中为需要进行回退的网络层。
"""
disable_names = []
disable_names.append('lm_head')

"""
6、执行PTQ量化校准 + 存储量化参数用于部署
"""
quant_config = QuantConfig(
    a_bit=16,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id,
    w_method='MinMax',
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

res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
for idx, item in enumerate(res):
    print(item)
```

在调用Calibrator.run()方法后，构建Calibrator时传入的model会被替换为伪量化模型，可以直接调用进行前向推理，用来测试对话效果，如果伪量化结果不理想，可以参考以下的方法进行调优。

## W8A16量化模型的精度调优步骤

### 1 调整离群值抑制

 msModelSlim工具使用AntiOutlierConfig生成离群值抑制配置，以使用离群值抑制功能。原理：通过抑制量化过程当中的异常值，从而提高量化模型的精度。W8A16量化建议使用m3，m3为AWQ算法，也是仅权重量化算法，可以理解为抑制权重中出现的异常值，但逻辑和激活值异常抑制算法是一样的。设置方式为：

```python
anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", dev_id=device_id)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()
```

补充：其中anti_method有六种算法，m3是用于权重量化的，另外五种是用于激活值量化的场景，如果是激活值精度调优建议从m1到m6逐一尝试。

### 2 量化参数选择

```python
quant_config = QuantConfig(
    a_bit=16,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id,
    w_method='MinMax',
    w_sym=True,
    mm_tensor=False
)
```

W8A16场景下，大部分模型采用最简单的MinMax算法精度就能控制在0.5%以内的精度损失。调优过程：MinMax->HQQ->GPTQ(耗时长，非必要情况不推荐)。
MinMax算法就是把简单的浮点数直接映射到int8的数据范围。
HQQ是一种无需训练数据，就能快速量化大型模型，且压缩质量与基于训练的方法相当的算法。
GPTQ是一种针对大规模预训练模型的高效后量化算法。
权重量化通常选择对称量化(w_sym=True)和per-channel量化(mm_tensor=False)。
增加回退层（建议最后进行调整），可以按照一定的经验，通过disable_names手动设置。

### 3 校准集调整

1.当算法层面无法提升精度时，可以增大校准数据集(10~50条)。
正常情况下，可以增加数据得到精度提升，但是到一定数据后，提高数据对精度影响有限。有些场景下，减少数据反而得到精度提升, 例如长数据场景。
2.针对特定场景切换成应用场景的数据作为校准集。
在选取时需要考虑模型部署时的具体推理场景，例如中文模型需要使用中文输入作为校准集；
英文模型使用英文输入；代码生成类模型则使用代码生成类任务；
中英文兼顾的模型考虑使用中英文混合的校准集合。
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

注： [Precision Tool 使用方法说明及数据集下载链接](../user_guide/feature_guide/traditional_quantization_v0/fake_quantization_accuracy_testing_tool.md)

### 4 量化回退

量化回退的原因：某些网络层对于量化比较敏感，量化后会带来较大的精度损失，这些层是不太适合量化的，应该使用浮点数进行计算，这个过程称之为回退（回退的都是线性层），可以通过设置disable_names控制哪些层应该被回退。
为什么回退的都是线性层：大模型中的线性层层数多、权重数量庞大且存在矩阵相乘（计算量大），通过量化线性层的权重和激活值，可以达到降低模型大小，减少计算量，降低内存占用，提升推理速度。
怎么判定敏感：终端的日志中会显示每一层算子激活量化输入的range_parm数值，range_parm数值越大越敏感。

仅权重量化操作流程：
1.判断需要回退的层，通过看日志中range_parm的数值，数值越大表明越需要回退。比如down_proj层，o_proj层。
2.操作方式：QuantConfig里设置参数，disable_names设置需要手动回退的量化敏感层。

另外，W8A16仅支持手动回退，不支持自动回退（设置disable_level='L0'）。
注：量化回退会造成一定的性能损失。

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

### 5 KV Cache int8量化

可在QuantConfig后调用kv_quant函数来开启KV Cache int8量化。

```python
quant_config = QuantConfig(
    a_bit=16,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id
).kv_quant()

```

长序列场景下KV Cache占用显存空间较大，通过KV Cache量化可以节约显存占用，增加并发数。
调用kv_quant函数会自动将QuantConfig中use_kvcache_quant设置为True。
use_kvcache_quant=True启用KV Cache量化，支持与W8A8、W8A16和稀疏量化同时使用。
