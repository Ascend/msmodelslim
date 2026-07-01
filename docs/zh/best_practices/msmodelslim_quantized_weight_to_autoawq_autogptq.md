# msModelSlim量化权重转AutoAWQ&AutoGPTQ使用指南

## 概述

msModelSlim权重格式与开源工具AutoAWQ、AutoGPTQ的格式存在差异，因此本文的目的是提供一份指南，用于将msModelSlim量化后的权重转换为与如上的开源工具格式一致的权重，以实现qwen2-7b W4A16转换后的权重能直接以huggingface形式加载。
本指南仅支持如下配置的权重转换：
W4A16 + per_group + AWQ
W4A16 + per_group + GPTQ
W4A16 + per_channel + GPTQ
W8A16 + per_group + GPTQ
W8A16 + per_channel + GPTQ

使用平台：
msModelSlim量化：NPU
转换脚本：CPU
AutoAWQ：GPU
AutoGPTQ：GPU

## msModelSlim量化

### 前期准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../install_guide/install_guide.md)。

参考大模型量化中的[使用前准备](../user_guide/feature_guide/traditional_quantization_v0/foundation_model_compression.md#11-使用前准备)，进行依赖安装。

### 量化使用说明

量化脚本跟正常的量化脚本一样，可以参考：[w8a8精度调优策略](w8a8_accuracy_tuning_policy.md) 。
本文以W4A16量化方式示例进行说明。需要注意的地方有三处:
a.在离群值抑制配置（AntiOutlierConfig）中，a_bit和w_bit应根据量化方式进行设置。当anti_method被设置为"m3"时，代表使用AWQ算法；而对于GPTQ算法，则不需要使用离群值抑制模块，此时可以将相关配置注释掉。

```python
anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", a_bit=16, w_bit=4, dev_id=device_id, w_sym=True)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()
```

b.QuantConfig配置
per_channel和per_group的参数配置是有差异的。
(1)per_group需要配置这三个参数：is_lowbit=True, open_outlier=False, group_size=128。
(2)per_channel场景下，如下的三个参数不需要配置，注释掉：is_lowbit=True, open_outlier=False, group_size=128。
(3)如果是AutoGPTQ需要更改w_method为='GPTQ', 另外开启GPTQ跑量化时间相对较长。
如下为AutoAWQ的per_group配置：

```python
quant_config = QuantConfig(
    a_bit=16,                      # 激活值量化位数
    w_bit=4,                       # 权重量化位数
    disable_names=disable_names,   # 手动回退的量化层名称
    mm_tensor=False,               # 默认True，表示使用per-tensor量化，False为per-channel量化
    dev_type='npu',                # 量化的工具为NPU
    dev_id=0,
    w_sym=True,                    # 对称量化
    w_method='MinMax',             # 权重量化策略
    is_lowbit=True,                # 如下为per_group场景下的设置，如果是per-channel量化注释掉如下三个参数
    open_outlier=False,
    group_size=128
)
```

c.关于保存的权重文件
本脚本仅支持未切片的safetensors权重转换，所以使用保存量化权重文件的时候，不要使用分片保存。
参考链接：[save()接口说明](../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_save().md)

```python
calibrator.save(output_path, safetensors_name=None, json_name=None, save_type=None, part_file_size=None)
```

### 转换脚本使用说明

转换脚本路径位于：[ms_to_vllm.py](https://gitcode.com/Ascend/msmodelslim/blob/master/example/ms_to_vllm.py)

经过上一步1.1使用msModelSlim对权重进行量化，生成quant_model_description_w4a16.json和quant_model_weight_w4a16.safetensors，再使用转换脚本ms_to_vllm.py进行权重格式转换，生成转换后的safetensors文件，用法如下：

```python
命令：
python ms_to_vllm.py --model {weighted_safetensors_path} --json {weighted_json_path} --save_path  {converted_safetensors_path} --w_bit {weight_bit} --target_tool  {target_convert_tool}

说明：
    model，必选参数，string类型，用于表示传入量化后的safetensors权重文件，可传入文件的绝对路径和相对路径
    json，必选参数，string类型，用于表示传入量化后的json权重描述文件，可传入文件的绝对路径和相对路径
    save_path，可选参数，string类型，默认值为./res.safetensors，另外save_path仅支持在已有的目录路径下创建保存文件，不支持创建目录
    w_bit，可选参数，int类型，可选值[4, 8]，默认值为4，表示量化的权重位数为4
    target_tool，可选参数，string类型，可选值[awq, gptq], 默认值为awq，表示转换的目标工具为AutoAWQ

使用示例：
首先将权重转换脚本拷贝到量化权重目录下，然后在该目录下执行如下命令，最终在该目录下生成转换后的权重脚本文件res.safetensors：
python ms_to_vllm.py --model ./quant_model_weight_w4a16.safetensors --json ./quant_model_description_w4a16.json --save_path res.safetensors --target_tool awq

```

## 开源工具AutoAWQ量化以及推理

### 前期准备

开源工具相关的环境配置、量化和推理参考 github 上的 readme.md，链接：[AutoAWQ README](https://github.com/casper-hansen/AutoAWQ)。

### 量化使用说明

AutoAWQ量化。需要注意的是，Version 使用 GEMM 时，如果没有传入数据集可能会报错，需要传入数据集 val.jsonl 文件，参考：[AutoAWQ issue #506](https://github.com/casper-hansen/AutoAWQ/issues/506)。
数据集获取地址：[val.jsonl.zst](https://huggingface.co/datasets/mit-han-lab/pile-val-backup/blob/main/val.jsonl.zst)。请注意 `trust_remote_code` 为 `True` 时可能执行浮点模型权重中的代码文件，请确保浮点模型来源安全可靠。
AutoAWQ量化脚本示例如下：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

model_path = 'qwen2_7b_instruct'  # 浮点模型权重路径
quant_path = 'quant_qwen2_7b_awq_4_g128'  # 浮点模型经过量化后的保存路径

# q_group_size和 msModelSlim量化的group_size对应，保持一致
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, low_cpu_mem_usage=True, use_cache=False, device_map='auto',
    local_files_only=True,
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

data = load_dataset("json", data_files='./val.jsonl')['train']

calib_data = [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')

```

### 推理使用说明

首先，修改AutoAWQ量化后权重路径的model.safetensors.index.json文件，请将文件中的weight_map中的权重文件名称修改为第1.2节中的转换脚本所生成的权重文件名，然后将权重文件替换为第1.2节中转换脚本所生成的权重文件，最后运行推理脚本。请注意`trust_remote_code`为`True`时可能执行浮点模型权重中的代码文件，请确保浮点模型来源安全可靠。

AutoAWQ推理脚本测试对话示例如下：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = './quant_qwen2_7b'  # 浮点模型经过量化后的保存路径

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True, local_files_only=True)

test_prompt = "what is deep learning:"
test_input = tokenizer(test_prompt, return_tensors="pt")
print("model is inferring...")
model.eval()
generate_ids = model.generate(
    test_input.input_ids.cuda(),
    attention_mask=test_input.attention_mask.cuda(),
    max_new_tokens=16
)

res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
for idx, item in enumerate(res):
    print(item)
```

如果没有使用AutoAWQ量化获取量化配置文件，直接使用msModelSlim转换的量化后模型进行推理，需要做以下步骤：

1.将浮点模型的原始配置文件复制到msModelSlim量化后转换生成的权重文件目录中。
2.修改量化后权重路径的 model.safetensors.index.json 文件，请将文件中的 weight_map 中的权重文件名称修改为第1.2节中的转换脚本所生成的权重文件名。
3.修改 config.json 文件，添加 quantization_config 参数，bits为量化的权重位数，group_size 和msModelSlim量化的 group_size 对应，保持一致。可参考第2.2节使用AutoAWQ量化后生成的 config.json 文件进行配置。
此处以Qwen2-7B-Instruct，W4A16+AWQ为例，在 config.json 添加 quantization_config 参数：

```json
{
  "model_type": "qwen2",
  "torch_dtype": "bfloat16",
  ··· 上述为原始json参数示例 ···
  ··· 添加下方参数 ···
  "quantization_config": {
    "bits": 4,
    "group_size": 64,
    "version": "gemm",
    "zero_point": true
  }
}
```

4.参考第2.3节推理脚本运行推理。

## 开源工具AutoGPTQ量化以及推理

### 前期准备

开源工具相关的环境配置、量化和推理参考 github 上的 readme.md，链接如下：[AutoGPTQ README](https://github.com/AutoGPTQ/AutoGPTQ)。

### 量化使用说明

msModelSlim转换为AutoGPTQ权重格式进行推理和AutoAWQ同理，首先去阅读AutoGPTQ的readme.md(链接如上第3.1节)，参考量化的示例，修改相关配置参数，然后进行量化，最后生成量化权重文件。
修改的配置包括路径和BaseQuantizeConfig接口，在BaseQuantizeConfig接口中，bits为量化的权重位数，对应msModelSlim中的w_bit；per_group场景下，group_size设置的值与msModelSlim一致，在per_channel场景下，group_size设置为-1。

### 推理使用说明

将经过msModelSlim量化以及经过转换脚本转换后的res.safetensors文件传入GPU生成的量化权重目录，替换掉之前的量化权重文件，文件名保持一致，其他文件不需要修改。
推理脚本示例如下：

```python
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_path = "./qwen2_7b_instruct"      # 浮点模型权重路径
quant_path = "./ms_to_gptq"             # 浮点模型经过量化后的保存路径

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

# 加载未量化的模型，默认情况下，模型总是会被加载到 CPU 内存中
model = AutoGPTQForCausalLM.from_quantized(quant_path, device="cuda:0")
print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))
```

如果没有使用autoGPTQ量化获取量化配置文件，直接使用msModelSlim转换的量化后模型进行推理，需要做以下步骤：

1.将浮点模型的原始配置文件 config.json，model.safetensors.index.json 复制到msModelSlim量化后转换生成的权重文件目录中。
2.修改量化后权重路径的 model.safetensors.index.json 文件，请将文件中的 weight_map 中的权重文件名称修改为第1.2节中的转换脚本所生成的权重文件名。
3.在量化后权重路径下新建 quantize_config.json 文件，bits为量化的权重位数，group_size 和msModelSlim量化的 group_size 对应，保持一致。可参考使用autoGPTQ量化后生成的 quantize_config.json 文件进行配置。
此处以Qwen2-7B-Instruct，W4A16+GPTQ为例，新建 quantize_config.json 文件：

```json
{
  "bits": 4,
  "group_size": 64
}
```

4.参考第3.3节推理脚本运行推理。

## 结论

经过上述步骤，成功完成msModelSlim在NPU上的量化，并且量化权重经过转换脚本转换后能够在AutoAWQ和AutoGPTQ推理成功。
