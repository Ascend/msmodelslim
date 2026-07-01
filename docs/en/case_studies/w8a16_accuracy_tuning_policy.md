# W8A16 Accuracy Tuning Strategy

## Overview

W8A16 quantization quantizes only the weights and is suitable for scenarios that require high accuracy.
The W8A16 accuracy tuning strategy combines the msModelSlim quantization tool with the precision test tool `precision_tool` for validation and tuning. Because W8A16 itself usually gives very high accuracy after quantization, most models can meet the requirement after quantization, so this document does not present a specific case study. If you find that W8A16 accuracy does not meet the requirement, you can follow the tuning strategy in this document. This document focuses only on the parameters and methods related to accuracy tuning.

Note:
Transformers version compatibility: The ChatGLM2-6B model depends on Transformers 4.40.2. If you see a Transformers-related error during runtime, try downgrading to 4.40.2 to resolve the compatibility issue.

## Preparation

Follow these two documents before you use the tool: 
Install the msModelSlim tool. For details, see [msModelSlim Installation Guide](../getting_started/install_guide.md). 
Install the dependencies for the large-model quantization tool. See [Preparations](../feature_guide/traditional_quantization_v0/foundation_model_compression.md#preparations).

## Code Example

Example of W8A16 quantization and pseudo-quantization accuracy testing on NPU:

```python
import os
import json
import torch
import torch_npu # Use this if you need to quantize on NPU.
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from precision_tool.precision_tool import PrecisionTest # `precision_tool` is used for pseudo-quantization accuracy testing.

SEQ_LEN_OUT = 100
device_id = 0
batch_size = 5

# If the NPU is used for quantization, enable binary compilation to avoid online operator compilation.
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

"""
2. Import the model.
"""
fp16_path = '/data/chatglm2-6b' # Path to the floating-point model

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
Evaluate the floating-point model on the dataset. This example uses boolq.
"""
precision_test = PrecisionTest(model, tokenizer, "boolq", batch_size, "npu")
precision_test.test()

"""
3. Obtain calibration data.
"""
# Data usually lives on CPU. When you quantize on NPU, you need to move it to the NPU device.
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

entry = "/path/to/calib_dataset" # In this example, calibration data is taken from precision_tool/dataset/boolq/dev.jsonl.
calib_set = []
i = 0
with open(entry, encoding="utf-8") as file:
    for line in file:
        data = json.loads(line) # Convert the string to a dictionary.
        while i < 50: # Get 50 calibration samples.
            calib_set.append(data)
            i += 1

dataset_calib = get_calib_dataset(tokenizer, calib_set)


"""
4. Add AntiOutlier outlier suppression for W8A16.
"""
anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", dev_id=device_id)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

"""
5. Set the rollback layer.
"""
"""
Some quantized layers have too much impact on accuracy, so they need to use floating-point weights for computation. disable_names contains the layers that need rollback.
"""
disable_names = []
disable_names.append('lm_head')

"""
6. Run PTQ quantization calibration and store quantization parameters for deployment.
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

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  # disable_level: automatically rolls back n linear layers.
calibrator.run()  # Run PTQ quantization calibration.
calibrator.save('/save/path', save_type=["safe_tensor", "numpy"]) # safe_tensor indicates safetensors weights, numpy indicates .npy weights.


"""
Evaluate the pseudo-quantized model on the dataset. This example uses boolq.
"""
precision_test = PrecisionTest(model, tokenizer, "boolq", batch_size, "npu")
precision_test.test()

"""
7. Optional. Run one round of pseudo-quantized inference for validation.
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

After calling the `Calibrator.run()` method, the tool replaces the `model` passed during `Calibrator` construction with the pseudo-quantization model. You can directly call this model to perform forward inference and test the chat result. If the pseudo-quantized result is not ideal, you can tune it with the following methods.

## Accuracy Tuning Steps for W8A16 Quantized Models

### 1. Outlier Suppression Adjustment

 msModelSlim uses `AntiOutlierConfig` to generate outlier suppression settings. The principle is to suppress outliers during quantization to improve accuracy. For W8A16 quantization, `m3` is recommended. `m3` is the AWQ algorithm, which is also a weight-only quantization algorithm. You can think of it as suppressing outliers in the weights, but the logic is the same as activation outlier suppression. The setting method is as follows:

```python
anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", dev_id=device_id)  
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()
```

Additional note: `anti_method` has six algorithms. `m3` is used for weight quantization. The other five are used for activation quantization scenarios. If you are tuning activation accuracy, try them from `m1` to `m6` one by one.

### 2. Quantization Parameter Selection

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

In the W8A16 scenario, most models can achieve a 0.5% accuracy loss with the simplest MinMax algorithm. Tuning path: MinMax -> HQQ -> GPTQ (slow and not recommended unless necessary).
MinMax maps floating-point values directly to the int8 range.
HQQ can quickly quantize LLMs without training data, and its compression quality is comparable to training-based methods.
GPTQ is an efficient post-training quantization algorithm for large pretrained models.
For Weight quantization, symmetric quantization (`w_sym=True`) and per_channel quantization (`mm_tensor=False`) are commonly used.
Quantization rollback should be the last step. You can set them manually with `disable_names` based on experience.

### 3. Calibration Set Adjustment

1. When algorithm changes do not improve accuracy, increase the calibration set size to 10 to 50 samples. <br>Normally, increasing the data volume improves accuracy, but beyond a certain threshold, the accuracy gains become limited. In some cases, reducing the data volume can actually improve accuracy (such as in long-sequence scenarios).
2. Switch to data from practical application scenarios as the calibration set for specific use cases. <br>Consider the actual inference scenario during selection. For example, use Chinese input for Chinese models, English input for English models, code-generation tasks for code-generation models,
and a mixed Chinese-English calibration set for bilingual models. 
3. Eliminate data with significant output differences between the original and quantized models when constructing the calibration dataset. 
4. Pay attention to the calibration set format. <br>In the following example, `get_calib_dataset` adjusts the calibration set format. Using boolq as the example, the boolq dataset format is `dict={"question":str, "title":str, "answer":bool, "passage":str}`, while the tokenizer expects `"str"` for a single prompt, `"List[str]"` for a batch or a single prompt, or `"List[List[str]]"` for batched prompts.

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

Note: [Precision Tool usage and dataset download links](../feature_guide/traditional_quantization_v0/fake_quantization_accuracy_testing_tool.md) 

### 4. Quantization Rollback

Reason for quantization rollback: Some network layers are sensitive to quantization, which can cause significant accuracy loss after quantization. These layers are not suitable for quantization and should use floating-point computation instead. This process is called rollback. The rolled-back layers are all linear layers, and you can use `disable_names` to control which layers should be rolled back.
Why only linear layers are rolled back: LLMs contain many linear layers, with a large number of weights and matrix multiplication operations that involve heavy computation. Quantizing the weights and activations of linear layers can reduce the model size, computation, and memory usage, and improve inference speed. 
How to determine sensitivity: The terminal logs show the `range_parm` value for the activation quantization input of each operator. A larger `range_parm` indicates higher sensitivity. 

Weight-only quantization workflow:

1. Determine which layers need rollback by checking `range_parm` in the log. The larger the value, the more likely the layer should be rolled back. For example, `down_proj` and `o_proj`.
2. Operation method: Set parameters in QuantConfig, and set disable_names to the quantization sensitive layer that needs to be manually rolled back.

In addition, W8A16 supports manual rollback only. It does not support automatic rollback (`disable_level='L0'`).
Note: Quantization rollback reduces performance to some extent.

The following example manually rolls back all `down_proj` layers in ChatGLM2-6B:

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

### 5. KV Cache INT8 Quantization

You can call `kv_quant` after `QuantConfig` to enable KV cache INT8 quantization.

```python
quant_config = QuantConfig(
    a_bit=16,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id
).kv_quant()

```

In long-sequence scenarios, KV cache takes up a large amount of memory. KV cache quantization can save memory and increase concurrency. 
Calling `kv_quant` automatically sets `use_kvcache_quant` in `QuantConfig` to `True`. 
`use_kvcache_quant=True` enables KV Cache quantization and it can be used together with W8A8, W8A16, and sparse quantization.
