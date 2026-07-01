# W8A8 Quantization Accuracy Tuning Strategy

## Overview

The W8A8 quantization accuracy tuning strategy is implemented by using the msModelSlim quantization tool and precision tool for accuracy verification and tuning. LLMs can lose a lot of accuracy after W8A8 quantization, so you can use this strategy for tuning.

Notes: 
 
1. Model-specific recommendation: If you use a ChatGLM model such as ChatGLM2-6B, set the thread count manually to improve runtime efficiency. Other models do not need extra thread settings.

2. Transformers version compatibility: The ChatGLM2-6B model depends on Transformers 4.40.2. If you see a Transformers-related error during runtime, try downgrading to 4.40.2 to resolve the compatibility issue.

```python
# Set the thread count.
export OMP_NUM_THREADS=48
```

## Preparation

Follow these two documents before you use the tool: 
Install the msModelSlim tool. For details, see [msModelSlim Installation Guide](../getting_started/install_guide.md). 
Install the dependencies for the large-model quantization tool. See [Preparations](../feature_guide/traditional_quantization_v0/foundation_model_compression.md#preparations).

## Code Example

Example of W8A8 quantization and fake-quantization accuracy testing on NPU:

```python
"""
1. Import dependencies
"""
import os
import json
import torch
import torch_npu # Use this if you need to quantize on NPU.
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from precision_tool.precision_tool import PrecisionTest # precision_tool is used for pseudo-quantization accuracy testing.

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
4. Add AntiOutlier outlier suppression for W8A8.
"""
anti_config = AntiOutlierConfig(anti_method="m2", dev_type="npu", dev_id=device_id)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()


"""
5. Set the rollback layer.
"""
"""
Some quantized layers have too much impact on accuracy, so they need to use floating-point weights for computation. disable_names contains the layers that need rollback.
"""
disable_names = []


"""
6. Run PTQ quantization calibration and store quantization parameters for deployment.
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

res = tokenizer.batch_decode(
    generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)
for idx, item in enumerate(res):
    print(item)

```

After calling the `Calibrator.run()` method, the tool replaces the `model` passed during `Calibrator` construction with the fake quantization model. You can directly call this model to perform forward inference and test dialogue effects. 
If the fake-quantization result is unsatisfactory, perform the following optimization operations:

## Accuracy Tuning Steps for W8A8 Quantized Models

### 1. Outlier Suppression Adjustment

Reason: By suppressing outliers during quantization, subsequent quantization can be improved.

```python
anti_config = AntiOutlierConfig(
    anti_method="m2",
    dev_type="npu",
    dev_id=device_id
)
```

Optimizable parameter `anti_method`: 

`m1`: SmoothQuant algorithm 
`m2`: upgraded SmoothQuant 
`m3`: AWQ algorithm (applicable to W8A16/W4A16) 
`m4`: optimized SmoothQuant algorithm 
`m5`: CBQ algorithm 
`m6`: Flex smooth quantization algorithm

W8A8 supports `m1`, `m2`, `m4`, `m5`, and `m6`.
You are advised to try `m1` through `m6`, because different models perform differently with different outlier suppression algorithms. Currently, `m2` has been adapted for the Qwen-VL and LLaVA-v1.5-7B multimodal models.

### 2. Quantization Parameter Selection

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

Optimizable parameters `disable_names`, `disable_level`, and `act_method`: 
Add rollback layers. You are advised to adjust this parameter last. You can manually set rollback layers by using `disable_names` based on experience, or use the automatic rollback function of `disable_level` to automatically roll back `Linear` layers that have a relatively large impact on accuracy according to certain criteria.

`disable_names`: specifies rollback layers based on theoretical experience and log information.
`disable_level='L0'`: enables automatic rollback.

`act_method`: specifies the activation quantization method. 
    The default value of `act_method` is `1`. The value of this parameter can be `1`, `2`, or `3`. 
    `1` indicates the min-max quantization method. 
    `2` indicates the histogram quantization method. 
    `3` indicates a hybrid quantization method that combines min-max and histogram. 
    In LLM scenarios, you are advised to use `3`.

### 3. Calibration Set Adjustment

1. When algorithm changes do not improve accuracy, increase the calibration set size to 10 to 50 samples. <br>(Normally, increasing the data volume improves accuracy, but beyond a certain threshold, the accuracy gains become limited. In some cases, reducing the data volume can actually improve accuracy, such as in long-sequence scenarios.) 
2. Switch to data from practical application scenarios as the calibration set for specific use cases. <br>When selecting data, consider the specific inference scenarios where the model will be deployed. For example, Chinese models require Chinese inputs as the calibration dataset, English models require English inputs, and code-generation models require code-generation tasks. Models supporting both languages should use a mixed Chinese-English calibration set. 
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

Note: You need to copy the [precision_tool folder](https://gitcode.com/Ascend/msmodelslim/tree/master/precision_tool) and [security folder](https://gitcode.com/Ascend/msmodelslim/tree/master/security) from the `msmodelslim` directory, place them in the same directory as the quantization script, and then put the test dataset into the `precision_tool` folder. For details, see [Precision Tool usage and dataset download links](../feature_guide/traditional_quantization_v0/fake_quantization_accuracy_testing_tool.md).

### 4. Quantization Rollback

Reasons for LLM quantization: Model quantization can reduce the model size, computing workload, and memory usage, thereby improving inference speed.

Why only linear layers are rolled back: LLMs contain many linear layers, with a large number of weights and matrix multiplication operations that involve heavy computation. Quantizing the weights and activations of linear layers can reduce the model size, computation, and memory usage, and improve inference speed.

Reason for quantization rollback: Some linear layers are sensitive to quantization, which can cause certain accuracy loss after quantization. These layers are not suitable for quantization and should use floating-point computation instead. This process is called rollback. You can use `disable_names` to control which linear layers should be rolled back.

How to determine sensitivity: The terminal logs show the `range_parm` value for the activation quantization input of each operator. A larger `range_parm` indicates higher sensitivity.  
Example terminal log entry:

```python
timestamp - msmodelslim-logger - INFO - use histogram observer:transformer.encoder.layers.27.mlp.dense_h_to_4h.quant_input, range_parm:41.21875
```

In this example, `range_parm:41.21875` is quite large compared with other layers in the log, which means the layer is sensitive and should be rolled back.

Quantization rollback approaches: Two approaches are available: manual rollback and automatic rollback (which can be used together). You are advised to start with manual rollback. If the rollback layers are unclear or the accuracy is unsatisfactory using manual rollback, automatic rollback can then be applied.

Note: Quantization rollback reduces performance to some extent.

#### Manual Rollback: `disable_names` 

`disable_names=[]` specifies rolled-back layer names. If it is left empty, rollback is not applied to any layer.

Rollback priorities (descending order):

1. Roll back `down_proj` layers first, since they are often accuracy-sensitive. These are the MLP down-sampling layers. If `down_proj` is not marked explicitly, check `out_features`. The smaller value is the down-sampling layer. 
2. Roll back `o_proj` layers next, since they are usually accuracy-sensitive. These are the last linear layers used in self-attention. The order printed in the model is only the initialization order. Check the actual call order in the model code. 
3. Use theoretical experience or the `range_parm` values in the terminal log to identify other sensitive layers and roll them back.

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

#### Automatic Rollback: `disable_level`

Automatic rollback ranks linear layers by their range_parm values (from large to small) and selectively applies rollback to those that have the greatest impact on accuracy.
Set `disable_level='Lx'`, where `x` is the number of linear layers to roll back automatically. The terminal will show the rolled back layer names. `disable_level='L0'` means no rollback. If `x` is larger than the number of model layers, all layers are rolled back and no error is reported.

### 5. KV Cache INT8 Quantization

You can call `kv_quant` after `QuantConfig` to enable KV cache INT8 quantization.

```python
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id
).kv_quant()
```

In long-sequence scenarios, KV cache takes up a large amount of memory. KV cache quantization can save memory and increase concurrency. 
Calling `kv_quant` automatically sets `use_kvcache_quant` in `QuantConfig` to `True`. 
`use_kvcache_quant=True` enables KV Cache quantization and it can be used together with W8A8, W8A16, and sparse quantization.

### 6. FA3 Quantization

[FA quantization](../feature_guide/traditional_quantization_v0/foundation_model_quantization_and_calibration.md#fa3-quantization) 

### 7. Accuracy Changes after Step-By-Step Tuning for ChatGLM2-6B

The accuracy values in the table below are obtained by applying the following steps in order, for reference only.

  | Step                | Parameter Changes                                                                                                                                                               | Accuracy|
  | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
  | Floating-point model accuracy          |     None                    | 0.794   |
  | Add `QuantConfig` quantization parameters.         | `disable_names = []`, `act_method = 3`, `disable_level = "L0"`, `dataset_calib` size is 2.                                                                                                          | 0.519   |
  | Add outlier suppression.            | anti_method = "m2"|  0.497 |
  | Increase boolq calibration data.           | Increase from 2 samples to 50 samples.|  0.505 |
  | Add quantization rollback (manual rollback).          | Manually roll back `mlp.down_proj` in all layers.                         | 0.793   |
  | Add quantization rollback (automatic rollback).         |        `disable_names = []`, `disable_level = "L28"`.                                                                                            | 0.791  |
  | Add quantization rollback (manual and automatic rollback).            | Manually roll back 10 layers in `disable_names`, `disable_level = "L28"`.|  0.795 |

### 8. Inference and Deployment after Model Quantization

After quantization, you can use MindIE, vLLM, and other tools for deployment and inference.
