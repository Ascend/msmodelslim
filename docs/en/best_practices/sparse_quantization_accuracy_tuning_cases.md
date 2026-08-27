# Sparse Quantization Accuracy Tuning Case

## Overview

This document provides sample code for sparse quantization and describes the tuning strategy for sparse quantization.

## Preparation

The code example uses the `precision_tool` tool. You can refer to [Precision Test Tool](../user_guide/feature_guide/traditional_quantization_v0/fake_quantization_accuracy_testing_tool.md) for setup.

Install msModelSlim. For details, see [msModelSlim Installation Guide](../install_guide/install_guide.md).

## Code Example

```python
import json
import torch
import argparse
import torch_npu # Use this if you need to quantize on NPU.
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from precision_tool.precision_tool import PrecisionTest # `precision_tool` is used for pseudo-quantization accuracy testing.

def parse_args():
    parser = argparse.ArgumentParser(description="Sparse quant demo")
    parser.add_argument("--model_path", type=str, default="/path/to/model", help="The path to model float weights")
    parser.add_argument("--save_path", type=str, default="./path/to/save", help="The path to save quant weights")
    parser.add_argument("--device", type=str, default="npu:0", help="The device to execute quant process")
    parser.add_argument("--calib_dataset_path", type=str, default="/path/to/dataset", help="The path to calibrate dataset, eg. boolq")
    parser.add_argument("--calib_dataset_count", type=int, default=50, help="The count of data to do calibration")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size when run precision tool")
    parser.add_argument("--fraction", type=float, default=0.01, help="Fraction to control sparse ratio")
    parser.add_argument("--do_smooth", type=bool, default=True, help="Enable the antioutlier for lowbit sparse quant mode")
    parser.add_argument("--co_sparse", type=bool, default=False, help="Enable the co_sparse mode sparse quant mode")
    parser.add_argument("--is_lowbit", type=bool, default=True, help="Enable the lowbit sparse quant mode")
    parser.add_argument("--use_sigma", type=bool, default=False, help="Enable sigma antioutlier protection in the lowbit sparse quant mode")
    return parser.parse_args()

def test_generate_oneshot(tokenizer, model):
    test_prompt = "Where is the capital of China?"
    test_input = tokenizer(test_prompt, return_tensors="pt")
    print("model is inferring...")
    model.eval()
    generate_ids = model.generate(
        test_input.input_ids.to(f"npu:{model.device.index}"),
        attention_mask=test_input.attention_mask.to(f"npu:{model.device.index}"),
        max_new_tokens=SEQ_LEN_OUT
    )
    out_str = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(out_str)


args = parse_args()

SEQ_LEN_OUT = 100

# If the NPU is used for quantization, enable binary compilation to avoid online operator compilation.
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

"""
1. Import the model.
"""
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=args.model_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map=args.device)

"""
2. Optional. Evaluate the floating-point model on the dataset. This example uses boolq.
"""
# precision_test = PrecisionTest(model, tokenizer, "boolq", args.batch_size, "npu")
# precision_test.test()

print("testing float weights...")
test_generate_oneshot(tokenizer, model)

"""
3. Obtain calibration data.
"""
def build_prompt(title, text, passage):
    prompt = f"{title} -- {passage}\nQuestion:{text}?\nAnswer:"
    return prompt

# The generated input must be on the same device as the model, or an error will be raised.
def get_calib_dataset(tokenizer, calib_list, device):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append([inputs.data['input_ids'].to(device), inputs.data['attention_mask'].to(device)])
    return calib_dataset

calib_set = []  # Start with an empty list, then read and append calibration data from the calibration set file.
# Generate calibration data from the calibration set folder.
with open(args.calib_dataset_path, encoding="utf-8") as file:
    i = 0
    for line in file:
        if i == args.calib_dataset_count:
            break
        data = json.loads(line) # Convert the string to a dictionary.
        calib_set.append(build_prompt(data["title"], data["question"], data["passage"]))
        i += 1

dataset_calib = get_calib_dataset(tokenizer, calib_set, device=args.device)

"""
4. Set the rollback layer.
"""
"""
Some quantized layers have too much impact on accuracy, so they need to use floating-point weights for computation. disable_names contains the layers that need rollback.
"""
disable_names = [f"model.layers.{i}.mlp.down_proj" for i in range(model.config.num_hidden_layers)]

"""
5. Run PTQ quantization calibration and store quantization parameters for deployment.
"""
quant_config = QuantConfig(
    a_bit=8,
    w_bit=4,                    # w_bit=4 and a_bit=8 enable sparse quantization.
    disable_names=disable_names,# Rollback layer configuration
    dev_type='npu',             # Device used for quantization
    dev_id=model.device.index,  # Device ID used for quantization. Required when you use NPU or CUDA.
    is_lowbit=args.is_lowbit,   # Enables low-bit mode.
    co_sparse=args.co_sparse,   # Enables co_sparse mode.
    fraction=args.fraction,     # Controls the sparsity ratio.
    do_smooth=args.do_smooth,   # If it is set to True, outlier suppression in low-bit mode is enabled.
    use_sigma=args.use_sigma,   # If it is set to True, Gaussian-based outlier protection in low-bit mode is enabled.
)

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  # disable_level: automatically rolls back n linear layers.
calibrator.run()  # Run PTQ quantization calibration.
calibrator.save(args.save_path, save_type=["safe_tensor"]) # safe_tensor means safetensors weights.

"""
6. Optional. Run one round of pseudo-quantized inference. A quick chat test can help you check whether there are obvious accuracy problems, such as garbled output.
"""
print("testing quantized weights...")
test_generate_oneshot(tokenizer, model)

"""
7. Optional. Evaluate the pseudo-quantized model on the dataset. This example uses boolq. Running dataset evaluation on the pseudo-quantized model is slow.
"""
precision_test = PrecisionTest(model, tokenizer, "boolq", args.batch_size, "npu")
precision_test.test()
```

# msModelSlim Sparse Quantization Accuracy Tuning Guide

## Basic Accuracy Tuning Strategy

Sparse quantization uses **the same basic accuracy tuning strategy** as the other quantization modes provided by msModelSlim: **adjust the calibration set**, **apply outlier suppression**, and **add quantization rollback**.

### Adjusting the Calibration Set

You can adjust the calibration set from two dimensions: **quantity** and **quality**.

**Calibration set quality means the data should adequately represent the dataset the quantized model will face.** You are advised to use test data as calibration data during quantization. For example, if the target dataset is boolq, you can randomly sample some examples as the calibration set. If there are multiple test datasets, you can sample several examples from each one and combine them into the calibration set.

**Increasing the number of calibration samples can improve calibration accuracy to some extent,** but the improvement is limited. Once the dataset reaches a certain size, additional samples no longer help much. You are advised to use 20 to 50 samples as the calibration set.

### Applying Outlier Suppression

Outlier suppression can improve quantization accuracy. In sparse quantization, there are currently two outlier suppression modes.

#### Low-bit Outlier Suppression

In low-bit mode (see [Sparse mode](#sparse-mode)), you can enable the built-in outlier suppression method by setting `do_smooth=True` in `QuantConfig`. **This method includes an automatic tuning mechanism, so it usually works slightly better than the AntiOutlier module. You are advised to use this method first.**

#### AntiOutlier

You can use the independent AntiOutlier module to suppress outliers in the model.

**For sparse quantization, the main difficulty lies in the weights, so it is recommended that you use the AWQ outlier suppression algorithm.** This algorithm works better for weight quantization. You can set `anti_method=m3` in `AntiOutlierConfig`.

### Adding Quantization Rollback

Some layers can cause a large accuracy drop after quantization. For such layers, **you can choose not to quantize them. This is called quantization rollback.**

#### Manual Rollback

The choice of layers to roll back usually depends on experience. For example, in Llama-series models, the down layers in the MLP are often rolled back. You can control the layers to roll back manually by setting `disable_names` in `QuantConfig`.

#### Automatic Rollback

The quantization tool also provides automatic rollback. You can set `disable_level` in `Calibrator` to control how many layers are rolled back automatically. For example, if you set `disable_level='L5'`, the tool rolls back at most 5 layers. It may roll back fewer than 5 layers depending on the model and input, and it does not include layers specified by `disable_names`.

Because there is currently no effective algorithm for determining quantization loss, this feature cannot guarantee a positive result. You should tune it carefully.

## Sparse-Specific Tuning Parameters

In addition to the basic accuracy tuning strategy, sparse quantization has several **tuning parameters that are unique to sparse quantization**.

### Sparse Mode

You can specify `use_sigma=True` in `QuantConfig`.

The tool currently provides two sparse quantization modes, which correspond to two different algorithm backends.

**Note that the two sparse modes are mutually exclusive, so you can choose only one.**

#### Low-bit Sparse Mode

You can set `is_lowbit=True` in `QuantConfig` to use low-bit sparse mode.

#### co_sparse Sparse Mode

You can set `co_sparse=True` in `QuantConfig` to use co_sparse sparse mode.

#### How to Choose

If you want to use outlier suppression, it is recommended that you prefer low-bit sparse mode. The built-in [low-bit outlier suppression](#low-bit-outlier-suppression) in this mode includes automatic tuning and usually works slightly better than co_sparse under the same configuration.

### Adjusting the Sparsity Ratio

You can tune accuracy by adjusting the sparsity ratio.

For sparse quantization, the usual rule is: **the higher the sparsity ratio, the lower the accuracy, but the better the performance. Conversely, the lower the sparsity ratio, the higher the accuracy, but the worse the performance.**

The tool currently provides two ways to adjust the sparsity ratio.

**These two methods are mutually exclusive, so you need to choose one.**

#### fraction

You can control the sparsity ratio by changing the `fraction` parameter in `QuantConfig`.

This parameter has a negative linear relationship with the sparsity ratio itself.

**Therefore, increasing `fraction` lowers the sparsity ratio, which improves model accuracy but reduces model performance. The same applies in reverse.**

**It is recommended that you tune this parameter in the 0.01 to 0.1 range.**

#### use_sigma + sigma_factor

You can enable Gaussian-distribution-based sparsity ratio control by setting `use_sigma=True` in `QuantConfig`. **This parameter is supported only in low-bit mode.**

After this is enabled, the sparsity ratio is no longer fixed. Instead, it depends on the distribution of the weights to be quantized. You can adjust the sparsity ratio by tuning `sigma_factor` in `QuantConfig`. **The smaller the `sigma_factor`, the lower the sparsity ratio and the higher the model accuracy. The same applies in reverse.**
