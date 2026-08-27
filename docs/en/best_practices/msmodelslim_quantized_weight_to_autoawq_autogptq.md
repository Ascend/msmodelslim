# msModelSlim Quantized Weights Conversion Guide for AutoAWQ and AutoGPTQ

## Overview

The weight formats used by msModelSlim differ from those used by the open-source tools AutoAWQ and AutoGPTQ. This guide explains how to convert msModelSlim quantized weights into weights that match the formats used by these tools, so that converted W4A16 weights for `qwen2-7b` can be loaded directly in Hugging Face format.
This guide supports weight conversion only for the following configurations:
W4A16 + per_group + AWQ
W4A16 + per_group + GPTQ
W4A16 + per_channel + GPTQ
W8A16 + per_group + GPTQ
W8A16 + per_channel + GPTQ

Supported platforms:
msModelSlim quantization: NPU
Conversion script: CPU
AutoAWQ: GPU
AutoGPTQ: GPU

## msModelSlim Quantization

### Preparation

Install msModelSlim. For details, see [msModelSlim Installation Guide](../install_guide/install_guide.md).

For dependency installation, see [Preparations](../user_guide/feature_guide/traditional_quantization_v0/foundation_model_compression.md#11-preparations) in the foundation model quantization guide.

### Quantization Instructions

The quantization script is the same as a normal quantization script. Refer to [W8A8 Accuracy Tuning Policy](w8a8_accuracy_tuning_policy.md).
This guide uses W4A16 quantization as an example. Pay attention to the following points:
a. In the outlier suppression configuration (`AntiOutlierConfig`), set `a_bit` and `w_bit` according to the quantization mode. When `anti_method` is set to `"m3"`, the AWQ algorithm is used. For GPTQ, you do not need the outlier suppression module, so you can comment out the related configuration.

```python
anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", a_bit=16, w_bit=4, dev_id=device_id, w_sym=True)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()
```

b. `QuantConfig` configuration
The parameters differ for `per_channel` and `per_group`.
(1) `per_group` requires these parameters: `is_lowbit=True`, `open_outlier=False`, `group_size=128`.
(2) In the `per_channel` scenario, you do not need the following three parameters. Comment them out: `is_lowbit=True`, `open_outlier=False`, `group_size=128`.
(3) If you use AutoGPTQ, change `w_method` to `'GPTQ'`. GPTQ quantization also takes longer to run.
The following is an example `per_group` configuration for AutoAWQ:

```python
quant_config = QuantConfig(
    a_bit=16,                      # Activation quantization bit width
    w_bit=4,                       # Weight quantization bit width
    disable_names=disable_names,   # Names of quantized layers that should be rolled back manually
    mm_tensor=False,               # Default: True for per_tensor quantization, False for per_channel quantization
    dev_type='npu',                # The quantization tool runs on NPU.
    dev_id=0,
    w_sym=True,                    # Symmetric quantization
    w_method='MinMax',             # Weight quantization algorithm configuration
    is_lowbit=True,                # Parameters for the per_group scenario. Comment out these three parameters for per_channel quantization.
    open_outlier=False,
    group_size=128
)
```

c. Saved weight files
This script supports conversion only for unsliced safetensors weights. When you save the quantized weight file, do not use sharded saving.
For reference, see the [`save()` API description](../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_save().md).

```python
calibrator.save(output_path, safetensors_name=None, json_name=None, save_type=None, part_file_size=None)
```

### Conversion Script Instructions

The conversion script is located at [`ms_to_vllm.py`](https://gitcode.com/Ascend/msmodelslim/blob/26.1.0/example/ms_to_vllm.py).

After you quantize the weights with msModelSlim in step 1.1 and generate `quant_model_description_w4a16.json` and `quant_model_weight_w4a16.safetensors`, run `ms_to_vllm.py` to convert the weight format and generate the converted `safetensors` file. Use the command as follows:

```python
Command:
python ms_to_vllm.py --model {weighted_safetensors_path} --json {weighted_json_path} --save_path  {converted_safetensors_path} --w_bit {weight_bit} --target_tool  {target_convert_tool}

Description:
    model: required, string. The quantized safetensors weight file to load. You can pass an absolute or relative path.
    json: required, string. The quantized JSON weight description file to load. You can pass an absolute or relative path.
    save_path: optional, string. Default: ./res.safetensors. save_path supports only saving files in an existing directory. It does not create directories.
    w_bit: optional, int. Valid values: [4, 8]. Default: 4, which means the quantized weight bit width is 4.
    target_tool: optional, string. Valid values: [awq, gptq]. Default: awq, which means the target conversion tool is AutoAWQ.

Example:
Copy the conversion script into the directory that contains the quantized weights, then run the following command in that directory. The converted weight file `res.safetensors` is generated in the same directory:
python ms_to_vllm.py --model ./quant_model_weight_w4a16.safetensors --json ./quant_model_description_w4a16.json --save_path res.safetensors --target_tool awq

```

## Open-Source AutoAWQ Quantization and Inference

### Preparation

For environment setup, quantization, and inference, refer to the `readme.md` on GitHub: [AutoAWQ README](https://github.com/casper-hansen/AutoAWQ).

### Quantization Instructions

AutoAWQ quantization needs special attention. When `Version` uses `GEMM`, an error may occur if you do not pass a dataset. In that case, pass the `val.jsonl` file. See [AutoAWQ issue #506](https://github.com/casper-hansen/AutoAWQ/issues/506).
Dataset download link: [val.jsonl.zst](https://huggingface.co/datasets/mit-han-lab/pile-val-backup/blob/main/val.jsonl.zst). If `trust_remote_code` is set to `True`, code files in the weights directory of the floating-point model may be executed. Ensure that the source of the floating-point model is secure.
Example AutoAWQ quantization script:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

model_path = 'qwen2_7b_instruct'  # Path to the floating-point model weights
quant_path = 'quant_qwen2_7b_awq_4_g128'  # Saving path for the quantized model

# q_group_size corresponds to the group_size used in msModelSlim quantization. Keep them consistent.
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

### Inference Instructions

First, modify the `model.safetensors.index.json` file in the AutoAWQ quantized weight directory. Update the weight file names in `weight_map` to the filename generated by the conversion script in section 1.2, then replace the weight file with the one generated by the conversion script in section 1.2 and run the inference script. If `trust_remote_code` is set to `True`, code files in the weights directory of the floating-point model may be executed. Ensure that the source of the floating-point model is secure.

The following is an example AutoAWQ inference script for a chat test:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = './quant_qwen2_7b'  # Saving path for the quantized model

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

If you do not use AutoAWQ quantization to generate the quantization config file and instead run inference directly with the model converted by msModelSlim, follow these steps:

1. Copy the floating-point model config file into the directory that contains the converted weights generated by msModelSlim quantization.
2. Modify the `model.safetensors.index.json` file in the quantized weight directory. Update the weight file names in `weight_map` to the weight file name generated by the conversion script in section 1.2.
3. Modify the `config.json` file and add the `quantization_config` parameter. Set `bits` to the quantized weight bit width, and keep `group_size` consistent with the `group_size` used by msModelSlim quantization. Refer to the `config.json` generated after AutoAWQ quantization in section 2.2.
    The following example uses Qwen2-7B-Instruct and W4A16 + AWQ. Add `quantization_config` to `config.json` as follows:

    ```json
    {
    "model_type": "qwen2",
    "torch_dtype": "bfloat16",
    ··· Example of the original JSON parameters ···
    ··· Add the following parameters. ···
    "quantization_config": {
        "bits": 4,
        "group_size": 64,
        "version": "gemm",
        "zero_point": true
    }
    }
    ```

4. Run inference by following the script in section 2.3.

## Open-Source AutoGPTQ Quantization and Inference

### Preparation

For environment setup, quantization, and inference, refer to the `readme.md` on GitHub: [AutoGPTQ README](https://github.com/AutoGPTQ/AutoGPTQ).

### Quantization Instructions

Converting msModelSlim weights to the AutoGPTQ format for inference is similar to AutoAWQ. First, read the AutoGPTQ `readme.md` linked in section 3.1, refer to the quantization example, modify the related configuration parameters, and then run quantization to generate the quantized weight file.
The modified configuration includes the path and the `BaseQuantizeConfig` interface. In `BaseQuantizeConfig`, `bits` is the quantized weight bit width and corresponds to `w_bit` in msModelSlim. In the `per_group` scenario, set `group_size` to the same value as msModelSlim. In the `per_channel` scenario, set `group_size` to `-1`.

### Inference Instructions

Put the `res.safetensors` file, after msModelSlim quantization and conversion by the conversion script, into the GPU-generated quantized weight directory. Replace the previous quantized weight file, keep the file name unchanged, and do not modify the other files.
Example inference script:

```python
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_path = "./qwen2_7b_instruct"      # Path to the floating-point model weights
quant_path = "./ms_to_gptq"             # Saving path for the quantized model

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

# Load the unquantized model. By default, the model is loaded into CPU memory.
model = AutoGPTQForCausalLM.from_quantized(quant_path, device="cuda:0")
print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))
```

If you do not use AutoGPTQ quantization to generate the quantization config file and instead directly infer with the model converted by msModelSlim, follow these steps:

1. Copy the floating-point model config files `config.json` and `model.safetensors.index.json` into the directory that contains the converted weights generated by msModelSlim quantization.
2. Modify the `model.safetensors.index.json` file in the quantized weight directory. Update the weight file names in `weight_map` to the weight file name generated by the conversion script in section 1.2.
3. Create a new `quantize_config.json` file in the quantized weight directory. Set `bits` to the quantized weight bit width, and keep `group_size` consistent with the `group_size` used by msModelSlim quantization. Refer to the `quantize_config.json` file generated after AutoGPTQ quantization.
    The following example uses Qwen2-7B-Instruct and W4A16 + GPTQ. Create `quantize_config.json` as follows:

    ```json
    {
    "bits": 4,
    "group_size": 64
    }
    ```

4. Run inference by following the script in section 3.3.

## Conclusion

With the steps above, you can successfully quantize the model with msModelSlim on NPU and then convert the quantized weights so that AutoAWQ and AutoGPTQ can run inference successfully.
