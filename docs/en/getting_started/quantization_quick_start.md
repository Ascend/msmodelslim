# Quantization Quickstart

## Overview

msModelSlim provides two quantization modes: **quick quantization (V1)** and **traditional quantization (V0)**.

- **Quick quantization (V1)**: Targeted at users with no experience, this mode allows users to quickly complete quantization by using commands and features out-of-the-box characteristics. The system automatically matches the configuration of best practices. Users only need to specify the necessary parameters. This mode also supports customized, refined, and hybrid quantization strategies and offers high flexibility.
- **Traditional quantization (V0)**: This mode executes quantization by using Python scripts. It is less generalized and readable than quick quantization and development has ceased. This mode is typically used for models unsupported by quick quantization.

This topic uses Qwen2.5-7B-Instruct as an example to describe how to complete quantization and perform inference by using the vLLM Ascend plugin (vllm-ascend).

## Environment Setup

### 1. Image Preparation

vllm-ascend provides Docker images for deployment. Pull the pre-built image from the [ascend/vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags) image repository. For details, see [vllm-ascend Quickstart](https://docs.vllm.ai/projects/ascend/en/latest/).

### 2. Installing msModelSlim in the Image

For details about the installation command, see [msModelSlim Installation Guide](./install_guide.md).

### 3. Downloading Original Floating-point Weights of the Model

For example, you can obtain the original model weights from [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

### 4. Installing Other Dependencies (Model-related. For details, see the Hugging Face Model Cards.)

```shell
pip install transformers==4.43.1
```

### 5. Preparing Calibration Data

Traditional quantization mode requires a calibration data file (`.jsonl` format) for calibration during the quantization process. Example data files, such as `boolq.jsonl` and `teacher_qualification.jsonl`, are located in the `example/common/` directory.

## Quantization Mode Selection

### Method 1: Quick quantization (Recommended)

Quick quantization is started by using commands. The system automatically matches the configuration of the best practices.

#### Syntax

```bash
msmodelslim quant [ARGS]
```

#### Parameters

For details about the parameters, see [Quick Quantization Parameters](../feature_guide/quick_quantization_v1/usage.md#parameters).

**Notes**:

- The configuration files in the best practices library are stored in `msmodelslim/lab_practice`.
- If no optimal configuration matching the conditions is found in the best practices library, the system will recommend other available configurations based on preset rules and ask you whether to use the recommended configuration to continue quantization.
- To print quantization run logs, you can set the environment variable `MSMODELSLIM_LOG_LEVEL` to `INFO` (default) or `DEBUG`.

#### Example

Quantize the Qwen2.5-7B-Instruct model in w8a8 mode by using the quick quantization feature:

```bash
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu:0,1 --model_type Qwen2.5-7B-Instruct --quant_type w8a8 --trust_remote_code True
```

where

- `${MODEL_PATH}` specifies the path of the original floating-point weights of Qwen2.5-7B-Instruct.
- `${SAVE_PATH}` specifies the user-defined path for saving the quantized weights.

### Method 2: Traditional quantization (Python scripts)

You can perform traditional quantization by using Python scripts.

#### Syntax

Each model has its corresponding quantization script. This topic uses the Qwen model as an example.

```bash
python3 example/Qwen/quant_qwen.py [ARGS]
```

#### Main parameter description

| Parameter| Purpose| Mandatory (Yes/No)| Description|
|---------|------|---------|------|
| `model_path` | Specifies the model path.| Yes| Set this parameter to the path of the original floating-point model weights.|
| `save_directory` | Specifies the save path for quantized weights.| Yes| Set this parameter to the directory for storing quantized weights.|
| `calib_file` | Specifies the calibration data file.| No| Set this parameter to the path of the calibration data file (in .jsonl format). By default, `example/common/teacher_qualification.jsonl` is used.|
| `w_bit` | Specifies weight quantization bits.| No| Default value: `8`. For quantization of foundation models, set this parameter to `8` or `16`. For sparse quantization, set it to `4`.|
| `a_bit` | Specifies activation quantization bits.| No| Default value: `8`. For quantization of foundation models, set this parameter to `8` or `16`. For sparse quantization, set it to `8`.|
| `device_type` | Specifies the device type.| No| Default value: `cpu`. Valid values: `cpu` or `npu`.|
| `act_method` | Specifies the activation quantization method.| No| Default value: `1`. `1` specifies the min-max quantization method, `2` specifies the histogram quantization method, and `3` specifies the automatic mixed precision quantization method (recommended).|
| `anti_method` | Specifies the outlier suppression method.| No| Valid values: `m1` (SmoothQuant), `m2` (enhanced SmoothQuant), `m3` (AWQ), `m4` (smooth optimization), `m5` (CBQ), and `m6` (Flex Smooth).|
| `model_type` | Specifies the model type.| No| For the Qwen model, valid values include `qwen1`, `qwen1.5`, `qwen2`, `qwen2.5`, and `qwen3`. Default value: `qwen2`.|
| `trust_remote_code` | Specifies whether to trust custom code.| No| Default value: `False`.|

**More parameters**: Traditional quantization supports additional advanced parameters, such as `disable_names` (manually fallen-back quantization layers), `fraction` (the proportion of abnormal values in sparse quantization), and `use_kvcache_quant` (KV cache quantization). For details about the parameters, see the `README.md` file in the directory of each model.

#### Examples

**Example 1: W8A8 quantization for Qwen2.5-7B-Instruct**

```bash
cd msmodelslim
python3 example/Qwen/quant_qwen.py \
    --model_path ${MODEL_PATH} \
    --save_directory ${SAVE_PATH} \
    --calib_file example/common/boolq.jsonl \
    --w_bit 8 \
    --a_bit 8 \
    --device_type npu \
    --trust_remote_code True
```

**Notes**:

- Quantization scripts for different models are stored in the corresponding subdirectories of the `example/` directory, such as `example/Qwen/quant_qwen.py` and `example/Llama/quant_llama.py`.
- For details about the quantization parameters and best practices of each model, see the `README.md` file in the directory of the corresponding model.
- To perform multi-rank quantization on the NPU, configure the following environment variables first:

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

## Quantization Output

After quantization is complete, the following files are generated in the save directory:

```yaml
├── config.json                          # Original model configuration file
├── generation_config.json               # Original generation configuration file 
├── quant_model_description.json         # Description file for quantized weights
├── quant_model_weight_w8a8.safetensors  # Quantized weight file
├── tokenizer_config.json                # Original tokenizer configuration file
├── tokenizer.json                        # Original tokenizer vocabulary
├── {model_type}_best_practice.yaml       # Quantization configuration protocol file
└── vocab.json                            # Original vocabulary mapping file (for some models)
```

**File description**:

- `quant_model_description.json` (or `quant_model_description_{quant_type}.json`) contains quantization parameters and configuration information. It describes the quantization type (such as `W8A8` or `FLOAT`) for each weight.
- `quant_model_weight_{quant_type}.safetensors` is the actual quantized model weight file.
- `{model_type}_best_practice.yaml` records complete configuration information used during quantization. You can use this file to reproduce the quantized weights.
- Other files are configuration and vocabulary files required for model inference. They obtained from the original floating-point directory.

## Using the Quantized Weights

After quantization is complete, you can use the generated quantized weights for inference. The usage method depends on the inference framework:

### 1. Using Quantized Weights in vllm-ascend

You can run the Docker container by referring to the vllm-ascend official tutorial for [Qwen3-32B-W4A4](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-32B-W4A4.html).

#### 1.1 Environment Setup and Model Directory Structure

Assume that you have completed `W8A8` quantization of `Qwen2.5-7B-Instruct` by using `msModelSlim`. The path for saving the quantized weights is as follows:

```bash
SAVE_PATH=/home/models/Qwen2.5-7B-w8a8
```

#### 1.2 Single-Rank Online Service Deployment

To provide an online service on the Ascend device by using vllm-ascend, run the following command:

```bash
vllm serve /home/models/Qwen2.5-7B-w8a8 \
  --served-model-name "Qwen2.5-7B-w8a8" \
  --max-model-len 4096 \
  --quantization ascend
```

Notes:

- The model path `/home/models/Qwen2.5-7B-w8a8` is the directory of the quantized model output by msModelSlim.
- `--quantization ascend` specifies the quantized inference backend adapted for Ascend and loads the weights generated by msModelSlim.
- Other parameters (such as `--max-model-len`) can be adjusted based on the actual service scenario.

After the service is started, you can send an inference request through an HTTP API. For example:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen2.5-7B-w8a8",
        "prompt": "what is large language model?",
        "max_tokens": 128,
        "top_p": 0.95,
        "top_k": 40,
        "temperature": 0.0
      }'
```

where

- The `model` field must be the same as the value of `--served-model-name`.
- During inference, you only need to specify the directory of the quantized model. You do not need to provide the original floating-point weights.

#### 1.3 Single-rank Offline Inference (Python API)

To directly load the quantized model in the Python script for offline inference, you can use the `LLM` API of vllm-ascend.

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is"
]

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=40,
)

llm = LLM(
    model="/home/models/Qwen2.5-7B-w8a8",  # Specifies the msModelSlim quantization output directory.
    max_model_len=4096,
    quantization="ascend",                # Enables Ascend quantization inference.
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Notes:

- `model` still points to the directory where the quantized weights are stored.
- `quantization="ascend"` must be explicitly set to enable the Ascend quantization inference path.
- Other sampling parameters can be adjusted based on the service requirements.

## Miscellaneous

### Supported Models and Quantization Types

You can check the support status of different models in the [foundation model support matrix](../model_support/foundation_model_support_matrix.md).

- Models marked with "quick quantization" in the support matrix support the quick quantization mode.
- All models that have quantization scripts in the `example/` directory support traditional quantization.

### Quantization Suggestions for Foundation Models

For large models (7B or larger), if you encounter insufficient memory issues, try the following methods:

1. **Layer-wise quantization**: This is enabled by default in quick quantization. For details, see [Layer-wise Quantization](../feature_guide/quick_quantization_v1/usage.md#Layer-wise-Quantization- and-Distributed-Layer-wise-Quantization). It is not supported in traditional quantization.
2. **Quantization on the CPU**: Set `--device cpu` (for quick quantization) or `--device_type cpu` (for traditional quantization). This method is slower but requires less memory.

### Supported Quantization Algorithms

For details about the algorithms supported by quick quantization, see [Algorithms Supported by Quick Quantization V1](../quantization_algorithms/README.md).

### FAQ

**Q: What Should I Do If Memory Is Insufficient During Quantization?**

A: You can try the following methods:

1. Use layer-wise quantization, which is enabled by default in quick quantization and is not supported in traditional quantization.
2. Use the CPU for quantization (set `--device cpu` or `--device_type cpu`). This method is slower but requires less memory.

**Q: What Should I Do If the Accuracy of the Quantized Model Decreases Significantly?**

A: You can try the following methods:

1. Use a higher-precision quantization type (for example, change from `w4a8` to `w8a8`).
2. Refer to the best practices configuration of the model in the `msmodelslim/lab_practice` directory.
3. Check whether the outlier suppression algorithm, quantization strategy, and calibration dataset are appropriate. For details, see the [Quantization Accuracy Tuning Guide](../case_studies/quantization_precision_tuning_guide.md).

**Q: How Can I Verify the Quantization Effect?**

A: You can use an inference framework to perform online or offline inference. Compare the output differences before and after quantization using the same input, and compare the differences in the [dataset scores](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/mode.html#id2).

**Q: How Do I Select a Quantization Mode?**

A: Select a quantization mode as follows:

- If the model supports quick quantization, use quick quantization.
- If the model does not support quick quantization, use traditional quantization (which is no longer updated).

**Q: What Is the Difference Between Weight Files Generated by Quick Quantization and Traditional Quantization?**

A: The weight files generated by the two methods use the same format and can both be used for inference. The major differences are as follows:

- Quick quantization uses the best practices configuration, which may contain specific optimizations.
- Traditional quantization supports the generation of an exclusive format for the MindIE inference framework to ensure compatibility with earlier versions. Quick quantization supports the AscendV1 format, which is compatible with multiple frameworks such as MindIE, vllm-ascend, and SGLang. For more information, see the description in [AscendV1Config](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quant_service/modelslim_v1/save/ascendv1.py#L87).
