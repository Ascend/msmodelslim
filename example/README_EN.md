# msModelSlim Recommended Practice Set

The msModelSlim recommended practice set provides quantitative practice cases of various large language models, multi-modal understanding models, and multi-modal generation models, helping users quickly get started with model quantization functions.

## Directory structure

### Quantification Description of Large Language Models

 * **[DeepSeek](./DeepSeek/)**    \- Quantification of DeepSeek Series Models
 * **[GLM](./GLM/)**    \- Quantification Description of GLM Series Models
 * **[GPT-NeoX](./GPT-NeoX/)**    \- Quantification of GPT-NeoX Series Models
 * **[HunYuan](./HunYuan/)**    \- Quantification of HunYuan Series Models
 * **[InternLM2](./InternLM2/)**    \- Quantification of InternLM2 Series Models
 * **[Llama](./Llama/)**    \- LLaMA Series Model Quantification Description
 * **[Qwen](./Qwen/)**    \- Qwen Series Model Quantification Description
 * **[Qwen3_5](./Qwen3_5/)**    \- Qwen3.5 Series Model Quantification Description
 * **[Qwen3-MOE](./Qwen3-MOE/)**    \- Qwen3-MOE Series Model Quantification Description
 * **[Qwen3-Next](./Qwen3-Next/)**    \- Qwen3-Next Series Model Quantification Description

### Multimodal Understanding Model Quantification

 * **[multimodal_vlm](./multimodal_vlm/)**    \- Multimodal Understanding Model Quantification Description
    
     * LLaVA Series Models
     * Qwen-VL Series Models
     * InternVL2 Series Models
     * Qwen2-VL Series Model
     * Qwen2.5-VL Series Model
     * Qwen2.5-Omni Series Model
     * Qwen3-VL Series Model
     * Qwen3-VL-MoE Series Model
     * GLM-4.1V Series Models
     * Qwen3-Omni Series Model
     * GLM-4.6V Model

### Multimodal Generation Model Quantification Description

 * **[multimodal_sd](./multimodal_sd/)**    \- Quantification description of the multi-modal generation model
    
     * Stable Diffusion Series Models
     * Flux Series Models
     * HunYuanVideo model
     * OpenSoraPlanV1_2 Series Models
     * Wan2.1 Series Model

### Other functions

 * **[common](./common/)**    \- General Tools and Calibration Data
 * **[osp1_2](./osp1_2/)**    \- Functions related to OpenSora Plan 1.2
 * **[ms_to_vllm.py](./ms_to_vllm.py)**    \- msModelSlim to vLLM format conversion tool

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../docs/zh/getting_started/install_guide.md)    .
 * Different model families may depend on specific versions. For details, see the specific description in each model directory.

### Using the Multicard Quantization Function

**Important Note: The Atlas 300I Duo card only supports single-card, single-chip processor quantization.**

To use NPU multi-card quantization, configure the following environment variables:

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
```

### Deq_scale Conversion to Int64 Postprocessing (deqscale2int64.py)

When the model defaults`torch_dtype`When bf16, the Ascend V1 saver will`deq_scale`Write as float32. If the inference side requires deq_scale in int64 format (adapted to the operator), you can use this script to post-process the saved quantization weights and convert deq_scale in the weights from float32/bf16 to int64. (The format is the same as that used when ascendv1 is saved in non-bf16 format.) The script does not depend on the NPU and supports two weight layouts: single file and fragment.

**Application scenario:**

 * During quantization, BF16 is used to save data by default. Deq_scale in the int64 format is expected to be used on the inference side such as MindIE and vLLM Ascend.
 * Convert the existing Ascend V1 quantization directory at a time without re-quantification.

**Example command:**

```shell
#Go to the sample directory (or change the path to example/deqscale2int64.py in the root directory of the project).
cd example

#In-place conversion (directly overwriting the safetensors file in the original weight directory)
python deqscale2int64.py --model_path {量化权重目录路径}

#Output to New Directory (Retain the original directory and copy the config and description files.)
python deqscale2int64.py --model_path {量化权重目录路径} --output_dir {输出目录路径}

#View only the key to be converted and do not write files.
python deqscale2int64.py --model_path {量化权重目录路径} --dry_run
```

**Parameter description:**

| Parameter name | meanings                   | Default value       | Description                                                                                                                                                            |
| -------------- | -------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path     | Quantified Weights Catalog | Mandatory           | Must contain`quant_model_description.json`And also the`quant_model_weights.safetensors`(single file) or`quant_model_weights.safetensors.index.json`\+ Fragmented file. |
| output_dir     | Output Directory           | Same as model_path. | If the parameter is not specified, the original data is overwritten. If you specify, the complete directory is copied and then converted in the output directory.      |
| dry_run        | Preview only, no write     | False               | Addition`--dry_run`Only the deq_scale keys to be converted and the number of deq_scale keys to be converted are printed. No file is modified.                          |

The script takes precedence over the`quant_model_description.json`W8A8/W8A8_MIX in`.deq_scale`Key identification of the item to be converted; If there is no description or no match is found, enter the key name (including the key name`.deq_scale`) and dtype (float32/bf16) identification. The deq_scale that is already int64 is skipped.
