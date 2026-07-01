# GLM4-MOE Quantification Description

## Model Introduction

[GLM-4.7](https://huggingface.co/zai-org/GLM-4.7)    Smart Spectrum AI is an open-source flagship large language model released on December 23, 2025. It focuses on Agentic Coding (agent programming), complex reasoning, and tool collaboration. It has outstanding performance in coding, long-range tasks, and front-end generation.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/en/getting_started/install_guide.md)    .
 * For the transformers version, the 4.57.3 version must be installed.
    
    ```bash
    pip install transformers==4.57.3
    ```

## Supported Model Versions and Quantification Policies

| Model Series | Model version | HuggingFace Link                       | W8A8 | W8A16 | W4A8 | W4A16 | W4A4 | sparse quantization | KV Cache | Attention | Quantization command    |
| ------------ | ------------- | -------------------------------------- | ---- | ----- | ---- | ----- | ---- | ------------------- | -------- | --------- | ----------------------- |
| **GLM4-MOE** | GLM-4.7       | https://huggingface.co/zai-org/GLM-4.7 | ✅    |       |      |       |      |                     |          |           | [W8A8](#glm-47-one-click-quantization-command-example)     |

**Description:**

 * " indicates that the quantification policy has passed the official verification of msModelSlim. The function is complete and the performance is stable. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the Quantization Command column to go to the specific quantization command.

## One-click quantization weight generation

One-Click Quantization Command Reference[One-Click Quantification User Guide](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/en/feature_guide/quick_quantization_v1/usage.md)    .

### GLM-4.7 One-Click Quantization Command Example

```python
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu:0,1,2,3,4,5,6,7 \
  --model_type GLM-4.7 \
  --quant_type w8a8 \
  --trust_remote_code True
```

 * of which`MODEL_PATH`is the path of the GLM-4.7 model,`SAVE_PATH`is the path for storing the quantized weight.
 * The quantization configuration file used by the one-click quantization command is[glm4_7_moe-w8a8-v1.yaml](../../lab_practice/glm4_moe/glm4_7_moe-w8a8-v1.yaml)    , you can view the specific quantization policy.
