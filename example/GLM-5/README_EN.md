# GLM5-MOE Quantification Description

## Model Introduction

[GLM-5](https://huggingface.co/zai-org/GLM-5)    It is an open-source flagship large language model released by AI on February 1, 2026. Compared with GLM-4.5, GLM-5 expands the parameter scale from 355B to 744B. The amount of pre-training data has also increased from 23 trillion tokens to 28.5 trillion tokens. GLM-5 also integrates the DeepSeek Sparse Attention Mechanism (DSA) to significantly reduce deployment costs while maintaining long-term context processing capabilities.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/en/getting_started/install_guide.md)    .
 * The transformers version needs to be configured and installed with the 5.2. 0 version.
    
    ```bash
    pip install transformers==5.2.0
    ```

## Supported Model Versions and Quantification Policies

| Model Series | Model version | HuggingFace Link                     | W8A8 | W8A16 | W4A8 | W4A16 | W4A4 | sparse quantization | KV Cache | Attention | Quantization command |
| ------------ | ------------- | ------------------------------------ | ---- | ----- | ---- | ----- | ---- | ------------------- | -------- | --------- | -------------------- |
| **GLM5-MOE** | GLM-5         | https://huggingface.co/zai-org/GLM-5 | ✅    |       | ✅    |       |      |                     |          |           |                      |

**Description:**

 * " indicates that the quantification policy has passed the official verification of msModelSlim. The function is complete and the performance is stable. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the Quantization Command column to go to the specific quantization command.

## One-click quantization weight generation

One-Click Quantization Command Reference[One-Click Quantification User Guide](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/en/feature_guide/quick_quantization_v1/usage.md)    .

### GLM-5 One-Click Quantization Command Example

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu:0 \
  --model_type GLM-5 \
  --quant_type w8a8 \
  --trust_remote_code True
```

 * of which`MODEL_PATH`is the path of the GLM-5 model,`SAVE_PATH`is the path for storing the quantized weight.
 * The quantization configuration file used by the one-click quantization command is[glm_5_w8a8.yaml](../../lab_practice/glm_5/glm_5_w8a8.yaml)    , you can view the specific quantization policy.
