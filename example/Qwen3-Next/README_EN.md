# Qwen3-Next Quantification Description

## Model Introduction

 * [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)    It represents the next generation of the foundation model introduced by the Qwen team, optimized for extreme context lengths and large-scale parameter efficiency. It introduces a number of architectural innovations that maximize performance while minimizing computational costs.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../docs/zh/getting_started/install_guide.md)    .
 * Requires transformers 4.57.0 or later:
    
     * pip install transformers==4.57.1

## Ascend AI processor support

 * Supports Atlas A2 training and inference products and Atlas A3 training and inference products.

## Supported Model Versions and Quantification Policies

| Model Series   | Model version               | HuggingFace Link                                                                        | W8A8 | W8A16 | W4A8 | W4A16 | W4A4 | sparse quantization | KV Cache | Attention | Quantization command                  |
| -------------- | --------------------------- | --------------------------------------------------------------------------------------- | ---- | ----- | ---- | ----- | ---- | ------------------- | -------- | --------- | ------------------------------------- |
| **Qwen3-Next** | Qwen3-Next-80B-A3B-Instruct | [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)     | ✅    |       |      |       |      |                     |          |           | [W8A8](#qwen3-next-80b-a3b-instruct)     |

**Description:**

 * " indicates that the quantification policy has been officially verified by msModelSlim and has complete functions and stable performance. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the Quantization Command column to go to the specific quantization command.

## Quantized weight generation

### Use Example

 * Please put the`${MODEL_PATH}`Replace with the actual floating-point weight path of the user,`${SAVE_PATH}`Replace with the path for storing the quantization weight.

#### Qwen3-Next-80B-A3B-Instruct

##### Qwen3-Next-80B-A3B-Instruct W8A8 Quantization

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3-Next-80B-A3B-Instruct --quant_type w8a8 --trust_remote_code True
```
