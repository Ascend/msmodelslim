# Qwen3.5 Quantification Description

## Model Introduction

**Qwen3.5 is the latest flagship multi-modal model of Qwen series. It adopts the Mixture of Experts (MoE) architecture, which significantly reduces inference costs while maintaining strong model capabilities. The core architecture features the following: native multi-modal capability (vision encoder + image and text integration), hybrid attention mechanism (conventional attention and linear-attention alternate), MTP multi-token prediction branch, and high-performance MoE expert routing and sharing expert mechanism.**

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../docs/zh/getting_started/install_guide.md)    .
 * For the transformers version, the 5.2. 0 version must be installed.
    
     * pip install transformers==5.2.0

## Ascend AI processor support

 * Atlas A2 training and inference products and Atlas A3 training and inference products are supported.

## Supported Model Versions and Quantification Policies

| Model Series      | Model version     | HuggingFace Link                                                          | W8A8 | W8A16 | W4A8 | W4A16 | W4A4 | sparse quantization | KV Cache | Attention | Quantization command                                              |
| ----------------- | ----------------- | ------------------------------------------------------------------------- | ---- | ----- | ---- | ----- | ---- | ------------------- | -------- | --------- | ----------------------------------------------------------------- |
| **Qwen3.5-MoE**   | Qwen3.5-397B-A17B | [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)           | ✅    |       | ✅    |       |      |                     |          |           | [W8A8](#qwen35-397b-a17b-w8a8-quantization)    /[W4A8](#qwen35-397b-a17b-w4a8-quantization)     |
| **Qwen3.5-MoE**   | Qwen3.5-122B-A10B | [Qwen3.5-122B-A10B](https://modelscope.cn/models/Qwen/Qwen3.5-122B-A10B)     | ✅    |       |      |       |      |                     |          |           | [W8A8](#qwen35-122b-a10b-w8a8-quantization)                                      |
| **Qwen3.5-MoE**   | Qwen3.5-35B-A3B   | [Qwen3.5-35B-A3B](https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B)         | ✅    |       |      |       |      |                     |          |           | [W8A8](#qwen35-35b-a3b-w8a8-quantification)                                        |
| **Qwen3.5-Dense** | Qwen3.5-27B       | [Qwen3.5-27B](https://modelscope.cn/models/Qwen/Qwen3.5-27B)                 | ✅    |       |      |       |      |                     |          |           | [W8A8](#qwen35-27b-w8a8-quantification)                                            |

**Description:**

 * " indicates that the quantification policy has passed the official verification of msModelSlim. The function is complete and the performance is stable. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the Quantization Command column to go to the corresponding quantization command.

## Quantized weight generation

### Use Example

 * Replace \{MODEL_PATH\} with the actual floating-point weight path and \{SAVE_PATH\} with the path for storing the quantization weight.

#### 1. Qwen3.5-397B-A17B

##### Qwen3.5-397B-A17B W8A8 Quantization

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-397B-A17B --quant_type w8a8 --trust_remote_code True
```

##### Qwen3.5-397B-A17B W4A8 Quantization

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-397B-A17B --quant_type w4a8 --trust_remote_code True
```

#### 2. Qwen3.5-122B-A10B

##### Qwen3.5-122B-A10B W8A8 Quantization

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-122B-A10B --quant_type w8a8 --trust_remote_code True
```

#### 3. Qwen3.5-35B-A3B

##### Qwen3.5-35B-A3B W8A8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-35B-A3B --quant_type w8a8 --trust_remote_code True
```

#### 4. Qwen3.5-27B

##### Qwen3.5-27B W8A8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shll
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-27B --quant_type w8a8 --trust_remote_code True
```
