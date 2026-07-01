# Qwen3-Omni Quantification Instructions

## Model Introduction

Qwen3-Omni is a multi-modal Omni model developed by the Alibaba Cloud Qwen team. It supports multi-modal understanding and generation of voice, image, video, and text. Currently, W8A8 quantization is supported for the following specifications:

 * **Qwen3-Omni-30B-A3B-Thinking: chain-thinking capability, 30B general parameter, 3B activation parameter MoE specifications**
 * **Qwen3-Omni-30B-A3B-Instruct: command fine tuning version, 30B general parameter, 3B activation parameter MoE specifications**

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/zh/getting_started/install_guide.md)    .
 * Note: Due to the particularity of transformers of later versions, PyTorch and torch_npu must be configured as compatible versions according to the installation guide.
 * For Qwen3-Omni, the transformers version must be 4.57. 3.
    
    ```bash
    pip install transformers==4.57.3
    ```

 * To be installed`qwen_omni_utils`(for multimodal data preprocessing):
    
    ```bash
    pip install qwen_omni_utils
    ```

 * You need to install the FFmpeg (for audio and video preprocessing) in the environment.
    
    ```bash
    #Ubuntu
      sudo apt-get update && sudo apt install -y ffmpeg
    
      #CentOS
      sudo yum install -y ffmpeg
    
      #Verifying the FFmpeg Installation
      ffmpeg -version
    ```

## Qwen3-Omni model currently validated quantization methods

| model                       | Raw floating-point weight                                                               | Quantization mode   | Inference Framework Support | Quantization command                     |
| --------------------------- | --------------------------------------------------------------------------------------- | ------------------- | --------------------------- | ---------------------------------------- |
| Qwen3-Omni-30B-A3B-Thinking | [Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)     | W8A8 Quantification | vLLM Ascend                 | [W8A8 Quantification](#qwen3-omni-30b-a3b-thinking--qwen3-omni-30b-a3b-instruct-w8a8-quantization)     |
| Qwen3-Omni-30B-A3B-Instruct | [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)     | W8A8 Quantification | vLLM Ascend                 | [W8A8 Quantification](#qwen3-omni-30b-a3b-thinking--qwen3-omni-30b-a3b-instruct-w8a8-quantization)     |

**Note: Click the link in the Quantization Command column to go to the specific quantization command.**

## Use Example

### Qwen3-Omni-30B-A3B-Thinking / Qwen3-Omni-30B-A3B-Instruct W8A8 Quantization

Quantification of this series of models has been integrated into[One-click quantization](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#参数说明)    . Will`--model_type`Set to the corresponding model name.`--quant_type`Set to`w8a8`That's it.

**Qwen3-Omni-30B-A3B-Thinking:**

```shell
msmodelslim quant \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --device npu \
    --model_type Qwen3-Omni-30B-A3B-Thinking \
    --quant_type w8a8 \
    --trust_remote_code True
```

**Qwen3-Omni-30B-A3B-Instruct:**

```shell
msmodelslim quant \
    --model_path ${model_path}  \
    --save_path ${save_path} \
    --device npu \
    --model_type Qwen3-Omni-30B-A3B-Instruct \
    --quant_type w8a8 \
    --trust_remote_code True
```

## Appendixes

### Related Resources

 * [One-Click Quantization Configuration Protocol Description](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#量化配置协议详解)    
 * [Configuration Description of the multimodal_vlm_modelslim_v1 Quantification Service](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#multimodal_vlm_modelslim_v1-配置详解)    
