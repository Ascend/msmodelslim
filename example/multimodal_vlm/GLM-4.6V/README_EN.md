# GLM-4.6V Quantification Description

## Model Introduction

[GLM-4.6V](https://z.ai/blog/glm-4.6v)    It is the latest iterative version of the intelligent spectrum multimodal large language model. GLM-4.6V (106B) is a basic model designed for cloud and high-performance cluster scenarios. The GLM-4.6V extends the context window to 128k lexicals in training and achieves state-of-the-art performance in visual understanding and reasoning in models of similar parameter sizes. It integrates the native function call capability. This effectively bridges the gap between "visual perception" and "executable action" and provides a unified technical foundation for multimodal agents in real-world business scenarios.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/zh/getting_started/install_guide.md)    .
 * For GLM-4.6V, the transformers version must be 5.0.0rc0:
    
    ```bash
    pip install transformers==5.0.0rc0
    ```

## GLM-4.6V Model Current Validated Quantification Method

| model    | Raw floating-point weight                            | Quantization mode                                              | Inference Framework Support                             | Quantization command                               |
| -------- | ---------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------- |
| GLM-4.6V | [GLM-4.6V](https://huggingface.co/zai-org/GLM-4.6V)     | W8A8 Hybrid Quantification (MoE Expert Dynamic Quantification) | MindIE to be supported. vLLM Ascend is being supported. | [W8A8 Hybrid Quantification](#glm-46v-w8a8-mixed-quantization)     |

**Note: Click the link in the Quantization Command column to go to the specific quantization command.**

## Calibration Data Description

Calibration data supported modes. For details, see.[Dataset Configuration Description](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#dataset---校准数据路径配置)    \:

For GLM-4.6V, a text prompt word is required for each sample during calibration.`text`and corresponding images`image`, the current missing sample is not supported.

## Generate quantified weights

### GLM-4.6V W8A8 Mixed Quantization

Quantification of the model has been integrated into[One-click quantization](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#参数说明)    .

```shell
msmodelslim quant \
    --model_path /path/to/GLM-4.6V_float_weights \
    --save_path /path/to/GLM-4.6V_quantized_weights \
    --device npu \
    --model_type GLM-4.6V \
    --quant_type w8a8 \
    --trust_remote_code True
```

## Appendixes

 * [Configuration of the multimodal_vlm_modelslim_v1 Quantization Service](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#multimodal_vlm_modelslim_v1-配置详解)    
