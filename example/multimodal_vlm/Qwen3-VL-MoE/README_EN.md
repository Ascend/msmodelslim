# Qwen3-VL-MoE Quantification Instructions

## Model Introduction

Qwen3-VL-MoE is a large-scale, multi-modal visual language Mixture-of-Experts (MoE) model developed by the Alibaba Cloud Qwen team. It has the following features:

 * **Sparse MoE architecture: Uses sparsely activated MoE architecture to significantly reduce computing costs while maintaining high performance**
 * **Multi-modal understanding capability: supports the joint understanding of images and texts, and performs multiple tasks such as image description and visual Q&A.**
 * **Large-scale parameter: 30B-A3B and 235B-A22B. A indicates the number of activated parameters.**
 * **3D fusion expert weight: Expert layer weights are converged and stored in the form of 3D tensors, requiring special quantitative processing.**

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/zh/getting_started/install_guide.md)    .
 * Note: Due to the speciality of transformers of a later version, the PyTorch and torch_npu must be installed in 2.7.1.
 * For Qwen3-VL-MoE, transformers version 4.57. 1 is required:
    
    ```bash
    pip install transformers==4.57.1
    ```

 * You also need to install the flax dependency:
    
    ```bash
    pip install flax
    ```

## Qwen3-VL-MoE model currently validated quantization methods

| model              | Raw floating-point weight                                                                | Quantization mode          | Inference Framework Support                             | Quantization command                                   |
| ------------------ | ---------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------- | ------------------------------------------------------ |
| Qwen3-VL-235B-A22B | [Qwen3-VL-235B-A22B](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/tree/main)     | W8A8 Hybrid Quantification | MindIE to be supported. vLLM Ascend is being supported. | [W8A8 Hybrid Quantification](#qwen3-vl-moe-model-currently-validated-quantization-methods)     |

Note:[Qwen3-VL-30B-A3B](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main)    The quantization precision has not been verified. Users can try to configure it according to actual requirements. However, the quantization effect and functional stability cannot be officially guaranteed.

**Description:**

 * Click the link in the quantization command column to go to the specific quantization command.
 * W8A8 hybrid quantization: Static quantization is used for Attention and regular MLP layers, and dynamic quantization is used for MoE experts.

## quantization characteristics

### Auto Conversion at the MoE Expert Layer

 * **3D weight splitting: Automatically add the fused 3D expert weight.**`(num_experts, hidden_size, expert_dim)`Split into separate`nn.Linear`layering
 * **Layer-by-layer processing: Based on the layer-by-layer loading mechanism of the v1 framework, MoE conversion is automatically completed when each layer is loaded.**
 * **Memory-friendly: The in-place policy is used during conversion to release the original 3D weight in a timely manner, greatly reducing memory usage.**

### Abnormal Value Suppression (Iterative Smooth)

 * **QuaRot algorithm: Uses the rotation-based outlier suppression algorithm to significantly smooth data distribution and effectively reduce quantization errors.**
 * **iter_smooth algorithm: Use the iterative smoothing algorithm to suppress abnormal activation values and improve the quantization precision.**
 * **Multi-seed graph type: supports multi-seed graph fusion, such as norm-linear and ov.**
 * **Adaptive configuration: Automatically identifies the MoE layer structure and applies appropriate smoothing policies to different layer types.**

### hybrid quantization strategy

 * **Attention layer: W8A8 static quantization (per_tensor activation), suitable for activating stable distribution layers.**
 * **MoE Experts: W8A8 dynamic quantization (per_token activation), adapting to the activation differences of different tokens and maintaining precision**
 * **Language part MLP gate layer: does not perform quantization, maintains floating-point precision, and ensures expert routing accuracy.**
 * **linear_fc2 layer of the visual part: precision-sensitive, no quantization, and floating-point precision**
 * **Merger and deepstack layers of the visual part: precision-sensitive, no quantization, and floating-point precision.**

### Level-by-level quantization

 * **Memory optimization: Supports layer-by-layer loading, quantification, and offloading, significantly reducing the video memory usage.**
 * **Supports single card: With the layer-by-layer quantization feature, large model quantization can be completed on the Atlas 800I A2 (64 GB) device.**

## Generate quantified weights

## Use Example

### Qwen3-VL-235B-A22B W8A8 hybrid quantization

Quantification of the model has been integrated into[One-click quantization](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#参数说明)    .

```shell
msmodelslim quant \
    --model_path /path/to/qwen3_vl_moe_float_weights \
    --save_path /path/to/qwen3_vl_moe_quantized_weights \
    --device npu \
    --model_type Qwen3-VL-235B-A22B \
    --quant_type w8a8 \
    --trust_remote_code True
```

## FAQ

### Q1: Why do MoE experts use dynamic quantization?

**A: The activation distribution of MoE experts varies greatly among tokens.**

 * **Static quantization (per_tensor): All tokens share a scale → High precision loss.**
 * **Dynamic quantization (per_token): Each token is scaled independently → with higher precision.**

This is a standard practice for MoE models, refer to best practices for models such as DeepSeek-V3.

### Q2: How do I customize the calibration dataset?

**A: There are several methods:**

1. Use the`lab_calib/calibImages/`Directory and customize the text prompt of all images: Configure the text prompt by the default_text field in the YAML configuration file.
2. Use the`lab_calib/calibImages/`Directory and customize the text prompt of each image: Add a JSON/JSONL file to the image directory. For details, see.[dataset - Calibration datapath configuration](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#dataset---校准数据路径配置)    ;
3. Use the customized image directory and customize the text prompt of all images in a unified manner. Modify the prompt in the YAML configuration file.`dataset`The field is the customized image directory, and the text prompt is configured by the default_text field in the YAML configuration file.
4. Use the custom image directory and customize the text prompt for each image: modify in the YAML configuration file.`dataset`The field is a customized image directory. Add a JSON/JSONL file to the image directory. For details, see.[dataset - Calibration datapath configuration](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#dataset---校准数据路径配置)    .

You are advised to use images similar to the actual application scenario as the calibration set. Generally, the number of images in the calibration set does not exceed 30.

## Appendixes

### Related Resources

 * [Configuration Description of multimodal_vlm_modelslim_v1](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#multimodal_vlm_modelslim_v1-配置详解)    
 * [One-Click Quantization Configuration Protocol Description](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#量化配置协议详解)    
 * [Description of the feature of layer-by-layer quantization](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#逐层量化及分布式逐层量化)    
 * [QuaRot Algorithm Description](../../../docs/zh/quantization_algorithms/outlier_suppression_algorithms/quarot.md)    
 * [Iterative Smooth Algorithm Description](../../../docs/zh/quantization_algorithms/outlier_suppression_algorithms/iterative_smooth.md)    
 * [LinearQuantProcess Linear Layer Quantization Processor Description](../../../docs/zh/quantization_algorithms/quantization_algorithms/linear_quant.md)    
