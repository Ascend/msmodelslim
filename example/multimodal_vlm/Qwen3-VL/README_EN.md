# Qwen3-VL Quantification Case

## Model Introduction

[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)    Qwen's visual-language model is a comprehensive upgrade in all aspects: better text understanding and generation, deeper visual perception and reasoning, extended context length, enhanced spatial and video dynamic understanding, and stronger agent interaction.

## Environment Configuration

 * For details about basic environment configuration, see.[Installation Guide](../../../docs/zh/getting_started/install_guide.md)    Note: Due to the particularity of transformers of later versions, the PyTorch and torch_npu versions must be 2.2 or later.
 * For Qwen3-VL, transformers version 4.57. 1 is required:
    
    ```bash
    pip install transformers==4.57.1
    ```

## Qwen3-VL Model Currently Validated Quantification Methods

| model                 | Raw floating-point weight                                                   | Quantization mode     | Inference Framework Support                                                                          | Quantization command                       |
| --------------------- | --------------------------------------------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| Qwen3-VL-4B-Instruct  | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)       | W8A8 Quantification   | MindIE 3.0.RC1 will support vLLM Ascend v0.13.0 and later versions.                                  | [W8A8 Quantification](#12-qwen3-vl-4b-instruct-w8a8-quantification)      |
| Qwen3-VL-8B-Instruct  | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)       | W8A8SC Quantification | MindIE 3.0.RC1 is expected to support vLLM Ascend. Currently, MindIE does not support this function. | [W8A8SC Quantification](#11-qwen3-vl-8b-instruct-w8a8sc-quantization-abnormal-value-suppression-algorithm-using-m2)     |
| Qwen3-VL-32B-Instruct | [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)     | W8A8 Quantification   | MindIE 3.0.RC1 will support vLLM Ascend v0.13.0 and later versions.                                  | [W8A8 Quantification](#13-qwen3-vl-32b-instruct-w8a8-quantification)         |

**Note: Click the link in the Quantization Command column to go to the specific quantization command.**

### Use Cases

 * To use NPU multi-card quantization, configure the multi-card environment variables first. (Atlas 300I Duo series products do not support multi-card quantization.)
    
    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

 * To load a custom model, call`from_pretrained`To be specified when the function`trust_remote_code=True`to ensure that the modified custom code file can be loaded correctly (ensure the security of the loaded custom code file).

#### 1. Qwen3-VL Series

##### 1.1 Qwen3-VL-8B-Instruct W8A8SC Quantization Abnormal Value Suppression Algorithm Using m2

This example generates the quantization weights for the Qwen3-VL-8B-Instruct model on the NPU. Use the m2 algorithm to suppress abnormal values.

Replace \{Floating Point Weight Path\} and \{W8A8S Quantization Weight Path\} with actual paths. \{Path of calibration set images\} The default value is ../calibImages. Two images in the ../calibImages directory are used as the calibration set. When deploying the quantization weight, if the precision of the application scenario is obviously lost, you can replace the image with another image (30 images are recommended) based on the actual scenario.

The Atlas 300I DUO uses the following sparse quantization method

 * sparse quantization
    
    ```shell
    python quant_qwen3vl.py \
      --model_path {浮点权重路径} \
      --save_directory {W8A8S量化权重路径} \
      --calib_images {校准集图片路径} \
      --w_bit 4 \
      --a_bit 8 \
      --device_type npu \
      --anti_method m2 \
      --is_lowbit True \
      --fraction 0.01 \
      --use_sigma True \
      --torch_dtype fp16 \
      --trust_remote_code True
    ```

 * Weight Compression
    
    **Note: MindIE must be installed before weight compression. For details, see.**[MindIE Installation Guide](https://www.hiascend.com/document/detail/zh/mindie/230/envdeployment/instg/mindie_instg_0001.html)    
    
    ```shell
    #The number of TPs is the number of tensor parallel parallels.
    export IGNORE_INFER_ERROR=1
    torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
    ```

###### Quantization parameter description

| Parameter name    | meanings                                                                                                               | Default value                                                                                                                                | How to Use                                                                                                                                                                                                                                                                                                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | floating-point weight path                                                                                             | No default value                                                                                                                             | Mandatory. Enter the original floating-point weight directory path.                                                                                                                                                                                                                                                                                                     |
| calib_images      | Calibration Set Picture Path                                                                                           | ../calibImages                                                                                                                               | Optional parameter; Enter the directory path for the calibration dataset. In this example, the image comes from the public dataset.[COCO](https://cocodataset.org/#download)    To ensure the quantization precision, the number of images needs to be expanded to 30 according to the example. You can replace the image with another image based on the actual scenario. |
| save_directory    | Quantifying Weight Path                                                                                                | No default value                                                                                                                             | Mandatory. Output the quantization weight path.                                                                                                                                                                                                                                                                                                                         |
| part_file_size    | Size of a weighted file, in GB.                                                                                        | If the default value is None, the size of a single weight file is not limited. In this case, only one quantitative weight file is generated. | Optional parameter; Specifies the size of a generated weight file. You need to customize the upper limit of the size of a single weight file.                                                                                                                                                                                                                           |
| w_bit             | Weight quantization bit                                                                                                | 8                                                                                                                                            | Optional parameter; In the Qwen3-VL quantization scenario, this parameter can be set to 4 or 8.                                                                                                                                                                                                                                                                         |
| a_bit             | Activated value quantization bit                                                                                       | 8                                                                                                                                            | Optional parameter; In the Qwen3-VL quantization scenario, the value can be set to 8.                                                                                                                                                                                                                                                                                   |
| device_type       | Quantify the type of running equipment                                                                                 | 'cpu'                                                                                                                                        | Optional parameter; Value range: \['cpu','npu'\].                                                                                                                                                                                                                                                                                                                       |
| trust_remote_code | Trust Custom Code                                                                                                      | False                                                                                                                                        | Optional parameter; specifies`trust_remote_code=True`Enable the modified custom code file to be loaded correctly (Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.)                                                                                                                                                 |
| anti_method       | Abnormal value suppression algorithm                                                                                   | 'm2'                                                                                                                                         | Optional parameter; Value range: \['m2'\]. 'm2' corresponds to the optimized Outlier Suppression Plus outlier suppression algorithm in the multi-modal understanding model scenario.                                                                                                                                                                                    |
| act_method        | activation value quantization method                                                                                   | 2                                                                                                                                            | Optional parameter; (1) 1 indicates the min-max quantization mode in the label-free scenario. (2) 2 indicates the histogram quantization mode in the Label-Free scenario. (3) 3 indicates the automatic hybrid quantization mode in the Label-Free scenario.                                                                                                            |
| open_outlier      | Indicates whether to enable weight abnormal value division.                                                            | True                                                                                                                                         | The value can be True or False. If this parameter is set to True, the weight abnormal value division function is enabled. Otherwise, the weight abnormal value division function is disabled.                                                                                                                                                                           |
| is_dynamic        | Whether dynamic quantization is used, that is, the activation quantization parameter in W8A8 is dynamically generated. | False                                                                                                                                        | The value can be True or False. If this parameter is set to True, dynamic quantization is used. Otherwise, dynamic quantization is not used.                                                                                                                                                                                                                            |
| is_lowbit         | Indicates whether to use the sparse quantization low bit algorithm.                                                    | False                                                                                                                                        | The value can be True or False. If this parameter is set to True, the sparse quantization low bit algorithm is used. Otherwise, the sparse quantization low bit algorithm is not used.`w4a8_dynamic per-group`Set this parameter to True in quantization scenarios.                                                                                                     |
| co_sparse         | Whether to enable the sparse quantization function                                                                     | False                                                                                                                                        | True: Use the sparse quantization function. False: The sparse quantization function is not used.                                                                                                                                                                                                                                                                        |
| fraction          | Proportion of Protected Abnormal Values During Model Weight Sparse Quantization                                        | 0.01                                                                                                                                         | Value range: \[0.01, 0.1\].                                                                                                                                                                                                                                                                                                                                             |
| use_sigma         | Whether to enable the sigma function                                                                                   | False                                                                                                                                        | True: The sigma function is enabled. False: The sigma function is disabled.                                                                                                                                                                                                                                                                                             |
| sigma_factor      | Coefficient of sigma in the sigma function                                                                             | 3.0                                                                                                                                          | The data type is float, the default value is 3.0, and the value range is \[1.0, 3.0\]. Note: This parameter is valid only when use_sigma is set to True.                                                                                                                                                                                                                |
| torch_dtype       | Set the data type of the loading weight.                                                                               | bf16                                                                                                                                         | Value range: \['bf16', 'fp16'\]. The default value is bf16.                                                                                                                                                                                                                                                                                                             |

 * For more parameter configuration requirements, see the parameters configured during quantization.[QuantConfig](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)    and quantization parameter configuration class[Calibrator](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)    .

##### 1.2 Qwen3-VL-4B-Instruct W8A8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#参数说明)    . Use the`model_type=Qwen3-VL-4B-Instruct`,`quant_type=w8a8`to use a custom configuration (e.g., specifying save options), you can`config_path`Designated[qwen3_vl_4b_w8a8.yaml](../../../lab_practice/qwen3_vl/qwen3_vl_4b_w8a8.yaml)    .

```shell
msmodelslim quant \
    --model_path /path/to/qwen3_vl_4b_float_weights \
    --save_path /path/to/qwen3_vl_4b_quantized_weights \
    --device npu \
    --model_type Qwen3-VL-4B-Instruct \
    --quant_type w8a8 \
    --trust_remote_code True
```

When using a custom profile:

```shell
msmodelslim quant \
    --model_path /path/to/qwen3_vl_4b_float_weights \
    --save_path /path/to/qwen3_vl_4b_quantized_weights \
    --device npu \
    --model_type Qwen3-VL-4B-Instruct \
    --config_path lab_practice/qwen3_vl/qwen3_vl_4b_w8a8.yaml \
    --trust_remote_code True
```

**Description:**

 * Qwen3-VL-4B-Instruct The default precision is`bfloat16`If the model weight path is modified,`config.json`medium`torch_dtype`To the`float16`Quantization may cause abnormal model precision.
 * If the hardware supports only float16 precision inference (for example, the Atlas 300I/300T series), the default precision is recommended.`bfloat16`After quantization, the model weight path is lowered.`config.json`medium`torch_dtype`Changed to`float16`To reason.

##### 1.3 Qwen3-VL-32B-Instruct W8A8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#参数说明)    .

```shell
msmodelslim quant \
    --model_path /path/to/qwen3_vl_float_weights \
    --save_path /path/to/qwen3_vl_quantized_weights \
    --device npu \
    --model_type Qwen3-VL-32B-Instruct \
    --quant_type w8a8 \
    --trust_remote_code True
```
