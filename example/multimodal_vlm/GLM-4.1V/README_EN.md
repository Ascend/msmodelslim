# GLM-4.1V Quantification Description

## Model Introduction

 * [GLM-4.1V-9B-Thinking](https://github.com/zai-org/GLM-V)    A multi-modal model launched by AI and Tsinghua University team. This model introduces thinking paradigms and comprehensively improves model capabilities through reinforcement learning (Reinforcement Learning with Curriculum Sampling, RLCS) based on course sampling.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/zh/getting_started/install_guide.md)    .
 * For the GLM-4.1V, the transformers version must be 4.53.0.
    
     * pip install transformers==4.53.0

## Currently Validated Quantification Methods for the GLM-4.1V Model

| model                | Raw floating-point weight                                                              | Quantization mode     | Inference Framework Support                                                                          | Quantization command                       |
| -------------------- | -------------------------------------------------------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| GLM-4.1V-9B-Thinking | [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking/tree/main)     | W8A8SC Quantification | MindIE 3.0.RC1 is expected to support vLLM Ascend. Currently, MindIE does not support this function. | [W8A8SC Quantification](#the-glm-41v-9b-thinking-w8a8sc-quantization-abnormal-value-suppression-algorithm-uses-m2)     |

## Generate quantified weights

 * Use quantitative weights in a unified manner.[quant_glm41v.py](./quant_glm41v.py)    Script generation. In section "Using Examples", the command for quickly starting the GLM-4.1V-9B-Thinking model quantization weight generation is provided.
 * To use NPU multi-card quantization, configure the multi-card environment variables first. (Atlas 300I DUO series products do not support multi-card quantization.)
    
    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

 * To load a custom model, call`from_pretrained`To be specified when the function`trust_remote_code=True`so that the modified custom code file can be loaded correctly. (Ensure the security of the loaded custom code file.)

## Example

### GLM-4.1V Series

#### The GLM-4.1V-9B-Thinking W8A8SC quantization abnormal value suppression algorithm uses m2

This example generates the quantization weights for the GLM-4.1V-9B-Thinking model on the NPU. Use the m2 algorithm to suppress abnormal values.

Replace \{Floating Point Weight Path\} and \{W8A8S Quantization Weight Path\} with actual paths. \{Path of the calibration set image\} defaults to "../calibImages". When deploying the quantization weight, if the precision of the application scenario is obviously lost, you can replace the image with another image (30 images are recommended) based on the actual scenario.

The Atlas 300I DUO uses the following sparse quantization compression (W8A8SC) method. The compression function is supported only by the Atlas 300I DUO.

 * sparse quantization
    
    ```shell
    python quant_glm41v.py \
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
    
    **Note: MindIE must be installed before weight compression.**
    
    ```shell
    #The number of TPs is the number of tensor parallel parallel devices.
    export IGNORE_INFER_ERROR=1
    torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
    ```

## Appendixes

### Quantization parameter description

| Parameter name    | Meaning:                                                                                                                                           | Optional/Mandatory | Default value                                                                                                                                | How to Use                                                                                                                                                                                                                                                                                                                         |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | floating-point weight path                                                                                                                         | Mandatory.         | No default value                                                                                                                             | Enter the original floating-point weight directory path.                                                                                                                                                                                                                                                                           |
| calib_images      | Calibration Set Picture Path                                                                                                                       | Optional.          | ../calibImages                                                                                                                               | Enter the directory path for the calibration dataset. In this example, the image comes from the public dataset.[COCO](https://cocodataset.org/#download)    . If the precision of the image is obviously lost in the application scenario, you can replace the image with another one (recommended: 30) based on the actual scenario. |
| save_directory    | Quantifying Weight Path                                                                                                                            | Mandatory.         | No default value                                                                                                                             | Output the quantization weight path.                                                                                                                                                                                                                                                                                               |
| part_file_size    | Size of a weighted file, in GB.                                                                                                                    | Optional.          | If the default value is None, the size of a single weight file is not limited. In this case, only one quantitative weight file is generated. | Specifies the size of a generated weight file. You need to customize the upper limit of the size of a single weight file.                                                                                                                                                                                                          |
| w_bit             | Weight quantization bit                                                                                                                            | Optional.          | 8                                                                                                                                            | This parameter can be set to 4 or 8 in the GLM-4.1V-9B-Thinking quantization scenario.                                                                                                                                                                                                                                             |
| a_bit             | Activated value quantization bit                                                                                                                   | Optional.          | 8                                                                                                                                            | This parameter can be set to 8 in the GLM-4.1V-9B-Thinking quantization scenario.                                                                                                                                                                                                                                                  |
| device_type       | Quantifying the Type of Operating Equipment                                                                                                        | Optional.          | 'cpu'                                                                                                                                        | Value range: \['cpu','npu'\].                                                                                                                                                                                                                                                                                                      |
| trust_remote_code | Trust Custom Code                                                                                                                                  | Optional.          | False                                                                                                                                        | Designated`trust_remote_code=True`Enable the modified custom code file to be loaded correctly Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.                                                                                                                                 |
| anti_method       | Abnormal value suppression algorithm                                                                                                               | Optional.          | 'm2'                                                                                                                                         | Value: \['m2'\]. 'm2' corresponds to the optimized Outlier Suppression Plus outlier suppression algorithm in the multi-modal understanding model scenario.                                                                                                                                                                         |
| act_method        | activation value quantization method                                                                                                               | Optional.          | 2                                                                                                                                            | (1) 1 indicates the min-max quantization mode in the Label-Free scenario. (2) 2 indicates the histogram quantization mode in the Label-Free scenario. (3) 3 indicates the automatic hybrid quantization mode in the Label-Free scenario.                                                                                           |
| open_outlier      | Indicates whether to enable weight abnormal value division.                                                                                        | Optional.          | True                                                                                                                                         | The value can be True or False. If this parameter is set to True, the weight abnormal value division function is enabled. Otherwise, the weight abnormal value division function is disabled.                                                                                                                                      |
| is_dynamic        | Whether dynamic quantization is used, that is, the activation quantization parameter in W8A8 is generated dynamically.                             | Optional.          | False                                                                                                                                        | The value can be True or False. If this parameter is set to True, dynamic quantization is used. Otherwise, dynamic quantization is not used.                                                                                                                                                                                       |
| is_lowbit         | Indicates whether to use the sparse quantization low bit algorithm.                                                                                | Optional.          | False                                                                                                                                        | The value can be True or False. If this parameter is set to True, the sparse quantization low bit algorithm is used. Otherwise, the sparse quantization low bit algorithm is not used.`w4a8_dynamic per-group`Set this parameter to True in quantization scenarios.                                                                |
| co_sparse         | Whether to enable the sparse quantization function                                                                                                 | Optional.          | False                                                                                                                                        | True: Use the sparse quantization function. False: The sparse quantization function is not used.                                                                                                                                                                                                                                   |
| fraction          | Proportion of Protected Abnormal Values During Model Weight Sparse Quantization                                                                    | Optional.          | 0.01                                                                                                                                         | Value range: \[0.01, 0.1\].                                                                                                                                                                                                                                                                                                        |
| do_smooth         | Whether to enable smooth quantization                                                                                                              | Optional.          | False                                                                                                                                        | True: indicates that smooth quantization is enabled. False: The smooth quantization function is disabled.                                                                                                                                                                                                                          |
| use_sigma         | Whether to enable the sigma function                                                                                                               | Optional.          | False                                                                                                                                        | True: The sigma function is enabled. False: The sigma function is disabled.                                                                                                                                                                                                                                                        |
| sigma_factor      | Coefficient of sigma in sigma function                                                                                                             | Optional.          | 3.0                                                                                                                                          | The data type is float. The default value is 3.0. The value range is \[1.0, 3.0\]. Note: This parameter is valid only when use_sigma is set to True.                                                                                                                                                                               |
| torch_dtype       | Set the data type of the loading weight.                                                                                                           | Optional.          | bf16                                                                                                                                         | Value range: \['bf16', 'fp16'\]. The default value is bf16.                                                                                                                                                                                                                                                                        |
| group_size        | Number of groups quantized by per-group                                                                                                            | Optional.          | 64                                                                                                                                           | Set to 64, 128, 256, 512.`w4a8_dynamic per-group`In quantization scenarios, only 256 bytes are supported.                                                                                                                                                                                                                          |
| mindie_format     | Whether the weight configuration file after the quantization of the multi-modal understanding model is compatible with the existing MindIE version | Optional.          | False                                                                                                                                        | On`mindie_format`The quantization weight format saved in is compatible with the current MindIE version.`mindie_format`The quantization weights saved in can be used in vLLM Ascend deployments.                                                                                                                                    |

 * For more parameter configuration requirements, see the parameters configured during quantization.[QuantConfig](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)    and quantization parameter configuration class[Calibrator](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)    
