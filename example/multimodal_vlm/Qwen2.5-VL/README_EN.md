# Qwen2.5-VL Quantification Description

## Model Introduction

 * [Qwen2.5-VL](https://qwenlm.github.io/zh/blog/qwen2.5-vl/)    It is the flagship visual language model of the Qwen model family developed by Alibaba Cloud and is a huge leap forward from the previous release of Qwen2-VL. The main features of the Qwen2.5-VL are as follows:
    
     * Perceive a richer world: The Qwen2.5-VL is not only good at recognizing common objects, such as flowers, birds, fish and insects, but it also analyzes text, charts, icons, graphics and layouts in images.
     * Agent: As a visual agent, Qwen2.5-VL can infer and use tools dynamically. It initially has the ability to use computers and mobile phones.
     * Understanding long videos and capturing events: The Qwen2.5-VL can understand more than an hour of video, and this time it has the new ability to capture events by pinpointing the relevant video footage.
     * Visual positioning: Qwen2.5-VL can accurately locate objects in images by generating bounding boxes or points, and provide stable JSON output for coordinates and attributes.
     * Structured output: Qwen2.5-VL supports structured output of invoices, forms, and tables, which is beneficial to applications in finance and business fields.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/zh/getting_started/install_guide.md)    .
 * Run the following command to install the qwen_vl_utils dependency:
    
     * pip install qwen_vl_utils
 * For Qwen2.5-VL, the transformers version must be 4.49.0.
    
     * pip install transformers==4.49.0

## Qwen2.5-VL model currently validated quantization methods

| model                   | Raw floating-point weight                                                                 | Quantization mode         | Inference Framework Support                                                        | Quantization command                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------- | ------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Qwen2.5-VL-7B-Instruct  | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)       | W8A8 static quantization  | MindIE 2.2.RC1 and later versions: vLLM Ascend v0.10.2rc2 and later versions:      | [W8A8 static quantization (m2)](#11-qwen25-vl-w8a8-static-quantization-abnormal-value-suppression-algorithm-use-m2)     |
| Qwen2.5-VL-72B-Instruct | [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/tree/main)     | W8A8 static quantization  | MindIE 2.2.RC1 and later versions: vLLM Ascend v0.10.2rc2 and later versions:      | [W8A8 static quantization (m2)](#11-qwen25-vl-w8a8-static-quantization-abnormal-value-suppression-algorithm-use-m2)     |
| Qwen2.5-VL-7B-Instruct  | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)       | W4A8 Dynamic Quantization | MindIE does not support this function. vLLM Ascend does not support this function. | [W4A8 Dynamic Quantization](#13-qwen25-vl-w4a8-dynamic-quantization-abnormal-value-suppression-algorithm-using-m4)         |
| Qwen2.5-VL-72B-Instruct | [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/tree/main)     | W4A8 Dynamic Quantization | MindIE does not support this function. vLLM Ascend does not support this function. | [W4A8 Dynamic Quantization](#13-qwen25-vl-w4a8-dynamic-quantization-abnormal-value-suppression-algorithm-using-m4)         |

**Description:**

 * Click the link in the Quantization Command column to go to the specific quantization command.

## Generate quantified weights

 * Use quantitative weights in a unified manner.[quant_qwen2_5vl.py](./quant_qwen2_5vl.py)    Script generation. The following provides the quick startup command for Qwen2.5-VL model quantization weight generation.

## Use Example

 * To use NPU multi-card quantization, especially for the Qwen2.5-VL-72B model, configure the multi-card environment variables first. (Atlas 300I Duo series products do not support multi-card quantization.)
    
    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

 * To load a custom model, call`from_pretrained`To be specified when the function`trust_remote_code=True`so that the modified custom code file can be loaded correctly. (Ensure the security of the loaded custom code file.)

### 1. Qwen2.5-VL series

#### 1.1 Qwen2.5-VL W8A8 Static Quantization Abnormal Value Suppression Algorithm Use m2

Generate Qwen2.5-VL model quantization weights. Use the m2 algorithm to suppress abnormal values and run on the NPU. Replace \{floating-point weight path\} and \{quantization weight path\} with actual paths. \{Path of the calibration image\} The default value is ../calibImages. If two images in the ../calibImages directory are used as an example, the number of images in the COCO data set needs to be expanded to 30 to ensure the precision during quantization. In addition, you can replace the image with another image based on the actual scenario.

```shell
#Used for MindIE deployment.
python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m2 --mindie_format

#Used for vLLM Ascend deployment.
python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m2
```

#### 1.2 Qwen2.5-VL W8A8 Static Quantization Abnormal Value Suppression Algorithm Use m4

Generate the Qwen2.5-VL model quantization weight. The abnormal value suppression uses the m4 algorithm and runs on the NPU. Replace \{floating-point weight path\} and \{quantization weight path\} with the actual paths. The default value of \{Path of calibration image\} is ../calibImages. If two images are stored in the ../calibImages directory, the number of images in the COCO data set needs to be expanded to 30 to ensure precision during quantization. In addition, you can replace the image with another image based on the actual scenario.

```shell
#Used for MindIE deployment.
python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m4 --mindie_format

#Used for vLLM Ascend deployment.
python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m4
```

#### 1.3 Qwen2.5-VL W4A8 Dynamic Quantization Abnormal Value Suppression Algorithm Using m4

Generate the Qwen2.5-VL model quantization weight. Use the 4-bit per-group quantization weight and the 8-bit per-token quantization activation value. Use the m4 algorithm to suppress abnormal values and run on the NPU. Replace \{floating-point weight path\} and \{quantization weight path\} with actual paths. \{Path of the calibration image\} The default value is ../calibImages. If two images in the ../calibImages directory are used as an example, the number of images in the COCO data set needs to be expanded to 30 to ensure the precision during quantization. In addition, you can replace the image with another image based on the actual scenario.

```shell
python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 4 --a_bit 8 --act_method 1 --device_type npu --trust_remote_code True --anti_method m4 --open_outlier False --is_dynamic True --is_lowbit True --group_size 256
```

## Appendixes

### Quantization parameter description

| Parameter name    | meanings                                                                                                                                           | Default value                                                                                                                                  | How to Use                                                                                                                                                                                                                                                                                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | Floating-Point Weighted Path                                                                                                                       | No default value.                                                                                                                              | Mandatory. Enter the Qwen2.5-VL raw floating-point weights directory path.                                                                                                                                                                                                                                                                                              |
| calib_images      | Calibration Set Picture Path                                                                                                                       | ../calibImages                                                                                                                                 | Optional parameter; Enter the directory path for the calibration dataset. In this example, the image comes from the public dataset.[COCO](https://cocodataset.org/#download)    To ensure the quantization precision, the number of images needs to be expanded to 30 according to the example. You can replace the image with another image based on the actual scenario. |
| save_directory    | Quantifying Weight Path                                                                                                                            | No default value                                                                                                                               | Mandatory. Output the quantization weight path.                                                                                                                                                                                                                                                                                                                         |
| part_file_size    | Size of a weighted file, in GB.                                                                                                                    | The default value is None, indicating that the size of a single weight file is not limited and only one quantitative weight file is generated. | Optional parameter; File size of the generated quantization weight file. You need to customize the upper limit of the size of a single quantization weight file.                                                                                                                                                                                                        |
| w_bit             | Weight quantization bit                                                                                                                            | 8                                                                                                                                              | Optional parameter; In the Qwen2.5-VL quantization scenario, this parameter can be set to 4 or 8.                                                                                                                                                                                                                                                                       |
| a_bit             | Activated value quantization bit                                                                                                                   | 8                                                                                                                                              | Optional parameter; In the Qwen2.5-VL quantization scenario, the value can be set to 8.                                                                                                                                                                                                                                                                                 |
| device_type       | Quantify the type of running equipment                                                                                                             | 'cpu'                                                                                                                                          | Optional parameter; Value range: \['cpu','npu'\].                                                                                                                                                                                                                                                                                                                       |
| trust_remote_code | Trust Custom Code                                                                                                                                  | False                                                                                                                                          | Optional parameter; specifies`trust_remote_code=True`Enable the modified custom code file to be loaded correctly (Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.)                                                                                                                                                 |
| anti_method       | Abnormal value suppression algorithm                                                                                                               | 'm2'                                                                                                                                           | Optional parameter; Value range: \['m2','m4'\]. 'm2' corresponds to the optimized Outlier Suppression Plus outlier suppression algorithm in the multi-modal understanding model scenario, and 'm4' corresponds to the Iterative Smooth outlier suppression algorithm.                                                                                                   |
| act_method        | activation value quantization method                                                                                                               | 2                                                                                                                                              | Optional parameter; (1) 1 indicates the min-max quantization mode in the label-free scenario. (2) 2 indicates the histogram quantization mode in the Label-Free scenario. (3) 3 indicates the automatic hybrid quantization mode in the Label-Free scenario.                                                                                                            |
| open_outlier      | Indicates whether to enable weight abnormal value division.                                                                                        | True                                                                                                                                           | The value can be True or False. If this parameter is set to True, the weight abnormal value division function is enabled. Otherwise, the weight abnormal value division function is disabled.                                                                                                                                                                           |
| is_dynamic        | Whether dynamic quantization is used, that is, the activation quantization parameter in W8A8 is dynamically generated.                             | False                                                                                                                                          | The value can be True or False. If this parameter is set to True, dynamic quantization is used. Otherwise, dynamic quantization is not used.                                                                                                                                                                                                                            |
| is_lowbit         | Indicates whether to use the sparse quantization low bit algorithm.                                                                                | False                                                                                                                                          | The value can be True or False. If this parameter is set to True, the sparse quantization low bit algorithm is used. Otherwise, the sparse quantization low bit algorithm is not used.`w4a8_dynamic per-group`Set this parameter to True in quantization scenarios.                                                                                                     |
| group_size        | Indicates the number of groups quantized by per-group.                                                                                             | 64                                                                                                                                             | Set to 64, 128, 256, 512.`w4a8_dynamic per-group`In the quantization scenario, only 256 bytes are supported.                                                                                                                                                                                                                                                            |
| mindie_format     | Whether the weight configuration file after the quantization of the multi-modal understanding model is compatible with the existing MindIE version | False                                                                                                                                          | On`mindie_format`The quantization weight format saved in is compatible with the current MindIE version.`mindie_format`Quantization weights saved in can be used for vLLM Ascend deployment.                                                                                                                                                                             |

 * For more parameter configuration requirements, see the parameters configured during quantization.[QuantConfig](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)    and quantization parameter configuration class[Calibrator](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)    
