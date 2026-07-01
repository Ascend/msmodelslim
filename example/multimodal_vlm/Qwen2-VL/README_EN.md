# Qwen2-VL Quantification Description

## Model Introduction

 * [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)    It is a new generation of Large Vision Language Model (LVLM) developed by Alibaba Cloud. Qwen2-VL is built on Qwen2, and has the following features compared to Qwen-VL:
    
     * Read images at different resolutions and different aspect ratios: Qwen2-VL has achieved a world-leading performance in visual comprehension benchmarks such as MathVista, DocVQA, RealWorldQA, MTVQA, and more.
     * Qwen2-VL can understand long videos longer than 20 minutes and supports applications such as Q&A, dialogue, and content creation based on video content.
     * Visual agents capable of operating mobile phones and robots: With the ability of complex reasoning and decision-making, Qwen2-VL can be integrated into mobile phones and robots to automatically operate according to the visual environment and text instructions.
     * Multilingual Support: To serve global users, in addition to English and Chinese, Qwen2-VL now supports understanding multilingual text in images, including most European languages, Japanese, Korean, Arabic, Vietnamese and more.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/zh/getting_started/install_guide.md)    .
 * The transformers version must be 4.46.0.
    
    ```bash
    pip install transformers==4.46.0
    ```

 * Other dependency packages need to be installed.
    
     * pip install qwen_vl_utils

## Currently Validated Quantification Methods for Qwen2-VL Models

| model                 | Raw floating-point weight                                                             | Quantization mode        | Inference Framework Support                                                                              | Quantization command                       |
| --------------------- | ------------------------------------------------------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| Qwen2-VL-7B-Instruct  | [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/tree/main)       | W8A8 static quantization | This function is supported by MindIE 2.1.RC1 and later versions. vLLM Ascend is not supported currently. | [W8A8 static quantization](#1-qwen2-vl-series)     |
| Qwen2-VL-72B-Instruct | [Qwen2-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct/tree/main)     | W8A8 static quantization | This function is supported by MindIE 2.1.RC1 and later versions. vLLM Ascend is not supported currently. | [W8A8 static quantization](#1-qwen2-vl-series)     |

**Description:**

 * Click the link in the Quantization Command column to go to the corresponding quantization command.

## Generate quantified weights

 * Use quantitative weights in a unified manner.[quant_qwen2vl.py](./quant_qwen2vl.py)    Script generation. The following provides the quick startup command for Qwen2-VL model quantization weight generation.

## Use Example

 * To use the NPU multi-card quantization, especially the Qwen2-VL-72B model, configure the multi-card environment variables first. (Atlas 300I Duo series products do not support multi-card quantization.)
    
    ```shell
    #Select multiple cards based on the site requirements. The following uses the 8-card quantization as an example:
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

 * To load a custom model, call`from_pretrained`To be specified when the function`trust_remote_code=True`so that the modified custom code file can be loaded correctly. (Ensure the security of the loaded custom code file.)

### 1. Qwen2-VL Series

#### Qwen2-VL W8A8 static quantization, using the abnormal value suppression m2 algorithm

Generate Qwen2-VL model quantization weights. Run on the NPU. Replace \{floating-point weight path\} and \{quantization weight path\} with actual paths. \{Path of the calibration image\} The default value is ../calibImages. If two images in the ../calibImages directory are used as an example, the number of images in the COCO data set needs to be expanded to 30 to ensure the precision during quantization. In addition, you can replace the image with another image based on the actual scenario.

```shell
python quant_qwen2vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m2 --mindie_format
```

#### Qwen2-VL W8A8 static quantization, using the abnormal value suppression m4 algorithm

Generate Qwen2-VL model quantization weights. Run on the NPU. Replace \{floating-point weight path\} and \{quantization weight path\} with actual paths. \{Path of the calibration image\} The default value is ../calibImages. If two images in the ../calibImages directory are used as an example, the number of images in the COCO data set needs to be expanded to 30 to ensure the precision during quantization. In addition, you can replace the image with another image based on the actual scenario.

```shell
python quant_qwen2vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m4 --mindie_format
```

## Appendixes

### Quantization parameter description

| Parameter name    | meanings                                                                                                                                           | Default value                                                                                                                                  | How to Use                                                                                                                                                                                                                                                                                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | Floating-Point Weighted Path                                                                                                                       | No default value                                                                                                                               | Mandatory. Enter the Qwen2-VL original floating-point weight directory path.                                                                                                                                                                                                                                                                                            |
| calib_images      | Calibration Set Picture Path                                                                                                                       | ../calibImages                                                                                                                                 | Optional parameter; Enter the directory path for the calibration dataset. In this example, the image comes from the public dataset.[COCO](https://cocodataset.org/#download)    To ensure the quantization precision, the number of images needs to be expanded to 30 according to the example. You can replace the image with another image based on the actual scenario. |
| save_directory    | Quantifying Weight Path                                                                                                                            | No default value                                                                                                                               | Mandatory. Output the quantization weight path.                                                                                                                                                                                                                                                                                                                         |
| part_file_size    | Size of a weighted file, in GB.                                                                                                                    | The default value is None, indicating that the size of a single weight file is not limited and only one quantitative weight file is generated. | Optional parameter; File size of the generated quantization weight file. You need to customize the upper limit of the size of a single quantization weight file.                                                                                                                                                                                                        |
| w_bit             | Weight quantization bit                                                                                                                            | 8                                                                                                                                              | Optional parameter; In the Qwen2-VL quantization scenario, this bit can be set to 8.                                                                                                                                                                                                                                                                                    |
| a_bit             | Activated value quantization bit                                                                                                                   | 8                                                                                                                                              | Optional parameter; In the Qwen2-VL quantization scenario, this bit can be set to 8.                                                                                                                                                                                                                                                                                    |
| device_type       | Quantifying the Type of Operating Equipment                                                                                                        | 'npu'                                                                                                                                          | Optional parameter; Value range: \['cpu', 'npu'\].                                                                                                                                                                                                                                                                                                                      |
| trust_remote_code | Trust Custom Code                                                                                                                                  | False                                                                                                                                          | Optional parameter; specifies`trust_remote_code=True`Enable the modified custom code file to be loaded correctly (Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.)                                                                                                                                                 |
| anti_method       | Abnormal value suppression algorithm                                                                                                               | 'm2'                                                                                                                                           | Optional parameter; Value range: \['m2','m4'\]. 'm2' corresponds to the optimized Outlier Suppression Plus outlier suppression algorithm in the multi-modal understanding model scenario, and 'm4' corresponds to the Iterative Smooth outlier suppression algorithm.                                                                                                   |
| mindie_format     | Whether the weight configuration file after the quantization of the multi-modal understanding model is compatible with the existing MindIE version | False                                                                                                                                          | Turn on`mindie_format`The quantization weight format saved in is compatible with the current MindIE version.                                                                                                                                                                                                                                                            |

 * For more parameter configuration requirements, see the parameters configured during quantization.[QuantConfig](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)    and quantization parameter configuration class[Calibrator](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)    
