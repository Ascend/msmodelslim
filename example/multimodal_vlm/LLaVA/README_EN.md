# LLaVA Quantification Description

## Model Introduction

 * [LLaVA (Large Language and Vision Assistant)](https://github.com/haotian-liu/LLaVA)    is a large multimodal model published by the University of Wisconsin-Madison, Microsoft Research, and Columbia University researchers. It can complete image description, visual Q&A, image query, and code writing based on images. It can also be used for multi-modal chat and scientific Q&A to help understand image content and generate corresponding natural language text.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/zh/getting_started/install_guide.md)    .
 * The transformers version must be 4.37.2.
    
    ```bash
    pip install transformers==4.37.2
    ```

## Currently Validated Quantification Methods for LLaVA Models

| model         | Raw floating-point weight                                                                                         | Quantization mode        | Inference Framework Support                                                        | Quantization command                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------ |
| LLaVA-v1.5-7B | [llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/a272c74b2481d8aff3aa6fc2c4bf891fe57334fb)     | W8A8 static quantization | MindIE does not support this function. vLLM Ascend does not support this function. | [W8A8 static quantization](#llava-v15-7b-w8a8-static-quantization)     |

**Description:**

 * Click the link in the Quantization Command column to go to the specific quantization command.

## Generate quantified weights

 * Use quantitative weights in a unified manner.[quant_llava.py](./quant_llava.py)    Script generation. The following provides the quick start command for generating the LLaVA model quantization weight.

## Use Example

 * If NPU multi-card quantization is required, configure environment variables to support multi-card quantization. (Atlas 300I Duo series products do not support multi-card quantization.)
    
    ```shell
    #Select multiple SIM cards based on the site requirements. The following uses two SIM cards as an example:
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

 * To load a custom model, call`from_pretrained`To be specified when the function`trust_remote_code=True`so that the modified custom code file can be loaded correctly. (Ensure the security of the loaded custom code file.)

### 1. LLaVA-v1.5-7B

#### LLaVA-v1.5-7B W8A8 Static Quantization

Generate the W8A8 quantization weight of the LLaVA-v1.5-7B model. Use the m2 algorithm to suppress abnormal values and run on the NPU. Replace \{floating-point weight path\} and \{quantization weight path\} with actual paths. The default value of \{Path of the calibration image\} is ../calibImages. You can replace it with another image based on the site requirements.

```shell
python quant_llava.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --mindie_format
```

## Appendixes

### Quantization parameter description

| Parameter name    | Meaning:                                                                                                                                           | Default value                                                                                                                                  | How to Use                                                                                                                                                                                                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | floating-point weight path                                                                                                                         | No default value                                                                                                                               | Mandatory. Enter the path to the LLaVA original floating-point weight directory.                                                                                                                                                                                                              |
| calib_images      | Calibration Set Picture Path                                                                                                                       | ../calibImages                                                                                                                                 | Optional parameter; Enter the directory path for the calibration dataset. In this example, the image comes from the public dataset.[COCO](https://cocodataset.org/#download)    . Example Select two of the pictures. You can replace the image with another image based on the actual scenario. |
| save_directory    | Quantifying Weight Path                                                                                                                            | No default value                                                                                                                               | Mandatory. Output the quantization weight path.                                                                                                                                                                                                                                               |
| part_file_size    | Size of a weighted file, in GB.                                                                                                                    | The default value is None, indicating that the size of a single weight file is not limited and only one quantitative weight file is generated. | Optional parameter; File size of the generated quantization weight file. You need to customize the upper limit of the size of a single quantization weight file.                                                                                                                              |
| w_bit             | Weight quantization bit                                                                                                                            | 8                                                                                                                                              | Optional parameter; This parameter can be set to 8 in the LLaVA quantization scenario.                                                                                                                                                                                                        |
| a_bit             | Activated value quantization bit                                                                                                                   | 8                                                                                                                                              | Optional parameter; This parameter can be set to 8 in the LLaVA quantization scenario.                                                                                                                                                                                                        |
| device_type       | Quantify the type of running equipment                                                                                                             | 'npu'                                                                                                                                          | Optional parameter; Value range: \['cpu', 'npu'\].                                                                                                                                                                                                                                            |
| trust_remote_code | Trust Custom Code                                                                                                                                  | False                                                                                                                                          | Optional parameter; specifies`trust_remote_code=True`Enable the modified custom code file to be loaded correctly (Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.)                                                                       |
| mindie_format     | Whether the weight configuration file after the quantization of the multi-modal understanding model is compatible with the existing MindIE version | False                                                                                                                                          | On`mindie_format`The quantization weight format saved in is compatible with the current MindIE version.                                                                                                                                                                                       |

 * For more parameter configuration requirements, see the parameters configured during quantization.[QuantConfig](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)    and quantization parameter configuration class[Calibrator](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)    
