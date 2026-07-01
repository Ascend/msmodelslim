# Qwen-VL Quantification Description

## Model Introduction

 * [Qwen-VL](https://github.com/QwenLM/Qwen-VL)    Large Vision Language Model (LVLM) developed by Alibaba Cloud. It is possible to generate text or detection boxes with images, text, and detection boxes as inputs. This series of models has excellent performance, supports multi-language dialogue, multi-graph interlaced dialogue, Chinese open domain location capability, fine-grained image recognition and understanding capability.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/zh/getting_started/install_guide.md)    .
 * To avoid similar[The SimSun.ttf file in the model directory cannot be read.](https://github.com/QwenLM/Qwen-VL/issues/319)    You are advised to manually download the file.[SimSun.ttf](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/SimSun.ttf)    Move to the original floating-point weight path and modify the FONT_PATH in tokenization_qwen.py. For example, run the following command:
    
    ```python
    #30 lines of code
    #FONT_PATH = try_to_load_from_cache("Qwen/Qwen-VL-Chat", "SimSun.ttf")
    #if FONT_PATH is None:
    #if not os.path.exists("SimSun.ttf"):
    #ttf = requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/SimSun.ttf")
    #open("SimSun.ttf", "wb").write(ttf.content)
    #FONT_PATH = "SimSun.ttf"
    FONT_PATH = "SimSun.ttf"
    ```

 * Modify the SUPPORT_CUDA field in the modeling_qwen.py file in advance.
    
    ```python
    #35 lines of code
    SUPPORT_CUDA = False
    ```

 * The following dependency must be installed:
    
    ```bash
    pip install transformers-stream-generator
    ```

## Qwen-VL Models Currently Validated Quantification Methods

| model   | Raw floating-point weight                                 | Quantization mode        | Inference Framework Support                                                        | Quantization command                      |
| ------- | --------------------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------- | ----------------------------------------- |
| Qwen-VL | [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL/tree/main)     | W8A8 static quantization | MindIE does not support this function. vLLM Ascend does not support this function. | [W8A8 static quantization](#1-qwen-vl-series)     |

**Description:**

 * You can click the link in the Quantization Command column to go to the corresponding quantization command.

## Generate quantified weights

 * Use quantitative weights in a unified manner.[quant_qwenvl.py](./quant_qwenvl.py)    Script generation. The following provides the quick startup command for Qwen-VL model quantization weight generation.

## Use Example

 * To use NPU multi-card quantization, configure environment variables first. Only one to three card quantization is supported. (Atlas 300I Duo series products do not support multi-card quantization.)
    
    ```shell
    #Select multiple SIM cards based on the site requirements. The following uses three SIM cards as an example:
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

 * To load a custom model, call`from_pretrained`To be specified when the function`trust_remote_code=True`so that the modified custom code file can be loaded correctly. (Ensure the security of the loaded custom code file)

### 1. Qwen-VL Series

#### Qwen-VL W8A8 Static Quantization

Generate the Qwen-VL model quantization weight. Abnormal value suppression uses the m2 algorithm and runs on the NPU. Replace \{floating-point weight path\} and \{quantization weight path\} with actual paths. The default value of \{Path of the calibration image\} is ../calibImages. You can replace it with another image based on the site requirements.

```shell
python quant_qwenvl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --mindie_format
```

## Appendixes

### Quantization parameter description

| Parameter name    | Meaning:                                                                                                                                           | Default Value                                                                                                                                  | How to Use                                                                                                                                                                                                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | Floating-Point Weighted Path                                                                                                                       | No default value                                                                                                                               | Mandatory. Enter the Qwen-VL original floating-point weight directory path.                                                                                                                                                                                                                   |
| calib_images      | Calibration Set Picture Path                                                                                                                       | ./calibImages                                                                                                                                  | Optional parameter; Enter the directory path for the calibration dataset. In this example, the image comes from the public dataset.[COCO](https://cocodataset.org/#download)    . Example Select two of the pictures. You can replace the image with another image based on the actual scenario. |
| save_directory    | Quantifying Weight Path                                                                                                                            | No default value                                                                                                                               | Mandatory. Output the quantization weight path.                                                                                                                                                                                                                                               |
| part_file_size    | Size of a weighted file, in GB.                                                                                                                    | The default value is None, indicating that the size of a single weight file is not limited and only one quantitative weight file is generated. | Optional parameter; File size of the generated quantization weight file. You need to customize the upper limit of the size of a single quantization weight file.                                                                                                                              |
| w_bit             | Weight quantization bit                                                                                                                            | 8                                                                                                                                              | Optional parameter; In the Qwen-VL quantization scenario, the value can be set to 8.                                                                                                                                                                                                          |
| a_bit             | Activated value quantization bit                                                                                                                   | 8                                                                                                                                              | Optional parameter; In the Qwen-VL quantization scenario, the value can be set to 8.                                                                                                                                                                                                          |
| device_type       | Quantifying the Type of Operating Equipment                                                                                                        | 'npu'                                                                                                                                          | Optional parameter; Value range: \['cpu', 'npu'\].                                                                                                                                                                                                                                            |
| trust_remote_code | Trust Custom Code                                                                                                                                  | False                                                                                                                                          | Optional parameter; specifies`trust_remote_code=True`Enable the modified custom code file to be loaded correctly (Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.)                                                                       |
| mindie_format     | Whether the weight configuration file after the quantization of the multi-modal understanding model is compatible with the existing MindIE version | False                                                                                                                                          | On`mindie_format`The quantization weight format saved in is compatible with the current MindIE version.                                                                                                                                                                                       |

 * For more parameter configuration requirements, see the parameters configured during quantization.[QuantConfig](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)    and quantization parameter configuration class[Calibrator](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)    
