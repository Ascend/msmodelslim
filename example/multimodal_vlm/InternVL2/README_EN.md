# InternVL 2.0 Quantification Description

## Model Introduction

 * [InternVL 2.0](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/)    It is a multi-modal model developed by Shanghai Artificial Intelligence Laboratory and SenseTime. Its upgraded version InternVL 2.0 has reached the level of the international top commercial closed-source model in many key evaluation indicators.

Scholar Vientiane supports image, video, text, voice, three-dimensional, medical and other modes, and can complete more than 100 downstream tasks, and the performance is comparable to that of dedicated task models. The model exhibits outstanding capabilities when dealing with complex multimodal data, especially for tasks such as math, scientific charts, general charts, document parsing, infographics, and OCR.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/zh/getting_started/install_guide.md)    .
 * The transformers version must be 4.46.0.
    
    ```bash
    pip install transformers==4.46.0
    ```

 * Other dependency packages need to be installed.
    
    ```bash
    pip install timm fastchat
    ```

## Current Validated Quantification Methods for InternVL 2.0 Models

| model         | Raw floating-point weight                                                  | Quantization mode        | Inference Framework Support                                                        | Quantization command                                 |
| ------------- | -------------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------- | ---------------------------------------------------- |
| InternVL2-8B  | [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main)       | W8A8 static quantization | MindIE does not support this function. vLLM Ascend does not support this function. | [W8A8 static quantization](#internvl2-8b-w8a8-static-quantization)      |
| InternVL2-40B | [InternVL2-40B](https://huggingface.co/OpenGVLab/InternVL2-40B/tree/main)     | W8A8 static quantization | MindIE does not support this function. vLLM Ascend does not support this function. | [W8A8 static quantization](#internvl2-40b-w8a8-static-quantization)     |

**Description:**

 * Click the link in the Quantization Command column to go to the specific quantization command.

## Generate quantified weights

 * Use quantitative weights in a unified manner.[quant_internvl2.py](./quant_internvl2.py)    Script generation. The following provides the quick start command for generating the InternVL 2.0 model quantization weight.

## Use Example

 * To use the NPU multi-card quantization, especially the InternVL2-40B model, configure the multi-card environment variables first. (Atlas 300I Duo series products do not support multi-card quantization.)
    
    ```shell
    #Select multi-card based on the actual situation. The following uses eight-card quantization as an example:
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

 * To load a custom model, call`from_pretrained`To be specified when the function`trust_remote_code=True`so that the modified custom code file can be loaded correctly. (Ensure the security of the loaded custom code file.)

### 1. InternVL 2.0 Series

#### InternVL2-8B W8A8 Static Quantization

Generate the quantization weight of the InternVL2-8B model. Use the m2 algorithm to suppress abnormal values. Currently, only m2 is supported. If the NPU runs, replace \{floating-point weight path\} and \{quantization weight path\} with actual paths. \{Path of the calibration image\} is ./textvqa_val by default. You need to manually download the corresponding[textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)    Dataset.

```shell
python quant_internvl2.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --is_8B_model --trust_remote_code True --mindie_format
```

#### InternVL2-40B W8A8 Static Quantization

Generate the quantization weight of the InternVL2-40B model. Use the m2 algorithm to suppress abnormal values. Currently, only m2 is supported. If the NPU runs, replace \{floating-point weight path\} and \{quantization weight path\} with actual paths. \{Path of the calibration image\} is ./textvqa_val by default. You need to manually download the corresponding[textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)    Dataset.

```shell
python quant_internvl2.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --mindie_format
```

```python
#If an Out of memory error occurs due to uneven memory allocation when running a model on an NPU with 32 GB memory, you can optimize memory usage by setting the memory limit (using the max_memory parameter) when loading the model. The sample code is as follows:
model = AutoModel.from_pretrained(
      args.model_path,
      torch_dtype=dtype,
      low_cpu_mem_usage=True,
      device_map=device_map,
      use_safetensors=True,
      trust_remote_code=True,
      max_memory={0: "20GB", 1: "20GB", 2: "20GB", 3: "20GB", 4: "20GB", 5: "20GB", 6: "20GB", 7: "20GB", "cpu": "20GB"}).eval()
```

## Appendixes

### Quantization parameter description

| Parameter name    | meanings                                                                                                                                           | Default value                                                                                                                                  | How to Use                                                                                                                                                                                                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | floating-point weight path                                                                                                                         | No default value                                                                                                                               | Mandatory. Enter the directory path for the InternVL 2.0 raw floating-point weights.                                                                                                                                                                              |
| calib_images      | Calibration Set Picture Path                                                                                                                       | ./textvqa_val                                                                                                                                  | Mandatory. Enter the directory path for the calibration dataset. In this example, the image comes from the public dataset.[textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)    . Only this calibration dataset is supported in the current example. |
| calib_num         | Randomly selected number from calibration data                                                                                                     | 30                                                                                                                                             | (Optional) Select a certain amount of data from the calibration set for calibration as needed. You are advised to select 30 data records.                                                                                                                         |
| save_directory    | Quantifying Weight Path                                                                                                                            | No default value                                                                                                                               | Mandatory. Output the quantization weight path.                                                                                                                                                                                                                   |
| part_file_size    | Size of a weighted file, in GB.                                                                                                                    | The default value is None, indicating that the size of a single weight file is not limited and only one quantitative weight file is generated. | Optional parameter; File size of the generated quantization weight file. You need to customize the upper limit of the size of a single quantization weight file.                                                                                                  |
| w_bit             | Weight quantization bit                                                                                                                            | 8                                                                                                                                              | Optional parameter; This parameter can be set to 8 in the InternVL 2.0 quantization scenario.                                                                                                                                                                     |
| a_bit             | Activated value quantization bit                                                                                                                   | 8                                                                                                                                              | Optional parameter; In the InternVL 2.0 quantization scenario, this parameter can be set to 8.                                                                                                                                                                    |
| act_method        | activation value quantization method                                                                                                               | 1                                                                                                                                              | Optional parameter; (1) 1 indicates the min-max quantization mode in the label-free scenario. (2) 2 indicates the histogram quantization mode in the Label-Free scenario. (3) 3 indicates the automatic hybrid quantization mode in the Label-Free scenario.      |
| device_type       | Quantify the type of running equipment                                                                                                             | 'npu'                                                                                                                                          | Optional parameter; Value range: \['cpu', 'npu'\].                                                                                                                                                                                                                |
| is_8B_model       | Whether to use the 8B model                                                                                                                        | Not Enabled                                                                                                                                    | Optional parameter; Select the 8-byte model or 40-byte model as required. If this option is selected, the 8-byte model is specified.                                                                                                                              |
| trust_remote_code | Trust Custom Code                                                                                                                                  | False                                                                                                                                          | Optional parameter; specifies`trust_remote_code=True`Enable the modified custom code file to be loaded correctly (Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.)                                           |
| mindie_format     | Whether the weight configuration file after the quantization of the multi-modal understanding model is compatible with the existing MindIE version | False                                                                                                                                          | On`mindie_format`The quantization weight format saved in is compatible with the current MindIE version.                                                                                                                                                           |

 * For more parameter configuration requirements, see the parameters configured during quantization.[QuantConfig](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)    and quantization parameter configuration class[Calibrator](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)    
