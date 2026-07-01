# Qwen3-MOE Quantification Description

## Model Introduction

 * [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)    ,[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)    Qwen3 is the latest generation of large language models in the Qwen family, offering comprehensive dense and mixed expert (MoE) models. Qwen3 is a breakthrough in reasoning, command-following, proxy capabilities, and multi-language support based on extensive training experience. The typical Qwen3-MoE structural models are Qwen3-235B-A22B and Qwen3-30B-A3B.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../docs/zh/getting_started/install_guide.md)    .
 * 4.51.0 must be installed for the transformers version.
    
     * pip install transformers==4.51.0

## Supported Model Versions and Quantification Policies

| Model Series  | Model version         | HuggingFace Link                                                                     | W8A8 | W8A16 | W4A8 | W4A16 | W4A4 | sparse quantization | KV Cache | Attention | Quantization command                                                    |
| ------------- | --------------------- | ------------------------------------------------------------------------------------ | ---- | ----- | ---- | ----- | ---- | ------------------- | -------- | --------- | ----------------------------------------------------------------------- |
| **Qwen3-MOE** | Qwen3-30B-A3B         | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)                              | ✅    |       | ✅    |       |      |                     |          |           | [W8A8](#qwen3-30b-a3b-w8a8-hybrid-quantization)    /[W4A8](#qwen3-30b-a3b-w4a8-hybrid-quantization)         |
|               | Qwen3-235B-A22B       | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)                          | ✅    |       | ✅    |       |      |                     |          |           | [W8A8](#qwen3-235b-a22b-w8a8-mixed-quantification)    /[W4A8](#qwen3-235b-a22b-w4a8-hybrid-quantification)     |
|               | Qwen3-Coder-480B-A35B | [Qwen3-Coder-480B-A35B](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct)     |      |       | ✅    |       |      |                     |          |           | [W4A8](#qwen3-coder-480b-a35b-w4a8-hybrid-quantization)                                   |

**Description:**

 * " indicates that the quantification policy has passed the official verification of msModelSlim. The function is complete and the performance is stable. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the quantization command column to go to the specific quantization command.

## Quantized weight generation

 * Quantified weights can be used[quant_qwen_moe_w8a8.py](./quant_qwen_moe_w8a8.py)    Script generation.

## Use Example

 * Replace \{floating-point weight path\} and \{quantization weight path\} with actual paths.
 * To use NPU multi-card quantization, configure environment variables first.
    
    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

### Qwen3-30B-A3B

#### Qwen3-30B-A3B W8A8 Hybrid Quantization

Generate Qwen3-30B-A3B model W8A8 hybrid quantization weight (attention:w8a8 quantization, MoE:w8a8 dynamic quantization)

```shell
python3 quant_qwen_moe_w8a8.py --model_path {浮点权重路径} \
--save_path {W8A8量化权重路径} \
--anti_dataset ../common/qwen3-moe_anti_prompt_50.json \
--calib_dataset ../common/qwen3-moe_calib_prompt_50.json \
--trust_remote_code True
```

#### Qwen3-30B-A3B W4A8 Hybrid Quantization

Generate Qwen3-30B-A3B model W4A8 hybrid quantization weight (Attention:w8a8 dynamic quantization, MoE:w4a8 dynamic quantization)

```shell
msmodelslim quant --model_type Qwen3-30B --model_path {浮点权重路径} --save_path {W4A8量化权重路径} --quant_type w4a8 --trust_remote_code True
```

### Qwen3-235B-A22B

#### Qwen3-235B-A22B W8A8 Mixed Quantification

Generate Qwen3-235B-A22B model W8A8 hybrid quantization weight (attention:w8a8 quantization, MoE:w8a8 dynamic quantization)

```shell
python3 quant_qwen_moe_w8a8.py --model_path {浮点权重路径} \
--save_path {W8A8量化权重路径} \
--anti_dataset ../common/qwen3-moe_anti_prompt_50.json \
--calib_dataset ../common/qwen3-moe_calib_prompt_50.json \
--trust_remote_code True \
--rot
```

#### Qwen3-235B-A22B W4A8 Hybrid Quantification

Generate Qwen3-235B-A22B model W4A8 hybrid quantization weight (Attention:w8a8 dynamic quantization, MoE:w4a8 dynamic quantization)

```shell
msmodelslim quant --model_type Qwen3-235B --model_path {浮点权重路径} --save_path {W4A8量化权重路径} --quant_type w4a8 --trust_remote_code True
```

### Qwen3-Coder-480B-A35B

#### Qwen3-Coder-480B-A35B W4A8 hybrid quantization

Generate Qwen3-Coder-480B-A35B model W4A8 hybrid quantization weight (Attention:w8a8 quantization, MoE:w4a8 dynamic quantization)

```shell
msmodelslim quant --model_type  Qwen3-Coder-480B-A35B --model_path {浮点权重路径} --save_path {W4A8量化权重路径} --quant_type w4a8 --trust_remote_code True
```

## Appendixes

### quant_qwen_moe_w8a8.py Quantization Parameters

| Parameter name    | Meaning:                                                          | Default value                            | How to Use                                                                                                                                                                                                                                     |
| ----------------- | ----------------------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | floating-point weight path                                        | No default value                         | Mandatory. Enter the Qwen3-MOE weight directory path.                                                                                                                                                                                          |
| save_path         | Quantifying Weight Path                                           | No default value                         | Mandatory. Path of the output quantization result.                                                                                                                                                                                             |
| layer_count       | Model Layers                                                      | 0                                        | Optional parameter; Number of actual quantized layers for debugging. 0 indicates that all layers are used.                                                                                                                                     |
| anti_dataset      | Path of the calibration dataset for the anti-outliers             | ../common/qwen3-moe_anti_prompt_50.json  | Optional parameter; Calibration dataset path for anti-outlier processing.                                                                                                                                                                      |
| calib_dataset     | Quantize Calibration Dataset Path                                 | ../common/qwen3-moe_calib_prompt_50.json | Optional parameter; Quantize the calibration set path.                                                                                                                                                                                         |
| batch_size        | Enter the batch size.                                             | 4                                        | Optional parameter; Batch size used when quantization calibration data is generated. A larger batch size indicates a faster calibration speed but requires more video memory and memory. If resources are insufficient, reduce the batch size. |
| mindie_format     | Whether to enable the old weight configuration file saving format | False                                    | On`mindie_format`The quantization weight format saved in is compatible with MindIE 2.1.RC1 and earlier versions.                                                                                                                               |
| trust_remote_code | Trust Custom Code                                                 | False                                    | Designated`trust_remote_code=True`Enable the modified custom code file to be loaded correctly Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.                                             |
| rot               | Enable preprocessing based on the rotation matrix.                | Not Enabled                              | Optional parameter; Enabled and specified.                                                                                                                                                                                                     |

Note: When loading a model through the transformers library in the quantization script, invoke the`from_pretrained`The function specifies`trust_remote_code=True`The modified modeling file can be loaded correctly. (Ensure that the source of the loaded modeling file is reliable to avoid potential security risks.)

For more parameter configuration requirements, see the parameters configured during quantization.[QuantConfig](../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)    and quantization parameter configuration class[Calibrator](../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)    
