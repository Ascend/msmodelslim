# Description of DeepSeek Quantification

## Model Introduction

 * [DeepSeek-LLM](https://github.com/deepseek-ai/deepseek-LLM)    Four models 7B Base, 7B Chat, 67B Base, and 67B Chat are trained from the Chinese and English mixed dataset containing 2 TB tokens.
 * [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)    Multi-head Latent Attention (MLA) is introduced, which uses low-rank key-value joint compression to eliminate the bottleneck of key-value cache during inference, thus supporting efficient inference. The DeepSeekMoE architecture is used in the FFN section, which enables stronger models to be trained at a lower cost.
 * [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder)    It consists of a series of code language models, all trained from scratch on 2T tags containing 87% code, 13% English and Chinese natural language. Each model is pre-trained in the project-level code corpus with 16K window size and extra blank-filling tasks to support project-level code completion and padding.
 * [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)    is an excellent hybrid expert (MoE) language model, with an overall parameter scale of 671 billion, with 3.7 billion parameters activated per token. The model has been innovated and optimized in the process of architecture design, training framework, pre-training and post-training.
 * [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)    The ability to perform complex tasks without human intervention is demonstrated through pure reinforcement learning, real reward mechanisms, and GRPO algorithms. Specifically, DeepSeek-R1 uses the large-scale reinforcement learning technology to significantly improve model performance with only a small amount of labeled data.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../docs/zh/getting_started/install_guide.md)    .
 * For the DeepSeek-V3/DeepSeek-R1 series models, complete the Check Before Running ([Check DeepSeek-V3 Before Running](#check-before-running)    /[Check DeepSeek-R1 Before Running](#check-before-running-1)    ).
 * Model quantization has high requirements on the video memory. Therefore, ensure that the video memory of a single SIM card is greater than or equal to 64 GB.

## Supported Model Versions and Quantification Policies

| Model Series       | Model version              | HuggingFace Link                                                                        | W8A8 | W8A16 | W4A8 | W8A8C8 | W4A8C8 | sparse quantization | KV Cache | Attention | FA3 Quantification | MTP quantization | Quantization command                                                                                                                                                                                                                                  |
| ------------------ | -------------------------- | --------------------------------------------------------------------------------------- | ---- | ----- | ---- | ------ | ------ | ------------------- | -------- | --------- | ------------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **DeepSeek-V2**    | DeepSeek-V2-Lite-Chat-16B  | [DeepSeek-V2-Lite-Chat-16B](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)      | ✅    | ✅     |      |        |        |                     |          |           |                    |                  | [W8A8](#deepseek-v2-w8a8-dynamic-quantification)    /[W8A16](#deepseek-v2-w8a16-quantification)                                                                                                                                                                                        |
|                    | DeepSeek-V2-Lite-Chat-236B | [DeepSeek-V2-Lite-Chat-236B](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)     | ✅    | ✅     |      |        |        |                     |          |           |                    |                  | [W8A8](#deepseek-v2-w8a8-dynamic-quantification)    /[W8A16](#deepseek-v2-w8a16-quantification)                                                                                                                                                                                        |
| **DeepSeek-Coder** | DeepSeek-Coder-33B         | [DeepSeek-Coder-33B](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct)       | ✅    | ✅     |      | ✅      |        |                     |          |           |                    |                  | [W8A8](#deepseek-coder-33b-w8a8-quantification)    /[W8A16](#deepseek-coder-33b-w8a16-quantification)    /[W8A8C8](#deepseek-coder-33b-w8a8c8-quantization)                                                                                                                                          |
| **DeepSeek-V3**    | DeepSeek-V3                |[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)|✅    |       |      |        |        ||          | | ✅ ||[W8A8](#deepseek-v3-w8a8-hybrid-quantization-mla-w8a8-quantizationmoe-w8a8-dynamic-quantization)/[FA3](#deepseek-v3-w8a8--fa3-hybrid-quantification)|
|| DeepSeek-V3.1              | [DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)                          | ✅    |       | ✅    | ✅      | ✅      |                     |          |           |                    | ✅                | [W8A8](#deepseek-v31-w8a8-hybrid-quantization--mtp-quantization)    /[W8A8C8](#deepseek-v31-w8a8c8-hybrid-quantization--mtp-quantization)    /[W4A8](#deepseek-v31-w4a8-hybrid-quantification)    /[W4A8C8](#deepseek-v31-w4a8c8-per-channel-quantization)    /[MTP quantization](#deepseek-v31-w8a8c8-hybrid-quantization--mtp-quantization)                                              |
|                    | DeepSeek-V3.2-Exp          | [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)                  | ✅    |       | ✅    |        |        |                     |          |           |                    |                  | [W8A8](#deepseek-v32-exp-including-mtp-layer-w8a8-hybrid-quantization)    /[W4A8](#deepseek-v32-exp-including-mtp-layer-w4a8-hybrid-quantization)                                                                                                                                                                                                    |
|                    | DeepSeek-V3.2              | [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)                          | ✅    |       |      |        |        |                     |          |           |                    |                  | [W8A8](#deepseek-v32-mtp-layer-included-w8a8-hybrid-quantization)                                                                                                                                                                                                                         |
| **DeepSeek-R1**    | DeepSeek-R1                | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)                              | ✅    |       | ✅    |        |        |                     |          |           | ✅                  | ✅                | [W8A8](#deepseek-r1-w8a8-hybrid-quantification)    /[W4A8](#deepseek-r1-w4a8-hybrid-quantification-the-first-three-levels-of-mlp-w8a8-dynamic-quantification-mla--sharing-experts-w8a8-dynamic-quantification-routing-experts-w4a8-dynamic-quantification)    /[W8A8 Dynamic](#deepseek-r1-w8a8-dynamic-quantization)    /[FA3](#deepseek-r1-w8a8--fa3-hybrid-quantification)    /[MTP quantization](#deepseek-r1-w8a8-hybrid-quantization--mtp-quantization)                                                             |
|                    | DeepSeek-R1-0528           | [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)                    | ✅    |       | ✅    | ✅      | ✅      |                     |          |           | ✅                  | ✅                | [W8A8](#deepseek-r1-0528-w8a8-hybrid-quantization--mtp-quantization)    /[W4A8](#deepseek-r1-0528-including-the-mtp-layer-w4a8-per-channel-quantization-the-routing-experts-at-the-non-mtp-layer-use-w4a8-per-channel-dynamic-quantization-and-other-linear-layers-use-w8a8-dynamic-quantization)    /[W8A8C8](#deepseek-r1-0528-w8a8c8-hybrid-quantization--mtp-quantization)    /[W4A8C8](#deepseek-r1-0528-including-the-mtp-layer-w4a8c8-per-channel-quantization)    /[MTP quantization](#deepseek-r1-0528-w8a8-hybrid-quantization--mtp-quantization)     |

**Description:**

 * " indicates that the quantification policy has passed the official verification of msModelSlim. The function is complete and the performance is stable. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the quantization command column to go to the specific quantization command.

## Quantized weight generation

 * Quantified weights can be used[quant_deepseek.py](./quant_deepseek.py)    ,[quant_deepseek_w8a8.py](./quant_deepseek_w8a8.py)    And to the[quant_deepseek_w4a8.py](./quant_deepseek_w4a8.py)    Script generation. The following provides the quick start command for generating the weight of the DeepSeek model.

### Quantization parameters in quant_deepseek.py

| Parameter name      | Meaning                                                                                                                      | Default Value                              | How to Use                                                                                                                                                                                                                                                                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path          | floating-point weight path                                                                                                   | No default value                           | Mandatory. Enter the DeepSeek weight directory path.                                                                                                                                                                                                                                                                                                       |
| save_directory      | Quantifying Weight Path                                                                                                      | No default value                           | Mandatory. Path of the output quantization result.                                                                                                                                                                                                                                                                                                         |
| part_file_size      | File size of the generated quantization weight, in GB.                                                                       | 5                                          | Optional parameter; File size of the generated quantization weight. The default value is 5 GB.                                                                                                                                                                                                                                                             |
| calib_texts         | Quantified calibration data                                                                                                  | No default value.                          | Optional parameter; Calibrate the dataset.                                                                                                                                                                                                                                                                                                                 |
| calib_file          | Quantified calibration data                                                                                                  | teacher_qualification.jsonl                | Optional parameter; JSON file for storing calibration data.                                                                                                                                                                                                                                                                                                |
| w_bit               | Weight quantization bit                                                                                                      | 8                                          | In the large model quantization scenario, this parameter can be set to 8 or 16. Set this parameter to 4 in the large model sparse quantization scenario.                                                                                                                                                                                                   |
| a_bit               | Activated value quantization bit                                                                                             | 8                                          | In the large model quantization scenario, this parameter can be set to 8 or 16. Set this parameter to 8 in the large model sparse quantization scenario.                                                                                                                                                                                                   |
| disable_names       | Name of the quantization layer to be manually rolled back                                                                    | Roll back all down_proj layers by default. | You can manually set this parameter based on the precision requirements. By default, all down_proj layers are rolled back.                                                                                                                                                                                                                                 |
| device_type         | Indicates the device type.                                                                                                   | cpu                                        | Value range: \['cpu','npu'\].                                                                                                                                                                                                                                                                                                                              |
| fraction            | Proportion of Protected Abnormal Values During Model Weight Sparse Quantization                                              | 0.01                                       | Value range: \[0.01, 0.1\].                                                                                                                                                                                                                                                                                                                                |
| act_method          | activation value quantization method                                                                                         | 1                                          | (1) 1 indicates the min-max quantization mode in the label-free scenario. (2) 2 indicates the histogram quantization mode in the Label-Free scenario. (3) 3 indicates the automatic hybrid quantization mode in the label-free scenario. It is recommended in the LLM large model scenario.                                                                |
| co_sparse           | Whether to enable the sparse quantization function                                                                           | False                                      | True: Use the sparse quantization function. False: The sparse quantization function is not used.                                                                                                                                                                                                                                                           |
| anti_method         | Outlier Suppression Parameters                                                                                               | No default value                           | 'm1': indicates the SmoothQuant algorithm. 'm2': enhanced SmoothQuant algorithm, which is recommended. 'm3': indicates the AWQ algorithm. 'm4': smooth optimization algorithm. 'm5': CBQ quantization algorithm.                                                                                                                                           |
| disable_level       | L Automatic Rollback Level                                                                                                   | L0                                         | A configuration example is as follows:'L0': default value, indicating that the rollback is not performed. 'L1': Roll back layer 1. 'L2': Roll back two layers. 'L3': Roll back to Layer 3. 'L4': Roll back four layers. 'L5': Roll back five layers.                                                                                                       |
| do_smooth           | Enable smooth quantization                                                                                                   | False                                      | True: indicates that smooth quantization is enabled. False: The smooth quantization function is disabled.                                                                                                                                                                                                                                                  |
| use_sigma           | Whether to enable the sigma function                                                                                         | False                                      | True: The sigma function is enabled. False: The sigma function is disabled.                                                                                                                                                                                                                                                                                |
| use_reduce_quant    | Indicates whether the weight is quantized by lccl all reduce.                                                                | False                                      | ID used for MindIE inference.                                                                                                                                                                                                                                                                                                                              |
| tp_size             | Simulate the number of cards in multi-card quantization.                                                                     | 1                                          | Value range: \[1, 2, 4, 8, 16\]. The default value is 1, indicating that the analog multi-card quantization function is disabled. When this parameter is set to 2, 4, 8, or 16, linear at the communication layer simulates multiple cards. Each card uses different scale and offset for quantization.                                                    |
| sigma_factor        | Coefficient of sigma in the sigma function                                                                                   | 3.0                                        | The data type is float. The default value is 3.0. The value range is \[1.0, 3.0\]. Note: This parameter is valid only when use_sigma is set to True.                                                                                                                                                                                                       |
| is_lowbit           | Indicates whether to enable the low bit quantization function.                                                               | False                                      | (1) When w_bit=4 and a_bit=8, large model sparse quantization is used, indicating that low-bit sparse quantization is enabled. (2) In other scenarios, the automatic quantization precision optimization function is enabled for large model quantization. Currently, the automatic precision optimization framework supports W8A8 and W8A16 quantization. |
| mm_tensor           | Whether to enable the mm_tensor quantization function                                                                        | True                                       | True: The mm_tensor quantization function is enabled. False: The mm_tensor quantization function is disabled.                                                                                                                                                                                                                                              |
| w_sym               | Whether to enable the w_sym quantization function                                                                            | True                                       | True: The w_sym quantization function is enabled. False: The w_sym quantization function is disabled.                                                                                                                                                                                                                                                      |
| use_kvcache_quant   | Specifies whether to use the kvcache quantization function.                                                                  | False                                      | True: Use the kvcache quantization function. False: The kvcache quantization function is not used.                                                                                                                                                                                                                                                         |
| use_fa_quant        | Whether to use FA3 quantization                                                                                              | False                                      | True: Use the FA3 quantization type. False: The FA3 quantization type is not used.                                                                                                                                                                                                                                                                         |
| fa_amp              | Number of layers that are automatically rolled back in the FA3 quantization scenario                                         | 0                                          | The data type is int, and the default value is 0. The value must be greater than or equal to 0 and less than or equal to the number of model layers. If the number of model layers exceeds the maximum number of model layers, the number of rollback layers is the maximum number of model layers.                                                        |
| open_outlier        | Indicates whether to enable weight abnormal value division.                                                                  | True                                       | True: indicates that weight abnormal value classification is enabled. False: Disable weight abnormal value division. Note: (1) This parameter takes effect only when lowbit is set to True. (2) In the per-group quantization scenario, is_lowbit needs to be set to True and open_outlier needs to be set to False.                                       |
| group_size          | Size of the group in the per-group quantization.                                                                             | 64                                         | The default value is 64, and the value can be 32, 64, or 128. Note: This parameter is applicable only to the per-group quantization scenario. Set is_lowbit to True and open_outlier to False.                                                                                                                                                             |
| is_dynamic          | Indicates whether to use the per-token dynamic quantization function.                                                        | False                                      | True: Use per-token dynamic quantization. False: Do not use per-token dynamic quantization.                                                                                                                                                                                                                                                                |
| input_ids_name      | Specifies the key name corresponding to the input ID in the word segmentation result.                                        | input_ids                                  | None                                                                                                                                                                                                                                                                                                                                                       |
| attention_mask_name | Specifies the key name corresponding to the attention mask in the word segmentation result.                                  | attention_mask                             | None                                                                                                                                                                                                                                                                                                                                                       |
| tokenizer_args      | User-defined parameter input when loading the user-defined tokenizer.                                                        | None                                       | The value is transferred in dictionary mode.                                                                                                                                                                                                                                                                                                               |
| disable_last_linear | Indicates whether to roll back the last linear layer.                                                                        | True                                       | True: Roll back the last linear layer. False: The last linear layer is not rolled back.                                                                                                                                                                                                                                                                    |
| model_name          | Model name. This parameter is optional.                                                                                      | None                                       | Controls the outlier suppression parameter.                                                                                                                                                                                                                                                                                                                |
| trust_remote_code   | Trust Custom Code                                                                                                            | False                                      | Designated`trust_remote_code=True`Enable the modified custom code file to be loaded correctly (Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.)                                                                                                                                                       |
| mindie_format       | Whether the weight configuration file after non-multimodal model quantization is compatible with the existing MindIE version | False                                      | On`mindie_format`The quantization weight format saved in is compatible with MindIE 2.1.RC1 and earlier versions.                                                                                                                                                                                                                                           |

Note: When loading a model through the transformers library in the quantization script, call`from_pretrained`The function specifies`trust_remote_code=True`The modified modeling file can be correctly loaded. (Ensure the security of the modeling file to be loaded.)

### quant_deepseek_w8a8.py/quant_deepseek_w4a8.py Quantization Parameters

| Parameter name    | Meaning                                                                                                                      | Default value                                       | How to Use                                                                                                                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | floating-point weight path                                                                                                   | No default value                                    | Mandatory. Enter the DeepSeek weight directory path.                                                                                                                                                                                           |
| save_path         | Save path of the quantization weight.                                                                                        | No default value.                                   | Mandatory. Path of the output quantization result.                                                                                                                                                                                             |
| layer_count       | Number of layers when the model is loaded                                                                                    | 0                                                   | Optional parameter; Number of actual quantized layers for debugging.                                                                                                                                                                           |
| anti_dataset      | Outlier Calibration Dataset Path                                                                                             | ../common/deepseek_anti_prompt.json                 | Optional parameter; Outlier suppression calibration set path.                                                                                                                                                                                  |
| calib_dataset     | Calibration Data Set File Path                                                                                               | ../common/deepseek_calib_prompt.json                | Optional parameter; Quantize the calibration set path.                                                                                                                                                                                         |
| batch_size        | Enter the batch size.                                                                                                        | 4(quant_deepseek_w8a8.py) 1(quant_deepseek_w4a8.py) | Optional parameter; Batch size used when quantization calibration data is generated. A larger batch size indicates a faster calibration speed but requires more video memory and memory. If resources are insufficient, reduce the batch size. |
| from_fp8          | Specify the original model as the FP8 weight.                                                                                | Not Enabled                                         | Optional parameter; This parameter is specified when it is enabled. It cannot coexist with from_bf16.                                                                                                                                          |
| from_bf16         | Specifies the BF16 weight as the original model.                                                                             | Not Enabled                                         | Optional parameter; This parameter is specified when it is enabled. It cannot coexist with from_fp8.                                                                                                                                           |
| mindie_format     | Whether the weight configuration file after non-multimodal model quantization is compatible with the existing MindIE version | False                                               | On`mindie_format`The quantization weight format saved in is compatible with MindIE 2.1.RC1 and earlier versions.                                                                                                                               |
| quant_mtp         | quantification mode                                                                                                          | none                                                | (Optional) none: The mtp weight is not saved. float: stores the mtp floating-point weight. mix: Saves the mtp mixed quantization weight.                                                                                                       |
| trust_remote_code | Trust Custom Code                                                                                                            | False                                               | Optional parameter; specifies`trust_remote_code=True`Enable the modified custom code file to be loaded correctly Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.                          |

### quant_deepseek_w8a8.py Extra Quantization Parameters

| Parameter name | Meaning:                                           | Default value | How to Use                                 |
| -------------- | -------------------------------------------------- | ------------- | ------------------------------------------ |
| fa_quant       | Specified FA Quantization                          | Not Enabled   | (Optional) Enabled and specified.          |
| dynamic        | Specify dynamic quantization                       | Not Enabled   | Optional parameter; Enabled and specified. |
| disable_anti   | Turn off outlier suppression                       | Not Enabled   | Optional parameter; Enabled and specified. |
| anti_method    | outlier suppression method                         | m4            | Optional parameter; Value range: m4, m6    |
| rot            | Enable preprocessing based on the rotation matrix. | Not Enabled   | Optional parameter; Enabled and specified. |

Note: When the transformers library is used to load a model in the quantization script, this function is invoked.`from_pretrained`The function specifies`trust_remote_code=True`The modified modeling file can be loaded correctly. (Ensure the security of the modeling file to be loaded.)

For more parameter configuration requirements, see the parameters configured during quantization.[QuantConfig](../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)    and quantization parameter configuration class[Calibrator](../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)    

## Use Example

Please put the`${model_path}`And to the`${save_path}`Replace it with the actual path of the user.

### DeepSeek-V2 Series

#### DeepSeek-V2 W8A16 Quantification

 * Generate the W8A16 quantization weight of the DeepSeek-V2 model, use the histogram quantization mode, and perform the calculation on the CPU.
    
    ```shell
    python3 quant_deepseek.py --model_path ${model_path} --save_directory ${save_path} --device_type cpu --act_method 2 --w_bit 8 --a_bit 16
    ```

#### DeepSeek-V2 W8A8 Dynamic Quantification

 * Generate the W8A8 Dynamic Quantization Weight of the DeepSeek-V2 model, use the histogram quantization mode, and perform the calculation on the CPU.
    
    ```shell
    python3 quant_deepseek.py --model_path ${model_path} --save_directory ${save_path} --device_type cpu --act_method 2 --w_bit 8 --a_bit 8  --is_dynamic True
    ```

**Note: To load a custom model, call**`from_pretrained`To be specified when the function`trust_remote_code=True`The modified custom code file can be loaded correctly. (Ensure the security of the loaded custom code file)

### DeepSeek-Coder Series

#### DeepSeek-Coder-33B W8A8 Quantification

 * Generate the W8A8 quantization weight of the DeepSeek-Coder-33B model, use the activation value quantization mode of the automatic mixing of min-max and histogram, and use the SmoothQuant enhanced algorithm to perform operations on the NPU.
    
    ```shell
    python3 quant_deepseek.py --model_path ${model_path} --save_directory ${save_path} --device_type npu --act_method 3 --anti_method m2 --w_bit 8 --a_bit 8 --model_name deepseek_coder
    ```

#### DeepSeek-Coder-33B W8A16 Quantification

 * Generate the W8A16 quantization weight of the DeepSeek-Coder-33B model and use the AWQ algorithm to perform the calculation on the NPU.
    
    ```shell
    python3 quant_deepseek.py --model_path ${model_path} --save_directory ${save_path} --device_type npu --anti_method m3 --w_bit 8 --a_bit 16 --model_name deepseek_coder
    ```

#### DeepSeek-Coder-33B W8A8C8 quantization

 * Generate the W8A8C8 quantization weight of the DeepSeek-Coder-33B model, use the histogram activation value quantization mode, and use the SmoothQuant enhanced algorithm to perform the calculation on the NPU.
    
    ```shell
    python3 quant_deepseek.py --model_path ${model_path} --save_directory ${save_path} --device_type npu --act_method 2 --anti_method m2 --w_bit 8 --a_bit 8 --use_kvcache_quant True --model_name deepseek_coder
    ```

### DeepSeek-V3 Series

#### Check Before Running

The DeepSeek-V3 model is large and needs to be manually adapted. To avoid wasting time, modify the related content according to the following mandatory check items before running the script:

1. (Skip this check item for the V3.2-Exp model.) The Ascend does not support the flash_attn library. Some code in modeling_deepseek.py in the weight folder needs to be commented out during running.
    ![img.png](img.png)    
2. Transformers 4.48.2 must be installed.
3. (Skip this check item for the V3.2-Exp model.) Currently, transformers do not support loading in the FP8 quantization format. You need to delete the following fields from the config.json file in the weight folder:
    ![img_1.png](img_1.png)    

#### DeepSeek-V3 W8A8 hybrid quantization (MLA w8a8 quantization,MOE w8a8 dynamic quantization)

Note: The script has built-in FP8 dequantization logic. Currently, the FP8 model weight and BF16 model weight converted from the official script can be input.

1. In the single quantification scenario, FP8 weights are directly quantified. Compared with the process of converting FP8 weights into BF16 weights using official scripts, FP8 weights are time-consuming and 1.3 TB BF16 weights are saved.

2. In multiple quantification scenarios, you are advised to use an official script to store the BF16 weight persistently and reuse the BF16 weight during subsequent quantification. The redundant FP8 dequantization step is omitted, which significantly improves efficiency.

 * Generate the W8A8 hybrid weight of the DeepSeek-V3 model.
    
    ```shell
    python3 quant_deepseek_w8a8.py --model_path ${model_path} --save_path ${save_path} --batch_size 4 --trust_remote_code True
    ```

#### DeepSeek-V3 W8A8 + FA3 Hybrid Quantification

 * Generate the W8A8 + FA3 hybrid weight of the DeepSeek-V3 model.
    
    ```shell
    python3 quant_deepseek_w8a8.py --model_path ${model_path} --save_path ${save_path} --batch_size 4 --fa_quant --trust_remote_code True
    ```

### DeepSeek-V3.1 Series

#### DeepSeek-V3.1 W8A8 hybrid quantization + MTP quantization

 * Generate DeepSeek-V3.1 W8A8 Hybrid Quantization + MTP Quantization
    
    ```shell
    python3 quant_deepseek_w8a8.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --batch_size 8 \
    --anti_dataset ../common/deepseek_anti_prompt_50_v3_1.json \
    --calib_dataset ../common/deepseek_calib_prompt_50_v3_1.json \
    --anti_method m4 \
    --quant_mtp mix \
    --rot \
    --trust_remote_code True
    ```

#### DeepSeek-V3.1 W8A8C8 Hybrid Quantization + MTP Quantization

 * Generate DeepSeek-V3.1 W8A8C8 Mixed Quantization + MTP Quantization
    
    ```shell
    python3 quant_deepseek_w8a8.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --batch_size 8 \
    --anti_dataset ../common/deepseek_anti_prompt_50_v3_1.json \
    --calib_dataset ../common/deepseek_calib_prompt_50_v3_1.json \
    --anti_method m4 \
    --quant_mtp mix \
    --rot \
    --fa_quant \
    --trust_remote_code True
    ```

#### DeepSeek-V3.1 W4A8 Hybrid Quantification

 * Generate the W4A8 hybrid weight of the DeepSeek-V3.1 model.
    
    ```shell
    python3 quant_deepseek_w4a8.py --model_path ${model_path} --save_path ${save_path} --anti_dataset ../common/deepseek_anti_prompt_50_v3_1.json --calib_dataset ../common/deepseek_calib_prompt_50_v3_1.json --quant_mtp mix  --batch_size 16 --trust_remote_code True
    ```

#### DeepSeek-V3.1 W4A8C8 per-channel Quantization

 * Generate the W4A8C8 per-channel quantitative weight of the DeepSeek-V3.1 model.
    
    ```shell
    msmodelslim quant \
     --model_path ${model_path} \
     --save_path ${save_path} \
     --model_type DeepSeek-V3.1 \
     --quant_type w4a8c8 \
     --trust_remote_code True
    ```

### DeepSeek-V3.2 Series

#### DeepSeek-V3.2-Exp (including MTP layer) W8A8 hybrid quantization

```shell
msmodelslim quant \
 --model_path ${model_path} \
 --save_path ${save_path} \
 --model_type DeepSeek-V3.2-Exp \
 --quant_type w8a8 \
 --trust_remote_code True
```

#### DeepSeek-V3.2-Exp (including MTP layer) W4A8 hybrid quantization

```shell
msmodelslim quant \
 --model_path ${model_path} \
 --save_path ${save_path} \
 --model_type DeepSeek-V3.2-Exp \
 --quant_type w4a8 \
 --trust_remote_code True
```

##### DeepSeek-V3.2 (MTP layer included) W8A8 hybrid quantization

```shell
msmodelslim quant \
 --model_path ${model_path} \
 --save_path ${save_path} \
 --model_type DeepSeek-V3.2 \
 --quant_type w8a8 \
 --trust_remote_code True
```

### DeepSeek-R1 Series

#### Check Before Running

The DeepSeek-R1 model is large and needs to be manually adapted. To avoid wasting time, modify the related content according to the following mandatory check items before running the script:

 1. The Ascend does not support the flash_attn library. Some code in modeling_deepseek.py in the weight folder needs to be commented out during running.
    ![img.png](img.png)    
 2. The transformers of the 4.48.2 version must be installed.
 3. Currently, transformers do not support loading in FP8 quantization format. You need to delete the following fields from the config.json file in the weight folder:
    ![img_1.png](img_1.png)    

#### DeepSeek-R1 W8A8 Hybrid Quantification

 * Generate the W8A8 hybrid weight of the DeepSeek-R1 model.
    
    ```shell
    python3 quant_deepseek_w8a8.py --model_path ${model_path} --save_path ${save_path} --batch_size 4 --trust_remote_code True
    ```

#### DeepSeek-R1 W8A8 + FA3 Hybrid Quantification

 * Generate the W8A8 + FA3 hybrid weight of the DeepSeek-R1 model.
    
    ```shell
    python3 quant_deepseek_w8a8.py --model_path ${model_path} --save_path ${save_path} --batch_size 4 --fa_quant --trust_remote_code True
    ```

#### DeepSeek-R1 W8A8 Dynamic Quantization

Note: If outlier suppression is performed for the current per-token quantization, the precision may be abnormal. It is recommended that the --disable_anti parameter be used together.

```shell
#Use the disable_anti parameter to reduce the time required for quantization.
python3 quant_deepseek_w8a8.py --model_path ${model_path} --save_path ${save_path} --dynamic --disable_anti --trust_remote_code True
```

#### DeepSeek-R1 W4A8 Hybrid Quantification (The first three levels of MLP: w8a8 dynamic quantification, MLA & sharing experts: w8a8 dynamic quantification, routing experts: w4a8 dynamic quantification)

Note: The script has built-in FP8 dequantization logic. Currently, the FP8 model weight and BF16 model weight converted from the official script can be input.

1. In single quantification scenarios, FP8 weights are directly quantified. Compared with the process of converting FP8 weights into BF16 weights using official scripts, FP8 weights are time-consuming and 1.3 TB BF16 weights are saved.

2. In multiple quantization scenarios, you are advised to use an official script to store the BF16 weight persistently and reuse the BF16 weight during subsequent quantization. The redundant FP8 dequantization step is omitted, which significantly improves efficiency.

3. If the number of TPs during inference is greater than 16, add the --mindie_format parameter during quantization.

 * Generate the W4A8 hybrid weight of the DeepSeek-R1 model.
    
    ```shell
    #The following command uses 10 calibration sets by default.
    python3 quant_deepseek_w4a8.py --model_path ${model_path} --save_path ${save_path} --trust_remote_code True
    
    #If you want to obtain higher precision, use 50 calibration sets. If the display memory is sufficient, try 16 batch_size to load the calibration sets.
    python3 quant_deepseek_w4a8.py --model_path ${model_path} --save_path ${save_path} --anti_dataset ../common/deepseek_anti_prompt_50.json --calib_dataset ../common/deepseek_calib_prompt_50.json  --batch_size 16 --trust_remote_code True
    ```

#### DeepSeek-R1 W8A8 Hybrid Quantization + MTP Quantization

 * Generate the W8A8 MTP Quantization Weight of the DeepSeek-R1 Model
    
    ```shell
    python3 quant_deepseek_w8a8.py --model_path ${model_path} --save_path ${save_path} --batch_size 4 --quant_mtp mix --trust_remote_code True
    ```

### DeepSeek-R1-0528 Series

#### DeepSeek-R1-0528 (including the MTP layer) W4A8 per-channel quantization (The routing experts at the non-MTP layer use w4a8 per-channel dynamic quantization, and other linear layers use w8a8 dynamic quantization.)

```shell
msmodelslim quant \
 --model_path ${model_path} \
 --save_path ${save_path} \
 --model_type DeepSeek-R1-0528 \
 --quant_type w4a8 \
 --trust_remote_code True
```

#### DeepSeek-R1-0528 (including the MTP layer) W4A8C8 per-channel quantization

```shell
msmodelslim quant \
 --model_path ${model_path} \
 --save_path ${save_path} \
 --model_type DeepSeek-R1-0528 \
 --quant_type w4a8c8 \
 --trust_remote_code True
```

#### DeepSeek-R1 0528 W8A8 Hybrid Quantization + MTP Quantization

 * Generate DeepSeek-R1 0528 Model W8A8 Hybrid Quantization + MTP Quantization
    
    ```shell
    python3 quant_deepseek_w8a8.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --batch_size 8 \
    --anti_dataset ../common/deepseek_calib_prompt_0528.json \
    --calib_dataset ../common/deepseek_calib_prompt_0528.json \
    --anti_method m4 \
    --quant_mtp mix \
    --rot \
    --trust_remote_code True
    ```

#### DeepSeek-R1 0528 W8A8C8 hybrid quantization + MTP quantization

 * Generate the DeepSeek-R1 0528 model W8A8C8 hybrid quantization + MTP quantization
    
    ```shell
    python3 quant_deepseek_w8a8.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --batch_size 8 \
    --anti_dataset ../common/deepseek_calib_prompt_0528.json \
    --calib_dataset ../common/deepseek_calib_prompt_0528.json \
    --anti_method m4 \
    --quant_mtp mix \
    --rot \
    --fa_quant \
    --trust_remote_code True
    ```

#### DeepSeek-V3/R1 Quantification QA

 * Q: Error This modeling file requires the following packages that were not found in your environment: flash_attn. Run'pip install flash_attn'
 * A: The flash_attn library is missing in the current environment and the Ascend does not support this library. You need to comment out some code in modeling_deepseek.py in the weight folder during running.
 * ![img.png](img.png)    
 * Q: Error if metadata.get("format") not in \["pt", "tf", "flax", "mix"\]: AttributeError: "NoneType" object has no attribute'get' is reported in modeling_utils.py.
 * A: This indicates that the metadata field is missing in the input weight. Therefore, transformers 4.48.2 must be installed.
 * Q: Unknown quantification type, got fp8 - supported types are: \['awq', 'bitsandbytes_4bit', 'bitsandbytes_8bit', 'gptq', 'aqlm', 'quanto', 'eetq', 'hqq', 'fbgemm_fp8'\]
 * A: Currently, transformers do not support loading in the FP8 quantization format. Therefore, you need to delete the following fields from the config.json file in the weight folder:
 * ![img_1.png](img_1.png)    
 * Q: The description file saved after quantization contains 61 layers of information, and the quantization type is float.
 * A: Layer 61 is the MTP layer. By default, the MTP layer is not quantized. Currently, the MTP can be quantified by setting the quant_mtp parameter to mix.
