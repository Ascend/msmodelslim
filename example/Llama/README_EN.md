# LLaMA Quantification Description

## Model Introduction

 * [LLaMA (Large Language Model Meta AI)](https://github.com/facebookresearch/llama/tree/llama_v1)    ,[LLaMA2 (Large Language Model Meta AI 2)](https://github.com/facebookresearch/llama)    And to the[LLaMA3 (Large Language Model Meta AI 3)](https://github.com/meta-llama/llama3)    , is an open and efficient large-scale basic language model released by Meta AI. It provides knowledge, text generation, language translation, language understanding, code writing, and interpretation through natural language interaction.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../docs/zh/getting_started/install_guide.md)    .

## Supported Model Versions and Quantification Policies

| Model Series | Model version         | HuggingFace link                                                      | W8A8 | W8A16 | W4A8 | sparse quantization | KV Cache | Attention | W4A8_DYNAMIC | Quantization command                                                                                                                                                                                                             |
| ------------ | --------------------- | --------------------------------------------------------------------- | ---- | ----- | ---- | ------------------- | -------- | --------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLaMA**    | LLaMA-33B             | [LLaMA-33B](https://github.com/facebookresearch/llama/tree/llama_v1)     |      |       |      | ✅                   |          |           |              | [sparseness](#llama-33b-sparse-quantization-configuration)                                                                                                                                                                                                     |
|              | LLaMA-65B             | [LLaMA-65B](https://github.com/facebookresearch/llama/tree/llama_v1)     |      | ✅     |      |                     |          |           |              | [W8A16](#llama-65b-w8a16-quantification)                                                                                                                                                                                                         |
| **LLaMA2**   | LLaMA2-7B             | [LLaMA2-7B](https://github.com/facebookresearch/llama/tree/v2)           | ✅    |       |      | ✅                   |          |           |              | [W8A8](#llama2-7b13b-w8a8-quantification)    /[sparseness](#llama2-7b13b-sparse-quantization-configuration)                                                                                                                                                                    |
|              | LLaMA2-13B            | [LLaMA2-13B](https://github.com/facebookresearch/llama/tree/v2)          | ✅    |       |      | ✅                   |          |           |              | [W8A8](#llama2-7b13b-w8a8-quantification)    /[sparseness](#llama2-7b13b-sparse-quantization-configuration)                                                                                                                                                                    |
|              | LLaMA2-70B            | [LLaMA2-70B](https://github.com/facebookresearch/llama/tree/v2)          | ✅    | ✅     |      |                     |          |           |              | [W8A8](#llama2-70b-npu-multi-sim-w8a8-quantization)    /[W8A16](#llama2-70b-w8a16-quantification)                                                                                                                                                                       |
| **LLaMA3**   | LLaMA3-70B            | [LLaMA3-70B](https://github.com/meta-llama/llama3)                       |      | ✅     |      |                     |          |           |              | [W8A16](#llama3-70b-w8a16-quantification)                                                                                                                                                                                                        |
| **LLaMA3.1** | LLaMA3.1-8B           | [LLaMA3.1-8B](https://github.com/meta-llama/llama3)                      | ✅    |       |      |                     |          |           |              | [W8A8](#llama31-8b-w8a8-quantification)                                                                                                                                                                                                          |
|              | LLaMA3.1-70B          | [LLaMA3.1-70B](https://github.com/meta-llama/llama3)                     | ✅    |       |      | ✅                   | ✅        | ✅         |              | [W8A8](#llama31-70b-w8a8-quantification)    /[KV Cache](#llama31-70b-w8a8-quantization-with-kv-cache-int8-quantization)    /[Attention](#llama31-70b-w8a8-quantification-with-attention-quantification)    /[PDMix+KV Cache](#llama31-70b-w8a8-pdmix-quantization-w8a8-dynamic-quantization-in-the-prefill-phase-and-w8a8-dynamic-quantization-in-the-decode-phase-kv-cache-int8-quantization)     |
|              | LLaMA3.1-8B-Instruct  | [LLaMA3.1-8B-Instruct](https://github.com/meta-llama/llama3)             |      |       |      |                     |          |           | ✅            | [W4A8_DYNAMIC](#llama31-8b-instruct-w4a8_dynamic-quantization)                                                                                                                                                                                 |
|              | LLaMA3.1-70B-Instruct | [LLaMA3.1-70B-Instruct](https://github.com/meta-llama/llama3)            |      |       |      |                     |          |           | ✅            | [W4A8_DYNAMIC](#llama31-70b-instruct-w4a8_dynamic-quantization)                                                                                                                                                                                |

**Description:**

 * " indicates that the quantification policy has passed the official verification of msModelSlim. The function is complete and the performance is stable. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * You can click the link in the Quantization Command column to go to the specific quantization command.

## Quantized weight generation

 * Use quantitative weights in a unified manner.[quant_llama.py](./quant_llama.py)    Script generation. The following provides the quick start command for generating LLaMA model quantization weights.

### Quantization parameter description

| Parameter name      | meanings                                                                                                                     | Default value                                                                                                                                      | How to Use                                                                                                                                                                                                                                                                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path          | Floating-Point Weighted Path                                                                                                 | No default value                                                                                                                                   | Mandatory. Enter the LLaMA weight directory path.                                                                                                                                                                                                                                                                                                          |
| save_directory      | Quantifying Weight Path                                                                                                      | No default value                                                                                                                                   | Mandatory. Path of the output quantization result.                                                                                                                                                                                                                                                                                                         |
| part_file_size      | Size of the generated quantization weight file, in GB.                                                                       | No default value                                                                                                                                   | Optional parameter; File size of the generated quantization weight file. You need to customize the maximum size of a single quantization weight file.                                                                                                                                                                                                      |
| calib_texts         | Quantified calibration data                                                                                                  | No default value                                                                                                                                   | Optional parameter; Calibration dataset.                                                                                                                                                                                                                                                                                                                   |
| calib_file          | Quantified calibration data                                                                                                  | teacher_qualification.jsonl                                                                                                                        | Optional parameter; JSON file for storing calibration data.                                                                                                                                                                                                                                                                                                |
| w_bit               | Weight quantization bit                                                                                                      | 8                                                                                                                                                  | In the large model quantization scenario, this parameter can be set to 8 or 16. Set this parameter to 4 in the large model sparse quantization scenario.                                                                                                                                                                                                   |
| a_bit               | Activated value quantization bit                                                                                             | 8                                                                                                                                                  | In the large model quantization scenario, this parameter can be set to 8 or 16. Set this parameter to 8 in the large model sparse quantization scenario.                                                                                                                                                                                                   |
| disable_names       | Name of the quantization layer to be manually rolled back                                                                    | W8a8 quantization is rolled back by default to all down_proj layers. LLaMA3 w8a16 quantization is rolled back by default to the first five layers. | You can manually set this parameter based on the precision requirements. By default, the dimension-reduced projection layer of the hidden layer is rolled back.                                                                                                                                                                                            |
| device_type         | Indicates the device type.                                                                                                   | cpu                                                                                                                                                | Value range: \['cpu', 'npu'\].                                                                                                                                                                                                                                                                                                                             |
| fraction            | Proportion of Protected Abnormal Values During Model Weight Sparse Quantization                                              | 0.01                                                                                                                                               | Value range: \[0.01, 0.1\].                                                                                                                                                                                                                                                                                                                                |
| act_method          | activation value quantization method                                                                                         | 1                                                                                                                                                  | (1) 1 indicates the min-max quantization mode in the label-free scenario. (2) 2 indicates the histogram quantization mode in the Label-Free scenario. (3) 3 indicates the automatic hybrid quantization mode in the label-free scenario. It is recommended in the LLM large model scenario.                                                                |
| co_sparse           | Whether to enable the sparse quantization function                                                                           | False                                                                                                                                              | True: Use the sparse quantization function. False: The sparse quantization function is not used.                                                                                                                                                                                                                                                           |
| anti_method         | Outlier Suppression Parameters                                                                                               | No default value                                                                                                                                   | 'm1': indicates the SmoothQuant algorithm. 'm2': enhanced SmoothQuant algorithm. 'm3': indicates the AWQ algorithm. 'm4': smooth optimization algorithm. 'm5': CBQ quantization algorithm. 'm6': Flex smooth quantization algorithm.                                                                                                                       |
| disable_level       | L Automatic Rollback Level                                                                                                   | L0                                                                                                                                                 | A configuration example is as follows:'L0': default value, indicating that the rollback is not performed. 'L1': Roll back layer 1. 'L2': Roll back two layers. 'L3': Roll back to Layer 3. 'L4': Roll back four layers. 'L5': Roll back five layers.                                                                                                       |
| do_smooth           | Enable smooth quantization                                                                                                   | False                                                                                                                                              | True: indicates that smooth quantization is enabled. False: The smooth quantization function is disabled.                                                                                                                                                                                                                                                  |
| use_sigma           | Whether to enable the sigma function                                                                                         | False                                                                                                                                              | True: The sigma function is enabled. False: The sigma function is disabled.                                                                                                                                                                                                                                                                                |
| use_reduce_quant    | Indicates whether the weight is quantized by lccl all reduce.                                                                | False                                                                                                                                              | ID used for MindIE inference.                                                                                                                                                                                                                                                                                                                              |
| tp_size             | Simulate the number of cards in multi-card quantization.                                                                     | 1                                                                                                                                                  | Value range: \[1, 2, 4, 8, 16\]. The default value is 1, indicating that the analog multi-card quantization function is disabled. When this parameter is set to 2, 4, 8, or 16, linear at the communication layer simulates multiple cards, and each card uses different scale and offset for quantization.                                                |
| sigma_factor        | Coefficient of sigma in the sigma function                                                                                   | 3.0                                                                                                                                                | The data type is float, the default value is 3.0, and the value range is \[1.0, 3.0\]. Note: This parameter is valid only when use_sigma is set to True.                                                                                                                                                                                                   |
| is_lowbit           | Indicates whether to enable the low bit quantization function.                                                               | False                                                                                                                                              | (1) When w_bit=4 and a_bit=8, large model sparse quantization is used, indicating that low-bit sparse quantization is enabled. (2) In other scenarios, the automatic quantization precision optimization function is enabled for large model quantization. Currently, the automatic precision optimization framework supports W8A8 and W8A16 quantization. |
| mm_tensor           | Whether to enable the mm_tensor quantization function                                                                        | True                                                                                                                                               | True: Enable the mm_tensor quantization function. False: The mm_tensor quantization function is disabled.                                                                                                                                                                                                                                                  |
| w_sym               | Indicates whether to enable the w_sym quantization function.                                                                 | True                                                                                                                                               | True: The w_sym quantization function is enabled. False: The w_sym quantization function is disabled.                                                                                                                                                                                                                                                      |
| use_kvcache_quant   | Specifies whether to use the kvcache quantization function.                                                                  | False                                                                                                                                              | True: Use the kvcache quantization function. False: The kvcache quantization function is not used.                                                                                                                                                                                                                                                         |
| use_fa_quant        | Whether to use FA3 for quantization                                                                                          | False                                                                                                                                              | True: Use the FA3 quantization type. False: The FA3 quantization type is not used.                                                                                                                                                                                                                                                                         |
| fa_amp              | Number of layers to be automatically rolled back in the FA3 quantization scenario                                            | 0                                                                                                                                                  | The data type is int, and the default value is 0. The value must be greater than or equal to 0 and less than or equal to the number of model layers. If the number of model layers exceeds the maximum number of model layers, the number of rollback layers is the maximum number of model layers.                                                        |
| open_outlier        | Indicates whether to enable weight abnormal value division.                                                                  | True                                                                                                                                               | True: indicates that weight abnormal value classification is enabled. False: Disable weight abnormal value division. Note: (1) This parameter is valid only when lowbit is set to True. (2) In the per_group quantization scenario, is_lowbit needs to be set to True and open_outlier needs to be set to False.                                           |
| group_size          | Size of the group in the per_group quantization.                                                                             | 64                                                                                                                                                 | The default value is 64. The value can be 32, 64, or 128. Note: This parameter applies only to the per_group quantization scenario. Set is_lowbit to True and open_outlier to False.                                                                                                                                                                       |
| is_dynamic          | Indicates whether to use the per-token dynamic quantization function.                                                        | False                                                                                                                                              | True: Use per-token dynamic quantization. False: Do not use per-token dynamic quantization.                                                                                                                                                                                                                                                                |
| input_ids_name      | Specifies the key name corresponding to the input ID in the word segmentation result.                                        | input_ids                                                                                                                                          | None                                                                                                                                                                                                                                                                                                                                                       |
| attention_mask_name | Specifies the key name corresponding to the attention mask in the word segmentation result.                                  | attention_mask                                                                                                                                     | None                                                                                                                                                                                                                                                                                                                                                       |
| tokenizer_args      | User-defined parameter input when loading the user-defined tokenizer.                                                        | None                                                                                                                                               | The value is transferred in dictionary mode.                                                                                                                                                                                                                                                                                                               |
| disable_last_linear | Indicates whether to roll back the last linear layer.                                                                        | True                                                                                                                                               | True: Roll back the last linear layer. False: The last linear layer is not rolled back.                                                                                                                                                                                                                                                                    |
| model_name          | Model name. This parameter is optional.                                                                                      | None                                                                                                                                               | This parameter controls the abnormal value suppression parameter.                                                                                                                                                                                                                                                                                          |
| model_type          | llama model type                                                                                                             | llama2                                                                                                                                             | If the LLaMA3 model is used, set this parameter to`llama3`If the LLaMA3.1 base model (8B/70B) is used, set this parameter to`llama3.1_fp`If the LLaMA3.1 Instruct model is used, set this parameter to`llama3.1_instruct`.                                                                                                                                 |
| anti_calib_file     | Outlier Suppression Calibration Data File                                                                                    | None                                                                                                                                               | Path of the calibration data file used for outlier suppression (.json or .jsonl format).                                                                                                                                                                                                                                                                   |
| disable_threshold   | Automatically select the threshold for the fallback layer.                                                                   | 0                                                                                                                                                  | If the value is greater than 0, the system automatically selects the layer to be rolled back based on the threshold. A larger value results in more layers to roll back.                                                                                                                                                                                   |
| pdmix               | Whether to use the PDMix quantization type                                                                                   | False                                                                                                                                              | True: Use the PDMix quantization type. False: The PDMix quantization type is not used.                                                                                                                                                                                                                                                                     |
| trust_remote_code   | Trust Custom Code                                                                                                            | False                                                                                                                                              | Designated`trust_remote_code=True`Enable the modified custom code file to be loaded correctly Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.                                                                                                                                                         |
| mindie_format       | Whether the weight configuration file after non-multimodal model quantization is compatible with the existing MindIE version | False                                                                                                                                              | True: enabled`mindie_format`The quantization weight format saved in is compatible with MindIE 2.1.RC1 and earlier versions.                                                                                                                                                                                                                                |

 * For more parameter configuration requirements, see the parameters configured during quantization.[QuantConfig](../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)    and quantization parameter configuration class[Calibrator](../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)    

## Use Example

 * Replace \{floating-point weight path\} and \{quantization weight path\} with actual paths.
 * To use NPU multi-card quantization, configure environment variables first to support multi-card quantization.
    
    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

 * To load a custom model, call`from_pretrained`To be specified when the function`trust_remote_code=True`Enable the modified custom code file to be loaded correctly. (Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.)

### LLaMA

#### LLaMA-65B W8A16 Quantification

```shell
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/teacher_qualification.jsonl --w_bit 8 --a_bit 16 --act_method 3 --trust_remote_code True
```

#### LLaMA 33B Sparse Quantization Configuration

```shell
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/boolq.jsonl --act_method 2 --do_smooth True --use_sigma True --is_lowbit True --co_sparse True --w_bit 4 --trust_remote_code True
```

### LLaMA2

#### LLaMA2-7B/13B W8A8 Quantification

 * Generate the llama2-7b quantization weight. The antioutlier uses the m1 algorithm to configure the weight. The min-max quantization mode is used. The calibration data set uses 50 pieces of BoolQ data and is calculated on the CPU.
    
    ```shell
    python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --device_type cpu --disable_level L0 --anti_method m1 --act_method 1 --trust_remote_code True
    ```

 * Generate llama2-13b quantization weights. The antioutlier uses the m2 algorithm to configure the weights. The min-max quantization mode is used. The calibration data set uses 50 pieces of BoolQ data and is calculated on the CPU.
    
    ```shell
    python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type cpu --disable_level L0 --anti_method m2 --act_method 1 --trust_remote_code True
    ```

#### LLaMA2 7B/13B Sparse Quantization Configuration

```shell
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W4A8S量化权重路径} --calib_file ../common/teacher_qualification.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --trust_remote_code True
```

#### LLaMA2-70B NPU Multi-SIM W8A8 Quantization

```shell
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --trust_remote_code True
```

#### LLaMA2-70B W8A16 Quantification

```shell
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/teacher_qualification.jsonl --w_bit 8 --a_bit 16 --act_method 3 --trust_remote_code True
```

### LLaMA3

#### LLaMA3-70B W8A16 Quantification

```shell
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --a_bit 16 --w_sym False --mm_tensor False --anti_method m3 --act_method 3 --model_type llama3 --trust_remote_code True
```

### LLaMA3.1

#### LLaMA3.1-8B W8A8 Quantification

```shell
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L0 --anti_method m1 --act_method 1 --model_type llama3.1_fp --trust_remote_code True
```

#### LLaMA3.1-70B W8A8 Quantification

```shell
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --anti_method m3 --act_method 3 --model_type llama3.1_fp --trust_remote_code True
```

#### LLaMA3.1-70B W8A8 quantization with KV cache int8 quantization

```shell
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --anti_method m3 --act_method 3 --model_type llama3.1_fp --use_kvcache_quant True --trust_remote_code True
```

#### LLaMA3.1-70B W8A8 Quantification with Attention Quantification

 * Currently, quantization weights can be generated only based on BF-16 weights.
 * To be modified`modeling_llama.py`The file and the`config.json`File. For details about the configuration method, see.[FA Quantification Instructions](../../docs/zh/quantization_algorithms/quantization_algorithms/fa3_quant.md)    .
 * Compared with W8A8 quantization, additional settings are required.`use_fa_quant`The parameter is True.
    
    ```shell
    python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --disable_level L5 --anti_method m4 --act_method 3 --model_type llama3.1_fp --use_fa_quant True --trust_remote_code True
    ```

#### LLaMA3.1-70B W8A8-pdmix quantization (W8A8 dynamic quantization in the prefill phase and W8A8 dynamic quantization in the decode phase) KV cache int8 quantization

```shell
python3 quant_llama.py --model_path {浮点权重路径} \
--save_directory {W8A8-pdmix量化权重路径} \
--calib_file ../common/llama_calib_prompt.jsonl  \
--anti_calib_file ../common/llama_anti_prompt.jsonl \
--device_type npu \
--anti_method m6 \
--act_method 3 \
--model_type llama3.1_fp \
--use_kvcache_quant True \
--disable_threshold 1 \
--pdmix True \
--trust_remote_code True
```

#### LLaMA3.1-8B-Instruct W4A8_DYNAMIC Quantization

```shell
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {量化权重路径} --w_bit 4 --device_type npu --anti_method m3 --act_method 1 --model_type llama3.1_instruct --is_lowbit True --mm_tensor False --open_outlier False --group_size 32 --is_dynamic True --anti_calib_file ../common/llama_anti_prompt.json --trust_remote_code True
```

#### LLaMA3.1-70B-Instruct W4A8_DYNAMIC Quantization

```shell
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {量化权重路径} --w_bit 4 --device_type npu --anti_method m3 --act_method 1 --model_type llama3.1_instruct --is_lowbit True --mm_tensor False --open_outlier False --group_size 32 --is_dynamic True --anti_calib_file ../common/llama_anti_prompt.json --trust_remote_code True
```
