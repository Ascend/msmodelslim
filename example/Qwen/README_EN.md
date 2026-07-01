# Qwen Quantification Description

## Model Introduction

 * The Qwen language model is a large-scale language model launched by Alibaba Group. It has powerful natural language processing capabilities and can understand and generate text. It is applied to intelligent customer service, content generation, and Q&A systems to help enterprises upgrade intelligently.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../docs/zh/getting_started/install_guide.md)    .
 * The 4.51.0 version must be installed for the Qwen3 transformers.
    
     * pip install transformers==4.51.0

## Supported Model Versions and Quantification Policies

| Model Series | Model version        | HuggingFace Link                                                                    | W8A8 | W8A16 | W4A16 | W4A4 | W16A16S (floating-point sparse) | sparse quantization | KV Cache | Attention | Quantization command                                                                                                                                                               |
| ------------ | -------------------- | ----------------------------------------------------------------------------------- | ---- | ----- | ----- | ---- | ------------------------------- | ------------------- | -------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Qwen**     | Qwen-7B              | [Qwen-7B](https://huggingface.co/Qwen/Qwen-7B/tree/main)                               | ✅    |       |       |      |                                 |                     |          |           | [W8A8](#qwen-14b-w8a8-quantification)                                                                                                                                                             |
|              | Qwen-14B             | [Qwen-14B](https://huggingface.co/Qwen/Qwen-14B/tree/main)                             | ✅    |       |       |      |                                 |                     |          |           | [W8A8](#qwen-14b-w8a8-quantification)                                                                                                                                                             |
|              | Qwen-72B             | [Qwen-72B](https://huggingface.co/Qwen/Qwen-72B/tree/main)                             |      | ✅     |       |      |                                 |                     |          |           | [W8A16](#qwen-72b-w8a16-quantification)                                                                                                                                                           |
| **Qwen1.5**  | Qwen1.5-14B          | [Qwen1.5-14B](https://huggingface.co/Qwen/Qwen1.5-14B/tree/main)                       | ✅    |       |       |      |                                 | ✅                   |          |           | [W8A8](#qwen15-14b-qwen15-32b-sparse-quantization)    /[sparseness](#qwen15-14b-qwen15-32b-sparse-quantization)                                                                                                     |
|              | Qwen1.5-32B          | [Qwen1.5-32B](https://huggingface.co/Qwen/Qwen1.5-32B/tree/main)                       | ✅    |       |       |      |                                 | ✅                   |          |           | [W8A8](#qwen15-14b-qwen15-32b-w8a8-quantization)    /[sparseness](#qwen15-14b-qwen15-32b-sparse-quantization)                                                                                                     |
|              | Qwen1.5-72B          | [Qwen1.5-72B](https://huggingface.co/Qwen/Qwen1.5-72B/tree/main)                       |      | ✅     |       |      |                                 |                     |          |           | [W8A16](#qwen15-72b-w8a16-quantification)                                                                                                                                                          |
|              | Qwen1.5-110B         | [Qwen1.5-110B](https://huggingface.co/Qwen/Qwen1.5-110B/tree/main)                     |      | ✅     |       |      |                                 |                     |          |           | [W8A16](#qwen15-110b-w8a16-quantification)                                                                                                                                                         |
| **Qwen2**    | Qwen2-7B             | [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B/tree/main)                             | ✅    |       |       |      |                                 | ✅                   |          |           | [W8A8](#qwen2-7b-w8a8-quantification)    /[sparseness](#qwen2-7b-sparse-quantization)                                                                                                                                |
|              | Qwen2-72B            | [Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B/tree/main)                           | ✅    | ✅     |       |      |                                 | ✅                   | ✅        |           | [W8A8](#qwen2-72b-w8a8-quantification)    /[W8A16](#qwen2-72b-w8a16-quantification)    /[sparseness](#qwen2-72b-sparse-quantization)    /[KV Cache](#qwen2-72b-kv-cache-w8a8-quantification)                                                         |
| **Qwen2.5**  | Qwen2.5-7B-Instruct  | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main)       | ✅    |       |       |      |                                 | ✅                   |          |           | [W8A8](#qwen25-7b-qwen25-14b-qwen25-32b-w8a8-quantization)    /[sparseness](#qwen25-coder-7b-sparse-quantization)                                                                                                 |
|              | Qwen2.5-Coder-7B     | [Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B/tree/main)             |      |       |       |      |                                 | ✅                   |          |           | [sparseness](#qwen25-coder-7b-sparse-quantization)                                                                                                                                                   |
|              | Qwen2.5-14B-Instruct | [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main)     | ✅    |       |       |      |                                 | ✅                   |          |           | [W8A8](#qwen25-7b-qwen25-14b-qwen25-32b-w8a8-quantization)    /[sparseness](#qwen25-coder-7b-sparse-quantization)                                                                                                 |
|              | Qwen2.5-32B-Instruct | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/main)     | ✅    |       |       |      |                                 |                     |          |           | [W8A8](#qwen25-7b-qwen25-14b-qwen25-32b-w8a8-quantization)                                                                                                                                      |
|              | Qwen2.5-72B-Instruct | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/tree/main)     | ✅    |       | ✅     |      |                                 |                     | ✅        | ✅         | [W8A8](#qwen25-7b-qwen25-14b-qwen25-32b-w8a8-quantization)    /[Attention](#qwen25-72b-supports-attention-quantization)    /[PDMix+KV Cache int8](#qwen25-72b-w8a8-pdmix-quantizationw8a8-dynamic-quantization-in-the-prefill-phase-and-w8a8-dynamic-quantization-in-the-decode-phase-quantization-with-kv-cache-int8)    /[W4A16](#qwen25-72b-instruct-w4a16-quantification)     |
| **Qwen3**    | Qwen3-8B             | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/main)                             |      |       |       |      |                                 | ✅                   |          |           | [sparseness](#qwen3-8b-sparse-quantization)                                                                                                                                                          |
|              | Qwen3-14B            | [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B/tree/main)                           | ✅    |       |       |      |                                 | ✅                   |          |           | [W8A8](#qwen3-14b-w8a8-quantification)    /[sparseness](#qwen3-14b-sparse-quantization)                                                                                                                              |
|              | Qwen3-32B            | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/tree/main)                           | ✅    |       |       | ✅    | ✅                               | ✅                   | ✅        |           | [W8A8](#qwen3-32b-w8a8-quantification)    /[sparseness](#qwen3-32b-sparse-quantization)    /[W4A4](#qwen3-32b-w4a4-flatquant-dynamic-quantization)    /[W16A16S](#qwen3-32b-w16a16s-floating-point-sparse-quantization)    /[W8A8C8](#qwen3-32b-w8a8c8-quantification)            |
| **QwQ**      | QwQ-32B              | [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B)                                   | ✅    |       |       |      |                                 | ✅                   |          |           | [W8A8](#qwq-32b-w8a8-quantification)    /[sparseness](#qwq-32b-sparse-quantization)                                                                                                                                  |

**Description:**

 * " indicates that the quantization policy has been officially verified by msModelSlim and has complete functions and stable performance. It is recommended that the quantization policy be used preferentially.
 * The space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the Quantization Command column to go to the specific quantization command.
 * The Qwen 1.5-14B/32B/72B model has passed the maintenance period due to the launch of a new version of the Qwen series with stronger capabilities. In the future, the Qwen 1.5-14B/32B/72B model will not be maintained in the Qwen series.

## Quantized weight generation

 * Qwen Series History v0 example can be used[quant_qwen.py](./quant_qwen.py)    Script generation; one-click quantification integrated or provided`lab_practice`It is recommended that you directly use the configuration scenario.`msmodelslim quant`. The following provides the quick start commands for each model.

### Description of W4A4 Flatquant Dynamic Quantization Parameters (w4a4.py)

| Parameter name    | Meaning:                                                                                                                     | Default Value        | How to Use                                                                                                                                                                                                                                                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | Floating-Point Weighted Path                                                                                                 | No default value.    | Mandatory. Enter the Qwen weight directory path.                                                                                                                                                                                                                                                                                       |
| save_directory    | Quantifying Weight Path                                                                                                      | No default value     | Mandatory. Path of the output quantization result.                                                                                                                                                                                                                                                                                     |
| layer_count       | Number of layers when the model is loaded                                                                                    | 0                    | The default value 0 indicates all layers of the quantization model. Used for debugging. The actual number of quantized layers. When this parameter is set to N, the quantization starts from layer 0 to layer (N-1). (If the value is 5, the five layers 0, 1, 2, 3, and 4 are quantized.) Value range: \[0, total number of layers\]. |
| calib_file        | Quantitative calibration data file                                                                                           | ../common/wiki.jsonl | Path of the data file used for calibration. The .jsonl file is supported. Each line in the file should contain the'inputs_pretokenized' field.                                                                                                                                                                                         |
| batch_size        | Batch size at calibration                                                                                                    | 4                    | Batch size used when quantization calibration data is generated. Value range: 1-16                                                                                                                                                                                                                                                     |
| mindie_format     | Whether the weight configuration file after non-multimodal model quantization is compatible with the existing MindIE version | False                | On`mindie_format`The quantization weight format saved in is compatible with MindIE iteration 4 versions earlier than B050.                                                                                                                                                                                                             |
| trust_remote_code | Trust Custom Code                                                                                                            | False                | Designated`trust_remote_code=True`Enable the modified custom code file to be loaded correctly Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.                                                                                                                                     |

## Use Example

 * Replace \{floating-point weight path\} and \{quantization weight path\} with actual paths.
 * If NPU multi-card quantization is required, configure environment variables to support multi-card quantization.
    
    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

 * To load a custom model, call`from_pretrained`To be specified when the function`trust_remote_code=True`The modified custom code file can be loaded correctly. (Ensure the security of the loaded custom code file.)

### 1. Qwen Series

#### Qwen-14b W8A8 Quantification

In the`{浮点权重路径}/modeling_qwen.py`General.`SUPPORT_CUDA = torch.cuda.is_available()`Manually set to`SUPPORT_CUDA = False`; Generate the Qwen-14b model quantization weight. The antioutlier uses the m2 algorithm to configure the weight, uses the min-max quantization mode, and performs the calculation on the NPU.

```shell
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu  --anti_method m2 --act_method 1 --model_type qwen1 --trust_remote_code True
```

#### Qwen-72b W8A16 Quantification

Generate the Qwen-72b model quantization weight. The activation value quantization uses the automatic hybrid quantization mode and is calculated on the CPU.

```shell
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --calib_file ../common/ceval.jsonl --w_bit 8 --a_bit 16 --device_type cpu  --act_method 3 --model_type qwen1 --trust_remote_code True
```

### 2. Qwen1.5 Series

#### Qwen1.5-14b, Qwen1.5-32b W8A8 Quantization

Generate the quantization weight, use the min-max quantization mode, use 50 pieces of BoolQ data in the calibration data set, and perform the calculation on the NPU.

```shell
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
```

#### Qwen1.5-14b, Qwen1.5-32b sparse quantization

Sparse quantization weights are generated using the following instructions

```shell
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --device_type npu --fraction 0.011 --use_sigma True --is_lowbit True --trust_remote_code True
```

#### Qwen1.5-72b W8A16 Quantification

```shell
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径}  --w_bit 8 --a_bit 16 --device_type npu --trust_remote_code True
```

##### Qwen1.5-110b W8A16 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A16量化权重路径} --device npu --model_type Qwen1.5-110B --quant_type w8a16 --trust_remote_code True
```

### 3. Qwen2 Series

#### Qwen2-7b W8A8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen2-7B --quant_type w8a8 --trust_remote_code True
```

##### Qwen2-7b Sparse Quantization

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8S量化权重路径} --device npu --model_type Qwen2-7B --quant_type w8a8s --trust_remote_code True
```

##### Qwen2-72b W8A8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen2-72B --quant_type w8a8 --trust_remote_code True
```

#### Qwen2-72b W8A16 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A16量化权重路径} --device npu --model_type Qwen2-72B --quant_type w8a16 --trust_remote_code True
```

##### Qwen2-72b Sparse Quantization

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8S量化权重路径} --device npu --model_type Qwen2-72B --quant_type w8a8s --trust_remote_code True
```

##### Qwen2-72b KV Cache W8A8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8C8量化权重路径} --device npu --model_type Qwen2-72B --quant_type w8a8c8 --trust_remote_code True
```

### 4. Qwen2.5 Series

#### Qwen2.5-7b, Qwen2.5-14b, Qwen2.5-32b W8A8 Quantization

```shell
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
```

#### Qwen2.5-72B supports attention quantization

 * To be modified`modeling_qwen2.py`The file and the`config.json`File. For details about the configuration method, see.[FA Quantification Instructions](../../docs/zh/quantization_algorithms/quantization_algorithms/fa3_quant.md)    .
 * Compared with W8A8 quantization, additional settings are required.`use_fa_quant`The parameter is True.
    
    ```shell
    python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --anti_method m4 --act_method 1 --use_fa_quant True --trust_remote_code True
    ```

#### Qwen2.5-Coder-7B Sparse Quantization

```shell
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W4A8量化权重路径} --calib_file ../common/humaneval_x.jsonl --w_bit 4 --a_bit 8 --device_type cpu --fraction 0.02 --co_sparse True  --use_sigma True --is_lowbit False --trust_remote_code True
```

#### Qwen2.5-72b W8A8-pdmix Quantization(W8A8 dynamic quantization in the prefill phase and W8A8 dynamic quantization in the decode phase) Quantization with KV cache int8

```shell
python3 quant_qwen_pdmix.py --model_path {浮点权重路径} \
--save_directory {W8A8-pdmix量化权重路径} \
--calib_file ../common/qwen_calib_prompt_72b_pdmix.json  \
--anti_calib_file ../common/qwen_anti_prompt_72b_pdmix.json \
--device_type npu \
--anti_method m6 \
--act_method 2 \
--use_kvcache_quant True \
--pdmix True \
--trust_remote_code True \
--disable_names model.layers.0.mlp.down_proj model.layers.1.mlp.down_proj model.layers.2.mlp.down_proj model.layers.79.mlp.down_proj
```

#### Qwen2.5-72B-Instruct W4A16 Quantification

When the mindie_format parameter is transferred, the quantization weight does not pack int4 into int8. When the mindie_format parameter is not transferred, the quantization weight packs int4 into int8, reducing storage space and loading time during model deployment.

```shell
python quant_qwen.py \
          --model_path {浮点权重路径} \
          --save_directory {量化权重路径} \
          --device_type npu \
          --calib_file ../common/qwen_mix_dataset.json \
          --w_bit 4 \
          --a_bit 16 \
          --is_lowbit True \
          --open_outlier False \
          --group_size 128 \
          --anti_method m3 \
          --trust_remote_code True
```

### 5. Qwen3 Series

#### Qwen3-32B W8A8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen3-32B --quant_type w8a8 --trust_remote_code True
```

#### Qwen3-32B W8A8C8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8C8量化权重路径} --device npu --model_type Qwen3-32B --quant_type w8a8c8 --trust_remote_code True
```

#### Qwen3-32B w8a8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    The inference engine MindIE and vLLM-Ascend support different quantization solutions. Use the`--scenario`Specifies the inference engine.

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen3-32B --quant_type w8a8 --scenario {场景标签} --trust_remote_code True
```

#### Qwen3-32B sparse quantization

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen3-32B --quant_type w8a8s --trust_remote_code True
```

#### Qwen3-32B W16A16S Floating Point Sparse Quantization

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3-32B --quant_type w16a16s --trust_remote_code True
```

#### Qwen3-32b W4A4 Flatquant Dynamic Quantization

```shell
python3 w4a4.py --model_path {浮点权重路径} --save_directory {w4a4量化权重路径} --calib_file ../common/qwen_qwen3_cot_w4a4.json --trust_remote_code True --batch_size 1
```

#### Qwen3-32b W4A4 Dynamic Quantization

```shell
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3-32B --quant_type w4a4 --trust_remote_code True
```

#### Qwen3-14B W8A8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen3-14B --quant_type w8a8 --trust_remote_code True
```

#### Qwen3-14B Sparse Quantization

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md).

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen3-14B --quant_type w8a8s --trust_remote_code True
```

#### Qwen3-8B Sparse Quantization

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md).

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type Qwen3-8B --quant_type w8a8s --trust_remote_code True
```

### QwQ Series

#### QwQ-32b W8A8 Quantification

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md).

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --device npu --model_type QwQ-32B --quant_type w8a8 --trust_remote_code True
```

##### QwQ-32b Sparse Quantization

Quantification of the model has been integrated into[One-click quantization](../../docs/zh/feature_guide/quick_quantization_v1/usage.md)    .

```shell
msmodelslim quant --model_path {浮点权重路径} --save_path {W8A8s量化权重路径} --device npu --model_type QwQ-32B --quant_type w8a8s --trust_remote_code True
```

## Appendixes

### Quantization parameter description

| Parameter name      | Meaning:                                                                                                                     | Default value                                                                                                            | How to Use                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path          | Floating-Point Weighted Path                                                                                                 | No default value                                                                                                         | Mandatory. Enter the Qwen weight directory path.                                                                                                                                                                                                                                                                                                                                   |
| save_directory      | Quantifying Weight Path                                                                                                      | No default value                                                                                                         | Mandatory. Path of the output quantization result.                                                                                                                                                                                                                                                                                                                                 |
| part_file_size      | Size of a weighted file, in GB.                                                                                              | Unlimited                                                                                                                | Optional parameter; File size of the generated quantization weight file. You can customize the maximum size of a single quantization weight file.                                                                                                                                                                                                                                  |
| calib_texts         | Quantization Calibration Data List                                                                                           | None                                                                                                                     | Calibrate the dataset.                                                                                                                                                                                                                                                                                                                                                             |
| calib_file          | Quantitative calibration data file                                                                                           | teacher_qualification.jsonl                                                                                              | JSON file for storing calibration data.                                                                                                                                                                                                                                                                                                                                            |
| w_bit               | Weight quantization bit                                                                                                      | 8                                                                                                                        | In the large model quantization scenario, this parameter can be set to 8 or 16. Set this parameter to 4 in the large model sparse quantization scenario.                                                                                                                                                                                                                           |
| a_bit               | Activated value quantization bit                                                                                             | 8                                                                                                                        | In the large model quantization scenario, this parameter can be set to 8 or 16. Set this parameter to 8 in the large model sparse quantization scenario.                                                                                                                                                                                                                           |
| disable_names       | Name of the quantization layer to be manually rolled back.                                                                   | For models earlier than Qwen2, roll back all c_proj layers. For other models, roll back all down_proj layers by default. | You can manually set this parameter based on the precision requirements. By default, the dimension-reduced projection layer of the hidden layer is rolled back.                                                                                                                                                                                                                    |
| device_type         | Indicates the device type.                                                                                                   | cpu                                                                                                                      | Value range: \['cpu','npu'\].                                                                                                                                                                                                                                                                                                                                                      |
| fraction            | Proportion of Protected Abnormal Values During Model Weight Sparse Quantization                                              | 0.01                                                                                                                     | Value range: \[0.01, 0.1\].                                                                                                                                                                                                                                                                                                                                                        |
| act_method          | activation value quantization method                                                                                         | 1                                                                                                                        | (1) 1 indicates the min-max quantization mode in the Label-Free scenario. (2) 2 indicates the histogram quantization mode in the label-free scenario. (3) 3 indicates the automatic hybrid quantization mode in the label-free scenario. It is recommended in the LLM large model scenario.                                                                                        |
| co_sparse           | Whether to enable the sparse quantization function                                                                           | False                                                                                                                    | True: Use the sparse quantization function. False: The sparse quantization function is not used.                                                                                                                                                                                                                                                                                   |
| anti_method         | Outlier Suppression Parameters                                                                                               | No default value.                                                                                                        | 'm1': indicates the SmoothQuant algorithm. 'm2': enhanced SmoothQuant algorithm. 'm3': indicates the AWQ algorithm. 'm4': smooth optimization algorithm. 'm5': CBQ quantization algorithm. 'm6': Flex smooth quantization algorithm.                                                                                                                                               |
| disable_level       | L Automatic Rollback Level                                                                                                   | L0                                                                                                                       | A configuration example is as follows:'L0': default value, indicating that the rollback is not performed. 'L1': Roll back layer 1. 'L2': Roll back two layers. 'L3': Roll back to Layer 3. 'L4': Roll back four layers. 'L5': Roll back five layers.                                                                                                                               |
| do_smooth           | Whether smooth quantization is used                                                                                          | False                                                                                                                    | True: Use smooth quantization. False: Not use smooth quantization.                                                                                                                                                                                                                                                                                                                 |
| use_sigma           | Whether to enable the sigma function                                                                                         | False                                                                                                                    | True: The sigma function is enabled. False: The sigma function is disabled.                                                                                                                                                                                                                                                                                                        |
| use_reduce_quant    | Whether the weight quantization is lccl all reduce quantization                                                              | False                                                                                                                    | ID used for MindIE inference.                                                                                                                                                                                                                                                                                                                                                      |
| tp_size             | Simulate the number of cards in multi-card quantization.                                                                     | 1                                                                                                                        | Value range: \[1, 2, 4, 8, 16\]. The default value is 1, indicating that the analog multi-card quantization function is disabled. When this parameter is set to 2, 4, 8, or 16, linear at the communication layer simulates multiple cards. Each card uses different scale and offset for quantization.                                                                            |
| sigma_factor        | sigma factor                                                                                                                 | 3.0                                                                                                                      | The value range is \[1.0, 3.0\]. The default value is 3.0.                                                                                                                                                                                                                                                                                                                         |
| is_lowbit           | Indicates whether to enable the low bit quantization function.                                                               | False                                                                                                                    | (1) When w_bit=4 and a_bit=8, a large model sparse quantization scenario is used, indicating that the low-bit sparse quantization function is enabled. (2) In other scenarios, the automatic quantization precision optimization function is enabled for large model quantization. Currently, the automatic precision optimization framework supports W8A8 and W8A16 quantization. |
| w_sym               | Whether the weight quantization is symmetrical                                                                               | True                                                                                                                     | True: Use symmetric quantization. False: Use asymmetric quantization.                                                                                                                                                                                                                                                                                                              |
| use_kvcache_quant   | Specifies whether to use the kvcache quantization function.                                                                  | False                                                                                                                    | True: Use the kvcache quantization function. False: The kvcache quantization function is not used.                                                                                                                                                                                                                                                                                 |
| use_fa_quant        | Whether to use FA3 for quantization                                                                                          | False                                                                                                                    | True: Use the FA3 quantization type. False: The FA3 quantization type is not used.                                                                                                                                                                                                                                                                                                 |
| fa_amp              | Number of layers that can be automatically rolled back                                                                       | 0                                                                                                                        | The data type is int, and the default value is 0. The value must be greater than or equal to 0 and less than or equal to the number of layers in the model. If the number of layers in the model exceeds the maximum number of layers in the model, the rollback layer number is used as the maximum number of layers in the model.                                                |
| open_outlier        | Indicates whether to enable weight abnormal value division.                                                                  | True                                                                                                                     | True: indicates that weight abnormal value classification is enabled. False: Disable weight abnormal value division. Note: (1) This parameter takes effect only when lowbit is set to True. (2) In the per_group quantization scenario, is_lowbit needs to be set to True and open_outlier needs to be set to False.                                                               |
| group_size          | Size of the group in the per_group quantization.                                                                             | 64                                                                                                                       | The default value is 64, and the value can be 32, 64, or 128. Note: This parameter is applicable only to the per_group quantization scenario. You need to set is_lowbit to True and open_outlier to False.                                                                                                                                                                         |
| is_dynamic          | Indicates whether to use the per-token dynamic quantization function.                                                        | False                                                                                                                    | True: Use per-token dynamic quantization. False: Do not use per-token dynamic quantization.                                                                                                                                                                                                                                                                                        |
| input_ids_name      | Specifies the key name corresponding to the input ID in the word segmentation result.                                        | input_ids                                                                                                                | None                                                                                                                                                                                                                                                                                                                                                                               |
| attention_mask_name | Specifies the key name corresponding to the attention mask in the word segmentation result.                                  | attention_mask                                                                                                           | None                                                                                                                                                                                                                                                                                                                                                                               |
| tokenizer_args      | User-defined parameter input when loading the user-defined tokenizer.                                                        | None                                                                                                                     | The value is transferred in dictionary mode.                                                                                                                                                                                                                                                                                                                                       |
| disable_last_linear | Indicates whether to roll back the last linear layer.                                                                        | True                                                                                                                     | True: Roll back the last linear layer. False: The last linear layer is not rolled back.                                                                                                                                                                                                                                                                                            |
| model_name          | Model name. This parameter is optional.                                                                                      | None                                                                                                                     | This parameter controls the abnormal value suppression parameter.                                                                                                                                                                                                                                                                                                                  |
| model_type          | Qwen Model Type                                                                                                              | qwen2                                                                                                                    | If the model earlier than Qwen2 is used, set this parameter to qwen1.                                                                                                                                                                                                                                                                                                              |
| anti_calib_file     | Outlier Suppression Calibration Data File                                                                                    | None                                                                                                                     | The calibration data file path (.json or .jsonl) used for outlier suppression.                                                                                                                                                                                                                                                                                                     |
| disable_threshold   | Automatic Rollback Threshold                                                                                                 | 0                                                                                                                        | If the value is greater than 0, the system automatically selects the layer to be rolled back based on the threshold.                                                                                                                                                                                                                                                               |
| pdmix               | Whether to use the PDMix quantization type                                                                                   | False                                                                                                                    | True: Use the PDMix quantization type. False: The PDMix quantization type is not used.                                                                                                                                                                                                                                                                                             |
| trust_remote_code   | Trust Custom Code                                                                                                            | False                                                                                                                    | Designated`trust_remote_code=True`Enable the modified custom code file to be loaded correctly (Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.)                                                                                                                                                                               |
| layer_count         | Number of layers when the model is loaded                                                                                    | 0                                                                                                                        | The default value 0 indicates all layers of the quantization model. It is used for debugging. The actual number of quantized layers. When this parameter is set to N, the quantization starts from layer 0 to layer (N-1). (If the value is 5, the five layers 0, 1, 2, 3, and 4 are quantized.) Value range: \[0, total number of layers in the model\].                          |
| mindie_format       | Whether the weight configuration file after non-multimodal model quantization is compatible with the existing MindIE version | False                                                                                                                    | On`mindie_format`The quantization weight format saved in is compatible with MindIE 2.1.RC1 and earlier versions.                                                                                                                                                                                                                                                                   |
| w_method            | weight quantification method                                                                                                 | MinMax                                                                                                                   | Value range: \['MinMax', 'GPTQ', 'HQ', 'NF'\].                                                                                                                                                                                                                                                                                                                                     |

 * For more parameter configuration requirements, see the parameters configured during quantization.[QuantConfig](../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)    and quantization parameter configuration class[Calibrator](../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)    
