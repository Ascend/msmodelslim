# Qwen3-VL 量化案例

## 模型介绍

  [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 是Qwen推出的视觉-语言模型，这一代在各个方面都进行了全面升级：更优秀的文本理解和生成、更深入的视觉感知和推理、扩展的上下文长度、增强的空间和视频动态理解能力，以及更强的代理交互能力。

## 环境配置

- 基础环境配置请参考《[msModelSlim工具安装指南](../../../docs/zh/install_guide/install_guide.md)》，注意：由于高版本transformers的特殊性，PyTorch及TorchNPU需要配置安装为≥2.2版本。
- 针对 Qwen3-VL，transformers 版本需要 4.57.1：

  ```bash
  pip install transformers==4.57.1
  ```

## Qwen3-VL模型当前已验证的量化方法

| 模型       | 原始浮点权重 | 量化方式 | 推理框架支持情况| 量化命令 |
|------------|-------------|---------|----------------|---------|
| Qwen3-VL-4B-Instruct | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | W8A8量化 | MindIE 预计3.0.RC1版本支持<br>vLLM Ascend v0.13.0及之后版本支持 | [W8A8量化](#qwen3-vl-4b-w8a8) |
| Qwen3-VL-8B-Instruct | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | W8A8SC量化 | MindIE 预计3.0.RC1版本支持<br>vLLM Ascend 当前不支持 | [W8A8SC量化](#qwen3-vl-w8a8sc) |
| Qwen3-VL-32B-Instruct | [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | W8A8量化 | MindIE 预计3.0.RC1版本支持<br>vLLM Ascend v0.13.0及之后版本支持 | [W8A8量化](#qwen3-vl-w8a8) |

**说明：**
  点击量化命令列中的链接可跳转到对应的具体量化命令。

### 使用案例

- 如果需要使用NPU多卡量化，请先配置多卡环境变量（Atlas 300I Duo 系列产品不支持多卡量化）：

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`，让修改后的自定义代码文件能够正确地被加载（请确保加载的自定义代码文件的安全性）。

#### 1. Qwen3-VL系列

##### <span id="qwen3-vl-w8a8sc">1.1 Qwen3-VL-8B-Instruct W8A8SC量化 异常值抑制算法使用m2</span>

该示例在NPU上生成Qwen3-VL-8B-Instruct模型的量化权重。使用m2算法进行异常值抑制。

请将{浮点权重路径}和{W8A8S量化权重路径}替换为用户实际路径。{校准集图片路径}默认为"../calibImages"，以当前"../calibImages"目录中2张图片作为校准集。部署量化权重时，如在使用场景精度出现明显掉点，用户可根据实际场景替换为其他图片（建议30张）。

Atlas 300I DUO 使用以下方法稀疏量化

- 稀疏量化

  ```shell
  python quant_qwen3vl.py \
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

- 权重压缩

  **注意**：权重压缩需要先安装MindIE，具体可参见[《MindIE安装指南》](https://www.hiascend.com/document/detail/zh/mindie/230/envdeployment/instg/mindie_instg_0001.html)

  ```shell
  # TP数为tensor parallel并行个数
  export IGNORE_INFER_ERROR=1
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
  ```

###### 量化参数说明

| 参数名 | 含义 | 默认值 | 使用方法 |
| ------ | ---- | --- | -------- |
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入原始浮点权重目录路径。 |
| calib_images | 校准集图片路径 | ../calibImages | 可选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[COCO](https://cocodataset.org/#download)。 为保证量化精度需要根据示例扩充到30张图片。用户可根据实际场景替换为其他图片。|
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化权重路径。 |
| part_file_size | 量化权重文件大小，单位是GB | 默认为None时不限制单个权重文件大小，此时只生成一个量化权重文件。 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的大小上限。|
| w_bit | 权重量化bit | 8 | 可选参数;<br>在Qwen3-VL量化场景下支持配置为4或8。|
| a_bit | 激活值量化bit | 8 | 可选参数;<br>在Qwen3-VL量化场景下支持配置为8。|
| device_type | 量化运行设备类型 | 'cpu' | 可选参数;<br>可选值：['cpu', 'npu']。 |
| trust_remote_code | 是否信任自定义代码 | False | 可选参数;<br>指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载（请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险）。 |
| anti_method | 异常值抑制算法 | 'm2' | 可选参数;<br>可选值：['m2']。'm2'对应多模态理解模型场景下优化后的Outlier Suppression Plus异常值抑制算法。 |
| act_method | 激活值量化方法 | 2 | 可选参数;<br>(1) 1代表Label-Free场景的min-max量化方式。<br>(2) 2代表Label-Free场景的histogram量化方式。<br>(3) 3代表Label-Free场景的自动混合量化方式。|
| open_outlier | 是否开启权重异常值划分 | True | 可以配置为True或者False。 <br>设置为True时开启权重异常值划分，反之则关闭。|
| is_dynamic | 是否使用动态量化，即W8A8中的激活量化参数动态生成 | False | 可以配置为True或者False。 <br>设置为True时使用动态量化，反之则不使用。|
| is_lowbit | 是否使用稀疏量化的lowbit算法 | False | 可以配置为True或者False。 <br>设置为True时，表示使用稀疏量化的lowbit算法，反之则不使用。 <br>在`w4a8_dynamic per-group`量化场景下需要设置为True。|
| co_sparse | 是否开启稀疏量化功能 | False | True: 使用稀疏量化功能；<br>False: 不使用稀疏量化功能。 |
| fraction | 模型权重稀疏量化过程中被保护的异常值占比  | 0.01 | 取值范围[0.01,0.1]。|
| use_sigma | 是否启动sigma功能 | False | True: 开启sigma功能；<br>False: 不开启sigma功能。 |
| sigma_factor | sigma功能中sigma的系数 | 3.0 | 数据类型为float，默认值为3.0，取值范围为[1.0, 3.0]。<br>说明：仅当use_sigma为True时生效。 |
| torch_dtype | 设置加载权重的数据类型 | bf16 | 可选值：['bf16', 'fp16']。默认值为bf16。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../../docs/zh/api_reference/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../../docs/zh/api_reference/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)。

##### <span id="qwen3-vl-4b-w8a8">1.2 Qwen3-VL-4B-Instruct W8A8量化</span>

该模型的量化已经集成至[一键量化](../../../docs/zh/user_guide/usage_quick_quantization.md#32-参数说明)。使用 `model_type=Qwen3-VL-4B-Instruct`、`quant_type=w8a8` 即可。若需使用自定义配置（如指定保存选项），可通过 `config_path` 指定 [qwen3_vl_4b_w8a8.yaml](../../../lab_practice/qwen3_vl/qwen3_vl_4b_w8a8.yaml)。

```shell
msmodelslim quant \
    --model_path /path/to/qwen3_vl_4b_float_weights \
    --save_path /path/to/qwen3_vl_4b_quantized_weights \
    --device npu \
    --model_type Qwen3-VL-4B-Instruct \
    --quant_type w8a8 \
    --trust_remote_code True
```

使用自定义配置文件时：

```shell
msmodelslim quant \
    --model_path /path/to/qwen3_vl_4b_float_weights \
    --save_path /path/to/qwen3_vl_4b_quantized_weights \
    --device npu \
    --model_type Qwen3-VL-4B-Instruct \
    --config lab_practice/qwen3_vl/qwen3_vl_4b_w8a8.yaml \
    --trust_remote_code True
```

**说明：**

- Qwen3-VL-4B-Instruct 默认精度为`bfloat16`，若修改模型权重路径下`config.json`中的`torch_dtype`为`float16`进行量化，可能会导致模型精度异常。
- 若硬件只支持float16精度推理（例如Atlas 300I/300T系列），建议采用默认精度`bfloat16`量化后将模型权重路径下`config.json`中的`torch_dtype`修改为`float16`进行推理。

##### <span id="qwen3-vl-w8a8">1.3 Qwen3-VL-32B-Instruct W8A8量化</span>

该模型的量化已经集成至[一键量化](../../../docs/zh/user_guide/usage_quick_quantization.md#32-参数说明)。

```shell
msmodelslim quant \
    --model_path /path/to/qwen3_vl_float_weights \
    --save_path /path/to/qwen3_vl_quantized_weights \
    --device npu \
    --model_type Qwen3-VL-32B-Instruct \
    --quant_type w8a8 \
    --trust_remote_code True
```
