# Qwen2.5-VL 量化说明

## 模型介绍

- [Qwen2.5-VL](https://qwenlm.github.io/zh/blog/qwen2.5-vl/) 是阿里云研发的 Qwen 模型家族的旗舰视觉语言模型，对比此前发布的 Qwen2-VL 实现了巨大的飞跃。Qwen2.5-VL 的主要特点如下所示：
  - 感知更丰富的世界：Qwen2.5-VL 不仅擅长识别常见物体，如花、鸟、鱼和昆虫，还能够分析图像中的文本、图表、图标、图形和布局。
  - Agent：Qwen2.5-VL 直接作为一个视觉 Agent，可以推理并动态地使用工具，初步具备了使用电脑和使用手机的能力。
  - 理解长视频和捕捉事件：Qwen2.5-VL 能够理解超过 1 小时的视频，并且这次它具备了通过精准定位相关视频片段来捕捉事件的新能力。
  - 视觉定位：Qwen2.5-VL 可以通过生成 bounding boxes 或者 points 来准确定位图像中的物体，并能够为坐标和属性提供稳定的 JSON 输出。
  - 结构化输出：对于发票、表单、表格等数据，Qwen2.5-VL 支持其内容的结构化输出，惠及金融、商业等领域的应用。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../docs/zh/install_guide/install_guide.md)。
- 还需要执行以下命令安装qwen_vl_utils依赖
  - pip install qwen_vl_utils
- 针对Qwen2.5-VL，transformers版本需要配置安装为4.49.0
  - pip install transformers==4.49.0

## Qwen2.5-VL模型当前已验证的量化方法

| 模型                    | 原始浮点权重                                                                            | 量化方式      | 推理框架支持情况                                                 | 量化命令                                                          |
| ----------------------- | --------------------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------- |
| Qwen2.5-VL-7B-Instruct  | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)   | W8A8静态量化  | MindIE 2.2.RC1及之后版本支持vLLM Ascend v0.10.2rc2及之后版本支持 | [W8A8静态量化(m2)](#1-qwen25-vl-w8a8静态量化-异常值抑制算法使用m2) |
| Qwen2.5-VL-72B-Instruct | [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/tree/main) | W8A8静态量化  | MindIE 2.2.RC1及之后版本支持vLLM Ascend v0.10.2rc2及之后版本支持 | [W8A8静态量化(m2)](#1-qwen25-vl-w8a8静态量化-异常值抑制算法使用m2) |
| Qwen2.5-VL-7B-Instruct  | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)   | W4A8动态量化  | MindIE当前不支持vLLM Ascend当前不支持                            | [W4A8动态量化](#3-qwen25-vl-w4a8动态量化-异常值抑制算法使用m4)     |
| Qwen2.5-VL-72B-Instruct | [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/tree/main) | W4A8动态量化  | MindIE当前不支持vLLM Ascend当前不支持                            | [W4A8动态量化](#3-qwen25-vl-w4a8动态量化-异常值抑制算法使用m4)     |
| Qwen2.5-VL-32B-Instruct | [Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/tree/main) | W8A8 静态量化 | MindIE 待支持vLLM Ascend 已支持                                  | [W8A8 静态量化](#4-qwen25-vl-32b-instruct-w8a8-静态量化)     |

**说明：**

- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 生成量化权重

- 量化权重统一使用[quant_qwen2_5vl.py](./quant_qwen2_5vl.py)脚本生成，以下提供Qwen2.5-VL模型量化权重生成快速启动命令。

## 使用示例

- 如果需要使用NPU多卡量化(特别是Qwen2.5-VL-72B这种大模型)，请先配置多卡环境变量（Atlas 300I Duo 系列产品不支持多卡量化）：

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`，让修改后的自定义代码文件能够正确地被加载。(请确保加载的自定义代码文件的安全性)

### 1 Qwen2.5-VL W8A8静态量化 异常值抑制算法使用m2

生成Qwen2.5-VL模型量化权重，异常值抑制使用m2算法，在NPU上运行，请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。{校准图片路径}默认为"../calibImages"，以当前"../calibImages"目录中2张图片为例，实际量化时为保证精度需要从COCO数据集中扩充到30张图片。此外，用户可根据实际场景替换为其他图片。

```shell
  # 用于MindIE部署
  python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m2 --mindie_format

  # 用于vLLM Ascend部署
  python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m2
```

### 2 Qwen2.5-VL W8A8静态量化 异常值抑制算法使用m4

生成Qwen2.5-VL模型量化权重，异常值抑制使用m4算法，在NPU上运行，请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。{校准图片路径}默认为"../calibImages"，以当前"../calibImages"目录中2张图片为例，实际量化时为保证精度需要从COCO数据集中扩充到30张图片。此外，用户可根据实际场景替换为其他图片。

```shell
  # 用于MindIE部署
  python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m4 --mindie_format

  # 用于vLLM Ascend部署
  python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m4
```

### 3 Qwen2.5-VL W4A8动态量化 异常值抑制算法使用m4

生成Qwen2.5-VL模型量化权重，使用4bit per-group量化权重，8bit per-token量化激活值，异常值抑制使用m4算法，在NPU上运行，请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。{校准图片路径}默认为"../calibImages"，以当前"../calibImages"目录中2张图片为例，实际量化时为保证精度需要从COCO数据集中扩充到30张图片。此外，用户可根据实际场景替换为其他图片。

```shell
  python quant_qwen2_5vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 4 --a_bit 8 --act_method 1 --device_type npu --trust_remote_code True --anti_method m4 --open_outlier False --is_dynamic True --is_lowbit True --group_size 256
```

### <span id="4-qwen25-vl-32b-instruct-w8a8-静态量化">4 Qwen2.5-VL-32B-Instruct W8A8 静态量化</span>

该模型的量化已经集成至《一键量化》文档中的"[参数说明](../../../docs/zh/user_guide/usage_quick_quantization.md#32-参数说明)"章节内容，具体量化命令参考如下：

```shell
msmodelslim quant \
    --model_path /path/to/qwen2_5_vl_float_weights \
    --save_path /path/to/qwen2_5_vl_quantized_weights \
    --device npu \
    --model_type Qwen2.5-VL-32B-Instruct \
    --quant_type w8a8 \
    --tag vLLM_Ascend Ascend_950 \
    --trust_remote_code True
```

## 附录

### 量化参数说明

| 参数名            | 含义                                                     | 默认值                                                       | 使用方法                                                                                                                                                                                |
| ----------------- | -------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | 浮点权重路径                                             | 无默认值                                                     | 必选参数；输入Qwen2.5-VL原始浮点权重目录路径。                                                                                                                                          |
| calib_images      | 校准集图片路径                                           | ../calibImages                                               | 可选参数；输入校准数据集的目录路径。本示例中图片来源于公开数据集[COCO](https://cocodataset.org/#download)。 为保证量化精度需要根据示例扩充到30张图片。用户可根据实际场景替换为其他图片。 |
| save_directory    | 量化权重路径                                             | 无默认值                                                     | 必选参数；输出量化权重路径。                                                                                                                                                            |
| part_file_size    | 量化权重文件大小，单位是GB                               | 默认为None，不限制单个权重文件大小，只生成一个量化权重文件。 | 可选参数；生成量化权重文件大小，请用户自定义单个量化权重文件的大小上限。                                                                                                                |
| w_bit             | 权重量化bit                                              | 8                                                            | 可选参数;在Qwen2.5-VL量化场景下支持配置为4或8。                                                                                                                                         |
| a_bit             | 激活值量化bit                                            | 8                                                            | 可选参数;在Qwen2.5-VL量化场景下支持配置为8。                                                                                                                                            |
| device_type       | 量化运行设备类型                                         | 'cpu'                                                        | 可选参数;可选值：['cpu', 'npu']。                                                                                                                                                       |
| trust_remote_code | 是否信任自定义代码                                       | False                                                        | 可选参数;指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载(请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险)。                                             |
| anti_method       | 异常值抑制算法                                           | 'm2'                                                         | 可选参数;可选值：['m2', 'm4']。'm2'对应多模态理解模型场景下优化后的Outlier Suppression Plus异常值抑制算法，'m4'对应Iterative Smooth异常值抑制算法。                                     |
| act_method        | 激活值量化方法                                           | 2                                                            | 可选参数;(1) 1代表Label-Free场景的min-max量化方式。(2) 2代表Label-Free场景的histogram量化方式。(3) 3代表Label-Free场景的自动混合量化方式。                                              |
| open_outlier      | 是否开启权重异常值划分                                   | True                                                         | 可以配置为True或者False。设置为True时开启权重异常值划分，反之则关闭。                                                                                                                   |
| is_dynamic        | 是否使用动态量化，即W8A8中的激活量化参数动态生成         | False                                                        | 可以配置为True或者False。设置为True时使用动态量化，反之则不使用。                                                                                                                       |
| is_lowbit         | 是否使用稀疏量化的lowbit算法                             | False                                                        | 可以配置为True或者False。设置为True时，表示使用稀疏量化的lowbit算法，反之则不使用。 在`w4a8_dynamic per-group`量化场景下需要设置为True。                                              |
| group_size        | per-group量化的分组数量                                  | 64                                                           | 设置为64，128，256，512。 在`w4a8_dynamic per-group`量化场景下仅支持256。                                                                                                             |
| mindie_format     | 多模态理解模型量化后的权重配置文件是否兼容MindIE现有版本 | False                                                        | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE当前的版本，不开启`mindie_format`时保存的量化权重可用于vLLM Ascend部署。                                                       |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../../docs/zh/api_reference/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../../docs/zh/api_reference/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)
