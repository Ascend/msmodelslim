# Qwen2-VL 量化说明

## 模型介绍

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) 是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM）的新一代。Qwen2-VL 基于 Qwen2 打造，相比 Qwen-VL，它具有以下特点：
    - 读懂不同分辨率和不同长宽比的图片：Qwen2-VL 在 MathVista、DocVQA、RealWorldQA、MTVQA 等视觉理解基准测试中取得了全球领先的表现。
    - Qwen2-VL 可以理解时长超过20分钟的长视频，并支持基于视频内容的问答、对话及内容创作等应用。
    - 能够操作手机和机器人的视觉智能体：借助复杂推理和决策的能力，Qwen2-VL 可集成到手机、机器人等设备，根据视觉环境和文字指令进行自动操作。
    - 多语言支持：为了服务全球用户，除英语和中文外，Qwen2-VL 现在还支持理解图像中的多语言文本，包括大多数欧洲语言、日语、韩语、阿拉伯语、越南语等。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../docs/zh/install_guide/install_guide.md)。
- transformers版本需要配置安装为4.46.0

  ```bash
  pip install transformers==4.46.0
  ```

- 另需安装其他依赖包：
    - pip install qwen_vl_utils

## Qwen2-VL模型当前已验证的量化方法

| 模型       | 原始浮点权重 | 量化方式 | 推理框架支持情况| 量化命令 |
|------------|-------------|---------|----------------|---------|
| Qwen2-VL-7B-Instruct | [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/tree/main) | W8A8静态量化 | MindIE 2.1.RC1及之后版本支持<br>vLLM Ascend当前不支持 | [W8A8静态量化](#1-qwen2-vl系列) |
| Qwen2-VL-72B-Instruct | [Qwen2-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct/tree/main) | W8A8静态量化 | MindIE 2.1.RC1及之后版本支持<br>vLLM Ascend当前不支持 | [W8A8静态量化](#1-qwen2-vl系列) |

**说明：**

- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 生成量化权重

- 量化权重统一使用[quant_qwen2vl.py](./quant_qwen2vl.py)脚本生成，以下提供Qwen2-VL模型量化权重生成快速启动命令。

## 使用示例

- 如果需要使用NPU多卡量化(特别是Qwen2-VL-72B这种大模型)，请先配置多卡环境变量（昇腾310P系列产品不支持多卡量化）：

  ```shell
  # 根据实际情况选择多卡，以下8卡量化为例：
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`，让修改后的自定义代码文件能够正确地被加载。(请确保加载的自定义代码文件的安全性)

### 1. Qwen2-VL系列

#### Qwen2-VL W8A8静态量化，使用异常值抑制m2算法

生成Qwen2-VL模型量化权重，在NPU上运行，请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。{校准图片路径}默认为"../calibImages"，以当前"../calibImages"目录中2张图片为例，实际量化时为保证精度需要从COCO数据集中扩充到30张图片。此外，用户可根据实际场景替换为其他图片。

  ```shell
  python quant_qwen2vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m2 --mindie_format
  ```

#### Qwen2-VL W8A8静态量化，使用异常值抑制m4算法

生成Qwen2-VL模型量化权重，在NPU上运行，请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。{校准图片路径}默认为"../calibImages"，以当前"../calibImages"目录中2张图片为例，实际量化时为保证精度需要从COCO数据集中扩充到30张图片。此外，用户可根据实际场景替换为其他图片。

  ```shell
  python quant_qwen2vl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --anti_method m4 --mindie_format
  ```

## 附录

### 量化参数说明

| 参数名 | 含义 | 默认值 | 使用方法 |
| ------ | ---- | --- | -------- |
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入Qwen2-VL原始浮点权重目录路径。 |
| calib_images | 校准集图片路径 | ../calibImages | 可选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[COCO](https://cocodataset.org/#download)。 为保证量化精度需要根据示例扩充到30张图片。用户可根据实际场景替换为其他图片。|
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化权重路径。 |
| part_file_size | 量化权重文件大小，单位是GB | 默认为None，不限制单个权重文件大小，只生成一个量化权重文件。 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的大小上限。|
| w_bit | 权重量化bit | 8 | 可选参数;<br>在Qwen2-VL量化场景下支持配置为8。|
| a_bit | 激活值量化bit | 8 | 可选参数;<br>在Qwen2-VL量化场景下支持配置为8。|
| device_type | 量化运行设备类型 | 'npu' | 可选参数;<br>可选值：['cpu', 'npu']。 |
| trust_remote_code | 是否信任自定义代码 | False | 可选参数;<br>指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载(请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险)。 |
| anti_method | 异常值抑制算法 | 'm2' | 可选参数;<br>可选值：['m2', 'm4']。'm2'对应多模态理解模型场景下优化后的Outlier Suppression Plus异常值抑制算法，'m4'对应Iterative Smooth异常值抑制算法。 |
| mindie_format | 多模态理解模型量化后的权重配置文件是否兼容MindIE现有版本 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE当前的版本。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../../docs/zh/api_reference/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../../docs/zh/api_reference/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)
