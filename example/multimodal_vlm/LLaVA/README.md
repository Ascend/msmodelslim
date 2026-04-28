# LLaVA 量化说明

## 模型介绍

- [LLaVA（Large Language and Vision Assistant）](https://github.com/haotian-liu/LLaVA)是一个多模态大模型，由威斯康星大学麦迪逊分校、微软研究院和哥伦比亚大学研究者共同发布。能完成图像描述、视觉问答、图像查询、根据图片写代码等任务，还能用于多模态聊天、科学问答，帮助理解图像内容并生成相应的自然语言文本。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../docs/zh/getting_started/install_guide.md)。
- transformers版本需要配置安装为4.37.2

  ```bash
  pip install transformers==4.37.2
  ```

## LLaVA模型当前已验证的量化方法

| 模型       | 原始浮点权重 | 量化方式 | 推理框架支持情况| 量化命令 |
|------------|-------------|---------|----------------|---------|
| LLaVA-v1.5-7B | [llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/a272c74b2481d8aff3aa6fc2c4bf891fe57334fb) | W8A8静态量化 | MindIE当前不支持<br>vLLM Ascend当前不支持 | [W8A8静态量化](#llava-v1-5-7b-w8a8) |

**说明：**

- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 生成量化权重

- 量化权重统一使用[quant_llava.py](./quant_llava.py)脚本生成，以下提供LLaVA模型量化权重生成快速启动命令。

## 使用示例

- 如果需要使用NPU多卡量化，请先配置环境变量以支持多卡量化（Atlas 300I Duo 系列产品不支持多卡量化）：

  ```shell
  # 根据实际情况选择多卡，以下2卡量化为例：
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`，让修改后的自定义代码文件能够正确地被加载。(请确保加载的自定义代码文件的安全性)
  
### 1. LLaVA-v1.5-7B<a id="llava-v1-5-7b-w8a8"></a>

#### <span id="llava-v1-5-7b-w8a8">LLaVA-v1.5-7B W8A8静态量化</span>

生成LLaVA-v1.5-7B模型W8A8量化权重，异常值抑制使用m2算法，在NPU上运行，请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。{校准图片路径}默认为"../calibImages"，用户可根据实际场景替换为其他图片。

  ```shell
  python quant_llava.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --mindie_format
  ```

## 附录

### 量化参数说明

| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入LLaVA原始浮点权重目录路径。|
| calib_images | 校准集图片路径 | ../calibImages | 可选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[COCO](https://cocodataset.org/#download)。 示例选取其中2张图片。用户可根据实际场景替换为其他图片。|
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化权重路径。|
| part_file_size | 量化权重文件大小，单位是GB | 默认为None，不限制单个权重文件大小，只生成一个量化权重文件。| 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的大小上限。|
| w_bit | 权重量化bit | 8 | 可选参数;<br>在LLaVA量化场景下支持配置为8。|
| a_bit | 激活值量化bit | 8 | 可选参数;<br>在LLaVA量化场景下支持配置为8。|
| device_type | 量化运行设备类型 | 'npu' | 可选参数;<br>可选值：['cpu', 'npu']。 |
| trust_remote_code | 是否信任自定义代码 | False | 可选参数;<br>指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载(请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险)。 |
| mindie_format | 多模态理解模型量化后的权重配置文件是否兼容MindIE现有版本 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE当前的版本。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../../docs/zh/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)
  