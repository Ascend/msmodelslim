# InternVL 2.0 量化说明

## 模型介绍

- [InternVL 2.0](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/) 是由上海人工智能实验室联合商汤科技推出的一款多模态大模型。其升级版本 InternVL 2.0 在多项关键评测指标上已达到国际顶尖商用闭源模型的水平。

书生·万象支持图像、视频、文本、语音、三维、医疗等多种模态，能完成百余种下游任务，性能可与专用任务模型媲美。在处理复杂的多模态数据时，该模型展现出卓越的能力，尤其在数学、科学图表、通用图表、文档解析、信息图表及OCR等任务中表现尤为突出。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../docs/zh/install_guide/install_guide.md)。
- transformers版本需要配置安装为4.46.0

  ```bash
  pip install transformers==4.46.0
  ```

- 另需安装其他依赖包：

  ```bash
  pip install timm fastchat
  ```

## InternVL 2.0模型当前已验证的量化方法

| 模型       | 原始浮点权重 | 量化方式 | 推理框架支持情况| 量化命令 |
|------------|-------------|---------|----------------|---------|
| InternVL2-8B | [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main) | W8A8静态量化 | MindIE当前不支持<br>vLLM Ascend当前不支持 | [W8A8静态量化](#internvl2-8b-w8a8静态量化) |
| InternVL2-40B | [InternVL2-40B](https://huggingface.co/OpenGVLab/InternVL2-40B/tree/main) | W8A8静态量化 | MindIE当前不支持<br>vLLM Ascend当前不支持 | [W8A8静态量化](#internvl2-40b-w8a8静态量化) |

**说明：**

- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 生成量化权重

- 量化权重统一使用[quant_internvl2.py](./quant_internvl2.py)脚本生成，以下提供InternVL 2.0模型量化权重生成快速启动命令。

## 使用示例

- 如果需要使用NPU多卡量化(特别是InternVL2-40B这种大模型)，请先配置多卡环境变量（昇腾310P系列产品不支持多卡量化）：

  ```shell
  # 根据实际情况选择多卡，以下8卡量化为例：
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`，让修改后的自定义代码文件能够正确地被加载。(请确保加载的自定义代码文件的安全性)

### 1. InternVL 2.0系列

#### InternVL2-8B W8A8静态量化

生成InternVL2-8B模型量化权重，异常值抑制使用m2算法（当前仅支持m2），在NPU上运行，请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。{校准图片路径}在示例中默认为"./textvqa_val"，需要手动下载对应的[textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)数据集。

  ```shell
  python quant_internvl2.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --is_8B_model --trust_remote_code True --mindie_format
  ```

#### InternVL2-40B W8A8静态量化

生成InternVL2-40B模型量化权重，异常值抑制使用m2算法（当前仅支持m2），在NPU上运行，请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。{校准图片路径}在示例中默认为"./textvqa_val"，需要手动下载对应的[textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)数据集。

  ```shell
  python quant_internvl2.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --mindie_format
  ```

  ```python
  # 若在32G显存的NPU上运行模型时，因显存分配不均导致出现显存不足（Out of memory）错误，可通过在模型加载时设置显存限制（使用max_memory参数）来优化显存使用，示例代码如下：
  model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        use_safetensors=True,
        trust_remote_code=True,
        max_memory={0: "20GB", 1: "20GB", 2: "20GB", 3: "20GB", 4: "20GB", 5: "20GB", 6: "20GB", 7: "20GB", "cpu": "20GB"}).eval()
```

## 附录

### 量化参数说明

| 参数名 | 含义 | 默认值 | 使用方法 |
| ------ | ---- | --- | -------- |
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入InternVL 2.0原始浮点权重目录路径。 |
| calib_images | 校准集图片路径 | ./textvqa_val | 必选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)。当前示例仅支持该校准数据集。|
| calib_num     | 从校准数据中随机选择的数量 | 30 | 可选参数；<br>根据需要从校准集中选择一定数量的数据用于校准。建议选取30个数据。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化权重路径。 |
| part_file_size | 量化权重文件大小，单位是GB | 默认为None，不限制单个权重文件大小，只生成一个量化权重文件。 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的大小上限。|
| w_bit | 权重量化bit | 8 | 可选参数;<br>在InternVL 2.0量化场景下支持配置为8。|
| a_bit | 激活值量化bit | 8 | 可选参数;<br>在InternVL 2.0量化场景下支持配置为8。|
| act_method | 激活值量化方法 | 1 | 可选参数;<br>(1) 1代表Label-Free场景的min-max量化方式。<br>(2) 2代表Label-Free场景的histogram量化方式。<br>(3) 3代表Label-Free场景的自动混合量化方式。|
| device_type | 量化运行设备类型 | 'npu' | 可选参数;<br>可选值：['cpu', 'npu']。 |
| is_8B_model      | 是否使用8B的模型  | 不开启  | 可选参数；<br>根据需要选择使用8B大小模型或40B大小模型，开启即指定8B大小模型。 |
| trust_remote_code | 是否信任自定义代码 | False | 可选参数;<br>指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载(请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险)。 |
| mindie_format | 多模态理解模型量化后的权重配置文件是否兼容MindIE现有版本 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE当前的版本。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../../docs/zh/api_reference/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../../docs/zh/api_reference/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)
