# 传统模型量化与校准

本文聚焦传统模型场景，包含 PyTorch/ONNX/MindSpore 训练后量化与量化感知训练。

## 1. 训练后量化（PyTorch）

### 1.1 简介

训练后量化工具需要用户提供PyTorch训练脚本或者pth文件，工具可自动对模型中的卷积和线性层（torch.nn.Linear和torch.nn.Conv2d）进行识别并量化，最终导出量化后的ONNX模型，量化后的模型可以在推理服务器上运行，达到提升推理性能的目的。量化过程中用户需自行提供模型与数据集，调用API接口完成模型的量化调优。

### 1.2 功能介绍

#### 1.2.1 自动混合精度量化算法

为了提升量化精度，训练后量化（PyTorch）算法内置了自动混合精度的模块，自动识别并回退量化敏感层为浮点计算，避免量化敏感层对精度造成较大损失。算法核心是：计算每个量化层量化前后输出的MSE，根据MSE的排序来衡量每一个量化层的量化敏感性，自动回退MSE最大的部分敏感层，从而提升量化的精度。

### 1.2.2 精度保持策略

为了进一步降低量化精度损失，训练后量化（PyTorch）工具内集成了多种精度保持策略，对权重的量化参数和取整方式进行优化。

- Easy Quant权重优化方法：利用输出相似性优化量化参数，减少输入输出张量的量化误差，推荐在data-free模式下使用，通常能够起到较好的改善效果。
- ADMM权重优化方法：使用交替优化的方法，对权重的量化参数进行迭代更新优化，推荐在label-free模式下使用，适当改善量化效果。
- Rounding取整优化：在量化中普通取整不是最优解，使用自适应取整的方式优化权重的取整能提高量化精度，推荐在label-free模式下使用，适当改善量化效果。

### 1.2.3 调用示例

```python
import torchvision

from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator

if __name__ == '__main__':
    MODEL_ARCH = "resnet50"
    SAVE_PATH = "./output"
    INPUTS_NAMES = ["input.1"]

    model = torchvision.models.resnet50(pretrained=True)
    model.eval()

    disable_names = []
    input_shape = [1, 3, 224, 224]
    keep_acc = {'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}

    quant_config = QuantConfig(
        disable_names=disable_names,  # 手动回退的量化层名称，要求格式list[str]，如精度太差，推荐回退量化敏感层，如分类层、输入层、检测head层等
        amp_num=0,  # 混合精度量化回退层数，要求格式int；默认为0
        input_shape=input_shape,  # 模型输入的shape，用于data-free量化构造虚拟数据
        keep_acc=keep_acc,  # 精度保持策略
        sigma=25,  # 大于0使用sigma统计方法；传入0值使用min-max统计方法。
    )

    calibrator = Calibrator(model, quant_config)

    calibrator.run()  # 执行量化算法

    calibrator.export_quant_onnx("resnet50", "./output", ["input.1"])  # 用来导出昇腾可部署的量化ONNX模型

```

### 1.2.4 多模态量化场景

说明
多模态量化，当前硬件限定 Atlas 800I A2 / 800T A2 / 900 A2, 当前量化已经支持但不仅限于SD3和opensora1.2。
多模态量化场景导入代码样例：

```python
import torch
from diffusers import StableDiffusion3Pipeline

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.pytorch.quant.ptq_tools import Calibrator, QuantConfig

pipe = StableDiffusion3Pipeline.from_pretrained(
    "/stable-diffusion-3-medium-diffusers/",
    torch_dtype=torch.float16,
    local_files_only=True
    ).to("npu") #模型路径
pipe.set_progress_bar_config(disable=True)
base = pipe
model = pipe.transformer

calib_dataset = safe_torch_load("sd3_calib_data_v3.pth", map_location="npu")
quant_config = QuantConfig(
    w_bit=8,
    a_bit=8,
    w_signed=True,
    a_signed=True,
    w_sym=True,
    a_sym=False,
    act_quant=True,
    act_method=1,
    quant_mode=1,
    disable_names=None,
    amp_num=0,
    keep_acc=None,
    sigma=25,
    device="npu" #设置模型运行device
)
calibrator = Calibrator(model, quant_config, calib_dataset)
calibrator.run()
calibrator.export_quant_safetensor("/output_path/")
```

### 1.2.5 校准数据获取方式

在上面的示例中，校准数据为 sd3_calib_data_v3.pth，其获取方式如下：
加载 SD3 预训练模型 --> 添加 Listener 类用于捕捉模型输入参数 --> 配置 calib_prompts --> 遍历 calib_prompts，输入 Listener 类中执行前向推理（num_inference_steps用于配置一个prompt生成多少个数据）--> 保存校准数据

代码参考如下：

```python
import torch
from diffusers import StableDiffusion3Pipeline
from ascend_utils.common.security import SafeWriteUmask

calib_data = []

class Listener(torch.nn.Module):
    def __init__(self, module):
        super(Listener, self).__init__()
        self.module = module
        self.inputs = []

    def forward(self, *args, **kwargs):
        sample = {}
        for k in kwargs:
            if isinstance(kwargs[k], torch.Tensor):
                sample[k] = kwargs[k].cpu()
            else:
                sample[k] = kwargs[k]
        self.inputs.append(sample)
        return self.module(*args, **kwargs)

pipe = StableDiffusion3Pipeline.from_pretrained(
    "path_to_stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    local_files_only=True
    )
pipe.to("npu")  # 使用gpu时则为pipe.to("cuda")

pipe.transformer = Listener(pipe.transformer)

# 校准prompts，根据需要配置多条
calib_prompts = ['a photo of a cat holding a sign that says hello world']

for prompt in calib_prompts:
    image = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=28,  # 此处配置为28，则一个prompt将会生成28条校准数据
        height=1024,
        width=1024,
        guidance_scale=7.0,
    ).images[0]

calib_data = pipe.transformer.inputs

with SafeWriteUmask(umask=0o377):
    torch.save(calib_data, "path_to_save/sd3_calib_data.pth")
```

## 2. 训练后量化（ONNX）

### 2.1 简介

当前训练后量化工具自动对ONNX模型中的卷积（Conv）和矩阵乘法（Gemm）进行识别和量化，并将量化后的模型保存为.onnx文件，量化后的模型可以在推理服务器上运行，达到提升推理性能的目的。量化过程中用户需自行提供模型与数据集，调用API接口完成模型的量化调优。

ONNX模型的量化可以采用不同的模式，包括Label-Free和Data-Free模式。这些模式在量化过程中是否需要数据集以及如何使用数据集方面有所不同。msModelSlim工具提供的squant_ptq接口和post_training_quant接口支持这两种量化模式，并且都可以处理静态和动态shape模型。

- Data-Free模式
    Data-Free模式则不需要数据集来进行量化校准。这种模式通常使用模型本身的统计信息或其他无需实际数据的技巧来估计量化参数。Data-Free模式的优势在于，它可以用于那些难以获取或无法获取真实数据的场景。当前以Data-Free模式（以squant_ptq接口为例）（无需校准数据集）为例演示量化步骤。

- Label-Free模式
    在Label-Free模式下，量化过程需要少量的数据集来校准量化因子。这种模式允许量化工具根据实际数据分布调整量化参数，从而提高量化后的模型精度。当前以Label-Free模式（以post_training_quant接口为例）为例演示量化步骤。

### 2.2 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../install_guide/install_guide.md)。

注意：当前 ONNX 量化功能暂不支持Python 3.12 及以上版本。 若需使用 ONNX 量化功能，请确保所用 Python 版本低于 3.12。

训练后量化前须执行命令安装依赖。

### 2.3 功能介绍

#### 2.3.1 Data-Free模式（以squant_ptq接口为例）

本节将以模型静态shape、动态shape、图优化场景分别介绍量化配置步骤，指导用户调用Python API接口对模型进行Data-Free模式的识别和量化，并将量化后的模型保存为.onnx文件，量化后的模型可以在推理服务器上运行。

功能实现流程
用户需准备好ONNX模型，调用squant_ptq接口生成量化配置脚本，运行脚本输出量化后的ONNX模型，并自行转换后进行推理。

图1 squant_ptq接口功能实现流程

![squant_ptq接口功能实现流程](./figures/[onnx]squant_ptq_api_implementation.png)

关键步骤说明如下：

用户准备onnx原始模型，使用QuantConfig配置量化参数，可基于如下场景进行配置。

静态/动态shape模型量化：用户基于量化要求进行配置，可以根据实际情况配置精度保持策略。动态shape场景下，需要手动开启is_dynamic_shape参数，并配置模型的input_shape。

图优化：针对静态shape模型，量化工具内置了多种图结构优化方法，支持对浮点模型和量化后模型进行图优化。使用graph_optimize_level参数开启并指定图优化级别，并支持通过
shut_down_structures参数指定需关闭优化的图结构。同时，在图优化过程中需要将ONNX模型转换为OM模型，用户可以通过om_method参数指定转换工具。

根据ONNX模型和调用OnnxCalibrator封装量化算法，可以根据模型量化情况配置精度保持策略。

初始化OnnxCalibrator后通过run()函数执行量化。

调用export_quant_onnx保存量化后的模型。

模型转换。

参考《ATC工具使用指南》或使用其他转换工具，将ONNX模型转换为OM模型，并进行推理。

如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install onnx --user。

```bash
pip3 install numpy              #Python版本为3.7.5至3.8时（包含3.7.5版本），numpy版本大于等于1.21.6；Python版本为3.8及以上时， numpy版本需大于等于1.23.0
pip3 install onnx               #部分版本存在安全漏洞，建议使用 >=1.16.2版本
pip3 install onnxruntime        #需大于等于1.14.1版本
pip3 install torch==2.1.0       #支持2.1.0，须为CPU版本的torch
pip3 install onnx-simplifier    #需大于等于0.3.10版本
```

静态shape模型量化步骤（以ResNet50为例）

用户需自行准备模型，本样例以ResNet50为例，参考对应[README](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/README.md#section741711594517)中“模型推理”章节，导出onnx文件。

新建模型的量化脚本resnet50_quant.py，编辑resnet50_quant.py文件，导入如下样例代码。

```python
from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig  # 导入squant_ptq量化接口
from msmodelslim import set_logger_level  # 可选，导入日志配置接口
set_logger_level("info")  # 可选，调整日志输出等级，配置为info时，启动量化任务后将打屏显示量化调优的日志信息

config = QuantConfig()   # 使用QuantConfig接口，配置量化参数，并返回量化配置实例，当前示例使用默认配置
input_model_path = "./resnet50_official.onnx"  # 配置待量化模型的输入路径，请根据实际路径配置
output_model_path = "./resnet50_official_quant.onnx"  # 配置量化后模型的名称及输出路径，请根据实际路径配置
calib = OnnxCalibrator(input_model_path, config)   # 使用OnnxCalibrator接口，输入待量化模型路径，量化配置数据，生成calib量化任务实例，其中calib_data为可选配置，可参考精度保持策略的方式三输入真实的数据
calib.run()   # 执行量化
calib.export_quant_onnx(output_model_path)  # 导出量化后模型
```

启动模型量化调优任务，并在指定的输出目录获取一个量化完成的模型。

```bash
python3 resnet50_quant.py
```

量化后的ONNX模型可参考[README](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/README.md#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)中“模型推理”章节，将ONNX模型转换为OM模型，并进行精度验证。若精度损失超过预期，可参考精度保持策略减少精度损失。

动态shape模型量化步骤（以YoloV5m为例）

用户需自行准备模型。以YoloV5m为例，可参考对应[README](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/README.md#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)中“模型推理”章节，获取6.1版本YoloV5m模型的权重文件后，并配置模型推理方式为nms_script，导出动态shape的onnx文件，导出命令参考如下：

```bash
bash pth2onnx.sh --tag 6.1 --model yolov5m --nms_mode nms_script
```

新建模型的量化脚本yolov5m_quant.py，编辑yolov5m_quant.py文件，导入如下样例代码。

```python
from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig  # 导入squant_ptq量化接口
from msmodelslim import set_logger_level  # 可选，导入日志配置接口
set_logger_level("info")  # 可选，调整日志输出等级，配置为info时，启动量化任务后将打屏显示量化调优的日志信息

config = QuantConfig(is_dynamic_shape = True, input_shape = [[1,3,640,640]])  # 使用QuantConfig接口，配置量化参数，返回量化配置实例，其中is_dynamic_shape和input_shape参数在动态shape场景下必须配置，其余参数使用默认配置
input_model_path = "./yolov5m.onnx"  # 配置待量化模型的输入路径，请根据实际路径配置
output_model_path = "./yolov5m_quant.onnx"  # 配置量化后模型的名称及输出路径，请根据实际路径配置
calib = OnnxCalibrator(input_model_path, config)   # 使用OnnxCalibrator接口，输入待量化模型路径，量化配置数据，生成calib量化任务实例，其中calib_data为可选配置，可参考精度保持策略的方式三输入真实的数据
calib.run()   # 执行量化
calib.export_quant_onnx(output_model_path)  # 导出量化后模型
```

启动模型量化调优任务，并在指定的输出目录获取一个量化完成的模型。

```bash
python3 yolov5m_quant.py
```

量化后的ONNX模型可参考[README](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/README.md#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)中“模型推理”章节，将ONNX模型转换为OM模型，并进行精度验证。若精度损失超过预期，可参考精度保持策略减少精度损失。

精度保持策略

为了进一步降低量化精度损失，Data-Free模式下集成了多种精度保持方式，具体如下：

方式一（推荐）：如果量化精度不达标，可使用精度保持策略来恢复精度。工具内集成了多种精度保持策略，对权重的量化参数和取整方式进行优化，可在keep_acc参数中配置优化策略恢复精度。当sigma参数配置为0时，即开启activation min-max量化，不建议同时使用keep_acc参数配置的优化策略。keep_acc的配置示例如下：

```python
config = QuantConfig(quant_mode=0,
                     keep_acc={'admm': [False, 1000], 'easy_quant': [True, 1000], 'round_opt': False}
)
```

方式二：为保证精度，模型分类层和输入层不推荐量化，可在disable_names中配置分类层和输入层名称。

方式三：若使用虚拟数据在Data-Free量化后的精度不达标，可以输入随机真实数据进行量化。比如输入其他数据集的一张图片或一条语句来当随机数据，由于真实数据的数据分布更优，精度也会有所提升。以输入一张真实图片为例，可参考如下代码对数据进行预处理，在量化步骤中作为calib_data传入。

```python
def get_calib_data():
    import cv2
    import numpy as np

    img = cv2.imread('/xxx/cat.jpg')
    img_data = cv2.resize(img, (224, 224))
    img_data = img_data[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    img_data /= 255.
    img_data = np.expand_dims(img_data, axis=0)
    return [[img_data]]
```

#### 2.3.2 Label-Free模式（以post_training_quant接口为例）

本节将以模型静态shape和动态shape两种场景分别介绍量化配置步骤，其中静态shape模型以ResNet50为例，动态shape模型以YoloV5m为例，指导用户调用Python API接口对模型进行Label-Free模式的识别和量化，并将量化后的模型保存为.onnx文件，量化后的模型可以在推理服务器上运行。

本章节示例调用post_training_quant接口进行Label-Free量化配置，若用户需自行配置精度保持策略，可以调用squant_ptq接口进行Label-Free量化，参考Data-Free模式（以squant_ptq接口为例）的配置步骤并注意修改quant_mode和calib_data参数。

前提条件
已参考[安装指南](../../install_guide/install_guide.md)完成开发环境配置。
训练后量化前须执行命令安装依赖。
如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install onnx --user。

```bash
pip3 install onnx==1.13.0
pip3 install onnxruntime==1.14.1
```

功能实现流程
用户需准备好ONNX模型和数据集，调用post_training_quant接口生成量化配置脚本，运行脚本输出量化后的ONNX模型，并自行转换后进行推理。

图2 post_training_quant接口功能实现流程

![post_training_quant接口功能实现流程](./figures/[onnx]post_training_quant_api_implementation.png)

关键步骤说明如下：

用户准备onnx原始模型和数据集，使用QuantConfig配置量化参数，可基于静态shape和动态shape场景进行自定义配置，关于校准数据的输入，可以参考数据预处理提供的两种方式进行配置。

调用run_quantize保存量化后的模型。

模型转换。

参考《ATC工具使用指南》或使用其他转换工具，将ONNX模型转换为OM模型，并进行推理。

静态shape模型量化步骤（以ResNet50为例）

本样例以ResNet50为例，参考[README](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/README.md#%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE%E9%9B%86)中“准备数据集”章节获取ImageNet数据集即可，无需预处理，同时参考“模型推理”章节导出onnx文件。

新建模型量化脚本resnet50_quant.py，编辑resnet50_quant.py文件，导入如下样例代码。

```python
from msmodelslim.onnx.post_training_quant import QuantConfig, run_quantize  # 导入post_training_quant量化接口
from msmodelslim.onnx.post_training_quant.label_free.preprocess_func import preprocess_func_imagenet  # 导入预置的ImageNet数据集预处理函数preprocess_func_imagenet
from msmodelslim import set_logger_level  # 可选，导入日志配置接口
set_logger_level("info")  # 可选，调整日志输出等级，配置为info时，启动量化任务后将打屏显示量化调优的日志信息

# 用户需要自行准备一小批校准数据集，读取数据集进行数据预处理，并将数据存入calib_data
def custom_read_data():
    calib_data = preprocess_func_imagenet("./data_path/")  # 调用数据集预处理函数，请根据数据集实际路径配置，不使用该预处理函数时请参考数据预处理自行配置
    return calib_data
calib_data = custom_read_data()

quant_config = QuantConfig(calib_data = calib_data, amp_num = 5)  # 使用QuantConfig接口，配置量化参数，返回量化配置实例

input_model_path = "./resnet50_official.onnx"  # 配置待量化模型的输入路径，请根据实际路径配置
output_model_path = "./resnet50_official_quant.onnx"  # 配置量化后模型的名称及输出路径，请根据实际路径配置

run_quantize(input_model_path,output_model_path,quant_config)  # 使用run_quantize接口执行量化，配置待量化模型和量化后模型的路径及名称，
```

启动模型量化调优任务，并在指定的输出目录获取一个量化完成的模型。

python3 resnet50_quant.py

量化后的ONNX模型可参考[README](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/README.md#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)中“模型推理”章节，将ONNX模型转换为OM模型，并进行精度验证。

动态shape模型量化步骤（以YoloV5m为例）

本样例以YoloV5m为例，参考对应[README](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/README.md#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)中“模型推理”章节，导出动态shape的onnx文件。

新建模型的量化脚本yolov5m_quant.py，编辑yolov5m_quant.py文件，导入如下样例代码。

```python
from msmodelslim.onnx.post_training_quant import QuantConfig, run_quantize  # 导入post_training_quant量化接口
from msmodelslim import set_logger_level  # 可选，导入日志配置接口
set_logger_level("info")  # 可选，调整日志输出等级，配置为info时，启动量化任务后将打屏显示量化调优的日志信息

# 用户需要自行准备一小批校准数据集，读取数据集进行数据预处理，并将数据存入calib_data，当前配置示例为空时，将随机生成校准数据
def custom_read_data():
    calib_data = []
    # 可读取数据集，进行数据预处理，将数据存入calib_data
    return calib_data
calib_data = custom_read_data()

quant_config = QuantConfig(calib_data = calib_data, amp_num = 5, is_dynamic_shape = True, input_shape = [[1,3,640,640]])  # 使用QuantConfig接口，配置量化参数，返回量化配置实例，当前示例中is_dynamic_shape和input_shape参数在动态shape场景下必须配置。

input_model_path = "./yolov5m.onnx"  # 配置待量化模型的输入路径，请根据实际路径配置
output_model_path = "./yolov5m_quant.onnx"  # 配置量化后模型的名称及输出路径，请根据实际路径配置

run_quantize(input_model_path,output_model_path,quant_config)  # 使用run_quantize接口执行量化，配置待量化模型和量化后模型的路径及名称，
```

启动模型量化调优任务，并在指定的输出目录获取一个量化完成的模型。

```bash
python3 yolov5m_quant.py
```

量化后的ONNX模型可参考[README](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/README.md#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)中“模型推理”章节，将ONNX模型转换为OM模型，并进行精度验证。

数据预处理

post_training_quant接口进行Label-Free量化配置时，用户需要自行准备一小批校准数据集，读取数据集进行数据预处理，并返回预处理后的校准数据以用于量化。现支持通过msModelSlim工具预置的数据集预处理函数和自行准备校准数据两种方式：

方式一：msModelSlim工具预置preprocess_func_imagenet和preprocess_func_coco函数，对ImageNet和COCO数据集进行预处理，请参见接口对应的调用示例进行配置。

方式二：自行准备校准数据集，并返回校准数据用于量化配置，配置要求请参见QuantConfig的calib_data参数，以输入单张图片为例进行配置：

```python
import cv2
import numpy as np
import torch
import torch_npu   # 若需要在cpu上进行量化，可忽略此步骤
…

calib_data = []
    image = cv2.imdecode(np.fromfile("./random_image.jpg", dtype=np.uint8), 1)  #请以数据集实际路径进行配置
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1) / 255
    image = image.unsqueeze(0)
    calib_data.append([np.array(image)])
```

#### 2.3.3 已验证模型

目前支持对包括但不限于表1和表2中的模型进行模型训练后量化。

[表格1 已验证模型列表（昇腾A2系列产品（Atlas 800I A2 推理产品/A200I A2 Box 异构组件）或昇腾310P系列产品）](onnx/onnx_verification_table_1.xlsx)

[表格2 已验证模型列表（Atlas 200/500 A2推理产品）](onnx/onnx_verification_table_2.xlsx)

## 3. 训练后量化（MindSpore）

### 3.1 简介

训练后量化过程中，用户需自行提供训练好的权重文件和一小批验证集用来矫正量化因子，调用API接口完成模型的调优过程。目前支持MindSpore框架模型的量化调优。模型量化期间，用户可手动配置参数，并使用部分数据完成对模型的校准，获取一个量化后的模型。

### 3.2 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../install_guide/install_guide.md)。

### 3.3 功能介绍

1.用户自行准备预训练模型和数据集。本样例以ResNet50模型为例，获取模型结构定义脚本，并参考[README](https://gitee.com/mindspore/models/blob/master/official/cv/ResNet/README_CN.md)下载所需数据集， 以cifar10数据集为例，在config/resnet50_cifar10_config.yaml里配置data_path和checkpoint_file_path，并在eval.py的基础上进行修改。

2.新建模型量化脚本resnet50_quant.py，将eval.py内容复制到该文件中，删除eval_net()函数中定义损失，定义metric，计算metric相关的代码，保留如下初始化模型和加载权重相关的代码。

需要根据实际情况修改的配置项包括：

- **config.data_path**：数据集路径
- **config.batch_size**：批次大小
- **config.eval_image_size**：评估图像尺寸
- **config.class_num**：分类数量
- **config.checkpoint_file_path**：预训练权重文件路径

```python
target = config.device_target
# init context
ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)
if target == "Ascend":
    device_id = int(os.getenv('DEVICE_ID'))
    ms.set_context(device_id=device_id)
# create dataset
dataset = create_dataset(dataset_path=config.data_path, do_train=False, batch_size=config.batch_size,
                            eval_image_size=config.eval_image_size,
                            target=target)
# define net
model = resnet(class_num=config.class_num)
# load checkpoint
param_dict = ms.load_checkpoint(config.checkpoint_file_path)
ms.load_param_into_net(model, param_dict)
model.set_train(False)
```

3.在resnet50_quant.py中导入量化接口

```python
from msmodelslim.mindspore.quant.ptq_quant.quantize_model import quantize_model
from msmodelslim.mindspore.quant.ptq_quant.create_config import create_quant_config
from msmodelslim.mindspore.quant.ptq_quant.save_model import save_model
```

4.（可选）调整日志输出等级，启动调优任务后，将打印显示量化调优的日志信息。

```python
import logging
logging.getLogger().setLevel(logging.INFO)    #请根据实际情况进行配置
```

5.准备预训练模型。
当前脚本中已存在加载预训练模型的相关代码，可跳过此步骤，其他模型请参考以下内容进行配置。

```python
model = get_user_network()   #加载模型结构，返回模型实例
load_checkpoint(ckpt_file_path, model)    #模型加载预训练参数，请根据实际情况配置
```

6.使用create_quant_config接口生成配置文件。

```python
config_file = "./quant_config.json"    #待生成的量化配置文件存放路径及名称，请根据实际情况配置
create_quant_config(config_file, model)
```

7.使用quantize_model接口修改原模型，进行量化算子插入，此处的input_data需要与预训练模型的input保持一致的shape。

```python
import mindspore as ms
from mindspore import dtype as mstype
input_data = ms.Tensor(np.random.uniform(size=[256, 3, 224, 224]), dtype=mstype.float32)    #请根据实际情况配置
model_calibrate = quantize_model(config_file, model, input_data)    #通过调用quantize_model接口生成的量化后的模型
```

8.对量化模型进行校准。校准过程中会使用少量数据集进行前向传播，以校准量化算子中的参数，提高量化后的精度。

```python
for i, data in enumerate(dataset.create_dict_iterator(num_epochs=1)):
    model(data['image'])
    if i >= 2:
        break
```

9.使用save_model接口保存量化后模型。

```python
file_name = "./quantized_model"    #指定量化后模型保存路径和文件名
save_model(file_name, model_calibrate, input_data, file_format="AIR")    #请根据需要配置量化后模型的格式
```

10.启动模型量化调优任务，并在步骤9指定的目录获取一个量化完成的模型。

```bash
python3 resnet50_quant.py
```

## 4. 量化感知训练

### 4.1 简介

量化感知训练会重新训练量化模型，从而减小模型大小，并且加快推理过程。当前支持对PyTorch框架的CNN类模型进行量化，并将量化后的模型保存为.onnx文件，量化过程中，需要用户自行提供模型与数据集，调用API接口完成模型的量化调优。

### 4.2 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../install_guide/install_guide.md)。
量化感知训练前须执行命令安装依赖。
如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install onnx --user。

### 4.3 功能介绍

#### 4.3.1 操作步骤

1.用户需自行准备模型、训练脚本和数据集，本样例以PyTorch框架的Resnet50和数据集ImageNet为例。

2.编辑训练脚本pytorch_resnet50_apex.py文件，导入如下接口。

3.在优化器初始化之前调用“qsin_qat”函数，将量化后模型替换为“qsin_qat”的输出模型。请参考QatConfig和qsin_qat进行配置。同时在训练代码中，需注意保存伪量化模型权重ckpt文件，在导出量化onnx时使用。

```python
quant_config = QatConfig(grad_scale=0.001)
quant_logger = get_logger()
model = qsin_qat(model, quant_config, quant_logger).to(model.device)     #根据实际情况配置待量化模型实例、量化配置和量化输出日志，注意需把模型按照原训练流程部署在NPU设备
```

4.调用原训练流程进行单卡训练，执行train_full_1p.sh启动单卡训练任务。

```bash
bash ./test/train_full_1p.sh --data_path=/datasets/imagenet  #请根据实际情况配置数据集路径
```

5.导出量化后的ONNX模型。在伪量化模型权重ckpt文件保存后，新建quant_deploy.py文件，添加如下代码，调用“save_qsin_qat_model”函数，请参考save_qsin_qat_model进行配置。

```python
import argparse
import os
import torch
from ascend_utils.common.security.pytorch import safe_torch_load
import models.image_classification.resnet as nvmodels

# 初始化模型
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='onnx bs')
parser.add_argument('--pretrained', default="./org_model_best.pth.tar", type=str,
                    help='use pre-trained model')
parser.add_argument('--quant_ckpt', default="./checkpoint_77.244_asym.pth.tar", type=str,
                    help='use pre-trained model')

args = parser.parse_args()

model = nvmodels.build_resnet("resnet50", "classic", is_training=False)
pretrained_dict = safe_torch_load(args.pretrained, map_location='cpu')["state_dict"]
model.load_state_dict(pretrained_dict, strict=False)
#保存量化后的ONNX模型
from msmodelslim.pytorch.quant.qat_tools import save_qsin_qat_model
#根据实际情况配置导出后模型文件名（文件后缀需为.onnx）、输入的shape、伪量化模型权重和onnx的输入名称
save_onnx_name='./resnet50.onnx'
dummy_input = torch.ones([args.batch_size, 3, 224, 224]).type(torch.float32)
saved_ckpt = args.quant_ckpt
input_names=['input1']
save_qsin_qat_model(model, save_onnx_name, dummy_input, saved_ckpt, input_names)
```

6.执行量化脚本，获取量化后的ONNX模型。

```bash
python3 quant_deploy.py
```
