# 稀疏量化精度调试案例

## 概述

本文将会提供稀疏量化的示例代码，并介绍稀疏量化的调优策略。

## 前期准备

代码示例中额外使用了`precision_tool`工具，可以参考该文档进行配置：[精度测试工具](../user_guide/feature_guide/traditional_quantization_v0/fake_quantization_accuracy_testing_tool.md)。

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../install_guide/install_guide.md)。

## 代码示例

```config
import json
import torch
import argparse
import torch_npu # 如果需要使用npu进行量化
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from precision_tool.precision_tool import PrecisionTest # precision_tool用于伪量化测精度

def parse_args():
    parser = argparse.ArgumentParser(description="Sparse quant demo")
    parser.add_argument("--model_path", type=str, default="/path/to/model", help="The path to model float weights")
    parser.add_argument("--save_path", type=str, default="./path/to/save", help="The path to save quant weights")
    parser.add_argument("--device", type=str, default="npu:0", help="The device to execute quant process")
    parser.add_argument("--calib_dataset_path", type=str, default="/path/to/dataset", help="The path to calibrate dataset, eg. boolq")
    parser.add_argument("--calib_dataset_count", type=int, default=50, help="The count of data to do calibration")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size when run precision tool")
    parser.add_argument("--fraction", type=float, default=0.01, help="Fraction to control sparse ratio")
    parser.add_argument("--do_smooth", type=bool, default=True, help="Enable the antioutlier for lowbit sparse quant mode")
    parser.add_argument("--co_sparse", type=bool, default=False, help="Enable the co_sparse mode sparse quant mode")
    parser.add_argument("--is_lowbit", type=bool, default=True, help="Enable the lowbit sparse quant mode")
    parser.add_argument("--use_sigma", type=bool, default=False, help="Enable sigma antioutlier protection in the lowbit sparse quant mode")
    return parser.parse_args()

def test_generate_oneshot(tokenizer, model):
    test_prompt = "Where is the capital of China?"
    test_input = tokenizer(test_prompt, return_tensors="pt")
    print("model is inferring...")
    model.eval()
    generate_ids = model.generate(
        test_input.input_ids.to(f"npu:{model.device.index}"),
        attention_mask=test_input.attention_mask.to(f"npu:{model.device.index}"),
        max_new_tokens=SEQ_LEN_OUT
    )
    out_str = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(out_str)


args = parse_args()

SEQ_LEN_OUT = 100

# 如果使用npu进行量化需开启二进制编译，避免在线编译算子
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

"""
1、导入相关模型
"""
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=args.model_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map=args.device)

"""
2、可选步骤：数据集测原始模型浮点精度（此示例中选择的是boolq）
"""
# precision_test = PrecisionTest(model, tokenizer, "boolq", args.batch_size, "npu")
# precision_test.test()

print("testing float weights...")
test_generate_oneshot(tokenizer, model)

"""
3、获取校准数据
"""
def build_prompt(title, text, passage):
    prompt = f"{title} -- {passage}\nQuestion:{text}?\nAnswer:"
    return prompt

# 生成输入要和模型所在设备保持一致，否则会报错
def get_calib_dataset(tokenizer, calib_list, device):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append([inputs.data['input_ids'].to(device), inputs.data['attention_mask'].to(device)])
    return calib_dataset

calib_set = []  # 初始化为空列表，后续从校准集文件中读取并追加相应的校准数据
# 从校准数据集文件夹中生成校准数据
with open(args.calib_dataset_path, encoding="utf-8") as file:
    i = 0
    for line in file:
        if i == args.calib_dataset_count:
            break
        data = json.loads(line) # 将字符串转换为字典
        calib_set.append(build_prompt(data["title"], data["question"], data["passage"]))
        i += 1

dataset_calib = get_calib_dataset(tokenizer, calib_set, device=args.device)

"""
4、回退层设置
"""
"""
因为一些量化后的网络层对精度影响太大了，所以需要让这些网络层使用浮点权重进行计算， disable_names中为需要进行回退的网络层。
"""
disable_names = [f"model.layers.{i}.mlp.down_proj" for i in range(model.config.num_hidden_layers)]

"""
5、执行PTQ量化校准 + 存储量化参数用于部署
"""
quant_config = QuantConfig(
    a_bit=8,
    w_bit=4,                    # w_bit=4 a_bit=8 代表开启稀疏量化
    disable_names=disable_names,# 回退层配置
    dev_type='npu',             # 运行量化的设备
    dev_id=model.device.index,  # 运行量化的设备ID，使用npu或cuda时需要指定
    is_lowbit=args.is_lowbit,   # 启用lowbit模式
    co_sparse=args.co_sparse,   # 启用co_sparse模式
    fraction=args.fraction,     # 控制稀疏率
    do_smooth=args.do_smooth,   # True则可以启用lowbit模式的异常值抑制
    use_sigma=args.use_sigma,   # True则可以启用lowbit模式中基于高斯分布的异常值保护
)

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  # disable_level: 自动回退n个linear
calibrator.run()  # 执行PTQ量化校准
calibrator.save(args.save_path, save_type=["safe_tensor"]) # "safe_tensor"对应safetensors格式权重

"""
6、可选步骤：伪量化验证一轮推理，先跑一轮对话可以快速判断是否出现明显的精度异常，如乱码等现象
"""
print("testing quantized weights...")
test_generate_oneshot(tokenizer, model)

"""
7、可选步骤：数据集测伪量化模型精度（此示例中选择的是boolq），伪量化跑数据集验证很慢
"""
precision_test = PrecisionTest(model, tokenizer, "boolq", args.batch_size, "npu")
precision_test.test()
```

# msModelSlim稀疏量化精度调试指导

## 基本精度调优策略

稀疏量化与msModelSlim所提供的其他量化方式**有着同样的基本精度调优策略**，分别是：**调整校准集**、**进行异常值抑制**、**添加回退层**。

### 调整校准集

可以从**数量**和**质量**两个维度调整校准集。

**校准集的质量体现在应该充分代表量化模型将会面对的数据集**，我们推荐使用测试数据集作为量化过程的校准数据，如目标数据集是boolq，则可随机抽取部分样本作为校准集。如果测试数据集有多个，则可以考虑每个数据集抽取若干个样本共同组成校准集。

**校准集的数量可以在一定程度上提高校准的精度**，但这种提升有限，在数据集增加到一定数量之后就不再会有效果，建议使用20~50条数据作为校准集。

### 进行异常值抑制

异常值抑制可以提升量化的精度，在稀疏量化中，目前有两种异常值抑制模式。

#### lowbit异常值抑制

lowbit模式（参考[稀疏模式](#稀疏模式)）下，可以通过在QuantConfig指定参数`do_smooth=True`启用该模式内部自带的异常值抑制方法，该**方法内含了自动调优机制，因此相较于AntiOutlier模块而言效果会略好一些，建议优先使用该方法**。

#### AntiOutlier

可以使用独立的异常值抑制模块AntiOutlier对模型进行异常值抑制。

**对于稀疏量化而言，量化的难度主要在于权重，因此建议使用AWQ异常值抑制算法**，该算法对于权重量化有更好的效果，可以通过在AntiOutlierConfig指定参数`anti_method=m3`。

### 增加回退层

部分层进行量化后，会对模型整网的精度带来较大的损失，针对这样的层，**可以选择不进行量化处理，这就是所谓的回退层**。

#### 手动回退

回退层的选择通常依赖于经验，例如在llama系模型中，MLP中的down层通常会被回退。可以通过指定`QuantConfig`中的`disable_names`参数手动控制需要回退的层。

#### 自动回退

量化工具还提供了自动回退的功能，可以通过指定`Calibrator`中的`disable_level`参数控制需要自动回退的层数，如令`disable_level='L5'`，量化工具将会最多回退5层（可能不到五层，取决于具体的模型和输入，且不包含disable_names所指定的层）。

由于目前没有一个较为有效的判断量化损失的算法，该功能无法保证可以带来正收益，建议多加调整。

## 独有调优参数

除了基本的精度调优策略之外，还有部分**稀疏量化独有的调优参数**。

### 稀疏模式

可以通过在`QuantConfig`中指定`use_sigma=True`

目前工具提供了两种稀疏量化的模式，分别对应了两种不同的算法后端。

**需要注意，两种稀疏模式是互斥的，只能二选一。**

#### lowbit稀疏模式

可以通过在`QuantConfig`中指定`is_lowbit=True`参数指定使用lowbit稀疏模式。

#### co_sparse稀疏模式

可以通过在`QuantConfig`中指定`co_sparse=True`参数指定使用co_sparse稀疏模式。

#### 如何选择

如果要使用异常值抑制功能，建议优先使用lowbit稀疏模式，该模式下的[lowbit异常值抑制](#lowbit异常值抑制)具备自动调优功能，同等配置下通常会比co_sparse模式的效果略好。

### 调整稀疏率

可以通过调整稀疏率的方式对精度进行调优。

对于稀疏量化而言，通常有这样的原则：**稀疏率越高，则精度越低，但性能越好；反之，稀疏率越低，则精度越高，但性能越差。**

量化工具目前提供了两种调整稀疏率的方式。

**这两种方式是互斥的，需要二选一。**

#### fraction

可以通过修改`QuantConfig`中的`fraction`参数控制调整稀疏率。

该参数直接与稀疏率本身成负线性关系。

**因此，将fraction调大，会导致稀疏率降低，进而导致模型精度提升，但模型性能下降，反之亦然。**

**建议在0.01~0.1范围内调整该参数。**

#### use_sigma + sigma_factor

可以通过在`QuantConfig`中指定`use_sigma=True`参数启用基于高斯分布的稀疏率控制，**该参数仅lowbit模式下支持**。

启用后，稀疏率不再是固定的比例，而是根据待量化权重的具体分布而定。可以通过调整`QuantConfig`中的`sigma_factor`参数调整稀疏率，**sigma_factor越小则稀疏率越低，模型精度越高，反之亦然**。
