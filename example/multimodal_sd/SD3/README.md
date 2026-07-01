# SD3-Medium量化使用说明

当前仅支持对SD3模型的transformer部分进行W8A8静态量化。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/getting_started/install_guide.md)。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接 | W8A8 | W8A16 | W4A16 | W4A4 | 时间步量化 | FA3量化 | 异常值抑制量化 | 量化命令 |
|---------|---------|---------------------------------------------------------------|-----|-------|-------|------|---------|----------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SD3** | SD3-Medium | [SD3-Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | ✅ |   |   |   |   |   |   | [W8A8](#sd3-medium-w8a8量化) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令

## 使用示例

### <span id="sd3-medium-w8a8量化">SD3-Medium W8A8量化</span>

我们提供了完整的量化启动脚本示例：[SD3/sd3_inference.py](./sd3_inference.py)，其启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
python /the/absolute/path/of/example/multimodal_sd/SD3/sd3_inference.py \
    --sd3_model_path "/path/to/stable-diffusion-3-medium-diffusers" \
    --prompt_path "example/multimodal_sd/SD3/calib_prompts.txt" \
    --width 1024 \
    --height 1024 \
    --infer_steps 28 \
    --seed 42 \
    --device "npu" \
    --save_path "./results/quant/images" \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8"
```

#### 校准数据Dump和量化的示例代码

```python
# 导入模型库
import os
import torch
from diffusers import StableDiffusion3Pipeline

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safetensors'  # 用于存放量化模型的文件夹

rank = get_rank()
is_distributed = rank >= 0  # 标记是否为分布式环境

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, get_rank_suffix_file(base_name="calib_data", ext="pth",
                                                                      is_distributed=is_distributed, rank=rank))

############################ 加载模型 ############################
def load_t2v_checkpoint(model_path):
    pipeline = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16).to('npu')
    return pipeline


pipeline = load_t2v_checkpoint("/path/to/stable-diffusion-3-medium-diffusers")  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='args')

    # 执行浮点模型推理

    pipeline(
        prompts=["A photo of an astronaut riding a horse on mars"],
        negative_prompts=[""],
        width=1024,
        height=1024,
        num_inference_steps=28,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据，校准数据需要提前dump生成
calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')
safetensors_name = get_rank_suffix_file(base_name='quant_model_weight_w8a8', ext='safetensors',
                                        is_distributed=is_distributed, rank=rank)
json_name = get_rank_suffix_file(base_name='quant_model_description_w8a8', ext='json',
                                 is_distributed=is_distributed, rank=rank)
# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
        "w8a8": W8A8ProcessorConfig(
            cfg = W8A8QuantConfig(
                act_method='minmax'
            ),
            disable_names=['context_embedder']
        ),
        "save": SaveProcessorConfig(
            output_path=SAFE_TENSOR_FOLDER,
            safetensors_name=safetensors_name,
            json_name=json_name,
            save_type=['safe_tensor'],
            part_file_size=None
        )
    },
    calib_data=calib_dataset,
    device='npu'
)

# python pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```

## 附录

### 运行参数说明

以下是使用[SD3/sd3_inference.py](./sd3_inference.py)进行SD3模型推理量化时的参数说明。

| 参数名 | 含义 | 使用限制 |
| ------ | ------ | ------ |
| sd3_model_path | SD3原始浮点模型路径 | 必选。<br>数据类型：字符串。无默认值。|
| prompt_path | 输入prompt（提示词）路径 | 可选。<br>数据类型：字符串。默认值"./calib_prompts.txt"。|
| width | 生成图像宽度 | 可选。<br>数据类型：整型。默认值1024。|
| height | 生成图像高度 | 可选。<br>数据类型：整型。默认值1024。|
| infer_steps | 推理步数 | 可选。<br>数据类型：整型。默认值28。|
| seed | prompt（提示词）随机种子 | 可选。<br>数据类型：整型。默认值42。|
| device | 模型运行设备 | 可选。<br>数据类型：字符串。默认值"npu"，当前仅支持npu。|
| save_path | 推理图像保存路径 | 可选。<br>数据类型：字符串。默认值"./results"。仅在do_save_img开启时生效。|
| do_quant | 是否进行量化 | 必选。<br>数据类型：布尔型。默认False，即不启动量化。只有显式传入 --do_quant 则变为True，在进行SD3模型推理量化时，必须使能该参数。|
| quant_type | 量化类型 | 可选。<br>数据类型：字符串。默认值"w8a8"，当前仅支持"w8a8"。|
| quant_weight_save_folder | 量化权重保存路径 | 必选。<br>数据类型：字符串。无默认值。|
| quant_dump_calib_folder | 量化校准数据保存路径 | 必选。<br>数据类型：字符串。无默认值。|
| do_save_img | 是否进行推理图像保存 | 可选。<br>数据类型：布尔型。默认False，即不启动推理图像保存。只有显式传入 --do_save_img 则变为True，启动图像保存。|
