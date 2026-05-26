# FLUX 量化使用说明

FLUX的推理量化依赖于FLUX.1-dev推理工程仓：[MindIE/FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev)，根据该工程仓完成配置后，使用以下示例代码进行量化。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/getting_started/install_guide/)。
- 硬件支持：Atlas 800I A2
- 软件支持：FLUX.1-dev推理工程仓

**注意**：diffusers仅支持>=0.33.0,<=0.33.1

## 操作流程

- 克隆工程仓代码；

- 非一键量化需要执行 `git checkout 12e09174353b1bd57bf7fcb80386f59b09fbbefe` 切换至指定版本；

- 完成后续配置与量化步骤。
  
**注意**：未使用指定版本可能导致兼容性问题或功能异常。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接 | 时间步量化 | FA3量化 | 异常值抑制量化 | 量化命令 |
|---------|---------|-------|-----|-------|-------|-------|
| **FLUX** | FLUX.1-dev | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main) | ✅ | ✅ | ✅ | [时间步量化](#flux-timestep-quantization) / [FA3量化](#flux-fa3-quantization) / [异常值抑制量化](#flux-outlier-suppression-quantization) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令

## 使用示例

### <span id="flux-timestep-quantization">FLUX 时间步量化</span>

**注意**: 在模型pipeline的去噪循环中，需要在每个timestep开始时调用`TimestepManager.set_timestep_idx()`来设置当前的时间步。

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager

# 在去噪循环中设置timestep
for step_id, t in enumerate(timesteps):
    # ----------- w8a8_timestep quantization -----------
    TimestepManager.set_timestep_idx(step_id)  # 必须在每个timestep开始时调用
    # ----------- w8a8_timestep quantization -----------

    model_output = pipeline(...)
    ...
```

例如在FLUX.1-dev/FLUX1dev/pipeline/pipeline_flux.py的FluxPipeline类的__call__函数中，添加如下代码：

```python
with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i,t in enumerate(timesteps):
        if self.interrupt:
            continue
        # -----------新增代码-----------
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
        TimestepManager.set_timestep_idx(i)
        # -----------新增代码-----------
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
```

#### 量化启动命令

示例的启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# do quant
python /the/absolute/path/of/example/multimodal_sd/Flux/inference_flux.py \
    --path ${model_path} \
    --save_path "./results/quant/img" \
    --device_id 0 \
    --device "npu" \
    --prompt_path "example/multimodal_sd/Flux/calib_prompts.txt" \
    --width 1024 \
    --height 1024 \
    --infer_steps 50 \
    --seed 42 \
    --use_cache \
    --device_type "A2-64g" \
    --batch_size 1 \
    --max_num_prompt 0 \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8_timestep"
```

#### 校准数据Dump和量化的示例代码

```python
import os
import torch

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import W8A8TimeStepProcessorConfig, W8A8TimeStepQuantConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safetensors'  # 用于存放量化模型的文件夹

rank = get_rank()
is_distributed = rank >= 0  # 标记是否为分布式环境

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, get_rank_suffix_file(base_name="calib_data", ext="pth",
                                                                      is_distributed=is_distributed, rank=rank))

############################ 加载模型 ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='timestep')

    # 执行浮点模型推理
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=20,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据
calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')

safetensors_name = get_rank_suffix_file('quant_model_weight_w8a8_timestep', 'safetensors', is_distributed, rank)
json_name = get_rank_suffix_file('quant_model_description_w8a8_timestep', 'json', is_distributed, rank)
# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
        "w8a8_timestep": W8A8TimeStepProcessorConfig(
            cfg=W8A8TimeStepQuantConfig(
                act_method='minmax'
            ),
            disable_names=get_disable_layer_names(
                model,
                layer_include='*',
                layer_exclude='*net.2*',
                ),
            timestep_sep=25,

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

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```

### <span id="flux-fa3-quantization">FLUX FA3+W8A8动态量化</span>

#### 使用quant_type参数进行一键量化

W8A8(MXFP8)+FA3(FP8动态)

```bash
msmodelslim quant \
    --model_path /path/to/flux1_float_weights \
    --save_path /path/to/flux1_quantized_weights \
    --device npu \
    --model_type FLUX.1-dev \
    --quant_type w8a8f8 \
    --trust_remote_code True
```

#### 脚本量化启动命令

W8A8(INT8)+FA3(INT8静态)

我们提供了完整的量化启动脚本示例：[Flux/inference_flux.py](./inference_flux.py)，其启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# do quant
python /the/absolute/path/of/example/multimodal_sd/Flux/inference_flux.py \
    --path ${model_path} \
    --save_path "./results/quant/img" \
    --device_id 0 \
    --device "npu" \
    --prompt_path "example/multimodal_sd/Flux/calib_prompts.txt" \
    --width 1024 \
    --height 1024 \
    --infer_steps 50 \
    --seed 42 \
    --use_cache \
    --device_type "A2-64g" \
    --batch_size 1 \
    --max_num_prompt 0 \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8_dynamic_fa3"
```

#### 校准数据Dump和量化的示例代码

```python

import os
import torch

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import FA3ProcessorConfig, W8A8DynamicQuantConfig, W8A8DynamicProcessorConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safetensors'  # 用于存放量化模型的文件夹

rank = get_rank()
is_distributed = rank >= 0  # 标记是否为分布式环境

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, get_rank_suffix_file(base_name="calib_data", ext="pth",
                                                                      is_distributed=is_distributed, rank=rank))

############################ 加载模型 ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='args')

    # 执行浮点模型推理
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=20,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据
calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')

safetensors_name = get_rank_suffix_file('quant_model_weight_w8a8_dynamic', 'safetensors', is_distributed, rank)
json_name = get_rank_suffix_file('quant_model_description_w8a8_dynamic', 'json', is_distributed, rank)
# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
    "fa3": FA3ProcessorConfig(),
    "w8a8_dynamic": W8A8DynamicProcessorConfig(
        cfg=W8A8DynamicQuantConfig(
            act_method='minmax'
        ),
        disable_names=[],

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

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```

### <span id="flux-outlier-suppression-quantization">FLUX 异常值抑制量化</span>

#### 量化启动命令

我们提供了完整的量化启动脚本示例：[Flux/inference_flux.py](./inference_flux.py)，其启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# do quant
python /the/absolute/path/of/example/multimodal_sd/Flux/inference_flux.py \
    --path ${model_path} \
    --save_path "./results/quant/img" \
    --device_id 0 \
    --device "npu" \
    --prompt_path "example/multimodal_sd/Flux/calib_prompts.txt" \
    --width 1024 \
    --height 1024 \
    --infer_steps 50 \
    --seed 42 \
    --use_cache \
    --device_type "A2-64g" \
    --batch_size 1 \
    --max_num_prompt 0 \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8_dynamic" \
    --anti_method "m4"
```

#### 校准数据Dump和量化的示例代码

```python

import os
import torch

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import M3ProcessorConfig, M4ProcessorConfig, M6ProcessorConfig, W8A8DynamicQuantConfig, \
    W8A8DynamicProcessorConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safetensors'  # 用于存放量化模型的文件夹

rank = get_rank()
is_distributed = rank >= 0  # 标记是否为分布式环境

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, get_rank_suffix_file(base_name="calib_data", ext="pth",
                                                                      is_distributed=is_distributed, rank=rank))

############################ 加载模型 ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='args')

    # 执行浮点模型推理
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=50,
        ...
    )
    # 保存校准数据
    dumper_manager.save(dump_data_path)

############################ 启动量化 ############################
# 加载校准数据
calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')
safetensors_name = get_rank_suffix_file('quant_model_weight_w8a8_dynamic', 'safetensors', is_distributed, rank)
json_name = get_rank_suffix_file('quant_model_description_w8a8_dynamic', 'json', is_distributed, rank)
# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
    "m4": M4ProcessorConfig(),
    "w8a8_dynamic": W8A8DynamicProcessorConfig(
        cfg=W8A8DynamicQuantConfig(
            act_method='minmax'
        ),
        disable_names=get_disable_layer_names(
            model,
            layer_include='*',
            layer_exclude='*net.2*',
        ),

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

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```

## 附录

### 运行参数说明

以下是使用[Flux/inference_flux.py](./inference_flux.py)进行FLUX.1-dev模型推理量化时的参数说明。量化启动命令未涉及参数对应的说明请见FLUX.1-dev推理工程仓[MindIE/FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev)

| 参数名 | 含义 | 使用限制 |
| ------ | ------ | ------ |
| path | FLUX.1-dev原始浮点模型路径 | 必选。<br>数据类型：字符串。无默认值。|
| save_path | 保存图像路径 | 可选。<br>数据类型：字符串。默认值"./res"。|
| device_id | 推理设备ID | 可选。<br>数据类型：整型。默认值0。|
| device | 推理设备类型 | 可选。<br>数据类型：字符串。默认值"npu"。<br>可选值："npu"或"cpu"。|
| prompt_path | 用于图像生成的文字描述提示的列表文件路径 | 可选。<br>数据类型：字符串。默认值"./calib_prompts.txt"。|
| prompt_type | 指定推理提示词类型 | 可选。<br>数据类型：字符串。默认值"plain"。可选值："plain"、"parti"、"hpsv2"。|
| num_images_per_prompt | 每个提示词生成的图像数量 | 可选。<br>数据类型：整型。默认值1。|
| max_num_prompt | 限制提示词数量（0意味着无限制）| 可选。<br>数据类型：整型。默认值0。|
| info_file_save_path | 保存图像信息的路径 | 可选。<br>数据类型：字符串。默认值"./image_info.json"。|
| width | 图像生成的宽度 | 可选。<br>数据类型：整型。默认值1024。|
| height | 图像生成的高度 | 可选。<br>数据类型：整型。默认值1024。|
| infer_steps | Flux图像推理步数 | 可选。<br>数据类型：整型。默认值50。|
| seed | 设置提示词随机种子 | 可选。<br>数据类型：整型。默认值42。|
| use_cache | 是否开启dit cache近似优化 | 可选。<br>数据类型：布尔型。默认值False。。只有显式传入 --use_cache 则变为True。|
| batch_size | 指定prompt的batch size | 可选。<br>数据类型：整型。默认值1。<br>说明：大于1时以list形式送入pipeline。|
| device_type | device类型 | 可选。<br>数据类型：字符串。默认值'A2-64g'。<br>可选值：'A2-32g-single'、'A2-32g-dual'或'A2-64g'。|
| do_quant | 是否进行量化 | 必选。<br>数据类型：布尔型。默认False，即不启动量化。只有显式传入 --do_quant 则变为True，在进行Flux.1-dev模型推理量化时，必须使能该参数。|
| quant_type | 指定量化类型 | 可选。<br>数据类型：字符串。默认值"w8a8_timestep"。<br>可选值："w8a8_timestep"、"w8a8_dynamic_fa3"或"w8a8_dynamic"。|
| anti_method | 指定异常值抑制方法 | 可选。<br>数据类型：字符串。默认值None。<br>可选值："m3"、"m4"或"m6"。|
| quant_weight_save_folder | 指定量化权重保存文件夹 | 必选。<br>数据类型：字符串。无默认值。|
| quant_dump_calib_folder | 指定量化校准数据保存文件夹 | 必选。<br>数据类型：字符串。无默认值。|
| data_split_num | 数据分片数量 | 可选。<br>数据类型：整型。默认值1。|
| data_split_id | 数据分片ID | 可选。<br>数据类型：整型。默认值0。|
| do_save_img | 是否进行推理图像保存 | 可选。<br>数据类型：布尔型。默认False，即不启动推理图像保存。只有显式传入 --do_save_img 则变为True，启动图像保存。|
