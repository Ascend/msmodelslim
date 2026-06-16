# HunyuanVideo 量化使用说明

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/getting_started/install_guide/)。
- HunyuanVideo浮点推理环境与依赖请参考[魔乐 HunyuanVideo](https://modelers.cn/models/MindIE/hunyuan_video) 及 [README](https://modelers.cn/models/MindIE/hunyuan_video/blob/main/README.md)，确保在量化前可正常完成浮点推理。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接 | W8A8 | W8A16 | W4A16 | W4A4 | 稀疏量化 | KV Cache | Attention | 时间步量化 | FA3量化 | 异常值抑制量化 | 量化命令 |
|---------|---------|---------------------------------------------------------------|-----|-------|-------|------|---------|----------|-----------|----------|----------|----------|----------|
| **HunyuanVideo** | HunyuanVideo-T2V-720P | [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) | ✅ | | | | | | | ✅ | ✅ | ✅ | [时间步量化](#hunyuanvideo-时间步量化) / [FA3量化](#hunyuanvideo-fa3-量化) / [异常值抑制量化](#hunyuanvideo-异常值抑制量化) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令

## 使用示例

### <span id="hunyuanvideo-时间步量化">HunyuanVideo 时间步量化</span>

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

例如在hunyuan_video/hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py的HunyuanVideoPipeline类的__call__函数中，添加如下代码：

```python
with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i,t in enumerate(timesteps):
        if self.interrupt:
            continue
        # -----------新增代码-----------
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
        TimestepManager.set_timestep_idx(i)
        # -----------新增代码-----------
        latent_model_input = (
            torch.cat([latents] * 2)
            if self.do_classifier_free_guidance
            else latents
        )
```

#### 量化启动命令

示例的启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# 根据使用卡数进行配置多卡环境变量和nproc_per_node，以下使用8卡为例
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=0

torchrun --nproc_per_node=8 /the/absolute/path/of/example/multimodal_sd/HunYuanVideo/sample_video.py \
    --model-base HunyuanVideo \
    --dit-weight HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --vae-path HunyuanVideo/hunyuan-video-t2v-720p/vae \
    --text-encoder-path HunyuanVideo/text_encoder \
    --text-encoder-2-path HunyuanVideo/clip-vit-large-patch14 \
    --model-resolution "720p" \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "example/multimodal_sd/HunYuanVideo/calib_prompts.txt" \
    --seed 42 \
    --flow-reverse \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --vae-parallel \
    --num-videos 1 \
    --save-path ./results \
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
safetensors_name = get_rank_suffix_file(base_name='quant_model_weight_w8a8_timestep', ext='safetensors',
                                        is_distributed=is_distributed, rank=rank)
json_name = get_rank_suffix_file(base_name='quant_model_description_w8a8_timestep', ext='json',
                                 is_distributed=is_distributed, rank=rank)
# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
        "w8a8_timestep": W8A8TimeStepProcessorConfig(
            cfg=W8A8TimeStepQuantConfig(
                act_method='minmax'
            ),
            disable_names=get_disable_layer_names(
              model,
              layer_include=['*double_blocks*', '*single_blocks*'],
              layer_exclude=['*img_mod*', '*modulation*', '*fc2*'],
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
with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
    quant_model(model, session_cfg)

```

### <span id="hunyuanvideo-fa3-量化">HunyuanVideo fa3 量化</span>

#### 使用quant_type参数进行一键量化

W8A8(MXFP8)+FA3(FP8动态)

```bash
msmodelslim quant \
    --model_path /path/to/hunyuan_float_weights \
    --save_path /path/to/hunyuan_quantized_weights \
    --device npu \
    --model_type HunyuanVideo \
    --quant_type w8a8f8 \
    --trust_remote_code True
```

完整 lab 配置见 [`lab_practice/hunyuan_video/hunyuan_video_w8a8f8_mxfp.yaml`](../../../lab_practice/hunyuan_video/hunyuan_video_w8a8f8_mxfp.yaml)。

##### 配置文件说明

###### 基础配置结构

```yaml
apiversion: multimodal_sd_modelslim_v1

spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: True
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: True
          method: "minmax"
      include:
        - "*"
      exclude:
        - "*mod*"
    - type: "online_quarot"
      include: ["*"]
    - type: "fa3_quant"
      qconfig:
        dtype: "fp8_e4m3"
        scope: "per_token"
        symmetric: True
        method: "minmax"
      include:
        - "*"

  dataset: hunyuanvideo

  save:
    - type: "mindie_format_saver"
      part_file_size: 0

  multimodal_sd_config:
    dump_config:
      enable_dump: False    # 全动态量化示例；静态/离群值抑制请改为 True
      capture_mode: "args"
      dump_data_dir: ""     # 空则使用 save_path；pth 见下文命名规则
    inference_config:       # 推荐；勿与已废弃的 model_config 同时配置
      model_resolution: "720p"
      video_size: [720, 1280]
      video_length: 129
      infer_steps: 50
```

**关键配置参数**

**量化配置 (process)**

- **linear_quant**：DiT 线性层 W8A8（MXFP8 per-block）；示例排除 `*mod*` 调制层。
- **online_quarot**：全图注意力在线旋转。
- **fa3_quant**：注意力 FA3 动态 FP8 量化。

**校准数据集 (dataset)**

- **作用**：指定 `index.json` / `index.jsonl` 或目录路径；短名称 `hunyuanvideo` 在 [`lab_calib`](../../../lab_calib) 下解析。
- **格式**：每条须含非空 `text`（校准 prompt）；**不提供** `image`（HunyuanVideo-T2V 为纯文本生成）。

**多模态配置 (multimodal_sd_config)**

- **dump_config**
  - `enable_dump`：是否 load/dump 校准数据；纯动态量化可设 `False`。
  - `capture_mode`：当前仅支持 `"args"`。
  - `dump_data_dir`：pth 根目录；为空时使用 `--save_path`。
- **inference_config**（推荐）：推理参数，字段须与 HunyuanVideo 推理仓 `hyvideo.config.parse_args` CLI 对齐，由适配器 Pydantic 校验后桥接到 `model_args`。合法字段以 `HunyuanVideoInferenceConfig` 为准（`extra=forbid`）。
- **model_config**（Legacy）：旧单体入口使用，**将废弃**；与 `inference_config` 不可同配。

**不在 `inference_config` 中配置的项**：`prompt` 来自 `dataset`；`model_base`、`dit_weight`、`vae_path`、`text_encoder_path`、`text_encoder_2_path` 由 `--model_path` 与适配器默认相对路径注入；`ulysses_degree` / `ring_degree` / `vae_parallel` / `use_cache` 等由适配器在量化路径固定为单卡默认值。

**`inference_config` 字段**（完整声明见 `msmodelslim/model/hunyuan_video/model_adapter.py` 中的 `HunyuanVideoInferenceConfig`）：

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `model_resolution` | 可选 | 分辨率档位，仅 `540p` 或 `720p`（默认 **720p**，与 t2v-720p 权重目录对齐）。须与 `video_size` 组合落在 `SUPPORT_SIZES[model_resolution]` 内。 |
| `video_size` | 可选 | 生成视频 **[高, 宽]**，对应 `--video-size`（`nargs=+`）。**720p** 可选：`(720,1280)`、`(1280,720)`、`(1104,832)`、`(832,1104)`、`(960,960)`；**540p** 可选：`(544,960)`、`(960,544)`、`(624,832)`、`(832,624)`、`(720,720)`。默认 `(720, 1280)`。YAML 可写列表 `[720, 1280]`。 |
| `video_length` | 可选 | 采样帧数，对应 `--video-length`；3D VAE 场景须满足 **4n+1**（默认 **129**）。 |
| `infer_steps` | 可选 | 去噪步数，对应 `--infer-steps`（默认 **50**）。若启用 cache/attentioncache 相关能力，步数须大于 cache 起始步（量化路径默认关闭 cache）。 |
| `seed` | 可选 | 随机种子，对应 `--seed`。省略为 `None`（推理仓按 `seed_type` 处理）；**≥0** 时校准时对每条样本在推理前固定种子。 |
| `neg_prompt` | 可选 | 负向 prompt，对应 `--neg-prompt`；省略为 `None`。 |
| `cfg_scale` | 可选 | Classifier-Free Guidance 强度，对应 `--cfg-scale`（默认 **1.0**）。 |
| `embedded_cfg_scale` | 可选 | 嵌入式 CFG 强度，对应 `--embedded-cfg-scale`（默认 **6.0**）。 |
| `num_videos` | 可选 | 每个 prompt 生成视频条数，对应 `--num-videos`（默认 **1**）。 |
| `flow_shift` | 可选 | Flow Matching scheduler 的 shift，对应 `--flow-shift`（默认 **7.0**）。 |
| `batch_size` | 可选 | 推理 batch 大小，对应 `--batch-size`（默认 **1**）。 |

#### 脚本量化启动命令

W8A8(INT8)+FA3(INT8静态)

**注意：**
Atlas 800I A2(8*64G)推理设备：支持4卡量化、6卡量化、8卡量化。

我们提供了完整的量化启动脚本示例：[HunYuanVideo/sample_video.py](./sample_video.py)，其启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# 根据使用卡数进行配置多卡环境变量和nproc_per_node，以下使用8卡为例
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=0
torchrun --nproc_per_node=8 /the/absolute/path/of/example/multimodal_sd/HunYuanVideo/sample_video.py \
    --model-base HunyuanVideo \
    --dit-weight HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --vae-path HunyuanVideo/hunyuan-video-t2v-720p/vae \
    --text-encoder-path HunyuanVideo/text_encoder \
    --text-encoder-2-path HunyuanVideo/clip-vit-large-patch14 \
    --model-resolution "720p" \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "example/multimodal_sd/HunYuanVideo/calib_prompts.txt" \
    --seed 42 \
    --flow-reverse \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --vae-parallel \
    --num-videos 1 \
    --save-path ./results \
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
safetensors_name = get_rank_suffix_file(base_name='quant_model_weight_w8a8_dynamic', ext='safetensors',
                                        is_distributed=is_distributed, rank=rank)
json_name = get_rank_suffix_file(base_name='quant_model_description_w8a8_dynamic', ext='json',
                                 is_distributed=is_distributed, rank=rank)
# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
        "fa3": FA3ProcessorConfig(),
        "w8a8_dynamic": W8A8DynamicProcessorConfig(
            cfg = W8A8DynamicQuantConfig(
                act_method = 'minmax'
            ),
            disable_names=get_disable_layer_names(
                model,
                layer_include=('*double_blocks*', '*single_blocks*'),
                layer_exclude=('*img_mod*', '*modulation*'),
            ),
        ),
        "save": SaveProcessorConfig(
            output_path=SAFE_TENSOR_FOLDER,
            safetensors_name=safetensors_name,
            json_name=json_name,
            save_type=["safe_tensor"],
            part_file_size=None,
        )
    },
    calib_data=calib_dataset[:20],
    device = "npu",
)

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```

### <span id="hunyuanvideo-异常值抑制量化">HunyuanVideo 异常值抑制量化</span>

#### 量化启动命令

我们提供了完整的量化启动脚本示例：[HunYuanVideo/sample_video.py](./sample_video.py)，其启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# 根据使用卡数进行配置多卡环境变量和nproc_per_node，以下使用8卡为例
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=0
torchrun --nproc_per_node=8 /the/absolute/path/of/example/multimodal_sd/HunYuanVideo/sample_video.py \
    --model-base HunyuanVideo \
    --dit-weight HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --vae-path HunyuanVideo/hunyuan-video-t2v-720p/vae \
    --text-encoder-path HunyuanVideo/text_encoder \
    --text-encoder-2-path HunyuanVideo/clip-vit-large-patch14 \
    --model-resolution "720p" \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "example/multimodal_sd/HunYuanVideo/calib_prompts.txt" \
    --seed 42 \
    --flow-reverse \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --vae-parallel \
    --num-videos 1 \
    --save-path ./results \
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
safetensors_name = get_rank_suffix_file(base_name='quant_model_weight_w8a8_dynamic', ext='safetensors',
                                        is_distributed=is_distributed, rank=rank)
json_name = get_rank_suffix_file(base_name='quant_model_description_w8a8_dynamic', ext='json',
                                 is_distributed=is_distributed, rank=rank)
# 量化配置
session_cfg = SessionConfig(
    processor_cfg_map={
        "m4": M4ProcessorConfig(),
        "w8a8_dynamic": W8A8DynamicProcessorConfig(
            cfg = W8A8DynamicQuantConfig(
                act_method = 'minmax'
            ),
            disable_names=get_disable_layer_names(
                model,
                layer_include=['*double_blocks*', '*single_blocks*'],
                layer_exclude=['*img_mod*', '*modulation*', '*fc2*'],
            ),
        ),
        "save": SaveProcessorConfig(
            output_path=SAFE_TENSOR_FOLDER,
            safetensors_name=safetensors_name,
            json_name=json_name,
            save_type=["safe_tensor"],
            part_file_size=None,
        )
    },
    calib_data=calib_dataset,
    device = "npu",
)

# pydantic库自带的数据类型校验
session_cfg.model_validate(session_cfg)

# 量化模型
quant_model(model, session_cfg)
```

## 附录

### 运行参数说明

以下是使用[HunYuanVideo/sample_video.py](./sample_video.py)进行HunyuanVideo模型推理量化时的参数说明。量化启动命令未涉及参数对应的说明请见HunyuanVideo推理工程仓[MindIE/hunyuan_video](https://modelers.cn/models/MindIE/hunyuan_video)

| 参数名 | 含义 | 使用限制 |
| ------ | ------ | ------ |
| model-base | HunyuanVideo权重路径，包含vae、text_encoder、Tokenizer、Transformer和Scheduler五个模型的配置文件及权重。 | 必选。<br>数据类型：字符串。默认值"ckpts"。|
| dit-weight | dit的权重路径 | 必选。<br>数据类型：字符串。默认值"ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"。|
| vae-path | VAE的权重路径 | 必选。<br>数据类型：字符串。默认值"vae"。|
| text-encoder-path | text_encoder的权重路径 | 必选。<br>数据类型：字符串。默认值"text_encoder"。|
| text-encoder-2-path | text_encoder_2的权重路径 | 必选。<br>数据类型：字符串。默认值"clip-vit-large-patch14"。|
| model-resolution | 分辨率 | 可选。<br>数据类型：字符串。默认值"540p"。|
| video-size | 生成视频的高和宽 | 可选。<br>数据类型：整型列表。默认值(720, 1280)。|
| video-length | 总帧数 | 可选。<br>数据类型：整型。默认值129。|
| infer-steps | 推理去噪总步数 | 可选。<br>数据类型：整型。默认值50。|
| prompt | 验证过程中用于采样的提示词 | 可选。<br>数据类型：字符串。默认值None。|
| seed | 验证过程的随机种子 | 可选。<br>数据类型：整型。默认值None。|
| flow-reverse | 是否反向流，如果反向, 学习或采样将从时间步1到时间步0 | 可选。<br>数据类型：布尔型。默认False。只有显式传入 --flow-reverse 则变为True|
| ulysses-degree | Ulysses长序列并行度 | 可选。<br>数据类型：整型。默认值1。|
| ring-degree | Ring并行度 | 可选。<br>数据类型：整型。默认值1。|
| vae-parallel | vae部分使能并行，目前只支持8卡、16卡并行时使用 | 可选。<br>数据类型：布尔型。默认False。只有显式传入 --vae-parallel 则变为True|
| num-videos | 每个prompt生成的视频数量 | 可选。<br>数据类型：整型。默认值1。|
| save-path | 生成视频的保存路径 | 可选。<br>数据类型：字符串。默认值'./results'。|
| do_quant | 是否进行量化 | 必选。<br>数据类型：布尔型。默认False，即不启动量化。只有显式传入 --do_quant 则变为True，在进行HunyuanVideo模型推理量化时，必须使能该参数。|
| quant_weight_save_folder | 量化权重保存文件夹 | 必选。<br>数据类型：字符串。无默认值。|
| quant_dump_calib_folder | 量化校准数据保存文件夹 | 必选。<br>数据类型：字符串。无默认值。|
| quant_type | 指定量化类型 | 可选。<br>数据类型：字符串。默认值"w8a8_timestep"。<br>可选值："w8a8_timestep"、"w8a8_dynamic_fa3"、"w8a8_dynamic"。|
| anti_method | 指定异常值抑制方法 | 可选。<br>数据类型：字符串。默认值None。<br>可选值：'m3'、'm4'、'm6'。|
| do_save_video | 是否进行推理视频保存 | 可选。<br>数据类型：布尔型。默认False，即不启动推理视频保存。只有显式传入 --do_save_video 则变为True，启动视频保存。|

### 相关资源

- [HunyuanVideo 模型仓库（魔乐）](https://modelers.cn/models/MindIE/hunyuan_video)
- 《[多模态生成模型接入指南（开发者）](../../../docs/zh/developer_guide/integrating_multimodal_generation_model.md)》
- [一键量化配置协议说明](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/feature_guide/quick_quantization_v1/usage/#%E9%87%8F%E5%8C%96%E9%85%8D%E7%BD%AE%E5%8D%8F%E8%AE%AE%E8%AF%A6%E8%A7%A3)
- [逐层量化特性说明](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/feature_guide/quick_quantization_v1/usage/#%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96%E5%8F%8A%E5%88%86%E5%B8%83%E5%BC%8F%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96)
