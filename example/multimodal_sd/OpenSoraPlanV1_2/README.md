# Open-Sora-Plan V1.2 量化使用说明

Open-Sora-Plan V1.2的推理量化依赖于推理工程仓：[MindIE/open_sora_planv1_2](https://modelers.cn/models/MindIE/open_sora_planv1_2)，根据该工程仓完成配置后，使用以下示例代码进行量化。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/getting_started/install_guide.md)。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接 | W8A8 | W8A16 | W4A16 | W4A4 | 稀疏量化 | KV Cache | Attention | 时间步量化 | FA3量化 | 异常值抑制量化 | 量化命令 |
|---------|---------|---------------------------------------------------------------|-----|-------|-------|------|---------|----------|-----------|----------|----------|----------|----------|
| **Open-Sora-Plan** | Open-Sora-Plan v1.2 | [Open-Sora-Plan v1.2](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0) | ✅ | | | | | | | | | | [W8A8静态量化](#open-sora-plan-v12-w8a8静态量化) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令

## 使用示例

### <span id="open-sora-plan-v12-w8a8静态量化">Open-Sora-Plan V1.2 W8A8静态量化</span>

#### 量化启动命令

我们提供了完整的量化启动脚本示例：[OpenSoraPlanV1_2/inference.py](./inference.py)，其启动命令可参考(请提前确保calib_prompts.txt权限不大于'0o640')：

```shell
# 根据使用卡数进行配置多卡环境变量和nproc_per_node，以下使用8卡为例
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"
export TASK_QUEUE_ENABLE=2
export HCCL_OP_EXPANSION_MODE="AIV"
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    /the/absolute/path/of/example/multimodal_sd/OpenSoraPlanV1_2/inference.py \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 93 \
    --height 720 \
    --width 1280 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt "example/multimodal_sd/OpenSoraPlanV1_2/calib_prompts.txt" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --tile_overlap_factor 0.125 \
    --max_sequence_length 512 \
    --dtype bf16 \
    --use_cfg_parallel \
    --algorithm "dit_cache" \
    --save_img_path "./results/quant/images" \
    --do_quant \
    --quant_weight_save_folder "./results/quant/safetensors" \
    --quant_dump_calib_folder "./results/quant/cache" \
    --quant_type "w8a8"
```

#### 校准数据Dump和量化的示例代码

```python
import os
import torch

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file

DUMP_CALIB_FOLDER = './results/quant/cache'  # 用于存放校准数据的文件夹
SAFE_TENSOR_FOLDER = './results/quant/safe_tensor'  # 用于存放量化模型的文件夹

rank = get_rank()
is_distributed = rank >= 0  # 标记是否为分布式环境

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, get_rank_suffix_file(base_name="calib_data", ext="pth",
                                                                      is_distributed=is_distributed, rank=rank))

############################ 加载模型 ############################
model_path = './model' #模型路径

def load_t2v_checkpoint(model_path):
    pass


pipeline = load_t2v_checkpoint(model_path)  # 加载模型

model = pipeline.transformer

############################ dump 校准数据 ############################
def run_model_and_save_images(pipeline, ...):
    # 原始模型推理过程
    pass

if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
    # 添加forward hook用于dump model的forward输入
    dumper_manager = DumperManager(model, capture_mode='args')

    # 执行浮点模型推理
    run_model_and_save_images(
        pipeline,
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
            cfg=W8A8QuantConfig(
                act_method='minmax'
            ),
            disable_names=get_disable_layer_names(model, layer_include=None,
                                                    layer_exclude=('*net.2*', '*adaln_single*'))
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

以下是使用[OpenSoraPlanV1_2/inference.py](./inference.py)进行Open-Sora-Plan V1.2模型推理量化时的参数说明。量化启动命令未涉及参数对应的说明请见Open-Sora-Plan V1.2推理工程仓[MindIE/open_sora_planv1_2](https://modelers.cn/models/MindIE/open_sora_planv1_2)

| 参数名 | 含义 | 使用限制 |
| ------ | ------ | ------ |
| model_path | Open-Sora-Plan V1.2原始浮点模型路径 | 必选。<br>数据类型：字符串。无默认值。|
| num_frames | 设置生成的总帧数 | 可选。<br>数据类型：整型。默认值93。|
| height | 指定生成视频的高度 | 可选。<br>数据类型：整型。默认值720。|
| width | 指定生成视频的宽度 | 可选。<br>数据类型：整型。默认值1280。|
| dtype | 指定用于推理的数据类型 | 可选。<br>数据类型：字符串。默认值'bf16'。<br>可选值：'bf16'或'fp16'。|
| cache_dir | 指定缓存目录，用于存储临时文件 | 可选。<br>数据类型：字符串。默认值'./cache_dir'。|
| ae | VAE的对视频的压缩规格 | 可选。<br>数据类型：字符串。默认值'CausalVAEModel_4x8x8'。|
| ae_path | 指定VAE模型权重配置路径 | 可选。<br>数据类型：字符串。默认值'CausalVAEModel_4x8x8'。|
| text_encoder_name | 指定text_encoder权重配置路径 | 可选。<br>数据类型：字符串。默认值'google/mt5-xxl'。|
| save_img_path | 指定生成视频的保存路径 | 可选。<br>数据类型：字符串。默认值"./sample_videos/t2v"。|
| guidance_scale | 指定引导比例，用于控制negative文本对视频生成的影响程度 | 可选。<br>数据类型：浮点型。默认值7.5。|
| num_sampling_steps | 指定采样步骤的数量，用于控制生成视频的多样性 | 可选。<br>数据类型：整型。默认值50。|
| fps | 指定生成视频的帧率 | 可选。<br>数据类型：整型。默认值24。|
| batch_size | 指定批处理大小，用于控制一次生成视频的数量 | 可选。<br>数据类型：整型。默认值1。|
| max_sequence_length | 指定最大序列长度，用于控制文本编码器的输入长度 | 可选。<br>数据类型：整型。默认值512。|
| text_prompt | 指定文本提示，可以是单个字符串或包含多个字符串的列表，也可以是包含多个字符串的文本文件路径 | 必选。<br>数据类型：字符串或字符串列表、txt文件。无默认值。|
| tile_overlap_factor | VAE tiling decode时重叠比例，用于控制生成视频的细节 | 可选。<br>数据类型：浮点型。默认值0.25。|
| algorithm | 指定使用的算法 | 可选。<br>数据类型：字符串。默认值None。<br>可选值：None、'dit_cache'或'sampling_optimize'。|
| use_cfg_parallel | 是否使用cfg并行，用于控制模型的并行计算方式 | 可选。<br>数据类型：布尔型。默认值False。只有显式传入 --use_cfg_parallel 则变为True。|
| test_time | 是否开启性能测试 | 可选。<br>数据类型：布尔型。默认值False。只有显式传入 --test_time 则变为True。|
| seed | 控制随机种子 | 可选。<br>数据类型：整型。默认值1234。|
| vae_parallel | 是否启用VAE并行计算 | 可选。<br>数据类型：布尔型。默认值False。只有显式传入 --vae_parallel 则变为True。|
| do_quant | 是否进行量化 | 必选。<br>数据类型：布尔型。默认False，即不启动量化。只有显式传入 --do_quant 则变为True，在进行Open-Sora-Plan v1.2模型推理量化时，必须使能该参数。|
| quant_type | 指定量化类型 | 可选。<br>数据类型：字符串。默认值"w8a8"。<br>可选值："w8a8"。|
| quant_weight_save_folder | 指定量化模型权重保存路径 | 必选。<br>数据类型：字符串。无默认值。|
| quant_dump_calib_folder | 指定量化校准数据保存路径 | 必选。<br>数据类型：字符串。无默认值。|
| do_save_video | 是否进行推理视频保存 | 可选。<br>数据类型：布尔型。默认False，即不启动推理视频保存。只有显式传入 --do_save_video 则变为True，启动视频保存。|
