# Open-Sora-Plan V1.2 Quantification Instructions

The inference quantification of Open-Sora-Plan V1.2 depends on the inference engineering repository.[MindIE/open_sora_planv1_2](https://modelers.cn/models/MindIE/open_sora_planv1_2)    After the configuration is complete based on the engineering warehouse, use the following sample code to quantify the configuration.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/en/getting_started/install_guide.md)    .

## Supported Model Versions and Quantification Policies

| Model Series       | Model version       | HuggingFace Link                                                                  | W8A8 | W8A16 | W4A16 | W4A4 | sparse quantization | KV Cache | Attention | time step quantization | FA3 Quantification | outlier suppression quantization | Quantization command                                      |
| ------------------ | ------------------- | --------------------------------------------------------------------------------- | ---- | ----- | ----- | ---- | ------------------- | -------- | --------- | ---------------------- | ------------------ | -------------------------------- | --------------------------------------------------------- |
| **Open-Sora-Plan** | Open-Sora-Plan v1.2 | [Open-Sora-Plan v1.2](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0)     | ✅    |       |       |      |                     |          |           |                        |                    |                                  | [W8A8 static quantization](#open-sora-plan-v12-w8a8-static-quantization)     |

**Description:**

 * " indicates that the quantification policy has passed the official verification of msModelSlim. The function is complete and the performance is stable. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the quantization command column to go to the specific quantization command.

## Use Example

### Open-Sora-Plan V1.2 W8A8 Static Quantization

#### Quantizing Start Command

A complete sample of the quantification startup script is provided:[OpenSoraPlanV1_2/inference.py](./inference.py)    For details about the startup command, see. (Ensure that the permission on calib_prompts.txt is not greater than '0o640'.)

```shell
#Configure the multi-card environment variables and nproc_per_node based on the number of used cards. The following uses eight cards as an example.
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

#### Sample Code for Calibration Data Dump and Quantization

```python
import os
import torch

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file

DUMP_CALIB_FOLDER = './results/quant/cache'  #Folder for storing calibration data
SAFE_TENSOR_FOLDER = './results/quant/safe_tensor'  #Folder for storing quantization models

rank = get_rank()
is_distributed = rank >= 0  #Marking a distributed environment

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, get_rank_suffix_file(base_name="calib_data", ext="pth",
                                                                      is_distributed=is_distributed, rank=rank))

############################ Load Model ############################
model_path = './model' #Model Path

def load_t2v_checkpoint(model_path):
    pass


pipeline = load_t2v_checkpoint(model_path)  #Load Model

model = pipeline.transformer

############################ dump Calibration Data ############################
def run_model_and_save_images(pipeline, ...):
    #Original Model Inference Process
    pass

if not os.path.exists(dump_data_path):  #Check whether the calibration data exists. If the calibration data does not exist, dump the data.
    #Add the forward hook for the forward input of the dump model.
    dumper_manager = DumperManager(model, capture_mode='args')

    #Perform floating-point model inference
    run_model_and_save_images(
        pipeline,
        ...
    )
    #Saving calibration data
    dumper_manager.save(dump_data_path)

############################ Start Quantization ############################
#Load the calibration data. The calibration data needs to be dumped and generated in advance.
calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')
safetensors_name = get_rank_suffix_file(base_name='quant_model_weight_w8a8', ext='safetensors',
                                        is_distributed=is_distributed, rank=rank)
json_name = get_rank_suffix_file(base_name='quant_model_description_w8a8', ext='json',
                                 is_distributed=is_distributed, rank=rank)
#Quantized configuration
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

#Data type verification in the pydantic library
session_cfg.model_validate(session_cfg)

#quantization model
quant_model(model, session_cfg)
```

## Appendixes

### Running parameter description

Here's how to use[OpenSoraPlanV1_2/inference.py](./inference.py)    This topic describes the parameters for performing Open-Sora-Plan V1.2 model inference and quantification. For details about parameters not involved in the quantization startup command, see the Open-Sora-Plan V1.2 inference engineering repository.[MindIE/open_sora_planv1_2](https://modelers.cn/models/MindIE/open_sora_planv1_2)    

| Parameter name           | Meaning:                                                                                                                                | Use Restrictions                                                                                                                                                                                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| model_path               | Open-Sora-Plan V1.2 Original Floating Point Model Path                                                                                  | Mandatory. Data type: character string. No default value.                                                                                                                                                                                                          |
| num_frames               | Sets the total number of frames generated.                                                                                              | Optional. Data type: Integer. Default value: 93.                                                                                                                                                                                                                   |
| height                   | Specify the height of the generated video                                                                                               | Optional. Data type: integer. Default value: 720.                                                                                                                                                                                                                  |
| width                    | Specify the width of the generated video                                                                                                | Optional. Data type: integer. The default value is 1280.                                                                                                                                                                                                           |
| dtype                    | Specify the data type used for inference                                                                                                | Optional. Data type: String. The default value is'bf16'. The value can be bf16 or fp16.                                                                                                                                                                            |
| cache_dir                | Specify the cache directory for storing temporary files.                                                                                | Optional. Data type: String. The default value is './cache_dir'.                                                                                                                                                                                                   |
| ae                       | Video Compression Specifications of the VAE                                                                                             | Optional. Data type: String. The default value is CausalVAEModel_4x8x8.                                                                                                                                                                                            |
| ae_path                  | Specifies the path for configuring the weight of a VAE model.                                                                           | Optional. Data type: String. The default value is CausalVAEModel_4x8x8.                                                                                                                                                                                            |
| text_encoder_name        | Specifies the path for configuring the text_encoder weight.                                                                             | Optional. Data type: String. The default value is'google/mt5-xxl'.                                                                                                                                                                                                 |
| save_img_path            | Specify the save path of the generated video.                                                                                           | Optional. Data type: String. Default value: "./sample_videos/t2v".                                                                                                                                                                                                 |
| guidance_scale           | Specifies the boot scale, which is used to control the extent to which negative text affects video generation.                          | Optional. Data type: floating point. Default value: 7.5.                                                                                                                                                                                                           |
| num_sampling_steps       | Specify the number of sampling steps used to control the variety of generated video                                                     | Optional. Data type: integer. Default value: 50.                                                                                                                                                                                                                   |
| fps                      | Specifies the frame rate for the generated video.                                                                                       | Optional. Data type: integer. Default value: 24.                                                                                                                                                                                                                   |
| batch_size               | Specify the batch size to control the number of videos generated at a time                                                              | Optional. Data type: integer. Default value: 1.                                                                                                                                                                                                                    |
| max_sequence_length      | Specifies the maximum sequence length, which is used to control the input length of the text encoder.                                   | Optional. Data type: integer. The default value is 512.                                                                                                                                                                                                            |
| text_prompt              | Specifies a text prompt, which can be a single string, a list of multiple strings, or a path to a text file containing multiple strings | Mandatory. Data type: string, string list, or TXT file. No default value is provided.                                                                                                                                                                              |
| tile_overlap_factor      | Overlap ratio during VAE tiling decode, which is used to control the details of the generated video.                                    | Optional. Data type: floating point. Default: 0.25.                                                                                                                                                                                                                |
| algorithm                | Specifies the algorithm used.                                                                                                           | Optional. Data type: String. The default value is None. The options are None,'dit_cache', or'sampling_optimize'.                                                                                                                                                   |
| use_cfg_parallel         | Whether to use cfg parallel, which is used to control the parallel computing mode of the model.                                         | Optional. Data type: Boolean. The default value is False. If --use_cfg_parallel is explicitly passed, it becomes True.                                                                                                                                             |
| test_time                | Whether to enable performance test                                                                                                      | Optional. Data type: Boolean. The default is False. If only --test_time is explicitly passed, becomes True.                                                                                                                                                        |
| seed                     | Control random seed                                                                                                                     | Optional. Data type: integer. The default value is 1234.                                                                                                                                                                                                           |
| vae_parallel             | Enable VAE parallel computing                                                                                                           | Optional. Data type: Boolean. The default value is False. If only --vae_parallel is explicitly passed, becomes True.                                                                                                                                               |
| do_quant                 | Quantification or not                                                                                                                   | Mandatory. Data type: Boolean. The default value is False, indicating that quantization is not enabled. If --do_quant is explicitly transferred, the value becomes True. This parameter must be enabled during Open-Sora-Plan v1.2 model inference quantification. |
| quant_type               | Refers to the quantitative type                                                                                                         | Optional. Data type: String. The default value is w8a8. The value can be w8a8.                                                                                                                                                                                     |
| quant_weight_save_folder | Indicates the path for saving the weight of the quantitative model.                                                                     | Mandatory. Data type: string. No default value.                                                                                                                                                                                                                    |
| quant_dump_calib_folder  | Indicates the path for saving quantitative calibration data.                                                                            | Mandatory. Data type: character string. No default value.                                                                                                                                                                                                          |
| do_save_video            | Indicates whether to save the inference video.                                                                                          | Optional. Data type: Boolean. The default value is False, indicating that the inference video is not saved. If only --do_save_video is explicitly transferred, the value becomes True and the video is saved.                                                      |
