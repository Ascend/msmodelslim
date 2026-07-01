# HunyuanVideo Quantification Instructions

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/en/getting_started/install_guide.md)    .

## Supported Model Versions and Quantification Policies

| Model Series     | Model version         | HuggingFace Link                                             | W8A8 | W8A16 | W4A16 | W4A4 | sparse quantization | KV Cache | Attention | time step quantization | FA3 Quantification | outlier suppression quantization | Quantization command                                                                                                                                  |
| ---------------- | --------------------- | ------------------------------------------------------------ | ---- | ----- | ----- | ---- | ------------------- | -------- | --------- | ---------------------- | ------------------ | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **HunyuanVideo** | HunyuanVideo-T2V-720P | [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)     | ✅    |       |       |      |                     |          |           | ✅                      | ✅                  | ✅                                | [time step quantization](#hunyuanvideo-time-step-quantization)    /[FA3 Quantification](#hunyuanvideo-fa3-quantization)    /[outlier suppression quantization](#quantization-of-hunyuanvideo-abnormal-value-suppression)     |

**Description:**

 * " indicates that the quantification policy has passed the official verification of msModelSlim. The function is complete and the performance is stable. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the quantization command column to go to the specific quantization command.

## Use Example

### HunyuanVideo time step quantization

**Note: In the denoising loop of the model pipeline, this interface needs to be invoked at the start of each timestep.**`TimestepManager.set_timestep_idx()`to set the current time step.

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager

#Set the timestep in the denoising cycle.
for step_id, t in enumerate(timesteps):
    #----------- w8a8_timestep quantization -----------
    TimestepManager.set_timestep_idx(step_id)  #Must be called at the start of each timestep
    #----------- w8a8_timestep quantization -----------

    model_output = pipeline(...)
    ...
```

For example, add the following code to the __call__ function in the HunyuanVideoPipeline class of hunyuan_video/hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py:

```python
with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i,t in enumerate(timesteps):
        if self.interrupt:
            continue
        #----------- New Code -----------
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
        TimestepManager.set_timestep_idx(i)
        #----------- New Code -----------
        latent_model_input = (
            torch.cat([latents] * 2)
            if self.do_classifier_free_guidance
            else latents
        )
```

#### Quantization Start Command

For details about the startup command example, see. (Ensure that the permission on calib_prompts.txt is not greater than '0o640'.)

```shell
#Configure the multi-card environment variables and nproc_per_node based on the number of used cards. The following uses eight cards as an example.
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

#### Sample Code for Calibration Data Dump and Quantization

```python
import os
import torch

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import W8A8TimeStepProcessorConfig, W8A8TimeStepQuantConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file

DUMP_CALIB_FOLDER = './results/quant/cache'  #Folder for storing calibration data
SAFE_TENSOR_FOLDER = './results/quant/safetensors'  #Folder for storing quantization models

rank = get_rank()
is_distributed = rank >= 0  #Marking a distributed environment

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, get_rank_suffix_file(base_name="calib_data", ext="pth",
                                                                      is_distributed=is_distributed, rank=rank))

############################ Load Model ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  #Load Model

model = pipeline.transformer

############################ dump Calibration Data ############################
if not os.path.exists(dump_data_path):  #Check whether the calibration data exists. If the calibration data does not exist, dump the data.
    #Add the forward hook for the forward input of the dump model.
    dumper_manager = DumperManager(model, capture_mode='timestep')

    #Perform floating-point model inference
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=20,
        ...
    )
    #Saving calibration data
    dumper_manager.save(dump_data_path)

############################ Start Quantification ############################
#Load calibration data
calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')
safetensors_name = get_rank_suffix_file(base_name='quant_model_weight_w8a8_timestep', ext='safetensors',
                                        is_distributed=is_distributed, rank=rank)
json_name = get_rank_suffix_file(base_name='quant_model_description_w8a8_timestep', ext='json',
                                 is_distributed=is_distributed, rank=rank)
#Quantized configuration
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

#Data type verification in the pydantic library
session_cfg.model_validate(session_cfg)

#quantization model
with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
    quant_model(model, session_cfg)
```

### HunyuanVideo fa3 quantization

The FA3+W8A8 dynamic quantization of the model has been integrated into one-click quantization.

#### Use the config_path parameter to specify the configuration file for one-click quantization

```bash
msmodelslim quant \
    --model_path /path/to/hunyuan_float_weights \
    --save_path /path/to/hunyuan_quantized_weights \
    --device npu \
    --model_type hunyuan_video \
    --config_path /lab_practice/hunyuan_video/hunyuan_video_w8a8f8_mxfp.yaml \
    --trust_remote_code True
```

#### Script Quantization Start Command

**Note: The Atlas 800I A2 (8 x 64 GB) inference device supports 4-card quantization, 6-card quantization, and 8-card quantization.**

A complete sample of the quantification startup script is provided:[HunYuanVideo/sample_video.py](./sample_video.py)    For details about the startup command, see. (Ensure that the permission on calib_prompts.txt is not greater than '0o640'.)

```shell
#Configure the multi-card environment variables and nproc_per_node based on the number of used cards. The following uses eight cards as an example.
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

#### Sample Code for Calibration Data Dump and Quantization

```python
import os
import torch

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import FA3ProcessorConfig, W8A8DynamicQuantConfig, W8A8DynamicProcessorConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file

DUMP_CALIB_FOLDER = './results/quant/cache'  #Folder for storing calibration data
SAFE_TENSOR_FOLDER = './results/quant/safetensors'  #Folder for storing quantization models

rank = get_rank()
is_distributed = rank >= 0  #Marking a distributed environment

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, get_rank_suffix_file(base_name="calib_data", ext="pth",
                                                                      is_distributed=is_distributed, rank=rank))

############################ Load Model ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  #Load Model

model = pipeline.transformer

############################ dump Calibration Data ############################
if not os.path.exists(dump_data_path):  #Check whether the calibration data exists. If the calibration data does not exist, dump the data.
    #Add the forward hook for the forward input of the dump model.
    dumper_manager = DumperManager(model, capture_mode='args')

    #Perform floating-point model inference
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=20,
        ...
    )
    #Saving calibration data
    dumper_manager.save(dump_data_path)

############################ Start Quantization ############################
#Load calibration data
calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')
safetensors_name = get_rank_suffix_file(base_name='quant_model_weight_w8a8_dynamic', ext='safetensors',
                                        is_distributed=is_distributed, rank=rank)
json_name = get_rank_suffix_file(base_name='quant_model_description_w8a8_dynamic', ext='json',
                                 is_distributed=is_distributed, rank=rank)
#Quantized configuration
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

#Data type verification in the pydantic library
session_cfg.model_validate(session_cfg)

#quantization model
quant_model(model, session_cfg)
```

### Quantization of HunyuanVideo Abnormal Value Suppression

#### Quantization Start Command

A complete sample of the quantification startup script is provided:[HunYuanVideo/sample_video.py](./sample_video.py)    For details about the startup command, see. (Ensure that the permission on the calib_prompts.txt file is not greater than '0o640'.)

```shell
#Configure the multi-card environment variables and nproc_per_node based on the number of used cards. The following uses eight cards as an example.
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

#### Sample Code for Calibration Data Dump and Quantization

```python
import os
import torch

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import M3ProcessorConfig, M4ProcessorConfig, M6ProcessorConfig, W8A8DynamicQuantConfig, \
  W8A8DynamicProcessorConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file

DUMP_CALIB_FOLDER = './results/quant/cache'  #Folder for storing calibration data
SAFE_TENSOR_FOLDER = './results/quant/safetensors'  #Folder for storing quantization models

rank = get_rank()
is_distributed = rank >= 0  #Marking a distributed environment

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, get_rank_suffix_file(base_name="calib_data", ext="pth",
                                                                      is_distributed=is_distributed, rank=rank))

############################ Load Model ############################
def load_pipeline():
    pass


pipeline = load_pipeline(...)  #Load model

model = pipeline.transformer

############################ dump Calibration Data ############################
if not os.path.exists(dump_data_path):  #Check whether the calibration data exists. If the calibration data does not exist, dump the data.
    #Add the forward hook for the forward input of the dump model.
    dumper_manager = DumperManager(model, capture_mode='args')

    #Perform floating-point model inference
    pipeline(
        prompt="A photo of an astronaut riding a horse on mars",
        num_inference_steps=50,
        ...
    )
    #Saving calibration data
    dumper_manager.save(dump_data_path)

############################ Start Quantization ############################
#Load calibration data
calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')
safetensors_name = get_rank_suffix_file(base_name='quant_model_weight_w8a8_dynamic', ext='safetensors',
                                        is_distributed=is_distributed, rank=rank)
json_name = get_rank_suffix_file(base_name='quant_model_description_w8a8_dynamic', ext='json',
                                 is_distributed=is_distributed, rank=rank)
#Quantified configuration
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

#Data type verification in the pydantic library
session_cfg.model_validate(session_cfg)

#Quantification model
quant_model(model, session_cfg)
```

## Appendixes

### Running parameter description

Here's how to use[HunYuanVideo/sample_video.py](./sample_video.py)    This topic describes the parameters for inference and quantification of the HunyuanVideo model. For details about parameters not involved in the quantization startup command, see the HunyuanVideo inference engineering library.[MindIE/hunyuan_video](https://modelers.cn/models/MindIE/hunyuan_video)    

| Parameter name           | Meaning:                                                                                                                                        | Use Restrictions                                                                                                                                                                                                                                              |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model-base               | HunyuanVideo weight path, including the configuration files and weights of the vae, text_encoder, Tokenizer, Transformer, and Scheduler models. | Mandatory. Data type: character string. The default value is ckpts.                                                                                                                                                                                           |
| dit-weight               | Weight path of dit.                                                                                                                             | Mandatory. Data type: character string. Default value: "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt".                                                                                                                                |
| vae-path                 | VAE Weighted Path                                                                                                                               | Mandatory. Data type: character string. The default value is "vae".                                                                                                                                                                                           |
| text-encoder-path        | Weight path of text_encoder.                                                                                                                    | Mandatory. Data type: character string. The default value is text_encoder.                                                                                                                                                                                    |
| text-encoder-2-path      | Weight path of text_encoder_2                                                                                                                   | Mandatory. Data type: character string. The default value is "clip-vit-large-patch14".                                                                                                                                                                        |
| model-resolution         | Resolution                                                                                                                                      | Optional. Data type: String. The default value is 540p.                                                                                                                                                                                                       |
| video-size               | Height and width of the generated video                                                                                                         | Optional. Data type: integer list. The default value is 720, 1280.                                                                                                                                                                                            |
| video-length             | Total number of frames                                                                                                                          | Optional. Data type: integer. Default value 129.                                                                                                                                                                                                              |
| infer-steps              | Total number of inference denoising steps                                                                                                       | Optional. Data type: integer. Default value: 50.                                                                                                                                                                                                              |
| prompt                   | Prompt word for sampling during validation                                                                                                      | Optional. Data type: String. Default value: None.                                                                                                                                                                                                             |
| seed                     | Random seed of the validation process                                                                                                           | Optional. Data type: integer. Default value: None.                                                                                                                                                                                                            |
| flow-reverse             | Indicates whether the flow is backward. If yes, the learning or sampling will be performed from time step 1 to time step 0.                     | Optional. Data type: Boolean. The default value is False. If only --flow-reverse is explicitly passed, the default value becomes True.                                                                                                                        |
| ulysses-degree           | Ulysses long sequence parallelism                                                                                                               | Optional. Data type: integer. Default value: 1.                                                                                                                                                                                                               |
| ring-degree              | Ring parallelism degree                                                                                                                         | Optional. Data type: Integer. Default value: 1.                                                                                                                                                                                                               |
| vae-parallel             | Enables the parallel function for some VAEs. Currently, only the parallel function is supported when eight or 16 cards are used.                | Optional. Data type: Boolean. The default value is False. If --vae-parallel is explicitly passed, the default value becomes True.                                                                                                                             |
| num-videos               | Number of videos generated by each prompt.                                                                                                      | Optional. Data type: integer. Default value: 1.                                                                                                                                                                                                               |
| save-path                | Path for storing the generated video.                                                                                                           | Optional. Data type: String. The default value is './results'.                                                                                                                                                                                                |
| do_quant                 | Quantized or not                                                                                                                                | Mandatory. Data type: Boolean. The default value is False, indicating that quantization is not enabled. If --do_quant is explicitly transferred, the value becomes True. This parameter must be enabled during HunyuanVideo model inference and quantization. |
| quant_weight_save_folder | Save folder for quantization weights                                                                                                            | Mandatory. Data type: character string. No default value.                                                                                                                                                                                                     |
| quant_dump_calib_folder  | Directory for storing quantitative calibration data                                                                                             | Mandatory. Data type: character string. No default value.                                                                                                                                                                                                     |
| quant_type               | Refers to the quantitative type                                                                                                                 | Optional. Data type: String. The default value is w8a8_timestep. The options are w8a8_timestep, w8a8_dynamic_fa3, and w8a8_dynamic.                                                                                                                           |
| anti_method              | Specify the method for suppressing abnormal values.                                                                                             | Optional. Data type: String. The default value is None. The options are'm3','m4', and'm6'.                                                                                                                                                                    |
| do_save_video            | Indicates whether to save the inference video.                                                                                                  | Optional. Data type: Boolean. The default value is False, indicating that the inference video is not saved. If only --do_save_video is explicitly transferred, the value becomes True and the video is saved.                                                 |
