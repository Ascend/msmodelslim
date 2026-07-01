# FLUX Quantification Instructions

The inference quantification of the FLUX depends on the FLUX.1-dev inference engineering repository.[MindIE/FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev)    After the configuration is complete based on the engineering warehouse, use the following sample code to quantify the configuration.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/en/getting_started/install_guide.md)    .
 * Supported hardware: Atlas 800I A2
 * Software support: FLUX.1-dev inference engineering repository, commit ID`12e09174353b1bd57bf7fcb80386f59b09fbbefe`

**Note: diffusers only supports > = 0.33. 0 and < = 0.33. 1**

## Operation Procedure

 * Clone the engineering warehouse code.
 * Execution`git checkout 12e09174353b1bd57bf7fcb80386f59b09fbbefe`Switch to the specified version.
 * Complete the subsequent configuration and quantization steps.

**Note: If the specified version is not used, compatibility issues or abnormal functions may occur.**

## Supported Model Versions and Quantification Policies

| Model Series | Model version | HuggingFace Link                                                             | time step quantization | FA3 Quantification | outlier suppression quantization | Quantization command                                                                                                                                                                |
| ------------ | ------------- | ---------------------------------------------------------------------------- | ---------------------- | ------------------ | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FLUX**     | FLUX.1-dev    | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main)     | ✅                      | ✅                  | ✅                                | [time step quantization](#flux-time-step-quantization)    /[Quantification of FA3](#flux-fa3w8a8-dynamic-quantization)    /[outlier suppression quantization](#flux-abnormal-value-suppression-quantification)     |

**Description:**

 * " indicates that the quantification policy has passed the official verification of msModelSlim. The function is complete and the performance is stable. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the quantization command column to go to the specific quantization command.

## Use Example

### FLUX time step quantization

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

For example, add the following code to the __call__ function of the FluxPipeline class of FLUX.1-dev/FLUX1dev/pipeline/pipeline_flux.py:

```python
with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i,t in enumerate(timesteps):
        if self.interrupt:
            continue
        #-----------New Code -----------
        from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
        TimestepManager.set_timestep_idx(i)
        #-----------New Code -----------
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
```

#### Quantization Start Command

For details about the startup command example, see. (Ensure that the permission on calib_prompts.txt is not greater than '0o640'.)

```shell
#do quant
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

############################ Start Quantization ############################
#Load calibration data
calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')

safetensors_name = get_rank_suffix_file('quant_model_weight_w8a8_timestep', 'safetensors', is_distributed, rank)
json_name = get_rank_suffix_file('quant_model_description_w8a8_timestep', 'json', is_distributed, rank)
#Quantized configuration
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

#Data type verification in the pydantic library
session_cfg.model_validate(session_cfg)

#quantization model
quant_model(model, session_cfg)
```

### FLUX FA3+W8A8 dynamic quantization

The FA3+W8A8 dynamic quantization of the model has been integrated into one-click quantization.

#### Use the config_path parameter to specify the configuration file for one-click quantization

```bash
msmodelslim quant \
    --model_path /path/to/flux1_float_weights \
    --save_path /path/to/flux1_quantized_weights \
    --device npu \
    --model_type FLUX.1-dev \
    --config_path /lab_practice/flux1/flux1_w8a8f8_mxfp.yaml \
    --trust_remote_code True
```

#### Script Quantization Start Command

A complete sample of the quantification startup script is provided:[Flux/inference_flux.py](./inference_flux.py)    For details about the startup command, see. (Ensure that the permission on calib_prompts.txt is not greater than '0o640'.)

```shell
#do quant
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


pipeline = load_pipeline(...)  #Load model

model = pipeline.transformer

############################ dump Calibration Data ############################
if not os.path.exists(dump_data_path):  #Check whether the calibration data exists. If the data does not exist, dump the data.
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

############################ Start Quantification ############################
#Load calibration data
calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')

safetensors_name = get_rank_suffix_file('quant_model_weight_w8a8_dynamic', 'safetensors', is_distributed, rank)
json_name = get_rank_suffix_file('quant_model_description_w8a8_dynamic', 'json', is_distributed, rank)
#Quantized configuration
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

#Data type verification in the pydantic library
session_cfg.model_validate(session_cfg)

#quantization model
quant_model(model, session_cfg)
```

### Flux Abnormal Value Suppression Quantification

#### Quantizing Start Command

A complete sample of the quantification startup script is provided:[Flux/inference_flux.py](./inference_flux.py)    For details about the startup command, see. (Ensure that the permission on calib_prompts.txt is not greater than '0o640'.)

```shell
#do quant
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
safetensors_name = get_rank_suffix_file('quant_model_weight_w8a8_dynamic', 'safetensors', is_distributed, rank)
json_name = get_rank_suffix_file('quant_model_description_w8a8_dynamic', 'json', is_distributed, rank)
#Quantified configuration
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

#Data type verification in the pydantic library
session_cfg.model_validate(session_cfg)

#Quantification model
quant_model(model, session_cfg)
```

## Appendixes

### Running parameter description

Here's how to use[Flux/inference_flux.py](./inference_flux.py)    This topic describes the parameters for inference and quantification of the FLUX.1-dev model. For details about parameters that are not involved in the quantization startup command, see the FLUX.1-dev inference engineering repository.[MindIE/FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev)    

| Parameter name           | Meaning:                                                         | Use Restrictions                                                                                                                                                                                                                                                   |
| ------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| path                     | FLUX.1-dev Original Floating Point Model Path                    | Mandatory. Data type: string. No default value.                                                                                                                                                                                                                    |
| save_path                | Save Image Path                                                  | Optional. Data type: String. The default value is ./res.                                                                                                                                                                                                           |
| device_id                | Inference device ID.                                             | Optional. Data type: integer. Default value: 0.                                                                                                                                                                                                                    |
| device                   | Inference Device Type                                            | Optional. Data type: String. The default value is npu. The value can be npu or cpu.                                                                                                                                                                                |
| prompt_path              | List file path for text description prompts for image generation | Optional. Data type: String. The default value is ./calib_prompts.txt.                                                                                                                                                                                             |
| prompt_type              | Specify the type of the inference prompt word.                   | Optional. Data type: String. The default value is plain. The options are plain, parti, and hpsv2.                                                                                                                                                                  |
| num_images_per_prompt    | Number of images generated per prompt word                       | Optional. Data type: integer. Default value: 1.                                                                                                                                                                                                                    |
| max_num_prompt           | Limit the number of prompt words (0 indicates no limit.)         | Optional. Data type: integer. Default value: 0.                                                                                                                                                                                                                    |
| info_file_save_path      | Path for saving image information.                               | Optional. Data type: String. The default value is ./image_info.json.                                                                                                                                                                                               |
| width                    | Width of the image generation                                    | Optional. Data type: integer. The default value is 1024.                                                                                                                                                                                                           |
| height                   | Height of image generation                                       | Optional. Data type: integer. The default value is 1024.                                                                                                                                                                                                           |
| infer_steps              | Flux image inference steps                                       | Optional. Data type: integer. Default value: 50.                                                                                                                                                                                                                   |
| seed                     | Set the random seed of the prompt word.                          | Optional. Data type: integer. Default value: 42.                                                                                                                                                                                                                   |
| use_cache                | Whether to enable dit cache approximation optimization           | Optional. Data type: Boolean. The default value is False.. If only --use_cache is explicitly passed in, becomes True.                                                                                                                                              |
| batch_size               | Specify the batch size of the prompt.                            | Optional. Data type: integer. The default value is 1. Note: If the value is greater than 1, the message is sent to the pipeline in list mode.                                                                                                                      |
| device_type              | Indicates the device type.                                       | Optional. Data type: String. The default value is A2-64g. Values: 'A2-32g-single', 'A2-32g-dual', or 'A2-64g'.                                                                                                                                                     |
| do_quant                 | Quantified or not                                                | Mandatory. Data type: Boolean. The default value is False, indicating that quantization is not enabled. If --do_quant is explicitly transferred, the value becomes True. This parameter must be enabled during inference and quantization of the Flux.1-dev model. |
| quant_type               | Refers to the quantitative type                                  | Optional. Data type: String. The default value is w8a8_timestep. The options are w8a8_timestep, w8a8_dynamic_fa3, or w8a8_dynamic.                                                                                                                                 |
| anti_method              | Specify the method for suppressing abnormal values.              | Optional. Data type: String. The default value is None. The options are "m3", "m4", or "m6".                                                                                                                                                                       |
| quant_weight_save_folder | Indicates the folder for saving quantitative weights.            | Mandatory. Data type: character string. No default value is available.                                                                                                                                                                                             |
| quant_dump_calib_folder  | Refers to the folder for saving quantitative calibration data.   | Mandatory. Data type: character string. No default value is available.                                                                                                                                                                                             |
| data_split_num           | Number of data fragments                                         | Optional. Data type: integer. Default value: 1.                                                                                                                                                                                                                    |
| data_split_id            | Data shard ID.                                                   | Optional. Data type: integer. Default value: 0.                                                                                                                                                                                                                    |
| do_save_img              | Indicates whether to save the inference image.                   | Optional. Data type: Boolean. The default value is False, indicating that the inference image saving function is not enabled. If --do_save_img is explicitly transferred, the value becomes True and the image is saved.                                           |
