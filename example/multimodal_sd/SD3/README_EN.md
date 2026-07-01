# SD3-Medium Quantization Instructions

Currently, W8A8 static quantization is supported only for the transformer part of the SD3 model.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/en/getting_started/install_guide.md)    .

## Supported Model Versions and Quantification Policies

| Model Series | Model version | HuggingFace Link                                                                      | W8A8 | W8A16 | W4A16 | W4A4 | time step quantization | FA3 Quantification | outlier suppression quantization | Quantization command        |
| ------------ | ------------- | ------------------------------------------------------------------------------------- | ---- | ----- | ----- | ---- | ---------------------- | ------------------ | -------------------------------- | --------------------------- |
| **SD3**      | SD3-Medium    | [SD3-Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)     | ✅    |       |       |      |                        |                    |                                  | [W8A8](#sd3-medium-w8a8-quantization)     |

**Description:**

 * ' indicates that the quantization policy has passed the official verification of msModelSlim and has complete functions and stable performance. It is recommended that the quantization policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the quantization command column to go to the specific quantization command.

## Use Example

### SD3-Medium W8A8 Quantization

A complete sample of the quantification startup script is provided:[SD3/sd3_inference.py](./sd3_inference.py)    For details about the startup command, see. (Ensure that the permission on calib_prompts.txt is not greater than '0o640'.)

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

#### Sample Code for Calibration Data Dump and Quantization

```python
#Importing a Model Library
import os
import torch
from diffusers import StableDiffusion3Pipeline

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig
from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file

DUMP_CALIB_FOLDER = './results/quant/cache'  #Folder for storing calibration data
SAFE_TENSOR_FOLDER = './results/quant/safetensors'  #Folder for storing quantization models

rank = get_rank()
is_distributed = rank >= 0  #Marking a distributed environment

dump_data_path = os.path.join(DUMP_CALIB_FOLDER, get_rank_suffix_file(base_name="calib_data", ext="pth",
                                                                      is_distributed=is_distributed, rank=rank))

############################ Load Model ############################
def load_t2v_checkpoint(model_path):
    pipeline = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16).to('npu')
    return pipeline


pipeline = load_t2v_checkpoint("/path/to/stable-diffusion-3-medium-diffusers")  #Load Model

model = pipeline.transformer

############################ dump Calibration Data ############################
if not os.path.exists(dump_data_path):  #Check whether the calibration data exists. If the data does not exist, dump the data.
    #Add the forward hook for the forward input of the dump model.
    dumper_manager = DumperManager(model, capture_mode='args')

    #Perform floating-point model inference

    pipeline(
        prompts=["A photo of an astronaut riding a horse on mars"],
        negative_prompts=[""],
        width=1024,
        height=1024,
        num_inference_steps=28,
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

#Data type verification in the Python pydantic library
session_cfg.model_validate(session_cfg)

#quantization model
quant_model(model, session_cfg)
```

## Appendixes

### Running parameter description

Here's how to use[SD3/sd3_inference.py](./sd3_inference.py)    Description of the parameters used for SD3 model inference quantification.

| Parameter name           | Meaning:                                          | Use Restrictions                                                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sd3_model_path           | SD3 Raw Floating Point Model Path                 | Mandatory. Data type: string. No default value.                                                                                                                                                                                                                   |
| prompt_path              | Enter the prompt path.                            | Optional. Data type: String. The default value is ./calib_prompts.txt.                                                                                                                                                                                            |
| width                    | Generate image width                              | Optional. Data type: integer. The default value is 1024.                                                                                                                                                                                                          |
| height                   | Generate Image Height                             | Optional. Data type: integer. The default value is 1024.                                                                                                                                                                                                          |
| infer_steps              | Inference Steps                                   | Optional. Data type: integer. Default value: 28.                                                                                                                                                                                                                  |
| seed                     | prompt: random seed                               | Optional. Data type: integer. Default value: 42.                                                                                                                                                                                                                  |
| device                   | model running equipment                           | Optional. Data type: String. The default value is npu. Currently, only npu is supported.                                                                                                                                                                          |
| save_path                | Path for storing inference images.                | Optional. Data type: String. The default value is ./results. This parameter is valid only when do_save_img is enabled.                                                                                                                                            |
| do_quant                 | Quantification or not                             | Mandatory. Data type: Boolean. The default value is False, indicating that the quantization function is not enabled. If --do_quant is explicitly transferred, the value becomes True. This parameter must be enabled during SD3 model inference and quantization. |
| quant_type               | Quantization type                                 | Optional. Data type: String. The default value is w8a8. Currently, only w8a8 is supported.                                                                                                                                                                        |
| quant_weight_save_folder | Save path of quantization weight.                 | Mandatory. Data type: string. No default value.                                                                                                                                                                                                                   |
| quant_dump_calib_folder  | Path for saving the quantization calibration data | Mandatory. Data type: character string. No default value.                                                                                                                                                                                                         |
| do_save_img              | Indicates whether to save the inference image.    | Optional. Data type: Boolean. The default value is False, indicating that the inference image saving function is not enabled. If --do_save_img is explicitly transferred, the value becomes True and the image is saved.                                          |
