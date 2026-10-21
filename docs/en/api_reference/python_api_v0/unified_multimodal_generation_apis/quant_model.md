# quant_model()

## Function

In multimodal generative model quantization, a unified quantization API must be called. This function invokes the core quantization logic based on the quantization session configuration to complete quantization.

## Prototype

```python
quant_model(model: nn.Module, session_cfg: SessionConfig)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ---------- | ---- | -------- |
| model | Input | The part of the multimodal generative model to be quantized. | Required.<br>Data type: nn.Module. Currently only the transformer part of the multimodal generative model is supported for quantization. After loading the full pipeline, select pipeline.transformer as the model. |
| session_cfg | Input | The quantization session configuration class, used to configure quantization-related parameters, calibration data, and runtime device. | Required.<br>Data type: SessionConfig. |

## Sample

```python
import torch
from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant.session.session import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig
from msmodelslim.quant.session.session import SessionConfig, quant_model

session_config = SessionConfig(
    processor_cfg_map={
        "w8a8": W8A8ProcessorConfig(
            cfg=W8A8QuantConfig(
                act_method='minmax'
            ),
            disable_names=[]
        ),
        "save": SaveProcessorConfig(
            output_path="./",
            safetensors_name=None,
            json_name=None,
            save_type=['safe_tensor'],
            part_file_size=None
        )
    },
    calib_data=safe_torch_load("calib_data.pth"),
    device="npu"
)

# Load the pipeline.
pipeline = load_pipeline(...)

model = pipeline.transformer

quant_model(model, session_config)
```
