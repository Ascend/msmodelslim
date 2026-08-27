# FakeQuantizeCalibrator

## Function

Converts a floating-point model to a fake quantization model based on quantization weights.

## Prototype

```python
FakeQuantizeCalibrator(model, dev_id, dev_type, description, safetensor)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The model. | Required.<br>Data type: nn.Module. |
| dev_id | Input | Device ID. | Required.<br>Data type: int.<br>Takes effect only when "dev_type" is configured as "npu". The Device ID specified by "dev_id" takes precedence over the Device ID configured by environment variables. |
| dev_type | Input | Device type. | Required.<br>Data type: str.<br>Options: ['cpu', 'npu'], default is 'cpu'. |
| description | Input | The JSON description file generated after quantization. | Required.<br>Data type: dict. |
| safetensor | Input | The weight file in safetensors format generated after quantization. | Required.<br>Data type: dict. |

## Sample

Configure all parameters in the QuantConfig initialization according to actual requirements.

```python
import torch
import json
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import FakeQuantizeCalibrator
if __name__ == '__main__':
    fp16_path = './chatglm2_6b/'  # File path
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=fp16_path, 
                                                 local_files_only=True, 
                                                 torch_dtype=torch.float32).cpu()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=fp16_path, local_files_only=True,)
    safetensor_dict = load_file('./quant_model_weight_w8a16.safetensors')  # Use the load_file() function to read a safetensor format file and parse it into a dictionary.
    with open('./quant_model_description_w8a16.json', 'r', encoding='utf-8') as file:
        description_dict = json.load(file)  # Use the json.load() function to read a file and parse it into a dictionary.
    fakecalibrator = FakeQuantizeCalibrator(model, None, "cpu", description_dict, safetensor_dict)
    model = fakecalibrator.model
```

## FAQ

1. The V0 pseudo-quantization interface has known accuracy issues. As V0 is no longer under active development, this will be addressed in the new pseudo-quantization implementation based on the V1 architecture.
