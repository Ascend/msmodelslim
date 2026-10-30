# create_quant_config

## Function

Post-training quantization API. It finds all quantizable layers based on the graph structure, automatically generates a quantization configuration file, and writes the quantization configuration information of the quantizable layers into the configuration file.

## Prototype

```python
create_quant_config(config_file, model)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| config_file | Input | The path and name for the quantization configuration file to be generated. Required. | Data type: string. The configuration file must have a .json suffix. If a file with the same name already exists in the specified path, calling this API will overwrite the existing file. |
| model | Input | The model instance to be quantized. Required. | Data type: MindSpore model. |

## Sample

```python
from msmodelslim.mindspore.quant.ptq_quant.create_config import create_quant_config
model = SampleModel()
config_file = "./test_config_file.json"
create_quant_config(config_file, model)
```
