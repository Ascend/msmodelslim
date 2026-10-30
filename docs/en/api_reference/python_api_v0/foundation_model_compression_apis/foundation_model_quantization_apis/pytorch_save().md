# save()

## Function

Quantization parameter configuration class, which encapsulates the quantization algorithm through the calibrator class to save the quantized weights and related parameters.

Note: Due to the risk of deserialization during the storage of quantization parameters, this risk has been mitigated by setting the permissions of the saved quantization result folder to 750, the quantized weight file to 400, and the quantized weight description file to 600 during the storage process.

## Prototype

```python
calibrator.save(output_path, safetensors_name=None, json_name=None, save_type=None, part_file_size=None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| output_path | Input | The save path for the quantized weights and related parameters. | Required.<br>Data type: string. |
| safetensors_name | Input | The name of the quantized weight file in safetensors format. | Optional.<br>Data type: string. |
| json_name | Input | The name of the JSON description file for the quantized weights in safetensors format. | Optional.<br>Data type: string. |
| part_file_size | Input | Size of each part when the weights are saved as safetensors files, in GB. | Optional.<br>Data type: int.<br>Defaults to None, which disables the sharded saving function. |
| save_type | Input | The save format for the quantized weights. | Optional.<br>Data type: list, element type: string.<br>The default value is ["safe_tensor"], meaning the quantized weights are saved in safetensors format. |

## Additional Parameters

- part_file_size
<br>When this parameter is set to an integer greater than 0, the sharded saving function is enabled. The weights are sharded according to the value set by the user (in GB), and the actually saved weights may be slightly larger than the set value.
- save_type
<br>This parameter supports three formats: "numpy", "safe_tensor", or "ascendV1". Users should choose based on the actual situation of model weight saving:
<br>(1) When set to "numpy", only the quantized weight file in npy format is exported. `Note: When the quantization type is W4A8_DYNAMIC, setting the "numpy" format for saving will cause an error.`
<br>(2) When set to "safe_tensor", only the safetensors quantized weight file and the json description file are exported. The safetensors weight file contains floating-point and quantized weights, including the quantized weights used by quantized layers and the original floating-point weights used by unquantized layers. The json description file contains all modules of the model and indicates the quantization or floating-point type of each module, such as FLOAT, W8A8, W8A16.
<br>(3) When set to "ascendV1", the model weights remain consistent with those exported using safe_tensor, but config.json and quant_model_description_{quant_type}.json will have slight modifications. `Note: When the quantization type is W4A16, weights are packed by default.` The modifications to the configuration files are as follows: The content of the config.json file remains consistent with the floating-point weights; meanwhile, the quant_model_description_{quant_type}.json file is renamed to quant_model_description.json, and a version description is added.

## Sample

Configure all parameters in the QuantConfig initialization according to actual needs.

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(dev_type='cpu', pr=0.5, mm_tensor=False)
model = AutoModel.from_pretrained('/chatglm2-6b', local_files_only=True, torch_dtype=torch.float32).cpu()   # Configure according to the actual path of the model.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run(int_infer=False)
quant_weight_save_path = "/path/to/save_path" # Configure according to the actual path.
calibrator.save(quant_weight_save_path)
```
