# export_quant_safetensor()

## Function

Quantization parameter configuration class that saves quantized weights and related parameters through the calibrator class encapsulating quantization algorithms.

Note: Considering the deserialization risk during the storage of quantization parameters, we have taken targeted measures in the storage process: setting the permission of the quantization result folder to 750, the permission of the quantized weight file to 400, and the permission of the quantized weight description file to 600 to mitigate this risk.

## Prototype

```python
calibrator.export_quant_safetensor(output_path, safetensors_name=None, json_name=None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| output_path | Input | The save path for the quantized weights and related parameters. | Required.<br>Data type: string. |
| safetensors_name | Input | The name of the safetensors format quantized weight file. | Optional.<br>Data type: string. |
| json_name | Input | The name of the safetensors format quantized weight JSON description file. | Optional.<br>Data type: string. |

The safetensors format weight file and JSON description file. The safetensors weight file contains floating-point and quantized weights, with quantized weights used for quantized layers and original floating-point weights used for unquantized layers. The JSON description file contains all modules of the model and indicates the quantization or floating-point type of each module, such as FLOAT or W8A8. The safetensors export format is required when exporting parameters for a multimodal quantized model for subsequent inference.

## Sample

Configure all parameters in the QuantConfig initialization according to actual requirements.

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(act_method=1, quant_mode=1,device="npu")
pipe = OpenSoraPipeline12.from_pretrained("open-sora/", local_files_only=True)
pipe = compile_pipe(pipe)
model = pipe.transformer   # Configure according to the actual path of the model.
calibrator = Calibrator(model, quant_config, calib_dataset)
calibrator.run()
calibrator.export_quant_safetensor("/output_path/")
```
