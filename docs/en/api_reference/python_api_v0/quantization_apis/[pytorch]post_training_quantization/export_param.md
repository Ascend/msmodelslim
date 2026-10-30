# export_param

## Function

Used to save the quantization parameters of inputs and weights for Conv2dQuantizer and LinearQuantizer as .npy files. These quantization parameters are used for subsequent inference, including the quantization input scale (input_scale), quantization input offset (input_offset), quantization weight scale (weight_scale), quantization weight offset (weight_offset), and quantization weight (quant_weight).

## Prototype

```python
export_param(save_path)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| save_path | Input | The save path for quantization parameters. | Required. Data type: string. |

## Sample

```python
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator
disable_names = []
input_shape = [1, 3, 224, 224]
quant_config = QuantConfig(disable_names=disable_names, amp_num=0, input_shape=input_shape)
calibrator = Calibrator(model, quant_config)
calibrator.run()
calibrator.export_param("./output")
```
