# get_quant_params

## Function

Used to obtain the quantization parameters for the inputs and weights of Conv2dQuantizer and LinearQuantizer, which are used for subsequent inference. These include the quantization input scale (input_scale), quantization input offset (input_offset), quantization weight scale (weight_scale), quantization weight offset (weight_offset), and quantization weight (quant_weight). Finally, these parameters are returned in the form of a dictionary.

## Prototype

```python
get_quant_params()
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| input_scale | Return | Input quantization scale. | Data type: dict. |
| input_offset | Return | Input quantization offset. | Data type: dict. |
| weight_scale | Return | Quantized weight scale. | Data type: dict. |
| weight_offset | Return | Quantized weight offset. | Data type: dict. |
| quant_weight | Return | Quantized weight. | Data type: dict. |

## Sample

```python
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator
disable_names = []
input_shape = [1, 3, 224, 224]
quant_config = QuantConfig(disable_names=disable_names, amp_num=0, input_shape=input_shape)
calibrator = Calibrator(model, quant_config)
calibrator.run()
input_scale, input_offset,weight_scale, weight_offset, quant_weight = calibrator.get_quant_params()
```
