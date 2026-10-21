# save()

## Function

Quantization parameter configuration class, which saves the quantized weights and related parameters by encapsulating the quantization algorithm through the calibrator class.

Note: Because deserialization risks exist when storing quantization parameters, set the permission of the saved quantization result directory to 750, the permission of the quantized weight file to 400, and the permission of the quantized weight description file to 600 during storage to reduce risks.

## Prototype

```python
calibrator.save(output_path="")
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| output_path | Input | The path to save the quantized weights and related parameters. | Required.<br>Data type: string. |

## Sample

```python
from msmodelslim.mindspore.llm_ptq import Calibrator, QuantConfig
quant_config = QuantConfig(disable_names=["lm_head"], fraction=0.01)
model = Model()    # Load according to the actual model situation.
calibrator = Calibrator(cfg=quant_config, model=model, model_ckpt="./model.ckpt", calib_data=dataset_calib)
calibrator.run()
calibrator.save("./quant_model.ckpt")
```
