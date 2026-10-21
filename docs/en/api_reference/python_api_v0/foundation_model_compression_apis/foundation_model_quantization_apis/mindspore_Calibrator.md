# Calibrator

## Function

Quantization parameter configuration class that encapsulates quantization algorithms through the Calibrator class.

## Prototype

```python
Calibrator(cfg: QuantConfig, model, model_ckpt, calib_data=None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| cfg | Input | The configured QuantConfig class. | Required.<br>Data type: QuantConfig. |
| model | Input | Model. | Required.<br>Data type: MindFormer model. |
| model_ckpt | Input | The ckpt file of model weights. | Required.<br>Data type: str. |
| calib_data | Input | Data for calibrating the Large Language Model (LLM) quantization. Input real data for quantization. | Optional.<br>Data type: object.<br>Default value: None.<br>Input template: \[[input1],[input2],[input3]]. |

## Sample

```python
from msmodelslim.mindspore.llm_ptq import Calibrator, QuantConfig
quant_config = QuantConfig(disable_names=["lm_head"], fraction=0.01)
model = Model()  # Load according to the actual model situation.
calibrator = Calibrator(cfg=quant_config, model=model, model_ckpt="./model.ckpt", calib_data=dataset_calib)
```
