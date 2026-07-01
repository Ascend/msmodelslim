# CalibratorAdapter

## Function

A quantization parameter configuration class for MindSpeed-LLM models, inheriting from Calibrator. Its external interface is consistent with Calibrator, and it does not support CPU-based quantization execution.

## Inheritance Relationship

CalibratorAdapter inherits from the Calibrator class and maintains the same external interface as Calibrator, including:

- run(): Executes the quantization process

- save(): Saves the quantized model and weights

## Prototype

```python
CalibratorAdapter(model, cfg: QuantConfig, calib_data=None, disable_level='L0', all_tensors=None, mix_cfg: Optional[dict] = None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The model. | Required.<br>Data type: PyTorch model. |
| cfg | Input | The configured QuantConfig class. | Required.<br>Data type: QuantConfig. |
| calib_data | Input | Data for LLM quantization calibration. Input real data for Label-Free quantization. | Optional.<br>Data type: object.<br>Default value: None, which is for the Data-Free scenario. It must be provided for the Label-Free scenario.<br>Input template: \[[input1],[input2],[input3]]. |
| disable_level | Input | Automatic fallback level. When model accuracy loss is significant, the level can be appropriately increased, but the number of fallback layers cannot exceed the total number of model layers. | Optional.<br>Data type: string.<br>Configuration examples are as follows: (1) 'L0': Default value, no fallback is performed. (2) 'L1': Fall back 1 layer. (3) 'L2': Fall back 2 layers. (4) 'L3': Fall back 3 layers. (5) 'L4': Fall back 4 layers. (6) 'L5': Fall back 5 layers.<br>And so on. |
| all_tensors | Input | {name:tensor} used for layer-by-layer quantization calibration. | Optional.<br>Data type: dict.<br>Default value: None. The default configuration can be used. |
| mix_cfg | Input | Mixed quantization configuration, specifying {wildcard or layer name of **nn.Module**: quantization type}.<br>Key: **Wildcards are case-sensitive. The implementation uses Python's standard library fnmatch, so please refer to fnmatch for related usage**. | Optional.<br>Data type: dict.<br>Default value: None.<br>Configuration example: {"\*down\*": "w8a16", "\*": "w8a8"}<br>**The quantization type (the value of the dictionary) must be an internally recognized configuration, otherwise an error will be reported.**<br>Currently supports w8a8, w8a16, w8a8_dynamic. |

## Sample

Configure all parameters in the QuantConfig initialization based on actual requirements.

```python
from msmodelslim.pytorch.mindspeed_adapter import ModelAdapter, CalibratorAdapter
model = ModelAdapter(model)
quant_config = QuantConfig(dev_type='npu', pr=0.5, mm_tensor=False)
calibrator = CalibratorAdapter(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()  # Same usage as Calibrator.
calibrator.save(quant_weight_save_path)  # Same usage as Calibrator.
```
