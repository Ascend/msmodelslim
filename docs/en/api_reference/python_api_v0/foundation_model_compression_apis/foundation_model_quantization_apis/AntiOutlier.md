# AntiOutlier

## Function

Constructs a class for outlier suppression, passing in the model, outlier suppression config, calibration data, and other parameters.

## Prototype

```python
AntiOutlier(model, calib_data=None, cfg=None, norm_class_name = None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The model used for large model outlier suppression. | Required.<br>Data type: PyTorch model. |
| calib_data | Input | Calibration data used for outlier suppression. | Optional.<br>Data type: object.<br>Default value: None.<br>Input template: \[[input1],[input2],[input3]]. |
| cfg | Input | The configured AntiOutlierConfig class. | Optional.<br>Data type: Config. |
| norm_class_name | Input | User-defined norm class name. | Optional.<br>Data type: str.<br>Default: None. If the system fails to automatically identify the norm, the user needs to manually input a custom norm class name, for example, norm_class_name = 'LlamaRMSNorm'. |

## Sample

Complete all parameter configurations during AntiOutlier initialization based on actual requirements.

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
anti_config = AntiOutlierConfig(anti_method="m2")
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process() 
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0') 
calibrator.run(int_infer=False) 
calibrator.save(quant_weight_save_path)
```
