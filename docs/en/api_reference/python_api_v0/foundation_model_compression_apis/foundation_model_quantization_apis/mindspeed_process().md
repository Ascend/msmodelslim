# process()

## Function

Uses a calibration dataset to perform an outlier suppression process, modifying the weights in the model to improve subsequent model quantization accuracy. This function takes no arguments.

## Prototype

```python
process()
```

## Sample

Configure all parameters in the AntiOutlierConfig initialization based on actual requirements.

```python
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig
from msmodelslim.pytorch.mindspeed_adapter import ModelAdapter, AntiOutlierAdapter, CalibratorAdapter
model = ModelAdapter(model)
anti_config = AntiOutlierConfig(anti_method="m5", dev_type='npu')
anti_outlier = AntiOutlierAdapter(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process() 
calibrator = CalibratorAdapter(model, quant_config, calib_data=dataset_calib, disable_level='L0') 
calibrator.run(int_infer=False) 
calibrator.save(quant_weight_save_path)
```
