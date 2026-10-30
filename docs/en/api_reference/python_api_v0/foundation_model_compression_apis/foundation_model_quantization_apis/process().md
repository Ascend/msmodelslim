# process()

## Function

Performs outlier suppression with no input parameters.

## Sample

Configure all parameters during QuantConfig initialization based on actual requirements.

```python
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
anti_config = AntiOutlierConfig(anti_method="m2")
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process() 
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0') 
calibrator.run(int_infer=False) 
calibrator.save(quant_weight_save_path)
```
