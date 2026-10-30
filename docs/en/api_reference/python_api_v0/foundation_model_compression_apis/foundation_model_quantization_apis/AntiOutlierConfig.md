# AntiOutlierConfig

## Function

Constructs a config for outlier suppression.

## Prototype

```python
AntiOutlierConfig(w_bit=8, a_bit=8, anti_method="m2", dev_type="cpu", dev_id=None, w_sym=True)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| w_bit | Input | Weight quantization bits. | Optional.<br>Data type: int.<br>Default is 8, modification is not supported. |
| a_bit | Input | Activation quantization bits. | Optional.<br>Data type: int.<br>Default is 8.<br>When anti_method is m3, modification to 16 is supported. |
| anti_method | Input | The method used for outlier suppression (anti_outlier). | Optional.<br>Data type: string.<br>Default is m2, optional values are 'm1', 'm2', 'm3', 'm4', 'm5', or 'm6'.<br>(1) 'm1': Outlier suppression method 1.<br>(2) 'm2': Outlier suppression method 2, recommended.<br>(3) 'm3': AWQ algorithm.<br>(4) 'm4': Smooth optimization algorithm.<br>(5) 'm5': CBQ quantization algorithm.<br>(6) 'm6': Flex smooth quantization algorithm.<br>Note: The m4 method does not support quantization scenarios for the telechat model. When the m3 method processes MOE models, it does not perform any processing on the expert structure. The m2 method does not support the w8a8-pertoken scenario for MOE models, and currently has been adapted for qwen-vl and llava-v1.5-7b multimodal models. |
| disable_anti_names | Input | Specifies certain layers to skip outlier suppression. | Optional.<br>Data type: list.<br>Default is [].<br>Takes effect only when anti_method is set to m6, and only supports passing in o layers. |
| flex_config | Input | Configuration file for the m6 method. | Optional.<br>Data type: dict.<br>Takes effect only when anti_method is set to m6. Contains two configuration items: alpha and beta, both of type float, with a value range of [0, 1], and a default value of None. Used to control the smoothness of the algorithm. If both alpha and beta are specified with specific values, those values are used directly; if either value is unspecified (None), the algorithm will automatically perform optimization to calculate the optimal alpha and beta values for outlier suppression. |
| dev_type | Input | Device type. | Optional.<br>Data type: string.<br>Optional values: ['cpu', 'npu'], default is 'cpu'. |
| dev_id | Input | Device ID. | Optional.<br>Data type: int.<br>Default is None.<br>Takes effect only when "dev_type" is configured as "npu". The Device ID specified by "dev_id" has a higher priority than the Device ID configured by environment variables. |
| w_sym | Input | Whether weights are symmetrically quantized. | Optional.<br>Data type: bool.<br>Default is True.<br>When anti_method is set to m3, it can be set to False, and must be consistent with the w_sym parameter setting in QuantConfig. |

## Sample

According to actual requirements, complete the configuration of all parameters during QuantConfig initialization.

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

When anti_method=m6:

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
# Specify that layer o does not perform outlier suppression.
keys = ['.o_proj']
disable_names = []
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        for key in keys:
            if key in name:
                disable_names.append(name)
# If both alpha and beta are set to specific values, configure directly using the specified values.
anti_config = AntiOutlierConfig(anti_method='m6',
                                disable_anti_names=disable_names,
                                flex_config={'alpha': 0.75,
                                             'beta': 0.1})
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process() 
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0') 
calibrator.run(int_infer=False) 
calibrator.save(quant_weight_save_path)
```
