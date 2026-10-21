# run()

## Function

Runs the quantization algorithm. After initializing the Calibrator, execute quantization through the run() function.

## Prototype

```python
calibrator.run(int_infer=False)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| int_infer | Input | Whether to use int8matmul for pseudo-quantization calculation. | Optional.<br>Data type: bool.<br>Default value: False.<br>This parameter is only applicable to W8A8 scenarios and is invalid in W8A16 scenarios. |

## Sample

Configure all parameters in QuantConfig initialization according to actual requirements.

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(dev_type='cpu', pr=0.5, mm_tensor=False)
model = AutoModel.from_pretrained('/chatglm2-6b', 
                                  local_files_only=True, 
                                  torch_dtype=torch.float32).cpu()   # Configure according to the actual path of the model.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run(int_infer=False) 
calibrator.save(quant_weight_save_path)
```
