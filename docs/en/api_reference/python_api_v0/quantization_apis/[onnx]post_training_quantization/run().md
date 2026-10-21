# run()

## Function

Runs the quantization algorithm. After initializing OnnxCalibrator, execute quantization via the run() function.

## Prototype

```python
from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig 
quant_config = QuantConfig(disable_names=[],
                     quant_mode=0,
                     amp_num=0)
input_model_path="/home/xxx/Resnet50/resnet50_pytorch.onnx"   # Configure according to the actual path of the model.
output_model_path="/home/xxx/Resnet50/resnet50_quant.onnx"    # Configure according to the actual path of the model.
calibrator = OnnxCalibrator(input_model_path, quant_config)
calibrator.run() 
calibrator.export_quant_onnx(output_model_path)
```
