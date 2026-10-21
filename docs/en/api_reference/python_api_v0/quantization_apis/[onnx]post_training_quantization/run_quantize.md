# run_quantize

## Function

Model quantization API. Quantizes the user-provided model according to the configured quantization parameters and saves the quantized model.

## Prototype

```python
run_quantize(input_model_path, output_model_path, quant_config)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| input_model_path | Input | Path and filename of the model to be quantized. | Required.<br>Data type: String. |
| output_model_path | Input | Path and filename of the quantized model. | Required.<br>Data type: String. |
| quant_config | Input | Quantization configuration instance generated based on QuantConfig. | Required.<br>Data type: QuantConfig. |

## Sample

```python
from msmodelslim.onnx.post_training_quant import QuantConfig, run_quantize
def custom_read_data():
    calib_data = []
    # TODO: Load the dataset, perform data preprocessing, and store the data into calib_data.
    return calib_data
calib_data = custom_read_data() 
quant_config = QuantConfig(calib_data=calib_data, amp_num=5)
input_model_path="/home/xxx/Resnet50/resnet50_pytorch.onnx"   # Configure according to the actual path of the model.
output_model_path="/home/xxx/Resnet50/resnet50_quant.onnx"    # Configure according to the actual path of the model.
run_quantize(input_model_path,output_model_path,quant_config)
```
