# save_qsin_qat_model

## Function

Quantized model saving API, which saves the quantized model as an .onnx model that can be used for inference on Ascend hardware.

## Prototype

```python
save_qsin_qat_model(model, save_onnx_name, dummy_input, saved_ckpt, input_names)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
|----------------| ------ | ------ |---------------------------|
| model | Input | The model instance to be quantized. | Required.<br>Data type: PyTorch model. |
| save_onnx_name | Input | The name of the .onnx model saved after quantization. | Required.<br>Data type: str. |
| dummy_input | Input | The input shape of the quantized model. | Required.<br>Data type: torch.Tensor. |
| saved_ckpt | Input | The saved quantization weights. | Required.<br>Data type: str. |
| input_names | Input | The input names for onnx. | Required.<br>Data type: list[str]. |

## Sample

```python
from msmodelslim.pytorch.quant.qat_tools import save_qsin_qat_model
save_onnx_name='./dest.onnx'      # Modify file name and path according to actual needs.
dummy_input = torch.ones([batch_size, 3, 224, 224]).type(torch.float32)
saved_ckpt = './saved_ckpt.pth'     # Modify file name and path according to actual needs.
input_names=['input']        # Modify file name according to actual needs.
save_qsin_qat_model(model, save_onnx_name, dummy_input, saved_ckpt, input_names)
```
