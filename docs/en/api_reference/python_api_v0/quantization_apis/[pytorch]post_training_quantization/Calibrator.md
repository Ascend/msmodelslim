# Calibrator

## Function

Quantization parameter configuration class that encapsulates quantization algorithms through the Calibrator class.

## Prototype

```python
Calibrator(model, cfg, calib_data=None, fuse_module_call_back=None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The model instance to be quantized. | Required.<br>Data type: PyTorch model. |
| cfg | Input | The configured QuantConfig class. | Required.<br>Data type: QuantConfig. |
| calib_data | Input | Model training data. Real data can be input for Label-Free quantization, or dummy data can be input to achieve Label-Free quantization. | Optional.<br>Data type: list[list[Torch.Tensor]] or list[Torch.Tensor].<br>If no data is input, the Label-Free quantization process will be automatically invoked when the model supports a single float format input and `input_shape` is specified. For models with multiple inputs or those requiring custom input formats, users can randomly construct input data to achieve Label-Free quantization. |
| fuse_module_call_back | Input | User-defined function for Batch Normalization (BN) fusion. This callback is invoked before quantization. | Optional.<br>Data type: function.<br>If the model structure is special and not a conv->bn parallel structure, the user needs to provide a custom fusion function. |

## Sample

```python
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator
disable_names = []
input_shape = [1, 3, 224, 224]
quant_config = QuantConfig(disable_names=disable_names, amp_num=0, input_shape=input_shape)
calib_data = []
image = cv2.imdecode(np.fromfile("./random_image.jpg", dtype=np.uint8), 1)
image = cv2.resize(image, (224, 224,), interpolation=cv2.INTER_CUBIC)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
image = torch.from_numpy(image).permute(2, 0, 1)/255
image = image.unsqueeze(0)
calib_data.append([image])     # Pass in a random image to improve accuracy.
calibrator = Calibrator(model, quant_config, calib_data=calib_data)
```
