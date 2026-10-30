# preprocess_func_imagenet

## Function

A function for image preprocessing, primarily used to preprocess the ImageNet dataset. It loads images, resizes them, converts color space, normalizes, and returns a batch of image data.

## Prototype

```python
preprocess_func_imagenet(data_path, height=224, width=224, batch_size=1)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| data_path | Input | Path to the dataset to be processed. | Required.<br>Data type: String. |
| height | Input | Image height. | Optional, default is 224.<br>Data type: int.<br>Value range: Integer > 0. |
| width | Input | Image width. | Optional, default is 224.<br>Data type: int.<br>Value range: Integer > 0. |
| batch_size | Input | Number of images per batch. | Optional, default is 1.<br>Data type: int.<br>Value range: Integer > 0. |

## Sample

```python
from msmodelslim.onnx.post_training_quant import QuantConfig
from msmodelslim.onnx.post_training_quant.label_free.preprocess_func import preprocess_func_imagenet
calib_data = preprocess_func_imagenet("./test/")
quant_config = QuantConfig(calib_data = calib_data, amp_num = 5)
```
