# preprocess_func_coco

## Function

A function for image preprocessing, primarily used to preprocess the COCO dataset. It reads image files into memory, resizes them to the specified height and width, converts them to floating-point type, normalizes pixel values to the range of 0 to 1, and returns image data with the specified batch size.

## Prototype

```python
preprocess_func_coco(data_path, height=320, width=320, batch_size=1)
```

## Parameters

| Parameter | Input/Return | Description| Constraints|
| ------ | ------ | ------ | ------ |
| data_path | Input| Path to the dataset to be processed.| Required.<br>Data Type: String.|
| height | Input| Image height.| Optional, default is 320.<br>Data Type: int.<br>Value Range: integer greater than 0.|
| width | Input| Image width.| Optional, default is 320.<br>Data Type: int.<br>Value Range: integer greater than 0.|
| batch_size | Input| Number of images used per batch.| Optional, default is 1.<br>Data Type: int.<br>Value Range: integer greater than 0.|

## Sample

```python
from msmodelslim.onnx.post_training_quant import QuantConfig
from msmodelslim.onnx.post_training_quant.label_free.preprocess_func import preprocess_func_coco
calib_data = preprocess_func_coco("./test/")
quant_config = QuantConfig(calib_data = calib_data, amp_num = 5)
```
