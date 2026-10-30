# qsin_qat

## Function

Model quantization API. Quantizes the user-provided model according to the configured quantization parameters.

## Prototype

```python
qsin_qat(model, quant_config, quant_logger)
```

## Parameters

| Parameter     | Input/Return | Description                                                      | Constraints                   |
|---------| ------ |---------------------------------------------------------|------------------------|
| model   | Input | The model instance to be quantized. | Required.<br>Data type: PyTorch model. |
|quant_config|Input|Quantization parameter configuration. | Required.<br>Data type: QatConfig.    |
|quant_logger|Input|Quantization output log. | Required.<br>Data type: log.       |

## Sample

```python
from msmodelslim.pytorch.quant.qat_tools import qsin_qat, QatConfig, get_logger
from torchvision.models import resnet50
import torch
model=resnet50()
quant_config = QatConfig()
quant_logger = get_logger()
model = qsin_qat(model, quant_config, quant_logger)
```
