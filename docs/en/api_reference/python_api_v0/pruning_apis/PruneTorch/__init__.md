# __init__

## Function

PruneTorch class method, which initializes the class for the user-input model.

## Prototype

```python
__init__(network, inputs)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ |-------------------------|
| network | Input | The model instance to be pruned. | Required.<br>Data type: PyTorch model. |
| inputs | Input | The input data of the model, used for parsing the model. | Optional.<br>Data type: Tensor. |

## Sample

```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
model = torchvision.models.vgg16(pretrained=False)
model.eval()
prune_torch = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
```
