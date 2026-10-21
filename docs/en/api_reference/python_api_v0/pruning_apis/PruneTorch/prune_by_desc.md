# prune_by_desc

## Function

Prunes the model during inference based on existing pruning information.

## Prototype

```python
prune_by_desc(desc)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ----- |--------------------------|
| desc | Input | Pruning information. | Required.<br>Data type: desc information returned by the prune or analysis API. |

## Sample

```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
model = torchvision.models.vgg16(pretrained=False)
model.eval()
prune_torch = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
desc = prune_torch.prune()
prune_torch.prune_by_desc(desc)
```
