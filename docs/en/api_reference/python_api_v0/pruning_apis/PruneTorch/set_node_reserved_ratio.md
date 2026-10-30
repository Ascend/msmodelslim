# set_node_reserved_ratio

## Function

A method of the PruneTorch class that configures the parameter ratio of operator nodes to be retained during model pruning.

## Prototype

```python
set_node_reserved_ratio(node_reserved_ratio)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ |--------------------------|
| node_reserved_ratio | Input | The maximum ratio of operator nodes to be pruned during the pruning process. | Required.<br>Data type: Float.<br>Value range: 0-1. |

## Sample

```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
model = torchvision.models.vgg16(pretrained=False)
model.eval()
prune_torch= PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
prune_torch= prune_torch.set_node_reserved_ratio(0.5)
```
