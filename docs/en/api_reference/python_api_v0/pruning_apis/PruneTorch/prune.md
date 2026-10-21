# prune

## Function

Pruning function that configures various parameters during the pruning process and returns pruning information, which can be used to perform pruning during evaluation.

## Prototype

```python
prune(reserved_ratio=0.75, un_prune_list=None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ |------------------------------------------------------------------------------------------------------------------------|
| reserved_ratio | Input | The retention ratio of pruning parameters. | Optional.<br>Data type: Float.<br>Default value: 0.75. Value range: [0, 1]. |
| un_prune_list | Input | Specifies layers that are not pruned. By default, the first and last layers are not pruned. | Optional.<br>Data type: list. Elements must be int or string.<br>Default value: None.<br>If an element is int, it indicates the layer index not to be pruned (only Conv2d and Linear operators to be pruned are counted).<br>If an element is string, it indicates the name of the operator in the network. |

## Sample

```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
model = torchvision.models.vgg16(pretrained=False)
model.eval()
prune_torch = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
desc = prune_torch.prune(0.5)
```
