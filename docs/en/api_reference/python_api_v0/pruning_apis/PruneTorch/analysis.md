# analysis

## Function

Pruning analysis function. It performs only the analysis operations during the pruning process and returns the number of parameters after pruning along with pruning information.

## Prototype

```python
analysis(reserved_ratio=0.75, un_prune_list=None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| reserved_ratio | Input | The retention ratio of the number of parameters after pruning. | Optional.<br>Data type: float.<br>Default value: 0.75. Value range: [0, 1]. |
| un_prune_list | Input | Specifies the layers that are not pruned. By default, the first and last layers are not pruned. | Optional.<br>Data type: list, elements must be int or string.<br>Default value: None.<br>If an element is int, it indicates the index of the layer not to be pruned (only Conv2d and Linear operators are counted).<br>If it is string, it indicates the name of the operator in the network. |

## Sample

```python
import torch
import torchvision
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
model = torchvision.models.vgg16(pretrained=False)
model.eval()
prune_torch = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
left_params, desc = prune_torch.analysis()
print(desc)
print(left_params)
```
