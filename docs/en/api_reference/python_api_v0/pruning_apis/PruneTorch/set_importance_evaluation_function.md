# set_importance_evaluation_function

## Function

PruneTorch class method, which configures a user-defined importance evaluation function during the pruning process. When no custom function is defined by the user, L1 regularization is used as the importance by default.

## Prototype

```python
set_importance_evaluation_function(importance_evaluation_function)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ |-------------------------|
| importance_evaluation_function | Input | Custom importance evaluation function, which must be a callable function. | Required.<br>Data type: function. |

## Sample

```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
model = torchvision.models.vgg16(pretrained=False)
model.eval()
prune_torch= PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
importance_evaluation_function = lambda chn_weight: torch.abs(chn_weight).mean().item()
prune_torch= prune_torch.set_importance_evaluation_function(importance_evaluation_function)
```
