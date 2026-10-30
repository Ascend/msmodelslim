# count_parameters

## Function

API for counting model parameters. It counts the number of parameters based on the model provided by the user.

## Prototype

```python
count_parameters(network)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| network | Input | The model to be low-rank decomposed. | Required.<br>Data type: PyTorch or MindSpore model. |

## Sample

```python
from ascend_utils.common.utils import count_parameters
print("Original model parameters:", count_parameters(network))
```
