# init

## Function

A method of the Decompose class that initializes the class with the user-provided model.

## Prototype

```python
__init__(model, config_file=None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The model to be low-rank decomposed. | Required.<br>Data type: PyTorch or MindSpore model. |
| config_file | Input | The file path and file name for saving the intermediate layer channel information after decomposition. | Optional.<br>Data type: str or None. The default value is None, indicating that the decomposed information is not saved. A string ending with .json can be specified. After calling the from_xxx() API, the channel information is saved to this file. |

## Sample

```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  # Call __init__ to initialize the class.
```
