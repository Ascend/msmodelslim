# from_ratio

## Function

A method of the Decompose class. Configures the decomposition ratio for each layer in low-rank decomposition of the model, calculates the number of channels after decomposition for each layer proportionally, and returns itself for chaining calls.

## Prototype

```python
from_ratio(channel_ratio, excludes=None, divisor=64)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| channel_ratio | Input | The decomposition ratio used to calculate the number of channels after decomposition for the layers to be decomposed. | Required.<br>Data type: float. Range: 0-1. |
| excludes | Input | Specifies the names of layers not to be decomposed. | Optional.<br>Data type: None, list, or tuple.<br>Default value: None. |
| divisor | Input | Specifies the divisor for the number of channels after decomposition. For example, if set to 16, the number of channels after decomposition will be a multiple of 16. | Optional.<br>Data type: int, must be greater than 0. Default value: 64.<br>Note: Setting divisor to 1 disables this feature. |

## Sample

```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  # Call __init__ to initialize the class.
decomposer = decomposer.from_ratio(0.5, divisor=16)  # Compute decomposition information in the ratio manner.
```
