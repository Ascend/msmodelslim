# from_fixed

## Function

A method of the Decompose class for configuring the decomposition rate of each layer during low-rank decomposition of a model. It specifies the channel number of the intermediate layer after decomposition for each layer using a fixed value, and returns self for method chaining.

## Prototype

```python
from_fixed(channel_fixed, excludes=None, divisor=64)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| channel_fixed | Input | The fixed number of decomposition channels for each layer. | Required.<br>Data type: integer, and must be greater than 0. |
| excludes | Input | Specifies the names of layers that are not to be decomposed. | Optional.<br>Data type: None, list, or tuple.<br>Default value: None. |
| divisor | Input | Specifies the multiple for the decomposed channels. For example, if 16 is specified, the number of decomposed channels will be a multiple of 16. | Optional.<br>Data type: int, must be greater than 0. Default value: 64.<br>Note: When divisor is set to 1, this feature is disabled. |

## Sample

```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  # Call __init__ to initialize the class.
decomposer = decomposer.from_fixed(64,divisor=16)  # Compute decomposition information in the fixed manner.
```
