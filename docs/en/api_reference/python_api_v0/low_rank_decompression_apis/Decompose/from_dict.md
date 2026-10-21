# from_dict

## Function

A method of the Decompose class. Configures the decomposition rate for each layer in model low-rank decomposition by specifying the number of channels after decomposition for each layer in a dictionary, and returns itself for method chaining.

## Prototype

```python
from_dict(channel_dict, excludes=None, divisor=64)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| channel_dict | Input | Specifies the decomposition method for each layer by name using a dictionary. | Required.<br>Data type: dict. The key is the layer name or a regular expression matching the layer name. The value can be an int, float, or "vbmf", corresponding to the logic of from_fixed, from_ratio, and from_vbmf respectively.<br>Note: When using complex regular expressions, users must ensure the safety of the expressions to avoid ReDoS attacks, which may cause slow program execution. |
| excludes | Input | Specifies the names of layers not to be decomposed. | Optional.<br>Data type: None, list, or tuple.<br>Default value: None. |
| divisor | Input | Specifies the multiple for the channel after decomposition. For example, if set to 16, the number of channels after decomposition will be a multiple of 16. | Optional.<br>Data type: int, must be greater than 0. Default value: 64. <br>Note: Setting divisor to 1 disables this feature. |

## Sample

```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  # Call __init__ to initialize the class.
decomposer = decomposer.from_dict({'feature.0': (64, 64), 'inner': 192, 'classifier.0': 128}, divisor=16)  # Compute decomposition information in the dict-style manner.
```
