# from_vbmf

## Function

A Decompose class method that configures the decomposition rate for low-rank decomposition of each layer in the model. It specifies the Variational Bayesian Matrix Factorization (VBMF) rank search method to automatically calculate the number of channels after decomposition, and returns itself for chained calls. This is typically suitable for models with pre-trained weights.

## Prototype

```python
from_vbmf(excludes=None, divisor=64)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| excludes | Input | Specifies the names of layers that are not to be decomposed. | Optional.<br>Data type: None, list, or tuple.<br>Default value: None. |
| divisor | Input | Specifies the multiplier for the channel after decomposition. For example, if 16 is specified, the number of channels after decomposition will be a multiple of 16. | Optional.<br>Data type: integer, greater than 0. Default value: 64.<br>Note: When divisor is set to 1, this feature is disabled. |

## Example

```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  # Call __init__ to initialize the class.
decomposer = decomposer.from_vbmf(divisor=16)  # Compute decomposition information in the VBMF manner.
```
