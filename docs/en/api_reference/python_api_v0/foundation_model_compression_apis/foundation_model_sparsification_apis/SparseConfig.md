# SparseConfig

## Function

Parameter configuration class for weight sparsification, which saves the parameters configured during the weight sparsification process.

## Prototype

```python
SparseConfig(mode='sparse', method='magnitude', sparse_ratio=0.5, progressive=False, uniform=True)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| mode | Input | Configures the tool to compression mode. | Optional.<br>Data type: String.<br>Default is 'sparse', available values are ['sparse']. |
| method | Input | Configures the specific sparse algorithm type. | Optional.<br>Data type: String.<br>Default is 'magnitude', available values are ['magnitude', 'hessian', 'par', 'par_v2']. |
| sparse_ratio | Input | Configures the weight sparsity ratio. | Optional.<br>Data type: float.<br>Default is 0.5, available range is (0, 1). |
| progressive | Input | Configures the progressive sparse mode. | Optional.<br>Data type: bool.<br>Default is False.|
| uniform | Input | Configures the global adaptive sparse mode. | Optional.<br>Data type: bool.<br>Default is True.<br>True: Enable uniform sparsity. False: Non-uniform sparsity.|

## Sample

```python
from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig
sparse_config = SparseConfig(method='magnitude', sparse_ratio=0.5)
```
