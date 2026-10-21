# Compressor

## Function

A weight sparsification parameter configuration class that encapsulates the sparsification algorithm through the Compressor class.

## Prototype

```python
Compressor(model, cfg)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The model to undergo weight sparsification. | Required.<br>Data type: nn.Module. |
| cfg | Input | The configured SparseConfig class. | Required.<br>Data type: SparseConfig. |

## Sample

```python
from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig, Compressor
sparse_config = SparseConfig(method='magnitude', sparse_ratio=0.5)
prune_compressor = Compressor(model, sparse_config)   # model is a PyTorch nn.Module model. 
```
