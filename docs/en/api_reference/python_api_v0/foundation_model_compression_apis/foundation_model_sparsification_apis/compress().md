# compress()

## Function

Runs the weight sparsification algorithm. After initializing the Compressor, execute weight sparsification through the compress() function.

## Prototype

```python
prune_compressor.compress(dataset)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| dataset | Input | Sparse calibration dataset. | Required.<br>Data type: list. |

## Sample

```python
import torch
from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig, Compressor
sparse_config = SparseConfig(method='magnitude', sparse_ratio=0.5)
# model is a PyTorch nn.Module model. A simple neural network model is used as an example.
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=True)
        self.linear2 = torch.nn.Linear(H, D_out, bias=True)
    def forward(self, x):
        x = self.linear1(x)
        y_pred = self.linear2(x)
        return y_pred
D_in, H, D_out = 100, 10, 1
model = TwoLayerNet(D_in, H, D_out)
prune_compressor = Compressor(model, sparse_config)
test_dataset = [torch.randn(64, D_in)]
prune_compressor.compress(dataset=test_dataset)
```
