# sparse_model_width

## Function

Sparse training configuration parameter interface. It determines the current sparsification stage, expands the model and weights, and calls the reset optimizer interface to convert the user-provided model into a sparse training model.

## Prototype

```python
sparse_model_width(model, optimizer, steps_per_epoch, epochs_each_stage)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The original model after initialization.<br>If the model has been wrapped with **torch.nn.parallel.DistributedDataParallel** in the original training script, the model must be in non-DDP mode. | Required.<br>Data Type: PyTorch model. |
| optimizer | Input | The optimizer after initialization. | Required.<br>Data Type: PyTorch optimizer, an instance of torch.optim.Optimizer. |
| steps_per_epoch | Input | The number of iterations per epoch, used to determine the current stage. | Required.<br>Data Type: int, must be greater than 0. |
| epochs_each_stage | Input | The number of epochs for each sparsification stage. | Required.<br>Data Type: list or tuple. Elements must be int.<br>Length must be greater than 2, and all elements except the last must be int values greater than 0. The last element can be -1.<br>Note: When the last element of epochs_each_stage is -1, it indicates that the third training stage will continue until the total number of epochs is reached. |

## Sample

```python
from msmodelslim.pytorch import sparse
model = sparse.sparse_model_width(model, optimizer, steps_per_epoch=100, epochs_each_stage=[1, 2, 1])
```
