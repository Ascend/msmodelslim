# decompose_network

## Function

Performs low-rank decomposition according to the decomposition rate configured for each layer by the user, and returns the decomposed model. This function recursively traverses and processes the model's submodules `named_children`, with the recursion depth depending on the depth of the network (that is, the number of module nesting levels). If the model structure is exceptionally complex or has extreme nesting (such as a custom ultra-deep network), it may trigger Python's recursion depth limit.

## Prototype

```python
decompose_network(do_decompose_weight=True, datasets=None, max_iter=-1)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| do_decompose_weight | Input | Specifies whether to perform weight decomposition and set it as the decomposed model. | Optional.<br>Data type: bool.<br>Default value is True. The value can be True or False. If set to False, the model is only converted to the low-rank decomposed model structure, and the weights of each layer are randomly initialized values. |
| datasets | Input | Input dataset. If the value is not None, data-aware decomposition using input data is required. | Optional.<br>Default value is None. If not None, a dataset that the model can directly iterate over must be input, with elements being dict, list, or tuple. |
| max_iter | Input | When datasets is not None, specifies the maximum number of iterations for fetching data from datasets during data-aware decomposition. | Optional.<br>Data type: int. Default value is -1, where -1 indicates using the entire dataset, and a value > 0 indicates the maximum number of iterations within the dataset. |

## Sample

```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model).from_ratio(0.5, divisor=16)  
model = decomposer.decompose_network(do_decompose_weight=True)
```
