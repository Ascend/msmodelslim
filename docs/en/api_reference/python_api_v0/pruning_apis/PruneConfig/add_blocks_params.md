# add_blocks_params

## Function

A method of the PruneConfig class, used to configure the blocks for model pruning based on custom parameters. If the steps selected by set_steps include "prune_blocks", this method must be called.

## Prototype

```python
add_blocks_params(pattern, layer_id_map)
```

## Parameters

| Parameter | Input/Return | Description | Constraints                                      |
| ------ | ------ | ------ |-------------------------------------------|
| pattern | Input | Regular expression for the name of the network layer to be pruned. | Required.<br>Data type: string.<br>For example, when the value is "bert\\.encoder\\.layer\\.(\d+)", it indicates selecting network layers that start with bert.encoder.layer and are followed by a number.<br>Note: When using complex regular expressions, users must ensure the security of the regular expression to avoid the risk of ReDoS attacks, otherwise it may cause slow program execution. |
| layer_id_map | Input | The matching relationship between the IDs of the network layers before and after pruning. | Required.<br>Data type: dict, with both key and value data types being int.<br>For example, when the value is {0: 0, 1: 2, 2: 4}, it means retaining the weights of bert.encoder.layer.0 to bert.encoder.layer.0, the weights of bert.encoder.layer.2 to bert.encoder.layer.1, and the weights of bert.encoder.layer.4 to bert.encoder.layer.2. That is, there are 5 bert.encoder.layer.x layers in the pre-trained weights, but only 3 bert.encoder.layer.x layers in the input model. Through layer_id_map, the weights are retained to the specified positions during pruning. |

## Sample

```python
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
prune_config = PruneConfig()
prune_config.set_steps(['prune_blocks']). \
  add_blocks_params('uniter\.encoder\.encoder\.blocks\.(\d+)\.', {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11})
```
