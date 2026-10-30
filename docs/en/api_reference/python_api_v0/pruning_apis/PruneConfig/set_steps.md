# set_steps

## Function

A method of the PruneConfig class that configures the steps for model pruning based on custom parameters.

## Prototype

```python
set_steps(steps)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ |-------------------------------------------|
| steps | Input | The steps for weight pruning. | Required.<br> Data type: list.<br> Values:<br> 1. "prune_bert_intra_block": Prunes pretrained weights based on the input model. It prunes weights in the pretrained model that have the same name but different shapes as those in the model, so that the shape of the pretrained weights matches the model. This can be specified independently.<br>Data type: string.<br> 2. "prune_blocks": Prunes pretrained weights based on the parameters of add_blocks_params(). It retains the weights of a layer with a specified ID to another layer. This can be specified independently. When specifying this step, add_blocks_params must also be configured for it to take effect.<br> |

## Sample

```python
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
prune_config = PruneConfig()
prune_config.set_steps([ 'prune_bert_intra_block'])
```
