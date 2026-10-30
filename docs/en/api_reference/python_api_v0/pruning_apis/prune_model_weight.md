# prune_model_weight

## Function

Model pruning API. Based on the original weights, a pruned Transformer model instance loaded with smaller parameters, and the pruning configuration passed to the API, it prunes the original weights and loads the pruned weights into the model instance with smaller parameters.

## Prototype

```python
prune_model_weight(model, config, weight_file_path)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ----- |-------------------------------------|
| model | Input | The model instance after pruning. | Required.<br> Data type: MindSpore or PyTorch model. |
| config | Input | The pruning configuration. | Required.<br> Data type: PruneConfig object. |
| weight_file_path | Input | The path and file name of the original model weight file before pruning. | Required.<br> Data type: string <br> The weight file for a MindSpore model must be in ckpt format, and the weight file for a PyTorch framework must be in pt/pth/pkl/bin format. |

## Sample

```python
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
from msmodelslim.common.prune.transformer_prune.prune_model import prune_model_weight
# Define the configuration class.
prune_config = PruneConfig()
prune_config.set_steps(['prune_blocks', 'prune_bert_intra_block']). \
  add_blocks_params('uniter\.encoder\.encoder\.blocks\.(\d+)\.', {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11})
# Pass in parameters to prune the model.
prune_model_weight(model, prune_config, weight_file_path = "xxx.ckpt")
```
