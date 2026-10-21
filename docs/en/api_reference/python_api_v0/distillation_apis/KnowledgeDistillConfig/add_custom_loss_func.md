# add_custom_loss_func

## Function

A KnowledgeDistillConfig class method that users call to add a custom loss function instead of using only the loss functions provided by the API. This method is not mandatory to call.

This method can only verify whether the loss function is a MindSpore model or a PyTorch model, and does not guarantee the availability or correctness of the user-defined custom loss function.

## Prototype

```python
add_custom_loss_func(name, instance)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| name | Input | The name of the custom loss function. | Required.<br>Data type: string. |
| instance | Input | The instance of the custom loss function. | Optional.<br>Data type: MindSpore model or PyTorch model. |

## Sample

```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
from mindspore.nn import Cell

# An instance of a user-defined loss function.
class CustomLoss(Cell):
    def __init__(self):
        # init
    def construct(self, logits_s, logits_t):
        # calculate loss by logits_s and logits_t
        return loss
custom_loss = CustomLoss()
# Define the configuration.
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label(0.5, 0) \
  .add_custom_loss_func("custom_loss", custom_loss) \
  .add_output_soft_label({
    't_output_idx': 0,
    's_output_idx': 0,
    "loss_func": [{"func_name": "custom_loss",
             "func_weight": 1}]
  })
```
