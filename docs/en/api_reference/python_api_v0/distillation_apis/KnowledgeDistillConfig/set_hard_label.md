# set_hard_label

## Function

A method of the KnowledgeDistillConfig class used to configure the loss weight and loss index of the student model during distillation. This method must be called.

## Prototype

```python
set_hard_label(weight, index)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| weight | Input | Weight of the hard loss. | Required.<br>Data type: float.<br>Recommended to be between 0 and 1. |
| index | Input | The output index of the student model. When the student model has multiple outputs, this index determines which output is used to calculate the loss.<br>Only needs to be configured for MindSpore models, typically 0. | Required.<br>Data type: int. |

## Sample

```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label(0.5, 0)
```
