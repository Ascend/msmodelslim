# get_distill_model

## Function

Model distillation API, which combines the user-provided teacher model and student model according to the distillation configuration, returns a DistillDualModels instance, and the user trains the DistillDualModels instance.

Due to differences in distillation implementation between PyTorch and MindSpore, there are also the following differences in the usage of the DistillDualModels instance.

- Under PyTorch, the DistillDualModels instance returns three pieces of data after forward propagation: the loss calculated from the soft label, the raw output of the student model, and the raw output of the teacher model. If the loss of the hard label is required, the user needs to calculate it manually based on the raw output of the student model and call the get_total_loss() method of the DistillDualModels instance to obtain the combined loss of the soft label and hard label.
- Under MindSpore, all losses are calculated automatically, eliminating the need for manual hard label calculation.

## Prototype

```python
get_distill_model(teacher, student, config)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| teacher | Input | The teacher model. | Required.<br>Data type: MindSpore model or PyTorch model. |
| student | Input | The student model. | Required.<br>Data type: MindSpore model or PyTorch model. |
| config | Input | The configuration for distillation. | Required.<br>Data type: KnowledgeDistillConfig object. |

## Sample

```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig, get_distill_model
# Define the configuration.
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label(0.5, 0) \
  .add_inter_soft_label({
    't_module': 'uniter.encoder.encoder.blocks.11.output',
    's_module': 'uniter.encoder.encoder.blocks.5.output',
    't_output_idx': 0,
    's_output_idx': 0,
    "loss_func": [{"func_name": "KDCrossEntropy",
             "func_weight": 1}],
    'shape': [2048]
  }) 
# Pass in parameters and return the distilled model.
distill_model = get_distill_model(teacher_model, student_model, distill_config)
```
