# add_inter_soft_label

## Function

A method of the KnowledgeDistillConfig class, used to configure the soft label for distillation, i.e., the mapping relationship between the soft labels of the student model and the teacher model. This method is not mandatory.

## Prototype

```python
add_inter_soft_label(config)
```

## Parameters

Parameter name: config; Data type: dict; The included configuration items are as follows:

| Configuration Item | Description | Constraints |
| --- | --- | --- |
| t_module | Configures the name of the layer in the teacher model used to compute the soft label. | Required.<br>Data type: string. |
| s_module | Configures the name of the layer in the student model used to compute the soft label. | Required.<br>Data type: string. |
| t_output_idx | Configures the output index of t_module.<br>If t_module has multiple outputs, use this parameter to specify the output used for loss calculation. If there is only one output, use 0. | Required.<br>Data type: int. |
| s_output_idx | Configures the output index of s_module.<br>If s_module has multiple outputs, use this parameter to specify the output used for loss calculation. If there is only one output, use 0. | Required.<br>Data type: int. |
| loss_func | Specifies the loss function for t_module and s_module. Each loss function is stored as a dictionary in this list. The dictionary contains the following fields:<br>(1) func_name: The name of the loss function. For MindSpore and PyTorch models, configure as KDCrossEntropy. Custom: Use the add_custom_loss_func method to add a new loss function.<br>(2) func_weight: The weight of the loss.<br>(3) temperature: The temperature for distillation.<br>(4) func_param: Parameters for some loss functions. | Required.<br>Data type: list. Parameters within the dictionary:<br>(1) func_name is required, Data type: string.<br>(2) func_weight is required, Data type: int.<br>(3) temperature is optional, Data type: float, default value is 1.<br>(4) func_param is optional, Data type: list, default value is []. |
| shape | The shape of t_module[t_output_idx].<br>Only required for MindSpore models. | Optional.<br>Data type: list.<br>Note that the shapes of t_module[t_output_idx] and s_module[s_output_idx] must be consistent. |

## Sample

```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label(0.5, 0) \
  .add_inter_soft_label({
                    "t_module": "uniter.encoder.encoder.blocks.11.output",
                    "s_module": "uniter.encoder.encoder.blocks.5.output",
                    "t_output_idx": 0,
                    "s_output_idx": 0,
                    "loss_func": [{"func_name": "KDCrossEntropy",
                                   "func_weight": 1,
                                   "temperature": 1,},
                    ],
                    "shape": [2048]
  })
```
