# add_output_soft_label

## Function

A method of the KnowledgeDistillConfig class, used to configure the soft label for distillation, that is, the mapping relationship between the soft labels of the student model and the teacher model. It is specifically used for the last layer of the model and is not a mandatory method to call.

## Prototype

```python
add_output_soft_label(config)
```

## Parameters

Parameter name: config; Data type: dict; The included configuration items are as follows:

| Configuration Item | Description | Usage Limitations |
| --- | --- | --- |
| t_output_idx | Used to configure the index of the t_module output.<br> If the t_module has multiple outputs, this parameter must be used to specify the output used for calculating the loss. If there is only one output, use 0. | Required.<br>Data type: int. |
| s_output_idx | Used to configure the index of the s_module output.<br>If the s_module has multiple outputs, this parameter must be used to specify the output used for calculating the loss. If there is only one output, use 0. | Required.<br>Data type: int. |
| loss_func | Used to specify the loss function for the t_module and s_module. Each loss function is stored as a dictionary in this list. The dictionary contains the following fields:<br>(1) func_name: The name of the loss function. Configure as KDCrossEntropy for MindSpore and PyTorch models. Custom: Use the add_custom_loss_func method to add a new loss function.<br>(2) func_weight: The weight of the loss.<br>(3) temperature: The temperature for distillation.<br>(4) func_param: Parameters for some loss functions. | Required.<br>Data type: list.<br>Parameters within the dictionary:<br>(1) func_name is required, Data type: string.<br>(2) func_weight is required, Data type: int.<br>(3) temperature is optional, Data type: float, Default value: 1.<br>(4) func_param is optional, Data type: list, Default value: []. |

## Sample

```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label(0.5, 0) \
  .add_output_soft_label({
    't_output_idx': 0,
    's_output_idx': 0,
    "loss_func": [{"func_name": "KDCrossEntropy",
             "func_weight": 1}]
      })
```
