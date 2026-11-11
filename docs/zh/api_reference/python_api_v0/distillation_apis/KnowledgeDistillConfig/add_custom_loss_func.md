# add_custom_loss_func

## 功能说明

KnowledgeDistillConfig类方法，用户调用该方法增加自定义 loss function，而不只是使用api提供的loss function，非必须调用的方法。

本方法只能对loss function是否为MindSpore模型或PyTorch模型进行校验，不保证用户自定义loss function的可用性、正确性。

## 函数原型

```python
add_custom_loss_func(name, instance)
```

## 参数说明

| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| name | 输入 | 自定义loss function名称。| 必选。<br>数据类型：string。 |
| instance | 输入 | 自定义loss function的实例。| 可选。<br>数据类型：MindSpore模型或PyTorch模型。 |

## 调用示例

```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
from mindspore.nn import Cell

#用户自定义loss function的实例
class CustomLoss(Cell):
    def __init__(self):
        # init
    def construct(self, logits_s, logits_t):
        # calculate loss by logits_s and logits_t
        return loss
custom_loss = CustomLoss()
#定义配置
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
