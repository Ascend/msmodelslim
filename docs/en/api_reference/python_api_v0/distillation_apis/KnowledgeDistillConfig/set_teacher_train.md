# set_teacher_train

## Function

A method of the KnowledgeDistillConfig class. By calling this method, the teacher model is set to training mode. Typically, during distillation, only the student model's gradients are computed and updated, while the teacher model is used solely for inference, making this method unnecessary. However, for special requirements, you can call this method directly to put the teacher model into a gradient update state. It takes no input parameters.

## Prototype

```python
set_teacher_train()
```

## Sample

```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label(0.5, 0)
distill_config.set_teacher_train()
```
