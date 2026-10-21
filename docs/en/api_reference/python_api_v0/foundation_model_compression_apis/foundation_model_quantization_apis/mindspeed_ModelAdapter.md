# ModelAdapter

## Function

The Model Adapter of MindSpeed, which converts the MindSpeed-LLM model into an LLM model that can be quantized by msModelSlim.

## Prototype

```python
ModelAdapter(model:nn.Module, dev_type='npu', forward_step=None, prefix='model.')
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The model. | Required.<br>Data type: MindSpeed-LLM model. |
| dev_type | Input | The device type of the device. | Optional.<br>Data type: string.<br>Optional values: ['npu', 'gpu'], Default Value: 'npu' |
| forward_step | Input | The function object that starts the MindSpeed-LLM model. A custom launcher can be defined when the model cannot be started normally. | Optional.<br>Data type: function object.<br>Default Value: None, which uses the built-in model launching method |
| prefix | Input | The prefix name of the model's state_dict. | Optional.<br>Data type: string.<br>Optional values: ["model.", "model.module."]<br>Default Value: "model". When the keys of the model's state_dict are inconsistent with the model's own structure, the prefix needs to be adjusted to ensure consistency; otherwise, the correct execution of model saving will be affected. For example, if the weight obtained by model.state_dict() is "language_model.output_layer.weight", but the actual model path of this weight is "model.language_model.output_layer.weight", this parameter should be set to 'model.' to correct the prefix name. |

## Sample

```python
from msmodelslim.pytorch.mindspeed_adapter import ModelAdapter
from megatron.inference.text_generation import generate
class GenerateForward:
    def __call__(self, model, x):
        return generate(model, x, tokens_to_generate=1)
model = ModelAdapter(model, forward_step=GenerateForward(), prefix='model.')
```
