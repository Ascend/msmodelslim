# Calibrator

## Function

Quantization parameter configuration class that encapsulates quantization algorithms through the Calibrator class.

## Prototype

```python
Calibrator(model, cfg: QuantConfig, calib_data=None, disable_level='L0', all_tensors=None, mix_cfg: Optional[dict] = None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The model. | Required.<br>Data type: PyTorch model. |
| cfg | Input | The configured QuantConfig class. | Required.<br>Data type: QuantConfig. |
| calib_data | Input | Calibration data for LLM quantization. Input real data for Label-Free quantization. | Optional.<br>Data type: object.<br>Default value is None, for Data-Free scenarios. Must be provided for Label-Free scenarios.<br>Input template: \[[input1],[input2],[input3]]. |
| disable_level | Input | The automatic fallback level. The level can be appropriately increased when the model accuracy loss is significant, but the number of fallback layers cannot exceed the total number of model layers. | Optional.<br>Data type: str.<br>Configuration examples are as follows: (1) 'L0': Default value, no fallback is performed. (2) 'L1': Fallback 1 layer. (3) 'L2': Fallback 2 layers. (4) 'L3': Fallback 3 layers. (5) 'L4': Fallback 4 layers. (6) 'L5': Fallback 5 layers.<br>And so on. |
| all_tensors | Input | {name:tensor} for layer-wise quantization calibration. | Optional.<br>Data type: dict.<br>Default value is None, use the default configuration. |
| mix_cfg | Input | Hybrid quantization configuration, specifying {wildcard or layer name of **nn.Module**: quantization type}.<br>Key: **Wildcards are case-sensitive. The implementation uses Python's standard library fnmatch, so please refer to fnmatch for related usage.** | Optional.<br>Data type: dict.<br>Default value is None.<br>Configuration example: {"\*down\*": "w8a16", "\*": "w8a8"}<br>**The quantization type (the value of the dictionary) must be an internally recognized configuration, otherwise an error will be reported.**<br>Currently supports w8a8, w8a16, w8a8_dynamic

### Hybrid Quantization Priority Description

When mix_cfg and the existing fallback mechanism are used simultaneously, the **matching priority** of different configurations is as follows, **matching from top to bottom in the following order. The first match takes effect, and no subsequent matching is performed**:

1. Fallback layers (rollback_names)

    Determined jointly by disable_names/disable_level (automatic fallback logic). disable_names is a parameter of QuantConfig, refer to [QuantConfig Parameter Description](./pytorch_QuantConfig.md#parameters).
    Once matched, it falls back to floating point, and mix_cfg is no longer considered.

2. Explicit layer name specification in mix_cfg

    If mix_cfg explicitly specifies the quantization type for a certain layer (with a one-to-one layer name match), it takes precedence over wildcard matching.

3. Wildcard rules in mix_cfg

    If the layer name successfully matches a wildcard pattern, the quantization type corresponding to that wildcard is used.

4. Default configuration (QuantConfig)

    If none of the above matches, the QuantConfig object passed via the cfg parameter is used as the default quantization type.

#### Sample of Priority

Assume mix_cfg is set to:

```python
mix_cfg = {
    "model.layers.0.mlp.down_proj": "w8a16",  # Layer name, matched successfully.
    "model.layers.1.mlp.down_pro?": "w8a16",  # fnmatch wildcard, matched successfully.
    "?q_proj": "w8a8_dynamic",  # fnmatch wildcard, matched failed.
    "*q_proj": "float",  # fnmatch wildcard, matched successfully.
    "model.layers.[012].mlp.down_proj": "w8a8_dynamic",  # fnmatch wildcard, matched successfully for 012, but since 0 and 1 are already matched, only 2 remains here.
    "model.layers.[!456789].mlp.down_proj": "w8a16",  # fnmatch wildcard, matched successfully for 0123. Similarly, since 0, 1, and 2 are already matched, only 3 remains here.
    "model.layers.4.mlp.down_proj": "w8a8_dynamic",  # fnmatch is case-sensitive, so no match here. Treated as w8a8 according to default QuantConfig.
    "model.layers.5.mlp.down_proj": "w8a16"  # Will be matched first by disable_names below, thus not actually taking effect.
}
```

And in addition, the following is also set:

```python
quant_config=QuantConfig(w_bit=8, a_bit=8)

disable_level="L1"
disable_names=["model.layers.5.mlp.down_proj"]
```

Then the result would be:

- model.layers.0.mlp.down_proj is expected to be w8a16

- model.layers.1.mlp.down_proj is expected to be w8a16

- model.layers.2.mlp.down_proj is expected to be w8a8_dynamic

- model.layers.3.mlp.down_proj is expected to be w8a16

- model.layers.4.mlp.down_proj is expected to be w8a8

- model.layers.5.mlp.down_proj is float

- All q_proj layers are float

- Some layers will fall back to float with higher priority according to the automatic fallback algorithm, which may include some of the layers expected above.

## Sample

Complete the configuration of all parameters in the QuantConfig initialization according to actual requirements.

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(dev_type='cpu', pr=0.5, mm_tensor=False)
model = AutoModel.from_pretrained('/chatglm2-6b', 
                                  local_files_only=True,
                                  torch_dtype=torch.float32).cpu()   # Configure according to the actual path of the model.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
```

### (Optional) INT8 Hybrid Quantization Sample

Configure all parameters in the QuantConfig initialization according to actual requirements.

```python
from transformers import AutoModel
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(dev_type='cpu', pr=0.5, mm_tensor=False)
model = AutoModel.from_pretrained('/chatglm2-6b', 
                                  local_files_only=True, 
                                  torch_dtype=torch.float32).cpu()   # Configure according to the actual path of the model.
mix_cfg = {
    "*down*": "w8a16",
    "*": "w8a8"
}
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0', mix_cfg=mix_cfg)
```
