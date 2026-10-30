# QuantConfig

## Function

Quantization parameter configuration class, which stores parameters configured during the quantization process.

## Prototype

```python
QuantConfig(disable_names=None, fraction=0.01)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| disable_names | Input | Names of nodes to be excluded from quantization, that is, the names of quantization layers to manually fallback.<br>If accuracy is poor, it is recommended to fallback quantization-sensitive layers, such as classification layers, input layers, detection head layers, etc. | Optional.<br>Data type: object. |
| fraction | Input | Sparse quantization precision control. | Optional.<br>Data type: float.<br>Value range [0.01, 0.1]. |

## Sample

```python
from msmodelslim.mindspore.llm_ptq import Calibrator, QuantConfig
quant_config = QuantConfig(disable_names=["lm_head"], fraction=0.04)
```
