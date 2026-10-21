# RARopeCompressor

## Function

Compression parameter configuration class. The weight file required for long sequence compression can be obtained through RARopeCompressor.

## Prototype

```python
RARopeCompressor(model, tokenizer, cfg)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The currently supported model. | Required.<br>Model type: PyTorch model. |
| tokenizer | Input | The tokenizer used to load the pre-trained model. | Required.<br>Type: AutoTokenizer. |
| cfg | Input | The configuration for RARopeCompressConfig. | Required.<br>Configuration class: RARopeCompressConfig. |

## Sample

```python
from msmodelslim.pytorch.ra_compression import RARopeCompressConfig, RARopeCompressor
config = RARopeCompressConfig(induction_head_ratio=0.14, echo_head_ratio=0.01)
ra = RARopeCompressor(model, tokenizer, config) 
```
