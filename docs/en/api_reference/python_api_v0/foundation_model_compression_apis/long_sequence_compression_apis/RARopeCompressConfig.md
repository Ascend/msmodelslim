# RARopeCompressConfig

## Function

When performing long sequence compression on a RoPE-encoded model, configure the retention ratios for Induction Head and Echo Head. The retained portions do not need compression, and only the unretained portions need to be compressed.

## Prototype

```python
RARopeCompressConfig(induction_head_ratio=0.14, echo_head_ratio=0.01)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| induction_head_ratio | Input | Controls the retention ratio of the Induction Head.<br>Description: Induction Head: When processing text sequences, it is used to attend to and predict the next token in the input sequence that is identical to the current token. | Optional.<br>Data type: float.<br>Default: 0.14, optional range: [0,1]. |
| echo_head_ratio | Input | Controls the retention ratio of the Echo Head.<br>Description: Echo Head. When processing text sequences, it is used to attend to tokens that appear earlier in the context and are identical to the current token. | Optional.<br>Data type: float.<br>Default: 0.01, optional range: [0,1]. |

## Sample

```python
from msmodelslim.pytorch.ra_compression import RARopeCompressConfig
config = RARopeCompressConfig(induction_head_ratio=0.14, echo_head_ratio=0.01)
```
