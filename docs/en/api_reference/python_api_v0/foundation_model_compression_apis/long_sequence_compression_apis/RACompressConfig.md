# RACompressConfig

## Function

Configures parameters for the compression process during long sequence compression.

## Prototype

```python
RACompressConfig(theta=0.00001, alpha=100)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| theta | Input | Contribution of the attention score, used to ensure the inference accuracy of the calibrated model. | Optional.<br>Data type: float.<br>Default value: 0.00001. Valid range: [0.00001, 0.001]. |
| alpha | Input | Calibration bias, used to ensure applicability breadth and control the window size. | Optional.<br>Data type: int.<br>Default value: 100. Valid range: [0, 10000]. |

## Sample

```python
from msmodelslim.pytorch.ra_compression import RACompressConfig
config = RACompressConfig(theta=0.00001, alpha=100)
```
