# CompressConfig

## Function

A parameter configuration class for weight compression, which stores the parameters configured during the weight compression process.

## Prototype

```python
CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=False, compress_disable_layers=None, record_detail_root='./', multiprocess_num=1)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| do_pseudo_sparse | Input | Whether to perform pseudo weight sparsification. | Optional.<br>Data type: bool.<br>Default: False. |
| sparse_ratio | Input | The sparse ratio for pseudo weight sparsification. | Optional.<br>Data type: float.<br>Default: 1, with a valid range of [0, 1]. |
| is_debug | Input | Whether to output detailed log information. | Optional.<br>Data type: bool.<br>Default: False. |
| compress_disable_layers | Input | Specifies the layers whose weights do not need to be compressed. | Optional.<br>Data type: list or tuple.<br>Default: None. |
| record_detail_root | Input | The storage path for intermediate results of the compression process. | Optional.<br>Data type: str.<br>Default: "./". |
| multiprocess_num | Input | The number of processes to start in multi-process compression mode. | Optional.<br>Data type: int.<br>Default: 1, using single-process compression mode.<br>Note: Users need to set the multiprocess_num parameter according to the device environment to accelerate weight compression. Setting an excessively large multiprocess_num value may lead to insufficient system resource requests, causing the compression program to fail. It is recommended that users first set the maximum number of processes that the current system can open, for example, ulimit -n 65536; then check the NPU usage of the current device to ensure there is no significant memory footprint. The recommended value for multiprocess_num is 8. |

## Sample

```python
from msmodelslim.modeslim.pytorch.weight_compression import CompressConfig
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, compress_disable_layers=None, record_detail_root='./record_root', multiprocess_num=8)
```
