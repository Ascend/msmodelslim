# export()

## Function

Weight compression parameter configuration class, which saves the compressed weights and related parameters by encapsulating the compression algorithm through the Compressor class.

Note: If the 'weight_path' parameter is used in the Compressor, the compressed weights must be exported via the 'compressor.export' function.

## Prototype

```python
compressor.export(arr, path, dtype=numpy.int8)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
|arr|Input|The return result of the compressor.run function.|Required.<br>Data type: dict.|
|path|Input|The save path for the compression result.|Required.<br>Data type: str.|
|dtype|Input|The save format of the compression result.|Optional.<br>Data type: numpy.dtype.<br>Default value: numpy.int8.|

## Sample

- Use the weight_path parameter to specify the path of the weight file to be compressed.

```python
import numpy
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, compress_disable_layers=None, record_detail_root='./record_root')
weight_save_path = './quant_weight.npy'  # Modify the path of the weight file to be compressed according to your actual situation.
compressor = Compressor(compress_config, weight_save_path)
compress_result_weight, compress_result_index, compress_result_info = compressor.run()
compressor.export(compress_result_weight, './compress_weight')
compressor.export(compress_result_index, './compress_index')
compressor.export(compress_result_info, './compress_info', dtype=numpy.int64)
```
