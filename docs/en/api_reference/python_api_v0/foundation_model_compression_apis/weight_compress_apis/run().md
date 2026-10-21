# run()

## Function

Runs the weight compression algorithm. After initializing the Compressor, execute weight compression through the run() function.

## Prototype

```python
compress_result_weight, compress_result_index, compress_result_info = compressor.run(weight_transpose=False)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| weight_transpose | Input | Whether the weights to be compressed need transposition. | Optional.<br>Data type: bool.<br>Default is False, meaning no transposition is needed. Can be set to True, meaning transposition is needed. |
| compress_result_weight | Return | The compressed weight result. | Data type: dict. |
| compress_result_index | Return | The compressed index result. | Data type: dict. |
| compress_result_info | Return | The compression information result. | Data type: dict. |

## Sample

- Use the weight_path parameter to perform weight compression.

```python
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, compress_disable_layers=None, record_detail_root='./record_root')
weight_save_path = './quant_weight.npy'  # Modify the path of the weight file to be compressed according to your actual situation.
compressor = Compressor(compress_config, weight_path=weight_save_path)
compress_result_weight, compress_result_index, compress_result_info = compressor.run() 
```
