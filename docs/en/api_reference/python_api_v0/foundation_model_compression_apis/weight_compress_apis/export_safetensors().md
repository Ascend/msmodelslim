# export_safetensors()

## Function

Saves the compressed weights as a safetensors format file and generates the corresponding description file.

Note: This function is required when selecting the weight and quant_model_description parameters in Compressor.

## Prototype

```python
Compressor.export_safetensors(path, safetensors_name=None, json_name=None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| path | Input | The save path for the compression results. | Required.<br>Data type: String. |
| safetensors_name | Input | The name of the safetensors format compressed weight file. | Optional.<br>Data type: String.<br>This parameter defaults to None, and the output file name is quant_model_weight_w8a8sc.safetensors. |
| json_name | Input | The name of the safetensors format compressed weight JSON description file. | Optional.<br>Data type: String.<br>This parameter defaults to None, and the output file name is quant_model_description_w8a8sc.json. |

## Sample

- Call the `run()` method of `Compressor` to perform weight compression.

```python
from safetensors.torch import load_file
import json
# Import weight compression interface.
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
# Prepare the weight file to be compressed and related compression config. Modify according to your actual situation.
weight_path = "./quant_model_weight_w8a8s.safetensors"       # Path of the weight file to be compressed
save_path = "./w8a8sc_llama2-7b"                          # Save path of the compressed weight file.
json_path = "./quant_model_description_w8a8s.json"          # Path of the description file of the weight file to be compressed
# Use CompressConfig interface to configure compression parameters and return a config instance.
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=save_path, multiprocess_num=8)
sparse_weight = load_file(weight_path)
with open(json_path, 'r') as f:
    quant_model_description = json.load(f)
# Use Compressor interface, passing in the loaded compression config and the weight file to be compressed
compressor = Compressor(compress_config, weight=sparse_weight, quant_model_description=quant_model_description)
compress_weight, compress_index, compress_info = compressor.run()
# Use the export_safetensors() interface to save the compressed result file.
compressor.export_safetensors(path=save_path, safetensors_name=None, json_name=None)
```
