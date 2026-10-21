# get_alibi_windows

## Function

After executing RACompressor, you can generate a .pt file at the specified path using the get_alibi_windows() function.

## Prototype

```python
RACompressor.get_alibi_windows(save_path)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| save_path | Input | The save path for the head compression window parameter file during long sequence compression. | Required.<br>Data type: str. |

## Sample

```python
from msmodelslim.pytorch.ra_compression import RACompressConfig, RACompressor
config = RACompressConfig(theta=0.00001, alpha=100)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="baichuan2-13b/float_path/", 
                                             local_files_only=True).float().cpu()    # Configure according to the actual path of the model.
ra = RACompressor(model, config) 
ra.get_alibi_windows('./alibi_windows.pt')
```
