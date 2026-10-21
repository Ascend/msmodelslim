# RACompressor

## Function

Compression parameter configuration class. The weight file (.pt file) required for long sequence compression can be obtained through RACompressor.

## Prototype

```python
RACompressor(model, cfg)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The currently supported model.<br>Note: Loading using the npu method is not supported. | Required.<br>Model type: PyTorch model. |
| cfg | Input | Configuration of RACompressConfig. | Required.<br>Data type: int.<br>Configuration class: RACompressConfig. |

## Sample

```python
from msmodelslim.pytorch.ra_compression import RACompressConfig, RACompressor
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="baichuan2-13b/float_path/", 
                                             local_files_only=True).float().cpu()  # Configure according to the actual path of the model.
config = RACompressConfig(theta=0.00001, alpha=100)
ra = RACompressor(model,config) 
```
