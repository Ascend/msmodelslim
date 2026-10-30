# get_compress_heads

## Function

After executing RARopeCompressor, you can call the get_compress_heads() function to generate a .pt file at the specified path.

## Prototype

```python
RARopeCompressor.get_compress_heads(save_path)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| save_path | Input | Path to save the head compression parameter file during long sequence compression. | Required.<br>Data type: String. |

## Sample

```python
import torch
from msmodelslim.pytorch.ra_compression import RARopeCompressConfig, RARopeCompressor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_npu
torch.npu.set_compile_mode(jit_compile=False)
 
config = RARopeCompressConfig(induction_head_ratio=0.14, echo_head_ratio=0.01)
 
save_path = "./win.pt" 
model_path = "/home/wgw/Meta-Llama-3.1-70B-Instruct/"
 
model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    ).eval()
 
tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left'
    ) 
 
ra = RARopeCompressor(model, tokenizer, config) 
ra.get_compress_heads(save_path)
```
