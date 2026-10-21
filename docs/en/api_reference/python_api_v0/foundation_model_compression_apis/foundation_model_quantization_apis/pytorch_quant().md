# quant()

## Function

The quant function is used to quantize the Q, K, and V tensors in a model to optimize model performance and efficiency.

## Prototype

```python
quant(states_tensor: torch.Tensor, qkv: str)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ---------- | ---- | -------- |
| states_tensor | Input | The tensor to be quantized, which can be a Q, K, or V tensor. | Required.<br>Data type: torch.Tensor. |
| qkv | Input | Specifies the tensor type represented by the current states_tensor, which can be "q", "k", or "v". | Required.<br>Data type: str. |

## Sample

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
from msmodelslim import logger 

self.fa_quantizer = FAQuantizer(self.config, logger)

# Note: The FA quantization function requires that Q, K, and V are all passed in once; otherwise, an incomplete input error will be raised.
query_states = self.fa_quantizer.quant(query_states, qkv="q")
key_states = self.fa_quantizer.quant(key_states, qkv="k")
value_states = self.fa_quantizer.quant(value_states, qkv="v")
```
