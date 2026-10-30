# FAQuantizer

## Function

Runs the quantization algorithm to quantize Q (Query), K (Key), and V (Value).

## Prototype

```python
FAQuantizer(config, logger)
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ---------- | ---- | -------- |
| config | Input | Configuration class instance containing model parameters. | Required.<br>For large language models with a built-in config.json, the config passed in.<br>Data type: PretrainedConfig<br>For multimodal models without a built-in config.json, the config passed in.<br>Data type: Object. Must contain the following three parameters: 'num_attention_heads', 'hidden_size', and 'num_key_value_heads'. |
| logger | Input | Logger instance for recording log information. | Required.<br>Data type: Logger. |

## Sample

Sample for LLMs:

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
from msmodelslim import logger 

self.fa_quantizer = FAQuantizer(self.config, logger)

query_states = self.fa_quantizer.quant(query_states, qkv="q")
key_states = self.fa_quantizer.quant(key_states, qkv="k")
value_states = self.fa_quantizer.quant(value_states, qkv="v")
```

Sample for multimodal models:

```python
# Instantiate the FAQuantizer class.
# --------------------fa3-----------------------------
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
from msmodelslim import logger 
from types import SimpleNamespace

config_dict = {
    'num_attention_heads': self.heads_num, 
    'hidden_size': self.hidden_size,
    'num_key_value_heads': self.heads_num,
    }

config = SimpleNamespace(**config_dict)
self.fa_quantizer = FAQuantizer(config, logger=logger)
# --------------------fa3-----------------------------
...   
# --------------------fa3-----------------------------
# Call the quant function of FAQuantizer to quantize the Q, K, and V matrices.
query = self.fa_quantizer.quant(query, qkv="q")
key = self.fa_quantizer.quant(key, qkv="k")
value = self.fa_quantizer.quant(value, qkv="v")
# --
```
