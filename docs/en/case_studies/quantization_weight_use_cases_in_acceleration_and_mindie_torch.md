# Use Cases for W8A8 Quantized Weights in the Acceleration Library Scenario

## Overview

The relationship between the quantization tool and MindIE is straightforward. msModelSlim provides quantization capabilities, and the MindIE acceleration library can load quantized weights for inference.

## Preparation

Install MindIE by following the [MindIE Installation Guide](https://www.hiascend.com/document/detail/zh/mindie/10RC3/envdeployment/instg/mindie_instg_0001.html), and configure MindIE LLM by following the [Configure MindIE LLM](https://www.hiascend.com/document/detail/zh/mindie/10RC3/envdeployment/instg/mindie_instg_0028.html) section in the same guide.

Install msModelSlim. For details, see [msModelSlim Installation Guide](../getting_started/install_guide.md).

## Quantized Weight Generation

  The following example uses Llama2-13b-hf.

  (1) Prepare the model weight files.

```text
├── config.json
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json
├── pytorch_model-00001-of-00006.bin
├── ...
├── pytorch_model-00006-of-00006.bin
├── pytorch_model.bin.index.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── tokenizer.model
```

  (2) Generate W8A8 quantized weights.

```shell
# Enter the acceleration library directory.
cd ${ATB_SPEED_HOME_PATH}
# Run the script to generate quantized weights.
bash examples/models/llama/generate_quant_weight.sh -src {floating_point_weight_path} -dst {W8A8_quantized_weight_save_path} -type llama2_13b_w8a8

# # The above command is used to generate W8A8 weights of the Llama-2-13b-hf model. Different model requires different parameter settings. Check the README file in advance.
# The config.json for W8A8 quantized weights should include the quantize field, and its value should be "w8a8".
# In addition to calling the msModelSlim quantization tool to generate weights and the weight description file, the MindIE quantization script also copies the tokenizer files and copies and modifies config.json into the quantized weight save path.
```

  (3) Directory layout after quantization:

```text
├─ config.json
├─ quant_model_weight_w8a8.safetensors
├─ quant_model_description_w8a8.json
├─ tokenizer_config.json
├─ tokenizer.json
└─ tokenizer.model
```

The quantized output contains the weight file `quant_model_weight_w8a8.safetensors` and the weight description file `quant_model_description_w8a8.json`.

The other files in the directory are required for inference, and they vary slightly by model.

In safetensors, the data is stored as a dictionary with two parts: the quantized weights and the floating-point weights that are not quantized. The key naming rule for the quantized weights is the name of each Linear layer plus the name of its corresponding weight.

The following is a partial view of `quant_model_weight_w8a8.safetensors` after quantization:

```text
{
  "model.embed_tokens.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_scale": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_offset": tensor([...]),
  "model.layers.0.self_attn.q_proj.quant_bias": tensor([...]),
  "model.layers.0.self_attn.q_proj.deq_scale": tensor([...]),
  "model.layers.0.self_attn.k_proj.weight": tensor([...]),
 ...
}
```

The following is a partial view of `quant_model_description_w8a8.json` after quantization:

```text
{
  "model_quant_type": "W8A8",                               # The overall quantization type is W8A8.
  "model.embed_tokens.weight": "FLOAT",                     # `embed_tokens` weights from the floating-point model
  "model.layers.0.self_attn.q_proj.weight": "W8A8",         # Quantized `quant_weight` added for `self_attn.q_proj` in layer 0
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8",    # Quantized `input_scale` added for `self_attn.q_proj` in layer 0
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8",   # Quantized `input_offset` added for `self_attn.q_proj` in layer 0
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8",     # Quantized `quant_bias` added for `self_attn.q_proj` in layer 0
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8",      # Quantized `deq_scale` added for `self_attn.q_proj` in layer 0
  "model.layers.0.self_attn.k_proj.weight": "W8A8",         # Quantized `quant_weight` added for `self_attn.k_proj` in layer 0
 ...
}
```

  (4) Run inference.
Using Llama2-13b-hf as an example, you can run a chat test with the following command. The inference prompt is "What is deep learning". For more details, see the Ascend community development guide: [MindIE LLM Development Guide](https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindiellm/llmdev/mindie_llm0281.html).

```shell
# Enter the acceleration library directory.
cd ${ATB_SPEED_HOME_PATH}
# Run the inference script.
bash examples/models/llama/run_pa.sh {W8A8_quantized_weight_path} 20
# Parameter 20 is the maximum output length.
```

Expected inference result:

```coldfusion
Question: "What's deep learning"
Answer: Deep learning is a subset of machine learning that uses artificial neural networks to learn from data.
```
