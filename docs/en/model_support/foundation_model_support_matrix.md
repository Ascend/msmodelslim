# Foundation Model Support Matrix

**Notes**:

- You can click the link in the **Model Type** column to go to the best practices page recommended by msModelSlim. This page describes the quantization commands and configurations in detail.
- √ indicates that the quantization strategy is officially verified by msModelSlim. It features complete functions and stable performance and is recommended for use.
- \- indicates that the quantization strategy has not been officially verified by msModelSlim. You can configure and try the strategy as needed, but the quantization effect and function stability cannot be guaranteed.
- The combination of "model name-quantization mode (such as w8a8s)" marked with "quick quantization" can be used to execute the following [quick quantization](../feature_guide/quick_quantization_v1/usage.md) command to perform model quantization after [installation](../getting_started/install_guide.md).
- Because newer versions of the Qwen series have been released with more powerful capabilities, the Qwen1.5-14B/32B/72B models have exceeded the maintenance period. Old models in this series will be sunset, and maintenance support for the quantization modes of their live-network versions will be discontinued.

```bash
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type ${MODEL_TYPE} --quant_type ${QUANT_TYPE} --trust_remote_code True
```

- For best practices not marked with "quick quantization", read the best practices page of the corresponding model type. Execute the command in the subdirectory of the model type within the [example](https://gitcode.com/Ascend/msmodelslim/tree/master/example) directory, such as [DeepSeek](https://gitcode.com/Ascend/msmodelslim/tree/master/example/DeepSeek) or [Qwen3](https://gitcode.com/Ascend/msmodelslim/tree/master/example/Qwen).

## Quantization Mode Naming Convention

The format of the quantization mode name is `W{weight_bit}A{activation_bit}[C{cache_bit}][S]`, where

- `{weight_bit}` indicates weight quantization bits (such as 8, 4, or 16).
- `{activation_bit}` indicates activation quantization bits (such as 8 or 16).
- `{cache_bit}` (optional) indicates KV cache quantization bits (such as 8).
- `S` (optional) indicates sparse quantization.

## Supported LLMs

!!! info "Note"
    This table contains a large amount of data. If it is not fully visible, use the **scroll bar** at the bottom or **hold the mouse wheel** to scroll horizontally.

<div class="custom-table">

<table>
  <thead>
    <tr>
      <th>Model Type</th>
      <th>Model Name</th>
      <th>Dependency Library</th>
      <th>w8a16<sup>1</sup></th>
      <th>w8a8</th>
      <th>w4a8</th>
      <th>w4a16</th>
      <th>w8a8c8<sup>2</sup></th>
      <th>w4a8c8<sup>2</sup></th>
      <th>w8a8s (Sparse Quantization)<sup>3</sup></th>
      <th>w16a16s (Floating-point Sparse Quantization)<sup>3</sup></th>
      <th>w4a4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="9"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/DeepSeek/README.md">DeepSeek series</a></strong></td>
      <td>DeepSeek-V2-16B</td>
      <td>-</td>
      <td>√</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-V2-236B</td>
      <td>-</td>
      <td>√</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-Coder-33B</td>
      <td>-</td>
      <td>√</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-V3</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-V3.1</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>√</td>
      <td>√</td>
      <td>-</td>
      <td>√</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-V3.2-Exp</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-V3.2</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-R1</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>√</td>
      <td>√</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-R1-0528</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>√</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>√</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="6"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/DeepSeek/DeepSeek-R1-Distill/README.md">DeepSeek-R1-Distill series</a></strong></td>
      <td>DeepSeek-R1-Distill-Llama-8B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-R1-Distill-Llama-70B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-R1-Distill-Qwen-1.5B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-R1-Distill-Qwen-7B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-R1-Distill-Qwen-14B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-R1-Distill-Qwen-32B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen/README.md">Qwen3 series</a></strong></td>
      <td>Qwen3-8B</td>
      <td>transformers==4.51.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-14B</td>
      <td>transformers==4.51.0</td>
      <td>-</td>
      <td>√ (quick quantization, supported only by MindIE)<sup>4</sup></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-32B</td>
      <td>transformers==4.51.0</td>
      <td>-</td>
      <td>√ (quick quantization, supported only by MindIE)<sup>4</sup></td>
      <td>-</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>√ (quick quantization)</td>
      <td>√</td>
    </tr>
    <tr>
      <td rowspan="3"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen3-MOE/README.md">Qwen3-MOE series</a></strong></td>
      <td>Qwen3-30B-A3B</td>
      <td>transformers==4.51.0</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-235B-A22B</td>
      <td>transformers==4.51.0</td>
      <td>-</td>
      <td>√</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-Coder-480B-A35B</td>
      <td>transformers==4.51.0</td>
      <td>-</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="4"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen3_5/README.md">Qwen3.5 series</a></strong></td>
      <td>Qwen3.5-397B-A17B</td>
      <td>transformers==5.2.0</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3.5-122B-A10B</td>
      <td>transformers==5.2.0</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3.5-35B-A3B</td>
      <td>transformers==5.2.0</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3.5-27B</td>
      <td>transformers==5.2.0</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen3-Next/README.md">Qwen3-Next series</a></strong></td>
      <td>Qwen3-Next-80B-A3B-Instruct</td>
      <td>transformers>=4.57.0</td>
      <td>-</td>
      <td>√ (quick quantization, supported only by vLLM Ascend)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="5"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen/README.md">Qwen2.5 series</a></strong></td>
      <td>Qwen2.5-7B-Instruct</td>
      <td>-</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen2.5-14B-Instruct</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen2.5-32B-Instruct</td>
      <td>-</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen2.5-72B-Instruct</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen2.5-Coder-7B-Instruct</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen/README.md">Qwen2 series</a></strong></td>
      <td>Qwen2-7B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen2-72B</td>
      <td>-</td>
      <td>√</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="7"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen/README.md">Qwen series</a></strong></td>
      <td>Qwen-7B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen-14B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen-72B</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen1.5-14B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen1.5-32B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen1.5-72B</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen1.5-110B</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen/README.md">QwQ series</a></strong></td>
      <td>QwQ-32B</td>
      <td>-</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/GLM-5/README.md">GLM5-MOE series</a></strong></td>
      <td>GLM-5</td>
      <td>transformers==5.2.0</td>
      <td>-</td>
      <td>√</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/GLM/README.md">GLM series</a></strong></td>
      <td>GLM-4-9B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/GLM4-MOE/README.md">GLM4-MOE series</a></strong></td>
      <td>GLM-4.7</td>
      <td>transformers==4.57.3</td>
      <td>-</td>
      <td>√ (quick quantization, supported only by vLLM Ascend)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/HunYuan/README.md">HunYuan series</a></strong></td>
      <td>Hunyuan-A52B-Instruct</td>
      <td>transformers>=4.48.2</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/InternLM2/README.md">InternLM series</a></strong></td>
      <td>InternLM2-20B</td>
      <td>-</td>
      <td>√</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="8"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/Llama/README.md">Llama series</a></strong></td>
      <td>LLaMA-33B</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>LLaMA-65B</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>LLaMA2-13B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>LLaMA2-7B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>LLaMA2-70B</td>
      <td>-</td>
      <td>√</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>LLaMA3-70B</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>LLaMA3.1-8B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>LLaMA3.1-70B</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

</div>

**Notes**:

- <sup>1</sup> The w8a16 quantization mode is supported only by MindIE.
- <sup>2</sup> Both KVCache and FA3 quantization are categorized under c8, as both methods quantize the K and V caches within the LLM. Only MindIE supports c8 quantization modes, such as w8a8c8 and w4a8c8.
- <sup>3</sup> For optimal performance, use the decompression features of the Atlas 300I Duo products after compression. Only MindIE supports sparse quantization modes, including w8a8s and w16a16s.
- <sup>4</sup> Only MindIE supports best practices that employ the PDMIX quantization scheme.

## Supported MLLMs

<div class="custom-table">

<table>
  <thead>
    <tr>
      <th>Model Type</th>
      <th>Model Name</th>
      <th>Dependency Library</th>
      <th>w8a8</th>
      <th>w8a8c8/w8a8f8</th>
      <th>w8a8s (Sparse Quantization)<sup>1</sup></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_vlm/Qwen3-VL/README.md">Qwen3-VL series</a></strong></td>
      <td>Qwen3-VL-4B-Instruct</td>
      <td>transformers==4.57.1</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-VL-8B-Instruct</td>
      <td>transformers==4.57.1</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>Qwen3-VL-32B-Instruct</td>
      <td>transformers==4.57.1</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_vlm/Qwen3-VL-MoE/README.md">Qwen3-VL-MoE series</a></strong></td>
      <td>Qwen3-VL-235B-A22B</td>
      <td>transformers==4.57.1, flax</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_vlm/Qwen3-Omni/README.md">Qwen3-Omni series</a></strong></td>
      <td>Qwen3-Omni-30B-A3B-Thinking</td>
      <td>transformers==4.57.3</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-Omni-30B-A3B-Instruct</td>
      <td>transformers==4.57.3</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_vlm/Qwen2.5-VL/README.md">Qwen2.5-VL series</a></strong></td>
      <td>Qwen2.5-VL-7B</td>
      <td>transformers==4.49.0, qwen_vl_utils</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-72B</td>
      <td>transformers==4.49.0, qwen_vl_utils</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_vlm/Qwen2.5-Omni/README.md">Qwen2.5-Omni series</a></strong></td>
      <td>Qwen2.5-Omni-7B</td>
      <td>transformers==4.57.3</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_vlm/Qwen2-VL/README.md">Qwen2-VL series</a></strong></td>
      <td>Qwen2-VL-7B</td>
      <td>transformers==4.46.0, qwen_vl_utils</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen2-VL-72B</td>
      <td>transformers==4.46.0, qwen_vl_utils</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_vlm/Qwen-VL/README.md">Qwen-VL series</a></strong></td>
      <td>Qwen-VL</td>
      <td>transformers-stream-generator</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_vlm/InternVL2/README.md">InternVL2 series</a></strong></td>
      <td>InternVL2-8B</td>
      <td>transformers==4.46.0, timm, fastchat</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>InternVL2-40B</td>
      <td>transformers==4.46.0, timm, fastchat</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_vlm/LLaVA/README.md">LLaVA series</a></strong></td>
      <td>LLaVA-1.5-7B</td>
      <td>transformers==4.37.2</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_vlm/GLM-4.1V/README.md">GLM-4.1V series</a></strong></td>
      <td>GLM-4.1V-9B-Thinking</td>
      <td>transformers==4.53.0</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_vlm/GLM-4.6V/README.md">GLM-4.6V</a></strong></td>
      <td>GLM-4.6V</td>
      <td>transformers==5.0.0rc0</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="7"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_sd/README.md"> Multimodal generative models</a></strong></td>
      <td>SD3-Medium</td>
      <td>diffusers</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Open-Sora-Plan v1.2</td>
      <td>huggingface_hub==0.25.2</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>FLUX.1-dev</td>
      <td>-</td>
      <td>√</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>HunyuanVideo</td>
      <td>-</td>
      <td>√</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Wan2.1</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Wan2.2</td>
      <td>-</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen-Image-Edit</td>
      <td>-</td>
      <td>-</td>
      <td>√ (quick quantization)</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

</div>

**Notes**:

- <sup>1</sup> For optimal performance, use the decompression features of the Atlas 300I Duo series after compression. Only MindIE supports sparse quantization modes.
- <sup>2</sup> FLUX.1-dev, HunyuanVideo, Wan2.2, and Qwen-Image-Edit-2509 support [MXFP quantization](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
