# Foundation Model Support Matrix

**Notes**:

- You can click a link in the **Model Type** column to go to the best practices page recommended by msModelSlim, which describes the quantization commands and configuration in detail.
- √ indicates that the quantization strategy has been officially verified by msModelSlim. It features complete functions and stable performance and is recommended for use.
- \- indicates that the quantization strategy has not been officially verified by msModelSlim. You can configure and try the strategy as needed, but the quantization effect and function stability cannot be guaranteed.
- The "model name-quantization mode (such as w8a8s)" combination marked with "one-click quantization" can be used to execute the following *[one-click quantization](../../user_guide/usage_one_click_quantization.md)* command line to perform model quantization after *[installation](../../install_guide/install_guide.md)*.
- Because newer versions of the Qwen series with more powerful capabilities have been released, the Qwen1.5-14B/32B/72B models have exceeded the maintenance period. Old models in this series will be sunset, and maintenance support for the quantization modes of their live-network versions will be discontinued.
- **Qwen3.5 / Qwen3.6** have native multimodal (Vision) capabilities. For calibration modality descriptions, see the *[Qwen3.5 Quantization Guide - Calibration Modality Support](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Qwen3_5/README.md)*.

```bash
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type ${MODEL_TYPE} --quant_type ${QUANT_TYPE} --trust_remote_code true
```

- For best practices not marked with "one-click quantization", read the best practices page of the corresponding model type and execute the command in the subdirectory of the corresponding model type in the [example](https://gitcode.com/Ascend/msmodelslim/tree/26.2.0/example) directory, such as [DeepSeek](https://gitcode.com/Ascend/msmodelslim/tree/26.2.0/example/DeepSeek) or [Qwen3](https://gitcode.com/Ascend/msmodelslim/tree/26.2.0/example/Qwen).

## Quantization Mode Naming Convention

The format of the quantization mode name is `W{weight_bit}A{activation_bit}[C{cache_bit}][S]`, where

- `{weight_bit}` indicates weight quantization bits (such as 8, 4, or 16).
- `{activation_bit}` indicates activation quantization bits (such as 8 or 16).
- `{cache_bit}` (optional) indicates KV cache quantization bits (such as 8).
- `S` (optional) indicates sparse quantization.

## Supported LLMs

> **Note**: This table contains a large amount of data. If it is not fully visible, use the **scroll bar** at the bottom or **hold the mouse wheel** to scroll horizontally.

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
      <th>w4a4c8<sup>2</sup></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="14"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/DeepSeek/README.md">DeepSeek series</a></strong></td>
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
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>DeepSeek-V3.1-Terminus</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
    </tr>
    <tr>
      <td>DeepSeek-V3.2-Exp</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
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
      <td>√ (one-click quantization)</td>
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
      <td>DeepSeek-V4-Flash</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
    </tr>
    <tr>
      <td>DeepSeek-V4-Flash-0731</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)<sup>5</sup></td>
    </tr>
    <tr>
      <td>DeepSeek-V4-Pro</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
    </tr>
    <tr>
      <td>DeepSeek-V4-Pro-0813</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)<sup>5</sup></td>
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
      <td>-</td>
    </tr>
    <tr>
      <td>DeepSeek-R1-0528</td>
      <td>transformers==4.48.2</td>
      <td>-</td>
      <td>√</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>√</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="6"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/DeepSeek/DeepSeek-R1-Distill/README.md">DeepSeek-R1-Distill series</a></strong></td>
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
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Qwen/README.md">Qwen3 series</a></strong></td>
      <td>Qwen3-8B</td>
      <td>transformers==4.51.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-14B</td>
      <td>transformers==4.51.0</td>
      <td>-</td>
      <td>√ (one-click quantization, supported only by MindIE)<sup>4</sup></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-32B</td>
      <td>transformers==4.51.0</td>
      <td>-</td>
      <td>√ (one-click quantization, supported only by MindIE)<sup>4</sup></td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>√ (one-click quantization)</td>
      <td>√</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="3"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Qwen3-MOE/README.md">Qwen3-MOE series</a></strong></td>
      <td>Qwen3-30B-A3B</td>
      <td>transformers==4.57.6</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-235B-A22B</td>
      <td>transformers==4.57.6</td>
      <td>-</td>
      <td>√</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-Coder-480B-A35B</td>
      <td>transformers==4.57.6</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="7"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Qwen3_5/README.md">Qwen3.5 series</a></strong></td>
      <td>Qwen3.5-397B-A17B</td>
      <td>transformers==5.2.0</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
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
      <td>√ (one-click quantization)</td>
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
      <td>Qwen3.5-35B-A3B</td>
      <td>transformers==5.2.0</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
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
      <td>Qwen3.5-27B</td>
      <td>transformers==5.2.0</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
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
      <td>Qwen3.5-4B</td>
      <td>transformers==5.2.0</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
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
      <td>Qwen3.6-27B</td>
      <td>transformers==5.2.0</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3.6-35B-A3B</td>
      <td>transformers==5.2.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)<sup>4</sup></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Qwen3-Next/README.md">Qwen3-Next series</a></strong></td>
      <td>Qwen3-Next-80B-A3B-Instruct</td>
      <td>transformers>=4.57.0</td>
      <td>-</td>
      <td>√ (one-click quantization, supported only by vLLM Ascend)</td>
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
      <td rowspan="5"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Qwen/README.md">Qwen2.5 series</a></strong></td>
      <td>Qwen2.5-7B-Instruct</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
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
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen2.5-32B-Instruct</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
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
      <td>Qwen2.5-72B-Instruct</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
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
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Qwen/README.md">Qwen2 series</a></strong></td>
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
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="7"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Qwen/README.md">Qwen series</a></strong></td>
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
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Qwen/README.md">QwQ series</a></strong></td>
      <td>QwQ-32B</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="5"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/GLM-5/README.md">GLM5-MOE series</a></strong></td>
      <td>GLM-5</td>
      <td>transformers==5.4.0</td>
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
      <td>GLM-5.1</td>
      <td>transformers==5.4.0</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
    </tr>
    <tr>
      <td>GLM-5.2</td>
      <td>transformers==5.12.0</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>√ (one-click quantization)</td>
    </tr>
    <tr>
      <td>GLM-5.3</td>
      <td>transformers==5.15.0</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
    </tr>
    <tr>
      <td>GLM-5.3-Flash</td>
      <td>transformers==5.16.0</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/GLM/README.md">GLM series</a></strong></td>
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
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/GLM4-MOE/README.md">GLM4-MOE series</a></strong></td>
      <td>GLM-4.7</td>
      <td>transformers==4.57.3</td>
      <td>-</td>
      <td>√ (one-click quantization, supported only by vLLM Ascend)</td>
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
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/GLM4-MOE-LITE/README.md">GLM4-MOE-LITE series</a></strong></td>
      <td>GLM-4.7-Flash</td>
      <td>transformers==5.3.0</td>
      <td>-</td>
      <td>√ (one-click quantization, supported only by sglang)</td>
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
      <td rowspan="1"><strong><a href="../../../../example/Hy3/README.md">Hy3 series</a></strong></td>
      <td>Hy3</td>
      <td>transformers==5.6.0</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
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
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/HunYuan/README.md">HunYuan series</a></strong></td>
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
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="../../../../example/MiniMax-M2/README.md">MiniMax-M2 series</a></strong></td>
      <td>MiniMax-M2.7</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="../../../../example/MiMo-V2-Flash/README.md">MiMo series</a></strong></td>
      <td>MiMo-V2-Flash</td>
      <td>transformers==4.57.6</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
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
      <td rowspan="1"><strong><a href="../../../../example/longcat_flash/README.md">LongCat-Flash series</a></strong></td>
      <td>LongCat-Flash-Chat</td>
      <td>transformers==4.55.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Step3p5/README.md">stepfun series</a></strong></td>
      <td>Step-3.5-Flash</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
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
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/InternLM2/README.md">InternLM series</a></strong></td>
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
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="8"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Llama/README.md">LLaMA series</a></strong></td>
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
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>

**Notes**:

- <sup>1</sup> The w8a16 quantization mode is supported only by MindIE.
- <sup>2</sup> Both KVCache and FA3 quantization are categorized under c8, as both quantize the K and V caches in the LLM. Only MindIE supports c8 quantization modes, including w8a8c8, w4a8c8, and w4a4c8.
- <sup>3</sup> For better results, use the decompression feature of Atlas inference products after compression. Only MindIE supports sparse quantization modes, including w8a8s and w16a16s.
- <sup>4</sup> Only MindIE supports best practices that use the PDMIX quantization scheme.
- <sup>5</sup> For DeepSeek-V4-Flash-0731 and DeepSeek-V4-Pro-0813, the w4a4c8 column is a W4A4C8 mixed-precision scheme: MXFP4 for routed experts, MXFP8 for attention and shared experts, and the official FP8 reused for KV. Use `--quant_type w4a4c8` during quantization. The verified scenarios are vLLM-Ascend + Ascend_950 and vLLM-Ascend + Atlas_350. For details, see the [DeepSeek quantization guide](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/DeepSeek/README.md).

## Supported MLLMs

> **Calibration modality**: For the calibration modalities supported by each model (text-only / image-text / audio-video, and so on) and the homogeneity constraints, refer to the corresponding example README. For example:
>
> - [Qwen3-Omni — Calibration Modality Support](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Qwen3-Omni/README.md): Text-only and any combination of image/audio/video; the same task must be homogeneous.
> - [Qwen3.5 — Calibration Modality Support](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Qwen3_5/README.md): Text-only and text+image (Qwen3.5 is listed in the LLM table, but its quantization follows the VLM path).

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
      <th>w4a8</th>
      <th>w4a8c8</th>
      <th>w4a4c8</th>
            <th>w4a4f4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Qwen3-VL/README.md">Qwen3-VL series</a></strong></td>
      <td>Qwen3-VL-4B-Instruct</td>
      <td>transformers==4.57.1</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-VL-8B-Instruct</td>
      <td>transformers==4.57.1</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-VL-32B-Instruct</td>
      <td>transformers==4.57.1</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="4"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Qwen3-VL-Embedding/README.md">Qwen3-VL-Embedding&Reranker series</a></strong></td>
      <td>Qwen3-VL-Embedding-2B</td>
      <td>transformers==4.57.1</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-VL-Embedding-8B</td>
      <td>transformers==4.57.1</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-VL-Reranker-2B</td>
      <td>transformers==4.57.1</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-VL-Reranker-8B</td>
      <td>transformers==4.57.1</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Qwen3-VL-MoE/README.md">Qwen3-VL-MoE series</a></strong></td>
      <td>Qwen3-VL-235B-A22B</td>
      <td>transformers==4.57.1, flax</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Qwen3-Omni/README.md">Qwen3-Omni series</a></strong></td>
      <td>Qwen3-Omni-30B-A3B-Thinking</td>
      <td>transformers==4.57.3</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-Omni-30B-A3B-Instruct</td>
      <td>transformers==4.57.3</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Gemma4/README.md">Gemma4 series</a></strong></td>
      <td>gemma-4-31B-it</td>
      <td>transformers==5.5.3</td>
      <td>√ (one-click quantization, supported only by vLLM Ascend; see also the README: MXFP8/MXFP4 hybrid, one-click quantization w4a4)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Qwen2.5-VL/README.md">Qwen2.5-VL series</a></strong></td>
      <td>Qwen2.5-VL-7B</td>
      <td>transformers==4.49.0, qwen_vl_utils</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-72B</td>
      <td>transformers==4.49.0, qwen_vl_utils</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Qwen2.5-Omni/README.md">Qwen2.5-Omni series</a></strong></td>
      <td>Qwen2.5-Omni-7B</td>
      <td>transformers==4.57.3</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Qwen2-VL/README.md">Qwen2-VL series</a></strong></td>
      <td>Qwen2-VL-7B</td>
      <td>transformers==4.46.0, qwen_vl_utils</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>Qwen2-VL-72B</td>
      <td>transformers==4.46.0, qwen_vl_utils</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Qwen-VL/README.md">Qwen-VL series</a></strong></td>
      <td>Qwen-VL</td>
      <td>transformers-stream-generator</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/InternVL2/README.md">InternVL2 series</a></strong></td>
      <td>InternVL2-8B</td>
      <td>transformers==4.46.0, timm, fastchat</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>InternVL2-40B</td>
      <td>transformers==4.46.0, timm, fastchat</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/LLaVA/README.md">LLaVA series</a></strong></td>
      <td>LLaVA-1.5-7B</td>
      <td>transformers==4.37.2</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/GLM-4.1V/README.md">GLM-4.1V series</a></strong></td>
      <td>GLM-4.1V-9B-Thinking</td>
      <td>transformers==4.53.0</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/GLM-4.6V/README.md">GLM-4.6V</a></strong></td>
      <td>GLM-4.6V</td>
      <td>transformers==5.0.0rc0</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Kimi-K2.5/README.md">Kimi-K2.5</a></strong></td>
      <td>Kimi-K2.5</td>
      <td>transformers==4.57.6, compressed-tensors==0.13.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Kimi-K2.6/README.md">Kimi-K2.6</a></strong></td>
      <td>Kimi-K2.6</td>
      <td>transformers==4.57.6, compressed-tensors==0.13.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
      <td>-</td>
      <td>√</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Kimi-K3/README.md">Kimi-K3</a></strong></td>
      <td>Kimi-K3</td>
      <td>transformers==4.57.6, compressed-tensors==0.13.0, fla-core</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)<sup>3</sup></td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/InternVL3_5/README.md">InternVL3_5 series</a></strong></td>
      <td>InternVL3_5-38B</td>
      <td>transformers==4.57.6</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/InternVL3_5-MoE/README.md">InternVL3_5-MoE series</a></strong></td>
      <td>InternVL3_5-241B-A28B</td>
      <td>transformers==4.57.6</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Gemma4-MoE/README.md">Gemma4-MoE series</a></strong></td>
      <td>gemma-4-26B-A4B-it</td>
      <td>transformers==5.5.3</td>
      <td>√ (see also the README: MXFP8/MXFP4 hybrid, one-click quantization w4a4)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/MiniMax-M3/README.md">MiniMax-M3 series</a></strong></td>
      <td>MiniMax-M3-preview</td>
      <td>transformers==5.12.0</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td rowspan="1"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Step-3.7-Flash/README.md">Step-3.7-Flash series</a></strong></td>
      <td>Step-3.7-Flash</td>
      <td>transformers>=4.57.1,&lt;5.0.0</td>
      <td>√ (one-click quantization, supported only by vLLM Ascend)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="7"><strong><a href="https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_sd/README.md">Multimodal Generative Models</a></strong></td>
      <td>SD3-Medium</td>
      <td>diffusers</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>Open-Sora-Plan v1.2</td>
      <td>huggingface_hub==0.25.2</td>
      <td>√</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>FLUX.1-dev</td>
      <td>-</td>
      <td>√</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>HunyuanVideo</td>
      <td>-</td>
      <td>√</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>Wan2.1</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    <td>-</td>
    </tr>
    <tr>
      <td>Wan2.2</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
    </tr>
    <tr>
      <td>Qwen-Image-Edit</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>√ (one-click quantization)</td>
    </tr>
  </tbody>
</table>
</div>

**Notes**:

- <sup>1</sup> For better results, use the decompression feature of Atlas inference products after compression. Only MindIE supports sparse quantization modes.
- <sup>2</sup> FLUX.1-dev, HunyuanVideo, Wan2.2, and Qwen-Image-Edit-2509 support [MXFP quantization](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
- <sup>3</sup> The w4a8 of Kimi-K3 contains two practices, INT and MXFP (which share `quant_type=w4a8`). MXFP requires additionally specifying `--tags vLLM-Ascend Ascend_950`. For details, see the [Kimi-K3 quantization case](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/multimodal_vlm/Kimi-K3/README.md).
- <sup>4</sup> The w8a8c8 of Qwen3.6-35B-A3B is a mixed-precision scheme: MXFP8 for linear layers (W8A8), MXFP4 for routed experts (W4A4), and FA3 quantization for attention (Q/K MXFP8 per-block dynamic + V MXFP8 per-channel static), with KV Cache quantization included. It is currently adapted only for vLLM-Ascend + Ascend_950, and `--tags vLLM-Ascend Ascend_950` must be specified during quantization. For details, see the [Qwen3.5 quantization guide](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/example/Qwen3_5/README.md).
