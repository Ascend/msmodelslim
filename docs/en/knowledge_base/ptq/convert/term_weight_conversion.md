# Weight Conversion Terminology Entry

> **Entry Category**: Quantization Tool Interface<br>
> **English Name**: Weight Conversion<br>
> **Application Domain**: Format / precision transformation of existing quantized weights, offline deployment

---

## 1. Overview

Weight conversion (Convert) is an offline, data-free quantization tool capability in msModelSlim. Without loading model code or relying on a calibration dataset, it performs a format or precision transformation on an existing checkpoint. Unlike regular [post-training quantization (PTQ)](../term_ptq.md), which requires calibration data, weight conversion operates directly on tensor-level IR (Intermediate Representation) and selects the shortest conversion path through an automatic routing mechanism. It applies to scenarios such as FP8 block dequantization, BF16 to MXFP8, and INT4 group dequantization. Weight conversion is usually a follow-up step to [LLM quantization](../llm/term_large_language_model_quantization.md), used to adjust the on-disk format of existing quantized weights so that they fit different inference frameworks.

## 2. Term Description

In real deployments it is common to have existing quantized weights (such as FP8 block and INT4 packed formats) that need to be converted to another precision or on-disk format. For example, FP8 weights may need to be dequantized to BF16 for inference in the HuggingFace ecosystem, or BF16 floating-point weights may need to be offline-quantized to W8A8 MXFP8 for deployment on an Ascend NPU. Regular one-click quantization requires loading model code and a calibration dataset, which is cumbersome and may introduce extra accuracy loss. Weight conversion provides a lightweight, purely data-driven transformation path that does not depend on model code, greatly simplifying format adaptation.

Weight conversion is based on the IR (Intermediate Representation) concept: tensors in a checkpoint are abstracted into IR nodes, and the shortest conversion path is selected automatically through a registered IR graph. Its core mechanisms include IR abstraction, automatic routing, the virtual module tree, and a seven-stage pipeline.

## 3. Core Principles

### 3.1 Mathematical Description

Weight conversion involves the following three core mathematical mappings:

#### 3.1.1 FP8 Block Dequantization (Lossless)

$$ x_{\text{bf16}} = x_{\text{fp8}} \cdot s_{\text{scale\_inv}} $$

Here $x_{\text{fp8}}$ is a weight in FP8 block format, and $s_{\text{scale\_inv}}$ is the reciprocal of the block-level scaling factor.

#### 3.1.2 INT4 Packed Dequantization (Lossless)

$$ x_{\text{bf16}} = (x_{\text{int4}} - z) \cdot \Delta $$

Here $x_{\text{int4}}$ is a weight in grouped INT4 packed format, $\Delta$ is the group scaling factor, and $z$ is the group zero point.

#### 3.1.3 FLOAT to MXFP8 Quantization (Lossy)

$$ \hat{x} = \text{round}\left(\frac{x}{\Delta_{\text{block}}}\right) $$

Here $\Delta_{\text{block}}$ is the block-level scaling factor; the MXFP8 format gathers statistics independently at block granularity.

### 3.2 Key Properties

- **Data-free property**: no calibration dataset and no model code are required; it is purely offline weight computation.
- **Lossless conversion**: `FP8_BLOCK → FLOAT` and `INT4_PACKED → FLOAT` are lossless conversions (dequantization only).
- **Lossy conversion**: `FLOAT → W8A8_MXFP8` is a lossy conversion (requantization introduces quantization error).
- **Automatic routing**: with `route: auto` it infers the source IR and selects the conversion path, supporting multi-step chaining.
- **Seven-stage pipeline**: from reading the catalog to writing to disk, the whole process requires no model forward pass.
- **Multiple save formats**: `ascend_v1` (MXFP8 deployment) and `huggingface` (HF ecosystem).
- **NPU / CPU**: use `--device npu --device_id` for NPU, and `parallel.cpu_workers` for CPU.

## 4. Process Flow

> For detailed algorithm steps and code-level analysis, see the *[Weight Conversion Usage Guide](./usage_weight_conversion.md)*.

### 4.1 Processing Flow

```mermaid
flowchart LR
    A[Source weight checkpoint] --> B[Read catalog and preprocess]
    B --> C[Virtual tree construction and IR routing]
    C --> D[IR conversion execution]
    D --> E[Save to the specified format]
```

- **Read catalog**: scan the checkpoint files to obtain all tensor keys and dtype information.
- **Preprocess**: optional; performs structural transforms on checkpoint keys (such as splitting fused gate/up).
- **Build virtual module tree**: organizes the module hierarchy in tick-tock mode and builds the `ModelFreeModule` tree.
- **Create IR tasks**: create an IR conversion task for each matching linear layer.
- **Route**: automatically select the conversion path according to the `route` configuration.
- **Execute**: call the corresponding IR conversion processor (`DequantToFloatProcessor`, `Int4PackedToFloatProcessor`, `MxFp8QuantProcessor`).
- **Save**: write to disk according to the configured `save.type` (`ascend_v1` or `huggingface`).

### 4.2 Configuration Example

The following is a minimal working YAML configuration snippet (FP8 block to BF16 as an example). For the detailed meaning and value ranges of each field, see the *[Weight Conversion Usage Guide](./usage_weight_conversion.md)*.

```yaml
apiversion: modelslim_convert
spec:
  linears:
    - match:
        - "model.layers.*.self_attn.q_proj"
        - "model.layers.*.self_attn.k_proj"
        - "model.layers.*.self_attn.v_proj"
        - "model.layers.*.self_attn.o_proj"
        - "model.layers.*.mlp.gate_proj"
        - "model.layers.*.mlp.up_proj"
        - "model.layers.*.mlp.down_proj"
      target: FLOAT
      route: auto
  save:
    - type: huggingface
      part_file_size: 4
  parallel:
    cpu_workers: 8
```

## 5. Applicable Scenarios and Limitations

### 5.1 Applicable Scenarios

- **FP8 block dequantization**: dequantize FP8 quantized weights to BF16 for inference in the HuggingFace ecosystem.
- **BF16 offline MXFP8 quantization**: offline-quantize BF16 floating-point weights to W8A8 MXFP8 for Ascend NPU inference.
- **FP8 directly to MXFP8**: automatic routing chains `FP8_BLOCK → FLOAT → W8A8_MXFP8` in a single step.
- **INT4 packed dequantization**: dequantize INT4 group-quantized weights to BF16.

### 5.2 Usage Limitations

- **Linear layers only**: only the modules matched by `linears.match` participate in the conversion; unmatched weights are kept as is.
- **Cannot replace regular quantization**: it cannot perform quantization that requires calibration (such as W8A8 INT8, which needs activation statistics).
- **MXFP8 format is unique**: MXFP8 must be written with `ascend_v1`; the HuggingFace format is not supported.
- **No model code loading**: it cannot handle structural transforms that require understanding the model structure.

## 6. Related Workflows

- *[Weight Conversion Usage Guide](./usage_weight_conversion.md)*: the complete workflow, command reference, and detailed configuration protocol for weight conversion.
- *[msmodelslim quant Command Line](../../../api_reference/cli/msmodelslim_quant.md)*: contains copy-ready weight conversion commands.
- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: the complete workflow for regular one-click quantization (including calibration).

## 7. Related Terms

- [LLM Quantization](../llm/term_large_language_model_quantization.md): a prerequisite term; weight conversion is usually follow-up processing on LLM quantized weights.
- [VLM Quantization](../vlm/term_vision_transformer_quantization.md): an application target; VLM quantized weights can also have their format adjusted through conversion.
- [DiT Quantization](../dit/term_diffusion_transformer_quantization.md): an application target; DiT quantized weights can also have their format adjusted through conversion.
- [PTQ](../term_ptq.md): the parent concept; weight conversion is a tool-chain step within the PTQ ecosystem.

## 8. Reference Documents

1. *[Weight Conversion Usage Guide](./usage_weight_conversion.md)*: detailed msModelSlim weight conversion documentation.
2. *[msModelSlim One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: the overall one-click quantization workflow.
3. *[Format Support Matrix](../../quantization_format/README.md)*: description of quantization formats and storage formats.
