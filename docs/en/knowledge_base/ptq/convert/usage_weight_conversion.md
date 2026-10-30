# Weight Conversion Usage Guide

## 1. Scope

This guide is intended for users who need to perform a format or precision transformation on **existing quantized / floating-point weights** (for example, dequantizing an FP8 block to BF16, offline-quantizing BF16 to W8A8 MXFP8, or dequantizing INT4 packed to BF16). **The focus is on providing ready-to-use recommended configurations and copy-ready commands, and on explaining the meaning of each configuration item and when it needs to be adjusted.**

Differences from regular one-click quantization:

| Comparison Item | Regular One-Click Quantization | Weight Conversion (modelslim_convert) |
|--------|--------------|-------------------------------|
| Requires a calibration set | Yes (activation statistics, etc.) | **No** |
| Requires `model_type` | Required | **Not required** |
| Requires `quant_type` | Required when using `--quant_type` to match a best practice; not required when using `--config` | **Not required** (the conversion configuration must be specified through `--config`) |
| Typical scenarios | Floating-point model → W8A8, etc. | FP8 → BF16, BF16 → MXFP8, FP8 → MXFP8, etc. |

For command-line conventions see [Step 3](#step-3-run-the-conversion-command); for the full parameter table see the *[msmodelslim quant Command Line](../../../api_reference/cli/msmodelslim_quant.md)*; for field descriptions see the *[modelslim_convert Configuration Description](../../../api_reference/config/quant/modelslim_convert.md)*.

## 2. Inputs and Deliverables

| Type | Name | Source or Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Source weight directory | Model download or local path | HuggingFace format, containing `config.json` and `*.safetensors` | Can be read; contains `model.safetensors.index.json` or single-file weights |
| Input | Conversion configuration YAML | Specified by `--config` | `apiversion: modelslim_convert`, containing `spec.linears`, `spec.save`, etc. | Configuration validation passes |
| Deliverable | Converted weight directory | Path specified by `--save_path` | The format corresponding to the target IR (`ascend_v1` / `huggingface`) | Contains complete weight files and can be loaded by the target inference framework |

`--save_path` must not point to the source model directory or one of its subdirectories, to avoid overwriting the source weights.

## 3. Process Overview

When getting started, reuse a set of **example configurations from the same family** to establish a baseline, then adjust by layer name, target IR, and save format.

```mermaid
flowchart LR
    A[Determine the source weight format and target IR] --> B[Write or reuse a conversion configuration]
    B --> C[Run the conversion command]
    C --> D[Validate the conversion result]
    D --> E[Deploy to the target inference framework]
```

## 4. Procedure

### Step 1: Confirm the Source and Target

**Action**: First determine three things -- what format the source weights are currently in (FP8 block / BF16 / INT4 packed), what IR the target output is (`W8A8_MXFP8` / `FLOAT`), and what the target deployment framework is (which determines the `save` format). Weight conversion is a **purely offline, data-free** operation: it needs no calibration set, no `model_type`, and no `quant_type`.

Currently supported conversions:

| Source Format | Target | Description | Lossy/Lossless |
|-------|---------|------|-----------|
| `FP8_BLOCK` | `FLOAT` | FP8 block weights are dequantized to BF16 | Lossless |
| `INT4_PACKED` | `FLOAT` | INT4 packed (grouped) weights are dequantized to BF16 | Lossless |
| `FLOAT` | `W8A8_MXFP8` | Offline MXFP8 quantization of BF16/FP16 floating-point weights | Lossy |
| `FP8_BLOCK` | `W8A8_MXFP8` | Dequantize first, then quantize to MXFP8 | Lossy (the second half is quantization) |

The tool identifies the source format automatically from the weights in the checkpoint.

### Step 2: Establish the Recommended Configuration

**Action**: The following gives the recommended configurations for the three most common conversion scenarios. If the target model already has an example conversion from the same family, reuse it first.

**Scenario A: FP8 block → BF16 (lossless HF dequantization)**

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
```

**Scenario B: BF16 → W8A8_MXFP8 (Ascend deployment)**

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
      target: W8A8_MXFP8
      route: auto
  save:
    - type: ascend_v1
      part_file_size: 4
  parallel:
    cpu_workers: 8
```

**Scenario C: FP8 block → W8A8_MXFP8**

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
      target: W8A8_MXFP8
      # Explicit path: dequantize first, then quantize. When omitted or set to auto, the tool chooses the same shortest path.
      route: [FP8_BLOCK, FLOAT, W8A8_MXFP8]
  save:
    - type: ascend_v1
      part_file_size: 4
```

Complete example configurations:

| Conversion Scenario | Configuration |
|---------|------|
| BF16 → W8A8_MXFP8 | [qwen3_8b_bf16_to_mxfp8.yaml](./qwen3_8b_bf16_to_mxfp8.yaml) |
| FP8 block → W8A8_MXFP8 | [qwen3_8b_fp8_to_mxfp8.yaml](./qwen3_8b_fp8_to_mxfp8.yaml) |
| FP8 block → BF16 (HF format) | [qwen3_8b_fp8_to_bf16.yaml](./qwen3_8b_fp8_to_bf16.yaml) |
| INT4 packed → BF16 | [kimi_k2_5_int4_per_group_to_bf16.yaml](./kimi_k2_5_int4_per_group_to_bf16.yaml) |

<a id="step-3-run-the-conversion-command"></a>

### Step 3: Run the Conversion Command

**Action**: The entry point is the same as one-click quantization, `msmodelslim quant`; use `--config` to specify the YAML from the previous step.

Minimal working command:

```bash
msmodelslim quant \
  --model_path "${MODEL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --config "${CONFIG_PATH}"
```

`${MODEL_PATH}` is the source weight directory, `${SAVE_PATH}` is the conversion output directory, and `${CONFIG_PATH}` is the YAML from the previous step (either an example file or a custom file). If the model directory contains custom code that must run during loading, additionally pass `--trust_remote_code true`.

NPU multi-card acceleration (recommended for large models / MoE):

```bash
msmodelslim quant \
  --model_path "${MODEL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --config "${CONFIG_PATH}" \
  --device npu \
  --device_id 0 1 2 3
```

`--device_id` specifies which NPUs to use. When running on CPU, set `parallel.cpu_workers` instead.

### Step 4: Select and Tune Parameters

**Action**: First confirm which layers were converted and whether the target format and the save format match; then tune parallelism if acceleration is needed.

`W8A8_MXFP8` pairs with `ascend_v1`, and `FLOAT` pairs with `huggingface`.

| Configuration Item | Meaning | Recommended Setting | Selection and Tuning Advice |
| --- | --- | --- | --- |
| `linears.match` | Module path patterns of the linear layers to be converted; `*` wildcards are supported. Only the matched Linear weights take part in IR conversion; **unmatched weights (such as Norm, Embedding, and Head) are copied as is** to the output directory. | List all Linear projection layers in the target model: `q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj`. | If the match range is too broad, the tool tries to convert non-Linear layers (which may fail); if it is too narrow, some Linear weights keep their original format. Check the checkpoint's `model.safetensors.index.json` to confirm the key names first, then write the wildcards from the layer-name prefixes and projection names. |
| `linears.target` | The target IR type of the conversion, which determines the numerical format of the output weights. Currently supported: `FLOAT` (BF16 floating point) and `W8A8_MXFP8` (Ascend MXFP8). | FP8 dequantization → `FLOAT`; Ascend deployment → `W8A8_MXFP8`. | `target` must match `save.type`: `W8A8_MXFP8` must be written with `ascend_v1`, and `FLOAT` with `huggingface`. Configuration validation checks this item. |
| `linears.route` | The IR conversion path from source to target. `auto` infers the source IR from the weights and takes the shortest path; it can also be written as an IR list, whose first element is the source IR and last element is `target`. | Defaults to `auto`. Write a list when a fixed path is needed, for example dequantization only `[FP8_BLOCK, FLOAT]`, or FP8 to MXFP8 `[FP8_BLOCK, FLOAT, W8A8_MXFP8]`. | Adjacent IRs in the list must have a registered conversion edge. See Scenario C. |
| `save.type` | The save format: `ascend_v1` (Ascend, corresponding to MXFP8) and `huggingface` / `hf` / `compressed_tensors` (HF ecosystem, corresponding to BF16 floating point). | `ascend_v1` (when the target IR is `W8A8_MXFP8`); `huggingface` (when the target IR is `FLOAT`). | The format must strictly match the target IR; a mismatch will prevent the target framework from loading the weights. |
| `save.part_file_size` | The size of each weight shard file, in GB; `0` means no sharding. | Defaults to `4` (4 GB shards). | Small models can use `0` for no sharding; large models should keep `4` to make huge checkpoints easier to manage. |
| `parallel.cpu_workers` | The number of parallel workers for CPU conversion: `1` for a single process, `>1` for multi-process parallelism (bypassing the GIL); defaults to `8`. NPU conversion always uses one child process per card and runs serially within a group (processes = cards), so this field does not apply there. | The default `8` suits most scenarios; increase it appropriately for very large models or MoE. | It takes effect only with `--device cpu`. The larger `cpu_workers` is, the higher the CPU parallelism, but it is limited by memory bandwidth and disk I/O; NPU multi-card parallelism is controlled by `--device_id` and needs no setting. |
| `preprocess` | Weight-graph preprocessing; performs structural transforms on checkpoint keys before the virtual module tree is built. It supports `rename` (renaming keys) and `convert` (split fused weights via chunk / merge weights). | Generally not needed; it must be configured when splitting a fused gate_up_proj in an MoE model. | Needed only when the checkpoint contains fused weights (such as `gate_up_proj`) that must be split into separate projections. See the MoE examples under the `convert/` directory. |
| `defaults.src_format` | The source weight format; `auto` infers it from the model adapter / weight directory. | Defaults to `auto`. | Generally no change is needed. The tool infers it automatically from the weight dtype in the checkpoint. |
| `defaults.dst_format` | The target save format; synonymous with `save.type`, and used as a fallback when `save` is empty. The code default is `ascendv1` (without an underscore), which corresponds to `ascend_v1` in `save.type`. | Defaults to `ascendv1`. | Generally no change is needed; control it directly through `save.type`. |
| `defaults.dst_ir` | The target IR type; when unset it is determined by the target format. | Defaults to `null`. | Generally no change is needed; control it directly through `linears.target`. |

> The device is determined by the command-line `--device`/`--device_id` (the YAML does not configure a device, matching the semantics of the quantization command): `--device npu` (the default) performs NPU conversion, and when `--device_id` is not passed it defaults to card 0; pass `--device_id 0 1 2 3` for multiple cards. `--device cpu` forces CPU conversion and ignores `--device_id`. If NPU is specified but no card is available in the environment (or the card ID is invalid), the task fails immediately instead of silently falling back to CPU.

### Parameter Combinations and Selection Order

1. **First determine the source format and target IR**: `FP8_BLOCK → FLOAT` (lossless dequantization), or `FLOAT → W8A8_MXFP8` (lossy quantization), or `FP8_BLOCK → W8A8_MXFP8` (automatic routing).
2. **Then fix the `linears.match` range**: follow the layer-name patterns of the example YAML for the same model family and keep the wildcard style consistent.
3. **Finally tune the parallelism and shard size**: `parallel.cpu_workers` affects conversion speed, and `part_file_size` affects output file management.

### Step 5: Converge the Parameter Plan Based on Results

**Action**: After the conversion finishes, check whether the output directory contains the expected weight files, and load them with the target inference framework to run a smoke test.

- **Lossless conversion** (FP8 / INT4 → BF16): no further conversion parameters need tuning.
- **Lossy conversion** (target is `W8A8_MXFP8`): if accuracy is insufficient, switch to regular quantization with a calibration set.
- **MoE split failure**: verify `num_experts` in `config.json` (it may also live under `text_config` / `language_config`). If the weights are already separate `gate_proj` / `up_proj`, do not configure `preprocess` splitting; just write `match` instead.

## 5. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| Weight Conversion | Offline, data-free weight format / precision transformation | [Weight Conversion Entry](./term_weight_conversion.md) |
| IR | Intermediate Representation, an abstract representation of tensor format | [Weight Conversion Entry - Core Principles](./term_weight_conversion.md#3-core-principles) |

## 6. Interface Documentation List

| API or Capability | Brief Description | Link |
| --- | --- | --- |
| `msmodelslim quant` | Command-line entry point, parameters, and weight conversion examples | [msmodelslim quant Command Line](../../../api_reference/cli/msmodelslim_quant.md) |
| modelslim_convert Configuration Description | Field descriptions for `linears` / `save` / `parallel` and so on | [modelslim_convert Configuration Description](../../../api_reference/config/quant/modelslim_convert.md) |
| Format Support Matrix | Description of quantization formats and storage formats | [Format Support Matrix](../../quantization_format/README.md) |
| AscendV1 Quantization Results | Description of the AscendV1 on-disk results | [One-Click Quantization Generated Results](../../quantization_format/ascendv1/ascendv1_usage.md) |
