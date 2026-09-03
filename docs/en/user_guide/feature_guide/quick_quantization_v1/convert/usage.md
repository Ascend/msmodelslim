# Weight Conversion Guide

<!-- md-trans-meta sourceCommit=7dbddaab8035898034bbdff70b0f57830595194e translatedAt=2026-09-02T01:22:23.476Z pushedAt=2026-09-02T01:42:16.283Z -->

## 1. Introduction

Weight conversion (Convert) is an **offline, data-free** capability in the one-click quantization system: it performs format/precision conversion on existing checkpoints and saves them to disk without loading model code or relying on calibration datasets.

The main differences from conventional one-click quantization (such as `modelslim_v1`) are as follows:

| Comparison Item | Conventional One-click Quantization | Weight Conversion (modelslim_convert) |
|--------|--------------|-------------------------------|
| Whether a calibration set is required | Yes (activation statistics, etc.) | **No** |
| Whether `model_type` is required | Yes | **Optional** (can be omitted when `apiversion: modelslim_convert` is set in the YAML) |
| Whether `quant_type` is required | Required for method 1 | **No** (the conversion configuration must be specified through `config_path`) |
| Typical scenario | Floating-point model → W8A8, etc. | FP8 → BF16, BF16 → MXFP8, FP8 → MXFP8, etc. |

The currently registered IR conversion edges include:

| Source IR | Target IR | Description | Lossy/Lossless |
|-------|---------|------|-----------|
| `FP8_BLOCK` | `FLOAT` | Dequantization of FP8 block weights to BF16 | Lossless |
| `FLOAT` | `W8A8_MXFP8` | Offline MXFP8 quantization of BF16/FP16 floating-point weights | Lossy |

When `route: auto` is configured, the tool **automatically infers the source IR** based on the tensor dtype and fields such as `weight_scale_inv` in the checkpoint, and selects the shortest conversion path on the IR graph. For example, for converting FP8 block weights to MXFP8, the actual path is `FP8_BLOCK → FLOAT → W8A8_MXFP8`.

**On-disk format constraints**:

- When the target IR is **`W8A8_MXFP8`**, it must be saved using **`ascend_v1`** (the Ascend NPU deployment path).
- When the target IR is **`FLOAT`** (for example, dequantizing FP8 to BF16), it can be saved using **`huggingface`**/**`compressed_tensors`** for inference in the Hugging Face ecosystem.

## 2. Preparations

1. Install the msModelSlim tool. For details, see [*msModelSlim Tool Installation Guide*](../../../../install_guide/install_guide.md).
2. Prepare the source weight directory, which must be a Hugging Face-style checkpoint (including `config.json` and `*.safetensors` or the `model.safetensors.index.json` shard index).
3. Write or select a YAML configuration to specify the matching rules for the linear layers to be converted, the target IR, and the on-disk format.

The document directory provides reference configurations for Qwen3-8B, which can be reused directly or modified as needed:

- [qwen3_8b_bf16_to_mxfp8.yaml](./qwen3_8b_bf16_to_mxfp8.yaml): BF16 → W8A8_MXFP8
- [qwen3_8b_fp8_to_mxfp8.yaml](./qwen3_8b_fp8_to_mxfp8.yaml): FP8 block → W8A8_MXFP8
- [qwen3_8b_fp8_to_bf16.yaml](./qwen3_8b_fp8_to_bf16.yaml): FP8 block → BF16 (HF format)

**Precautions**:

1. Weights that are not matched in `linears.match` (such as `embed_tokens`, `lm_head`, LayerNorm, etc.) are **retained as-is** and written to the output directory.
2. To print runtime logs, set the environment variable `MSMODELSLIM_LOG_LEVEL` (optional values: `INFO` (default), `DEBUG`).

## 3. Quick Start

### 3.1 Command Format

Weight conversion reuses the one-click quantization CLI entry point and specifies the `modelslim_convert` configuration through **`--config_path`**:

```bash
msmodelslim quant [ARGS]
```

**Minimal command form** (convert scenario):

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config_path ${CONFIG_PATH}
```

Here, `${CONFIG_PATH}` points to a YAML file with `apiversion: modelslim_convert`.

### 3.2 Parameter Description

| Parameter Name | Optional/Required | Parameter Description |
|----------|-----------|----------|
| model_path | Required | Source weight directory path.<br>Type: Str. |
| save_path | Required | Path for saving the converted weights.<br>Type: Str. |
| config_path | Required | Path to the conversion configuration YAML.<br>1. Type: Str.<br>2. The `apiversion` in the YAML must be `modelslim_convert`.|
| h, help | Optional | Command-line help information. |

**Precautions**:

1. Weight conversion involves only the above three parameters of one-click quantization (except `help`).

### 3.3 Usage Examples

#### 3.3.1 Example 1: Dequantizing FP8 Block Weights to BF16 (HF Format)

Dequantize the linear layers in the Qwen3-8B FP8 checkpoint to BF16 and save them to disk in Hugging Face/compressed_tensors format:

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config_path ./qwen3_8b_fp8_to_bf16.yaml
```

Where:

- `${MODEL_PATH}` is the source directory containing FP8 block weights (`.weight` + `.weight_scale_inv`)
- `${SAVE_PATH}` is the output directory
- For the configuration file, see [qwen3_8b_fp8_to_bf16.yaml](./qwen3_8b_fp8_to_bf16.yaml)

#### 3.3.2 Example 2: Performing Offline Quantization of BF16 Weights to W8A8_MXFP8 (AscendV1 Format)

Quantize the linear layers of the Qwen3-8B BF16 floating-point checkpoint to W8A8_MXFP8 for Ascend NPU inference:

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config_path ./qwen3_8b_bf16_to_mxfp8.yaml
```

For the configuration file, see [qwen3_8b_bf16_to_mxfp8.yaml](./qwen3_8b_bf16_to_mxfp8.yaml).

#### 3.3.3 Example 3: Directly Converting FP8 Block Weights to W8A8_MXFP8

When you already have FP8 block weights and want to deploy them to the Ascend MXFP8 inference path, you can complete dequantization and MXFP8 quantization in one step (`route: auto` automatically chains `FP8_BLOCK → FLOAT → W8A8_MXFP8`):

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config_path ./qwen3_8b_fp8_to_mxfp8.yaml
```

For the configuration file, see [qwen3_8b_fp8_to_mxfp8.yaml](./qwen3_8b_fp8_to_mxfp8.yaml).

### 3.4 Output Description

The output directory depends on `save.type` in the YAML file:

| save.type | Typical Target IR | Output Feature |
|-----------|-------------|----------|
| `ascend_v1` | `W8A8_MXFP8` | Generates AscendV1 quantized weights such as `quant_model_description.json` and `quant_model_weights*.safetensors`. For details, see *[One-click Quantization Generation Result](../quantization_result.md)* |
| `huggingface` / `compressed_tensors` | `FLOAT` | Generates HF-style `config.json`, `model*.safetensors`, and so on, with weights in BF16 floating-point format |

Regardless of the on-disk format, the weights of nonlinear layers that are not included in `linears.match` are copied from the source checkpoint to the output directory.

## 4. Detailed Description of the Conversion Configuration Protocol

### 4.1 Configuration Protocol Overview

#### 4.1.1 Basic Structure

The weight conversion configuration is described in YAML, with the following fixed top-level structure:

```yaml
apiversion: modelslim_convert   # Protocol version, fixed value
spec:                           # Specific configuration of the conversion task
  preprocess: [ ]               # Optional: weight graph structure preprocessing
  linears: [ ]                  # Required: linear layer matching and target IR
  save: [ ]                     # Required: on-disk format
  parallel: { }                 # Optional: parallelism and device
```

Unlike the `modelslim_v1` quantization configuration, `modelslim_convert` **does not include** calibration-related fields such as `runner`, `process`, and `dataset`. The conversion pipeline follows a fixed order: reading catalog → preprocessing → building virtual module tree → IR routing → converting → saving to disk.

#### 4.1.2 Protocol Version Description

| Parameter | Optional/Required | Description |
|------|-----------|------|
| apiversion | Required | Fixed to `"modelslim_convert"`, used to select the convert quantization service backend. |
| spec | Required | Conversion rules, on-disk saving, and parallel parameters. |

### 4.2 modelslim_convert Configuration Details

#### 4.2.1 linears - Linear Layer Conversion Rules

**Function**: Declares which modules participate in IR conversion, as well as the target precision/format.

**Features**:

- **List structure**: Multiple `match` groups can be configured, with each group sharing the same `target` and `route`.
- **Wildcard matching**: `match` supports the `*` wildcard (for example, `model.layers.*.self_attn.q_proj`).
- **Automatic routing**: When `route: auto` is set, the tool infers the source IR and selects the shortest conversion chain; an IR sequence can also be explicitly specified.

**Field description**:

| Field | Type | Description |
|------|------|------|
| match | String list | Module path pattern to be converted, supporting the `*` wildcard. |
| target | IRKind | Target IR. Currently supported: `FLOAT` (BF16 floating-point), `W8A8_MXFP8`. |
| route | `"auto"` or IRKind list | IR path constraint from source to target. Defaults to `"auto"`. |

**Configuration example** (Qwen3-8B dense, 7×36 linear layers converted to MXFP8):

```yaml
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
```

**Source IR inference rules** (when `route: auto`):

- `weight_scale_inv` exists in the checkpoint or the weight dtype is float8 → the source IR is `FP8_BLOCK`
- Only BF16/FP16 `.weight` exists → the source IR is `FLOAT`

#### 4.2.2 save - On-disk Format

**Function**: Specifies the save format and sharding strategy for the conversion result.

**Configuration example**:

```yaml
# AscendV1 (MXFP8 deployment)
spec:
  save:
    - type: ascend_v1
      part_file_size: 4

# HuggingFace / compressed_tensors (BF16 floating-point export)
spec:
  save:
    - type: huggingface
      part_file_size: 4
```

**Field description**:

| Field | Function | Description |
|------|------|------|
| type | Saver type | `ascend_v1`/`ascendv1`/`ascendv1_saver` → AscendV1; `huggingface`/`hf`/`compressed_tensors` → HF ecosystem format. |
| part_file_size | Shard size | Unit: GB; `0` means no sharding, type is int. |

**Correspondence between format and target IR**:

| target IR | Recommended save.type | Description |
|-----------|----------------|------|
| `W8A8_MXFP8` | `ascend_v1` | **Required**; MXFP8 weights are intended only for Ascend NPU deployment. |
| `FLOAT` | `huggingface` | HF-side inference scenarios such as FP8 dequantization. |

If `target` is `W8A8_MXFP8` while `save.type` is `huggingface`, configuration validation will report an error.

#### 4.2.3 parallel - Parallel Configuration

**Function**: Controls the parallelism and device strategy for IR task execution.

**Configuration example**:

```yaml
spec:
  parallel:
    workers: 8              # Number of processes/workers
```

**Field description**:

| Field | Default Value | Description |
|------|--------|------|
| workers | 1 | `1`: single process; `>1`: multi-process parallelism. |

#### 4.2.4 (Optional) preprocess - Weight Graph Preprocessing

**Function**: Performs structural transformation on checkpoint keys (such as splitting fused gate/up) before building the virtual module tree.

**Supported types**:

| type | Description |
|------|------|
| rename | Renames checkpoint keys according to a pattern. |
| convert | Performs operations such as chunk (splitting fused weights) or merge (merging gate/up) on a group of source keys; must be used together with `source`, `target`, and `ops`. |

**convert field description** (when `type: convert`):

| Field | Optional/Required | Description |
|------|-----------|------|
| source | Required | List of source module path patterns to be transformed, supporting `*` wildcard. |
| target | Required | List of target module path patterns after transformation. The number of entries must match the number of split results in `ops`. |
| ops | Required | List of structural transformation operators, which perform operations on the matched source weights in order. See below for details. |

**ops operator description**:

`ops` is a list structure, where each element describes one structural transformation step. During the preprocessing phase, only the keys and metadata in the checkpoint are rewritten, and **large tensors are not materialized**; the actual slicing/merging is completed when the subsequent virtual modules are loaded.

| Operator type | Description | Typical scenario |
|-----------|------|----------|
| `chunk` | Splits fused weights into multiple logical sub-weights along a specified dimension | `gate_up_proj` → `gate_proj` + `up_proj` |
| `merge` | Merges multiple logical sub-weights into fused weights | The inverse operation of `chunk` |

The fields supported by each operator are as follows:

| Field | Optional/Required | Applicable Operator | Default Value | Description |
|------|-----------|----------|--------|------|
| type | Required | All | — | Operator type, which can be `chunk` or `merge`. |
| dim | Optional | `chunk` / `merge` | `1` for `chunk`, `0` for `merge` | The tensor dimension along which splitting or merging is performed. |
| projections | Optional | `chunk` | `["gate_proj", "up_proj"]` | The logical projection names of each sub-weight after splitting, which must correspond one-to-one with the entries in the `target` list. |

**Precautions**:

1. When splitting `*.mlp.experts.gate_up_proj` in an MoE model, the tool reads `num_experts` from the model `config.json`; an error is reported if it is missing.
2. Multiple `ops` entries can be configured and are executed sequentially in the order of the YAML list.
3. `chunk` is internally mapped to `split_fused_gate_up`, and `merge` is mapped to `merge_gate_up`.

**rename example**:

```yaml
spec:
  preprocess:
    - type: rename
      patterns:
        - from: "model.layers.0.mlp.gate_up_proj.weight"
          to: "model.layers.0.mlp.gate_proj.weight"
```

**convert (chunk) example**:

Split the fused `gate_up_proj` in an MoE model into per-expert `gate_proj` and `up_proj`:

```yaml
spec:
  preprocess:
    - type: convert
      source:
        - "model.layers.*.mlp.experts.gate_up_proj"
      target:
        - "model.layers.*.mlp.experts.*.gate_proj.weight"
        - "model.layers.*.mlp.experts.*.up_proj.weight"
      ops:
        - type: chunk
          dim: 1
          projections: ["gate_proj", "up_proj"]
```

**convert (merge) example**:

Merge the independent `gate_proj`/`up_proj` back into the fused `gate_up_proj` (the inverse operation of chunk):

```yaml
spec:
  preprocess:
    - type: convert
      source:
        - "model.layers.*.mlp.experts.*.gate_proj.weight"
        - "model.layers.*.mlp.experts.*.up_proj.weight"
      target:
        - "model.layers.*.mlp.experts.gate_up_proj"
      ops:
        - type: merge
          dim: 0
```

#### 4.2.5 Complete Configuration Example

The following example is equivalent to [qwen3_8b_fp8_to_mxfp8.yaml](./qwen3_8b_fp8_to_mxfp8.yaml) and shows the complete spec for FP8 → MXFP8:

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
    workers: 8
```

## 5. Appendix

### 5.1 Related Materials

- Overall one-click quantization process and conventional quantization configuration: *[One-click Quantization Complete Guide](../usage.md)*
- AscendV1 quantization weight file description: *[One-click Quantization Results](../quantization_result.md)*
- Format support matrix: *[Format Support Matrix](../../../quantization_formats/README.md)*

### 5.2 FAQs

#### 5.2.1 Q1: How Do I Choose Between Weight Conversion and One-Click Quantization?

- If you already have **FP8 / BF16 or other checkpoints** and only need to change the precision or on-disk format, **without recalibration**, use weight conversion (this document).
- If you start from the **original floating-point model** and need a calibration set to collect activation statistics and perform complete quantization such as W8A8, use *One-Click Quantization Guide*.

#### 5.2.2 Q2: Why Must Mxfp8 Be Saved to Disk Using ascend_v1?

W8A8_MXFP8 weights are designed for the Ascend NPU inference stack, and their metadata and packing method are incompatible with HF `compressed_tensors`. Only BF16 floating-point export (such as FP8 dequantization) should use `huggingface` for saving.

#### 5.2.3 Q3: How Do I Determine Which Layers to Write in linears.match?

1. Check the `model.safetensors.index.json` of the source checkpoint or the key list of a single file.
2. Configure match only for the **Linear weights** whose IR needs to be changed; Norm, Embedding, Head, and the like are usually excluded and are automatically passed through.
3. You can refer to the example YAML in the documentation directory of the same model family, and adjust the wildcards according to the layer name prefixes and projection names.

#### 5.2.4 Q4: What Is an Appropriate Value for Workers?

- For small models (such as 8B dense): `workers: 4~8` is usually sufficient to fully utilize the CPU.
- For very large models or MoE: you can increase `workers` appropriately.
- The calibration set and NPU are irrelevant to this process; convert is pure offline weight computation.

#### 5.2.5 Q5: When Does Route Need to Be Explicitly Specified?

In most scenarios, `route: auto` is sufficient. An explicit path is needed only when you want to **force an intermediate IR** (for example, to debug a single step `FP8_BLOCK → FLOAT`):

```yaml
route:
  - FP8_BLOCK
  - FLOAT
  - W8A8_MXFP8
```

Each step of an explicit route must have a corresponding conversion edge on the IR graph registered by the tool; otherwise, an error is reported.
