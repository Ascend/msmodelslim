# Quantization-Sensitive Layer Analysis Tool Usage Guide

<!-- md-trans-meta sourceCommit=3b90bb107b98d64c7bdc9bb6de93e6cf3e430cc1 translatedAt=2026-08-19T03:15:45.027Z pushedAt=2026-08-19T03:34:34.717Z -->

## 1. Overview

`analyze` is the quantization-sensitive layer analysis function interface in the msModelSlim tool. It is used to analyze the quantization sensitivity of each layer in a model, helping users identify quantization-sensitive layers for targeted optimization.

The following figure shows the end-to-end quantization workflow, where **sensitivity layer analysis** (`msmodelslim analyze`) belongs to the **solution design** stage: when writing or iterating the quantization YAML, layer/structure sensitivity sorting is obtained based on the calibration dataset, and is used to determine which layers to fall back—i.e., which layers should be quantized less or not at all. After analysis, the log may output a ready-to-copy YAML snippet. For the format, see [Output Description](#37-output-description).

```text
                                                YAML                    Weights
    Floating-point weights─────▶ Scheme design ─────▶ Model quantization ─────▶ Evaluation ────▶ Quantized weights
                                  ▲                                                       │
                                  └──────── Accuracy and performance feedback ────────────┘
```

**Common practices for modifying the YAML based on sensitivity layer analysis results**:

- **Fall back to floating point**: For the top-ordered layers, use **`exclude`** (or narrow **`include`**) in the quantization YAML so that these layers **do not participate** in the current quantization and still execute along the **floating-point path** during inference. If grouped modules such as QKV are involved, it is recommended to apply a consistent strategy to the same group to avoid accuracy discontinuity.

- **Increase bit width under low-bit settings**: In an overall **low-bit scheme such as W4/A4**, you do not need to raise the bit width of the entire network for layers with **high sensitivity** in the analysis result. Instead, you can **add or override configurations only for these layers** in `spec.process`, raising the **4-bit to 8-bit** for that segment while keeping the remaining layers at low bit width to achieve local precision replacement.

- **Iterate with evaluation**: When the evaluation accuracy does not meet the target, select the layers to fall back or increase bit width first based on the accuracy gap and the sensitive layer sorting. You can first **fall back a larger batch of layers** (or adopt a more conservative YAML strategy) to quickly meet the accuracy target; after the target is met, gradually **reduce the number of fallback layers** and converge incrementally between accuracy and compression.

After the sensitive layer analysis is complete, the console outputs the sensitive layer sorting result in a format compliant with the quantization YAML configuration. You can directly copy and paste it into the include or exclude of the quantization configuration (usually used for the Processor of `type: linear_quant`). Check whether the layer name wildcards cover the expected range, and then perform quantization with the new configuration.

## 2. Prerequisites

Install the msModelSlim tool. For details, see [msModelSlim Tool Installation Guide](../../../install_guide/install_guide.md).

## 3. Feature Introduction

### 3.1 Feature Description

- **Multi-dimensional analysis** can accurately evaluate layer sensitivity from multiple dimensions such as data distribution, robustness, kurtosis characteristics, attention, and hierarchical output differences. The analysis hierarchies are as follows:

  - **linear** measurement algorithms: `std`, `quantile`, `kurtosis`

  - **attention** measurement algorithm: `mse`

  - **layer** measurement algorithms: `mse_layer_wise`, `mse_model_wise`

- **Flexible configuration**: Supports custom calibration datasets (JSON/JSONL format), layer name matching, and a rich set of parameter options to meet quantization requirements in different scenarios.
- **Intelligent output**: Supports printing the Top K sensitive layer list. The actual number of printed layers may be greater than or equal to the target number, for example, QKV are printed together.

### 3.2 Precautions

- The transformers version depends on the model and is unrelated to the quantization feature.
- The actual number of fallback layers is limited by the inference engine implementation, so it may differ slightly from the `topk` parameter setting.
- The default value of `topk` is `15`, which serves only as an empirical fallback value for reference. If the printed layers involve QKV, QKV will be output together.
- Due to security specifications, `trust_remote_code` defaults to `False`.
- Sensitivity layer analysis currently supports only large language models.

### 3.3 Command Format

```bash
msmodelslim analyze [scope] [option]
```

Where `scope` is used to specify the analysis scope (linear/attention/layer level, etc.).

Values:

- `linear`: linear layer sensitivity analysis, whose output result is the linear layer sorting result.
- `attn`: attention structure sensitivity analysis, whose output result is the attention layer sorting result.
- `layer`: sensitivity analysis of the entire Decoder hierarchy, where the output result is the sorting result of the entire layer.

> [!note] Compatibility Description
>
> - **The old command is still available but will be sunset in the future**. The old syntax `msmodelslim analyze --model_type ... --model_path ...` (without explicitly specifying scope) is still supported, but under the old syntax `--metrics` supports only four types: `"std"`, `"quantile"`, `"kurtosis"`, and `"attention_mse"`, where `"attention_mse"` corresponds to the new syntax `msmodelslim analyze attn --metrics mse`. Under the new syntax, the optional value of `--metrics` for the attn scope is `"mse"`, and the name `"attention_mse"` is no longer retained.
> - **The new syntax is recommended**. Explicitly specifying the scope (`linear`/`attn`/`layer`) provides clearer help and more stable parameter semantics.

### 3.4 Parameter Description

#### 3.4.1 Common Parameters (Shared by All Scopes)

| Name | Type | Default Value | Description | Example Value |
|------|------|--------|------|--------|
| `--model_type` | `str` | - | Model type, used to specify the model architecture to be analyzed | `Qwen2.5-7B-Instruct` |
| `--model_path` | `str` | - | Path to the original model. An absolute path is recommended. | `/path/Qwen2.5-7B-Instruct` |
| `--device` | `str` | `npu` | The target device for running the analysis. Optional values: `npu`, `cpu`. | `npu` |
| `--calib_dataset` | `str` | `"mix_calib.jsonl"` | Path to the calibration dataset file. JSON/JSONL formats are supported, ending with .json or .jsonl. Both absolute and relative paths are supported. | `/path/data.jsonl` |
| `--topk` | `int` | `15` | Number of `TopK`-sensitive layers to output, which is an integer greater than 0. The recommended range is 10 to 20. | `15` |
| `--trust_remote_code` | `bool` | `False` | Whether to trust remote code. Users need to ensure security by themselves. Optional values: `True`, `False`. | `False` |
| `-h, --help` | - | - | Help information for command-line parameters | - |

#### 3.4.2 `linear` Parameters (Linear Layer Analysis)

Command Format:

```bash
msmodelslim analyze linear [general_options] [linear_options]
```

| Name | Type | Default Value | Description | Example Value |
|------|------|--------|------|--------|
| `--pattern` | `List[str]` | `["*"]` | List of layer names to be analyzed, supporting wildcard matching. Multiple patterns can be set, separated by spaces. | `"*down_proj"` `"*up_proj"` |
| `--metrics` | `str` | `"kurtosis"` | Measurement algorithm used for analysis. Optional values: `"std"`, `"quantile"`, `"kurtosis"` | `"kurtosis"` |

#### 3.4.3 `attn` Parameters (attention structure analysis)

Command Format:

```bash
msmodelslim analyze attn [general_options] [attn_options]
```

| Name | Type | Default Value | Description | Example Value |
|------|------|--------|------|--------|
| `--metrics` | `str` | `"mse"` | Measurement algorithm used for analysis. Optional Value: `"mse"` | `"mse"` |

#### 3.4.4 `layer` Parameters (decoder Hierarchical Output)

Command Format:

```bash
msmodelslim analyze layer [general_options] [layer_options]
```

| Name | Type | Default Value | Description | Example Value |
|------|------|--------|------|--------|
| `--quant_modules` | `List[str]` | `["*"]` | Target module list, supporting wildcard matching, used to specify the module range participating in the comparison. | `"*self_attn*"` `"*mlp*"` |
| `--metrics` | `str` | `"mse_layer_wise"` | Measurement algorithm used for analysis. Optional values: `"mse_model_wise"`, `"mse_layer_wise"` | `"mse_layer_wise"` |

#### 3.4.5 Parameter Selection Precautions

**model_type Support Description**

- For most sensitive layer analysis metrics (metrics under `linear`/`layer`), the optional `model_type` is consistent with ModelslimV1 quantization.
- `attn --metrics mse` is not within the above scope. It requires the model adapter of the corresponding `model_type` to implement **AttentionMSEAnalysisInterface** before it can be used. For the list of models supported by this algorithm, see [Attention MSE Algorithm Applicable Requirements](../../quantization_algorithms/sensitive_layer_analysis/attention_mse.md#applicable-requirements).

**Model Path and Dataset Requirements**

- **`model_path`**: Must be a real absolute or relative path that contains valid weights and configuration.
- **`calib_dataset`**: Must be in `.json`/`.jsonl` format. JSON is a list of strings, where each item represents a piece of calibration text, while JSONL contains one JSON object per line. You can directly use the calibration dataset under the [`lab_calib`](../../../../../lab_calib/) sample calibration dataset directory provided by the tool. If the dataset uses a relative path, the resolution rules are as follows: it is first searched under the command startup path and used directly if found; if not found, a dataset with the same name is matched under the sample calibration dataset directory; if neither is found, an exception is thrown.

**Hierarchy Selection**

- `linear`: Performs fine-grained sensitivity analysis on individual linear layers and supports precise fallback for each layer. It is not recommended for expert layers in MoE models, as the large number of linear layers and the fact that not all participate in activation make the analysis results of limited value.
- `attn`: Performs sensitivity analysis on attention modules. It is currently mostly used in scenarios that need to work with Flash Attention 3 activation quantization to help identify attention layers that require fallback.
- `layer`: Performs overall sensitivity analysis on Decoder blocks and outputs block-granularity sorting results, which is used for scenarios that require fallback of an entire layer or block (Attention/MoE/MLP).

For the selection of different algorithms (metrics) at each hierarchy, see [Analysis Algorithm Description](#35-analysis-algorithm-description) below.

### 3.5 Analysis Algorithm Description

Sensitivity measurement is divided into three categories by analysis scope and output granularity. For detailed descriptions of `--metrics` under each category, see the following subsections.

#### 3.5.1 Linear

Performs sensitivity analysis on **individual linear layers** (and convolution layers supported by the implementation) in the model, with the result being **linear-layer-granularity** sorting. The optional `--metrics` values include `std`, `quantile`, and `kurtosis`; none of them requires the model adapter to additionally implement analysis interfaces. For detailed descriptions of each algorithm, see:

- [Std: Sensitive Layer Analysis Algorithm Description](../../quantization_algorithms/sensitive_layer_analysis/std.md)
- [Quantile: Sensitive Layer Analysis Algorithm Description](../../quantization_algorithms/sensitive_layer_analysis/quantile.md)
- [Kurtosis: Sensitive Layer Analysis Algorithm Description](../../quantization_algorithms/sensitive_layer_analysis/kurtosis.md)

> **Recommendation**: For `linear`, `kurtosis` is the preferred choice. This metric is sensitive to activation spikes and can effectively identify the impact of extreme values on quantization. If the data contains many outliers, `quantile` can be used in combination. If you focus on the range-to-dispersion ratio, `std` can be used in combination.

#### 3.5.2 attn (Attention Structure)

Perform sensitivity analysis on the **attention structure** in the model, with the result being sorting at the **attention module granularity**. The optional `--metrics` value is `mse`. The model adapter corresponding to `model_type` must implement **AttentionMSEAnalysisInterface**. For detailed descriptions of each algorithm, see:

- [Attention MSE (mse): Sensitive Layer Analysis Algorithm Description](../../quantization_algorithms/sensitive_layer_analysis/attention_mse.md)

#### 3.5.3 layer (Decoder Hierarchical Output)

Performs sensitivity analysis on the **Decoder block**, with the result sorted at **hierarchical granularity**, used for whole-layer fallback or whole-block fallback (such as the entire attention/MLP block). The optional `--metrics` values include `mse_layer_wise` and `mse_model_wise`; neither requires the model adapter to additionally implement an analysis interface. For detailed descriptions of each algorithm, see:

- [Hierarchical MSE (mse_layer_wise): Sensitive Layer Analysis Algorithm Description](../../quantization_algorithms/sensitive_layer_analysis/mse_layer_wise.md)
- [Model-level MSE (mse_model_wise): Sensitive Layer Analysis Algorithm Description](../../quantization_algorithms/sensitive_layer_analysis/mse_model_wise.md)

> **Recommendation**: For `layer`, `mse_layer_wise` is the preferred choice. `mse_model_wise` relies on chained forward propagation to pass the output of the previous layer into the next layer to simulate the real inference path. On some architectures, subsequent layers may be skipped because tensor shapes cannot be aligned, and the calibration scale and intermediate cache also increase memory stress.

### 3.6 Analysis Example

In the following examples, `${model_path}` indicates the original model path, and `${calib_dataset}` indicates the calibration dataset file path. Replace them with actual values as needed.

Linear analysis example:

```bash
msmodelslim analyze linear \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics kurtosis \
    --calib_dataset ${calib_dataset} \
    --pattern "*.down_proj*" "*.o_proj*" \
    --topk 15 \
    --device npu
```

Attn analysis example:

```bash
msmodelslim analyze attn \
    --model_type DeepSeek-V3 \
    --model_path ${model_path} \
    --metrics mse \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

Layer analysis example:

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --metrics mse_layer_wise \
    --quant_modules "*mlp*" \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

### 3.7 Output Description

After the analysis is run, the console outputs two key pieces of information: a **sensitive layer sorting list** (with `Score` from high to low, where a higher score indicates that the layer is more sensitive to quantization) and a **YAML configuration snippet that can be pasted directly**. In the YAML snippet, the fallback layer list has been generated in sensitivity order, and users can copy it directly into the quantization configuration file for use.

Depending on the scope, the output format is divided into the following three categories.

> [!NOTE] NOTE
>
>
> - When performing linear hierarchy analysis, for grouped modules such as QKV (`q_proj`, `k_proj`, `v_proj`), the same group has the same score and is listed together in the output.
> - Certain grouped modules must be excluded at the same time, such as `up_proj` and `gate_proj` in MLP. If only one of them is rolled back, the model may fail to be deployed.

#### 3.7.1 Linear Output

The output results are sorted by layer granularity, with each row containing the specific layer name and the corresponding sensitivity score:

```text
=== Layer Analysis Results (std method) ===
Patterns analyzed: ['*']
Total layers analyzed: 252
Layer Sensitivity Scores (higher score = more sensitive to quantization):
--------------------------------------------------------------------------------
  1. model.layers.6.mlp.down_proj                       | Score:   2.1326e+02
  2. model.layers.14.mlp.down_proj                      | Score:   1.6952e+02
  3. model.layers.3.mlp.gate_proj                       | Score:   1.5288e+02
  ...
--------------------------------------------------------------------------------
Top 80 most sensitive layers selected for disable_names

=== YAML Format for quantization ===

top 80:
  - 'model.layers.6.mlp.down_proj'
  - 'model.layers.14.mlp.down_proj'
  - 'model.layers.3.mlp.gate_proj'
  ...

=== End of YAML Format ===
```

#### 3.7.2 Attn Output

The output result is sorted by attention module granularity, with each line containing the attention module name and the corresponding sensitivity score:

```text
=== Layer Analysis Results (mse method) ===
Patterns analyzed: ['*']
Total layers analyzed: 36
Layer Sensitivity Scores (higher score = more sensitive to quantization):
--------------------------------------------------------------------------------
  1. model.layers.33.self_attn                          | Score:   2.0504e+00
  2. model.layers.24.self_attn                          | Score:   4.3414e-01
  3. model.layers.31.self_attn                          | Score:   4.2244e-01
  ...
--------------------------------------------------------------------------------
Top 36 most sensitive layers selected for disable_names

=== YAML Format for quantization ===

top 36:
  - 'model.layers.33.self_attn'
  - 'model.layers.24.self_attn'
  - 'model.layers.31.self_attn'
  ...

=== End of YAML Format ===
```

#### 3.7.3 Layer Output

The output result is sorted by Decoder block granularity. In the YAML, fallback layer names contain wildcards by default (for example, `model.layers.2.*`), indicating that all submodules within that Decoder layer are to be fallen back. If you only need to fall back a specific structure within that layer (for example, only MLP or only Attention), you can replace the wildcard with the specific module name in the YAML, or use `include` in the quantization configuration to specify the range of submodules to be quantized.

```text
=== Layer Analysis Results (mse_layer_wise method) ===
Patterns analyzed: ['*']
Total layers analyzed: 36
Layer Sensitivity Scores (higher score = more sensitive to quantization):
--------------------------------------------------------------------------------
  1. model.layers.2                                     | Score:   2.4396e+03
  2. model.layers.35                                    | Score:   1.3626e+02
  3. model.layers.34                                    | Score:   1.2008e+02
  ...
--------------------------------------------------------------------------------
Top 36 most sensitive layers selected for disable_names

=== YAML Format for quantization ===

top 36:
  - 'model.layers.2.*'
  - 'model.layers.35.*'
  - 'model.layers.34.*'
  ...

=== End of YAML Format ===
```

## 4. FAQs

### 4.1 Symptom: The calibration dataset file has an incorrect format or cannot be read

**Solution**:

1. Confirm that the file format is a supported JSON or JSONL format.
2. Ensure that each record contains the required fields.
3. Verify that the file path is correct.
4. Confirm that the calibration dataset file is readable.

### 4.2 Symptom: What happens when an unsupported model_type is entered

**Solution**:
When the entered model_type is not in the supported list:

- The system prints a warning log, prompting that the default model is used.
- The default model is automatically used for processing.
- The optimal analysis result may not be achieved.

- **Recommendation**: Use a standard `model_type` that is consistent with the selected `--metrics` and the conventions in [Parameter Selection Precautions](#345-parameter-selection-precautions) to achieve optimal compatibility and analysis results.
