# msmodelslim analyze Command-Line API Document

## 1. Function Description

`msmodelslim analyze` performs sensitive layer analysis on the model before quantization, listing the layer names from the highest sensitivity score to the lowest and providing guidance for configuring quantization fallback layers. By analysis granularity, it is divided into four scopes: `linear` (individual linear layers), `layer` (grouped by layer/block), `attn` (Attention modules), and `attn_head` (Attention Head / `ra_compress`). The analysis computes sensitivity metrics on the target device and outputs the names of the most sensitive layers according to `--top_k`; `attn_head` outputs `head.pt`. By default the results are only printed to the log; once `--save_path` is specified, the results are written to files (`linear`/`layer`/`attn` as YAML, `attn_head` as `head.pt`). Among these, `--top_k` and `--calibration_dataset` are provided only by `linear`/`layer`/`attn`; `attn_head` (`ra_compress`) neither outputs `disable_names` nor provides these two parameters, and its calibration input is constructed by the algorithm itself at runtime (the first token plus a random repeated segment), without reading an external calibration set.

Command boundaries: this command only performs analysis and outputs results; it does not quantize, and it requires the model adapter to implement the analysis interface. Single-device and multi-device (DP) analysis are supported; for multi-device runs, use `--device npu --device_id 0 1 ...` (or the compatible form `--device npu:0,1,...`), executed by `DPLayerWiseRunner`. For operational steps, see *[Sensitive Linear Layer Analysis Usage Guide](../../user_guide/usage_sensitive_linear_analysis.md)*, *[Layer-wise Sensitive Layer Analysis Usage Guide](../../user_guide/usage_sensitive_layer_analysis.md)*, *[Attention Sensitive Layer Analysis Usage Guide](../../user_guide/usage_sensitive_attn_analysis.md)*, and *[Attention Head Screening and Analysis Usage Guide](../../user_guide/usage_sensitive_attn_head_analysis.md)*.

## 2. Command Format

```text
msmodelslim analyze linear --model_type <model_type> --model_path <model_path> [--device <device>] [--device_id <id> ...] [--calibration_dataset <file>] [--top_k <n>] [--trust_remote_code [<BOOL>]] [--log_level <level>] [-v] [-q] [--metrics <metrics>] [--patterns <pattern> ...] [--save_path <path>]

msmodelslim analyze layer --model_type <model_type> --model_path <model_path> [--device <device>] [--device_id <id> ...] [--calibration_dataset <file>] [--top_k <n>] [--trust_remote_code [<BOOL>]] [--log_level <level>] [-v] [-q] [--metrics <metrics>] [--quant_modules <module> ...] [--save_path <path>]

msmodelslim analyze attn --model_type <model_type> --model_path <model_path> [--device <device>] [--device_id <id> ...] [--calibration_dataset <file>] [--top_k <n>] [--trust_remote_code [<BOOL>]] [--log_level <level>] [-v] [-q] [--metrics <metrics>] [--save_path <path>]

msmodelslim analyze attn_head --model_type <model_type> --model_path <model_path> [--device <device>] [--device_id <id> ...] [--trust_remote_code [<BOOL>]] [--log_level <level>] [-v] [-q] [--metrics <metrics>] [--save_path <path>]
```

Symbols:

- `<scope>` (`linear`/`layer`/`attn`/`attn_head`) is a positional parameter and is also the subcommand name.
- Values in angle brackets must be replaced; parameters in square brackets are optional.
- `--trust_remote_code` is a Boolean parameter that takes an optional value: passing it without a value enables it (equivalent to `true`), and an explicit `true`/`false` may also be passed (case-insensitive; the legacy `True`/`False` spellings are also accepted).
- `--patterns`, `--quant_modules`, and `--device_id` accept one or more values; `...` means that several space-separated values may follow.
- Each scope has different specific options: `linear` has `--metrics` and `--patterns`; `layer` has `--metrics` and `--quant_modules`; `attn` has only `--metrics`; `attn_head` has only `--metrics` (`ra_compress` only). `--save_path` is a common parameter, and what it writes differs by scope: `linear`/`layer`/`attn` output an analysis result YAML, while `attn_head` outputs `head.pt`.
- `--top_k` and `--calibration_dataset` are provided only by `linear`/`layer`/`attn`: `attn_head` does not output `disable_names`, and its calibration input is constructed by the algorithm itself at runtime, so this scope does not provide these two parameters (passing them explicitly raises `unrecognized arguments`, and they are not listed in the help information either).
- When `<scope>` is omitted (and this is not a help request), execution automatically proceeds as `linear`; `msmodelslim analyze` has no other positional parameters. This syntax summary describes the parameter structure and is not a copy-ready command; for copy-ready commands, see [Usage Examples](#5-usage-examples).

## 3. Parameter List

Common parameters (shared by all scopes):

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--model_type` | None | `string` | Single value | Required | None | Model type name, such as `Qwen2.5-7B-Instruct` or `Qwen3-32B` | Type of the model to analyze, which must match a name in the *[Foundation Model Support Matrix](../../knowledge_base/model/README.md)*; when no value is passed, or when the passed name does not exist, execution falls back to the default adapter `default`. |
| `--model_path` | None | `string` | Single value | Required | None | Original model weight directory (must exist and be readable) | Weight directory of the model to analyze. |
| `--device` | None | `string` | Single value | Optional | `npu` | `npu`, `cpu`; the legacy spelling `npu:0,1,...` is also accepted | Device type to run on. The recommended form is `--device npu` together with `--device_id`; the legacy `npu:0,1` is automatically split into `--device npu --device_id 0 1`. |
| `--device_id` | None | `list[int]` | One or more values, space-separated | Optional | None | Non-negative integer indices, such as `0 1 2 3` | NPU indices used for the analysis; when the length is > 1, multi-device DP is enabled (`DPLayerWiseRunner`). Multiple indices cannot be specified in the CPU scenario. |
| `--save_path` | None | `string` | Single value | Optional | None (printed to the log only) | A writable directory or file path: for `linear`/`layer`/`attn` a `.yaml`/`.yml` file or a directory; for `attn_head` a `.pt` file or a directory | Path to which the analysis results are written. For `linear`/`layer`/`attn`: when the path ends with `.yaml`/`.yml`, it is written to that file name, otherwise it is treated as a directory and the file is named `<model_type>-<metric>.yaml`; for `attn_head`: when the path ends with `.pt`, it is written to that file name, otherwise `head.pt` is written into the directory. When it is not specified, the results are only printed to the log (console). |
| `--trust_remote_code` | None | `bool` | Boolean parameter: passing it without a value enables it; an explicit `true`/`false` value may also be given | Optional | `false` | `true`/`false` (case-insensitive; the legacy `True`/`False` spellings are also accepted) | Whether to trust and execute custom Python code in the model directory; enable it only when the code source is confirmed trustworthy. |
| `--log_level` | None | `string` | Single value | Optional | `info` | `debug`, `info`, `warning`, `error` | Log level. |
| `-v` / `--verbose` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Increase the level of output detail (equivalent to `--log_level debug`). |
| `-q` / `--quiet` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Suppress non-error output (equivalent to `--log_level error`). |
| `-h` / `--help` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Displays the help information and exits. Used after `analyze`, it displays the scope list; used after `<scope>`, it displays the parameter description of that scope. |
| `-V` / `--version` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Displays the version information and exits (a top-level global parameter that can be used after any subcommand, for example, `msmodelslim analyze --version`). |

`linear`/`layer`/`attn` common parameters (not provided by `attn_head`):

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--calibration_dataset` | `--calib_dataset` (legacy alias) | `string` | Single value | Optional | `mix_calib.jsonl` | For LLMs: `.json` / `.jsonl`; for VLMs: a text-and-image directory name (such as `calibImages`) or a path | Calibration dataset, used for metric computation in `linear`/`layer`/`attn`. |
| `--top_k` | `--topk` (legacy alias) | `int` | Single value | Optional | `15` | An integer greater than 0 | Number of the most sensitive layers written to `disable_names` (an empirical value, for reference only). |

Positional parameters:

| Positional Parameter | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|----------|------|----------|-----------|--------|----------------|------|
| `scope` | `string` | Positional parameter (subcommand) | Optional | `linear` is injected when it is not specified and this is not a help request | `linear`, `layer`, `attn`, `attn_head` | Analysis granularity. When omitted, it defaults to `linear`; when help is requested (`-h`/`--help`), nothing is injected. |

`linear`-specific parameters:

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--metrics` | None | `string` | Single value | Optional | `kurtosis` | `std`, `quantile`, `kurtosis` | Linear layer sensitivity metric: `std` (standard deviation), `quantile` (quantile), `kurtosis` (kurtosis). |
| `--patterns` | `--pattern` (legacy alias) | `list` | One or more values, space-separated | Optional | `['*']` | Module name list, wildcards supported; `*` means all | Filters the linear layers to display; `*` means all. |

`layer`-specific parameters:

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--metrics` | None | `string` | Single value | Optional | `mse_layer_wise` | `mse_model_wise`, `mse_layer_wise` | Layer-wise sensitivity metric: compute the error by layer, or across the model as a whole. |
| `--quant_modules` | None | `list` | One or more values, space-separated | Optional | `['*']` | Module name list, wildcards supported; `*` means all | Modules brought into the quantization range of the quantization pipeline in layer-wise analysis; `*` means all. |

`attn`-specific parameters:

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--metrics` | None | `string` | Single value | Optional | `mse` | `mse` | Attention module sensitivity metric; only `mse` is supported, and it applies to all Attention modules. |

`attn_head`-specific parameters:

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--metrics` | None | `string` | Single value | Optional | `ra_compress` | `ra_compress` | Attention Head screening metric; only `ra_compress` is supported, and it applies to LLMs only, not to multimodal understanding or multimodal generation. |

## 4. Parameter Relationships

- `scope` determines the available `--metrics` values and the specific parameters; mixing across scopes (for example, `analyze attn --patterns`) causes argument parsing to fail, reporting an unsupported parameter or value.
- When `scope` is omitted, execution defaults to `linear`; with `msmodelslim analyze -h`/`--help`, no scope is injected and the scope help is displayed.
- When the name passed to `--model_type` does not exist, execution falls back to the default adapter `default`.
- `--device` accepts `npu`/`cpu`; multi-device analysis uses `--device npu --device_id 0 1 ...` (a length > 1 enables `DPLayerWiseRunner`). The legacy spelling `--device npu:0,1` is accepted and is automatically converted to the recommended form above.
- `--save_path` is a common parameter shared by all scopes: `linear`/`layer`/`attn` output an analysis result YAML, and when a directory is passed, the file `<model type>-<metric>.yaml` is generated in that directory; `attn_head` outputs `head.pt`, and when a directory is passed, `head.pt` is generated. When it is not specified, the results are written only to the console.
- The historical usage `--metrics attention_mse` (when scope is omitted) is converted to `analyze attn --metrics mse` and `--patterns` is discarded; this behavior exists only for backward compatibility, the command gives a deprecation notice, and it is not recommended as a formal usage.
- `--top_k` must be an integer greater than 0; `--calibration_dataset`: for LLMs it is `.json`/`.jsonl`, and for VLMs it may be a text-and-image directory name (such as `calibImages`). Both are provided only by `linear`/`layer`/`attn`; `attn_head` has neither parameter (passing one raises `unrecognized arguments`, and neither is listed in the help information).
- `--save_path` is available for all four scopes, and what is written differs by scope: `linear`/`layer`/`attn` output a quantization YAML (`top <n>:` plus a `disable_names` list; for the `layer` scope, the whole-block fallback name is `<block>.*`), and when a directory is specified the file name is `<model_type>-<metric>.yaml`; `attn_head` outputs `head.pt` (containing the two dictionaries `prefix_matching` and `copying`), delivered for use by the long-sequence KV Cache compression capability of the subsequent MindIE inference framework, and it is not written into the quantization configuration. A directory that does not exist is created automatically; when this parameter is not specified, the results are only printed to the log.
- `attn_head` / `ra_compress` supports LLMs only; it does not apply to multimodal understanding or multimodal generation models. This scenario does not provide `--top_k` or `--calibration_dataset`, and the calibration input is constructed by the algorithm itself at runtime; for details, see *[RA Compress Usage Guide](../../knowledge_base/quantization_algorithms/ra_compress/usage_ra_compress.md)*.

## 5. Usage Examples

### 5.1 Minimal Runnable Scenario (Linear Layer Sensitivity Analysis)

The scope is omitted, so execution proceeds as the default `linear`:

```bash
msmodelslim analyze \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH}
```

`${MODEL_TYPE}` is the model type name, and `${MODEL_PATH}` is the weight directory. When `scope` is omitted, the analysis proceeds as the default `linear`, using the default metric `kurtosis`, the default `--top_k 15`, and the default calibration set `mix_calib.jsonl`; it outputs the 15 most sensitive layer names, which can be written into the `exclude` of `linear_quant` in the quantization YAML.

### 5.2 Scope, the Metric, and topk

```bash
msmodelslim analyze linear \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --metrics kurtosis \
  --top_k 15
```

The `linear` scope is specified explicitly, the kurtosis metric is used, and the 15 most sensitive layer names are output according to `--top_k 15`.

### 5.3 Attention MSE Analysis

```bash
msmodelslim analyze attn \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --metrics mse
```

Computes the MSE sensitivity for all Attention modules.

### 5.4 Sensitive Layer Analysis on Specified Modules

```bash
msmodelslim analyze layer \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --metrics mse_layer_wise \
  --quant_modules ${MODULE_1} ${MODULE_2}
```

`${MODULE_1}` and `${MODULE_2}` are the names of modules brought into the quantization range of the quantization pipeline (wildcards are supported; separate several values with spaces).

### 5.5 Multi-Device DP Sensitive Layer Analysis

```bash
msmodelslim analyze layer \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --metrics mse_layer_wise \
  --device npu \
  --device_id 0 1 2 3 \
  --calibration_dataset mix_calib.jsonl
```

When `--device_id` specifies several NPU indices, `DPLayerWiseRunner` is enabled for multi-device sensitive layer analysis. Compatible form: `--device npu:0,1,2,3` (automatically split into `--device npu --device_id 0 1 2 3`).

### 5.6 Attention Head Analysis

```bash
msmodelslim analyze attn_head \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --metrics ra_compress \
  --device npu \
  --save_path ./head_result
```

Applies to LLMs only; multimodal understanding and multimodal generation models are not supported. This scenario does not support `--calibration_dataset` or `--top_k`: the algorithm constructs a controlled random calibration input itself at runtime (the first token plus a random repeated segment), and an external calibration set takes no part in the computation. The resulting `head.pt` is delivered to the MindIE inference framework for long-sequence KV Cache compression: users can complete long-sequence compression on this basis, and the compressed model can be used for subsequent inference deployment; for the specific list of adapted quantization models, see the "Models Supported by MindIE" section of *[What Is MindIE](https://www.hiascend.com/document/detail/en/mindie/10RC3/whatismindie/mindie_what_0001.html)*. This artifact is not written into the quantization configuration. For the calibration input conventions, see *[RA Compress Attention Head Screening and Analysis Usage Guide](../../knowledge_base/quantization_algorithms/ra_compress/usage_ra_compress.md)*.

## 6. Exit Codes and Exception Handling

Normal operation raises no exception; on failure, an exception is thrown. Common causes: a mismatch between the scope and the `--metrics` value, an invalid `--device`/`--device_id` value, a calibration set whose format does not conform, a `--top_k` that is not an integer greater than 0, a model adapter that does not implement the analysis interface, a multimodal generation model being rejected, and `ra_compress` used on a non-LLM.

## 7. Security Notes

- `--trust_remote_code true` executes custom Python code in the model directory; enable it only when the source is confirmed trustworthy.
