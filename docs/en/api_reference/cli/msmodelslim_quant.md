# msmodelslim quant Command-Line API Document

## 1. Function Description

`msmodelslim quant` is the one-click quantization command. It loads the original model weights, performs weight and activation quantization, and exports deployable quantized weights and a description file. There are two sources of configuration: use `--quant_type` to automatically match a best-practice YAML in `lab_practice` by model and quantization type; or use `--config` to specify a user YAML directly (supporting `modelslim_v1`, multimodal, and the `modelslim_convert` pure weight-conversion protocols).

Command boundaries: the supported devices are `npu` and `cpu`, and multi-device runs specify the index list with `--device_id`; scenario tag matching and dumping the `--debug` debug context to disk are also supported. For operational steps such as calibration data preparation and deployment, see *[One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md)*; for the fields of the quantization task YAML, see *[Quantization Task Configuration Reference Navigation](../config/quant/README.md)*, and for the auto-tuning configuration, see *[Auto Tuning Configuration Reference Navigation](../config/tuning/README.md)*. For more related items, see [Referenced Configurations](#5-referenced-configurations).

## 2. Command Format

```text
msmodelslim quant [--model_type <model_type>] --model_path <model_path> --save_path <save_path> [--device <device>] [--device_id <id> ...] [--config <config> | --quant_type <quant_type>] [--trust_remote_code [<BOOL>]] [--debug] [--tags <tag> ...] [--log_level <level>] [-v] [-q]
```

Symbols:

- Values in angle brackets must be replaced; parameters in square brackets are optional.
- `--config` and `--quant_type` form a mutually exclusive group, denoted by `|`; the two must not be passed at the same time.
- `--trust_remote_code` is a Boolean parameter: passing it without a value enables it (equivalent to `true`), and an explicit `true`/`false` may also be passed (case-insensitive; the legacy `True`/`False` spellings are also accepted).
- `--debug` is a Boolean switch that takes no value.
- `--tags` accepts one or more values; `...` means that several space-separated values may follow.
- This command has no positional parameters; all parameters are passed as options. This syntax summary describes the parameter structure and is not a copy-ready command; for copy-ready commands, see [Usage Examples](#7-usage-examples).

## 3. Parameter List

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--model_type` | None | `string` | Single value | Conditionally required (required on the normal quantization path; may be omitted when `--config` points to a configuration with `apiversion: modelslim_convert`) | None | Model type name, such as `Qwen2.5-7B-Instruct`, `Qwen3-32B`, or `transformers` | Specifies the type of the model to quantize, used to load the corresponding model adapter and match a best practice; when the passed name does not exist, execution falls back to the default adapter `default`. `transformers` loads the generic transformers-based base-model adapter; it may be omitted only when `--config` points to a configuration with `apiversion: modelslim_convert`. |
| `--model_path` | None | `string` | Single value | Required | None | Original model weight directory (must exist and be readable) | Weight directory of the model to quantize. The weight-conversion scenario needs only `--model_path` and `--save_path`. |
| `--save_path` | None | `string` | Single value | Required | None | Output directory (must be writable) | Directory in which the quantized weights and the description file are saved. The weight-conversion scenario needs only `--model_path` and `--save_path`. |
| `--device` | None | `string` | Single value | Optional | `npu` | `npu`, `cpu` | Device type to run on; specify multi-device indices with `--device_id`. |
| `--device_id` | None | `list` | One or more values, space-separated | Optional | None | A list of non-negative integers, such as `0` or `0 1 2 3` | Device indices; used to specify multiple NPU devices. |
| `--config` | `-c`, `--config_path` (legacy alias) | `string` | Single value | Optional | None | Path to a readable YAML file | Explicitly specified quantization configuration YAML; once loaded it is adopted directly, and best-practice matching by `--quant_type` and `--tags` is ignored. Mutually exclusive with `--quant_type`; when neither `--config` nor `--quant_type` is passed, a best practice is matched by the default quantization type `w8a8`. |
| `--quant_type` | None | `string` | Single value | Optional | None (a best practice is matched by `w8a8` when neither this parameter nor `--config` is provided) | `w4a4`, `w4a8`, `w4a4c8`, `w4a4f8`, `w4a8c8`, `w8a16`, `w8a8`, `w8a8s`, `w8a8c8`, `w8a8f8`, `w4a4f4`, `w16a16s` | Quantization type, used to match a best-practice YAML by model and quantization type. Mutually exclusive with `--config`; when neither `--config` nor `--quant_type` is passed, a best practice is matched by the default quantization type `w8a8`. |
| `--trust_remote_code` | None | `bool` | Boolean parameter: passing it without a value enables it; an explicit `true`/`false` value may also be given | Optional | `false` | `true`/`false` (case-insensitive; the legacy `True`/`False` spellings are also accepted) | Whether to trust and execute custom Python code in the model directory; enable it only when the code source is confirmed trustworthy. |
| `--debug` | None | `bool` | Boolean switch without a value | Optional | `False` | Enabled when passed | Enables debug mode, writing the intermediate quantization context to `${SAVE_PATH}/debug_info/` (`debug_info.json` and `debug_info.safetensors`). |
| `--tags` | `--tag` (legacy alias) | `list` | One or more values, space-separated | Optional | None | Scenario tags, which must match tag names in the best practice's `verified_tags` (case-insensitive), such as `vLLM-Ascend`, `Atlas_A2_Inference`, and `MindIE` | Matches a best practice that carries verified scenario tags; several tags must all occur in the same scenario, and when no hardware-type tag is provided the current device type is matched automatically. |
| `--log_level` | None | `string` | Single value | Optional | `info` | `debug`, `info`, `warning`, `error` | Log level. |
| `-v` / `--verbose` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Increase the level of output detail (equivalent to `--log_level debug`). |
| `-q` / `--quiet` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Suppress non-error output (equivalent to `--log_level error`). |
| `-h` / `--help` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Displays the help information and exits (for example, `msmodelslim quant -h`). |
| `-V` / `--version` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Displays the version information and exits (a top-level global parameter that can be used after any subcommand, for example, `msmodelslim quant --version`). |

## 4. Parameter Relationships

- `--config` and `--quant_type` are mutually exclusive; passing both at the same time raises an error.
- Passing neither is a supported usage: a best practice is matched by the default quantization type `w8a8`; when no best practice is matched, a prompt is given and the command waits for confirmation (enter `Y`/`y` to continue; entering anything else exits). This prompt is interactive input, so in a non-interactive environment without standard input it waits forever; prefer `--config` or an explicit `--quant_type` instead.
- Once `--config` is specified, that configuration is adopted directly, and best-practice matching by `--quant_type` and `--tags` is ignored.
- On the normal quantization path, `--model_type` must be provided; it may be omitted only when `--config` points to a configuration with `apiversion: modelslim_convert`, in which case the run is a pure weight conversion that does not load model code, and `--model_type` and the quantization-related parameters have no effect.
- When the name passed to `--model_type` does not exist, execution falls back to the default adapter `default`.
- When `--model_type` specifies `transformers`: if `--quant_type` is used, only `w8a16`/`w8a8` are supported; if `--config` is used, currently only LLM quantization configurations (`apiversion: modelslim_v1`) are supported, and multimodal understanding (`multimodal_vlm_modelslim_v1`) and multimodal generation (`multimodal_sd_modelslim_v1`) configurations are rejected.
- When several values are specified for `--tags`, they must all occur in the same verified scenario; when no hardware-type tag is provided, the current device type is matched automatically.
- Once `--debug` is enabled, the quantization context is written to `${SAVE_PATH}/debug_info/`.

**Best-Practice Match Search Priority** (the search priority, from highest to lowest, when matching a best-practice YAML through `--quant_type`):

1. Best-practice policy YAML with the model-specified quantization method plus the specified scenario tags (`--tags`).
2. Best-practice policy YAML with the model-specified quantization method and scenario tags ignored; the user is asked whether to adopt it.
3. Default-practice policy YAML with the model-specified quantization method and scenario tags ignored (a default-practice policy does not guarantee normal accuracy); the user is asked whether to adopt it.
4. Best-practice policy YAML with the model-recommended quantization method (W8A8) plus the specified scenario tags; the user is asked whether to adopt it.
5. Best-practice policy YAML with the model-recommended quantization method (W8A8) and scenario tags ignored; the user is asked whether to adopt it.
6. Default-practice policy YAML with the model-recommended quantization method (W8A8) and scenario tags ignored (a default-practice policy does not guarantee normal accuracy); the user is asked whether to adopt it.

> **Note**: the configuration files in the best-practice library are located in the [`msmodelslim/lab_practice`](../../../../lab_practice/) directory.

## 5. Referenced Configurations

For the protocol overview, see *[Quantization Task Configuration Reference Navigation](../config/quant/README.md)*; if the configuration is used as an auto-tuning template, see also *[Auto Tuning Configuration Reference Navigation](../config/tuning/README.md)*.

| Related Parameter | Configuration Name | Reference Relationship | Configuration Document |
|----------|----------|----------|----------|
| `--config` | `modelslim_v1` | Loads the entire quantization YAML | *[modelslim_v1 Configuration Reference](../config/quant/modelslim_v1.md)* |
| `--config` | `multimodal_vlm_modelslim_v1` | Multimodal understanding model quantization YAML | *[multimodal_vlm_modelslim_v1 Configuration Reference](../config/quant/multimodal_vlm_modelslim_v1.md)* |
| `--config` | `multimodal_sd_modelslim_v1` | Multimodal generation model quantization YAML | *[multimodal_sd_modelslim_v1 Configuration Reference](../config/quant/multimodal_sd_modelslim_v1.md)* |
| `--config` | `modelslim_convert` | Weight conversion YAML; `--model_type` is not needed | *[modelslim_convert Configuration Reference](../config/quant/modelslim_convert.md)* |
| `--quant_type` | `lab_practice` best-practice YAML | Matched by model and quantization type | *[One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md)* |

## 6. Environment Variables

| Environment Variable | Type | Required/Optional | Default | Value Range or Format | Description |
|----------|------|-----------|--------|----------------|------|
| `MSMODELSLIM_CUSTOM_PRACTICE_REPO` | `string` | Optional | None (not set) | Directory path | Directory of a custom best-practice repository; once set, it takes part in best-practice matching together with the official `lab_practice`. |
| `MSMODELSLIM_LOG_LEVEL` | `string` | Optional | `INFO` | `INFO`, `DEBUG` | Set the log level; once set, logs at that level and above are printed. |

## 7. Usage Examples

### 7.1 One-Click Quantization by the Default Quantization Type (Minimal Runnable Scenario)

Contains only the minimum parameters required to complete one quantization run:

```bash
msmodelslim quant \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH}
```

`${MODEL_TYPE}` is the model type name (such as `Qwen2.5-7B-Instruct`), `${MODEL_PATH}` is the floating-point weight directory, and `${SAVE_PATH}` is the quantization output directory. When `--quant_type` and `--config` are not specified, a best practice is matched by the default quantization type `w8a8` and quantization is performed; if model loading requires custom code inside the model directory, additionally pass `--trust_remote_code true`.

### 7.2 Quantization with the Transformers Generic Base-Model Adapter (LLMs Only)

```bash
msmodelslim quant \
  --model_type transformers \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --quant_type w8a16
```

### 7.3 Explicitly Specifying the Quantization Type and the Device

```bash
msmodelslim quant \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --quant_type w8a8c8 \
  --device npu \
  --device_id 0 1 2 3
```

`--quant_type w8a8c8` means 8-bit weight, 8-bit activation, and 8-bit KVCache quantization; `--device_id 0 1 2 3` uses 4 NPU devices. For the weight-conversion command, see [7.7](#77-weight-conversion-with-an-explicitly-specified-device).

### 7.4 Using a Custom Configuration File

```bash
msmodelslim quant \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config ${CONFIG_PATH}
```

`${CONFIG_PATH}` points to a quantization YAML that follows the V1 protocol; once it is specified, that configuration is adopted directly and no best-practice matching is performed. For field descriptions, see the cited configuration document. For a weight-conversion example, see [7.7](#77-weight-conversion-with-an-explicitly-specified-device).

### 7.5 Matching a Best Practice with Scenario Tags

```bash
msmodelslim quant \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --quant_type w8a8 \
  --tags MindIE Atlas_A2_Inference
```

The several tags after `--tags` must all occur in the same verified scenario; when there is no exact match, the command gives a prompt and waits for confirmation (enter `Y`/`y` to continue; entering anything else exits); when a standby configuration is hit, the command prompts you to switch to the standby configuration, which still requires user confirmation.

### 7.6 Pure Weight Conversion (No Quantization)

When `--config` points to a configuration with `apiversion: modelslim_convert`, only the weight format is converted, and neither `--model_type` nor `--quant_type` is needed:

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config ${CONFIG_PATH}
```

`${MODEL_PATH}` is the source weight directory, `${SAVE_PATH}` is the conversion output directory, and `${CONFIG_PATH}` is the conversion YAML. For the configuration protocol and fields, see *[modelslim_convert Configuration Reference](../config/quant/modelslim_convert.md)*; for the operational steps and an example YAML, see *[Weight Conversion Usage Guide](../../knowledge_base/ptq/convert/usage_weight_conversion.md)*.

### 7.7 Weight Conversion with an Explicitly Specified Device

For multi-device NPU runs, add `--device npu` and `--device_id`:

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config ${CONFIG_PATH} \
  --device npu \
  --device_id 0 1 2 3
```

## 8. Exit Codes and Exception Handling

Normal operation raises no exception; on failure, an exception is thrown. Common causes: passing `--config` and `--quant_type` at the same time, a missing `--model_type` on the normal quantization path, YAML validation failure, and an invalid `--device_id` index or one that exceeds the available devices.

## 9. Security Notes

- `--trust_remote_code true` executes custom Python code in the model directory; enable it only when the source is confirmed trustworthy.
- `--save_path` writes the quantization result files; make sure the directory is writable and that unexpected data will not be overwritten.
- `--debug` writes the intermediate quantization context (including tensor data) to `${SAVE_PATH}/debug_info/`; manage that directory according to your organization's security policy.
