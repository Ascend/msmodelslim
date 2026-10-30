# msmodelslim tune Command-Line API Document

## 1. Function Description

`msmodelslim tune` is the auto-tuning command. It reads a tuning YAML containing `strategy` and `evaluation`, runs the loop "generate a quantization configuration → quantize → evaluate accuracy", and iteratively searches, based on the evaluation results, for a quantization configuration that meets the accuracy expectation, until the strategy generator is exhausted, the `--timeout` expires, or the maximum number of iterations is reached. Results such as the quantized model and the evaluation history are written to `--save_path`; after tuning finishes, the final best practice can be saved to a custom best-practice repository.

Command boundaries: this command does not expand on how to write specific tuning strategies and evaluation configurations; for field descriptions, see *[Auto Tuning Configuration Reference Navigation](../config/tuning/README.md)*, whose strategy templates reference the quantization task protocol described in *[Quantization Task Configuration Reference Navigation](../config/quant/README.md)*; for operational steps, see *[Auto Tuning Usage Guide](../../user_guide/usage_auto_precision_tuning.md)*.

## 2. Command Format

```text
msmodelslim tune --model_path <model_path> --save_path <save_path> --config <config> [--model_type <model_type>] [--device <device>] [--device_id <id> ...] [--timeout <timeout>] [--trust_remote_code [<BOOL>]] [--log_level <level>] [-v] [-q]
```

Symbols:

- Values in angle brackets must be replaced; parameters in square brackets are optional.
- `--model_path`, `--save_path`, and `--config` are required parameters.
- `--config` is the required path of the tuning YAML; it has the same name as the `--config` of the one-click quantization command but a different meaning (the one-click quantization `--config` is optional).
- `--timeout` is passed as an integer number of seconds (for example, `7200` means 2 hours); the old duration-string spellings (such as `2H` and `3D4H`) are kept only for backward compatibility, and new usages should use seconds uniformly.
- `--trust_remote_code` is a Boolean parameter: passing it without a value enables it (equivalent to `true`), and an explicit `true`/`false` may also be passed (case-insensitive; the legacy `True`/`False` spellings are also accepted).
- This command has no positional parameters. This syntax summary describes the parameter structure and is not a copy-ready command; for copy-ready commands, see [Usage Examples](#7-usage-examples).

## 3. Parameter List

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--model_type` | None | `string` | Single value | Optional | `default` | Model type name, such as `Qwen2.5-7B-Instruct` or `Qwen3-32B` | Type of the model to tune, which must match a name in the *[Foundation Model Support Matrix](../../knowledge_base/model/README.md)*; when it is not passed, the default adapter `default` is used. |
| `--model_path` | None | `string` | Single value | Required | None | Original model weight directory (must exist and be readable) | Weight directory of the model to tune. |
| `--save_path` | None | `string` | Single value | Required | None | Output directory (must be writable) | Directory in which the tuning results, the quantized model, and the history records are saved. |
| `--config` | `-c` | `string` | Single value | Required | None | Path to a readable YAML file | Tuning configuration YAML, containing the `strategy` and `evaluation` fields. |
| `--device` | None | `string` | Single value | Optional | `npu` | `npu`, `cpu` | Device type to run on; specify multi-device indices with `--device_id`. |
| `--device_id` | None | `list` | One or more values, space-separated | Optional | None | A list of non-negative integers, such as `0` or `0 1 2 3` | Device indices, used to specify multiple NPU devices. |
| `--timeout` | None | `int` (seconds) | Single value | Optional | None (no limit) | An integer number of seconds, such as `7200` (2 hours) or `3600` (1 hour); the old duration-string spelling (such as `1D2H30M15S`, with the units in the fixed order D/H/M/S) is kept only for backward compatibility | Tuning wall-clock timeout; when the timeout is reached, the current tuning run stops. |
| `--trust_remote_code` | None | `bool` | Boolean parameter: passing it without a value enables it; an explicit `true`/`false` value may also be given | Optional | `false` | `true`/`false` (case-insensitive; the legacy `True`/`False` spellings are also accepted) | Whether to trust and execute custom Python code in the model directory; enable it only when the code source is confirmed trustworthy. |
| `--log_level` | None | `string` | Single value | Optional | `info` | `debug`, `info`, `warning`, `error` | Log level. |
| `-v` / `--verbose` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Increase the level of output detail (equivalent to `--log_level debug`). |
| `-q` / `--quiet` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Suppress non-error output (equivalent to `--log_level error`). |
| `-h` / `--help` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Displays the help information and exits (for example, `msmodelslim tune -h`). |
| `-V` / `--version` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Displays the version information and exits (a top-level global parameter that can be used after any subcommand, for example, `msmodelslim tune --version`). |

## 4. Parameter Relationships

- This command loads the tuning YAML through the required `--config`.
- When `--timeout` is not set, the tuning wall-clock time is not limited; once it is set, the current tuning run stops when the timeout is reached.
- `--device` accepts only `npu`/`cpu`, and multi-device indices are specified through `--device_id`.
- When `--model_type` is not passed, the default value `default` is used.

## 5. Referenced Configurations

| Related Parameter | Configuration Name | Reference Relationship | Configuration Document |
|----------|----------|----------|----------|
| `--config` | Auto tuning configuration | Loads `strategy` and `evaluation` | *[Auto Tuning Configuration Reference Navigation](../config/tuning/README.md)* |

## 6. Environment Variables

| Environment Variable | Type | Required/Optional | Default | Value Range or Format | Description |
|----------|------|-----------|--------|----------------|------|
| `MSMODELSLIM_CUSTOM_PRACTICE_REPO` | `string` | Optional | None (not set) | Directory path | After tuning finishes, the final best practice is saved to this custom best-practice repository; when it is not set, the command only records a "saving is not supported" notice. |
| `MSMODELSLIM_LOG_LEVEL` | `string` | Optional | `INFO` | `INFO`, `DEBUG` | Set the log level; once set, logs at that level and above are printed. |

## 7. Usage Examples

### 7.1 Minimal Tuning Command

Contains only the minimum parameters required to complete one auto-tuning run:

```bash
msmodelslim tune \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config ${CONFIG_PATH}
```

`${MODEL_PATH}` is the weight directory, `${SAVE_PATH}` is the tuning result directory, and `${CONFIG_PATH}` is the tuning YAML (containing `strategy` and `evaluation`). When `--model_type` is not specified, the default adapter `default` is used; specifying it explicitly is recommended.

### 7.2 Explicitly Specifying the Model Type, the Device, and the Timeout

```bash
msmodelslim tune \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config ${CONFIG_PATH} \
  --timeout 7200 \
  --device npu \
  --device_id 0 1 2 3
```

`--timeout` set to `7200` means running for at most 7200 seconds (2 hours); `--device_id` set to `0 1 2 3` means using 4 NPU devices.

## 8. Exit Codes and Exception Handling

Normal operation raises no exception (including when the policy iteration completes or the maximum number of iterations is reached); the results are found in the quantized model, the results, and the history records under `${SAVE_PATH}`. On failure, an exception is thrown. Common causes: `--config` YAML validation failure, an invalid `--timeout` format, and failure to load the model adapter.

## 9. Security Notes

- `--trust_remote_code true` executes custom Python code in the model directory; enable it only when the source is confirmed trustworthy.
- `--save_path` writes the tuning results and history records; make sure the directory is writable and that unexpected data will not be overwritten.
