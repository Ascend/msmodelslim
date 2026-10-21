# msmodelslim eval Command-Line API Document

## 1. Function Description

`msmodelslim eval` is the fake-quantization inference verification command. Aimed at small-sample inference scenarios such as bad cases, it validates the accuracy of quantization artifacts. The command reads a quantization export directory, rebuilds a fake-quantized model structure from the floating-point model structure, loads the weights layer by layer to drive native inference, and outputs the inference results.

Command boundaries: only text-generation models of LLMs and multimodal understanding models (VLMs) are supported; diffusion-style image/video generation models such as DiT are **not supported**. Currently only AscendV1 format exports are supported.

## 2. Command Format

```text
msmodelslim eval --model_type <model_type> --model_path <model_path> --prompt_file <prompt_file> [--device <device>] [--device_id <id> ...] [--max_new_tokens <n>] [--log_level <level>] [-v] [-q]
```

Symbols:

- Values in angle brackets must be replaced; parameters in square brackets are optional.
- This command has no positional parameters; all parameters are passed as options. This syntax summary describes the parameter structure and is not a copy-ready command; for copy-ready commands, see [Usage Examples](#5-usage-examples).

## 3. Parameter List

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--model_type` | None | `string` | Single value | Required | None | Model type name, such as `Qwen3-32B` or `Qwen3.6-27B` | Specifies the type of the model to verify, used to load the corresponding model adapter |
| `--model_path` | None | `string` | Single value | Required | None | Quantization export directory (must exist and be readable) | Quantized weight directory to verify |
| `--prompt_file` | None | `string` | Single value | Required | None | Absolute or relative file path. For LLMs, the sample file extension must be `.json` or `.jsonl`; for VLMs, the multimodal sample file must be named `index.json` or `index.jsonl` | Inference sample file. |
| `--device` | None | `string` | Single value | Optional | `npu` | `npu`, `cpu`; `npu` is recommended | Device type to run on |
| `--device_id` | None | `list` | One or more values, space-separated | Optional | None | A list of non-negative integers, such as `0` or `0 1 2 3` | Device indices; passing several values enables multi-device sample-parallel (DP) inference, while a single value or nothing means single-device inference. |
| `--max_new_tokens` | None | `int` | Single value | Optional | `1` | A positive integer (≥ 1) | Number of new tokens generated per sample (excluding the prompt part). |
| `--log_level` | None | `string` | Single value | Optional | `info` | `debug`, `info`, `warning`, `error` | Log level. |
| `-v` / `--verbose` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Increase the level of output detail (equivalent to `--log_level debug`). |
| `-q` / `--quiet` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Suppress non-error output (equivalent to `--log_level error`). |
| `-V` / `--version` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Displays the version information and exits (a top-level global parameter that can be used after any subcommand, for example, `msmodelslim eval --version`). |
| `-h` / `--help` | None | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Displays the help information of this command and exits (for example, `msmodelslim eval -h` or `msmodelslim eval --help`). |

## 4. Parameter Relationships

- Device indices may be specified with `--device_id` only when `--device npu` is specified.

## 5. Usage Examples

### 5.1 Minimal Runnable Single-Device Scenario

Contains only the minimum parameters required to complete one fake-quantization inference run:

```bash
msmodelslim eval \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --prompt_file ${PROMPT_FILE}
```

`${MODEL_TYPE}` is the model type name (such as `Qwen3-32B`), `${MODEL_PATH}` is the quantization export directory, and `${PROMPT_FILE}` is the path to the inference sample file. By default it runs on `npu`, and `--max_new_tokens` takes the default value `1`, meaning that only 1 token is generated per sample.

### 5.2 Multi-Device Parallel Inference

```bash
msmodelslim eval \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --prompt_file ${PROMPT_FILE} \
  --device npu \
  --device_id 0 1 2 3 \
  --max_new_tokens 5
```

`--device npu --device_id 0 1 2 3` uses 4 NPUs for data parallelism: samples are sharded round-robin across the devices for inference, and the results are merged in global sample order. When the number of samples is not divisible by the number of devices, the shortfall is filled by copying from the beginning; the results of the filler samples are discarded during merging and do not affect the output order.

## 6. Exit Codes and Exception Handling

Normal operation raises no exception, and `token_ids` and `generated_text` can be inspected in the log. On failure, an exception is thrown. Common causes: `--model_type`/`--model_path`/`--prompt_file` missing or a path that is not readable, a sample file that is empty or whose format does not conform to the convention of the corresponding model, and an invalid `--device_id` index or one that exceeds the available devices.

## 7. Security Notes

- `--prompt_file` is read locally; make sure the directory access permissions and security meet expectations.
