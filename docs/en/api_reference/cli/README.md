# msmodelslim Command-Line API Overview

`msmodelslim` is the command-line entry point of MindStudio msModelSlim and provides four kinds of subcommands: quantization, sensitive layer analysis, fake-quantization accuracy evaluation, and auto tuning. This page collects the global parameters, the subcommand list, and the documentation entry point for each subcommand; for the parameter details, parameter relationships, and usage examples of a subcommand, see the corresponding document.

## 1. Command Format

```text
msmodelslim <command> [<args>]
```

Symbols:

- `<command>` is the subcommand name (a positional parameter); when it is omitted or an unknown subcommand is passed, the top-level help is printed.
- Values in angle brackets must be replaced; parameters in square brackets are optional.
- `-h`/`--help` and `-V`/`--version` can be used at the top level or after any subcommand (for example, `msmodelslim quant --version`).

## 2. Subcommand List

| Subcommand | Function | Documentation |
|--------|------|------|
| `quant` | One-click quantization or weight conversion, saving the quantized weights and the description file | *[msmodelslim quant Command-Line API Document](msmodelslim_quant.md)* |
| `analyze` | Sensitive layer analysis before quantization, listing the layer names from the highest sensitivity score to the lowest | *[msmodelslim analyze Command-Line API Document](msmodelslim_analyze.md)* |
| `eval` | Fake-quantized model accuracy evaluation based on an AscendV1 export | *[msmodelslim eval Command-Line API Document](msmodelslim_eval.md)* |
| `tune` | Automatically tunes the quantization configuration to reach the target accuracy | *[msmodelslim tune Command-Line API Document](msmodelslim_tune.md)* |

## 3. Global Parameters

Global parameters are parsed at the top level and apply to all subcommands:

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--help` | `-h` | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Displays the help information. |
| `--version` | `-V` | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Displays the version information (version number, build information, repository address) and exits. This parameter is handled before the subcommand is parsed, so it takes effect wherever it is passed. |

## 4. Common Parameters

The following parameters are registered separately by each subcommand (all of `quant`, `analyze`, `eval`, and `tune` provide them) and are written after the subcommand:

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--log_level` | None | `string` | Single value | Optional | `info` | `debug`, `info`, `warning`, `error` | Log level. When passed explicitly, it takes precedence over `-v`/`-q`. |
| `--verbose` | `-v` | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Increase the level of output detail (equivalent to `--log_level debug`). |
| `--quiet` | `-q` | `bool` | Boolean switch without a value | Optional | Off | Enabled when passed | Suppress non-error output (equivalent to `--log_level error`). |

## 5. Usage Examples

### 5.1 View the Version and the Top-Level Help

```bash
msmodelslim --version
msmodelslim -h
```

`-V`/`--version` outputs `msmodelslim <version> (<git hash>)` and the build information; `-h`/`--help` outputs the subcommand list. Neither depends on the model or the weight paths.

### 5.2 View the Help for a Subcommand

```bash
msmodelslim quant -h
msmodelslim analyze linear -h
```

The top-level help lists only the subcommands; to see the parameter description of a subcommand, add `-h`/`--help` after that subcommand.

## 6. Exit Codes and Exception Handling

Normal operation raises no exception. On failure, an exception is thrown; see the error log for the specific reason. `-V`/`--version` and `-h`/`--help` end normally and do not throw an exception.
