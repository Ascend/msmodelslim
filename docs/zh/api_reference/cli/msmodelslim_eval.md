# msmodelslim eval 命令行 API 文档

## 1. 功能说明

`msmodelslim eval` 是伪量化推理验证命令，面向 bad case 等小推理样本场景，对量化产物做精度验证。命令读取一个量化导出目录，基于浮点模型结构重建伪量化模型结构，逐层加载权重驱动原生推理，输出推理结果。

命令边界：仅支持大语言模型（LLM）与多模态理解模型（VLM）的文本生成类模型，**不支持** DiT 等扩散式图像/视频生成模型。当前仅支持 AscendV1 格式导出件。

## 2. 命令格式

```text
msmodelslim eval --model_type <model_type> --model_path <model_path> --prompt_file <prompt_file> [--device <device>] [--device_id <id> ...] [--max_new_tokens <n>] [--log_level <level>] [-v] [-q]
```

符号说明：

- 尖括号内为需替换的值，方括号内为可选参数。
- 本命令无位置参数，全部参数通过选项传入。本语法摘要用于说明参数结构，不作为可复制命令；可复制命令见[使用示例](#5-使用示例)。

## 3. 参数列表

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--model_type` | 无 | `string` | 单值 | 必选 | 无 | 模型类型名称，如 `Qwen3-32B`、`Qwen3.6-27B` | 指定待验证模型类型，用于加载对应模型适配器 |
| `--model_path` | 无 | `string` | 单值 | 必选 | 无 | 量化导出目录（需存在且可读） | 待验证的量化权重目录 |
| `--prompt_file` | 无 | `string` | 单值 | 必选 | 无 | 文件绝对/相对路径。LLM 要求样本文件扩展名为 `.json` 或 `.jsonl`；VLM 多模态样本文件须命名为 `index.json` 或 `index.jsonl` | 推理样本文件。 |
| `--device` | 无 | `string` | 单值 | 可选 | `npu` | `npu`、`cpu`，推荐使用 `npu` | 运行设备类型 |
| `--device_id` | 无 | `list` | 一次接收多个值，空格分隔 | 可选 | 无 | 非负整数列表，如 `0` 或 `0 1 2 3` | 设备索引；传入多个值时启用多卡样本并行（DP）推理，单值或不传为单卡推理。 |
| `--max_new_tokens` | 无 | `int` | 单值 | 可选 | `1` | 正整数（≥ 1） | 每个样本新生成的 token 数（不含 prompt 部分）。 |
| `--log_level` | 无 | `string` | 单值 | 可选 | `info` | `debug`、`info`、`warning`、`error` | 日志级别。 |
| `-v` / `--verbose` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 提高输出详细程度（等价 `--log_level debug`）。 |
| `-q` / `--quiet` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 抑制非错误输出（等价 `--log_level error`）。 |
| `-V` / `--version` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 显示版本信息后退出（顶层全局参数，可在任意子命令后使用，如 `msmodelslim eval --version`）。 |
| `-h` / `--help` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 显示本命令帮助信息后退出（如 `msmodelslim eval -h` 或 `msmodelslim eval --help`）。 |

## 4. 参数关系

- 仅当指定 `--device npu` 时，可通过 `--device_id` 指定设备索引。

## 5. 使用示例

### 5.1 单卡最小可运行场景

只包含完成一次伪量化推理所需的最少参数：

```bash
msmodelslim eval \
  --model_type "${MODEL_TYPE}" \
  --model_path "${MODEL_PATH}" \
  --prompt_file "${PROMPT_FILE}"
```

`${MODEL_TYPE}` 为模型类型名称（如 `Qwen3-32B`），`${MODEL_PATH}` 为量化导出目录，`${PROMPT_FILE}` 为推理样本文件路径。默认在 `npu` 上运行，`--max_new_tokens` 取默认值 `1`，即每个样本仅生成 1个 token。

### 5.2 多卡并行推理

```bash
msmodelslim eval \
  --model_type "${MODEL_TYPE}" \
  --model_path "${MODEL_PATH}" \
  --prompt_file "${PROMPT_FILE}" \
  --device npu \
  --device_id 0 1 2 3 \
  --max_new_tokens 5
```

`--device npu --device_id 0 1 2 3` 使用 4张 NPU 做数据并行：样本按卡轮转分片进行推理，按全局样本序归并结果。样本数不能被卡数整除时，不足部分会从头复制补位，补位样本的结果在归并时被丢弃，不影响输出顺序。

## 6. 退出码与异常处理

正常运行则无异常，可在日志中查看 `token_ids` 与 `generated_text`。失败时抛出异常。常见原因：`--model_type`/`--model_path`/`--prompt_file` 缺失或路径不可读、样本文件为空或格式不符合对应模型的约定、`--device_id` 索引非法或超出可用设备。

## 7. 安全说明

- `--prompt_file` 从本地读取，请确认目录访问权限与安全符合预期。
