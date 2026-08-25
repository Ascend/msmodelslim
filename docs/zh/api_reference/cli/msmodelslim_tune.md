# msmodelslim tune 命令行 API 文档

## 1. 功能说明

`msmodelslim tune` 是自动调优命令，读取包含 `strategy` 与 `evaluation` 的调优 YAML，循环执行「生成量化配置 → 量化 → 精度评估」，根据评估结果迭代搜索满足精度期望的量化配置，直至策略生成器耗尽、达到 `--timeout` 超时或达到最大迭代次数。量化模型与评估历史等结果写入 `--save_path`；调优结束后，最终最佳实践可保存到自定义最佳实践仓库。

命令边界：本命令不展开具体调优策略与评估配置的编写；字段说明见《[自动调优配置协议说明](../config/auto_precision_tuning.md)》，操作步骤见《[自动调优使用说明](../../user_guide/usage_auto_precision_tuning.md)》。

## 2. 命令格式

```text
msmodelslim tune --model_path <model_path> --save_path <save_path> --config <config> [--model_type <model_type>] [--device <device>] [--device_id <id> ...] [--timeout <timeout>] [--trust_remote_code [<BOOL>]] [--log_level <level>] [-v] [-q]
```

符号说明：

- 尖括号内为需替换的值，方括号内为可选参数。
- `--model_path`、`--save_path`、`--config` 为必选参数。
- `--config` 是必选的调优 YAML 路径；与一键量化命令的 `--config` 同名，但语义不同（一键量化的 `--config` 可选）。
- `--timeout` 为时长字符串，如 `2H`、`3D4H`。
- `--trust_remote_code` 是可选值布尔参数：不带值传入即开启（等价 `true`），或跟 `true`/`false`（大小写不敏感，兼容遗留 `True`/`False`、`yes`/`no`、`on`/`off`）。
- 本命令无位置参数。本语法摘要用于说明参数结构，不作为可复制命令；可复制命令见「使用示例」。

## 3. 参数列表

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--model_type` | 无 | `string` | 单值 | 可选 | `default` | 模型类型名称，如 `Qwen2.5-7B-Instruct`、`Qwen-QwQ-32B` | 待调优模型类型；不传时使用默认适配器 `default`。 |
| `--model_path` | 无 | `string` | 单值 | 必选 | 无 | 原始模型权重目录（需存在且可读） | 待调优模型的权重目录。 |
| `--save_path` | 无 | `string` | 单值 | 必选 | 无 | 输出目录（需可写） | 调优结果、量化模型与历史记录的保存目录。 |
| `--config` | 无 | `string` | 单值 | 必选 | 无 | 可读 YAML 文件路径 | 调优配置 YAML，含 `strategy` 与 `evaluation` 字段。 |
| `--device` | 无 | `string` | 单值 | 可选 | `npu` | `npu`、`cpu` | 运行设备类型；多卡索引请用 `--device_id` 指定。 |
| `--device_id` | 无 | `list` | 一次接收多个值，空格分隔 | 可选 | 无 | 非负整数列表，如 `0` 或 `0 1 2 3` | 设备索引，用于指定多个 NPU 设备。 |
| `--timeout` | 无 | `string` | 单值 | 可选 | 无（不限时） | 形如 `1D2H30M15S`，单位固定顺序 D/H/M/S，可省略任意一段（如 `1D2H`、`30M`、`10S`），至少含一个单位，字母大写 | 调优墙钟超时；到达超时时间后停止本次调优。 |
| `--trust_remote_code` | 无 | `bool` | 可选值（不带值传入即开启，或跟 `true`/`false`） | 可选 | `false` | `true`/`false`（大小写不敏感；兼容 `True`/`False`、`yes`/`no`、`on`/`off`） | 是否信任并执行模型目录中的自定义 Python 代码；仅在确认代码来源可信时开启。 |
| `--log_level` | 无 | `string` | 单值 | 可选 | `info` | `debug`、`info`、`warning`、`error` | 日志级别。 |
| `-v` / `--verbose` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 提高输出详细程度（等价 `--log_level debug`）。 |
| `-q` / `--quiet` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 抑制非错误输出（等价 `--log_level error`）。 |

## 4. 参数关系

- 本命令通过必选的 `--config` 加载调优 YAML；与一键量化命令的 `--config` 同名但语义不同，本命令不接收 `--config_path`。
- `--timeout` 未设置时不限制调优墙钟时间；设置后超时即停止当前调优。
- `--device` 仅接受 `npu`/`cpu`，多卡索引通过 `--device_id` 指定。
- `--model_type` 不传时使用默认值 `default`。

## 5. 引用的配置

| 关联参数 | 配置名称 | 引用关系 | 配置文档 |
|----------|----------|----------|----------|
| `--config` | 自动调优配置 | 加载 `strategy` 与 `evaluation` | 《[自动调优配置协议说明](../config/auto_precision_tuning.md)》 |

## 6. 环境变量

| 环境变量 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|----------|------|-----------|--------|----------------|------|
| `MSMODELSLIM_CUSTOM_PRACTICE_REPO` | `string` | 可选 | 无（未设置） | 目录路径 | 调优结束后将最终最佳实践保存到该自定义最佳实践仓库；未设置时命令仅记录「不支持保存」的提示。 |
| `MSMODELSLIM_LOG_LEVEL` | `string` | 可选 | `INFO` | `INFO`、`DEBUG` | 设置日志级别；设置后打印同级及以上日志。 |

## 7. 使用示例

### 7.1 最小调优命令

只包含完成一次自动调优所需的最少参数：

```bash
msmodelslim tune \
  --model_path "${MODEL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --config "${CONFIG_PATH}"
```

`${MODEL_PATH}` 为权重目录，`${SAVE_PATH}` 为调优结果目录，`${CONFIG_PATH}` 为调优 YAML（含 `strategy` 与 `evaluation`）。未指定 `--model_type` 时使用默认适配器 `default`，推荐显式指定。

### 7.2 指定模型类型与超时

```bash
msmodelslim tune \
  --model_type "${MODEL_TYPE}" \
  --model_path "${MODEL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --config "${CONFIG_PATH}" \
  --timeout 2H \
  --device npu \
  --device_id 0 1 2 3
```

`--timeout 2H` 表示最多运行2H；`--device_id 0 1 2 3` 使用4个 NPU 设备。

## 8. 退出码与异常处理

| 退出码或异常 | 含义 | 处理建议 |
|--------------|------|----------|
| `0` | 调优流程正常结束（含策略迭代完成、超时或达到最大迭代次数） | 查看 `${SAVE_PATH}` 下的量化模型、结果与历史记录。 |
| 非 `0` | 失败 | 查看错误日志。常见原因：`--config` YAML 校验失败、`--timeout` 格式非法、模型适配器加载失败。 |

## 9. 安全说明

- `--trust_remote_code true` 会执行模型目录中的自定义 Python 代码，仅在确认来源可信时开启。
- `--save_path` 会写入调优结果与历史记录；请确认目录可写且不会覆盖非预期数据。
