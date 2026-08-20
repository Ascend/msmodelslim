# msmodelslim analyze 命令行 API 文档

## 1. 功能说明

`msmodelslim analyze` 在量化前对模型做敏感层分析，输出建议回退或重点关注的层名列表，供后续写入量化配置的 `exclude` 等字段。按分析粒度分为三个 scope：`linear`（逐个线性层）、`layer`（按层/块分组）、`attn`（Attention 模块）。分析使用校准数据集在目标设备上计算敏感度指标，并按 `--topk` 输出敏感度最高的层名（`disable_names`）。

命令边界：本命令只做分析与结果输出，不执行量化，且需要模型适配器实现分析接口。操作步骤见《[线性层敏感层分析使用指南](../../user_guide/usage_sensitive_linear_analysis.md)》、《[层级敏感层分析使用指南](../../user_guide/usage_sensitive_layer_wise_analysis.md)》、《[Attention 敏感层分析使用指南](../../user_guide/usage_sensitive_attn_analysis.md)》。

## 2. 命令格式

```text
msmodelslim analyze linear --model_type <model_type> --model_path <model_path> [--device <device>] [--calib_dataset <calib_dataset>] [--topk <topk>] [--trust_remote_code <True|False>] [--metrics <metrics>] [--pattern <pattern> ...]

msmodelslim analyze layer --model_type <model_type> --model_path <model_path> [--device <device>] [--calib_dataset <calib_dataset>] [--topk <topk>] [--trust_remote_code <True|False>] [--metrics <metrics>] [--quant_modules <module> ...]

msmodelslim analyze attn --model_type <model_type> --model_path <model_path> [--device <device>] [--calib_dataset <calib_dataset>] [--topk <topk>] [--trust_remote_code <True|False>] [--metrics <metrics>]
```

符号说明：

- `<scope>`（`linear`/`layer`/`attn`）为位置参数，也是子命令名。
- 尖括号内为需替换的值，方括号内为可选参数。
- `--trust_remote_code` 是显式值布尔参数，必须紧跟字面量 `True` 或 `False`。
- `--pattern`、`--quant_modules` 一次接收多个值，`...` 表示可跟多个以空格分隔的值。
- 各 scope 的专有选项不同：`linear` 有 `--metrics`、`--pattern`；`layer` 有 `--metrics`、`--quant_modules`；`attn` 只有 `--metrics`。
- 省略 `<scope>` 时（非帮助请求）自动按 `linear` 执行；本命令无其他位置参数。本语法摘要用于说明参数结构，不作为可复制命令；可复制命令见「使用示例」。

## 3. 参数列表

公共参数（所有 scope 通用）：

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--model_type` | 无 | `string` | 单值 | 必选 | 无 | 模型类型名称，如 `Qwen2.5-7B-Instruct`、`Qwen-QwQ-32B` | 待分析模型类型，须与支持矩阵中的名称一致。 |
| `--model_path` | 无 | `string` | 单值 | 必选 | 无 | 原始模型权重目录（需存在且可读） | 待分析模型的权重目录。 |
| `--device` | 无 | `string` | 单值 | 可选 | `npu` | `npu`、`cpu` | 运行设备类型；与 `msmodelslim quant --device` 不同，不支持 `npu:0,1,2,3` 这类索引列表。 |
| `--calib_dataset` | 无 | `string` | 单值 | 可选 | `mix_calib.jsonl` | 可直接给文件路径，或给 `lab_calib` 目录下的文件名；后缀须为 `.json` 或 `.jsonl` | 校准数据集。 |
| `--topk` | 无 | `int` | 单值 | 可选 | `15` | 大于0的整数 | 输出到 `disable_names` 的最高敏感层数（经验值，仅供参考）。 |
| `--trust_remote_code` | 无 | `bool` | 显式值（必须跟字面量 `True` 或 `False`） | 可选 | `False` | 字面量 `True` 或 `False`（大小写敏感） | 是否信任并执行模型目录中的自定义 Python 代码；仅在确认代码来源可信时开启。 |

位置参数：

| 位置参数 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|----------|------|----------|-----------|--------|----------------|------|
| `scope` | `string` | 位置参数（子命令） | 可选 | 未指定且非帮助请求时注入 `linear` | `linear`、`layer`、`attn` | 分析粒度。省略时默认 `linear`；请求帮助（`-h`/`--help`）时不注入。 |

`linear` 专有参数：

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--metrics` | 无 | `string` | 单值 | 可选 | `kurtosis` | `std`、`quantile`、`kurtosis` | 线性层敏感度指标：标准差、分位数、峰度。 |
| `--pattern` | 无 | `list` | 一次接收多个值，空格分隔 | 可选 | `['*']` | 通配符模式列表 | 过滤要展示的线性层；`*` 表示全部。 |

`layer` 专有参数：

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--metrics` | 无 | `string` | 单值 | 可选 | `mse_layer_wise` | `mse_model_wise`、`mse_layer_wise` | 层级敏感度指标：按层计算误差或按模型整体计算误差。 |
| `--quant_modules` | 无 | `list` | 一次接收多个值，空格分隔 | 可选 | `['*']` | 通配符模块列表 | 映射到 pipeline 量化范围的模块通配符；`*` 表示全部。 |

`attn` 专有参数：

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--metrics` | 无 | `string` | 单值 | 可选 | `mse` | `mse` | Attention 模块敏感度指标，仅支持 `mse`，作用于全部 Attention 模块。 |

## 4. 参数关系

- `scope` 决定可用的 `--metrics` 取值与专有参数；跨 scope 混用（如 `analyze attn --pattern`）会导致参数解析失败，提示不支持的参数或取值。
- 省略 `scope` 时默认按 `linear` 执行；`msmodelslim analyze -h`/`--help` 时不注入 scope，展示 scope 帮助。
- `--device` 仅接受 `npu`/`cpu`，不支持 `npu:0,1,2,3` 索引列表写法。
- 历史用法 `--metrics attention_mse`（省略 scope 时）会转换为 `analyze attn --metrics mse` 并丢弃 `--pattern`；该行为仅为向后兼容，命令会给出废弃提示，不推荐作为正式用法。
- `--topk` 必须为大于0的整数；`--calib_dataset` 后缀必须为 `.json` 或 `.jsonl`。

## 5. 使用示例

### 5.1 最小可运行场景（线性层敏感度分析）

省略 scope，按默认 `linear` 执行：

```bash
msmodelslim analyze \
  --model_type "${MODEL_TYPE}" \
  --model_path "${MODEL_PATH}"
```

`${MODEL_TYPE}` 为模型类型名称，`${MODEL_PATH}` 为权重目录。省略 `scope` 时默认按 `linear` 分析，使用默认指标 `kurtosis`、默认 `--topk 15` 与默认校准集 `mix_calib.jsonl`，输出敏感度最高的15个层名（`disable_names`），可写入量化 YAML 的 `exclude`。

### 5.2 指定 linear 指标与 topk

```bash
msmodelslim analyze linear \
  --model_type "${MODEL_TYPE}" \
  --model_path "${MODEL_PATH}" \
  --metrics kurtosis \
  --topk 15
```

显式指定 `linear` scope，使用峰度指标，并按 `--topk 15` 输出高敏感层名。

### 5.3 Attention MSE 分析

```bash
msmodelslim analyze attn \
  --model_type "${MODEL_TYPE}" \
  --model_path "${MODEL_PATH}" \
  --metrics mse
```

对全部 Attention 模块计算 MSE 敏感度。

### 5.4 层级分析与量化模块

```bash
msmodelslim analyze layer \
  --model_type "${MODEL_TYPE}" \
  --model_path "${MODEL_PATH}" \
  --metrics mse_layer_wise \
  --quant_modules "${MODULE_1}" "${MODULE_2}"
```

`${MODULE_1}`、`${MODULE_2}` 为要映射到 pipeline 量化范围的模块通配符（空格分隔多个值）。

## 6. 退出码与异常处理

| 退出码或异常 | 含义 | 处理建议 |
|--------------|------|----------|
| `0` | 分析成功 | 按输出的层名更新量化配置（如写入 `exclude`）。 |
| 非 `0` | 失败 | 查看错误日志。常见原因：scope 与 `--metrics` 取值不匹配、`--device` 取值非法、`--calib_dataset` 后缀非 `.json`/`.jsonl`、`--topk` 非大于0的整数、模型适配器未实现分析接口。 |

## 7. 安全说明

- `--trust_remote_code True` 会执行模型目录中的自定义 Python 代码，仅在确认来源可信时开启。
