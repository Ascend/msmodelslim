# msmodelslim analyze 命令行 API 文档

## 1. 功能说明

`msmodelslim analyze` 在量化前对模型做敏感层分析，输出建议回退或重点关注的层名列表，供后续写入量化配置的 `exclude` 等字段。按分析粒度分为四个 scope：`linear`（逐个线性层）、`layer`（按层/块分组）、`attn`（Attention 模块）、`attn_head`（Attention Head / `ra_compress`）。分析在目标设备上计算敏感度指标，并按 `--top_k` 输出敏感度最高的层名（`disable_names`；`attn_head` 输出 `head.pt`）。默认只把结果打印到日志；指定 `--save_path` 后结果落盘为文件（`linear`/`layer`/`attn` 为 YAML，`attn_head` 为 `head.pt`）。其中 `linear`/`layer`/`attn` 使用 `--calibration_dataset` 指定的校准数据集；`attn_head`（`ra_compress`）不读取外部校准集，其校准输入由算法在运行时自行构造（首 token + 随机重复段）。

命令边界：本命令只做分析与结果输出，不执行量化，且需要模型适配器实现分析接口。支持单卡与多卡（DP）分析，多卡时使用 `--device npu --device_id 0 1 ...`（或兼容写法 `--device npu:0,1,...`），由 `DPLayerWiseRunner` 执行。操作步骤见《[线性层敏感层分析使用指南](../../user_guide/usage_sensitive_linear_analysis.md)》、《[层级敏感层分析使用指南](../../user_guide/usage_sensitive_layer_analysis.md)》、《[Attention 敏感层分析使用指南](../../user_guide/usage_sensitive_attn_analysis.md)》、《[Attention Head 筛选分析使用指南](../../user_guide/usage_sensitive_attn_head_analysis.md)》。

## 2. 命令格式

```text
msmodelslim analyze linear --model_type <model_type> --model_path <model_path> [--device <device>] [--device_id <id> ...] [--calibration_dataset <file>] [--top_k <n>] [--trust_remote_code [<BOOL>]] [--log_level <level>] [-v] [-q] [--metrics <metrics>] [--patterns <pattern> ...] [--save_path <path>]

msmodelslim analyze layer --model_type <model_type> --model_path <model_path> [--device <device>] [--device_id <id> ...] [--calibration_dataset <file>] [--top_k <n>] [--trust_remote_code [<BOOL>]] [--log_level <level>] [-v] [-q] [--metrics <metrics>] [--quant_modules <module> ...] [--save_path <path>]

msmodelslim analyze attn --model_type <model_type> --model_path <model_path> [--device <device>] [--device_id <id> ...] [--calibration_dataset <file>] [--top_k <n>] [--trust_remote_code [<BOOL>]] [--log_level <level>] [-v] [-q] [--metrics <metrics>] [--save_path <path>]

msmodelslim analyze attn_head --model_type <model_type> --model_path <model_path> [--device <device>] [--device_id <id> ...] [--trust_remote_code [<BOOL>]] [--log_level <level>] [-v] [-q] [--metrics <metrics>] [--save_path <path>]
```

符号说明：

- `<scope>`（`linear`/`layer`/`attn`/`attn_head`）为位置参数，也是子命令名。
- 尖括号内为需替换的值，方括号内为可选参数。
- `--trust_remote_code` 是可选值布尔参数：不带值传入即开启（等价 `true`），或跟 `true`/`false`（大小写不敏感，兼容遗留 `True`/`False`、`yes`/`no`、`on`/`off`）。
- `--patterns`、`--quant_modules`、`--device_id` 一次接收多个值，`...` 表示可跟多个以空格分隔的值。
- 各 scope 的专有选项不同：`linear` 有 `--metrics`、`--patterns`；`layer` 有 `--metrics`、`--quant_modules`；`attn` 只有 `--metrics`；`attn_head` 有 `--metrics`（仅 `ra_compress`）。`--save_path` 为四个 scope 通用的落盘选项，但落盘格式不同：`linear`/`layer`/`attn` 写 YAML，`attn_head` 写 `head.pt`。
- `--calibration_dataset` 对 `attn_head` 不生效：该场景的校准输入由算法在运行时自行构造，传入的校准集仅参与格式与安全检查，不影响分析结果。
- 省略 `<scope>` 时（非帮助请求）自动按 `linear` 执行；本命令无其他位置参数。本语法摘要用于说明参数结构，不作为可复制命令；可复制命令见「使用示例」。

## 3. 参数列表

公共参数（所有 scope 通用）：

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--model_type` | 无 | `string` | 单值 | 必选 | 无 | 模型类型名称，如 `Qwen2.5-7B-Instruct`、`Qwen3-32B` | 待分析模型类型，须与《[大模型支持矩阵](../../knowledge_base/model/README.md)》中的名称一致；未传值或传入的名称不存在时回退到默认适配器 `default` 执行。 |
| `--model_path` | 无 | `string` | 单值 | 必选 | 无 | 原始模型权重目录（需存在且可读） | 待分析模型的权重目录。 |
| `--device` | 无 | `string` | 单值 | 可选 | `npu` | `npu`、`cpu`；兼容遗留写法 `npu:0,1,...` | 运行设备类型。推荐写法为 `--device npu` 再配合 `--device_id`；遗留 `npu:0,1` 会自动拆分为 `--device npu --device_id 0 1`。 |
| `--device_id` | 无 | `list[int]` | 一次接收多个值，空格分隔 | 可选 | 无 | 非负整数索引，如 `0 1 2 3` | 分析所用 NPU 索引；长度 > 1 时启用多卡 DP（`DPLayerWiseRunner`）。CPU 场景不可指定多个索引。 |
| `--calibration_dataset` | `--calib_dataset`（遗留别名） | `string` | 单值 | 可选 | `mix_calib.jsonl` | LLM：`.json` / `.jsonl`；VLM：图文目录名（如 `calibImages`）或路径 | 校准数据集，用于 `linear`/`layer`/`attn` 的指标计算；`attn_head`（`ra_compress`）不读取该校准集（内部构造随机校准输入），此时仅参与格式与安全检查。 |
| `--top_k` | `--topk`（遗留别名） | `int` | 单值 | 可选 | `15` | 大于0的整数 | 输出到 `disable_names` 的最高敏感层数（经验值，仅供参考；`attn_head` 无此参数）。 |
| `--save_path` | 无 | `string` | 单值 | 可选 | 无（仅打印到日志） | 可写目录或文件路径：`linear`/`layer`/`attn` 为 `.yaml`/`.yml` 或目录；`attn_head` 为 `.pt` 或目录 | 分析结果的落盘路径。`linear`/`layer`/`attn`：路径以 `.yaml`/`.yml` 结尾时按该文件名写入，否则视为目录并按 `<model_type>-<metrics>.yaml` 命名；`attn_head`：路径以 `.pt` 结尾时按该文件名写入，否则写入目录下的 `head.pt`。不指定时结果仅打印到日志（控制台）。 |
| `--trust_remote_code` | 无 | `bool` | 可选值（不带值传入即开启，或跟 `true`/`false`） | 可选 | `false` | `true`/`false`（大小写不敏感；兼容 `True`/`False`、`yes`/`no`、`on`/`off`） | 是否信任并执行模型目录中的自定义 Python 代码；仅在确认代码来源可信时开启。 |
| `--log_level` | 无 | `string` | 单值 | 可选 | `info` | `debug`、`info`、`warning`、`error` | 日志级别。 |
| `-v` / `--verbose` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 提高输出详细程度（等价 `--log_level debug`）。 |
| `-q` / `--quiet` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 抑制非错误输出（等价 `--log_level error`）。 |
| `-h` / `--help` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 显示帮助信息后退出。在 `analyze` 后使用显示 scope 列表，在 `<scope>` 后使用显示该 scope 的参数说明。 |
| `-V` / `--version` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 显示版本信息后退出（顶层全局参数，可在任意子命令后使用，如 `msmodelslim analyze --version`）。 |

位置参数：

| 位置参数 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|----------|------|----------|-----------|--------|----------------|------|
| `scope` | `string` | 位置参数（子命令） | 可选 | 未指定且非帮助请求时注入 `linear` | `linear`、`layer`、`attn`、`attn_head` | 分析粒度。省略时默认 `linear`；请求帮助（`-h`/`--help`）时不注入。 |

`linear` 专有参数：

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--metrics` | 无 | `string` | 单值 | 可选 | `kurtosis` | `std`、`quantile`、`kurtosis` | 线性层敏感度指标：`std`（标准差）、`quantile`（分位数）、`kurtosis`（峰度）。 |
| `--patterns` | `--pattern`（遗留别名） | `list` | 可接收单个或多个值，空格分隔 | 可选 | `['*']` | 模块名称列表，支持通配符；`*` 表示全部 | 过滤要展示的线性层；`*` 表示全部。 |

`layer` 专有参数：

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--metrics` | 无 | `string` | 单值 | 可选 | `mse_layer_wise` | `mse_model_wise`、`mse_layer_wise` | 层级敏感度指标：按层计算误差或按模型整体计算误差。 |
| `--quant_modules` | 无 | `list` | 可接收单个或多个值，空格分隔 | 可选 | `['*']` | 模块名称列表，支持通配符；`*` 表示全部 | 层级分析中纳入量化流水线（pipeline）量化范围的模块；`*` 表示全部。 |

`attn` 专有参数：

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--metrics` | 无 | `string` | 单值 | 可选 | `mse` | `mse` | Attention 模块敏感度指标，仅支持 `mse`，作用于全部 Attention 模块。 |

`attn_head` 专有参数：

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--metrics` | 无 | `string` | 单值 | 可选 | `ra_compress` | `ra_compress` | Attention Head 筛选指标，仅支持 `ra_compress`；仅适用于 LLM，不支持多模态理解 / 多模态生成。 |

## 4. 参数关系

- `scope` 决定可用的 `--metrics` 取值与专有参数；跨 scope 混用（如 `analyze attn --patterns`）会导致参数解析失败，提示不支持的参数或取值。
- 省略 `scope` 时默认按 `linear` 执行；`msmodelslim analyze -h`/`--help` 时不注入 scope，展示 scope 帮助。
- `--model_type` 传入的名称不存在时回退到默认适配器 `default` 执行。
- `--device` 接受 `npu`/`cpu`；多卡分析使用 `--device npu --device_id 0 1 ...`（长度 > 1 启用 `DPLayerWiseRunner`）。兼容遗留写法 `--device npu:0,1`，会自动转换为上述推荐形式。
- `--save_path` 为全部 scope 通用的公共参数：`linear`/`layer`/`attn` 输出分析结果 YAML，传入目录时在该目录下生成 `<模型类型>-<指标>.yaml`；`attn_head` 输出 `head.pt`，传入目录时生成 `head.pt`。未指定时结果仅输出到控制台。
- 历史用法 `--metrics attention_mse`（省略 scope 时）会转换为 `analyze attn --metrics mse` 并丢弃 `--patterns`；该行为仅为向后兼容，命令会给出废弃提示，不推荐作为正式用法。
- `--top_k` 必须为大于0的整数；`--calibration_dataset`：LLM 为 `.json`/`.jsonl`，VLM 可为图文目录名（如 `calibImages`）；对 `attn_head` 不生效。
- `--save_path` 对四个 scope 均可用，落盘内容随 scope 不同：`linear`/`layer`/`attn` 输出量化 YAML（`top <n>:` 加 `disable_names` 列表，`layer` scope 的整块回退名为 `<block>.*`），指定目录时文件名为 `<model_type>-<metrics>.yaml`；`attn_head` 输出 `head.pt`（含 `prefix_matching`/`copying` 两个字典），交付给后续 MindIE 推理框架的长序列 KV Cache 压缩能力使用，不写入量化配置。目录不存在时会自动创建；不指定该参数时结果只打印到日志。
- `attn_head` / `ra_compress` 仅支持 LLM；多模态理解与多模态生成模型不适用。该场景不使用外部校准集，校准输入由算法在运行时自行构造，详见《[RA Compress 使用指南](../../knowledge_base/quantization_algorithms/ra_compress/usage_ra_compress.md)》。

## 5. 使用示例

### 5.1 最小可运行场景（线性层敏感度分析）

省略 scope，按默认 `linear` 执行：

```bash
msmodelslim analyze \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH}
```

`${MODEL_TYPE}` 为模型类型名称，`${MODEL_PATH}` 为权重目录。省略 `scope` 时默认按 `linear` 分析，使用默认指标 `kurtosis`、默认 `--top_k 15` 与默认校准集 `mix_calib.jsonl`，输出敏感度最高的15个层名，可写入量化 YAML 中 `linear_quant` 的 `exclude`。

### 5.2 指定 scope、指标与 topk

```bash
msmodelslim analyze linear \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --metrics kurtosis \
  --top_k 15
```

显式指定 `linear` scope，使用峰度指标，并按 `--top_k 15` 输出敏感度最高的15个层名。

### 5.3 Attention MSE 分析

```bash
msmodelslim analyze attn \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --metrics mse
```

对全部 Attention 模块计算 MSE 敏感度。

### 5.4 指定限定模块进行敏感层分析

```bash
msmodelslim analyze layer \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --metrics mse_layer_wise \
  --quant_modules ${MODULE_1} ${MODULE_2}
```

`${MODULE_1}`、`${MODULE_2}` 为纳入量化流水线量化范围的模块名称（支持通配符，空格分隔多个值）。

### 5.5 多卡 DP 敏感层分析

```bash
msmodelslim analyze layer \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --metrics mse_layer_wise \
  --device npu \
  --device_id 0 1 2 3 \
  --calibration_dataset mix_calib.jsonl
```

`--device_id` 指定多个 NPU 索引时启用 `DPLayerWiseRunner` 做多卡敏感层分析。兼容写法：`--device npu:0,1,2,3`（会自动拆分为 `--device npu --device_id 0 1 2 3`）。

### 5.6 注意力头分析

```bash
msmodelslim analyze attn_head \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --metrics ra_compress \
  --device npu \
  --save_path ./head_result
```

仅适用于大语言模型；多模态理解与多模态生成模型不支持。该场景无需指定 `--calibration_dataset`：算法会在运行时自行构造受控的随机校准输入（首 token + 随机重复段），外部校准集不参与计算。产出的 `head.pt` 交付给 MindIE 推理框架用于长序列 KV Cache 压缩：用户可据此完成长序列压缩，压缩后的模型可用于后续推理部署；具体已适配的量化模型列表，请参见《[MindIE 是什么](https://www.hiascend.com/document/detail/zh/mindie/10RC3/whatismindie/mindie_what_0001.html)》的“MindIE 支持模型列表”章节。该产物不写入量化配置。校准输入约定详见《[RA Compress 注意力头分析使用指南](../../knowledge_base/quantization_algorithms/ra_compress/usage_ra_compress.md)》。

## 6. 退出码与异常处理

正常运行则无异常，按输出的层名更新量化配置（如写入 `exclude`）；`attn_head` 场景检查 `head.pt`。失败时抛出异常。常见原因：scope 与 `--metrics` 取值不匹配、`--device`/`--device_id` 取值非法、校准集格式不符、`--top_k` 非大于0的整数、模型适配器未实现分析接口、多模态生成模型被拦截、`ra_compress` 用于非 LLM。

## 7. 安全说明

- `--trust_remote_code true` 会执行模型目录中的自定义 Python 代码，仅在确认来源可信时开启。
