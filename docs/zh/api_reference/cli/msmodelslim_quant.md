# msmodelslim quant 命令行 API 文档

## 1. 功能说明

`msmodelslim quant` 是一键量化命令，加载原始模型权重并执行权重/激活量化，导出可部署的量化权重与描述文件。配置来源有两种：通过 `--quant_type` 按模型与量化类型自动匹配 `lab_practice` 中的最佳实践 YAML；或通过 `--config` 直接指定用户 YAML（支持 `modelslim_v1`、多模态以及 `modelslim_convert` 纯权重转换等协议。

命令边界：设备支持 `npu`、`cpu`，多卡通过 `--device_id` 指定索引列表；还支持场景标签匹配与 `--debug` 调试上下文落盘。校准数据准备与部署等操作步骤见《[一键量化完整指南](../../user_guide/usage_quick_quantization.md)》；量化任务 YAML 字段见《[量化任务配置说明导航](../config/quant/README.md)》，自动调优配置见《[自动调优配置说明导航](../config/tuning/README.md)》。更多关联项见[引用的配置](#5-引用的配置)。

## 2. 命令格式

```text
msmodelslim quant [--model_type <model_type>] --model_path <model_path> --save_path <save_path> [--device <device>] [--device_id <id> ...] [--config <config> | --quant_type <quant_type>] [--trust_remote_code [<BOOL>]] [--debug] [--tags <tag> ...] [--log_level <level>] [-v] [-q]
```

符号说明：

- 尖括号内为需替换的值，方括号内为可选参数。
- `--config` 与 `--quant_type` 属于互斥组，用 `|` 表示，二者不能同时传入。
- `--trust_remote_code` 是布尔参数：不带值传入即开启（等价 `true`），也可显式传入 `true`/`false`（大小写不敏感，兼容遗留写法 `True`/`False`）。
- `--debug` 是不带值的布尔开关。
- `--tags` 可接收单个或多个值，`...` 表示可跟多个以空格分隔的值。
- 本命令无位置参数，全部参数通过选项传入。本语法摘要用于说明参数结构，不作为可复制命令；可复制命令见[使用示例](#7-使用示例)。

## 3. 参数列表

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--model_type` | 无 | `string` | 单值 | 条件必选（普通量化路径必选；`--config` 指向 `apiversion: modelslim_convert` 的配置时可省略） | 无 | 模型类型名称，如 `Qwen2.5-7B-Instruct`、`Qwen3-32B`、`transformers` | 指定待量化模型类型，用于加载对应模型适配器并匹配最佳实践；传入的名称不存在时回退到默认适配器 `default` 执行。`transformers` 用于加载基于 transformers 的通用基础模型适配器；仅当 `--config` 指向 `apiversion: modelslim_convert` 的配置时可省略。 |
| `--model_path` | 无 | `string` | 单值 | 必选 | 无 | 原始模型权重目录（需存在且可读） | 待量化模型的权重目录。权重转换场景仅需 `--model_path` 与 `--save_path`。 |
| `--save_path` | 无 | `string` | 单值 | 必选 | 无 | 输出目录（需可写） | 量化权重与描述文件的保存目录。权重转换场景仅需 `--model_path` 与 `--save_path`。 |
| `--device` | 无 | `string` | 单值 | 可选 | `npu` | `npu`、`cpu` | 运行设备类型；多卡索引请用 `--device_id` 指定。 |
| `--device_id` | 无 | `list` | 可接收单个或多个值，空格分隔 | 可选 | 无 | 非负整数列表，如 `0` 或 `0 1 2 3` | 设备索引；用于指定多个 NPU 设备。 |
| `--config` | `-c`、`--config_path`（遗留别名） | `string` | 单值 | 可选 | 无 | 可读 YAML 文件路径 | 显式指定的量化配置 YAML；加载后直接采用，并忽略 `--quant_type` 与 `--tags` 的最佳实践匹配。与 `--quant_type` 互斥，`--config` 与 `--quant_type` 两者都不传时按默认量化类型 `w8a8` 匹配最佳实践。 |
| `--quant_type` | 无 | `string` | 单值 | 可选 | 无（与 `--config` 均未提供时按 `w8a8` 匹配） | `w4a4`、`w4a8`、`w4a4c8`、`w4a4f8`、`w4a8c8`、`w8a16`、`w8a8`、`w8a8s`、`w8a8c8`、`w8a8f8`、`w4a4f4`、`w16a16s` | 量化类型，用于按模型与量化类型匹配最佳实践 YAML。与 `--config` 互斥，`--config` 与 `--quant_type` 两者都不传时按默认量化类型 `w8a8` 匹配最佳实践。 |
| `--trust_remote_code` | 无 | `bool` | 不带值传入即开启，也可显式传值 `true`/`false` | 可选 | `false` | `true`/`false`（大小写不敏感；兼容遗留写法 `True`/`False`） | 是否信任并执行模型目录中的自定义 Python 代码；仅在确认代码来源可信时开启。 |
| `--debug` | 无 | `bool` | 不带值开关 | 可选 | `False` | 传入即启用 | 启用调试模式，将量化中间上下文写入 `${SAVE_PATH}/debug_info/`（`debug_info.json` 与 `debug_info.safetensors`）。 |
| `--tags` | `--tag`（遗留别名） | `list` | 可接收单个或多个值，空格分隔 | 可选 | 无 | 场景标签，须与最佳实践 `verified_tags` 中的标签名一致（大小写不敏感），如 `vLLM-Ascend`、`Atlas_A2_Inference`、`MindIE` | 匹配带已验证场景标签的最佳实践；多个标签须同时出现在同一场景中，未提供硬件类型标签时自动匹配当前设备类型。 |
| `--log_level` | 无 | `string` | 单值 | 可选 | `info` | `debug`、`info`、`warning`、`error` | 日志级别。 |
| `-v` / `--verbose` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 提高输出详细程度（等价 `--log_level debug`）。 |
| `-q` / `--quiet` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 抑制非错误输出（等价 `--log_level error`）。 |
| `-h` / `--help` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 显示帮助信息后退出（如 `msmodelslim quant -h`）。 |
| `-V` / `--version` | 无 | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 显示版本信息后退出（顶层全局参数，可在任意子命令后使用，如 `msmodelslim quant --version`）。 |

## 4. 参数关系

- `--config` 与 `--quant_type` 互斥，同时传入会报错。
- 两者都不传是受支持的用法：按默认量化类型 `w8a8` 匹配最佳实践；未匹配到最佳实践时会给出提示并等待确认（输入 `Y`/`y` 继续，输入其他内容则退出）。该提示为交互式输入，在无标准输入的非交互环境下会一直等待，建议改用 `--config` 或显式指定 `--quant_type`。
- 指定 `--config` 后直接采用该配置，`--quant_type` 与 `--tags` 的最佳实践匹配均被忽略。
- `--model_type` 在普通量化路径下必须提供；仅当 `--config` 指向 `apiversion: modelslim_convert` 的配置时可省略，此时为纯权重转换，无需加载模型代码，`--model_type` 及量化相关参数均不生效。
- `--model_type` 传入的名称不存在时回退到默认适配器 `default` 执行。
- `--model_type` 指定 `transformers` 时，若使用 `--quant_type`，仅支持 `w8a16`/`w8a8`；若使用 `--config`，当前仅支持大语言模型量化配置（`apiversion: modelslim_v1`），多模态理解（`multimodal_vlm_modelslim_v1`）与多模态生成（`multimodal_sd_modelslim_v1`）配置会被拒绝。
- `--tags` 指定多个值时须同时出现在同一已验证场景；未提供硬件类型标签时自动匹配当前设备类型。
- `--debug` 启用后量化上下文写入 `${SAVE_PATH}/debug_info/`。

## 5. 引用的配置

协议总览见《[量化任务配置说明导航](../config/quant/README.md)》；若配置用于自动调优模板，另见《[自动调优配置说明导航](../config/tuning/README.md)》。

| 关联参数 | 配置名称 | 引用关系 | 配置文档 |
|----------|----------|----------|----------|
| `--config` | `modelslim_v1` | 加载整份量化 YAML | 《[modelslim_v1 配置说明](../config/quant/modelslim_v1.md)》 |
| `--config` | `multimodal_vlm_modelslim_v1` | 多模态理解模型量化 YAML | 《[multimodal_vlm_modelslim_v1 配置说明](../config/quant/multimodal_vlm_modelslim_v1.md)》 |
| `--config` | `multimodal_sd_modelslim_v1` | 多模态生成模型量化 YAML | 《[multimodal_sd_modelslim_v1 配置说明](../config/quant/multimodal_sd_modelslim_v1.md)》 |
| `--config` | `modelslim_convert` | 权重转换 YAML；不需要 `--model_type` | 《[modelslim_convert 配置说明](../config/quant/modelslim_convert.md)》 |
| `--quant_type` | `lab_practice` 最佳实践 YAML | 按模型与量化类型匹配 | 《[一键量化完整指南](../../user_guide/usage_quick_quantization.md)》 |

## 6. 环境变量

| 环境变量 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|----------|------|-----------|--------|----------------|------|
| `MSMODELSLIM_CUSTOM_PRACTICE_REPO` | `string` | 可选 | 无（未设置） | 目录路径 | 自定义最佳实践仓库目录；设置后与官方 `lab_practice` 一并参与最佳实践匹配。 |
| `MSMODELSLIM_LOG_LEVEL` | `string` | 可选 | `INFO` | `INFO`、`DEBUG` | 设置日志级别；设置后打印同级及以上日志。 |

## 7. 使用示例

### 7.1 按默认量化类型一键量化（最小可运行场景）

只包含完成一次量化所需的最少参数：

```bash
msmodelslim quant \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH}
```

`${MODEL_TYPE}` 为模型类型名称（如 `Qwen2.5-7B-Instruct`），`${MODEL_PATH}` 为浮点权重目录，`${SAVE_PATH}` 为量化输出目录。未指定 `--quant_type` 与 `--config` 时，默认按量化类型 `w8a8` 匹配最佳实践并执行量化；若模型加载需要模型目录内的自定义代码，再补充 `--trust_remote_code true`。

### 7.2 使用 transformers 通用基础模型适配器量化（仅适用于大语言模型）

```bash
msmodelslim quant \
  --model_type transformers \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --quant_type w8a16
```

### 7.3 显式指定量化类型与设备

```bash
msmodelslim quant \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --quant_type w8a8c8 \
  --device npu \
  --device_id 0 1 2 3
```

`--quant_type w8a8c8` 表示权重8bit、激活8bit、KVCache 8bit 量化；`--device_id 0 1 2 3` 使用 4 个 NPU 设备。权重转换的命令见 [7.7](#77-显示指定设备进行权重转换)。

### 7.4 使用自定义配置文件

```bash
msmodelslim quant \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config ${CONFIG_PATH}
```

`${CONFIG_PATH}` 指向符合 V1 协议的量化 YAML；指定后直接采用该配置，不再做最佳实践匹配。字段说明见引用的配置文档。权重转换示例见 [7.7](#77-显示指定设备进行权重转换)。

### 7.5 使用场景标签匹配最佳实践

```bash
msmodelslim quant \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --quant_type w8a8 \
  --tags MindIE Atlas_A2_Inference
```

`--tags` 后的多个标签须同时出现在同一已验证场景中；未精确匹配时命令会给出提示并等待确认（输入 `Y`/`y` 继续，输入其他内容则退出）；命中备用（standby）配置时提示改用备用配置，仍需用户确认。

### 7.6 纯权重转换（不执行量化）

`--config` 指向 `apiversion: modelslim_convert` 的配置时，仅做权重格式转换，无需 `--model_type` 和 `--quant_type`：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config ${CONFIG_PATH}
```

`${MODEL_PATH}` 为源权重目录，`${SAVE_PATH}` 为转换输出目录，`${CONFIG_PATH}` 为转换 YAML。配置协议与字段见《[modelslim_convert 配置说明](../config/quant/modelslim_convert.md)》，操作步骤与示例 YAML 见《[权重转换使用指南](../../knowledge_base/ptq/convert/usage_weight_conversion.md)》。

### 7.7 显示指定设备进行权重转换

NPU 多卡时加上 `--device npu` 和 `--device_id`：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config ${CONFIG_PATH} \
  --device npu \
  --device_id 0 1 2 3
```

## 8. 退出码与异常处理

正常运行则无异常，`${SAVE_PATH}` 下生成量化权重与描述文件。失败时抛出异常。常见原因：`--config` 与 `--quant_type` 同时传入、普通量化路径缺少 `--model_type`、YAML 校验失败、`--device_id` 索引非法或超出可用设备。

## 9. 安全说明

- `--trust_remote_code true` 会执行模型目录中的自定义 Python 代码，仅在确认来源可信时开启。
- `--save_path` 会写入量化结果文件；请确认目录可写且不会覆盖非预期数据。
- `--debug` 会把量化中间上下文（含张量数据）写入 `${SAVE_PATH}/debug_info/`，请按组织安全策略管理该目录。
