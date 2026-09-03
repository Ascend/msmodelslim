---
name: evaluation-cfg
description: Generate msmodelslim evaluation YAML configuration (service_oriented + aisbench + vllm-ascend). Use when user asks for evaluation config generation.
license: Apache-2.0
metadata:
  version: 0.9.5
  domain: quantization
  framework: msmodelslim
  aliases:

    - msmodelslim-evaluation-config
    - evaluation-yaml

  trigger_intents:

    - 生成评测配置
    - 写测评yaml
    - 评测配置怎么写

  keywords:

    - evaluation config
    - aisbench
    - vllm-ascend
    - service_oriented

  # subagent 绑定声明：id = 委派标识（subagent_type）；bind = 委派时须绑定加载的 skill 根目录（含 references/、scripts/）
  subagent:
    id: "evaluation/evaluation-cfg"
    bind:

      - "evaluation/evaluation-cfg"

---

# 评测配置生成（evaluation-cfg）

## 概述

本 Skill 负责生成 `evaluation/evaluate` / `run_evaluation.py` 所需的单文件评测 YAML 配置。本 Skill 仅生成 Evaluation YAML。不要据此推断 Practice YAML 的 spec.save 应使用 `compressed_tensors`；Practice 的 save 以 strategy/practice-cfg 为准，默认情况下使用 `ascendv1_saver`。

**核心功能**：

- 生成包含 `demand`（目标精度）、`evaluation`（AISBench）、`inference_engine`（vLLM-Ascend）三个模块的完整 YAML
- 确保三个模块之间的字段保持一致（模型名、服务地址、端口）

**不适用**：

- 生成其他类型的配置（如量化策略配置）
- 执行评测或分析评测结果等配置生成以外的任务

**模板参考**：[evaluation_config.example.yaml](assets/evaluation_config.example.yaml)

**权威参考**：Evaluation YAML 的完整 schema 与字段定义以《[evaluation_service_oriented 配置](../../../docs/zh/api_reference/config/tuning/evaluation_service_oriented.md)》为准；本文只保留编排特有的生成规则（config_name 查找、VLM 图片路径处理、一致性检查）。

## 输入

执行时从上下文中提取以下信息：

| 参数 | 说明 | 缺省时默认值 |
|------|------|--------|
| 模型名称 | 量化后的模型标识符 | 从上下文获取 |
| 服务地址 | 推理服务 host | `localhost` |
| 服务端口 | 推理服务 port | `8000` |
| 设备类型 | 推理后端设备 | `ascend` |
| 设备索引 | 用户选择的物理设备索引列表，如 `[7]` | 必须从上下文获取 |
| 目标数据集 | 要评测的数据集列表 | 从上下文获取 |
| 精度目标 | 每个数据集的目标精度百分比 | 从上下文获取 |
| 精度容差 | 允许的精度波动范围 | 从上下文获取 |
| `allowed_local_media_path` | VLM 路径任务的显式覆盖目录 | `null`；优先从数据集 README 自动推导 |

## 文件生成规则

### 文件生成步骤

1. 在工作目录生成一个 YAML 文件，包含以下结构：

```yaml
type: service_oriented

demand:
  expectations:
    - dataset: <数据集名称>
      target: <目标精度>
      tolerance: <容差>

evaluation:
  type: aisbench
  precheck: [...]  # 可选
  aisbench: { ... }
  datasets:
    <数据集名称>:
      config_name: <ais_bench 注册名>
      # ...
  host: <服务地址>
  port: <服务端口>
  served_model_name: <模型名称>

inference_engine:
  type: vllm-ascend
  env_vars:
    ASCEND_RT_VISIBLE_DEVICES: <设备索引以逗号连接后的字符串>
  served_model_name: <模型名称>
  host: <服务地址>
  port: <服务端口>
  args: { ... }
```

1. 文件生成后，执行文件检查。如果未通过检查，需要修正后重新生成，直到生成的文件满足所有要求。
2. 如果用户提供了参考的测评配置，则尽可能地按照用户的配置。如果进行了修改，则需要向用户回显该修改，给出简要的原因解释，但不必中断流程向用户确认。
3. 文件生成并验证通过后，返回文件路径。

### 关键必填字段填写要求

| 路径 | 类型 | 说明 |
|------|------|------|
| `type` | string | 必须为 `service_oriented` |
| `demand.expectations` | list | 至少包含一项 |
| `demand.expectations[].dataset` | string | 必须存在于 `evaluation.datasets` |
| `demand.expectations[].target` | float | 必须 > 0 |
| `demand.expectations[].tolerance` | float | 必须 ≥ 0 |
| `evaluation.type` | string | 必须为 `aisbench` |
| `evaluation.aisbench.request_rate` | float | 每秒发送的请求数，必须大于 `0`；< 0.001 为不限速发送，默认设为 `0.0001` |
| `evaluation.datasets` | dict | 必须非空 |
| `evaluation.datasets.*.config_name` | string | AISBench 注册名；输入未指定时按[注册名查找与任务选择](references/how_to_find_aisbench_config_name.md)确定 |
| `evaluation.host` | string | 与 `inference_engine.host` 保持一致 |
| `evaluation.port` | int | 与 `inference_engine.port` 保持一致 |
| `evaluation.served_model_name` | string | 与 `inference_engine.served_model_name` 保持一致 |
| `inference_engine.type` | string | 必须为 `vllm-ascend` |
| `inference_engine.env_vars.ASCEND_RT_VISIBLE_DEVICES` | string | 用户选择的物理设备索引，以逗号连接；如 `device_indices=[7]` 时填写 `"7"` |
| `inference_engine.args.served-model-name` | string | 与 `served_model_name` 保持一致 |
| `inference_engine.args.tensor-parallel-size` | int | 等于 `device_indices` 的元素数量 |

### VLM 字段填写参考

VLM 测评仍然生成同一类 `service_oriented + aisbench + vllm-ascend` YAML，不新增顶层结构。与 LLM 配置相比，需要考虑以下字段：

| 路径 | 类型 | 说明 |
|------|------|------|
| `evaluation.aisbench.max_out_len` | int | 取值优先级：模型卡 / 数据集官方配置 / 用户明确给出 > 与 `max-model-len` 的余量约束；**无明确依据时不得拍脑袋填 32768**（见「长度配置一致性规则」，且必须 `< max-model-len`） |
| `evaluation.aisbench.batch_size` | int | 请求最大并发数；无通用默认值，信息不足时使用 `16` |
| `inference_engine.args.max-model-len` | int | VLM 默认 `65536`；模型、数据集或用户明确指定时使用指定值，且不能超过模型上限 |
| `inference_engine.args.max-num-batched-tokens` | int | 默认取 `min(max-model-len, 33792)`；显存充足但吞吐不足时再提高 |
| `inference_engine.args.allowed-local-media-path` | string | 仅图片路径任务填写；必须是经过校验的可信绝对目录 |

### VLM 图片输入处理

1. 按[注册名查找与任务选择](references/how_to_find_aisbench_config_name.md)确定或校验 `config_name`，取得 `media_input_type` 和可选的 `candidate_local_media_path`。
2. `media_input_type` 为 `text` 或 `base64` 时，不生成 `allowed-local-media-path`。
3. `media_input_type` 为 `local_path` 时，优先使用显式传入的 `allowed_local_media_path`，否则使用 `candidate_local_media_path`。
4. 将选定目录规范化为绝对路径，并确认它已存在、是目录、可被本次 vLLM 服务访问，且任务运行时发送的媒体文件位于该目录下。禁止使用 `/` 等过宽目录；多个路径任务必须共享一个安全的可信根目录。
5. 校验通过后写入 `inference_engine.args.allowed-local-media-path`。没有候选目录或校验失败时不得生成 YAML；返回 `VALIDATION_ERROR`，说明数据集、`selected_config_name`、候选路径和失败原因，请主 Agent 向用户确认后通过 `allowed_local_media_path` 重试。

### 长度配置一致性规则（LLM/VLM 通用，硬门禁）

vLLM 约束 `prompt + output ≤ context`：若 `evaluation.aisbench.max_out_len ≥ inference_engine.args.max-model-len`，服务会拒绝**全部**请求（`VLLMValidationError`），评测出 `accuracy=0.0`，且会被误判为达标基线（曾实测发生，见 F0-5）。因此：

- **必须满足**：`evaluation.aisbench.max_out_len < inference_engine.args.max-model-len`，建议 `max_out_len ≤ max-model-len − 512` 预留 prompt 余量。
- `max_out_len` 取值优先级：模型卡 / 数据集官方配置 / 用户明确给出 > 与 max-model-len 的余量约束。无明确依据时不得拍脑袋填 32768。
- LLM 的 `max-model-len` 默认锚点：读模型目录 `config.json` 的 `max_position_embeddings`（如 Qwen3-4B = 32768）；该字段缺失时以模型卡 / README 声明为准。
- 生成配置后必须复核该关系，不满足即修正（见「文件检查步骤」第 6条），**禁止**输出不满足此关系的 YAML。

**取值冲突裁决顺序**：`max-model-len` 与 `max_out_len` 存在多个候选来源时按以下优先级，并在回传 `derivation` 中说明采用了哪条：

1. **用户参考配置 / 已实测跑通的配置（一致性优先）**：量化评测与浮点评测必须同口径（尤其 `evaluation.aisbench` 与 `max-model-len`），以参考配置为准——即使模型 `config.json` 的 `max_position_embeddings` 更大（如 40960 > 参考的 32768），仍沿用参考值以保证 FP/量化可比；
2. **模型 `config.json` 的 `max_position_embeddings`**（无参考配置时）；
3. **模型卡 / README 声明**。

仅当用户明确需要更长上下文（参考值成为瓶颈）时，才允许偏离参考值，且必须在回传中显式说明偏离与理由，由编排层确认。

### 文件检查步骤（直接检查即可，无需写检查脚本）

1. 确保所有必填字段存在且符合格式要求
2. 确保生成的 YAML 文件语法正确，可以被 YAML 解析器成功解析
3. 如果你在测浮点模型精度基线，则 `demand.expectations[].target` 和 `demand.expectations[].tolerance` **必须**都设置为 100 进行占位。
4. 确保测评配置一致性，你应确保测评浮点权重和量化权重的配置的通用参数一致，尤其是 `evaluation.aisbench`、`inference_engine.args.max-model-len`**必须**保持一致。在不一致的情况下，你应该修改当前生成的配置文件。例如先前生成了浮点的测评配置且已经测评过了，则你应该修改当前生成的量化测评配置。
5. 检查 `inference_engine.env_vars.ASCEND_RT_VISIBLE_DEVICES` 与用户选择的 `device_indices` 完全一致，且 `tensor-parallel-size` 等于设备数量。
6. **长度门禁（LLM/VLM 通用）**：检查 `evaluation.aisbench.max_out_len` **必须 <** `inference_engine.args.max-model-len`（见「长度配置一致性规则」）。不满足则修正后重新生成，**禁止**输出不满足此关系的配置。
7. 对 VLM 配置，按“VLM 图片输入处理”完成任务选择和媒体路径校验。

## 执行约束

**绝对禁止**：

- 不得阅读任何源码文件
- 不得使用向量检索 / 语义检索工具搜索配置；信息以本 Skill 引用的模板、references 及允许读取的文档为准

**允许**：

- 使用 `assets/evaluation_config.example.yaml` 作为模板
- 读取本 Skill 直接引用的 `references/` 文件
- 读取 AISBench 数据集 README、模型官方 README/模型卡，以及模型目录中的 `config.json`、`generation_config.json`
- 读取用户提供的已有 Evaluation YAML、AISBench 生成配置、summary、prediction 和服务日志，用于确认推理模式、长度截断与资源参数；只读分析，不修改这些评测产物

## 常见错误

| 错误类型 | 描述 | 修复方法 |
|----------|------|----------|
| 数据集未统一 | `expectations` 中的 dataset 不在 `datasets` 中 | 同步添加或删除 |
| 服务地址不一致 | `evaluation` 与 `inference_engine` 的 host/port 不统一 | 统一设置 |
| 模型名不一致 | `served_model_name` 在三处不统一 | 统一设置 |
| 命名规则错误 | `args` 内使用了 snake_case 而非 kebab-case | 转换为 kebab-case（如 `served_model_name` → `served-model-name`） |
| 配置名错误 | `config_name` 与 ais_bench 注册名不匹配 | 查询正确的注册名 |
| 长度越界 | `evaluation.aisbench.max_out_len ≥ inference_engine.args.max-model-len` → vLLM 拒绝全部请求、accuracy=0.0 | 按「长度配置一致性规则」修正 `max_out_len`（留 prompt 余量） |
| VLM 图片输入方式错误 | 任务输入方式与服务能力不匹配，或路径任务缺少可信媒体根目录 | 按 VLM 图片输入处理流程选择任务或返回主 Agent 确认路径 |

## 经验条目（Experiences，追加制）

> 追加规范见 `skills/README.md`「经验条目」。连续编号 `[E-序号]`；正文保留权威展开，本表只做索引 + 元数据登记；来源：实测编号（F0-x / Dx / Tx）| 用户反馈 | 代码实证；验证状态：已回归 | 待验证 | 已上流 docs。

| 条目 | 主题 | 适用条件 / 触发信号 | 结论要点（一句话） | 正文位置 | 来源 | 验证状态 |
|------|------|------|------|------|------|------|
| E-001 | 长度配置一致性门禁 | 生成/修改任意评测配置 | `max_out_len ≥ max-model-len` → vLLM 拒绝**全部**请求、accuracy=0.0 且会被误判达标基线（曾实测）；必须 `max_out_len < max-model-len`，建议留 512 余量 | 长度配置一致性规则 | 实测 F0-5 | 已回归 |
| E-002 | max-model-len 默认锚点 | LLM 配置且用户未给明确值时 | 读模型目录 `config.json` 的 `max_position_embeddings`（如 Qwen3-4B=32768）；字段缺失以模型卡/README 为准 | 长度配置一致性规则 | 代码实证 | 已回归 |
| E-003 | 取值冲突裁决顺序 | `max-model-len` / `max_out_len` 存在多候选来源 | 用户参考/已实测跑通配置（FP/量化同口径一致性优先）> config.json > 模型卡；仅用户明确需更长上下文才偏离并在 `derivation` 显式说明 | 取值冲突裁决顺序 | 用户反馈 | 已回归 |
| E-004 | FP/量化配置同口径 | 量化评测与浮点基线并存时 | `evaluation.aisbench` 与 `inference_engine.args.max-model-len` 必须一致，不一致时修改当前生成的文件（浮点已测则改量化配置） | 文件检查步骤 4 | 实测 F0-5 | 已回归 |
| E-005 | VLM 媒体路径安全 | `media_input_type=local_path` | 禁止 `/` 等过宽目录；多路径任务共享一个安全可信根目录；无候选/校验失败返回 `VALIDATION_ERROR` 交由用户确认，不硬出 YAML | VLM 图片输入处理 | 代码实证 | 待验证 |
