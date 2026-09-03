---
name: sensitive-layer-analysis
description: 敏感层分析执行。调优闭环中通过 msmodelslim analyze layer 计算各 Decoder Block/分析单元的量化敏感度得分，产出 analysis_result.yaml 供回退层选择与后续各轮复用；可由 strategy/practice-cfg 步骤②再委派（嵌套委派），也可独立触发直接分析。
license: Apache-2.0
metadata:
  version: 0.1.0
  domain: quantization
  framework: msmodelslim
  protocol: cli
  skill_class: tool
  aliases:

    - sensitivity-analysis
    - layer-sensitivity

  trigger_intents:

    - 敏感层分析
    - 分析哪些层敏感
    - 生成敏感层排序

  keywords:

    - msmodelslim analyze
    - mse_layer_wise
    - sensitive layer
    - 敏感层
    - analysis_result.yaml

  # subagent 绑定声明：id = 委派标识（subagent_type）；bind = 委派时须绑定加载的 skill 根目录（含 references/、scripts/）
  # sensitive-layer-analysis 依赖 installation：msmodelslim CLI 可用性校验
  subagent:
    id: "sensitive-layer-analysis"
    bind:

      - "sensitive-layer-analysis"
      - "installation"

---

# 敏感层分析（sensitive-layer-analysis）

## 职责

通过 `execute` 调用 **msmodelslim CLI** 的 `analyze layer`，计算各分析单元的量化敏感度得分，将结果写入 `{save_path}/analysis_result.yaml`（score 降序排列），供回退层选择使用。

**不解决什么**：

- 不生成 / 修改 Practice YAML → `strategy/practice-cfg`
- 不执行量化 → `quantization`
- 不执行评测 → `evaluation/evaluate`
- 不做策略终止决策 → `tuning`

**权威参考**（CLI 参数、命令格式、算法原理等以 docs 为准，不在此重复）：

- `msmodelslim analyze` 参数与命令 → 《[msmodelslim analyze CLI](../../docs/zh/api_reference/cli/msmodelslim_analyze.md)》
- 层级敏感层分析完整流程 → 《[层级敏感层分析使用指南](../../docs/zh/user_guide/usage_sensitive_layer_analysis.md)》
- 分析算法与 metric 语义 → 《[量化算法总览](../../docs/zh/knowledge_base/quantization_algorithms/README.md)》

闭环使用 `layer` scope。

## 使用导引（直接使用）

用户要求「敏感层分析 / 分析哪些层量化敏感 / 生成敏感层排序」时，按以下路径独立执行，不依赖调优编排：

| 用户输入 | 执行路径 |
|----------|----------|
| 已有基准 Practice | 从基准 Practice 的 `spec.process[type=linear_quant].include` 读取分析范围，按下方契约执行 `msmodelslim analyze layer` |
| 无基准 Practice | 分析范围缺省 `["*"]`，直接执行；如需回退层选择建议，结合支持矩阵与量化调优策略 |

执行前须校验 `msmodelslim analyze --help` 可正常执行（环境不满足时按已绑定的 `installation` skill 内容在本会话内处理）。本 skill 为**叶子 subagent**：被委派后只执行分析，不再向下委派（链深上限 2，见 `subagent_io_protocol.md`「子任务再委派」）。

## 委派契约（SUBAGENT_IO v1）

本 skill 支持两种委派方：

1. **主 Agent 直接委派**（独立使用/循环前一次性分析）；
2. **`strategy/practice-cfg` 再委派**（嵌套委派，见 `subagent_io_protocol.md`「子任务再委派」）：practice-cfg 在其步骤②检查 `analysis_result.yaml` 是否需要分析，需要时以本契约为准发起再委派，并把其 `input`（`model_type`/`model_path`/`save_path`/`device`/`calib_dataset`）与本 skill 的 input 对齐后透传，`practice_path` 指向 practice-cfg 步骤①产出的基准 Practice。

委派本 subagent 时，`input` 按下表填写；`commands` 与信封格式见 `tuning/references/subagent_io_protocol.md`。

| input 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | ✓ | msModelSlim 注册的模型适配器名称 |
| `model_path` | string | ✓ | 模型路径 |
| `save_path` | string | ✓ | 工作目录；分析报告写入 `{save_path}/analysis_result.yaml` |
| `device` | string | ✓ | 设备，如 `npu:0`；EP 多卡场景按编排层给定列表 |
| `calib_dataset` | string \| null | | 校准数据集覆盖值；缺省时按闭环约定解析（见下） |
| `practice_path` | string \| null | | 基准 Practice YAML 路径；提供时从中读取 `include` 作为分析范围 |

**回传 `output`**：`analysis_result_path`（`{save_path}/analysis_result.yaml`）、`success`（bool）、`stderr`（失败时摘要）、`commands`。

## 执行步骤

### ① 参数与前置校验

- `model_type` / `model_path` / `save_path` / `device` 齐备且路径存在可读；
- `msmodelslim analyze --help` 可执行（环境不满足时按 `installation` skill 处理）；
- 若提供 `practice_path`，读取基准 Practice 的 `spec.process[type=linear_quant].include` 作为分析范围（`analysis_include_patterns`）；缺失或为空时使用 `["*"]`。

### ② 执行分析

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path "${model_path}" \
    --metrics mse_layer_wise \
    --calibration_dataset "${effective_calib_dataset}" \
    --quant_modules "${analysis_include_patterns[@]}" \
    --top_k 999 \
    --device npu \
    > "${save_path}/analysis_console.log" 2>&1
```

> 参数名以 docs 为准：校准数据集为 `--calibration_dataset`，敏感层数量为 `--top_k`（不要使用遗留的 `--calib_dataset` / `--topk` 写法）。

### ③ 解析并写出结果

CLI 将各层 Score 写入 `analysis_console.log`。解析后写入 `{save_path}/analysis_result.yaml`：

> **直出优先（若支持）**：若当前版本 `msmodelslim analyze layer --help` 提供 `--save_path`，可将其直接指向 `{save_path}/analysis_result.yaml` 由 CLI 直出结果。**必须核验直出文件结构与本节规范一致**（`layer_scores` 按 score 降序、含 `method`/`patterns`、层名不加通配符）；不一致时仍按本节的日志解析路径写出。实测中不同版本/适配器对 `--save_path` 的结构输出存在差异，未核验前不得假设与日志解析结果等同。

```yaml
layer_scores:
  - name: "model.layers.0"
    score: 12.5
  - name: "model.layers.15"
    score: 8.3
  # ... 按 score 降序排列
method: "mse_layer_wise"
patterns:
  - "*"
```

- `score` 越高，层越敏感，量化时越容易造成精度损失。
- `layer_scores[].name` 保存 ModelSlim 返回的原始 Decoder Block 名称，不在分析结果中预先追加通配符。
- 供 `linear_quant.exclude` 使用时，对 layer scope 的名称追加 `.*`，例如将 `model.layers.15` 转换为 `model.layers.15.*`；名称已以 `.*` 结尾时不得重复追加。

**成功判定**：exit code 为 0 且 `save_path` 下生成 `analysis_result.yaml`；失败 → 报命令名 + stderr 关键摘要，立即中止，不兜底续跑，等待编排层决策。

## 闭环关键约定

- `effective_calib_dataset`：优先使用显式传入的 `calib_dataset`；未传入时，`modelslim_v1` 使用 `mix_calib.jsonl`，`multimodal_vlm_modelslim_v1` 使用 `calibImages`。
- `analysis_include_patterns`：从基准 Practice 的 `spec.process[type=linear_quant].include` 读取；缺失或为空时使用 `["*"]`。多个模式必须作为同一个 `--quant_modules` 后的独立数组元素参数传递，不得拼接成一个字符串。
- 分析输出必须直接写入日志文件（`> file 2>&1`），不得依赖 `tee`；外层执行超时时先确认原分析进程是否仍在运行，**避免**上一个分析进程未结束时再次拉起一个新的分析进程。
- 使用 `mse_layer_wise` 时 `--top_k` 固定使用 `999`（覆盖当前模型适配器全部分析单元的兼容上限，不代表语言层数或模型实际总层数；分析单元可能包括语言 Decoder 层、视觉模块整体、多模态投影层、MTP 及适配器定义的其他单元）。
- 支持指标：`mse_layer_wise`（默认，逐块 MSE）、`mse_model_wise`（可选，整模型 MSE）。

## 结果在调优中的使用

敏感度得分在调优任务开始时计算一次，写入 `{save_path}/analysis_result.yaml`，各轮复用。每轮根据策略从预计算的得分排序中选择回退层，无需重新调用分析命令。

> **复用判定归属**：被 `strategy/practice-cfg` 再委派时，是否跳过分析（复用已有 `analysis_result.yaml`）由 practice-cfg 判断；本 skill 只负责在**实际执行分析**时产出结果并确保其结构符合「解析并写出结果」规范。若委派方传入的 `practice_path` 与已缓存结果不一致（基准 Practice 变化），本 skill 应在执行分析时覆盖旧结果。

选择回退层时需遵守**同分同退约束**：分析结果按 score 降序排列，分数相同的层作为一个整体（同分组），`--top_k` 参数选取的是前 K 个**同分组**而非前 K 个单独层。在调优过程中，同分组内的层必须同时回退或同时保留，不可拆分。

## 基准 Practice 范围约束

- 分析前读取基准 Practice，将 `spec.process[type=linear_quant].include` 作为 `--quant_modules`。
- 基准 Practice 中的静态 `exclude` 记录为 `protected_exclude`，包括视觉编码器、投影层及其他不参与自动调优的模块；它不参与二分搜索，任何轮次都不得删除。
- `multimodal_vlm_modelslim_v1` 使用的 `calib_dataset` 必须与后续 Practice 的 `spec.dataset` 解析到同一份多模态校准数据，并满足当前模型适配器的要求。

## 工具不可用时的经验规则

当 `msmodelslim analyze` 失败（非 0 exit code）或超时时，按以下步骤占位：

1. **获取语言 Decoder 层数 N**：从 `<model_path>/config.json` 读取 `num_hidden_layers`。嵌套 config 依次查顶层、`text_config`、`language_config`、`thinker_config.text_config` 等同名字段；该值仅用于构造语言层经验排序，不代表模型适配器的全部分析单元数量。
2. **构造经验排序**：层序上前 2-4层 + 后 2-4层视为更敏感，中间段相对低敏感。
3. **写出结果文件**：将经验排序按上方「解析并写出结果」的格式写入 `{save_path}/analysis_result.yaml`，确保后续步骤无需区分数据来源。

仅作占位，**弱于**精确分析。数据集、模型加载、schema 或参数错误必须立即失败返回，不得用经验规则掩盖。

## 错误处理

| 错误类型 | 处理 |
|----------|------|
| `msmodelslim` 未安装 | 按 installation skill 安装后重试（已绑定） |
| 路径不存在 / 配置解析失败 | 检查路径与参数后重试或中止 |
| 分析失败 / 超时 | 先确认原进程状态；确属失败再按经验规则占位（仅工具不可用时） |
| 数据集 / 模型 / schema 错误 | 立即失败返回，禁止经验规则掩盖 |

## 被 `tuning` 编排

调优编排中，本 skill 在「量化配置调优」阶段被 `strategy/practice-cfg` 步骤②**再委派**（嵌套委派，`commands.name = sensitive_layer_analysis`）：practice-cfg 判定需要分析时发起再委派，本 skill 产出 `analysis_result.yaml` 供各轮复用；不参与策略决策、Practice 写出与校验。

## 经验条目（Experiences，追加制）

> 追加规范见 `skills/README.md`「经验条目」。连续编号 `[E-序号]`；正文保留权威展开，本表只做索引 + 元数据登记；来源：实测编号（F0-x / Dx / Tx）| 用户反馈 | 代码实证；验证状态：已回归 | 待验证 | 已上流 docs。

| 条目 | 主题 | 适用条件 / 触发信号 | 结论要点（一句话） | 正文位置 | 来源 | 验证状态 |
|------|------|------|------|------|------|------|
| E-001 | 参数名以 docs 为准 | 执行 `analyze layer` 时 | 校准数据集用 `--calibration_dataset`、层数用 `--top_k`，勿用遗留 `--calib_dataset` / `--topk` | ② 执行分析 | 代码实证 | 已回归 |
| E-002 | `--top_k` 固定 999 | 使用 `mse_layer_wise` 指标 | 999 是覆盖当前适配器全部分析单元的兼容上限，**不代表语言层数**；分析单元可能含视觉模块/投影/MTP | 闭环关键约定 | 代码实证 | 已回归 |
| E-003 | 直出优先需核验结构 | CLI 提供 `--save_path` 时 | 直出文件必须先核验与本节规范一致（`layer_scores` 降序、含 `method`/`patterns`、层名不加通配符）；不同版本/适配器结构输出存在实测差异 | ③ 解析并写出结果 | 代码实证 | 待验证 |
| E-004 | 同分同退约束 | 选择回退层时 | 分数相同的层必须作为整体同时回退/保留，不可拆分 | 结果在调优中的使用 | 用户反馈 | 已回归 |
| E-005 | 工具不可用的经验占位 | `analyze` 失败/超时且确属工具不可用 | 用 config.json `num_hidden_layers`（嵌套 config 逐层查）构造前/后 2-4层敏感的经验排序；**弱于精确分析**，数据/模型/schema 错误禁止占位掩盖 | 工具不可用时的经验规则 | 用户反馈 | 待验证 |
| E-006 | 复用判定归属 practice-cfg | 被再委派时 | 是否跳过分析（复用 `analysis_result.yaml`）由 practice-cfg 判定；基准 Practice 变化时本 skill 须覆盖旧结果 | 复用判定归属 | 用户反馈 | 已回归 |
