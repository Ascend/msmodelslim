---
name: practice-cfg
description: Use when 量化调优闭环中需要生成或修改一轮调优所需的 Practice YAML：读取/生成基准 Practice、委派敏感层分析（sensitive-layer-analysis）、按策略决策回退层、写出 YAML 并校验。
license: Apache-2.0
metadata:
  version: 0.1.1
  domain: quantization
  framework: msmodelslim
  protocol: cli
  skill_class: tool
  aliases:

    - practice-cfg
    - tune-practice

  trigger_intents:

    - 生成量化配置
    - 修改 practice
    - 依据敏感层分析结果生成量化配置

  keywords:

    - yaml_validation_validate
    - exclude
    - practice yaml

  # subagent 绑定声明：id = 委派标识（subagent_type）；bind = 委派时须绑定加载的 skill 根目录（含 references/、scripts/）
  # practice-cfg 依赖 strategy 父级：references/（Practice 格式）与 standing_high* 策略文档在其下；敏感层分析契约见 sensitive-layer-analysis
  subagent:
    id: "strategy/practice-cfg"
    bind:

      - "strategy/practice-cfg"
      - "strategy"
      - "sensitive-layer-analysis"

---

# 量化配置生成（practice-cfg）

## 概述

在量化调优闭环中，根据敏感层分析和上轮评测结果，**生成或修改**一轮调优所需的 Practice YAML，确保其通过校验后交付给后续 quantization 执行。

| 负责 | 不负责（编排层） |
|------|------------------|
| 敏感层分析（**再委派** `sensitive-layer-analysis`） | 缓存查询（`accuracy_lookup`） |
| 策略生成/修改 Practice YAML | 量化执行（`msmodelslim quant`） |
| YAML 校验（`validate_practice_yaml.py`） | 评测执行（`run_evaluation.py`） |
| | 缓存写入（`accuracy_append`） |
| | 历史记录（`history_append`） |
| | 策略终止决策 |

## 接口

**输入**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_type` | `str` | 模型类型名 |
| `model_path` | `str` | 模型路径 |
| `save_path` | `str` | 工作目录，Practice YAML 写入此目录下 |
| `device` | `str` | 分析设备，如 `"npu"`、`"npu:0"`、`"gpu:0,1"` |
| `strategy` | `str` | 调优策略：`"standing_high"` 或 `"standing_high_with_experience"` |
| `calib_dataset` | `str \| None` | 可选的校准数据集覆盖值；默认值见 [敏感层分析](../../sensitive-layer-analysis/SKILL.md) |
| `max_iterations` | `int` | 最大迭代轮次，由用户指定 |
| `round` | `int` | 当前调优轮次，用于生成本轮 Practice 文件名 |
| `prev_result` | `dict \| None` | 上轮评测结果（EvaluateResult 结构），首轮为 `None` |
| `anchor_practice` | `str \| None` | 当前已知最优且达标的 Practice YAML 路径（锚点） |

**产出**：`practice_path`（合法的 Practice YAML 文件路径）

**工具**：`sensitive-layer-analysis` subagent（敏感层分析，再委派）、`scripts/validate_practice_yaml.py`（校验）

## 执行步骤

### 步骤总览

```text
        ┌─────────────────────┐
        │ ① 读取/生成基准      │  ← 确定 schema 与静态量化边界
        │    Practice         │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │   ② 敏感层分析       │  ← 只执行一次；再委派
        │ (委派 sensitive-     │    sensitive-layer-analysis
        │  layer-analysis)     │
        └──────────┬──────────┘
                   │ 敏感度得分文件（各轮复用）
                   ▼
     (>>> 每轮循环 <<<)  ◄──────────────────┐
                   ▼                        │
        ┌─────────────────────────────────┐  │
        │ ③ 根据策略选择回退层              │  │
        │   + 生成/修改 Practice YAML      │  │
        └──────────┬──────────────────────┘  │
                   │                          │
                   ▼                          │
        ┌─────────────────────┐              │
        │ ④ 校验 Practice YAML │              │
        │ (validate_practice_  │              │
        │  yaml)               │              │
        └──────────┬──────────┘              │
                   │ practice_path            │
                   ▼                          │
         后续：quantization ──→ 下一轮 ─────┘
```

**调度优化**：敏感层分析由子任务执行、可能耗时较长，期间占用的卡由 `sensitive-layer-analysis` 自行管理；本 skill 不涉及量化/评测的并行调度（那是编排层决策），不做跨 skill 并行安排。

### ① 读取或生成基准 Practice

优先从 Practice 仓库中查找与当前 `model_type` 匹配的已验证 Practice；存在多个候选时，返回候选项，由主 Agent 确认后继续。未找到时，按照 [量化配置格式](../references/practice_yaml_format.md) 生成保守基准 Practice，保存为 `{save_path}/practice_base.yaml`。基准 Practice 必须在敏感层分析前确定并通过校验。

从基准 Practice 中继承：

- `apiversion`
- 目标量化处理器的 `include`
- 因模型能力或已验证经验而存在的静态 `exclude`
- 当前 schema 要求的其他静态字段，如 VLM 的 `spec.default_text`。

将静态排除项记录为 `protected_exclude`，在全部调优轮次中保持不变。每轮最终写入的 `exclude` 为 `protected_exclude ∪ tuning_exclude`；调优只能增减 `tuning_exclude`，不得删除静态排除项。

### ② 敏感层分析（委派 `sensitive-layer-analysis`）

本步骤通过**再委派** `sensitive-layer-analysis` subagent 完成（嵌套委派，见 `subagent_io_protocol.md`「子任务再委派」），本 skill 不直接执行 `msmodelslim analyze`。

**前置检查（复用判定）**：每个调优任务只执行一次分析，结果写入 `{save_path}/analysis_result.yaml`，供后续各轮复用。再委派前先检查该文件：

- 已存在且结构有效 → **跳过本次分析**（回传 `commands` 中 `sensitive_layer_analysis.skipped: true`），直接复用；
- 已存在但结构校验失败 → 覆盖旧结果，重新委派分析；
- 不存在 → 委派分析。

结构校验规则见 [sensitive-layer-analysis SKILL.md](../../sensitive-layer-analysis/SKILL.md)「解析并写出结果」。

**委派契约**（`subagent_type: "sensitive-layer-analysis"`，完整字段见其 SKILL.md）：

| input 字段 | 来源 |
|------|------|
| `model_type` / `model_path` / `save_path` / `device` | 透传本 skill 入参 |
| `calib_dataset` | 显式传入则透传；否则由 `sensitive-layer-analysis` 按协议默认（`modelslim_v1`→`mix_calib.jsonl`、`multimodal_vlm_modelslim_v1`→`calibImages`） |
| `practice_path` | 步骤 ① 产出的基准 Practice（用于确定分析范围 `--quant_modules`） |

**执行要求**（委托方须遵守、校验由 `sensitive-layer-analysis` 完成）：

- 使用与后续 Practice `spec.dataset` 一致的校准数据；
- 根据目标量化处理器的 `include` 确定分析范围；
- 不得将静态 `exclude` 中的模块作为可调回退项。

分析命令、设备绑定、指标选择、日志保存、成功判定及结果转换均以 `sensitive-layer-analysis` 契约为准，不在此重复定义。

**降级**：若环境不支持 subagent 内再委派，按 `sensitive-layer-analysis` SKILL.md 内容在本会话内完成其职责，保留同样 input/output 契约；`commands` 仍须含 `sensitive_layer_analysis`。

**失败处理**：子任务失败/超时按 `subagent_io_protocol.md`「子任务再委派」归并到本 skill 的 `output`/`error` 上抛，**不做二次占位**；经验规则占位仅由 `sensitive-layer-analysis` 在工具不可用时于其内部执行（见该 SKILL「工具不可用时的经验规则」），本 skill 不重复该规则。

---

### ③ 策略生成/修改 Practice 并写出 YAML 文件

**目的**：根据预计算的敏感度得分和当前轮次的策略需要，选择本轮回退层并确定离群值抑制策略，构造完整的 Practice YAML 内容，并**写入磁盘文件**。

**输入**：

- 敏感度得分文件 `{save_path}/analysis_result.yaml`（步骤 ② 产出，各轮复用）
- 上轮评测结果 `prev_result`（首轮为 `None`）
- 当前已知最优且达标的配置（锚点）

**具体动作**：

1. **确定本轮改动**（一次只改一两处字段，从预计算的敏感度得分中选择回退层，遵守同分同退约束）
2. **构造完整的 Practice YAML 内容**：继承基准 Practice 的 `apiversion` 和静态字段，仅修改当前策略允许的调优字段，详见 [量化配置格式](../references/practice_yaml_format.md)
3. **写出文件**：将 YAML 内容写入 `{save_path}/practice_round_{N}.yaml`（N 为当前轮次），得到 `practice_path`

| 改动项 | 说明 | 对应 YAML 位置 |
|--------|------|----------------|
| 调整 `tuning_exclude` | 增减敏感层回退；最终与 `protected_exclude` 取并集 | `spec.process[].exclude` |

**修改粒度**：

- **一次只改一两处字段**，避免多因素同时变化导致无法归因

**exclude 设计原则**：

- 优先覆盖敏感层排序中 **score 最高的层**
- **同分同退**：敏感度分数相同的层必须作为一个整体同时回退或同时保留
- 回退位置经验优先级：靠近输入的前若干层 > 靠近输出的后若干层 > 语义敏感子模块（部分 MLP / attention 层）
- 回退级别按层组离散化（如前 2层、前 4层、前 4 + 后 4层……），便于二分搜索

**离群值抑制叠加原则**：

- 先上单一、简单的抑制（如仅 `iter_smooth`）
- 确认瓶颈后再考虑更强或组合策略
- **二分阶段抑制组合固定，只动回退刻度**；摸高阶段才允许切换抑制

> 调优策略由入参 `strategy` 决定。`"standing_high"` 详见 [standing_high 策略](../standing_high/strategy_standing_high.md)；`"standing_high_with_experience"` 详见 [standing_high_with_experience 策略](../standing_high_with_experience/strategy_standing_high_with_experience.md)。

**始终保留锚点**：掉精度时可回滚到上一已知达标配置。

---

### ④ 校验 Practice YAML

**脚本调用**：

```bash
python skills/strategy/practice-cfg/scripts/validate_practice_yaml.py --practice-path /path/to/practice.yaml
```

> **执行位置注意（实测 F0-2）**：本脚本会 `import msmodelslim`。若在 msmodelslim **仓库根目录内**执行且 msmodelslim 为源码安装（`bash install.sh` 为复制式，源码树缺 `config/`），会命中残缺源码包报 `SecurityError: msmodelslim/config doesn't exist`。建议在**仓库外**工作目录以**脚本绝对路径**调用（此时命中 site-packages 完整安装），或确认源码树内已具备可用的 `config`（如符号链接）。脚本本身不依赖 cwd（跨 skill 依赖按 `__file__` 定位）。

**返回**：

```json
{
    "ok": true,
    "valid": true,
    "errors": []
}
```

**校验内容**：

1. **YAML 语法**：能否正常解析
2. **Schema 校验**：是否可被 `PracticeConfig.model_validate` 通过（字段名、类型、必填项）
3. **业务规则**：如 `label` 必须是 dict 而非字符串、`type` 与字段是否匹配

**错误处理**：

| 错误类型 | 说明 | 动作 |
|----------|------|------|
| `parse_error` | YAML 语法错误 | 修正语法后重试 |
| `schema_error` | 字段缺失/类型不对 | 修正字段后重试 |
| `business_rule_error` | 业务逻辑违规 | 按提示修正后重试 |

`valid=false` 时**不可继续后续步骤**，必须修正后重新校验。

> YAML 字段名、类型、必填项等 schema 细节见 [量化配置格式](../references/practice_yaml_format.md)。

---

## 产出

`practice_path`（合法的 Practice YAML 文件路径），交付给编排层进行缓存查询后，传递给 quantization 执行量化。

## 约束汇总

| 约束 | 说明 |
|------|------|
| ② 在首轮前调用一次 | 敏感度得分每个调优任务计算一次，各轮复用 |
| 一次只改一两处 | exclude 或离群值抑制，避免多因素同时变化 |
| 保留锚点 | 始终保留一份当前已知最优且达标的配置，掉精度可回滚 |
| 校验必过 | `valid=false` 时不可继续，必须修正后重新校验 |
| save固定 | `spec.save` 字段除非用户指定，默认情况下必须为 `ascendv1_saver` |

## 常见错误

- 回退层选择时拆分同分同退组（应整体回退或整体保留）
- `metadata.label` 写成字符串而非 dict
- `valid=false` 仍继续后续步骤
- 命令行参数 `--device` 未使用 `npu:0` 这种格式，错误地使用了 `DeviceType.NPU`

## 经验条目（Experiences，追加制）

> 追加规范见 `skills/README.md`「经验条目」。连续编号 `[E-序号]`；正文保留权威展开，本表只做索引 + 元数据登记；来源：实测编号（F0-x / Dx / Tx）| 用户反馈 | 代码实证；验证状态：已回归 | 待验证 | 已上流 docs。

| 条目 | 主题 | 适用条件 / 触发信号 | 结论要点（一句话） | 正文位置 | 来源 | 验证状态 |
|------|------|------|------|------|------|------|
| E-001 | 校验脚本执行位置 | `validate_practice_yaml.py` 调用前 | 脚本 `import msmodelslim`，在仓库根内跑命中源码树残缺包（缺 `config/`）报 `SecurityError`；在**仓库外** cwd + 脚本绝对路径调用，命中 site-packages 完整安装 | ④ 校验 Practice YAML | 实测 F0-2 | 已回归 |
| E-002 | 单变量改动原则 | 每轮修改 Practice 时 | 一次只改一两处字段（exclude / 离群值抑制），避免多因素同时变化导致无法归因 | ③ 策略生成/修改 | 用户反馈 | 已回归 |
| E-003 | 回退位置经验优先级 | 从敏感度得分选择回退层 | 靠近输入的前若干层 > 靠近输出的后若干层 > 语义敏感子模块（部分 MLP/attention）；回退级别按层组离散化（前 2/4/前4+后4…）便于二分 | ③ exclude 设计原则 | 代码实证 | 待验证 |
| E-004 | protected_exclude 不可删 | 基准 Practice 继承静态 exclude | 静态排除项记录为 `protected_exclude`，全轮次不变；每轮最终 `exclude = protected_exclude ∪ tuning_exclude` | ① 读取基准 Practice | 用户反馈 | 已回归 |
| E-005 | 保留锚点 | 掉精度需回滚时 | 始终保留当前已知最优且达标的 Practice 作为锚点（`anchor_practice`），掉精度可回滚 | ③ 始终保留锚点 | 用户反馈 | 已回归 |
| E-006 | save 默认固定 | 生成/修改 Practice 时 | `spec.save` 除非用户明确指定，默认必须为 `ascendv1_saver`（evaluation-cfg 的 `compressed_tensors` 只属于评测配置，勿跨用） | 约束汇总 | 用户反馈 | 已回归 |
