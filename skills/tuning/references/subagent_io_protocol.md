# 主 Agent ↔ Subagent 交互协议（SUBAGENT_IO v1）

编排层通过子 agent 委派工具（如 `task`）委派 subagent。主 Agent 在委派描述、Subagent 在最终回复中，须使用统一的 `subagent-io` 机器可读块。

各 subagent 的 `input` / `output` 字段定义见对应 reference；本文只规定**围栏格式、信封字段、职责边界**。

## 适用 subagent

自动调优流程中，以下 subagent **均须**遵守本协议。绑定关系见下方「subagent ↔ skill 绑定」。

| subagent | 职责 | 字段定义 | 绑定 skills（frontmatter 声明） |
|----------|------|----------|--------------------------------|
| `adaptation/calibration/analyze` | 适配前模型分析 | [prepare_model.md](./prepare_model.md) | `adaptation/calibration/analyze` |
| `adaptation/calibration` | 模型适配与验证 | [prepare_model.md](./prepare_model.md) | `adaptation/calibration` |
| `adaptation/calibration/dequant` | 原生量化模型反量化适配 | [prepare_model.md](./prepare_model.md) | `adaptation/calibration/dequant` |
| `adaptation/calibration/verify` | 适配功能性验收（四步验证；calibration 内部门禁 + 主 Agent 验收门禁） | [prepare_model.md](./prepare_model.md) | `adaptation/calibration/verify` |
| `adaptation/ep` | MoE 多卡 EP 并行适配（多卡条件触发，归入模型准备） | [prepare_model.md](./prepare_model.md) | `adaptation/ep` |
| `evaluation/dataset-compression-herding` | 数据集 coreset 压缩（一次性数据准备，无门禁） | [quantization_tuning.md](./quantization_tuning.md) | `evaluation/dataset-compression-herding` |
| `strategy/standing_high_with_experience/expert-rules` | 结构化回退专家经验 | [quantization_tuning.md](./quantization_tuning.md) | `strategy/standing_high_with_experience/expert-rules` |
| `evaluation/evaluation-cfg` | 生成测评配置 | [quantization_tuning.md](./quantization_tuning.md) | `evaluation/evaluation-cfg` |
| `strategy/practice-cfg` | 生成 Practice 配置（步骤②可再委派 `sensitive-layer-analysis`） | [quantization_tuning.md](./quantization_tuning.md) | `strategy/practice-cfg`, `strategy`, `sensitive-layer-analysis` |
| `sensitive-layer-analysis` | 敏感层分析（一次，产出 analysis_result.yaml；可作为 practice-cfg 子任务或独立使用） | [自身 SKILL.md](../../sensitive-layer-analysis/SKILL.md) | `sensitive-layer-analysis`, `installation` |
| `quantization` | 执行量化 | [quantization_tuning.md](./quantization_tuning.md) | `quantization`, `installation` |
| `evaluation/evaluate` | 执行精度评测 | [quantization_tuning.md](./quantization_tuning.md) | `evaluation/evaluate`, `installation` |

> **环境能力与降级策略（三级）**：本协议假设环境支持「skill 目录即 subagent」的委派机制，即 `subagent_type` 取值为 `skills/` 下的 SKILL 目录路径（如 `strategy/practice-cfg`、`adaptation/ep`）。委派前**必须探测**运行环境能力并选定执行形态：
>
> - **L2（skill-as-subagent 可用）**：按 `task` 工具以 SKILL 目录路径发起委派。
> - **L1（普通 subagent 可用、skill-as-subagent 不可用）**：主 Agent 读取目标 skill frontmatter 的 `metadata.subagent.bind`，将 bind 列表内 skill 的内容（SKILL.md + references/scripts）注入委派描述后，发起普通 subagent 委派（行为等价）。
> - **L0（主 Agent 也无法委派 subagent）**：主 Agent 按对应 SKILL 的提示词内容在本会话内完成其职责，并保留同样的 input/output 契约格式。

## subagent ↔ skill 绑定

绑定关系**内置于各 subagent 对应 SKILL.md 的 frontmatter**（`metadata.subagent`），不依赖平台侧的 subagent 配置。每个可作为 subagent 委派的 skill 在 frontmatter 中声明：

```yaml
metadata:
  subagent:
    id: "<subagent_type 标识>"   # 与委派时的 subagent_type 一致
    bind:                        # 委派该 subagent 时须绑定加载的 skill 根目录列表（含其 references/、scripts/）
      - "<skill 根目录>"
      - "<另一 skill 根目录>"
```

**解析规则**：

- `metadata.subagent.id` 即该 skill 的委派标识（`subagent_type`），与 [quantization_tuning.md](./quantization_tuning.md) / [prepare_model.md](./prepare_model.md) 各委派模板中的 `subagent_type` 保持一致（`routing.md` 只做路由判定，契约已在 `prepare_model.md`）。
- `metadata.subagent.bind` 列出委派时须**加载给该 subagent** 的 skill 根目录：`bind` 首个元素恒为主 skill 自身；后续元素为跨 skill 依赖（如 `strategy/practice-cfg` 依赖 `strategy` 父级目录下的 `references/` 与 `standing_high*` 策略文档，以及 `sensitive-layer-analysis` 的敏感层分析契约；`quantization` / `evaluation/evaluate` 依赖 `installation` 做环境校验）。绑定粒度是 skill 根目录（含 `references/`、`scripts/` 子目录）。
- 委派时：读取目标 skill frontmatter 的 `metadata.subagent.bind`，把列表内的 skill 全部作为该 subagent 的可用上下文。平台支持绑定则按 `bind` 加载；平台不支持（无 subagent 绑定能力）时，主 Agent 将 `bind` 列表中的 skill 内容（SKILL.md + 被引用的 references/scripts）注入委派描述，行为等价。
- 新增可委派 subagent 时，必须在其 SKILL.md frontmatter 补齐 `metadata.subagent` 声明，并在上表登记。

## 子任务再委派（嵌套委派）

协议允许**两级委派链**：主 Agent → 被委派 subagent → 该 subagent 再委派其 bind 列表内的子 skill。链条深度**上限为 2**（主 Agent 不再直接接触孙级 subagent），适用于"父 subagent 的某个内部步骤本身就是独立 skill 职责"的场景（如 `strategy/practice-cfg` 步骤②敏感层分析再委派 `sensitive-layer-analysis`）。

**规则**：

- 父 subagent 对主 Agent 承担**全部责任**：再委派的子任务失败 = 父步骤失败，父 subagent 不得把孙级错误原样上抛，须归并到自身 `output` / `error` 中；
- 父 subagent 的 `output` 须**汇总子任务产物路径**（如 `analysis_result_path`），并保持 `commands` 审计完整（子任务执行的命令以其在父 `commands` 中的条目体现，见「`commands` 字段」）；
- 子任务沿用同一 SUBAGENT_IO v1 信封（委派块含 `input`，回传块含 `output`/`error`），父 subagent 负责把孙级回传的机器可读块解析并转述，**不得**伪造孙级 output；
- **委派对象判定**：可再委派的对象 = 父 subagent 自身 SKILL.md 中**显式声明将委派**的子 skill（如 `strategy/practice-cfg` 步骤②声明委派 `sensitive-layer-analysis`），**而非其 `metadata.subagent.bind` 列表的全部元素**。`bind` 仅表达"加载上下文"：被加载的 skill 既有可委派子级，也可能有仅供参考的父级/同级（如 `strategy/practice-cfg` 的 bind 含父级 `strategy`，仅作参考，不可委派）。是否委派、委派谁一律以 SKILL 正文为准；
- **叶子限制**：被再委派的子 subagent 为**叶子**，不得继续向下委派（链深上限即止于 2）。叶子 skill 的 bind 中即使含其他 skill（如 `sensitive-layer-analysis` 含 `installation`），也只作为本会话内直接使用的参考上下文；
- **commands 归属**：父 `output.commands` 中对应子任务的条目，其 `command` 可能由子任务实际执行后回传、父原样收录；审计以父 `commands` 为准（等价可复现），不要求区分实际执行者；
- **环境降级与上报代下发（父 subagent 无再委派能力时）**：父 subagent 若发现自己无再委派能力（无 `task` 工具 / 平台禁止嵌套委派），**不得静默 in-session 吞掉子任务**。此时在回传 `output` 中携带 `subagent_dispatch_requests`（数组，每项含 `delegation_id`/`subagent_type`/`input`/`bind`），请求主 Agent 代拉起真实子 subagent；主 Agent 收到后代为委派，并在下一条消息中把各子任务回传的 subagent-io 块**原样转述**给父 subagent，父 subagent 继续完成剩余流程并归并结果。**仅当**主 Agent 也确认无法委派（L0）时，父 subagent 才允许按目标 skill 的 SKILL.md 内容 in-session 完成职责并保留同样的 input/output 契约格式。

**职责边界（三级）**：

| 角色 | 写什么 | 读什么 |
|------|--------|--------|
| **主 Agent** | 委派描述（含 `input`） | 父 subagent 回传 `output` / `error` |
| **父 subagent**（可再委派） | 向主 Agent 回传自身 `output`/`error`；再委派时写子任务委派块（含 `input`） | 主 Agent 委派 `input`；子任务回传 `output`/`error` |
| **子 subagent** | 向父 subagent 回传 `output`/`error` | 父 subagent 委派 `input` |

主 Agent 与各级 subagent 均**不得**伪造下一级的 `output`；结论须来自机器可读块。

## 职责边界

| 角色 | 写什么 | 读什么 |
|------|--------|--------|
| **主 Agent** | 委派描述中的 subagent-io 块（含 `input`） | Subagent 回传 subagent-io 块中的 `output` / `error` |
| **Subagent** | 最终回复中的 subagent-io 块（含 `status` + `output` 或 `error`） | 主 Agent 委派 subagent-io 块中的 `input` |

主 Agent **不得**伪造 Subagent 的 `output`；汇总结论须来自 Subagent 回传的 subagent-io 块。

## 消息结构（委派与回传统一）

每条消息由两部分组成：

- **块外**（可选）：≤3行纯文本摘要
- **块内**（必选）：有且仅有一个 ` ```subagent-io v1 ` 围栏块

完整形态参考如下：

````markdown
<可选摘要，≤3行>

```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "<subagent 名称>",
  ...
}

```text
````

约束：

1. 禁止第二个 subagent-io 块或重复 JSON
2. 块外**禁止**：长参数列表、SKILL 全文、完整 YAML/日志正文、重复 `input` 已有字段的执行细节
3. JSON 须可解析；`protocol` 固定为 `subagent.subagent_io`
4. 委派块**不含** `status` / `output` / `error`；回传块**不含** `input`

### 委派信封（主 Agent → 委派描述）

块外摘要规则见上文。以下**仅展示块内** subagent-io 围栏内容：

```subagent-io v1
{
  "protocol": "subagent.subagent_io",
  "subagent_type": "<与 task 参数 subagent_type 一致>",
  "input": { }
}
```

`input` 字段见上表对应 reference 中的 subagent 字段表。

### 回传信封（Subagent → 最终回复）

块外摘要规则见上文。以下**仅展示块内** subagent-io 围栏内容：

成功时（`status: "ok"` 时填 `output`，**不填** `error`）：

```subagent-io v1
{
  "protocol": "subagent.subagent_io",
  "subagent_type": "<本 subagent 名称>",
  "status": "ok",
  "output": { }
}
```

失败时（`status: "failed"` 时填 `error`，**不填** `output`）：

```subagent-io v1
{
  "protocol": "subagent.subagent_io",
  "subagent_type": "<本 subagent 名称>",
  "status": "failed",
  "error": {
    "code": "UNKNOWN_ERROR",
    "message": "简短错误描述"
  }
}
```

`output` / `error` 内具体字段见对应 reference，不在此重复。

### `commands` 字段（回传 `output` 内，涉及 CLI/脚本时必填）

当 subagent 通过 `execute` 运行 shell 命令或脚本时，须在 `output.commands` 中列出**实际执行**（或等价、可复现）的命令，供审计日志追溯。

每项结构：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 步骤标识，如 `quantize`、`sensitive_layer_analysis` |
| `command` | string | 完整 shell 命令；未执行时可省略 |
| `skipped` | bool | 未执行时为 `true` |
| `reason` | string | 跳过原因（可选） |
| `exit_code` | int | 命令退出码（执行类节点必填） |
| `fix` | string | 错误修复类节点：错误摘要 + 修复动作（可选，遇错必填） |
| `output_path` | string | 该步骤关键产物路径（可选） |

各 subagent 要求的 `name` 见 `quantization_tuning.md` 对应小节。

### 执行者关键节点记录与审计汇总（G8）

- **执行者（子 subagent / 降级执行者）责任**：凡涉及 CLI/脚本的子任务，回传 `output.commands` 时必须记录**关键节点**——核心命令行（`name`+`command`+`exit_code`）、遇错时的修复（`fix`：错误摘要 + 修复动作）、关键产物路径（`output_path`）。**不得省略**；省略即该子任务缺审计项。降级执行（in-session 完成）同样须照此记录。
- **主编排责任（审计汇总）**：每个子任务回传后，主编排在工作目录 `{save_path}/audit/audit_log.{json,md}` **按子任务聚合**该任务的 `subagent_type` / `input`（委派输入）/ `commands`（含 exit/fix/output_path，原样收录）/ `deliverables`（产物清单）/ `status`+`output`\|`error` / 时间戳。人类打开 `audit_log` 即可一次性复盘全流程。审计日志与最终交付物一并归档，不得删除。
- **禁止虚构**：主编排不得编造执行者的 `commands`/`fix`；执行者漏报时如实标注（`commands: []` 或“未回传”），并可在主 Agent 侧附观测到的命令（注明来源为审计观测而非执行者回传）。

## 反例

| 反例 | 问题 |
|------|------|
| 整段自然语言参数列表、无 subagent-io 块 | 无法解析 |
| 块内缺少必填字段 | 委派不合规，须修正后重委派 |
| 块外重复 `input` 路径/设备说明或写执行步骤 | 违反「块外 ≤3行摘要」 |
| 回传只有 Markdown 表格或纯自然语言（如「全部任务完成。」） | 无 subagent-io 块，不得作为有效结论 |
| 在 `output` 中粘贴完整 YAML / 日志正文 | 应只回传路径等结构化字段 |
| 回传同时含 `output` 与 `error`，或 `status` 与内容不匹配 | 信封字段冲突 |

调优闭环四类 subagent 完整示例见 `quantization_tuning.md` 各小节；适配类两类见 `prepare_model.md`。

## 回传检查（主 Agent）

`task` 返回 Subagent 原文，不附带校验标志。须从回传中解析 subagent-io 块：

- `status: "ok"` → 读 `output`
- `status: "failed"` → 读 `error`
- 无块或无法解析 → 重试或判该步失败，**不得**用自然语言摘要代替
