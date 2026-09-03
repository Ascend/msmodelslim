# msModelSlim Skills

本目录存放可供 Agent 加载的 Skills，覆盖模型量化与精度调优的完整闭环：安装、适配、调优策略、量化执行、评测与端到端编排。

## 目录结构

| 领域 | 顶层 Skill | 说明 | 可委派子 Skill（供编排） |
|---|---|---|---|
| 端到端编排 | [tuning/](tuning/) | 全自动量化调优编排者（workflow） | — |
| 安装与环境 | [installation/](installation/) | 安装与环境校验 | `installation`（自身） |
| 模型适配 | [adaptation/](adaptation/) | 校准适配 + EP 并行适配 | `adaptation/calibration`、`adaptation/calibration/analyze`、`adaptation/calibration/dequant`、`adaptation/calibration/verify`、`adaptation/ep` |
| 敏感层分析 | [sensitive-layer-analysis/](sensitive-layer-analysis/) | 敏感层分析执行 | `sensitive-layer-analysis`（自身） |
| 调优策略 | [strategy/](strategy/) | 调优策略与 Practice 生成 | `strategy/practice-cfg`、`strategy/standing_high_with_experience/expert-rules` |
| 量化执行 | [quantization/](quantization/) | 执行 `msmodelslim quant` | `quantization`（自身） |
| 评测 | [evaluation/](evaluation/) | 评测配置 / 执行 / 评测集 herding 压缩 | `evaluation/evaluation-cfg`、`evaluation/evaluate`、`evaluation/dataset-compression-herding` |
| 资料管理 | [docs-management/](docs-management/) | 资料管理 | — |

调优闭环脚本共享库（无 SKILL）：[tuning-loop-lib/](tuning-loop-lib/)。

## 调用方式

同一套 Skill 支持两种入口，共用同一批子 Skill 与脚本，差异仅在「由谁驱动」。

**1. 直接使用（独立完成任务）**

用户针对单一需求（安装、写适配器、生成 Practice、执行量化、跑评测等）直接触发对应 Skill，由 Skill 按自身流程独立完成，不依赖 `tuning` 编排。

- 领域 Skill（`adaptation` / `strategy` / `evaluation`）是**领域入口**，内部路由到子 Skill 完成具体工作；
- 单任务 Skill（`installation` / `quantization` / `sensitive-layer-analysis`）自身即可独立完成，无子级。

**2. 端到端调优（由 `tuning` 编排）**

用户要求「自动量化调优」时，由 `tuning` 作为编排者接管，按「环境准备 → 模型准备 → 量化配置调优 → 结果输出」四阶段，委派各可委派子 Skill 协作完成：

- 委派对象是带 `metadata.subagent` 声明的**可委派子 Skill**（见「subagent ↔ skill 绑定」）；
- 顶层领域 Skill 不直接参与委派，作为领域入口供直接使用；
- 编排层与子 Skill 通过 SUBAGENT_IO v1 契约衔接（字段定义见 `tuning/references/`）。

## 内容组织

skills 文档遵循「docs 为权威、skill 只写增量」的分工，避免重复：

1. **docs（权威）**：安装指南、CLI 参数、YAML schema、算法原理、模型接入等完整细节以 `docs/` 为准，skills 内只链接、不复写；
2. **skill（增量）**：只保留执行所需的行为增量——触发条件、执行步骤、门禁与实战经验（docs 未覆盖项）；
3. **编排契约**：`tuning/references/` 只定义委派时机、input/output 字段与判定标准，指向各子 Skill，不重复实现细节。

## subagent ↔ skill 绑定

调优闭环中可作为 subagent 委派的 skill（如 `strategy/practice-cfg`、`adaptation/ep`、`quantization` 等），其绑定关系**内置于各自 SKILL.md 的 frontmatter**：`metadata.subagent` 声明 `id`（委派标识，即 `subagent_type`）与 `bind`（委派时须绑定加载的 skill 根目录列表，含自身）。绑定清单与解析规则见 [tuning/references/subagent_io_protocol.md](tuning/references/subagent_io_protocol.md)「subagent ↔ skill 绑定」章节。

部分被委派 subagent 可**再委派**其 `bind` 内的子 skill（嵌套委派，链深上限 2），如 `strategy/practice-cfg` 步骤②再委派 `sensitive-layer-analysis` 完成敏感层分析，主 Agent 无需感知；规则见该文档「子任务再委派」章节。

## Skill 编写规范

所有 `SKILL.md` 遵循统一结构与命名，保证**人类可读、Agent 可执行、经验可积累**。任何新增 / 修改 skill 均须满足本规范。

### 1. 标题与命名

- 一级标题统一为 `# <中文名>（<frontmatter name>）`，如 `# 敏感层分析（sensitive-layer-analysis）`；其中 name 必须与 frontmatter `name` 一致。
- 小节名一律使用中文；`## Overview`、`## Skill: xxx` 等英文标题统一改写为 `## 概述` 等中文小节。
- 同级小节标题保持同构：同为动作就同为动宾（如 `## 检查清单` 不混用 `## 清单检查`），同为门禁就同用规范门禁名（见下）。

### 2. 统一小节骨架

**执行 / 决策型 skill**（会被委派或独立执行完整任务的 skill）按需选用以下小节，顺序推荐为：

```markdown

## 职责（或 ## 概述）      # 解决什么 / 不解决什么 / 权威参考指向 docs

## 输入 / ## 输出           # 委派或直用契约的字段表（类型/必填/说明）

## 流程                    # 编号步骤（1. 2. 3. 或 ①②③），可用决策树/流程图画清分支

## 硬门禁 / ## 红线 / ## 检查清单 / ## 通过-失败标准   # 见「门禁范式」

## 常见错误 / ## 错误处理   # 本 skill 高频错误 → 处置

## 经验条目（Experiences）  # 追加制登记表，见「经验条目规范」

## 参考资料                # 链接 references/ 与 docs/
```

**路由 / 入口型 skill**（`adaptation`、`evaluation`、`strategy`）：只保留「子目录表 + 使用方式」即可，**不加**经验条目区（见经验条目适用层）。

### 3. 门禁范式（统一命名，不得使用异名）

| 范式 | 语义 | 格式 |
|------|------|------|
| `## 硬门禁` | 流程**启动前**必须满足；不满足即停、**不得带病进入**后续步骤 | 条目化「必须…，否则…」 |
| `## 红线` | **贯穿全流程**的禁止行为 / 边界 | `- 不得…` / `- 禁止…` 清单 |
| `## 检查清单` | 执行前逐项自检 | `- [ ]` 勾选项 |
| `## 通过/失败标准` | 判定本 skill 是否完成 | 显式「通过 = …；失败 = …」 |
| `## 常见错误` / `## 错误处理` | 高频异常 → 处置 | 表格（错误 / 原因 / 修复） |

- 同语义门禁不得再自造名称（如「硬性门禁」「红线和原则」须归入上表）；正文内的「必须 / 禁止」措辞不受限。
- 门禁必须是**可判定**的：给出判定输入（exit code、文件存在性、schema 校验、数值门限等），不能只写"注意"。

### 4. 经验条目规范（显式、可追加）

统一格式为一张**追加制登记表**，置于各 skill 的 `## 经验条目（Experiences）` 小节：

```markdown

## 经验条目（Experiences，追加制）

> 追加规范：连续编号 `[E-序号]`（跨 skill 不要求全局唯一，按文件内递增）；正文保留权威展开，本表只做索引 + 元数据登记（结论一句话）；来源三选一：实测编号（F0-x / Dx / Tx）| 用户反馈 | 代码实证；验证状态三选一：已回归 | 待验证 | 已上流 docs。经验上流到 docs 后改为 `已上流 docs`，可在表内保留指针。

| 条目 | 主题 | 适用条件 / 触发信号 | 结论要点（一句话） | 正文位置 | 来源 | 验证状态 |
|------|------|------|------|------|------|------|
```

- **不记录日期/时间**：经验条目表不含日期列；时间戳对检索与复用无意义，勿追加。
- **经验条目 ≠ 常见错误**：常见错误表解决"出错怎么修"；经验条目沉淀"正确认知 / 最佳做法 / 边界结论"（含其适用条件与验证状态）。
- **新增经验必须登记**：任何实测中发现、被用户反馈纠正、或经代码实证的结论，应追加为一条 `[E-序号]`，而非仅散落在正文。
- **淘汰与上流**：随版本过期的条目整行删除；被 docs 收录的经验改验证状态为 `已上流 docs` 并保留指针，遵循「docs 权威、skill 只写增量」原则。

### 5. 经验条目的适用层

| 适用层 | skill | 说明 |
|------|------|------|
| **强适用**（必须带经验条目区） | `sensitive-layer-analysis`、`strategy/practice-cfg`、`evaluation/evaluation-cfg`、`evaluation/evaluate`、`quantization`、`adaptation/calibration/analyze`、`adaptation/calibration/dequant`、`adaptation/calibration/verify`、`adaptation/calibration/layer_wise`、`evaluation/dataset-compression-herding` | 反复实操、每次运行都可能产出新经验（环境坑 / CLI 差异 / schema 边界 / 结构判定信号） |
| **部分适用**（沉淀原则级条目） | `tuning`、`docs-management`、`adaptation/calibration`（主流程）、`adaptation/ep` | 只登记编排 / 文档管理 / 适配流程级原则；具体操作坑**下沉**给其子 skill 登记，避免编排层膨胀 |
| **弱适用 / 不加** | `adaptation`、`evaluation`、`strategy`（路由入口） | 只做路由分发；如确有"哪类模型走哪条路径"的教训，登记到对应子 skill |
| **经验库载体** | `strategy/standing_high_with_experience/expert-rules` | 全体系经验的 L1/L2/L3 检索索引与专家库，见其 SKILL.md「三级知识结构」；新模型个案按 L3 约定追加 `models/<vendor>/<model>.md` |

### 6. 风格约定

- 语言以中文为主；代码、命令、CLI 参数、YAML 字段名保留原文（不加翻译）。
- 决策树 / 流程优先用文本图（text 代码块）或 Markdown 表格表达，少用无法在终端阅读的图形字符。
- "权威参考以 docs 为准"的内容一律用链接指向 `docs/`，skill 内不复写长文。
- **遵从资料规范公共校验 CE-05 / CE-06**（《[公共校验清单](../docs/zh/contributing/development_guide/docs_standards/00_common_checklist.md)》）：引用 `docs/` 标题写作 `《[正式标题](相对路径)》`；中文语境数字与量词/单位不留空格（如 `前2层`、`≤2卡`）。
- **CE-04**：全文（不含 fenced code 内示例）仅一个一级标题。analyze 的「分析报告」是子 agent **落盘产物模板**（`{save_path}/model_analysis_report.md`），其 `# 分析报告` 仅出现在代码块示例中，不是本 skill 的第二 H1。
- 对用户 / 编排层的**必做沟通**（阻塞、需确认、风险提示）用「必须 / 不得」显式写出，并给出话术要点。
