---
name: example
title: 新增示例场景
triggers: ["新增示例", "示例场景", "样例场景"]
target_dir: docs/zh/<领域>/<name>/
create_subdir: true
subdir_name: "<name>：英文小写，多词以下划线连接，与目标目录既有文档命名一致"
templates:
  - "02_process_guide_template.md → <name>.md（使用指南）"
---

# 新增示例场景

> **样例说明**：本文件是场景文档的**样例**，仅演示结构、字段与正文写法，**不作为已登记的真实场景**。新增真实场景时复制本文件为 `<name>.md`，改写 `target_dir` / `templates` 与正文（含可选的"场景特有校验清单"）即可。frontmatter 字段含义见 [README](README.md)。

## 目标形态

描述该场景新增资料后的目标形态：在哪个目录、产出哪几篇文档、各属什么文档类型。示例：

- 在 `docs/zh/<领域>/` 下新增一篇使用指南 `<name>.md`，套 [02 流程模板](../../../docs/zh/contributing/development_guide/docs_standards/02_process_guide_template.md)。

## 命名规则

说明目录 / 文件命名规则（英文小写、连接符风格、与既有文档一致等）。

## 操作步骤

1. 确认 `<name>` 命名，检查目标目录下无同名文档；同名则转为"更新"职责。
2. 按 `create_subdir` 建子目录（`create_subdir: true` 时）。
3. 复制 `templates` 中各模板为目标文件，**保留 `{{ }}` 占位符与 `[OPTIONAL]` 标记**，供用户填充。
4. 用户填充后**分别校验**本场景特有校验清单与文档类型清单（本例[《流程指南校验清单》](../../../docs/zh/contributing/development_guide/docs_standards/02_process_guide_checklist.md)），并叠加[《公共校验清单》](../../../docs/zh/contributing/development_guide/docs_standards/00_common_checklist.md)，综合结果。
5. 修复至无 ERROR；WARN 项提示用户决定是否处理。

## 场景特有校验清单

仅列本场景特有、文档类型清单未覆盖的条目（示例，新增场景时按需编写；不重复类型清单）：

| 编号 | 级别 | 校验条目 |
| --- | --- | --- |
| SC-01 | ERROR | 本场景特有的必填章节 / 字段存在 |
| SC-02 | WARN | 本场景特有的命名或格式要求满足 |

## 校验依据

- **场景特有校验清单**：正文"场景特有校验清单"小节（若有），仅本场景特有条目。
- **文档类型清单**：按插入模板对应的类型清单（本例[《流程指南校验清单》](../../../docs/zh/contributing/development_guide/docs_standards/02_process_guide_checklist.md)）；类型清单已覆盖的条目不写入场景特有清单。
- **公共兜底**：[《公共校验清单》](../../../docs/zh/contributing/development_guide/docs_standards/00_common_checklist.md) 的 CE-01~CE-04、CW-01~CW-02。
- 三类清单**分别逐条执行并综合结果**；修复优先级：CE → 类型前缀序 → 场景特有序号。

## 编写注意事项

写真实场景文档时注意：

- **操作步骤须含"同名判定"**：明确目标目录 / 文件已存在时的处理——目录不存在 → 按新增；目录在但空 → 复用继续新增；目录在但只缺部分文档 → 补插缺失文件；同名文档已存在 → 转"更新"职责。
- **命名规则给显式替换示例**：除规则描述外，补一行 `<占位>` → 实际名称的替换实例（如 `flex_smooth_quant`），减少执行者猜测。
- **场景特有校验清单仅列本场景特有项**：不重复文档类型清单条目；校验按"场景 + 类型 + 公共"分别执行并综合结果。
- **通用规则不外提**：待建词条 TODO 登记、链接校验基准、公共清单执行口径（CE-02 / CW-01）等属 skill 通用规则（见 `references/validation_rules.md`），不写入场景文档；模板脚手架章节（如 01 词条的「排版规范」）由公共校验 CE-02 兜底，场景文档无需重复。
