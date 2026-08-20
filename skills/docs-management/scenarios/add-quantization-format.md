---
name: add-quantization-format
title: 新建量化格式资料
triggers:
  - "新增量化格式"
  - "新建量化格式资料"
  - "新增格式词条"
  - "量化格式文档"
  - "add quantization format"
target_dir: docs/zh/knowledge_base/quantization_format/<format_name>/
create_subdir: true
subdir_name: "<format_name>：英文小写，多词以下划线连接，与代码/地图命名一致"
templates:
  - "01_term_glossary_template.md → term_<format_name>.md（格式词条）"
  - "02_process_guide_template.md → <format_name>_usage.md（使用指南）"
---

# 新建量化格式资料

## 目标形态

在 `docs/zh/knowledge_base/quantization_format/<format_name>/` 下新增：

- 格式词条 `term_<format_name>.md`：套 [01 词条模板](../../../docs/zh/contributing/development_guide/docs_standards/01_term_glossary_template.md)，词条类别为**量化数据格式**；须含量化模式支持情况与交付件说明；H2 建议对齐 `## 1. 概述` / `## 2. 词条介绍` / `## 3. 关联流程` / `## 4. 关联词条`（格式专属内容放在「2. 词条介绍」下）。
- 使用指南 `<format_name>_usage.md`：套 [02 流程模板](../../../docs/zh/contributing/development_guide/docs_standards/02_process_guide_template.md)；操作步骤须覆盖**确认模式支持（或适配）、配置、执行**。

并在《[量化格式](../../../docs/zh/knowledge_base/quantization_format/README.md)》总词条中登记该格式（格式地图、关联词条 / 关联流程）。目录约定见资料规范 [§3.3](../../../docs/zh/contributing/development_guide/docs_standards/README.md#quantization-format-docs)。

## 命名规则

- `<format_name>`：英文小写，多词以下划线连接（如 `compressed_tensors`、`ascendv1`、`mindie_sd`）。
- 词条文件：`term_<format_name>.md`（符合 01 清单 G01）；使用指南：`<format_name>_usage.md`。
- 替换示例：`<format_name>` → `my_new_format` 时，目录为 `quantization_format/my_new_format/`，文件为 `term_my_new_format.md` 与 `my_new_format_usage.md`。

## 操作步骤

1. **确认命名与同名判定**：检查 `docs/zh/knowledge_base/quantization_format/<format_name>/`。目录不存在 → 按新增；目录在但为空 → 复用继续新增；目录在但只缺词条或使用指南之一 → 补插缺失文件；同名 `term_<format_name>.md` / `<format_name>_usage.md` 已存在 → 转「更新」职责，不重复覆盖用户内容。
2. **建子目录**：`create_subdir: true` 时创建 `<format_name>/`。
3. **插入模板**：将 `templates` 中模板复制为目标文件 `term_<format_name>.md` 与 `<format_name>_usage.md`，**保留 `{{ }}` 占位符与 `[OPTIONAL]` 标记**，供专家填充。
4. **专家填充**：替换占位符；词条填写模式支持与交付件，支持表「量化模式词条」列优先链到 `quantization_mode/` 现行子目录下对应 **`term_*.md` 词条**（如 `linear_layer_quantization/term_w8a8_static.md`）；类别索引可链 `fa_quantization/README.md` 等；无独立词条则链 `quantization_mode/term_quantization_mode.md`（总览），禁止空 `[]()`、禁止链向不存在的旧名（如 `w8a8_static.md`、`quantization_mode.md`）。使用指南写清确认模式支持（或适配）/ 配置 / 执行，并与词条互相链接；删除模板注释。关联词条用列表写法：`- [词条名](路径)：关系类型，说明`（关系类型取 01 清单 L03 枚举）。同步更新兄弟格式词条交叉链接（关系类型可用「其他」表示并列格式），以及 `iformat_integration_guide.md` 中对本格式的引用（若适用）。
5. **更新总词条**：在 `quantization_format/README.md` 中：（1）格式地图表新增一行（词条链到 `term_<format_name>.md`、使用指南链到 `<format_name>_usage.md`，相对路径可解析）；（2）「关联词条」补充该格式；（3）「关联流程」补充其使用指南（若适用）。
6. **校验**：分别执行本场景特有清单（QF-*）、词条清单中**适用于量化数据格式的条目**、流程指南清单与公共清单，合并输出结论表（编号 / 级别 / 文档 / 条目 / 结论 / 理由）；任一 ERROR 未解决即不通过。对 01 清单中算法百科专属硬性要求（如强制「数学描述」专节、强制 H2 前 `---`），量化数据格式词条可按不适用记 N/A；**G01 文件命名 `term_*` 对格式词条适用，不得记 N/A**。
7. **修复至通过**：按 CE → 类型清单 → QF 优先级修复；WARN 提示用户是否处理。

## 场景特有校验清单

| 编号 | 级别 | 校验条目 |
| --- | --- | --- |
| QF-01 | ERROR | 《量化格式》README 格式地图中存在新格式的词条与使用指南链接，且相对路径可解析；「关联词条」「关联流程」已同步纳入该格式（若适用） |
| QF-02 | ERROR | 子目录同时存在 `term_<format_name>.md` 与 `<format_name>_usage.md` |
| QF-03 | ERROR | 使用指南操作步骤覆盖三类：**确认模式支持**（标题可用「适配」等等价表述）、**配置**、**执行**（标题或正文可明确识别即可） |
| QF-04 | ERROR | 格式词条包含「量化模式支持」与「交付件 / 导出产物」相关章节或等价表格 |
| QF-05 | ERROR | 词条与使用指南互相链接；总词条、接入指南，以及已引用本格式的兄弟词条/使用指南中，无指向本格式的断链（目标须为 `term_<format_name>.md` 或现行路径） |
| QF-06 | WARN | 若代码已注册该格式 `type`，接入指南或词条中实现位置与配置 `type` 字符串一致 |
| QF-07 | ERROR | 支持表「量化模式词条」列均为可点击相对链接，禁止空链接占位 `[]()`。优先指向 `quantization_mode/` 下现行 `term_*.md`（或类别 `README.md`）；若目标文件尚不存在，允许统一指向 `quantization_mode/term_quantization_mode.md`（总览）并建议顶部 TODO 登记，不得留悬空路径 |

## 校验依据

- **场景特有校验清单**：上文 QF-*。
- **文档类型清单**：词条用 [01 清单](../../../docs/zh/contributing/development_guide/docs_standards/01_term_glossary_checklist.md)（格式词条适用 G01 `term_*`；数学描述专节 / H2 前 `---` 等可按步骤 6 N/A）；使用指南用 [02 清单](../../../docs/zh/contributing/development_guide/docs_standards/02_process_guide_checklist.md)。
- **公共兜底**：[《公共校验清单》](../../../docs/zh/contributing/development_guide/docs_standards/00_common_checklist.md) 的 CE-01~CE-06、CW-01~CW-02。
- 三类清单分别逐条执行并综合结果；校验结论与理由须在日志（对话报告表）中呈现。

## 编写注意事项

- 操作步骤须含同名判定（见步骤 1）。
- 场景特有清单仅列本场景特有项，不重复 01/02 清单条目。
- 总词条格式地图与关联词条 / 关联流程更新属于本场景必做步骤，不得省略。
- 交叉链接（兄弟格式关联词条、iformat 术语表等）属于 QF-05 范围，不得只改总词条而遗留断链。
- 参考现网样例：`ascendv1/term_ascendv1.md`、`compressed_tensors/term_compressed_tensors.md`、`mindie_sd/term_mindie_sd.md`。
