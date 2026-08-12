# 校验规则

按文档类型对照 `docs/zh/contributing/development_guide/docs_standards/` 下对应校验清单执行校验。**新增场景校验 = 场景特有校验清单（若配置）+ 文档类型清单 + 《[公共校验清单](../../../docs/zh/contributing/development_guide/docs_standards/00_common_checklist.md)》，分别执行并综合结果**（见 [scenarios/ 场景文档](../scenarios/README.md)，场景特有校验清单构成见下文「校验清单构成」节）。

## 校验清单构成

- **《[公共校验清单](../../../docs/zh/contributing/development_guide/docs_standards/00_common_checklist.md)》**：适用于所有文档，含通用 ERROR（CE-01~CE-04：占位符 / 模板注释 / 敏感信息 / 标题）与 WARN（CW-01~CW-02：可选章节 / 渲染与链接）。
- **文档类型清单**：`NN_*_checklist.md`，类型专属条目。
- **场景特有校验清单**：可选。场景文档正文"场景特有校验清单"小节，仅含该场景特有、文档类型清单未覆盖的条目（如场景独有章节、专属字段、特殊命名），不重复类型清单；校验时与文档类型清单**分别执行并综合结果**。

## 文档类型 → 校验清单

| 文档类型 | 文档类型清单 |
| --- | --- |
| 术语词条 | 《[术语词条校验清单](../../../docs/zh/contributing/development_guide/docs_standards/01_term_glossary_checklist.md)》 |
| 流程指南 / 使用指南 | 《[流程指南校验清单](../../../docs/zh/contributing/development_guide/docs_standards/02_process_guide_checklist.md)》 |
| 案例参考 | 《[通用案例校验清单](../../../docs/zh/contributing/development_guide/docs_standards/03_general_case_checklist.md)》 |
| 量化配置文档 | 《[量化配置文档校验清单](../../../docs/zh/contributing/development_guide/docs_standards/04_quantization_config_document_checklist.md)》 |
| CLI 文档 | 《[命令行 API 文档校验清单](../../../docs/zh/contributing/development_guide/docs_standards/05_cli_api_contract_checklist.md)》 |
| 模板体系外（安装指南 / 快速入门 / FAQ 等） | 无文档类型清单，以《[公共校验清单](../../../docs/zh/contributing/development_guide/docs_standards/00_common_checklist.md)》的 ERROR 条目（CE-01~CE-04）为兜底 |

模板体系外文档校验口径见规范《[msModelSlim 资料规范](../../../docs/zh/contributing/development_guide/docs_standards/README.md)》第 2.2 节：至少满足《[公共校验清单](../../../docs/zh/contributing/development_guide/docs_standards/00_common_checklist.md)》的 ERROR 条目。

## 公共清单执行口径

- **CE-02（模板注释）**：覆盖模板自带的教学性 / 脚手架章节，不仅限 `> **注释：**` 块引用。发布文档中此类"指导作者写作"的模板指令章节应删除；确属发布内容需保留的，须改写为正式说明并去除占位示例。判定依据见《[公共校验清单](../../../docs/zh/contributing/development_guide/docs_standards/00_common_checklist.md)》 CE-02。
- **CW-01（可选章节）**：删除不适用 `[OPTIONAL]` 章节时，其后章节按顺序重新编号、不留空号；保留的章节剥除 `[OPTIONAL]` 前缀。判定依据见《[公共校验清单](../../../docs/zh/contributing/development_guide/docs_standards/00_common_checklist.md)》 CW-01。

## 校验报告格式

逐条输出表格：

```text
| 编号 | 级别 | 文档 | 条目 | 失败原因 | 修复建议 |
| CE-01 | ERROR | term_xxx.md | 模板占位符 | 残留 {{ term_name }} | 替换为实际内容 |
```

- **ERROR**：必须修复，任一 ERROR 未解决即校验**不通过**。
- **WARN**：建议修复，仅 WARN 不满足时结论为**需修订**。
- 全部条目满足：**通过**。

## 修复优先级

先公共后类型：CE 条目优先；再按文档类型清单编号顺序修复（词条 TE- / TW- 序；流程 PE- / PW- 序；案例 GC 序；配置 QE- / QW- 序；CLI LE- / LW- 序）。ERROR 条目优先于 WARN 条目，避免后续条目连锁失败。

## 豁免机制

某条目确有合理原因无法满足时，在文档顶部以 HTML 注释声明：

```text
<!-- waiver: TE-28 原因：纯概念术语无公式 -->
```

校验时跳过对应条目并记录豁免原因。豁免不得用于规避 ERROR 类条目超过2项，WARN 条目不设豁免数量限制。

## 词条待建 TODO 登记（统一动作，TE-33 / TW-05）

**所有词条的统一动作**：新增词条时，凡 `## 关联词条` 链接到尚未创建的 `term_*.md`，一律在**词条文件顶部以 TODO 列表登记**（如 `<!-- TODO: term_xxx.md 待建 -->`）。该要求适用于一切词条类文档，是 skill 通用规则，不写入具体场景文档。

校验时：已登记待建词条豁免 TE-33（ERROR）与 TW-05（WARN）的存在性检查；未登记则报告 ERROR，提示登记或修正链接。不设中心化待建清单。

**与 TE-17 的分工**：TE-17（链接路径可解析，ERROR）只校验相对路径能否在仓库内解析——待建词条链接路径合法即满足，与待建登记无关。目标文件**存在性**由 TE-33 / TW-05 管：已登记 TODO 的待建词条链接，其目标文件不存在属"已规划"，TE-33 通过、TW-05 豁免，仅记录待建，**不**报 TE-17 存在性类 ERROR。

## 链接校验基准

相对链接以文档**上库后的最终位置**（target_dir）为基准解析，已上库的相对路径为准。文档在模拟 / 临时目录（如试运行、草稿）时仍按最终位置校验，不因临时所在位置改变链接写法；插入模板时按 `target_dir` 相对深度书写链接。模板注释中的相对路径示例按上库后位置书写。

## 校验触发时机

- 新增资料填充完成后立即全量校验。
- 修改既有文档后做增量校验（仅校验变更文件及受其引用影响的文件）。
- 删除 / 下线文档后做断链复查（见 [SKILL.md 职责 3](../SKILL.md#职责-3--删除--下线资料完整流程)）。
