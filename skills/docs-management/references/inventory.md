# 资料盘点规则

## 目的

扫描 `docs/zh/`，输出资料清单，反映资料现状与结构。

## 步骤

1. 遍历 `docs/zh/` 各一级目录：`api_reference`、`best_practices`、`contributing`、`install_guide`、`knowledge_base`、`legal`、`quick_start`、`release_notes`、`support`、`user_guide`。
2. `knowledge_base/` 按领域子目录分类：`model`、`quantization_algorithms`、`quantization_format`、`tuning_strategies`、`ptq`、`parallel`。
3. 按区域分类统计，每个区域下列出文档文件与子目录。
4. 对每个文档标注：
   - 所属区域。
   - 文档类型：5类模板文档（术语词条 `term_*.md` / 流程指南 / 案例参考 / 量化配置文档 / CLI 文档）、自由文档、README。
   - 是否模板体系内（对应哪个模板 / 校验清单）。

## 输出格式

按区域分组，以表格汇总：

```text
| 区域 | 文档 | 类型 | 备注 |
```

## 注意

- 只盘点 `docs/zh/` 中文树。
- 模板与校验清单本身（`docs_standards/` 下）属于治理层文档，单独标注。
- 词条按领域就近放置于 `knowledge_base/<领域>/` 下，文件名为 `term_<english_name>.md`；盘点时归入所属领域目录。
- 盘点只报告现状，不修改任何文档。
