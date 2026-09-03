# 接口文档生成器

从源码注解生成 `docs/zh/api_reference/config/` 下的量化配置文档（模板 04）。请改源码注解后重新生成，不要手工改带 `generated-by` 标记的配置文档。

命令行 API 文档（模板 05）**不自动抽取**：对照 `msmodelslim.cli` 实现与 `--help`，按《[命令行 API 文档模板](../../../docs/zh/contributing/development_guide/docs_standards/05_cli_api_contract_template.md)》撰写，提交前按《[命令行 API 文档校验清单](../../../docs/zh/contributing/development_guide/docs_standards/05_cli_api_contract_checklist.md)》检查。现稿在 `docs/zh/api_reference/cli/`。

配置生成器的设计说明见《[接口文档自动生成设计](../../../docs/zh/contributing/design/api_reference_docgen.md)》。内部类以 YAML `type` 的 `_` 前缀判定；配置「含义」只来自 `Field(description=)`，跨字段约束只来自 validator docstring。

## 量化配置（模板 04）

从 Pydantic `model_json_schema()` 生成 `docs/zh/api_reference/config/`，并按类型放入 `task/`、`processor/`、`format/`、`tuning/` 等子目录。服务规格（spec）不再单独成页，而是随对应 task 页展开。字段类型、枚举、边界以 JSON Schema 为准。

日常重新生成使用驱动脚本（默认目标 + 展开嵌套 + 清理过期文档）：

```bash
python3 skills/docs-management/scripts/gen_config_api_docs.py
python3 skills/docs-management/scripts/gen_config_api_docs.py --dry-run
python3 skills/docs-management/scripts/gen_config_api_docs.py --check
```

基础脚本 `gen_quant_config_docs.py` 提供目标 / 展开 / 清理等基础能力，供需要精确控制时使用：

```bash
python3 skills/docs-management/scripts/gen_quant_config_docs.py --targets processor --dry-run
python3 skills/docs-management/scripts/gen_quant_config_docs.py --check
```

| 参数 | 含义 |
|------|------|
| `--output` | 输出目录，默认 `docs/zh/api_reference/config` |
| `--dry-run` | 只打印将写入的路径 |
| `--check` | 对比已生成文档与当前注解，不一致时退出码为 1 |
| `--update-mkdocs` | 写入 `mkdocs.yml` 导航；默认不改该文件 |

取值范围只收录 JSON Schema 里的 `enum` / `const` / `minimum` / `maxLength` 等。写在 `AfterValidator` 闭包里的长度或区间不会进入文档，应改成 `Field(max_length=)` / `ge` / `le`。

生成规则：

- 每个对外 Pydantic 配置类生成一份 Markdown；`type` 以 `_` 开头的内部配置不生成。
- 文档按类型子目录组织：任务配置 `task/`、处理器 `processor/`、保存格式 `format/`、自动调优 `tuning/`；服务规格随 task 页展开，不单独生成 `spec/` 页面。
- 参数表「字段路径」为相对当前配置的字段名；子标题不显示 YAML 路径后缀。
- 参数列表按配置类名分块组织，块标题带可见编号（如 `<h3 id="2-1-...">2.1类名</h3>`、`<h4 id="...">2.x 派生类名</h4>`）；每个块含自身参数表与「配置约束」子项；嵌套配置（含 task 的 spec）展开进对应子块；嵌套块若类 docstring 首段非空，标题后先输出该句类概述。
- `PracticeConfig`（`BaseQuantConfig` 子类，含 `metadata` → `Metadata`）作为任务基类配置独立成 `task/practice_config.md` 页，被调优策略引用时链接到该页。
- type/mode 分派字段（`process`、`save`、`strategy`、`evaluation`、`select_best`、`operations`、`preprocess` 等）渲染为基础类块（`<h3 id="…">基础类名（按 type 分派）</h3>`，含基础类参数表与「派生类」列表）+ 各派生类 `<h4 id="…">` 子块；同一页面同一基础类只渲染一次，分派字段的「引用配置」列统一指向基础类块锚点。
- 配置块标题与页内跳转统一用 HTML 标签（`<h3 id>` / `<h4 id>` / `<a href="#…">`），不使用 Markdown 的 `{#anchor}` 属性语法。
- 不再生成 `config/README.md` 索引页。
- 文件名优先使用公开 `type`（如 `linear_quant.md`），否则使用类名的 snake_case。
- 顶层任务配置保留稳定文件名：`modelslim_v1.md`、`multimodal_vlm_modelslim_v1.md`、`multimodal_sd_modelslim_v1.md`、`modelslim_convert.md`、`practice_config.md`。
- 手写文档 `processor_group.md`、`auto_precision_tuning.md` 不会被覆盖。
- 「完整配置参考」仅当能抽出可加载的完整 YAML（真实字段路径 + 根配置校验通过、`apiversion` 不为 `Unknown`）时生成；自动调优配置无 `apiversion` 根，改用类内 `json_schema_extra.examples` 声明的子树示例组装成完整的 `strategy` + `evaluation` 形态。
- 生成文件带 HTML 注释标记 `generated-by: skills/docs-management/scripts/gen_quant_config_docs.py`。
