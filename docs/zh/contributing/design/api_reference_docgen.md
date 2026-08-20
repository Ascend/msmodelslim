# 接口文档自动生成设计

本文说明 `skills/docs-management/scripts/` 如何从源码抽出**量化配置**接口文档（模板 04）。用法见《[接口文档生成器](../../../../skills/docs-management/scripts/README.md)》；模板与校验清单见《[量化配置文档模板](../development_guide/docs_standards/04_quantization_config_document_template.md)》。

命令行 API（模板 05）不走生成器：对照 CLI 实现与 `--help` 撰写，见《[命令行 API 文档模板](../development_guide/docs_standards/05_cli_api_contract_template.md)》与 `docs/zh/api_reference/cli/`。

## 1. 背景与目标

配置接口文档必须与实现一致：字段名、类型、默认值、互斥关系和 `type` 分派都来自代码。手工维护会在新增处理器、改校验器后立刻过期。

生成器要同时满足：

- **一结构体一份文档**，结构对齐模板 04，便于检索。
- **机器可核对**：`--check` 对比仓库中的 Markdown 与当前源码，CI 或本地可发现漂移。
- **中文可读**：含义与约束来自源码注解；生成器不编造字段说明。
- **可单测**：渲染与注解解析不依赖导入完整 `msmodelslim`。

不生成使用步骤。操作指南仍在 `docs/zh/user_guide/`，接口文档只回答「能填什么、默认是什么、和谁互斥」。

## 2. 抽取规则

生成器只搬运源码里已经存在的契约，不编造接口语义。缺内容时改注解，不改脚本里的中文对照表。

### 2.1 内部类格式

配置类若声明了唯一的 YAML `type` 字面量，且该字面量以 `_` 开头，则视为**内部类**。这是唯一的机器判定规则，例如 `_auto_save`、`_adapt_rotation_stage1`、`_abstract_flex_smooth_base_`。

| 判定 | 生成器行为 |
|------|------------|
| 内部 `type` 且类是公开分派基类（`AutoProcessorConfig` / `QuantFormatConfig`） | 不单独成文，不进入 `process[]` / `save[]` 索引 |
| 内部 `type` 且类是公开配置的嵌套字段载体（如 stage1/stage2） | 可作为嵌套配置成文；示例不得把该 `type` 写成用户可填的分派名 |
| 公开 `type`（不以 `_` 开头） | 独立成文，进入对应分派索引 |

用户可复制的示例里禁止出现内部 `type`：`_auto_save` 替换为第一个公开保存格式；其它内部 `type` 从示例中删除（由父配置写入）。

新增内部实现时：给 `type` 以 `_` 开头的 Literal，不要再向生成器加跳过名单。

### 2.2 内容从哪来

| 文档位置 | 源 | 缺失时 |
|----------|----|--------|
| 参数「含义」 | `Field(description=)` | 写 `—`，去补 description，禁止在生成器写死中文 overlay |
| 配置概述 | 类 docstring 首段 | 回退「这是××配置，位于××路径」，去补类 docstring |
| 配置约束（参数列表内每个配置块的子项） | field / model validator 的 docstring | 「无。」，去补 validator docstring |
| 取值范围中的数字与枚举 | JSON Schema：`enum` / `const` / `minimum` / `maxLength` 等；对应写在 `Field(ge=…, max_length=…)`、`Literal`、`Enum` | Schema 没有的约束（如纯 `AfterValidator`）不会出现，应改成 Field 约束 |
| 类型、默认值、必选 | JSON Schema 的 `type` / `default` / `required`；`default_factory` 未进 Schema 时回退 Field | 与实现一致 |

## 3. 总体架构

生成器拆成两层：

| 层 | 文件 | 职责 |
|------|------|------|
| 纯抽取 / 渲染 | `quant_config_docgen.py` | JSON Schema → FieldRecord → Markdown。不导入 `msmodelslim`。 |
| 产品装配 | `gen_quant_config_docs.py` | 发现真实类、算 YAML 路径、写文件；仅 `--update-mkdocs` 时改 `mkdocs.yml`。 |

```text
Pydantic 配置类
        │
        ▼
  model_json_schema（文档用 Generator）
        │
        ▼
  quant_config_docgen.py（JSON Schema → FieldRecord → Markdown）
        ▲
        │ 装配路径 / 引用图 / 示例外壳
        │
  gen_quant_config_docs.py
        │
        ▼
  docs/zh/api_reference/config/<分类>/*.md
```

> 具体抽取目标、展开策略与后处理由驱动脚本 `gen_config_api_docs.py` 控制；`gen_quant_config_docs.py` 提供 `--targets` / `--expand-nested` / `--prune` 等基础能力。

生成文件第一行带 HTML 标记（`generated-by: skills/docs-management/scripts/gen_quant_config_docs.py`）。带该标记的文件视为生成稿，应改源码后重新生成，不要直接改 Markdown。

CLI 文档不在本生成器范围内，也不使用 `generated-by` 标记。

手写保留：`processor_group.md`、`auto_precision_tuning.md`。生成器写入前跳过这两个文件名。

## 4. 量化配置生成器

### 4.1 要做什么

把每个对外 Pydantic 配置类渲染成一份模板 04 文档，按类型子目录组织输出。对外类包括：

- 任务根配置：`ModelslimV1QuantConfig` 等四个 `apiversion` 入口，以及任务基类实践配置 `PracticeConfig`（`BaseQuantConfig` 子类，含 `metadata` → `Metadata`）。
- 服务规格：`*ServiceConfig`。
- 处理器：`AutoProcessorConfig` 注册表中 `type` 不以 `_` 开头的子类。
- 保存格式：`QuantFormatConfig` 的公开子类。
- 自动调优：`TuningPlanConfig`、具体策略与评估服务。
- 从上述类的字段注解递归到的嵌套 BaseModel（如 `QConfig`、`Metadata`）。
- 权重转换侧额外种子：`RenamePreprocessConfig` 等。

输出目录：`docs/zh/api_reference/config/`，按分类放入子目录：

| 分类 | 子目录 |
|------|--------|
| 任务配置 | `task/` |
| 服务规格 | 默认不单独成页：展开进对应 task 页的「参数列表」；基础脚本独立成文时放 `spec/` |
| 处理器 | `processor/` |
| 保存格式 | `format/` |
| 自动调优 | `tuning/` |
| 嵌套配置（仅基础脚本独立成文时） | `nested/` |

不再生成 `config/README.md` 索引页；入口文档（使用指南）直接链接到各分类下的具体页面。

### 4.2 为什么这样切

YAML 是「根配置 → spec → process[] / save[] → 嵌套对象」的树，但用户检索是按 `type`（`fa3_quant`、`linear_quant`）或类名。一份巨型 YAML 说明无法按处理器查阅，也无法给「被引用的配置」做反向链接。

因此采用 **一模型一文档 + 引用图（正向）**：每份文档只回答「本类字段指向哪些配置」，`type` 分派字段（`process`、`save`、`strategy`、`evaluation`、`select_best`、`operations`、`preprocess` 等）用「基础类块 + 派生类列表 + 派生类子块」表达：基础类块说明该字段按 `type`/`mode` 分派并列出全部公开子类型，而不是只写基类名。被引用信息收敛到参数表的「引用配置」列，不再单设「引用的配置 / 被引用的配置」小节。

内部配置按[第2.1节](#21-内部类格式)判定，不把内部 `type` 写进公开分派索引。驱动脚本默认把嵌套内部配置展开进上级文档「参数列表」内对应类名子块，减少跳转；task 根配置同时把对应的 spec（服务规格）展开进「参数列表」，task 与 spec 合并为一页，不再单独生成 `spec/` 页面。`PracticeConfig` 是 `BaseQuantConfig` 子类（任务配置），虽然会被调优策略引用，仍独立成 `task/practice_config.md` 页，在任务侧集中展示 `apiversion + spec + metadata`（`Metadata`）的完整实践配置结构。

### 4.3 如何实现

`build_records()` 按固定顺序工作。

#### 4.3.1 发现类

1. 导入已知模块，让处理器 / 保存格式完成注册。
2. 种子 = 五个根配置（四个 `apiversion` 入口 + `PracticeConfig`） ∪ 公开处理器 ∪ 公开保存格式 ∪ 转换侧额外模型 ∪ 自动调优目标（`TuningPlanConfig` + 公开策略 + 评估服务）。
3. 对每个种子，遍历 `model_fields`，用 `collect_nested_models()` 从 `Optional` / `List` / `Union` / `Annotated` 中取出嵌套 BaseModel，递归加入。
4. `_should_keep()` 过滤：非 Pydantic、跳过名单、测试模块、内部 `type`、`AutoSaverBaseConfig` 中间层。

公开处理器来自 `AutoProcessorConfig._registry`，不是 `__subclasses__()`，以免漏掉未形成继承树但已注册的实现。保存格式来自 `QuantFormatConfig.__subclasses__()`。

#### 4.3.2 分类与文件名

| 判定 | 目录分组 |
|------|----------|
| `BaseQuantConfig` 子类 | 任务配置 |
| 类名以 `ServiceConfig` 结尾（非自动调优评估服务） | 服务规格 |
| `QuantFormatConfig` 子类 | 保存格式 |
| `AutoProcessorConfig` 子类 | 处理器 |
| 自动调优目标（`TuningPlanConfig`、具体策略、评估服务） | 自动调优 |
| 其余 | 嵌套配置 |

文件名（slug）优先用字段 `type: Literal["xxx"]` 的唯一字面量；否则 snake_case 类名。五个任务配置（四个 `apiversion` 入口 + `PracticeConfig`）和部分重名 `QuantStrategyConfig` 用覆盖表固定文件名，避免链接抖动。

#### 4.3.3 YAML 路径

从根配置 BFS：父路径 + 字段名，列表字段加 `[]`。字段注解指向 `AutoProcessorConfig` 时，所有公开处理器共享 `spec.process[]`；指向保存器时共享 `spec.save[]`。

个别嵌套类若从多条路径可达，用覆盖表钉死用户最常见的路径，例如 `QConfig` → `spec.process[].qconfig.weight`。未从根走到的类回退为 `spec.process[]` / `spec.save[]` / `spec`。

字段表里的「字段路径」= 字段名本身（相对当前配置），不叠加整条 YAML 路径。子标题不再显示 `（ctx_path）` 路径后缀；配置的归属关系由「参数列表」的块层级（普通嵌套平铺、type 分派的基础类块 + 派生类子块）表达。

#### 4.3.4 字段抽取

`extract_fields()` 取 `model_json_schema(mode="serialization")`，再映射为参数表。序列化 Schema 与 YAML 形状一致：用 alias（如 `from_` → `from`），不含 `exclude=True` 字段。`computed_field` 即使进了 Schema 也会丢掉，因为用户不能在 YAML 里填写它们。

| 表格列 | 来源 |
|--------|------|
| 类型 | Schema `type` / `anyOf` / `$ref`，映射为 `string` / `int` / `object` / `list[…]` / `… / null` |
| 必选 | Schema `required` |
| 默认值 | Schema `default`；没有时回退 Field / `default_factory`（工厂值常不进 Schema） |
| 取值范围或格式 | Schema `enum` / `const` / `minimum` / `maxLength` / `pattern` 等 |
| 含义 | Schema `description`（即 `Field(description=)`）；没有则为 `—` |
| 引用配置 | Schema `$ref` 对应的嵌套模型（展开进本页时写页内 HTML 锚点 `<a href="#anchor">`；独立页面写相对链接）；type 分派字段统一指向基础类块锚点「本页 <a href="#anchor">§x.y</a>」 |

`AfterValidator` **不会**进入 JSON Schema。长度、区间必须写成 `Field(max_length=N)` / `ge` / `le`，或 `Literal` / `Enum`，才会出现在文档里。生成器不再解析校验闭包，也不维护函数名到中文的对照表。

无法 JSON 化的类型（如 `torch.dtype`）在文档用 Generator 里回退为 `string`，以免整份 Schema 失败；若该字段是 `computed_field`，随后仍会被丢掉。

配置约束（参数列表内每个配置块的子项）来自 `__pydantic_decorators__` 上 field / model validator 的 docstring。没有 docstring 则该块写「无。」。类概述优先用类 docstring 首段；没有则回退「`X` 是Y类别。」。展开进参数列表的嵌套配置块（`<h3>` / `<h4>`）若类 docstring 首段非空，也会在标题后、参数表前输出该句类概述（如 `Metadata` 的“量化配置元数据：…”）。

发现类、YAML 路径、引用图仍读 Python 注解与注册表：JSON Schema 里 `process` 的 items 只 `$ref` 到分派基类，不会展开全部公开处理器。

#### 4.3.5 引用图

每个类先算「引用的配置」`nested_refs`（正向图）：

- 字段注解指向 `AutoProcessorConfig`：对该字段列出全部公开处理器，关系为 `type 分派`。
- 指向保存格式：列出全部公开保存器，关系为 `type 分派`。
- 指向策略 / 评估服务基类：列出公开策略 / 评估服务，关系为 `type 分派`。
- 单一基础类 + 已注册派生类（如 `TLQOpConfig` → `MinmaxTuneOpConfig` / `RoundTuneOpConfig` / `TrainableSmoothOpConfig`）：派生类作为同路径 `嵌套对象` 引用，使该字段成为分派组。
- `ModelslimConvertServiceConfig.preprocess`：`rename` / `convert` 两类预处理步骤，关系为 `type 分派`，分派基础类名合成 `PreprocessConfig`。
- 普通嵌套 BaseModel：一条 `嵌套对象`。处理器 / 保存器子类在这条路径上跳过，避免与分派展开重复。

不再渲染「被引用的配置」小节（内部仍保留 `parents` 反查表供示例 YAML 上溯宿主）。展开嵌套时，BFS 从 `nested_refs` 出发，把可达的嵌套配置按引用处上下文路径渲染进「参数列表」内对应类名子块；task 页把 spec 一并展开。对未独立成页的 spec 的引用（如调优策略的 `template` 指向 `ModelslimV1ServiceConfig`）重定向到所属 task 页的 2.2 小节锚点（task 页块序固定：根块 §2.1，spec 为首个嵌套块 §2.2）；跨目录链接按输出子目录计算相对路径。

**type 分派渲染**：同一字段路径下引用数 ≥2 或含 `type 分派` 即判为分派组。渲染成一个 `<h3 id="…">2.x 基础类名（按 type 分派）</h3>` 块：块内含基础类参数表（基础类本身是真实模型时）与「派生类」列表（每项 = 类名、`type` 值、一行说明、本页子块锚点或独立页相对链接），各派生类参数表 + 配置约束作为 `<h4 id="…">2.x 派生类名</h4>` 子块排在其后。同一页面同一基础类只渲染一次，后续分派字段的「引用配置」列别名指向该块锚点（例如 task 页的 `spec.process[]` 与 `spec.prior[].process` 共用同一个 `AutoProcessorConfig` 块）。「引用配置」列对分派字段统一写「本页 <a href="#anchor">§x.y</a>」指向基础类块锚点，不再逐个子类型平铺、也不再写「按 `type` 分派，见对应配置文档」汇总文案。配置块标题带可见编号，且标题与页内跳转统一用 HTML 标签（`<h3 id>` / `<h4 id>` / `<a href="#…">`），不用 Markdown 的 `{#anchor}` 属性语法。

#### 4.3.6 示例 YAML

「完整配置参考」按模板 04 / QE-05 生成：语法正确、放在真实字段路径、可被根配置加载。抽不出满足这些条件的示例则整节省略。

1. 优先用模型级 JSON Schema `examples` / `example`。
2. 否则从默认值、唯一 `Literal`/`Enum`、以及参数表中的枚举 / 数值下界抽出**本配置**必选字段和带默认值的可选字段。枚举跳过 `placeholder`，并优先取典型值（如 `int8`、`per_channel`）。无约束的必选字符串不编造 `"example"`；量化算法名 `method` 取仓库中已注册的典型值 `minmax`。
3. 用 `wrap_example_at_path()` 套进 `apiversion` + `spec`：路径中带 `[]` 的段写成单元素列表（`spec.process[]`、`spec.save[]`、`spec.prior[]` 等）。嵌套配置挂到引用它的公开 `type` 宿主下，沿「被引用的配置」上溯；兄弟 Literal（如 `stage: 1`）标明分派。任务根配置的 `apiversion` 取服务类 Literal，禁止写成 `Unknown`。内部 `type` 的 Union 载体（如 `stage_config`）展平到父级，与加载器实际接受的 YAML 一致。
4. 内部 `type` 仍经 `rewrite_example_internal_types()` 处理。`AutoSaverConfigList` 不允许空 `save` 时补一层公开保存格式。
5. 用对应任务根配置 `model_validate` 校验；失败则尝试给空的嵌套对象列表补一项后再校验，仍失败则不输出第 3 节（完整配置参考）。

**自动调优示例（无 apiversion 根）**：`TuningPlanConfig` 没有 `apiversion` 字段，且其 `strategy` / `precheck` 分派依赖插件注册表，docgen 环境未注册插件时整体校验会失败。因此各策略 / 评估类在类内通过 `model_config.json_schema_extra.examples` 声明**自身子树的完整示例**（参考 `docs/zh/knowledge_base/tuning_strategies/` 下的配置，去掉触发插件分派的可选 `precheck` 块）。生成器为 tuning 根走专用路径：取页面自身类的子树示例 + 兄弟类型的默认子树（strategy 页配默认 evaluation，evaluation 页配默认 strategy，`TuningPlanConfig` 取两者默认），组装成完整的 `strategy` + `evaluation` 形态；校验时先尝试 `TuningPlanConfig.model_validate`，失败则回退为对具体策略 / 评估类分别 `model_validate`。

#### 4.3.7 导航

默认不改 `mkdocs.yml`。仅当显式传入 `--update-mkdocs` 时，才替换 `# BEGIN GENERATED QUANT CONFIG NAV` … `# END GENERATED QUANT CONFIG NAV`。分组顺序固定为：任务配置、服务规格、处理器、保存格式、自动调优、嵌套配置。手写的组合处理器与自动调优条目接在生成块末尾。不再生成 `config/README.md` 索引页。

开发树里 `config.ini` 在仓库根 `config/`，包代码期望 `msmodelslim/config`。生成前若缺失则建临时 symlink，结束后删除，避免导入失败。

## 5. 命令行 API 文档（不自动抽取）

命令行接口文档对照 `msmodelslim.cli` 实现与 `--help`，按模板 05 撰写，不从 argparse 生成。现稿：`docs/zh/api_reference/cli/msmodelslim_{quant,analyze,tune}.md`。改 CLI 行为时同步改对应 Markdown，提交前按《[命令行 API 文档校验清单](../development_guide/docs_standards/05_cli_api_contract_checklist.md)》检查。

## 6. 模板章节对照

### 6.1 模板 04（配置）

| 章节 | 自动来源 | 缺口 |
|------|----------|------|
| 1 概述 | 类 docstring；类名与源码相对路径 | 无类 docstring 时概述偏模板化 |
| 2 参数列表 | JSON Schema；按配置类名分块组织，块标题带可见编号（`2.1 类名`、`2.x 派生类名`）；每个块含参数表与「配置约束」子项；嵌套配置（含 task 的 spec）展开进本节对应类名子块，有类 docstring 时块首输出一句类概述；type 分派字段渲染为基础类块（含参数表与派生类列表）+ 各派生类 `<h4>` 子块；标题与页内跳转用 HTML 标签 | 无 `description` 时含义为 `—`；未写入 Schema 的 AfterValidator 不出现 |
| 3 完整配置参考 | 默认可抽出值 + 真实 YAML 路径外壳，并用根配置校验（条件出现） | 缺必选且无法从约束抽出，或校验失败则整节省略 |
| 4/5 兼容性、注意事项 | 不生成 | 有破坏性变更时需手写或另文 |

### 6.2 模板 05（CLI）

不由生成器填写。章节内容对照 argparse 实现、运行时校验与业务语义撰写；不得保留模板占位符或填写说明。

## 7. 校验与测试

- `--check`：按当前源码重新渲染配置文档，与磁盘文件逐字节比较，不一致打印 `drift:` 并以退出码 1 失败。用于防止生成稿被手工改乱，或源码改了却没重新生成。
- `--dry-run`：只打印将写入的路径。
- 生成器脚本作为 `skills/docs-management/` skill 的一部分维护，不再单独保留单测；正确性通过 `--check` 与文档门禁校验。

## 8. 维护约定

改配置接口行为时同步改源码注解，然后重新生成：

```bash
# 日常重新生成（默认目标 + 展开嵌套 + 清理过期文档）
python3 skills/docs-management/scripts/gen_config_api_docs.py
```

改命令行行为时对照实现更新 `docs/zh/api_reference/cli/`，不运行配置生成器。

| 想改的内容 | 改哪里 |
|------------|--------|
| 字段类型、默认值、choices、必选 | Pydantic Field |
| 配置「含义」 | `Field(description=)` |
| 配置概述 | 类 docstring |
| 跨字段约束 | validator docstring |
| 长度 / 区间数字 | `Field(max_length=)` / `ge` 等（进入 JSON Schema） |
| CLI 参数、互斥、示例、安全说明 | `docs/zh/api_reference/cli/`（模板 05） |
| 完整配置参考 | 默认可抽出值；须能通过根配置校验并位于真实路径。无法抽出则不生成该节 |
| 是否内部类 | `type` Literal 是否以 `_` 开头 |

新增公开处理器：实现并注册到 `AutoProcessorConfig`，保证 `type` 不是 `_` 前缀，重新生成即可出现文档、索引和所有 `process[]` 分派表。不必改生成器。
