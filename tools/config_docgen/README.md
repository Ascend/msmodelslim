# 配置文档生成工具

`tools/config_docgen/` 是量化配置文档生成的唯一项目级工具入口，也是本仓库配置文档生成、注解规范、维护流程和故障排查的唯一完整手册。其他开发者文档、skill 文档和脚本目录只链接到本手册，不再维护重复规范。

生成器从 Pydantic 模型、字段注解、配置类 docstring、validator 文档约束和 JSON Schema 抽取配置事实，并从 `templates/` 的 YAML 读取完整示例和场景，生成 `docs/zh/api_reference/config/` 下可版本管理的 Markdown。源码注解是字段语义、概述、约束和注意事项的唯一事实来源；带 `generated-by` 标记的生成稿属于构建产物，禁止手工修改。

覆盖范围为全部量化配置类，包括任务配置、服务规格、处理器、保存格式和自动调优。英文文档和 V0 历史量化模块不纳入本项目级生成链路。

## 1. 设计思路

生成链路分为四步：

1. **发现配置类**：由 `catalog.py` 注册来源模块、分类、slug、公开分派关系和目标关键字；`discovery.py` 导入模块并发现 Pydantic 配置类。
2. **抽取源码事实**：`schema.py` 读取字段、类型、默认值、枚举、`Field` 约束、类 docstring 和 validator 文档约束；`examples.py` 严格加载 YAML 示例与场景。
3. **组装页面模型**：`records.py` 使用 `FieldRecord`、`ScenarioRecord` 和 `ModelRecord` 承载中间结果，计算嵌套引用、派生类关系和相关链接。
4. **渲染并检查**：`renderer.py` 生成稳定编号和 Markdown；`generate.py` 负责写盘、删除陈旧页面、注解审计、逐字节 drift 检查和退出码。

生成过程不读取现有生成稿作为事实来源，因此手工修改生成稿不会反向影响源码；`--check` 会重新生成到临时目录并比较仓库中的文件。

## 2. 入口与接口

在仓库根目录执行：

```bash
# 推荐入口；在仓库根目录执行
python -m tools.config_docgen

# 兼容直接执行入口；可从任意工作目录调用
python tools/config_docgen/main.py

# 一致性门禁
python -m tools.config_docgen --check
```

`__main__.py`、`main.py` 和 `generate.py` 共享同一套参数、抽取逻辑和退出码。新脚本、文档及门禁统一使用 `python -m tools.config_docgen`；`main.py` 仅保留为兼容直接执行入口。

Python 调用接口为：

```python
from tools.config_docgen.main import main

exit_code = main(["--check"])
```

## 3. 输入与输出

输入包括：

- `msmodelslim/core/quant_service/`、`msmodelslim/core/practice/` 中的任务配置和服务规格；
- `msmodelslim/processor/` 中的处理器配置；
- `msmodelslim/format/` 中的保存格式配置；
- `msmodelslim/core/tune_strategy/`、`msmodelslim/infra/` 中的自动调优配置；
- `tools/config_docgen/templates/` 中维护的精选完整 YAML 示例；
- `catalog.py` 中的目录、slug、来源模块和分派关系；
- 配置类的 Pydantic schema、字段注解、docstring、validator 文档约束和 `config_constraint`。

输出为 `docs/zh/api_reference/config/` 下的生成稿，按 `quant/`、`processor/`、`format/`、`tuning/` 分类。每个生成页顶部包含不可手改提示和 `generated-by` 标记；`README.md` 及 `MANUAL_KEEP` 中的手写页不由生成器覆盖。

## 4. 命令行参数

| 参数 | 作用 |
| --- | --- |
| `--output PATH` | 指定输出目录，默认 `docs/zh/api_reference/config`。 |
| `--targets SELECTOR...` | 按分类、类名或 slug 选择目标，可重复选择；拼写错误返回非零退出码。 |
| `--expand-nested` | 将服务规格和嵌套内部配置展开到直接引用它的上级页面，默认开启。 |
| `--no-expand-nested` | 不展开嵌套配置。 |
| `--prune` | 清理不再由当前完整 catalog 生成的旧页面。 |
| `--no-prune` | 保留范围外页面，只覆盖本次选中目标。 |
| `--dry-run` | 只报告将写入或删除的路径，不创建输出目录。 |
| `--check` | 检查注解完整性、生成稿逐字节漂移、示例 YAML 和 stale 生成页；失败退出码为 1。 |

全量生成默认 `--prune`；指定 `--targets` 的局部生成默认 `--no-prune`。局部生成即使显式传入 `--prune`，也只按完整 catalog 清理，不会误删其他有效分类页面。未知 target 返回非零退出码，不允许“检查 0 个文件”通过门禁。

常用示例：

```bash
# 全量重新生成
python -m tools.config_docgen

# 只预览 selected processor 和 quant
python -m tools.config_docgen --targets processor quant --dry-run

# 局部重生成，不清理其他页面
python -m tools.config_docgen --targets LinearProcessorConfig

# 提交前或 CI 一致性门禁
python -m tools.config_docgen --check
```

## 5. 抽取规则

| 文档部分 | 源码事实来源 | 生成规则 |
| --- | --- | --- |
| 页面文件与标题 | 配置类、公开 `type`、`catalog.py` 中的 slug 规则 | 每个公开配置类生成一页；任务配置使用稳定文件名；内部实现不进入公开分派索引。 |
| 配置概述 | 配置类自身 docstring 首段 | 只读取首段；首段内硬换行合并，固定小节和后续自由段落不进入概述；不继承基类 docstring。 |
| 配置说明 | 配置类 docstring 的 `说明` 小节 | 将列表项作为概述后的独立项目列表展示；不混入首段。未加固定标题的自由段落不参与抽取。 |
| 字段路径与含义 | `Field(description=...)` | 每个字段必须有非空描述；别名按输出字段名处理。 |
| 字段类型、默认值、枚举、边界 | Pydantic JSON Schema 和 `Field` 约束 | 类型、默认值、`Literal` / `Enum`、`ge` / `le` / `gt` / `lt`、长度等直接渲染。 |
| 配置约束 | 配置类 docstring 的 `约束` 小节、validator docstring、`AfterValidator.config_constraint` | 按源码声明顺序合并并去重；内部归一化 validator 使用 `@config_internal` 排除。 |
| 注意事项 | 配置类 docstring 的 `注意事项` 小节 | 按源码声明顺序合并并去重；仅当源码显式声明时生成“注意事项”。 |
| 配置场景示例 | `examples.py` 的 `SCENARIO_EXAMPLE_FILES` 注册项和对应 YAML | 有稳定端到端用例时登记 `title`、可选 `description` 和 YAML 文件；未登记不生成。 |
| type/mode 分派 | 公开分派基类和派生配置 | 基础类展示参数表与派生类表格，各派生类在页内展开；公开链接自动识别。 |
| 完整配置参考 | 精选 YAML、确定性默认值和根模型校验 | 依次尝试调优子树装配、`practice_config.yaml`、类名映射的 YAML 和 schema 默认值补全；只有能通过根配置校验才输出。 |
| 相关链接 | 嵌套引用和公开分派 | 仅渲染真实存在且可解析的目标，不生成悬空链接。 |

已知事实源模块导入失败、未发现配置类、unknown target、示例 YAML 非法或生成稿漂移都会返回非零退出码，不允许静默忽略。

## 6. 注解规范

### 6.1 字段语义

每个用户可见字段都必须提供非空 `Field(description=...)`：

```python
from pydantic import BaseModel, Field


class ExampleConfig(BaseModel):
    """示例配置，说明该配置的用途。"""

    num_samples: int = Field(
        default=128,
        gt=0,
        description="校准样本数量；增大该值会提高校准耗时。",
    )
```

字段 `description` 只描述字段本身，不承载整类配置背景、跨字段约束或完整示例。枚举优先使用 `Literal` / `Enum`，数值边界优先使用 `Field(ge=..., le=..., gt=..., lt=...)`，使约束进入 JSON Schema。

### 6.2 配置概述与类级说明

配置类必须提供自身 docstring。首段是“配置概述”，应说明配置对象是什么、解决什么问题、在 YAML 中的使用位置；首段不要依赖基类继承。需要进入生成文档的补充说明必须放在 `说明：` 小节中，未加固定标题的自由段落只保留在源码，不参与抽取：

```python
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ExampleProcessorConfig(BaseModel):
    """示例处理器配置，用于在校准前执行示例变换；位于 process 列表的一项。

    说明：
    - 该处理器只处理指定类型的算子。
    - 多个处理器按 `process` 中的顺序执行。

    约束：
    - `targets` 中的算子必须已在当前模型中注册。
    - `alpha` 和 `beta` 同时为空时，处理器使用默认变换强度。

    注意事项：
    - 该配置会增加校准阶段的内存占用。
    """

    type: Literal["example_processor"] = Field(
        default="example_processor",
        description="处理器类型，固定为 `example_processor`。",
    )
    alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="主变换强度；未设置时由处理器选择默认值。",
    )
    beta: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="辅助变换强度；未设置时由处理器选择默认值。",
    )
    targets: List[str] = Field(
        default_factory=lambda: ["*"],
        description="参与变换的算子名称模式；`*` 表示全部算子。",
    )
```

固定小节约定：

- 只识别 `说明：`、`约束：`、`注意事项：` 三个固定标题；
- 每个小节使用 `-` 列表项，条目可以跨多行；
- 补充说明在“配置概述”中作为独立项目列表展示，不混入首段概述；
- 首段后未加固定标题的说明文字或列表不会进入生成文档，无需额外的“禁止抽取”字段；
- `约束` 与 validator 约束合并到“配置约束”；
- `注意事项` 按源码声明顺序合并到“注意事项”；
- 生成器只读取配置类自身的 docstring，不继承基类内容；
- 字段默认值、枚举和边界不要在小节中重复抄写，应由 Pydantic schema 提供。

### 6.3 validator 约束

Pydantic validator 的 docstring 是跨字段规则的源码说明。能由字段类型或 `Field` 表达的枚举、长度和数值边界不要重复写进 validator。

通用 `AfterValidator` 使用 `@config_constraint(...)` 声明短约束标签：

```python
from msmodelslim.utils.validation.pydantic import config_constraint


@config_constraint("长度不超过 200 个字符")
def validate_name(value, info):
    """校验名称长度。"""
    ...
```

参数化 validator 直接使用装饰器格式化实际值：

```python
def validate_str_length(max_len: int = 200):
    @config_constraint("字符串长度 ≤ {max_len}", max_len=max_len)
    def validator(value, info):
        """字符串长度不能超过上限。"""
        ...

    return validator
```

当同一个 validator 用于列表元素时，生成器会自动补充列表上下文。例如：

```python
include: Optional[List[Annotated[str, AfterValidator(validate_str_length())]]] = Field(
    default=None,
    description="包含的模块名称模式；不设置表示全部匹配。",
)
```

该字段的取值范围会渲染为 `列表元素：字符串长度 ≤ 200；或 null`，明确表示上限作用于每个字符串元素，而不是列表元素个数。

不传格式化参数时，`config_constraint` 按字面值保存约束文本；固定文本中的字面花括号使用 `{{` / `}}` 转义。不要再在函数定义外手工给 `validator.config_constraint` 赋值。

仅用于归一化、默认值补全或兼容转换、不构成用户配置规则的 validator，使用 `@config_internal` 标记，使其不进入用户文档。不得用该标记规避真实的约束文档检查。

### 6.4 场景示例 YAML

类级说明、约束和注意事项只写在当前配置类自身的 docstring 中，不再从 `model_config` 或 `json_schema_extra` 读取。需要展示稳定端到端场景时，在 `examples.py` 的 `SCENARIO_EXAMPLE_FILES` 中登记 YAML 文件：

```python
SCENARIO_EXAMPLE_FILES = {
    "ExampleProcessorConfig": (
        ScenarioSpec(
            title="典型用法",
            description="展示最小可运行组合。",
            file="example_processor_scenario.yaml",
        ),
    )
}
```

对应 YAML 必须是根节点为对象的合法配置文件。生成器按类名挂载场景并渲染为“配置场景示例”；标题和描述登记在 `ScenarioSpec`，完整配置正文只维护在 YAML。场景 YAML 不替代运行期校验，也不得用来承载字段语义、约束或注意事项。

### 6.5 内部类型与公开分派

公开处理器、保存器和自动调优配置必须能被用户通过 `type` / `mode` 选择，并进入目录和交叉链接。仅用于内部嵌套实现的类型使用以下划线开头的分派值；内部 validator 使用 `@config_internal`。内部实现不得出现在公开派生类表格或可复制的完整示例中。

### 6.6 生成稿要求

生成稿顶部必须保留不可手改提示和 `generated-by` 标记。不得直接修改生成页面；需要修正内容时，应修改源码注解、示例 YAML 或 catalog，然后重新生成并提交。

## 7. YAML 完整示例与场景

完整示例和场景统一放在 `templates/`，由 `examples.py` 显式登记。README 只负责说明和索引，不作为机器可读示例来源。

| 文件 | 对应配置类 |
| --- | --- |
| `templates/practice_config.yaml` | `PracticeConfig` |
| `templates/linear_processor.yaml` | `LinearProcessorConfig` |
| `templates/adapt_rotation_processor.yaml` | `AdaptRotationProcessorConfig` |
| `templates/modelslim_convert_quant.yaml` | `ModelslimConvertQuantConfig` |
| `templates/rename_pattern.yaml` | `RenamePattern` |
| `templates/convert_op.yaml` | `ConvertOpConfig` |
| `templates/group_processor.yaml` | `GroupProcessorConfig` |
| `templates/tlq_op.yaml` | `TLQOpConfig` |
| `templates/quant_save_processor.yaml` | `QuantSaveProcessorConfig` |
| `templates/strategy_standing_high.yaml` | `StandingHighStrategyConfig` |
| `templates/strategy_standing_high_with_experience.yaml` | `StandingHighWithExperienceStrategyConfig` |
| `templates/strategy_binary_fallback.yaml` | `BinaryFallbackStrategyConfig` |
| `templates/evaluation_service_oriented.yaml` | `ServiceOrientedEvaluateServiceConfig` |
| `templates/unary_analysis_ra_compress.yaml` | `UnaryAnalysisProcessorConfig`（场景示例） |

YAML 约定：

- 扩展名统一为 `.yaml`，根节点必须是对象；
- 示例应接近用户可复制执行的形态，避免注释性伪代码和内部占位对象；
- 一个文件只服务一个页面或一个明确场景；
- 示例必须能通过对应根配置或调优根配置校验；
- 需要内部类型时，YAML 可暂用下划线开头的内部 `type`，由生成器统一重写为公开类型；
- 不在 YAML 中重复生成器会补齐的 `apiversion`、`spec` 或保存器，除非该文件明确登记为完整根示例。

新增精选示例时：

1. 在 `templates/` 新增 YAML；
2. 在 `examples.py` 的 `EXPLICIT_EXAMPLE_FILES`、`TUNING_EXAMPLE_FILES` 或 `SCENARIO_EXAMPLE_FILES` 中登记；
3. 确认示例通过对应根配置或调优根配置校验；
4. 重新运行生成器并检查“完整配置参考”diff；
5. 运行 `--check` 验证注册完整性和生成稿一致性。

## 8. 模块职责

| 文件 | 职责 |
| --- | --- |
| `main.py` | 兼容直接执行入口；也可用于 Python 调用，并可从任意工作目录定位仓库根目录。 |
| `__main__.py` | 推荐模块入口；支持 `python -m tools.config_docgen`。 |
| `generate.py` | CLI 参数、注解审计、退出码、一致性检查提示、Windows 进程内路径兼容；保留兼容入口。 |
| `discovery.py` | 配置类发现、分类、YAML 路径、引用图、示例装配、写盘与 prune。 |
| `examples.py` / `templates/` | 完整示例和场景 YAML、注册表及严格加载器。 |
| `records.py` | `FieldRecord`、`ScenarioRecord`、`ModelRecord` 数据契约。 |
| `schema.py` | Pydantic JSON Schema、字段、类型、默认值、约束和 validator 文档事实抽取。 |
| `catalog.py` | slug、目录、分类、事实源模块、目标关键字等集中注册表。 |
| `renderer.py` | Markdown 页面组织、章节、编号、链接和生成标记。 |

## 9. 维护流程与门禁

修改量化配置类时按以下顺序操作：

1. 修改字段、默认值、枚举或运行期校验；
2. 同步修改 `Field(description=...)` 和配置类 docstring 的首段、`说明`、`约束`、`注意事项`；
3. 同步修改 validator docstring 与参数化的 `config_constraint`；稳定场景同步更新 `templates/` YAML 和 `examples.py` 注册；
4. 判断完整示例使用自动装配还是更新 `templates/*.yaml`，并同步 `examples.py` 注册；
5. 运行 `python -m tools.config_docgen`；
6. 检查生成稿 diff，确认字段含义、默认值、枚举、边界、约束和示例均符合预期；
7. 运行 `python -m tools.config_docgen --check`；
8. 提交源码、示例 YAML 和生成稿。

不要只修改生成稿而不修改源码注解。`python -m tools.config_docgen --check` 用于本地检查，并由 UT 流水线 `test/run_ut.sh` 自动执行；全量检查应在 30 秒内完成，不成为流水线瓶颈。

## 10. 故障排查

| 输出 | 处理 |
| --- | --- |
| `drift:` | 重新生成并提交生成稿。 |
| `stale:` | 运行全量生成清理旧页，并修正导航残留链接。 |
| 缺少字段 description | 为字段补 `Field(description=...)`。 |
| 缺少配置类概述 docstring | 补配置类自身 docstring 首段。 |
| validator 缺少约束 docstring | 补与实现一致的约束说明。 |
| AfterValidator 缺少约束说明 | 给最终 validator 函数添加 `@config_constraint(...)`，参数化约束使用格式化参数。 |
| 场景 YAML 未渲染 | 检查 `SCENARIO_EXAMPLE_FILES` 的类名、标题和文件登记，并运行 `--check`。 |
| 配置示例不是合法 YAML / 根节点必须是对象 | 修正 `templates/` 中对应文件并重新生成。 |
| 示例校验失败 | 检查 YAML 对应的配置类、`kind`、挂载路径和必填字段。 |
| unknown target | 检查 `--targets` 拼写；门禁不允许零目标通过。 |
| 缺少依赖或模块导入失败 | 在仓库开发环境中安装项目运行依赖，确认 `msmodelslim/config` 已按开发安装流程初始化。 |
