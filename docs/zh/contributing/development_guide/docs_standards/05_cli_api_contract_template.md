# {{ command_name }} 命令行 API 文档

> **注释：** 将 command_name 替换为命令名称。所有 {{ placeholder_name }} 占位符必须替换为实际内容；占位符名称只能使用英文小写字母、数字和下划线，多个单词以下划线连接。发布文档前删除所有填写说明，并从保留的可选章节标题中删除 [OPTIONAL] 前缀。

## 功能说明

> **注释：** 说明该命令解决的问题、主要能力和适用场景，不在此处展开参数细节。

{{ command_summary }}

## 命令格式

> **注释：** 给出完整的命令语法摘要。必选参数直接列出，可选参数使用方括号标识；尖括号表示需要替换的值。不接收值的布尔开关不得添加值占位符；可重复参数、一次接收多个值的参数及互斥参数组应使用与 --help 一致的表示方式，并在摘要后解释。此处用于说明语法，不作为可复制执行的命令；可执行命令写入“使用示例”。

```text
{{ executable_name }} {{ subcommand_name }} --{{ required_argument_name }} <{{ required_argument_value_name }}> [--{{ optional_argument_name }} <{{ optional_argument_value_name }}>]
```

> **注释：** 说明命令格式中位置参数、选项参数、互斥参数组、可重复参数等符号的含义。没有特殊语法时填写“无”。

{{ command_syntax_description }}

## 参数列表

> **注释：** 参数名称和别名必须与 --help 和实现代码一致；没有别名时填写“无”。类型使用 string、int、float、bool、list 等明确名称。传入形式应区分不带值的布尔开关（如 --debug）、需要显式值的布尔参数（如 --trust_remote_code true）、单值参数以及可重复或一次接收多个值的参数，并说明列表分隔方式。必选参数的默认值填写“无”；取值范围或格式应包含枚举值、数值边界、路径要求、大小写规则及参数间约束。每个参数单独成行，不得合并描述。

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--{{ required_argument_name }}` | {{ required_argument_alias }} | `{{ required_argument_type }}` | {{ required_argument_input_form }} | 必选 | 无 | {{ required_argument_constraint }} | {{ required_argument_description }} |
| `--{{ optional_argument_name }}` | {{ optional_argument_alias }} | `{{ optional_argument_type }}` | {{ optional_argument_input_form }} | 可选 | `{{ optional_argument_default }}` | {{ optional_argument_constraint }} | {{ optional_argument_description }} |

> **注释：** 存在位置参数时保留下表并逐项填写；不存在时删除本段说明和下表。

| 位置参数 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|----------|------|----------|-----------|--------|----------------|------|
| `{{ positional_argument_name }}` | `{{ positional_argument_type }}` | {{ positional_argument_input_form }} | {{ positional_argument_requirement }} | {{ positional_argument_default }} | {{ positional_argument_constraint }} | {{ positional_argument_description }} |

## 参数关系

> **注释：** 逐条说明互斥、依赖、至少选一项、同时使用时的优先级以及重复传入时的处理规则。没有参数关系时填写“无”。

- {{ argument_relationship }}

## [OPTIONAL] 引用的配置

> **注释：** 仅当命令通过 --config 等参数加载或依赖 YAML 配置、配置类或量化方案时保留本节。链接应指向对应配置文档的字段说明或独立配置文档；不得只链接源码。没有引用配置时删除本节。

| 关联参数 | 配置名称 | 引用关系 | 配置文档 |
|----------|----------|----------|----------|
| `--{{ referenced_param_name }}` | `{{ referenced_config_name }}` | {{ reference_relationship }} | [{{ referenced_document_name }}]({{ referenced_document_link }}) |

## [OPTIONAL] 环境变量

> **注释：** 仅当存在命令专属环境变量，或环境变量会显著改变本命令行为时保留本节。全局日志级别、全局缓存目录等通用环境变量不在单个 CLI 契约中重复说明。默认值必须与代码一致；敏感变量不得填写真实凭据。

| 环境变量 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|----------|------|-----------|--------|----------------|------|
| `{{ environment_variable_name }}` | `{{ environment_variable_type }}` | {{ environment_variable_requirement }} | `{{ environment_variable_default }}` | {{ environment_variable_constraint }} | {{ environment_variable_description }} |

## [OPTIONAL] 使用示例

> **注释：** 仅当需要展示典型场景或参数组合时保留本节。第一条示例应为最小可跑场景，只包含完成核心功能所需的最少参数；后续示例用于展示常见参数组合或高级场景。示例必须可复制，路径、模型名等变量使用大写 shell 变量，并在代码块后解释变量和预期结果。不得包含真实密钥或内部地址。

### {{ example_scenario_name }}

{{ example_scenario_description }}

```bash
{{ executable_name }} {{ subcommand_name }} \
  --{{ example_argument_name }} {{ example_argument_value }}
```

> **注释：** 解释示例中的变量、预期行为和关键输出。

{{ example_result_description }}

## [OPTIONAL] 退出码与异常处理

> **注释：** 仅当命令定义了可供脚本判断的稳定退出码，或存在需要用户处理的典型异常时保留本节。多数命令只需说明“0 表示成功，非 0 表示失败，具体原因见错误日志”；只有稳定退出码或典型异常需要细分处理时才保留下表。

| 退出码或异常 | 含义 | 处理建议 |
|--------------|------|----------|
| `{{ exit_code }}` | {{ exit_code_description }} | {{ error_resolution }} |

## [OPTIONAL] 安全说明

> **注释：** 仅当命令涉及远程代码、凭据、文件覆盖、外部输入、网络访问或其他安全风险时保留本节。说明风险、触发条件和规避措施。

- {{ security_notice }}
