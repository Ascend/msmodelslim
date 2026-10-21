# {{ command_name }} Command-Line API Document

> **Note**: replace command_name with the command name. All `{{ placeholder_name }}` placeholders must be replaced with actual content; placeholder names may contain only lowercase English letters, digits, and underscores, with multiple words connected by underscores. Before publishing the document, delete all filling instructions and remove the [OPTIONAL] prefix from the headings of the optional sections that are kept.

## 1. Function Description

> **Note**: describe the problem this command solves, its main capabilities, and the applicable scenarios; do not expand the parameter details here.

{{ command_summary }}

## 2. Command Format

> **Note**: give a complete command syntax summary. List required parameters directly and mark optional parameters with square brackets; angle brackets denote values that need to be replaced. A Boolean switch that does not accept a value must not be given a value placeholder; repeatable parameters, parameters that accept multiple values at once, and mutually exclusive parameter groups should use the same representation as `--help` and be explained after the summary. This is used to describe the syntax and is not a command that can be copied and executed; executable commands are written into "Usage Examples".

```text
{{ executable_name }} {{ subcommand_name }} --{{ required_argument_name }} <{{ required_argument_value_name }}> [--{{ optional_argument_name }} <{{ optional_argument_value_name }}>]
```

> **Note**: explain the meaning of the symbols for positional parameters, option parameters, mutually exclusive parameter groups, and repeatable parameters in the command format. When there is no special syntax, write "None".

{{ command_syntax_description }}

## 3. Parameter List

> **Note**: parameter names and aliases must be consistent with `--help` and the implementation code; when there is no alias, write "None". Types use explicit names such as string, int, float, bool, and list. The passing form should distinguish a Boolean switch without a value (such as --debug), a Boolean parameter that requires an explicit value (such as --trust_remote_code true), a single-value parameter, and a repeatable parameter or one that accepts multiple values at once, and should explain how the list is separated. The default value of a required parameter is written as "None"; the value range or format should include enumeration values, numeric boundaries, path requirements, case rules, and constraints between parameters. Each parameter occupies its own line and must not be merged into one description.

| Parameter | Alias | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|------|------|------|----------|-----------|--------|----------------|------|
| `--{{ required_argument_name }}` | {{ required_argument_alias }} | `{{ required_argument_type }}` | {{ required_argument_input_form }} | Required | None | {{ required_argument_constraint }} | {{ required_argument_description }} |
| `--{{ optional_argument_name }}` | {{ optional_argument_alias }} | `{{ optional_argument_type }}` | {{ optional_argument_input_form }} | Optional | `{{ optional_argument_default }}` | {{ optional_argument_constraint }} | {{ optional_argument_description }} |

> **Note**: when positional parameters exist, keep the table below and fill it in item by item; when they do not exist, delete this note and the table below.

| Positional Parameter | Type | Passing Form | Required/Optional | Default | Value Range or Format | Description |
|----------|------|----------|-----------|--------|----------------|------|
| `{{ positional_argument_name }}` | `{{ positional_argument_type }}` | {{ positional_argument_input_form }} | {{ positional_argument_requirement }} | {{ positional_argument_default }} | {{ positional_argument_constraint }} | {{ positional_argument_description }} |

## 4. Parameter Relationships

> **Note**: explain item by item the rules for mutual exclusion, dependencies, selecting at least one option, the priority when several are used together, and the handling of repeated passing. When there are no parameter relationships, write "None".

- {{ argument_relationship }}

## 5. [OPTIONAL] Referenced Configurations

> **Note**: keep this section only when the command loads or depends on YAML configurations, configuration classes, or quantization schemes through parameters such as --config. Links should point to the field descriptions of the corresponding configuration document or to a standalone configuration document; do not link only to the source code. When there are no referenced configurations, delete this section.

| Related Parameter | Configuration Name | Reference Relationship | Configuration Document |
|----------|----------|----------|----------|
| `--{{ referenced_param_name }}` | `{{ referenced_config_name }}` | {{ reference_relationship }} | [{{ referenced_document_name }}]({{ referenced_document_link }}) |

## 6. [OPTIONAL] Environment Variables

> **Note**: keep this section only when command-specific environment variables exist or when environment variables significantly change the behavior of this command. General environment variables such as the global log level and the global cache directory are not described again in a single CLI contract. Default values must be consistent with the code; sensitive variables must not be filled with real credentials.

| Environment Variable | Type | Required/Optional | Default | Value Range or Format | Description |
|----------|------|-----------|--------|----------------|------|
| `{{ environment_variable_name }}` | `{{ environment_variable_type }}` | {{ environment_variable_requirement }} | `{{ environment_variable_default }}` | {{ environment_variable_constraint }} | {{ environment_variable_description }} |

## 7. [OPTIONAL] Usage Examples

> **Note**: keep this section only when typical scenarios or parameter combinations need to be shown. The first example should be a minimal runnable scenario containing only the minimum parameters required to complete the core function; subsequent examples are used to show common parameter combinations or advanced scenarios. The examples must be copyable, variables such as paths and model names should use uppercase shell variables, and the variables and expected results should be explained after the code block. Real keys or internal addresses must not be included.

### 7.1 {{ example_scenario_name }}

{{ example_scenario_description }}

```bash
{{ executable_name }} {{ subcommand_name }} \
  --{{ example_argument_name }} {{ example_argument_value }}
```

> **Note**: explain the variables in the example, the expected behavior, and the key output.

{{ example_result_description }}

## 8. [OPTIONAL] Exit Codes and Exception Handling

> **Note**: keep this section only when the command defines stable exit codes that scripts can check, or when there are typical exceptions that users need to handle. For most commands it is enough to state "0 means success, non-0 means failure, and the specific reason is in the error log"; keep the table below only when stable exit codes or typical exceptions require differentiated handling.

| Exit Code or Exception | Description | Recommended Action |
|--------------|------|----------|
| `{{ exit_code }}` | {{ exit_code_description }} | {{ error_resolution }} |

## 9. [OPTIONAL] Security Notes

> **Note**: keep this section only when the command involves remote code, credentials, file overwriting, external input, network access, or other security risks. Describe the risks, trigger conditions, and mitigations.

- {{ security_notice }}
