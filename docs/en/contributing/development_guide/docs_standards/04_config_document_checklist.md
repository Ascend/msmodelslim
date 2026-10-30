# Configuration Document Checklist

> Used to check configuration documents that are created or reworked based on the *[Configuration Document Template](04_config_document_template.md)*, applicable to task configurations (including `PracticeConfig`), service specifications, and the configurations that reference them. For the general check items, see the *[Common Checklist](00_common_checklist.md)*; for the check mechanism (severity levels, conclusion rules, check execution, and the waiver mechanism), see Section 4.2 of the *[msModelSlim Documentation Standards](./README.md)*. The following are the check items specific to the configuration document type, separated into tables by severity level.

## 1. Check Items

### 1.1 ERROR (Mandatory)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| FE-01 | Configuration overview metadata | The metadata table of "Configuration Overview" contains the configuration class and a source link, both of which can be located in the source code of the target version. | Add the missing fields and fix the broken links |
| FE-02 | Parameter list completeness and references | The "Parameter List" is organized in blocks by configuration class name and covers all fields defined directly by the configuration class and those exposed to users through inheritance; the field path is the field name relative to the current configuration; nested configurations (including the `spec` of a task configuration) have been expanded into sub-blocks for the corresponding class name, or linked to a standalone document through the "Referenced Config" column; the "Referenced Config" column covers all nested configurations used by this configuration and the `type` and `mode` dispatch targets, with links pointing to anchors on this page or to standalone configuration documents; the same base class is rendered only once on the same page. | Add the missing blocks or fields, delete the fields that do not exist, and fix the field paths and reference links |
| FE-03 | Consistent field attributes | The type, default value, and required or optional status of each field are consistent with the source code definition and runtime validation. | Fix the inconsistent field attributes |
| FE-04 | Complete configuration reference | Check only when the "Complete Configuration Reference" section exists: the YAML syntax is correct, the configuration is placed under the actual field path, and it can be successfully parsed by the configuration loading entry point. | Fix the YAML syntax or the path level |

### 1.2 WARN (Recommended)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| FW-01 | Configuration overview content | "Configuration Overview" describes the quantization stage controlled by this configuration, the applicable models or scenarios, and its position in the complete configuration. | Add or fix the configuration overview |
| FW-02 | Complete parameter descriptions | The "Value Range or Format" of each field includes enumeration values, numeric boundaries, and path formats; when enumeration is not possible, a clear constraint description is given. | Add the value range or format |
| FW-03 | Division of labor between constraints and notes | "Configuration Constraints" describes only the rules that the parser, validator, or loading logic can determine, including mutual exclusion, dependencies, priorities, and fallback, and is filled with "- None" when there are no additional constraints; "Notes" describes only runtime practice issues, including performance, accuracy, and video memory, and does not repeat the rules already given in "Configuration Constraints". | Move the practice constraints that cannot be checked to "Notes", and move the checkable rules to "Configuration Constraints" of the corresponding configuration block |
| FW-04 | Complete configuration reference | It is recommended to write a "Complete Configuration Reference" section that provides a complete YAML example placed under the actual field path and loadable directly. | Add the "Complete Configuration Reference" section |
| FW-05 | Example value constraints | Check only when the "Complete Configuration Reference" section exists: the example values satisfy the parameter constraints and are consistent with the parameter list; complex nested configurations indicate their type through `type` and do not use an ellipsis as a substitute. | Fix the inconsistent example values and expand the omitted parts |
| FW-06 | Compatibility and migration | Check only when the "Compatibility and Migration" section exists: deprecated fields have been provided with a replacement and a migration method. | Add the replacement and the migration method |
| FW-07 | Anchor style | In-page jumps use HTML anchors `<a href="#anchor">§x.y</a>` and do not use the Markdown `{#anchor}` syntax. | Unify them to the HTML anchor style |

## 2. Usage Notes

- **Validation scope**: configuration documents written based on the *[Configuration Document Template](04_config_document_template.md)*, stored in `docs/zh/api_reference/config/`; documents produced by the generator carry a `generated-by` marker and are not edited manually.
