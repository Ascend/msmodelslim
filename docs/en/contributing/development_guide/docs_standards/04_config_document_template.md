# {{ config_name }} Configuration Reference

> **Note**: replace config_name with the configuration name. All `{{ placeholder_name }}` placeholders must be replaced with actual content; placeholder names may contain only lowercase English letters, digits, and underscores, with multiple words connected by underscores. Before publishing the document, delete all filling instructions and remove the [OPTIONAL] prefix from the headings of the optional sections that are kept.

## 1. Configuration Overview

> **Note**: describe the processing stage controlled by this configuration, the applicable models or scenarios, and its position in the complete configuration.

{{ config_summary }}

> **Note**: fill in the source link that defines this configuration and the corresponding class name. The source link should point to the stable repository file address for the target release version, tag, or commit number; when there are multiple entry points, list them item by item.

| Item | Content |
|------|------|
| Config Class | `{{ config_class_name }}` |
| Source | [{{ source_file_name }}]({{ source_file_link }}) |

## 2. Parameter List

> **Note**: list all fields defined directly by this configuration and those opened up to users through inheritance. Field paths use the **field name relative to the current configuration** and do not prepend the whole YAML path; the position of the configuration in YAML is given by "Configuration Overview" or by the block heading. Types use explicit names such as string, int, float, bool, object, and list. The default value of a required field is written as "None"; optional fields must give their actual default value. The value range or format should include enumeration values, numeric boundaries, path formats, and combination constraints. When an object or list is defined by another configuration, link to the corresponding document or in-page anchor in the "Referenced Config" column.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `{{ required_field_path }}` | `{{ required_field_type }}` | Required | None | {{ required_field_constraint }} | {{ required_field_description }} | {{ required_field_reference }} |
| `{{ optional_field_path }}` | `{{ optional_field_type }}` | Optional | `{{ optional_field_default }}` | {{ optional_field_constraint }} | {{ optional_field_description }} | {{ optional_field_reference }} |

> **Note**: the parameter list is organized in blocks by configuration class name, and there are no longer separate "Configuration Constraints", "Referenced Configurations", or "Referenced By" sections:
>
> - Root configuration block: `<h3 id="2-1-...">2.1 Class name</h3>`, followed by the parameter table and the "Configuration Constraints" sub-items.
> - Nested configurations (including the spec of a task configuration) are expanded into the corresponding class name sub-block on this page, using `<h3>`/`<h4>` headings (headings carry visible numbers, such as `2.2 Class name`); if the first paragraph of the class docstring is not empty, output a one-sentence class overview after the heading and before the parameter table.
> - `type`/`mode` dispatch fields are rendered as a base class block `<h3 id="...">2.x Base class name (dispatched by type)</h3>`, which contains the base class parameter table (when the base class itself is a real configuration) and a "derived classes" list (class name, discriminator value, one-sentence description, in-page anchor, or standalone page link); the parameter table and "Configuration Constraints" of each derived class follow as `<h4>` sub-blocks.
> - The same base class is rendered only once on the same page, and the "Referenced Config" column of subsequent dispatch fields points to the anchor of that base class block.
> - In-page jumps use HTML anchors `<a href="#anchor">§x.y</a>` and do not use the Markdown `{#anchor}` attribute syntax.
> - The "Configuration Constraints" of each configuration block lists only the rules that the YAML parser, configuration model, validator, or loading logic can determine (mutual exclusion, dependencies, priorities, inheritance, overrides, fallback, and so on); when there are no additional constraints, write "- None.". Runtime practice issues such as performance, accuracy, resources, and data representativeness are written into "Notes".

## 3. [OPTIONAL] Complete Configuration Reference

> **Note**: provide a complete typical YAML that is syntactically correct, can be copied directly, and requires only the business paths to be replaced before use. The example must place the current configuration in the actual complete field path: a top-level task configuration should contain apiversion and spec; nested or referenced configurations should be placed in the real reference positions such as spec.process[] and spec.save[], and the fields of a sub-configuration must not be mistakenly placed directly under spec. The example must contain the required fields of this configuration and the key typical optional fields; for complex nested configurations, expand only the configuration level that this document is responsible for and one necessary level of sub-configuration, indicate the sub-configuration type through type or a discriminator field, and link to the "sub-configuration index" or the corresponding configuration document. An ellipsis must not be used in place of YAML. The example values must satisfy the parameter constraints and stay consistent with the parameter list and the reference relationships. When a complete example that can be successfully parsed by the configuration loading entry point cannot be provided, this section may be omitted.

```yaml
{{ complete_config_yaml }}
```

> **Note**: explain the scenario adopted by the complete example, the values that users need to replace, and the execution entry point.

{{ complete_config_description }}

## 4. [OPTIONAL] Compatibility and Migration

> **Note**: keep this section only when the configuration has version differences, deprecated fields, old field mappings, or compatibility behavior during migration. Describe the affected versions, replacement fields, priorities, and removal plan.

| Old Field or Behavior | New Field or Behavior | Affected Version | Migration Method |
|--------------|--------------|------------|----------|
| `{{ deprecated_field_name }}` | `{{ replacement_field_name }}` | {{ affected_version }} | {{ migration_instruction }} |

## 5. [OPTIONAL] Notes

> **Note**: keep this section only when there are practice issues that cannot be determined directly by the YAML parser, configuration model, validator, or loading logic, but that affect performance, accuracy, video memory, devices, data representativeness, security, or runtime stability. Rules such as field validity, mutual exclusion and dependencies, and default/override/fallback are written into the "Configuration Constraints" of each configuration block.

- {{ config_notice }}
