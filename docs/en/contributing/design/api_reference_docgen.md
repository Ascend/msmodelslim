# Design of Automatic Interface Documentation Generation

This document explains how `skills/docs-management/scripts/` extracts **quantization configuration** interface documentation (Template 04) from the source code. For usage, see the [Interface Documentation Generator](../../../../skills/docs-management/scripts/README.md); for the template and the checklist, see the [Configuration Document Template](../development_guide/docs_standards/04_config_document_template.md).

The command-line API (Template 05) does not go through the generator: write it against the CLI implementation and `--help`; see the [Command-Line API Document Template](../development_guide/docs_standards/05_cli_api_contract_template.md) and `docs/en/api_reference/cli/`.

## 1. Background and Objectives

Configuration interface documentation must be consistent with the implementation: field names, types, default values, mutual exclusion relationships, and `type` dispatch all come from the code. Manually maintained documentation goes stale as soon as a processor is added or a validator is changed.

The generator must satisfy all of the following at the same time:

- **One document per struct**, with a structure aligned with Template 04, for ease of retrieval.
- **Machine-verifiable**: `--check` compares the Markdown in the repository with the current source code, so drift can be found in CI or locally.
- **Readable**: the meaning and the constraints come from the source annotations; the generator does not fabricate field descriptions.
- **Unit-testable**: rendering and annotation parsing do not require importing the complete `msmodelslim`.

Usage steps are not generated. Operation guides remain under `docs/en/user_guide/`; the interface documentation only answers "what can be filled in, what the default is, and what it is mutually exclusive with".

## 2. Extraction Rules

The generator only carries over contracts that already exist in the source code and does not fabricate interface semantics. When content is missing, change the annotation instead of the lookup table in the script.

### 2.1 Internal Class Format

A configuration class is treated as an **internal class** if it declares a single YAML `type` literal and that literal starts with `_`. This is the only machine-decidable rule; examples include `_auto_save`, `_adapt_rotation_stage1`, and `_abstract_flex_smooth_base_`.

| Criterion | Generator Behavior |
|------|------------|
| Internal `type` and the class is a public dispatch base class (`AutoProcessorConfig` / `QuantFormatConfig`) | Not documented separately and not included in the `process[]` / `save[]` index |
| Internal `type` and the class is the nested-field carrier of a public configuration (such as stage1/stage2) | May be documented as a nested configuration; examples must not write this `type` as a dispatch name that users can fill in |
| Public `type` (does not start with `_`) | Documented independently and included in the corresponding dispatch index |

Internal `type` values must not appear in examples that users can copy: `_auto_save` is replaced by the first public save format, and other internal `type` values are removed from the example (they are written by the parent configuration).

When adding an internal implementation: give the `type` a Literal that starts with `_`, instead of adding another skip list to the generator.

### 2.2 Where the Content Comes From

| Document Position | Source | When Missing |
|----------|----|--------|
| Parameter "Meaning" | `Field(description=)` | Write `—` and add the description instead; do not hard-code an overlay in the generator |
| Configuration overview | The first paragraph of the class docstring | Fall back to "this is an XX configuration, located at the XX path" and add the class docstring instead |
| Configuration constraints (the sub-items of each configuration block in the parameter list) | The docstring of the field / model validator | "None."; add the validator docstring instead |
| Numbers and enumerations in the value range | JSON Schema: `enum` / `const` / `minimum` / `maxLength` and so on, corresponding to `Field(ge=..., max_length=...)`, `Literal`, and `Enum` | Constraints that the schema does not contain (such as a pure `AfterValidator`) do not appear and should be changed into Field constraints |
| Type, default value, mandatory | The `type` / `default` / `required` of the JSON Schema; fall back to Field when `default_factory` does not enter the schema | Consistent with the implementation |

## 3. Overall Architecture

The generator is split into two layers:

| Layer | File | Responsibility |
|------|------|------|
| Pure extraction / rendering | `quant_config_docgen.py` | JSON Schema → FieldRecord → Markdown. Does not import `msmodelslim`. |
| Product assembly | `gen_quant_config_docs.py` | Discovers the real classes, computes the YAML paths, and writes the files; modifies `mkdocs.yml` only when `--update-mkdocs` is passed. |

```text
Pydantic Config Class
        │
        ▼
  model_json_schema (Generator for documentation)
        │
        ▼
  quant_config_docgen.py (JSON Schema → FieldRecord → Markdown)
        ▲
        │ Assembly paths / reference graph / example shell
        │
  gen_quant_config_docs.py
        │
        ▼
  docs/en/api_reference/config/<category>/*.md
```

> The specific extraction targets, expansion strategy, and post-processing are controlled by the driver script `gen_config_api_docs.py`; `gen_quant_config_docs.py` provides the basic capabilities such as `--targets` / `--expand-nested` / `--prune`.

The first line of a generated file carries an HTML marker (`generated-by: skills/docs-management/scripts/gen_quant_config_docs.py`). A file with this marker is regarded as generated output; change the source code and regenerate it rather than editing the Markdown directly.

CLI documentation is not covered by this generator and does not use the `generated-by` marker.

Manually maintained files preserved: `README.md` (configuration reference navigation), `auto_precision_tuning.md`, and `processor_group.md` (moved to `user_guide/usage_processor_group.md`). The generator skips these file names before writing and cleaning up.

## 4. Quantization Configuration Generator

### 4.1 What to Do

Render each externally visible Pydantic configuration class as one Template 04 document, and organize the output by type subdirectory. The externally visible classes include:

- Task root configurations: the four `apiversion` entry points such as `ModelslimV1QuantConfig`, plus the task base-class practice configuration `PracticeConfig` (a `BaseQuantConfig` subclass, containing `metadata` → `Metadata`).
- Service specifications: `*ServiceConfig`.
- Processors: the subclasses in the `AutoProcessorConfig` registry whose `type` does not start with `_`.
- Save formats: the public subclasses of `QuantFormatConfig`.
- Auto Tuning: `TuningPlanConfig`, the concrete strategies, and the evaluation services.
- Nested BaseModels reached recursively from the field annotations of the preceding classes (such as `QConfig` and `Metadata`).
- Additional seeds on the weight conversion side: `RenamePreprocessConfig` and others.

Output directory: `docs/en/api_reference/config/`, with files placed into subdirectories by category:

| Category | Subdirectory |
|------|--------|
| Task Configuration | `quant/` |
| Service specifications | Not a separate page by default: expanded into the "Parameter List" of the corresponding task (quant) page; placed in `spec/` when the base script generates it as a standalone document |
| Processor | `processor/` |
| Save Format | `format/` |
| Auto Tuning | `tuning/` |
| Nested configurations (only when the base script generates them as standalone documents) | `nested/` |

The **content** of `config/README.md` is not generated (that page is a manually written navigation page protected by `MANUAL_KEEP`): protocol selection and the `type` index and grouping are maintained manually, and the generator is responsible only for the individual configuration pages; the entry document (usage guide) links directly to the specific page under each category.

### 4.2 Why the Split Is Done This Way

YAML is a tree of "root configuration → spec → process[] / save[] → nested objects", but users search by `type` (`fa3_quant`, `linear_quant`) or by class name. One giant YAML description cannot be consulted per processor, and it cannot provide backlinks for "the configuration being referenced".

Therefore **one document per model + a (forward) reference graph** is adopted: each document only answers "which configurations do the fields of this class point to", and `type` dispatch fields (`process`, `save`, `strategy`, `evaluation`, `select_best`, `operations`, `preprocess`, and so on) are expressed as "base class block + derived class list + derived class sub-blocks": the base class block states that the field is dispatched by `type`/`mode` and lists all public subtypes, rather than writing only the base class name. Reference information is consolidated into the "Referenced Config" column of the parameter table, and a separate "referenced configurations / configurations that reference it" section is no longer provided.

Internal configurations are determined as described in [Section 2.1](#21-internal-class-format) and internal `type` values are not written into the public dispatch index. By default, the driver script expands nested internal configurations into the corresponding class-name sub-block of the "Parameter List" in the parent document to reduce navigation; for the task root configuration, the corresponding spec (service specification) is also expanded into the "Parameter List", so that the task and the spec are merged into one page and a separate `spec/` page is no longer generated. `PracticeConfig` is a `BaseQuantConfig` subclass (a task configuration); although it is referenced by tuning strategies, it is still documented independently as `quant/practice_config.md`, presenting on the task side the complete practice configuration structure of `apiversion + spec + metadata` (`Metadata`).

### 4.3 How It Is Implemented

`build_records()` works in a fixed order.

#### 4.3.1 Discovering Classes

1. Import the known modules so that the processors / save formats complete their registration.
2. Seeds = the five root configurations (the four `apiversion` entry points + `PracticeConfig`) ∪ public processors ∪ public save formats ∪ extra models on the conversion side ∪ Auto Tuning targets (`TuningPlanConfig` + public strategies + evaluation services).
3. For each seed, traverse `model_fields` and use `collect_nested_models()` to take the nested BaseModels out of `Optional` / `List` / `Union` / `Annotated`, adding them recursively.
4. `_should_keep()` filters out non-Pydantic classes, the skip list, test modules, internal `type` values, and the `AutoSaverBaseConfig` intermediate layer.

Public processors come from `AutoProcessorConfig._registry` rather than `__subclasses__()`, so that registered implementations that do not form an inheritance tree are not missed. Save formats come from `QuantFormatConfig.__subclasses__()`.

#### 4.3.2 Classification and File Names

| Criterion | Directory Group |
|------|----------|
| `BaseQuantConfig` subclass | Task Configuration |
| Class name ends with `ServiceConfig` (not an Auto Tuning evaluation service) | Service Specification |
| `QuantFormatConfig` subclass | Save Format |
| `AutoProcessorConfig` subclass | Processor |
| Auto Tuning targets (`TuningPlanConfig`, concrete strategies, evaluation services) | Auto Tuning |
| Others | Nested Configuration |

The file name (slug) prefers the unique literal of the field `type: Literal["xxx"]`; otherwise it is the snake_case class name. The five task configurations (the four `apiversion` entry points + `PracticeConfig`) and some duplicate `QuantStrategyConfig` names use an override table to fix the file names and avoid link churn.

#### 4.3.3 YAML Paths

BFS from the root configuration: parent path + field name, with `[]` appended to list fields. When a field annotation points to `AutoProcessorConfig`, all public processors share `spec.process[]`; when it points to a saver, they share `spec.save[]`.

If an individual nested class is reachable through multiple paths, an override table pins down the path most commonly used by users, for example `QConfig` → `spec.process[].qconfig.weight`. Classes that are not reached from the root fall back to `spec.process[]` / `spec.save[]` / `spec`.

The "Field Path" in the field table = the field name itself (relative to the current configuration) and does not include the whole YAML path. Subheadings no longer show the `(ctx_path)` path suffix; the ownership relationship of configurations is expressed by the block hierarchy of the "Parameter List" (ordinary nesting is flattened, while type dispatch uses a base class block + derived class sub-blocks).

#### 4.3.4 Field Extraction

`extract_fields()` takes `model_json_schema(mode="serialization")` and maps it into a parameter table. The serialization schema has the same shape as YAML: it uses aliases (such as `from_` → `from`) and excludes fields with `exclude=True`. `computed_field` entries are dropped even if they enter the schema, because users cannot fill them in the YAML.

| Table Column | Source |
|--------|------|
| Type | Schema `type` / `anyOf` / `$ref`, mapped to `string` / `int` / `object` / `list[...]` / `... / null` |
| Required | Schema `required` |
| Default | Schema `default`; falls back to Field / `default_factory` when absent (factory values often do not enter the schema) |
| Value Range or Format | Schema `enum` / `const` / `minimum` / `maxLength` / `pattern` and so on |
| Description | Schema `description` (that is, `Field(description=)`); `—` when absent |
| Referenced Config | The nested model corresponding to Schema `$ref` (when expanded into this page, an in-page HTML anchor `<a href="#anchor">` is written; for a standalone page, a relative link is written); type dispatch fields uniformly point to the base class block anchor `"This page <a href="#anchor">§x.y</a>"` |

`AfterValidator` **does not** enter the JSON Schema. Lengths and ranges must be written as `Field(max_length=N)` / `ge` / `le`, or as `Literal` / `Enum`, to appear in the documentation. The generator no longer parses validation closures and does not maintain function names to the lookup table.

Types that cannot be converted to JSON (such as `torch.dtype`) fall back to `string` in the documentation Generator, so that the whole schema does not fail; if the field is a `computed_field`, it is still dropped afterwards.

Configuration constraints (the sub-items of each configuration block in the parameter list) come from the docstrings of the field / model validators on `__pydantic_decorators__`. If there is no docstring, the block says "None.". The class overview prefers the first paragraph of the class docstring; if there is none, it falls back to "`X` is a Y category.". If the first paragraph of the class docstring of a nested configuration block expanded into the parameter list (`<h3>` / `<h4>`) is not empty, that class overview sentence is also output after the heading and before the parameter table (for example `Metadata`'s "quantization configuration metadata: ...").

Class discovery, YAML paths, and the reference graph still read Python annotations and registries: the items of `process` in the JSON Schema only `$ref` to the dispatch base class and do not expand all public processors.

#### 4.3.5 Reference Graph

For each class, first compute the "referenced configurations" `nested_refs` (forward graph):

- A field annotation points to `AutoProcessorConfig`: list all public processors for that field, with the relationship `type dispatch`.
- It points to a save format: list all public savers, with the relationship `type dispatch`.
- It points to a strategy / evaluation service base class: list the public strategies / evaluation services, with the relationship `type dispatch`.
- A single base class + registered derived classes (such as `TLQOpConfig` → `MinmaxTuneOpConfig` / `RoundTuneOpConfig` / `TrainableSmoothOpConfig`): the derived classes are referenced as same-path `nested objects`, making the field a dispatch group.
- `ModelslimConvertServiceConfig.preprocess`: two kinds of preprocessing steps, `rename` / `convert`, with the relationship `type dispatch`; the dispatch base class name is composed as `PreprocessConfig`.
- An ordinary nested BaseModel: one `nested object`. Processor / saver subclasses are skipped on this path to avoid duplication with the dispatch expansion.

The "referenced configurations" section is no longer rendered (the `parents` reverse lookup table is still kept internally so that example YAML can be traced back to its host). When expanding nesting, the BFS starts from `nested_refs` and renders the reachable nested configurations into the corresponding class-name sub-block of the "Parameter List" according to the context path at the reference site; the task (quant) page expands the spec as well. References to a spec that is not documented as a standalone page (for example a tuning strategy's `template` pointing to `ModelslimV1ServiceConfig`) are redirected to the Section 2.2 anchor of the owning task page (the block order of a task page is fixed: the root block is §2.1 and the spec is the first nested block, §2.2); configuration block numbering is at most three levels (`2.x.y` / `<h4>`), and anything deeper is promoted to a sibling of the grandparent with continued numbering; cross-directory links compute their relative path from the output subdirectory.

**type dispatch rendering**: a field path with 2 or more references, or one that contains `type dispatch`, is judged to be a dispatch group. It is rendered as one `<h3 id="...">2.x Base Class Name (Dispatched by Type)</h3>` block: the block contains the base class parameter table (when the base class itself is a real model) and a "derived classes" list (each item = class name, `type` value, a one-line description, and an in-page sub-block anchor or a standalone-page relative link), followed by the parameter table + configuration constraints of each derived class as sibling or lower-level numbered sub-blocks (still subject to the three-level numbering limit). The same base class is rendered only once per page, and the alias in the "Referenced Config" column of subsequent dispatch fields points to that block's anchor (for example, `spec.process[]` and `spec.prior[].process` on a task page share the same `AutoProcessorConfig` block). For dispatch fields, the "Referenced Config" column uniformly writes `"This page <a href="#anchor">§x.y</a>"` to point to the base class block anchor, instead of listing each subtype or writing the summary text "dispatched by `type`, see the corresponding configuration document". Configuration block headings carry visible numbers, and both headings and in-page navigation uniformly use HTML tags (`<h3 id>` / `<h4 id>` / `<a href="#...">`) rather than Markdown's `{#anchor}` attribute syntax.

#### 4.3.6 Example YAML

The "Complete Configuration Reference" is generated according to Template 04 / FE-04: syntactically correct, placed at a real field path, and loadable by the root configuration. If no example satisfying these conditions can be extracted, the whole section is omitted.

1. Prefer the model-level JSON Schema `examples` / `example`.
2. Otherwise, extract the mandatory fields and the optional fields with default values **of this configuration** from the default values, the unique `Literal`/`Enum` values, and the enumerations / numeric lower bounds in the parameter table. Enumerations skip `placeholder` and prefer typical values (such as `int8` and `per_channel`). An unconstrained mandatory string is not fabricated as `"example"`; for the quantization algorithm name `method`, the typical registered value `minmax` in the repository is used.
3. Use `wrap_example_at_path()` to wrap it into `apiversion` + `spec`: segments containing `[]` in the path are written as single-element lists (`spec.process[]`, `spec.save[]`, `spec.prior[]`, and so on). A nested configuration is attached under the public `type` host that references it, traced upward along the "referenced configurations"; a sibling Literal (such as `stage: 1`) marks the dispatch. The `apiversion` of a task root configuration takes the service class Literal and must not be written as `Unknown`. Union carriers of internal `type` values (such as `stage_config`) are flattened to the parent level, consistent with the YAML that the loader actually accepts.
4. Internal `type` values are still processed by `rewrite_example_internal_types()`. When `AutoSaverConfigList` does not allow an empty `save`, one layer of public save format is added.
5. Validate with the `model_validate` of the corresponding task root configuration; if that fails, try adding one item to the empty nested object list and validate again; if it still fails, Section 3 (Complete Configuration Reference) is not output.

**Auto Tuning example (no apiversion root)**: `TuningPlanConfig` has no `apiversion` field, and its `strategy` / `precheck` dispatch depends on the plugin registry, so overall validation fails when no plugin is registered in the docgen environment. Therefore each strategy / evaluation class declares a **complete example of its own subtree** inside the class through `model_config.json_schema_extra.examples` (refer to the configurations under `docs/en/knowledge_base/tuning_strategies/`, removing the optional `precheck` block that triggers plugin dispatch). The generator follows a dedicated path for the tuning root: take the subtree example of the page's own class plus the default subtree of the sibling type (a strategy page is paired with the default evaluation, an evaluation page with the default strategy, and `TuningPlanConfig` takes the defaults of both), and assemble them into a complete `strategy` + `evaluation` form; during validation it first tries `TuningPlanConfig.model_validate`, and on failure falls back to `model_validate` on the concrete strategy / evaluation classes separately.

#### 4.3.7 Navigation

`mkdocs.yml` is not modified by default. Only when `--update-mkdocs` is passed explicitly is the block from `# BEGIN GENERATED QUANT CONFIG NAV` to `# END GENERATED QUANT CONFIG NAV` replaced. The group order is fixed as: Task Configuration, Service Specification, Processor, Save Format, Auto Tuning, Nested Configuration. Manually written composite processor and Auto Tuning entries follow at the end of the generated block; `config/README.md` is a manually written navigation page and does not take part in generation or cleanup.

In the development tree, `config.ini` is under `config/` at the repository root, while the package code expects `msmodelslim/config`. If it is missing before generation, a temporary symlink is created and removed afterwards to avoid an import failure.

## 5. Command-Line API Documentation (Not Extracted Automatically)

Command-line interface documents are written against the `msmodelslim.cli` implementation and `--help`, following Template 05, and are not generated from argparse. Current documents: `docs/en/api_reference/cli/msmodelslim_{quant,analyze,tune}.md`. When the CLI behavior changes, update the corresponding Markdown at the same time, and check it before submission against the [Command-Line API Document Checklist](../development_guide/docs_standards/05_cli_api_contract_checklist.md).

## 6. Template Section Mapping

### 6.1 Template 04 (Configuration)

| Section | Automatic Source | Gap |
|------|----------|------|
| 1 Overview | The class docstring; the class name and the relative source path | Without a class docstring, the overview is rather templated |
| 2 Parameter List | JSON Schema; organized into blocks by configuration class name, with visible numbers in the block headings (`2.1 Class name`, `2.x Derived class name`); each block contains a parameter table and a "Configuration Constraints" sub-item; nested configurations (including the task spec) are expanded into the corresponding class-name sub-block of this section, and a class overview sentence is output at the start of the block when a class docstring exists; type dispatch fields are rendered as a base class block (containing a parameter table and a derived class list) + `<h4>` sub-blocks for each derived class; headings and in-page navigation use HTML tags | When there is no `description`, the meaning is `—`; AfterValidator entries not written into the schema do not appear |
| 3 Complete Configuration Reference | Extractable default values + a real YAML path shell, validated with the root configuration (appears conditionally) | If a mandatory field is missing and cannot be extracted from the constraints, or validation fails, the whole section is omitted |
| 4/5 Compatibility, Cautions | Not generated | Must be written manually or in a separate document when there are breaking changes |

### 6.2 Template 05 (CLI)

Not filled in by the generator. The section content is written against the argparse implementation, runtime validation, and business semantics; template placeholders or filling instructions must not be left in place.

## 7. Validation and Testing

- `--check`: re-renders the configuration documents from the current source code and compares them byte by byte with the files on disk; on a mismatch it prints `drift:` and fails with exit code 1. It is used to prevent generated output from being scrambled by manual edits, or the source from being changed without regeneration.
- `--dry-run`: prints only the paths that would be written.
- The generator scripts are maintained as part of the `skills/docs-management/` skill and no separate unit tests are kept; correctness is ensured through `--check` and the documentation gate checks.

## 8. Maintenance Conventions

When you change the behavior of the configuration interface, update the source annotations at the same time and then regenerate:

```bash
# Daily regeneration (default targets + expand nesting + clean up expired documents)
python3 skills/docs-management/scripts/gen_config_api_docs.py
```

When you change the command-line behavior, update `docs/en/api_reference/cli/` against the implementation and do not run the configuration generator.

| What You Want to Change | Where to Change It |
|------------|--------|
| Field type, default value, choices, mandatory | Pydantic Field |
| Configuration "meaning" | `Field(description=)` |
| Configuration overview | Class docstring |
| Cross-field constraints | Validator docstring |
| Length / range numbers | `Field(max_length=)` / `ge` and so on (they enter the JSON Schema) |
| CLI parameters, mutual exclusion, examples, security notes | `docs/en/api_reference/cli/` (Template 05) |
| Complete configuration reference | Extractable default values; they must pass root configuration validation and be located at a real path. If they cannot be extracted, the section is not generated |
| Whether a class is internal | Whether the `type` Literal starts with `_` |

To add a public processor: implement it and register it with `AutoProcessorConfig`, make sure `type` does not have the `_` prefix, and regenerate; the document, the index, and all `process[]` dispatch tables then appear. The generator does not need to be changed.
