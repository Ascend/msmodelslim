# msModelSlim Documentation Standards

## 1. Overview

### 1.1 Purpose

This standard defines the requirements for writing, reviewing, and maintaining msModelSlim documentation, providing a unified basis for documentation work. Its main purposes are as follows:

- **Ensuring quality**: define the writing requirements and check mechanism for each document type, so that the documentation is accurate, structurally complete, and clearly expressed, meeting the needs of the target readers.
- **Guiding authoring**: provide document classification methods, templates, and writing requirements, so that documentation authors have clear rules to follow, reducing the cost of writing and learning.
- **Supporting review**: provide an objective, executable basis for checking, so that documentation review conclusions are verifiable and subjective judgment and communication costs are reduced.
- **Serving readers**: organize the documentation from the perspective of user business, so that it is easy to search, navigate, and understand, enabling readers to quickly find the information they need.
- **Supporting evolution**: provide a stable standards foundation for the continuous maintenance, expansion, and iteration of the documentation system.

### 1.2 Scope

This standard applies to the documentation under the `docs/` directory and covers the whole process of documentation from planning and writing to review and maintenance; the Chinese documentation is the mandatory scope, and other language versions are for reference.

- **Mandatory scope**: the `docs/zh/` Chinese documentation tree is the mandatory scope and the direct basis for documentation checking and review.
- **Reference scope**: the `docs/en/` English tree and other language versions are implemented for reference only and are not included in the mandatory check scope.
- **Coverage of all document types**: this includes both the documents within the template system (5 types in total, see [Section 2.2](#22-document-types)) and the documents outside the template system, such as installation guides, quick start guides, FAQs, release notes, and legal statements.
- **Out of scope**: content carried by code, such as code comments and commit messages, is not within the scope of this standard; for its requirements, see the *[Coding Standards](../coding_standards.md)*.

### 1.3 Target Readers

This standard is intended for all roles involved in msModelSlim documentation work. The focus of each role is as follows:

| Reader Role | Main Purpose |
| --- | --- |
| Documentation author | Select the document type according to the document classification method, write the content with the corresponding template, and submit after self-checking against the checklist. |
| Documentation reviewer | Determine whether a document passes according to the checklist, and give a verifiable review conclusion and revision suggestions. |
| Documentation maintainer | According to the document lifecycle and navigation relationships, add, update, merge, or retire documentation as features evolve, keeping the documentation system consistent. |
| SIG review member | According to Section 4.2 and the standard maintenance section, review documentation waiver requests and standard changes, and steer the evolution of the standard itself. |

> Ordinary users do not need to read this standard when consulting msModelSlim documentation; they can read the user-facing documentation content directly.

## 2. Documentation System

### 2.1 Documentation Tiers

The `docs/zh` documentation is divided into six tiers, each defined as follows:

| Tier | Directory | Role |
| --- | --- | --- |
| Getting started tier | `install_guide/`, `quick_start/` | Getting-started guidance: installation and environment preparation, quick start |
| Usage tier | `user_guide/`, `best_practices/` | Operational knowledge: business workflow guides, tool usage guides, and practical cases |
| Knowledge tier | `knowledge_base/` | Domain knowledge: concepts such as quantization algorithms, tuning strategies, and quantization formats, and their applications |
| Interface tier | `api_reference/` | Interface reference: API documentation such as command-line and configuration references |
| Support tier | `support/` (FAQ), `release_notes/`, `legal/` | Auxiliary information: frequently asked questions, release notes, and legal statements |
| Governance tier | `contributing/` | Governance standards: contribution guides, design documents, development standards, and documentation standards |

### 2.2 Document Types

**Documents within the template system (5 types)**: the structure follows the corresponding template, and the corresponding checklist is used before submission:

| Type | Tier | Role | Template | Checklist |
| --- | --- | --- | --- | --- |
| Terminology entry | Knowledge tier | Declarative knowledge: describes the definition, principles, properties, and relationships of a term | *[Terminology Entry Template](./01_term_glossary_template.md)* | *[Terminology Entry Checklist](./01_term_glossary_checklist.md)* |
| Workflow guide | Usage tier | Procedural knowledge: describes the applicable scope, inputs and deliverables, and procedure | *[Workflow Guide Template](./02_process_guide_template.md)* | *[Workflow Guide Checklist](./02_process_guide_checklist.md)* |
| Case reference | Usage tier | Practical case: records the background, environment, steps, results, and lessons | *[Case Reference Template](./03_general_case_template.md)* | *[Case Reference Checklist](./03_general_case_checklist.md)* |
| Configuration document | Interface tier | Interface reference: describes configuration fields, constraints, and values | *[Configuration Document Template](./04_config_document_template.md)* | *[Configuration Document Checklist](./04_config_document_checklist.md)* |
| CLI document | Interface tier | Interface contract: describes command names, parameters, and invocation conventions | *[Command-Line API Document Template](./05_cli_api_contract_template.md)* | *[Command-Line API Document Checklist](./05_cli_api_contract_checklist.md)* |

> **Audience and level of detail**: usage guides (workflow guides) target beginner users and use detailed steps to guide them through selecting parameters, filling in configurations, and executing commands; API documents and configuration documents target advanced users for quick reference and only need to specify reference information such as command names, parameters, and values, without expanding the steps.

**Documents outside the template system**: installation guides, quick start guides, FAQs, release notes, legal statements, and the description files under each directory do not use a template and are distributed across multiple tiers (for example, the installation guide and quick start guide in the getting started tier, and the FAQ and release notes in the support tier). They follow the general writing requirements in the *[Common Checklist](00_common_checklist.md)* and must at least satisfy the items marked as ERROR therein.

### 2.3 Choosing a Document Type

Determine the document type according to the nature and purpose of the content to be written:

| Content Characteristic | Document Type |
| --- | --- |
| Description of the definition, principles, and properties of a concept or algorithm | Terminology entry |
| Description of the procedure and execution flow of a task | Workflow guide (including usage guides) |
| Record of the background, process, results, and lessons of a practical activity | Case reference |
| Description of the fields, constraints, and values of a configuration interface | Configuration document |
| Name, parameters, and invocation conventions of a command-line interface | CLI document |
| Auxiliary description without a fixed structure | Document outside the template system |

## 3. Directory Structure

### 3.1 docs Directory Structure

In addition to the Chinese documentation `docs/zh/`, `docs/` also contains the English documentation `docs/en/`, the site home page `docs/index.md`, and the static asset directories `docs/assets/` and `docs/stylesheets/`. This standard takes `docs/zh/` as the mandatory scope (see [Section 1.2](#12-scope)), and the documentation directories are arranged in the tier order of [Section 2.1](#21-documentation-tiers) as follows:

```text
docs/en/
├── install_guide/      # Installation guide
├── quick_start/        # Quick start guide
├── user_guide/         # User guide: describes business workflows and tool features
├── best_practices/     # Typical cases
├── knowledge_base/     # Business knowledge base: introduces term concepts such as PTQ and W8A8, and organizes knowledge such as quantization algorithms
├── api_reference/      # Interface reference: makes it easy for users to look up interface and configuration information
├── support/            # Problem support
├── release_notes/      # Release notes directory
├── legal/              # All legally related files such as licenses, statements, and disclaimers
└── contributing/       # Contribution guide directory
    ├── contributing_guide.md    # Contribution guide
    ├── design/                  # Design documents
    └── development_guide/       # Technical documents
```

### 3.2 Naming Conventions

- Directory and file names use lowercase English letters, with words connected by underscores, and must not contain Chinese characters, spaces, or special characters.
- The entry description file of a directory is uniformly named `README.md`.
- If a directory or file name must be consistent with a code entity (such as a Python class name), the first rule does not apply.

## 4. Writing and Checking

### 4.1 Writing Requirements

Documentation writing requirements are divided into general requirements and type-specific requirements. The former do not vary with the document type, while the latter differ by document type:

- **General requirements**: apply to all documents (including those outside the template system). They cover writing conventions unrelated to the document type, such as heading structure, punctuation, terminology, links, figures, tables, and formulas, and are uniformly given in the general items of the *[Common Checklist](00_common_checklist.md)*. The general requirements are identical for all document types, so no template is provided and they are carried directly by the checklist items.
- **Type-specific requirements**: apply to the 5 document types within the template system. They cover the section structure and the required and optional sections, as given in the corresponding template. Their specific criteria are given in the checklist for the corresponding type; for the mapping between templates and checklists, see [Section 2.2](#22-document-types).

### 4.2 Check Requirements

Documentation checking is based on the general items in the *[Common Checklist](00_common_checklist.md)* and the specific items in the checklist for the corresponding type. Before writing, read the corresponding template and the *[Common Checklist](00_common_checklist.md)*; before submission, check item by item against the two sets of items above.

- **Severity levels**: `ERROR` items are mandatory, and `WARN` items are recommended. The level of an item is determined as follows: those with objective criteria that affect readers' comprehension are classified as `ERROR`; those requiring subjective judgment or having many reasonable exceptions are classified as `WARN`. When a single requirement involves both kinds of criteria, it is split into two items with separate levels.
- **Conclusion rules**: if any `ERROR` item is not satisfied, the check conclusion is **fail**; if only `WARN` items are not satisfied, the check conclusion is **revision required**; if all items are satisfied, the check conclusion is **pass**.
- **Check execution**: run a full check after adding a document, and an incremental check after modifying a document (checking only the changed files and the files they affect). The check may be performed by a person or an AI agent, verifying item by item against the checklist.
- **Waiver mechanism**: if a reasonable cause genuinely prevents satisfying an item, declare `<!-- waiver: number Reason: xxx -->` at the top of the document; the item is then skipped during checking and the reason is recorded. Waivers must not be used to bypass more than 2 `ERROR` items; there is no upper limit on the number of waived `WARN` items.

## 5. Documentation Navigation

Documents should establish navigation relationships through links, which fall into two categories: the business workflow diagram and the knowledge graph.

### 5.1 Business Workflow Diagram

The business workflow diagram expresses the hierarchical relationship between business operation workflows and tool usage workflows.

- **Linking requirement**: a business operation workflow document should link to every tool usage workflow document it depends on; when it depends on multiple tools, link to each of them separately, nesting level by level.
- **Coverage requirement**: apart from business operation workflow documents, every workflow document should be linked by an upper-level workflow document; no workflow document may remain unlinked.
- **External tool requirement**: when a workflow uses a third-party tool outside this repository, provide a link to that tool's official usage guide. External links pointing to documents outside the repository are not subject to the templates and checklists of this standard.
- **Node requirement**: each node in the diagram should be a workflow guide document that follows the *[Workflow Guide Template](./02_process_guide_template.md)*; a business operation workflow is itself an independent workflow document and must not be replaced by a virtual grouping.
- **Scope requirement**: terminology entries, interface documents, and cases must not be nodes of the business workflow diagram; they should be referenced by workflow documents in the form of links.

### 5.2 Knowledge Graph

The knowledge graph expresses the relationships between term concepts.

- **Linking requirement**: a terminology entry should link to other related entries and indicate the relationship type, and should also link to related workflow documents.
- **Connectivity requirement**: each entry should be linked by at least one other entry, and all entries should form a weakly connected graph with no isolated entries.

## 6. Document Lifecycle

- **Addition and release**: first determine the document type according to [Section 2.3](#23-choosing-a-document-type), then determine the tier and directory according to [Section 2.1](#21-documentation-tiers) and [Section 3.1](#31-docs-directory-structure), write the content with the corresponding template, and complete the check according to [Section 4.2](#42-check-requirements). When releasing, the navigation relationships specified in [Section 5](#5-documentation-navigation) must be wired up.
- **Review and release**: a document must pass the check in [Section 4.2](#42-check-requirements) and be confirmed by a documentation reviewer before release.
- **Changes**: when features, configurations, or interfaces change, reader feedback is received, or the underlying template and checklist are updated, the document should be updated accordingly and re-checked according to [Section 4.2](#42-check-requirements).
- **Merging, splitting, and renaming**: when documents are merged, split, or renamed, delete the original document files and update all links pointing to the original documents in [Section 5](#5-documentation-navigation) and in other documents so that they point to the merged, split, or renamed documents.
- **Deprecation and retirement**: when a feature is retired or a document is replaced by another document, declare this in the release notes and handle all links pointing to that document (either repoint them to the replacement document or delete them).

## 7. Standard Maintenance

- **Maintenance responsibility**: this standard is maintained by the SIG, which accepts revision suggestions, organizes reviews, drives synchronous updates of the templates and checklists, and maintains the revision history.
- **Feedback channels**: when documentation authors, reviewers, or maintainers find that the standard is missing, contradictory, or inapplicable, they should submit revision suggestions through a repository issue or the regular SIG meeting, explaining the affected scope and the rationale.
- **Review**: changes to the standard must pass a SIG review before being merged, and the reason for the change and the affected scope must be described in the revision history.
- **Synchronous evolution**: when a template is added or adjusted, the corresponding checklist must be updated at the same time; when a document type is added, allocate a type code according to [Section 7.1](#71-checklist-authoring-conventions). The document type table and the directory structure in this standard are maintained in sync.
- **Effectiveness and transition**: this standard takes effect for newly written and newly revised documents from the date it is merged; already released documents are checked against this standard at their next substantive revision. When batch remediation is genuinely required, the SIG sets a transition period and describes it in the revision history.

### 7.1 Checklist Authoring Conventions

The item numbering and type codes of the checklists follow the conventions below. When a type-specific checklist references an item of the *[Common Checklist](00_common_checklist.md)*, it writes the number directly, for example "see CE-05".

**Numbering format**: in the checklists defined by this standard, all items use `<Type Code><Level Code>-<two-digit serial number>`, for example `CE-05` denotes the 5th `ERROR` item of the common checklist. Within a single checklist, `ERROR` and `WARN` items are numbered consecutively from `01`, with no gaps.

**Type code registry**:

| Checklist | Type Code | ERROR Number | WARN Number |
| :--- | :--- | :--- | :--- |
| Common checklist | C | CE-nn | CW-nn |
| Terminology entry checklist | T | TE-nn | TW-nn |
| Workflow guide checklist | P | PE-nn | PW-nn |
| Case reference checklist | G | GE-nn | GW-nn |
| Configuration document checklist | F | FE-nn | FW-nn |
| Command-line API document checklist | A | AE-nn | AW-nn |

**Uniqueness of type codes**: a type code is unique among the checklists defined by this standard and must not be reused. When adding a document type, first allocate a type code in this table, then write the corresponding checklist, and register the template and checklist according to [Section 2.2](#22-document-types).

**How optional sections are referenced**: whether an optional section exists is defined solely by the `[OPTIONAL]` marker in the template; checklists do not list them again. When a checklist references an optional section in a template, it should locate it by section name and must not use the section number from the template: on release, optional sections are deleted and renumbered according to `CE-10` of the common checklist, so the numbers are not consistent across documents of the same type. Check requirements for optional sections are written as conditional items beginning with "check only when the 'section name' section exists"; if the section is not written, no check is performed.

**Exception handling**: a checklist must not set exception clauses for a specific document type or a specific document, so that it does not lose its binding force by accumulating clauses such as "not mandatory" or "except" case by case. If an item does not apply to a document type as a whole, define the validation scope in the "Usage Notes" section of that checklist; if an individual document genuinely cannot satisfy an item for a reasonable cause, declare it at the top of the document according to the waiver mechanism in [Section 4.2](#42-check-requirements).

### 7.2 Revision History

| Date | Version | Change Description |
| --- | --- | --- |
| 2026-08-05 | v1.0 | First release. |
| 2026-08-07 | v1.1 | The entry document type was renamed to "terminology entry", and the template and checklist files were renamed to term_glossary; book-title marks in the body text and broken examples were corrected. |
| 2026-08-12 | v1.2 | Added links to book-title-marked title citations (*Workflow Guide Template*, *Terminology Entry Template*, *Common Checklist*) to align with CE-05 and CE-12. |
| 2026-08-12 | v1.3 | Unified the anchor text of book-title-marked citations to the `*[Title](link)*` form, and converged the CE-05 example to a single canonical form. |
| 2026-08-19 | v1.4 | The quantization configuration document template and checklist were adjusted along with the generator structure: the parameter list is blocked by configuration class, nested blocks output a class overview, the separate "referenced configurations / referenced by" sections were removed, and anchors were unified to HTML. |
| 2026-09-12 | v2.0 | Breaking revision: (1) restructured the standard so that writing requirements and check requirements are separated, with the general writing requirements moved to the *[Common Checklist](00_common_checklist.md)* and the check mechanism moved to Section 4.2; (2) unified the checklist system, changing the numbering to `<Type Code><Level Code>-serial number` and registering type codes, aligning type naming with the type table and renaming files accordingly, and reclassifying `ERROR` and `WARN` by "objective criteria + impact on readers' comprehension", which substantially reduced the number of items and eliminated duplication with the common checklist; (3) moved the checklist authoring conventions (numbering rules, type code registry, optional section references, exception handling) up to Section 7.1 and prohibited type-level or document-level exceptions within a checklist; (4) eliminated dual sources: optional sections are defined solely by the template `[OPTIONAL]` marker and are no longer listed in checklists, and the "Complete Configuration Reference" is established as optional but recommended. |
| 2026-09-12 | v2.1 | Corrected wording and punctuation in the body text so that the standard itself satisfies the check requirements it publishes: (1) unified quoted parentheses to Chinese quotation marks (`""`) instead of corner brackets, aligning with CE-09; (2) removed extra spaces between numbers and quantifiers, aligning with CE-06. Only wording is affected; no item criterion is changed. |
| 2026-09-14 | v2.2 | Completed the heading numbering requirements: (1) added `CE-13` heading numbering to the common checklist, clarifying that the level-1 heading carries no number while level-2 and headings below carry visible numbers aligned with their levels, which removes the problem that "renumbering" in `CE-10` previously had no explicit basis; (2) reduced duplication, keeping only the heading separator in `TW-04` and renaming the anchor style in `FW-07` to keep only the anchor form; (3) restored the heading numbering of the common checklist itself for consistency with the other checklists. |
