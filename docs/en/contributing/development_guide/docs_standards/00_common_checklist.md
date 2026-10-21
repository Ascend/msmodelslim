# Common Checklist

> This checklist is part of the *[msModelSlim Documentation Standards](./README.md)* and lists the general check items applicable to all documents, each with a criterion and a recommended action. For the check mechanism (severity levels, conclusion rules, check execution, and the waiver mechanism), see Section 4.2 of the *[msModelSlim Documentation Standards](./README.md)*; for the item numbering rules and type codes, see [Section 7.1](./README.md#71-checklist-authoring-conventions). Type-specific checklists reference the general items in this checklist, which does not define any type-specific items.
> Scope: documents within the template system must satisfy all items in this checklist; documents outside the template system (installation guides, quick start guides, FAQs, release notes, and so on) must satisfy at least the ERROR items.

## 1. General Check Items

### 1.1 ERROR (Mandatory)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| CE-01 | Placeholder content | The document contains no leftover template placeholders (such as `{{ name }}`) or placeholder samples (such as `xxx`, `TODO`, `FIXME`, `to be added`, or `TBD`). | Replace them with actual content |
| CE-02 | Template comments and scaffolding | The document contains no leftover template comments or template scaffolding sections. Template comments are comments that guide the author, usually starting with `> **Note**:` and containing hints such as "replace xxx with", "required", or "optional", and they are not limited to the blockquote form; template scaffolding sections are instructional sections used only to guide authoring and unrelated to published content. | Delete the template comments and template scaffolding sections, or rewrite them as formal text for readers |
| CE-03 | Sensitive information | The document contains no keys, tokens, internal addresses, personal paths, or other sensitive information. | Remove the sensitive information and replace it with a placeholder description |
| CE-04 | Heading structure | The document has exactly one level-1 heading, excluding example headings inside code blocks; heading levels progress step by step from level 2, without skipping levels and with continuous levels. | Fix the headings or adjust the heading levels |
| CE-05 | Book-title marks | When citing the title of another document, set the title off with book-title marks, which are rendered as italics in English; a linked citation is written as `*[Target Document Official Title](relative-path)*`. | Add book-title marks around the document title |
| CE-06 | Number and unit spacing | In a Chinese-language context, leave no space between a number and a quantifier or unit, for example write `≤5个`, `5类`, and `80~200字`, not `≤ 5 个` or `5 类`. | Remove the space between the number and the unit |
| CE-07 | Formulas | Formulas are written in LaTeX syntax, with inline formulas in `$...$` and block formulas in `$$...$$`; formula markup must not be wrapped in code block markers. Below each block formula, describe the meaning of each variable in a list. | Use LaTeX syntax and add descriptions of the formula variables |
| CE-08 | Images | Images are placed in the `figures/` directory at the same level as the document and referenced by relative paths. | Move the images to the sibling `figures/` directory and fix the reference paths |
| CE-09 | Punctuation | In a Chinese-language context, use Chinese punctuation and do not mix Chinese and English punctuation; separate coordinate words with the enumeration comma, and separate coordinate quotation marks or book-title marks with an enumeration comma rather than placing them adjacent; quotation marks, parentheses, and book-title marks must appear in pairs. | Fix the punctuation |
| CE-10 | Optional sections | For sections marked with `[OPTIONAL]` in the template, those that do not apply at release time have been deleted in full; those retained have had the `[OPTIONAL]` prefix removed and have been renumbered in order without leaving gaps. | Delete the inapplicable `[OPTIONAL]` sections, and remove the prefix from and reorder the retained ones |
| CE-11 | Markdown structure | Markdown tables have the same number of columns as the header; code blocks are annotated with a language. | Complete the missing table cells or header, and annotate the language of code blocks |
| CE-12 | Documentation links | Inline documentation links are relative paths and resolve to files that actually exist in the repository. | Fix the broken links and change them to resolvable relative paths |
| CE-13 | Heading numbering | The level-1 heading carries no number; level-2 and headings below carry visible numbers, the numbers are consecutive at the same level, and the numbers align with the heading levels (`x` for `##`, `x.x` for `###`, `x.x.x` for `####`). The same applies when in-page anchors are carried by HTML headings. | Add numbers to the headings, or fix the numbers and levels |

### 1.2 WARN (Recommended)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| CW-01 | Language and sentence structure | The document uses a user-facing technical documentation style; well-established general abbreviations (such as AI, API, FAQ, CLI, and SIG) may be used directly, while other abbreviations are spelled out in full when they first appear; sentences are complete and unambiguous in meaning. | Adjust the style, spell out uncommon abbreviations, and rewrite sentences with incomplete elements or unclear meaning |
| CW-02 | Consistent terminology | The same concept is referred to by the same name throughout the document and is not confused with other meanings. | Unify the name of a concept and eliminate mixed usage |
| CW-03 | Heading readability | Headings reflect the document topic; it is recommended that heading levels do not exceed four. | Adjust the wording or the depth of the headings |
| CW-04 | Image display size | Image references specify the display size through the `width` attribute of the HTML `<img>` tag, so that oversized images do not hinder reading. | Specify the display size for oversized images |
| CW-05 | Rendered result | Line breaks and indentation render as expected. | Adjust the line breaks or indentation |
| CW-06 | Command and path generalization | Commands and paths in code blocks are generalized with environment variables rather than hard-coded; system paths and fixed installation locations are exceptions. | Replace the concrete values in commands and paths with environment variables |
| CW-07 | External links | External links use the HTTPS protocol. | Replace them with HTTPS links |
