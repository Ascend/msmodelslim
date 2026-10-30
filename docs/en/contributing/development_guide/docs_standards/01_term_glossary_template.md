# {{ term_name }}

> **Note**:
> This file is the **entry template** of the Quantization Terminology Encyclopedia, used to guide developers in adding or completing terminology encyclopedia entries in the msModelSlim knowledge base.
> The level-1 heading contains only the entry name `{{ term_name }}`; do **not** append a suffix such as "Terminology Entry".
> `{{ placeholder }}` is a placeholder to be filled in, and `[OPTIONAL]` sections are kept as needed. The notes inside the template are marked with `> **Note**:` and must be deleted from the formal entry.
> The examples in the template (such as the link `XX...`, the path `xxx.md`, and configuration snippets) are all placeholder examples, and it is normal for such links to be invalid; in a formal entry they must be replaced with the actual document title and relative path.
> After filling it in, check item by item for low-level errors against the *[Terminology Entry Checklist](01_term_glossary_checklist.md)*.
>
> **Template flexibility**: the template sections fall into two categories: "Generally Required" and "As Needed ([OPTIONAL])":
>
> - "Generally Required" sections (Overview and Term Description) must be kept for all entries;
> - "As Needed ([OPTIONAL])" sections (Related Workflows, Related Terms, and Reference Documents) are kept as needed.
>
> **Note: an entry may omit irrelevant sections as appropriate; there is no need to fill in irrelevant content merely to align with the template structure.**

---

>
> **Entry Category**: {{ term_category }}<br>
> **English Name**: {{ english_name }}<br>
> **Application Domain**: {{ application_domain }}<br>
> **First Proposed**: {{ first_proposed }}
>
> **Note**: the fields above are required and are used to quickly locate the basic attributes of an entry. The filling requirements for each field are as follows:
>
> - Entry Category: the value must be within the enumeration set allowed by the template (Quantization Basic Concept / Quantization Algorithms / Outlier Suppression Algorithms / Sensitive Layer Analysis Algorithms / Auto Tuning Strategies / Quantization Data Format / Quantization Parameters / Quantization Configuration Items / Quantization Tool Interface / Quantization Evaluation Metrics / Other).
> - English Name: keep it consistent with the `english_name` in the file name `term_<english_name>.md` (the letter case may differ, but the spelling must match).
> - Application Domain: briefly describe the domains to which the entry applies.
> - First Proposed: fill it in only when there is a clear source (authors, conference or journal, year); if there is no clear source, delete the line.
> - For field checking, see items TE-03, TW-02, TW-03, and TW-10 of the *[Terminology Entry Checklist](01_term_glossary_checklist.md)*.

## 1. Overview

{{ overview }}

> **Note**: overview requirements:
>
> 1. Complete the definition in one paragraph, keeping the length moderate; being concise and accurate is enough.
> 2. Cover three core elements: what it is, what problem it solves, and what its core characteristics are.
> 3. Terminology encyclopedia entries mentioned in the text that already exist or will exist should be presented as hyperlinks: `[entry_name](./term_xxx.md)`. The anchor text of an entry link is the entry name, without book-title marks (a level-1 heading contains no book-title marks); book-title marks are added only when a document title is mentioned separately in the body text, according to CE-05 of the *[Common Checklist](00_common_checklist.md)*.
> 4. Avoid expanding algorithm details or restating complete formulas.
> 5. Related terms mentioned in the overview should be presented as hyperlinks, covering the terminology entries that already exist in the body text or will exist in the future, to facilitate the construction of the knowledge graph.

## 2. Term Description

{{ term_introduction }}

> **Note**: the **Overview** already completes the definition in one paragraph (what it is, what problem it solves, and what its core characteristics are); the term description is **not** a repeated short introduction of the overview, nor is its **length limited**.
> Under this section, expand freely with subsections such as `2.1`, `2.2`, and `2.2.1` as the entry itself requires, for example the definition, principles, use cases, comparison with related types, mode specification, and quantization formulas.
> Align the numbering with the Markdown heading levels: use `##` (level 2) for `x`, `###` (level 3) for `x.x`, and `####` (level 4) for `x.x.x`; do not write `2.4.1` as a level-3 heading.
> Content such as use cases, comparisons, and relationships with other modes must be **subsections** of the "Term Description" section and must not be promoted to a section at the same level as "Term Description".
> The subsequent optional sections listed in the template (Related Workflows, Related Terms, and Reference Documents) remain independent sections, numbered after the term description (usually 3, 4, and 5).
> Authoring suggestions:
>
> - You may refer to the way similar entries are written in the existing knowledge base and keep the style and granularity consistent;
> - You may also start from first principles and introduce the entry from the user's perspective: what the term is, why it is worth knowing, and its relationship to existing concepts.

## 3. [OPTIONAL] Related Workflows

{{ related_processes }}

> **Note**: requirements for the related workflow list:
>
> 1. List the "workflow guide" documents related to this term (such as *One-Click Quantization V1*, *Sensitive Layer Analysis Usage Guide*, and *Auto Tuning Usage Guide*).
> 2. Each item must be a hyperlink pointing to an actual workflow document; writing only the name is not allowed.
> 3. Briefly describe the relationship between the workflow and this term (for example, "this algorithm is used as pre-processing", or "this workflow can trigger the algorithm corresponding to this term").
> 4. If there is no related workflow, this section may be omitted entirely.
>
> Example:
>
> - *[One-Click Quantization V1](../../../user_guide/usage_one_click_quantization.md)*: this algorithm is integrated by default as a pre-processing step for outlier suppression.
> - *[Sensitive Layer Analysis Usage Guide](../../../user_guide/usage_sensitive_layer_analysis.md)*: the analysis results can be used to roll back the layers that use this algorithm.

## 4. [OPTIONAL] Related Terms

{{ related_terms }}

> **Note**: requirements for the related term list:
>
> 1. Use hyperlinks to point to other terminology encyclopedia entries: `[entry_name](./term_xxx.md)`.
> 2. Avoid plain-text entry names (a clickable jump must be guaranteed).
> 3. Related terms should cover the necessary relationships and be supplemented appropriately according to the breadth of the entry's knowledge.
> 4. After each related term, it is recommended to briefly describe the relationship type in parentheses or after a colon. Relationship types include but are not limited to:
>    the parent concept / a subordinate concept / a similar algorithm / a comparison algorithm / a prerequisite term / a follow-up term / a companion term / an application target / an evaluation metric.
> 5. If this section references terms that are not explained in the body of this entry, they should also be hyperlinks.
> 6. The anchor text of an entry link is the entry name, without book-title marks (consistent with the level-1 heading of the entry); book-title marks are added only when a document title is mentioned separately in the body text, according to CE-05 of the *[Common Checklist](00_common_checklist.md)*.

## 5. [OPTIONAL] Reference Documents

{{ references }}

> **Note**: requirements for reference documents:
>
> 1. Academic papers: title, authors, conference or journal, year, and link (arXiv or official).
> 2. Internal tool documents: title and a relative-path link.
> 3. Reference documents should preferentially add classic papers and the official documents of authoritative specifications, covering the original source, the in-tool implementation, and further reading.
> 4. List multiple entries in a numbered list.
>
> Example:
>
> 1. Xiao G, et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." arXiv preprint arXiv:2211.10438, 2022. <https://arxiv.org/abs/2211.10438>
