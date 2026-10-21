# Terminology Entry Checklist

> Used to check terminology entry documents that are created or reworked based on the *[Terminology Entry Template](01_term_glossary_template.md)*. For the general check items, see the *[Common Checklist](00_common_checklist.md)*; for the check mechanism (severity levels, conclusion rules, check execution, and the waiver mechanism), see Section 4.2 of the *[msModelSlim Documentation Standards](./README.md)*. The following are the check items specific to the terminology entry type, separated into tables by severity level.

## 1. Check Items

### 1.1 ERROR (Mandatory)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| TE-01 | File naming | Ordinary entries are named `term_<english_name>.md`, where the English name is all lowercase and multiple words are connected by underscores; category entries and index documents are named `README.md`. | Rename the files according to the convention |
| TE-02 | Required sections | All four of the following are present: the level-1 heading, the top information box, `## 1. Overview`, and `## 2. Term Description`, in the same order as the template. | Add the missing sections and adjust the order |
| TE-03 | Top information box fields | The top information box contains at least the "Entry Category" and "English Name" fields. | Add the missing fields |
| TE-04 | Form of related term items | Check only when the "Related Terms" section exists: each item is a Markdown hyperlink whose anchor text matches the target entry name and which indicates the relationship type. | Change them to hyperlinks, unify the anchor text, and add the relationship type |
| TE-05 | Overview not empty | `## 1. Overview` has been filled in and is not empty. | Add the overview |
| TE-06 | Registration of pending entries | Check only when the "Related Terms" section exists: entries that are linked but not yet created have been registered in the TODO list at the top of the entry file. | Register the pending entry at the top of the entry file |
| TE-07 | Entry uniqueness | The same term has only one entry file under the same domain directory, that is, the files are named differently and the English names are not duplicated. | Merge or rename the entry files |

### 1.2 WARN (Recommended)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| TW-01 | Level-1 heading | The level-1 heading on the first line of the file corresponds to the entry name, without appending suffixes such as "Terminology Entry". | Adjust the level-1 heading |
| TW-02 | Consistent English name | The "English Name" in the top information box is spelled the same as the English name in the file name; the letter case may differ. | Unify the English name in the two places |
| TW-03 | Entry category value | The "Entry Category" value is within the enumeration set allowed by the template; non-top-level entries fill in a link to the parent entry. | Fix the value or add the link to the parent entry |
| TW-04 | Heading separator | There is a `---` separator line between adjacent level-2 headings, and it is not added before level-3 or level-4 headings. | Add or remove the `---` separator |
| TW-05 | Related workflow links | Check only when the "Related Workflows" section exists: each item is a hyperlink pointing to a workflow guide that actually exists, and at least one item describes its relationship to the workflow. | Fix the links and add the relationship descriptions |
| TW-06 | Term hyperlinks | Terms mentioned in the overview and the body text are presented as hyperlinks rather than plain text whenever they have or will have a standalone entry. | Change them to hyperlinks |
| TW-07 | Three elements of the overview | `## 1. Overview` answers three questions: what the concept is, what problem it solves, and what its core characteristics are. | Rewrite the overview |
| TW-08 | Reference document fields | Check only when the "Reference Documents" section exists: academic references include the title, authors, conference or journal, year, and link. | Complete the reference document fields |
| TW-09 | Reference document order | Check only when the "Reference Documents" section exists: the first entry is an authoritative source. For algorithm entries the first entry is the original paper, and for data format entries the first entry is the corresponding standard document (such as the IEEE 754 or OCP MX specification). | Reorder the reference documents to put the authoritative source first |
| TW-10 | Source of the first proposal | The "First Proposed" field in the top information box has a clear source, that is, the authors, conference or journal, and year; when there is no clear source, information without a source such as "msModelSlim proprietary algorithm" is not filled in. | Add the real source or delete the line |
| TW-11 | External link sources | Check only when the "Reference Documents" section exists: external links come from sources worth citing, that is, one of arXiv, an official conference or journal, or a document in the msModelSlim repository. | Replace them with links to authoritative sources |
| TW-12 | Backlinks | Each entry is referenced by at least one other entry in its "Related Terms" section, that is, the in-degree is not less than 1. | Add the backlinks |
| TW-13 | Weakly connected graph | The terminology entries as a whole form a weakly connected graph, that is, starting from any entry one can reach any other entry through related-term links. | Add related terms across clusters |

## 2. Usage Notes

- **Validation scope**: the `term_*.md` terminology entries in each domain and its subdirectories under [`docs/en/knowledge_base/`](../../../knowledge_base/README.md), placed in the directory of the domain they belong to. The repository organizes entry documents in three tiers -- domain, category, and entry: entry documents include `term_*.md` and `<format_name>.md` for quantization formats, and are checked against this checklist; category entries and index documents are `README.md`, are not required to use the `term_*` naming, are checked only for their structure, and do not take part in the in-degree and connectivity statistics of TW-12 and TW-13.
- **Supplement to CE-12**: a link to an entry that has not yet been created but has been registered in the TODO list at the top of the entry file counts as "planned" and is not counted as a broken link when checking CE-12 of the common checklist; if it is not registered, it counts as a CE-12 failure and the author is prompted to register it or fix the link. For the registration requirements, see TE-06.
