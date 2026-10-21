# Workflow Guide Checklist

> Used to check workflow guide documents that are created or reworked based on the *[Workflow Guide Template](02_process_guide_template.md)*. For the general check items, see the *[Common Checklist](00_common_checklist.md)*; for the check mechanism (severity levels, conclusion rules, check execution, and the waiver mechanism), see Section 4.2 of the *[msModelSlim Documentation Standards](./README.md)*. The following are the check items specific to the workflow guide type, separated into tables by severity level.

## 1. Check Items

### 1.1 ERROR (Mandatory)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| PE-01 | Core sections | The document has the three core sections "Scope", "Inputs and Deliverables", and "Procedure". | Add the missing sections |
| PE-02 | Inputs and deliverables | The "Inputs and Deliverables" section contains at least one input item and at least one deliverable item. | Add an input item or a deliverable item |
| PE-03 | Procedure | The "Procedure" section contains at least one step, and each step contains both "Action" and "Output". | Add steps, or add the action and output for a step |

### 1.2 WARN (Recommended)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| PW-01 | Title and scope | The title expresses the workflow topic; the scope describes the applicable objects, scenarios, or boundaries. | Adjust the title or the scope |
| PW-02 | Inputs and deliverables | Input and deliverable items are locatable and describe the necessary source, constraint, or acceptance method. | Add the source, constraint, or acceptance method |
| PW-03 | Step completeness | The steps are in a reasonable order; inputs, pre-execution checks, pass conditions, audit records, and troubleshooting have been added as needed. | Add the step elements |
| PW-04 | Commands and references | Commands, variables, and paths have the necessary descriptions; referenced capabilities have an evidence source. | Add the descriptions or evidence sources |
| PW-05 | Special scenarios | When accuracy, performance, security, product form, resources, release, or runtime are involved, the corresponding baseline, limits, acceptance, monitoring, or rollback arrangements have been described. | Add the corresponding arrangements |
| PW-06 | Process relationships and prerequisites | Check only when the "Process Relationships and Prerequisites" section exists: the parent process, prerequisites, and follow-up operations are described clearly and match the actual workflow. | Fix or add the relevant descriptions |
| PW-07 | Process overview | Check only when the "Process Overview" section exists: the section precedes the "Procedure" section and contains a flowchart, a Mermaid diagram, or another schematic. | Adjust the section position or add a schematic |
| PW-08 | Acceptance conditions | Check only when the "Acceptance Conditions" section exists: each item describes the pass requirements at the level of the whole workflow. | Add the acceptance items |
| PW-09 | Troubleshooting | Check only when the "Troubleshooting" section exists: each item describes the symptom and the handling or rollback direction. | Add the troubleshooting items |
| PW-10 | Case list, terminology, and interface documentation list | Check only when the "Case List", "Terminology", or "Interface Documentation List" section exists: each item keeps only a name and a one-sentence description together with a clickable hyperlink, instead of expanding details in long body text. | Reduce them to a name, a one-sentence description, and a link |
| PW-11 | Security notes | Check only when the "Security Notes" section exists: the security notes match the actual risks of the workflow. | Fix the security notes |
| PW-12 | Product form and resource constraints | Check only when the "Product Form and Resource Constraints" section exists: the product form, dependencies, and capacity constraints match the actual workflow. | Fix the product form and resource constraints |

## 2. Usage Notes

- **Validation scope**: workflow guides and usage guides written based on the *[Workflow Guide Template](02_process_guide_template.md)*, including business workflow guides and tool usage guides.
