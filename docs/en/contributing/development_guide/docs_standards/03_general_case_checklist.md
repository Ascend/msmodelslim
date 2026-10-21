# Case Reference Checklist

> Used to check case reference documents that are created or reworked based on the *[Case Reference Template](03_general_case_template.md)*, and to verify how they connect with the related workflow guide. For the general check items, see the *[Common Checklist](00_common_checklist.md)*; for the check mechanism (severity levels, conclusion rules, check execution, and the waiver mechanism), see Section 4.2 of the *[msModelSlim Documentation Standards](./README.md)*. The following are the check items specific to the case reference type, separated into tables by severity level.

## 1. Check Items

### 1.1 ERROR (Mandatory)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| GE-01 | File naming and title | The file name is `{model_name}_{scenario_type}_case.md`, and the title is the readable form `{model_name}-{scenario_type} Case`. | Rename the file and adjust the title according to the convention |
| GE-02 | Main sections | The five sections "Case Background", "Environment and Versions", "Inputs and Deliverables", "Procedure", and "Results and Lessons Learned" are present, in the same order as the template. | Add the missing sections and adjust the order |
| GE-03 | Covered step correspondence | "Case Background" lists the names of the workflow steps actually covered, and these steps all have a counterpart in "Procedure"; steps that are not covered have been deleted. | Add the missing steps, or delete the steps that are not covered |
| GE-04 | Environment and versions | The actual dependencies of this case have been filled in, rather than template examples or vague descriptions; the "Product form" item is included and states whether it is restricted, and each dependency is given a version number or link. | Add the actual dependencies, product form, or version information |
| GE-05 | Inputs and deliverables | Listed item by item, with a header containing the type, name, source or location, format or constraint, and acceptance method; each column contains the actual values for this case, which are locatable and verifiable. | Add the header or the items, and replace the example values with actual values |
| GE-06 | Procedure | The "Procedure" section contains at least one step, and each retained step contains both "Operations" and "Output". | Add the missing steps, or add the operations and output for a step |
| GE-07 | Summary of key results | The "Summary of Key Results" table has been filled in with the key step, key operation, metric, change, and note; the metric and change are verifiable factual data, or the reason why no numeric value can be given has been stated. | Add the summary table, or add the data and the explanation of the reason |
| GE-08 | Related workflow link | A hyperlink pointing to the related workflow guide has been provided. Link validity is checked by CE-12 of the common checklist; this item only checks whether the link exists. | Add the related workflow link |

### 1.2 WARN (Recommended)

| No. | Check Item | Criterion | Recommended Action |
| :--- | :--- | :--- | :--- |
| GW-01 | Case background description | "Case Background" describes the objective and the usage scenario of this case. | Add the objective and the scenario |
| GW-02 | No duplicated content | Workflow-level standard clauses are not repeated in "Case Background", and general gates such as pre-execution checks, pass conditions, and troubleshooting from the related workflow are not repeated in "Procedure". | Streamline the background and steps, and delete the duplicated standard clauses and gates |
| GW-03 | Step quality | The steps are arranged in execution order; the operations are concrete and executable, with key commands or configurations given as code blocks; the output and record are traceable; if "Next step" is written, it points to a subsequent step and is consistent with the order. | Adjust the step order, or add the commands, artifact paths, and records |
| GW-04 | Prerequisites | "Prerequisites for this case" contains only the conditions that must be satisfied before the case starts, not workflow-level clauses or vague descriptions. | Fix the prerequisites |
| GW-05 | Workflow connection | The case scenario falls within the applicable scope of the related workflow; the deliverables do not conflict with the key deliverables required by the related workflow. | Fix the related workflow, or adjust the deliverables |
| GW-06 | Reproducibility | The case can be reproduced according to the document, and the required inputs, commands, and artifact paths are all locatable. | Add the information needed for reproducibility |
| GW-07 | Consistent conclusions | The conclusions are consistent with the "Summary of Key Results" and the deliverable acceptance data, without exaggerating unverified gains. | Fix the conclusions |
| GW-08 | Transferable lessons | "Lessons Learned" contains transferable conclusions and states the applicable boundaries (data, load, version, product form, and so on), rather than restating the procedure. | Rewrite the lessons learned and add the applicable boundaries |
| GW-09 | Exception handling | Check only when the "Exception Handling" section exists: the content describes the pitfalls, investigation, and rollback of this case, rather than general gate clauses. | Rewrite it as the pitfalls of this case |
| GW-10 | Appendix | Check only when the "Appendix" section exists: materials such as configurations, scripts, issues, PRs, and logs are accessible or reproducible. | Add the materials or links |
| GW-11 | Entry link | If the "Case List" of the related workflow needs to include this case, the entry link has been updated, or the pending inclusion has been registered in the related workflow. | Update the entry link or register the pending inclusion |

## 2. Usage Notes

- **Validation scope**: case reference documents written based on the *[Case Reference Template](03_general_case_template.md)*, stored in `docs/en/best_practices/`.
