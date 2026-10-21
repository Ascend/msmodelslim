# {{model_name}}-{{scenario_type}} Case

> **Note**: suggested file naming: {{model_name}}_{{scenario_type}}_case.md, for example qwen3_32b_w8a8_precision_tuning_case.md. After copying, replace every {{placeholder}}; sections marked [OPTIONAL] are kept as needed or deleted in full; delete notes of this kind after the document is filled in.

## 1. Case Background

{{case_goal}}

> **Note**: describe the objective and scenario of this case, and the names of the workflow steps actually covered.

**Related process**: *{{related_workflow_title}}* ({{related_workflow_link}})

> **Note**: when filling in a case, replace this with a hyperlink pointing to the corresponding workflow guide; do not repeat the workflow specification here.

## 2. Environment and Versions

| Item | Version or Configuration |
| --- | --- |
| Product form | {{product_form}} (whether restricted: {{product_form_restricted}}) |
| CANN | {{cann_version_or_link}} |
| PyTorch | {{pytorch_version_or_link}} |
| TorchNPU | {{torch_npu_version_or_link}} |
| vLLM Ascend | {{vllm_ascend_version_or_link}} |
| MindIE | {{mindie_version_or_link}} |
| MindIE SD | {{mindie_sd_version_or_link}} |
| Other dependencies | {{extra_dependencies}} |

> **Note**: for Product form, fill in the actual hardware form (such as Atlas 800I A2 products) and state whether it is restricted (restricted / not restricted; when restricted, state the applicable boundary). Add or remove dependency items as needed; if several items are alternatives to each other (such as multiple inference frameworks), keep only the one actually used in this case. For version requirements, you may first fill in the version number and replace it with a document or release page link once that version is officially released; for the remaining items, give a link to the version currently used.

**Prerequisites for this case**:

- {{prerequisite_fact_1}}
- {{prerequisite_fact_2}}

> **Note**: write only the conditions that must be satisfied before this case starts.

## 3. Inputs and Deliverables

| Type | Name | Source or Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | {{input_name}} | {{input_source}} | {{input_constraint}} | {{input_acceptance_method}} |
| Deliverable | {{deliverable_name}} | {{deliverable_location}} | {{deliverable_constraint}} | {{deliverable_acceptance_method}} |

> **Note**: fill in the actual paths or values of this case; each item must be locatable and verifiable; add or remove rows as needed.
>
> Example (delete after the document is filled in):
>
> - Input / floating-point model / `${MODEL_PATH}` / contains config.json, weights, and tokenizer / can be loaded by the corresponding Transformers version
> - Input / calibration and evaluation data / lab_calib or a business evaluation dataset / matches the task and language / the samples can be preprocessed by the model
> - Deliverable / quantized weight result / `${SAVE_PATH}` (the output directory of the quantization workflow) / compatibility guarantee / the md5 values of the quantized weights or the results of a safetensors comparison
> - Deliverable / evaluation report / `AISBench outputs/{timestamp}/` / comparison between floating-point and quantized results / compared item by item against the acceptance threshold

## 4. Procedure

### Step {{step_number}}: {{step_name}}

**Objective**: {{step_objective}}  
**Inputs**: {{step_input}}  
**Operations**: {{step_action}}  
**Output**: {{step_output}}  
**Record**: {{execution_record}}  
**[OPTIONAL] Next step**: {{next_step}}

> **Note**: copy this section in execution order and delete the steps that are not covered. Operations must be executable (put commands and configurations in code blocks); the output and record must be traceable. "Next step" is optional; delete the whole line if it is not needed. For pre-execution checks, pass conditions, and failure handling, see the related workflow; do not repeat them here.

## 5. Results and Lessons Learned

### 5.1 Summary of Key Results

| Step | Key Operation | Metric | Change | Note |
| --- | --- | --- | --- | --- |
| {{path_step_name}} | {{path_step_action}} | {{path_step_metric}} | {{path_step_delta}} | {{path_step_note}} |

> **Note**: summarize the key steps and metric changes of this case with facts; add or remove rows as needed.

### 5.2 Lessons Learned

1. **{{lesson_title_1}}**: {{lesson_body_1}}
2. **{{lesson_title_2}}**: {{lesson_body_2}}
3. **{{lesson_title_3}}**: {{lesson_body_3}}

> **Note**: write transferable conclusions and state the applicable boundaries (data, load, version, and so on).

## 6. [OPTIONAL] Exception Handling

- {{exception_handling}}

> **Note**: record only the pitfalls and the investigation or rollback actions of this case; for general gates, see the related workflow.

## 7. [OPTIONAL] Appendix

- Configuration or script paths: {{appendix_config_path}}
- Related issues / PRs: {{appendix_issue_pr}}
- Evaluation logs or supplementary data: {{appendix_extra_data}}

> **Note**: add reproducible materials or external links as needed.
