# {{model_name}}-{{scenario_type}}案例

> **注释：** 文件命名建议：{{model_name}}_{{scenario_type}}_case.md，例如 qwen3_32b_w8a8_precision_tuning_case.md。复制后替换全部 {{placeholder}}；带 [OPTIONAL] 的章节按需保留或整节删除；填写完成后删除本类说明注释。

## 1. 案例背景

{{case_goal}}

> **注释：** 说明本次目标、场景，以及实际覆盖的流程步骤名称。

**关联流程**：《[{{related_workflow_title}}]({{related_workflow_link}})》

> **注释：** 链接至对应流程指南，不在此重复流程规范。

## 2. 环境与版本

| 项 | 版本或配置 |
| --- | --- |
| 产品形态 | {{product_form}}（是否限定：{{product_form_restricted}}） |
| CANN | {{cann_version_or_link}} |
| PyTorch | {{pytorch_version_or_link}} |
| TorchNPU | {{torch_npu_version_or_link}} |
| vLLM Ascend | {{vllm_ascend_version_or_link}} |
| MindIE | {{mindie_version_or_link}} |
| MindIE-SD | {{mindie_sd_version_or_link}} |
| 其他依赖 | {{extra_dependencies}} |

> **注释：** 产品形态填写实际硬件形态（如Atlas 800I A2），并说明是否限定（限定 / 不限定；限定时写明适用边界）。依赖项按需增删，多选一时仅保留实际使用项。版本需求可先填写版本号，待该版本正式上线后替换为文档或发布页链接；其余项给出当前使用的版本链接。

**本次前置事实**：

- {{prerequisite_fact_1}}
- {{prerequisite_fact_2}}

> **注释：** 仅写本案例开始前已满足的事实。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | {{input_name}} | {{input_source}} | {{input_constraint}} | {{input_acceptance_method}} |
| 交付件 | {{deliverable_name}} | {{deliverable_location}} | {{deliverable_constraint}} | {{deliverable_acceptance_method}} |

> **注释：** 填写本案例实际路径或取值；每项须可定位、可验证；按需增删行。
>
> 示例（填写完成后删除）：
>
> - 输入 / 浮点模型 / `${MODEL_PATH}` / 含 config.json、权重、tokenizer / 可被对应 Transformers 版本加载
> - 输入 / 校准与测评数据 / lab_calib 或业务测评集 / 与任务、语言匹配 / 样本可被模型预处理
> - 交付件 / 量化权重结果 / `${SAVE_PATH}`（量化流程输出目录） / 兼容性保证 / 量化权重的 md5 值或 safetensors 比对结果
> - 交付件 / 测评报告 / `AISBench outputs/{timestamp}/` / 浮点与量化对比 / 与验收阈值逐项比对

## 4. 操作步骤

### 步骤 {{step_number}}：{{step_name}}

**目标**：{{step_objective}}  
**输入**：{{step_input}}  
**操作**：{{step_action}}  
**输出**：{{step_output}}  
**记录**：{{execution_record}}  
**下一步**：{{next_step}}

> **注释：** 按执行顺序复制本节，未覆盖的步骤删除。操作须可执行（命令与配置放入代码块）；输出与记录须可追溯。执行前检查、通过条件、失败处置见关联流程，此处不重复。

## 5. 结果与经验

### 5.1 关键结果汇总

| 步骤 | 关键操作 | 指标 | 变化 | 备注 |
| --- | --- | --- | --- | --- |
| {{path_step_name}} | {{path_step_action}} | {{path_step_metric}} | {{path_step_delta}} | {{path_step_note}} |

> **注释：** 用事实汇总本次关键步骤与指标变化；按需增删行。

### 5.2 经验总结

1. **{{lesson_title_1}}**：{{lesson_body_1}}
2. **{{lesson_title_2}}**：{{lesson_body_2}}
3. **{{lesson_title_3}}**：{{lesson_body_3}}

> **注释：** 写可迁移结论，并注明适用边界（数据、负载、版本等）。

## 6. [OPTIONAL] 异常处理

- {{exception_handling}}

> **注释：** 仅记录本次踩坑与排查/回退动作；通用门禁见关联流程。

## 7. [OPTIONAL] 附录

- 配置或脚本路径：{{appendix_config_path}}
- 相关 Issue / PR：{{appendix_issue_pr}}
- 评测日志或补充数据：{{appendix_extra_data}}

> **注释：** 按需补充可复现材料或外链。
