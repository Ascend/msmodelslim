---
name: evaluation
description: |
  量化模型测评：生成 Evaluation YAML、执行评测、以及 AISBench 评测集 herding 压缩（coreset）。
  触发：执行测评、生成评测配置、压缩评测集、量化调优需要子集/全集切换时。
metadata:
  version: 0.1.0
  domain: quant
  framework: msmodelslim
  skill_class: workflow
---

# 测评（evaluation）

| 目录 | 用途 |
|---|---|
| [evaluation-cfg/](evaluation-cfg/SKILL.md) | 生成评测 YAML |
| [evaluate/](evaluate/SKILL.md) | 按 Evaluation YAML 执行评测 |
| [dataset-compression-herding/](dataset-compression-herding/SKILL.md) | 评测集 herding 压缩，供调优切换 config_name |

## 使用方式

- **直接使用**：用户要求「执行测评 / 生成评测配置 / 压缩评测集」时，进入对应子 Skill（`evaluate` / `evaluation-cfg` / `dataset-compression-herding`），由本 Skill 独立完成，不依赖 `tuning`。
- **被 `tuning` 编排**：端到端调优在「量化配置调优」阶段委派 `evaluation/evaluation-cfg` 与 `evaluation/evaluate`，在「压缩数据集确认」阶段委派 `evaluation/dataset-compression-herding`，委派契约见 `tuning/references/quantization_tuning.md`。本 Skill 是**领域入口**，自身不作为 subagent 被委派。
