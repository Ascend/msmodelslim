---
name: strategy
description: |
  量化调优策略与 Practice YAML 生成。支持 standing_high 与 standing_high_with_experience；
  含 Practice YAML 生成与校验、策略决策。敏感层分析为独立一级 Skill（sensitive-layer-analysis），
  practice-cfg 在首轮按该 Skill 执行。当用户要生成/修改 practice、选择调优策略或专家回退经验时使用。
metadata:
  version: 0.1.1
  domain: quant
  framework: msmodelslim
  skill_class: workflow
---

# 调优策略（strategy）

| 目录 | 用途 |
|---|---|
| [practice-cfg/](practice-cfg/SKILL.md) | 公共：基准 Practice、Practice YAML 生成与校验 |
| [standing_high/](standing_high/strategy_standing_high.md) | standing_high 策略 |
| [standing_high_with_experience/](standing_high_with_experience/strategy_standing_high_with_experience.md) | 含专家经验；见 [expert-rules/](standing_high_with_experience/expert-rules/SKILL.md) |

敏感层分析为独立一级 Skill：[sensitive-layer-analysis](../sensitive-layer-analysis/SKILL.md)（闭环中由 `practice-cfg` 在生成基准 Practice 后**再委派**执行一次，结果供各轮复用；也支持独立触发）。

策略入参仅上述二者。策略原理见《[自动调优策略总览](../../docs/zh/knowledge_base/tuning_strategies/README.md)》。

## 使用方式

- **直接使用**：用户要求「生成 / 修改 Practice、选择调优策略、查询专家回退经验」时，进入对应子 Skill（`practice-cfg` / `standing_high*` / `expert-rules`），由本 Skill 独立完成，不依赖 `tuning`。
- **被 `tuning` 编排**：端到端调优在「量化配置调优」阶段委派 `strategy/practice-cfg`（生成 Practice）与 `strategy/standing_high_with_experience/expert-rules`（结构化回退意见），委派契约见 `tuning/references/quantization_tuning.md`。本 Skill 是**领域入口**，自身不作为 subagent 被委派。
