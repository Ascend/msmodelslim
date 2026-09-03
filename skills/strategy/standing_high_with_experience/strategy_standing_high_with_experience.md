# standing_high_with_experience 调优策略

在 standing_high 基础上引入**专家经验**，实现结构感知的量化配置。根据模型结构类型自动匹配预定义的 QConfig 模板和离群值抑制策略。

最大迭代轮次由入参 `max_iterations` 指定。

> 算法原理与 YAML 字段详解以 docs 为准：《[Standing High With Experience 调优算法](../../../docs/zh/knowledge_base/tuning_strategies/standing_high_with_experience/standing_high_with_experience.md)》、《[strategy_standing_high_with_experience 配置](../../../docs/zh/api_reference/config/tuning/strategy_standing_high_with_experience.md)》。QConfig 模板、离群值抑制策略与结构-量化映射的**权威配置源**为 `msmodelslim/core/tune_strategy/common/config_builder/expert_experience/expert_experience.yaml`，本文不复制其内容。

---

## 工作流程

```text
1. 用户指定 quant_type（w8a8/w4a8）+ structure_configs
2. 读取专家经验配置（expert_experience.yaml）
3. 根据 quant_type 选择离群值抑制策略和结构映射
4. 根据 structure_configs 为不同结构类型分配 QConfig
5. 构建完整的调优配置
6. 根据模型能力过滤不支持的离群值抑制策略
7. 委托 standing_high 策略执行实际调优
```

---

## 支持的量化类型

- `w8a8`：8-bit 权重 + 8-bit 激活
- `w4a8`：4-bit 权重 + 8-bit 激活

（`expert_experience.yaml` 的 `supported_quant_types` 为权威范围，以此为准）

---

## 调优搜索空间

| 配置项 | 说明 |
|--------|------|
| **离群值抑制策略候选** | 由专家经验自动生成 |
| **允许的数据集列表** | 由专家经验自动生成 |
| **最大回退层数** | 由专家经验自动生成 |

控制摸高算法的搜索边界。

---

## 编排衔接

- 二分前由主 Agent 委派 `strategy/standing_high_with_experience/expert-rules` 取得结构化回退意见（`experience_hints`），作为 practice-generator 生成 Practice 的初值；委派契约见 `tuning/references/quantization_tuning.md`「结构化回退经验」。
- 每轮 Practice 仍须继承基准 Practice 的 `apiversion`、`include`、静态 `exclude` 等 schema 专属字段；`exclude` 合并规则见 `strategy/references/practice_yaml_format.md`。
