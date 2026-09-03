# standing_high 调优策略

基础摸高策略，通过二分搜索找到最小回退，再逐步减少回退提升量化覆盖率。

最大迭代轮次由入参 `max_iterations` 指定。

> 算法原理与 YAML 字段详解以 docs 为准：《[Standing High 调优算法](../../../docs/zh/knowledge_base/tuning_strategies/standing_high/standing_high.md)》、《[strategy_standing_high 配置](../../../docs/zh/api_reference/config/tuning/strategy_standing_high.md)》。本文只保留与编排层（`tuning` / `quantization_tuning.md`）的衔接约定。

---

## 算法流程

```text
0. 零回退测试（下界 = 0 层回退）
   ├─ 生成无回退层的 Practice → 量化 → 评测
   └─ 若精度达标 → 该配置即全局最优（量化覆盖率最大），直接输出，结束
       （这是唯一「达标即退出」的情形；其余轮次达标只标记上界，见 quantization_tuning.md）

1. 二分搜索最小回退层数
   ├─ 下界 = 0（未达标），上界 = 全部敏感层回退
   ├─ 每次取上下界中位数层回退 → 量化 → 评测
   ├─ 精度达标 → 标记上界 = 本轮，并记录该配置
   └─ 精度不达标 → 下界 = 本轮
   → 上界与下界不可再二分（差距 ≤ 配对粒度）时收敛，输出上界配置
```

> 退出语义与编排层一致：**「某轮达标」不是退出条件**，达标只标记上界；仅在「0层回退（下界）即达标」时直接输出（此时已无可量化更多层的空间）。二分收敛后，若上界配置从未实际量化评测过（上界 = 全部敏感层回退，见 quantization_tuning.md 风险说明），按编排层约定决定是否抽查量化一次。

---

## 配置边界

- 本策略不依赖模型模态；所有受支持的模型共用上述二分搜索和摸高算法，具体配置结构由基准 Practice 决定。
- 每轮 Practice 必须继承基准 Practice 的 `apiversion`、`include`、静态 `exclude` 及其他 schema 专属字段。策略只能调整 `tuning_exclude`，最终写入的 `exclude` 为静态排除项与 `tuning_exclude` 的并集。
- 二分搜索须遵守 gate_proj/up_proj 配对完整性约束（见 quantization_tuning.md「二分搜索约束」）。
- 具体 YAML 字段和量化边界见 [量化配置格式](../references/practice_yaml_format.md)，敏感层排序规则见 [敏感层分析](../../sensitive-layer-analysis/SKILL.md)。
