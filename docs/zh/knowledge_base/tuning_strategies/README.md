---
toc_depth: 3
---
# 自动调优策略总览

msModelSlim 通过自动化策略搜索最优量化配置。下表总结了目前支持的调优策略及其主要特性。使用方法请参见《[自动调优使用指南](../../user_guide/usage_auto_precision_tuning.md)》，配置协议请参见《[自动调优配置协议说明](../../api_reference/config/auto_precision_tuning.md)》。

| 策略名称 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- |
| **Standing High** | 结合离群值策略，在满足精度条件下基于二分法尽量减少回退层数 | 需精细控制模板与策略，需要提供完整量化配置 | [查看详情](standing_high/standing_high.md) |
| **Standing High With Experience** | 仅需量化类型与结构配置，根据专家经验自动生成量化配置 | 熟悉模型结构，无需提供完整量化配置 | [查看详情](standing_high_with_experience/standing_high_with_experience.md) |
| **Binary Fallback** | 仅二分搜索最小回退前缀，template 为完整 PracticeConfig | 离群值抑制固定在 template，需精细指定回退路径 | [查看详情](binary_fallback/binary_fallback.md) |

## 策略选择建议

- **已有完整量化 YAML**：优先使用 **Standing High** 或 **Binary Fallback**，在固定离群值抑制与量化模板上自动搜索回退层。
- **仅知量化类型与模型结构**：可使用 **Standing High With Experience**，由专家经验自动生成量化配置。
- **敏感层排序辅助回退**：可结合《[线性层敏感层分析使用指南](../../user_guide/usage_sensitive_linear_analysis.md)》、《[层级敏感层分析使用指南](../../user_guide/usage_sensitive_layer_wise_analysis.md)》、《[Attention 敏感层分析使用指南](../../user_guide/usage_sensitive_attn_analysis.md)》与《[量化算法总览](../quantization_algorithms/README.md#敏感层分析算法)》中的 metrics。
