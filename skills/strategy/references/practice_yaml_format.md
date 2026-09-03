# 量化配置格式（Practice YAML）

## 权威 schema 参考（docs）

Practice YAML 的完整字段 schema、处理器类型、QConfig 取值与保存配置**以 docs 为准**，本文件只保留调优闭环特有的约束：

| 内容 | docs 位置 |
|------|-----------|
| 配置类与字段总览（`PracticeConfig` / `metadata`） | 《[PracticeConfig 配置说明](../../../docs/zh/api_reference/config/task/practice_config.md)》 |
| 任务规格 `spec`（按 `apiversion`） | 《[modelslim_v1](../../../docs/zh/api_reference/config/task/modelslim_v1.md)》、《[multimodal_vlm_modelslim_v1](../../../docs/zh/api_reference/config/task/multimodal_vlm_modelslim_v1.md)》、《[multimodal_sd_modelslim_v1](../../../docs/zh/api_reference/config/task/multimodal_sd_modelslim_v1.md)》 |
| 处理器类型（`linear_quant` / `flex_smooth_quant` / `iter_smooth` / `flex_awq_ssz` / `quarot` 等） | 《[config/processor/ 目录](../../../docs/zh/api_reference/config/processor/)》 |
| QConfig 字段取值（`dtype` / `scope` / `symmetric` / `method` / `ext`） | 《[linear_quant](../../../docs/zh/api_reference/config/processor/linear_quant.md)》 等 |
| 保存配置（`ascendv1_saver`） | 《[ascendv1_saver](../../../docs/zh/api_reference/config/format/ascendv1_saver.md)》 |
| 完整配置示例与协议详解 | 《[权重量化使用指南](../../../docs/zh/user_guide/usage_weight_quantization.md)》、《[modelslim_v1](../../../docs/zh/api_reference/config/task/modelslim_v1.md)》、《[multimodal_vlm_modelslim_v1](../../../docs/zh/api_reference/config/task/multimodal_vlm_modelslim_v1.md)》、《[multimodal_sd_modelslim_v1](../../../docs/zh/api_reference/config/task/multimodal_sd_modelslim_v1.md)》 |

## 调优闭环特有约束

以下约束为调优编排（`strategy/practice-cfg` + `tuning`）使用 Practice YAML 时的附加规则，docs 未覆盖：

### 静态排除与调优排除

1. 优先选择与当前 `model_type` 和量化方案匹配的已验证 Practice，并原样继承其中的静态 `exclude`。
2. 静态排除项记录为 `protected_exclude`，自动调优过程中不得删除。
3. 每轮最终写入 YAML 的 `exclude` 为 `protected_exclude ∪ tuning_exclude`，并保持稳定顺序、去除重复项。
4. 若没有匹配的已验证 VLM Practice，应根据实际模型结构生成保守基线，确保视觉编码器和多模态投影层不在目标量化处理器的作用范围内；无法确认量化范围时立即返回，不得仅根据通用模块名称推断。

### VLM 专属约束

- `spec.default_text`：默认值为 `"Describe this image in detail."`。
- `spec.dataset`：VLM 校准数据集，默认使用 `calibImages`。
- `spec.process[].include`：继承基准 Practice 中对应处理器的配置；自动调优过程中不得扩大其作用范围。
- `spec.process[type=linear_quant].exclude`：由 `protected_exclude` 与 `tuning_exclude` 合并生成。`protected_exclude` 包含基准 Practice 的静态 `exclude`，以及为保护视觉编码器、多模态投影层等非调优模块而增加的固定排除项；`tuning_exclude` 由敏感层搜索增减。

### 常见错误（编排侧）

- `metadata.label` 写成字符串而非 dict
- `type` 与字段不匹配（如 `flex_awq_ssz` 缺少 `qconfig`）
- `dataset` 指向不存在的路径，或使用了当前安装环境中不存在的短名称
- `save` 字段的 `type` 不为 `"ascendv1_saver"`
