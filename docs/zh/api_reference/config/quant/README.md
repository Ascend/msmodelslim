# 量化任务配置说明导航

本目录是 msModelSlim 量化任务配置（YAML）的字段级参考文档，按先选 `apiversion`、再展开 `spec`的方式组织。使用流程与端到端操作步骤见《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》；处理器与保存格式总表见《[配置说明导航](../README.md)》；自动调优见《[自动调优配置说明导航](../tuning/README.md)》。

## 1. 怎么用这套文档

1. **确定协议**：YAML 根节点的 `apiversion` 决定 `spec` 的结构，按任务场景在下方[任务配置](#2-任务配置按-apiversion-选择)表中选择对应页面。各协议可用的 `process` / `save` 子类型并不相同，以对应任务页为准。
2. **查任务字段**：打开对应协议页的参数列表，根块为任务配置（如 `ModelslimV1QuantConfig`），嵌套块为 `spec` 及展开的服务配置。
3. **查处理器 / 保存格式**：`spec.process[]`、`spec.save[]` 由 `type` 分派，字段见《[配置说明导航](../README.md)》的处理器、保存格式表，或直接打开 `../processor/`、`../format/` 下对应页面。
4. **查最佳实践包装**：一键量化 / 调优模板若带 `metadata`，见《[practice_config](practice_config.md)》；其内 `task` 仍对应某一 `apiversion` 任务配置。

> 本页为手写导航；本目录其余页面由 `skills/docs-management/scripts/gen_config_api_docs.py`（或 `gen_quant_config_docs.py`）从源码 Pydantic 注解生成。修改字段说明请改源码 `Field(description=)` 等注解后重新生成，不要直接编辑带 `generated-by` 标记的页面。

## 2. 任务配置（按 `apiversion` 选择）

| 配置 | `apiversion` | 说明 |
|------|--------------|------|
| [modelslim_v1](modelslim_v1.md) | `modelslim_v1` | 大语言模型量化与权重转换的默认协议，含 `runner`/`process`/`save`/`dataset`。 |
| [multimodal_vlm_modelslim_v1](multimodal_vlm_modelslim_v1.md) | `multimodal_vlm_modelslim_v1` | 多模态理解（VLM）模型量化协议。 |
| [multimodal_sd_modelslim_v1](multimodal_sd_modelslim_v1.md) | `multimodal_sd_modelslim_v1` | 多模态生成（DiT/SD）模型量化协议，含 dump 与推理参数。 |
| [modelslim_convert](modelslim_convert.md) | `modelslim_convert` | 纯权重转换协议，用于权重重命名、结构替换与格式转换。 |
| [practice_config](practice_config.md) | `modelslim_v1`（在 `task` 内） | 最佳实践配置：`metadata` + `task`，被自动调优策略引用。 |
