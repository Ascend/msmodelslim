# 量化格式支持矩阵

## 简介

msModelSlim 支持多种量化权重落盘格式。格式决定量化结果的**文件结构、张量命名与元数据组织方式**；量化算法决定**量化过程**（校准、离群值抑制等）。本文档帮助您根据目标推理框架选择合适的导出格式。

> 如需接入新的量化格式，请参见《[量化格式接入指南](../../development_guide/iformat_integration_guide.md)》。

## 格式对比矩阵

| 格式 | YAML 配置 | 目标推理框架 | 支持量化模式（概要） | 分布式导出 | 详细说明 |
|------|-----------|-------------|---------------------|-----------|----------|
| AscendV1 | `- type: "ascendv1_saver"`<br>`part_file_size: 4` | MindIE、vLLM Ascend | W8A8 / W8A16 / W4A8 / W4A4 / MXFP / KV Cache / FA 等 20+ | 支持 | [AscendV1 格式说明](ascendv1.md) |
| MindIE-SD | `- type: "mindie_format_saver"`<br>`part_file_size: 0` | MindIE（多模态生成） | 多模态生成模型专用 | 支持 | [MindIE 保存器配置](../feature_guide/quick_quantization_v1/usage.md#6341-mindie_format_saver) |
| compressed-tensors | `- type: "compressed_tensors"`<br>`part_file_size: 4` | vLLM | W8A8 Static / W8A8 Dynamic | 不支持 | [compressed-tensors 格式说明](compressed_tensors.md) |

## YAML 配置示例

### AscendV1（默认，昇腾推理）

```yaml
spec:
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

### compressed-tensors（vLLM 等）

```yaml
spec:
  save:
    - type: "compressed_tensors"
      part_file_size: 4
```

### MindIE-SD（多模态生成）

```yaml
spec:
  save:
    - type: "mindie_format_saver"
      part_file_size: 0
```

## 格式 vs 量化算法

| 概念 | 说明 | 文档位置 |
|------|------|----------|
| **量化格式** | 量化权重的落盘结构与加载协议 | 本章节 |
| **量化算法** | 校准、离群值抑制、自动调优等计算过程 | 《[算法总览](../quantization_algorithms/README.md)》 |
| **量化模式** | 如 w8a8、w4a8 等比特组合策略 | 《[大模型支持矩阵](../model_support/foundation_model_support_matrix.md)》 |

## 相关文档

- 《[一键量化使用指南](../feature_guide/quick_quantization_v1/usage.md)》 — save 配置详解
- 《[一键量化生成结果](../feature_guide/quick_quantization_v1/quantization_result.md)》 — 输出文件概览（QuaRot、debug 等）
- 《[权重在加速库/MindIE 中的使用](../../case_studies/quantization_weight_use_cases_in_acceleration_and_mindie_torch.md)》
