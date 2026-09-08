# Quantization Format Support Matrix

<!-- md-trans-meta sourceCommit=3b90bb107b98d64c7bdc9bb6de93e6cf3e430cc1 translatedAt=2026-08-20T03:17:20.519Z pushedAt=2026-08-20T03:18:43.387Z -->

## Introduction

msModelSlim supports multiple quantization weight on-disk persistence formats. The format determines the **file structure, tensor naming, and metadata organization** of the quantization result; the quantization algorithm determines the **quantization process** (calibration, outlier suppression, etc.). This document helps you select an appropriate export format based on the target inference framework.

> To integrate a new quantization format, see *[Quantization Format Integration Guide](../../development_guide/iformat_integration_guide.md)*.

## Format Comparison Matrix

| Format | Target Inference Framework | Supported Quantization Mode | Distributed Export | Details |
|------|-------------|---------------------|-----------|----------|
| AscendV1 | MindIE, vLLM Ascend | W8A8 / W8A16 / W4A8 / W4A4 / MXFP / KV Cache / FA, 20+ in total | Supported | *[AscendV1 Format Description](ascendv1.md)* |
| MindIE-SD | MindIE (multimodal generation) | Dedicated to multimodal generation models | Supported | *[MindIE Saver Configuration](../feature_guide/quick_quantization_v1/usage.md#5341-mindie_format_saver)* |
| compressed-tensors | vLLM | W8A8 Static / W8A8 Dynamic | Not supported | *[compressed-tensors Format Description](compressed_tensors.md)* |

## YAML Configuration Example

### AscendV1 (Default, Ascend Inference)

```yaml
spec:
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

### compressed-tensors (vLLM, etc.)

```yaml
spec:
  save:
    - type: "compressed_tensors"
      part_file_size: 4
```

### MindIE-SD (Multimodal Generation)

```yaml
spec:
  save:
    - type: "mindie_format_saver"
      part_file_size: 0
```

## Format vs. Quantization Algorithm

| Concept | Description | Document location |
|------|------|----------|
| **Quantization format** | The on-disk persistence structure and loading protocol of quantized weights | This chapter |
| **Quantization algorithm** | Computational processes such as calibration, outlier suppression, and automatic tuning | *[Algorithm Overview](../quantization_algorithms/README.md)* |
| **Quantization mode** | Bit combination strategies such as w8a8 and w4a8 | *[Foundation Model Support Matrix](../model_support/foundation_model_support_matrix.md)* |

## Related Documents

- *[One-Click Quantization Usage Guide](../feature_guide/quick_quantization_v1/usage.md)* — save configuration details

- *[One-Click Quantization Results](../feature_guide/quick_quantization_v1/quantization_result.md)* — output file overview (QuaRot, debug, etc.)

- *[Using Quantized Weights in Acceleration Libraries/MindIE](../../best_practices/quantization_weight_use_cases_in_acceleration_and_mindie_torch.md)*
