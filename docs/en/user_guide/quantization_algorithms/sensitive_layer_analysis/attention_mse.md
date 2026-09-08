# Attention MSE (mse): Sensitive Layer Analysis Algorithm

<!-- md-trans-meta sourceCommit=3909078a95b1f3d532dd72a3a08143f4d01c1435 translatedAt=2026-08-19T08:16:49.136Z pushedAt=2026-08-19T08:17:52.373Z -->

## Introduction

- **Overview**: `mse` (Mean Squared Error) is used for **attn** scope analysis: forward inference is performed using floating-point weights and quantized weights respectively, the MSE is calculated for the output of the same attention module, and the results are output as **attention module granularity** ranking.

- **Core idea**: Directly measure the output drift of the attention subsystem under quantized weights; a larger value indicates that the attention layer is more sensitive to weight quantization.

## Prerequisites

Install the msModelSlim tool. For details, see the [msModelSlim Tool Installation Guide](../../../install_guide/install_guide.md).

## Principle

1. For the same calibration sample, perform forward propagation using **floating-point weights** and **quantized weights** respectively, and collect tensors at the output of the attention module.

2. Calculate the MSE between the floating-point and quantized outputs of the same layer and the same sample:

    $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float}}^{(i)} - y_{\text{quant}}^{(i)})^2$$

3. **Interpretation**: The larger the MSE, the more sensitive the attention module is to the current quantization configuration.

## Applicable Requirements

- **Recommended scenario**: When weight quantization or sensitivity evaluation is required for the **Attention** structure.

- **Model adaptation (mandatory)**: The model adapter corresponding to `model_type` must implement `AttentionMSEAnalysisInterface` and provide the module class name and output extraction function. Otherwise, an error is reported during the analysis phase.

- **model_type**: The tool currently implements interface adaptation only for the following models. Other `model_type` values will report an error or require you to implement the interface in the adapter yourself.

| model_type       |
| ---------------- |
| DeepSeek-V3      |
| DeepSeek-V3-0324 |
| DeepSeek-R1      |
| DeepSeek-R1-0528 |
| DeepSeek-V3.1    |

## Feature Description

### Usage Instructions

This analysis relies on the tool attaching a hook to the attention submodule and reading its forward output. Because the attention class names and the shape of the `forward` return values differ across models, they cannot be uniformly inferred by the framework. Therefore, `AttentionMSEAnalysisInterface` (which declares the class name to be hooked and how to extract the tensor used for MSE computation from the `forward` return value) must be implemented in the **model adapter** of the target `model_type` before `msmodelslim analyze attn --metrics mse` can be used on that model. The following describes the interface conventions. If the interface is not implemented or its implementation does not match the model structure, an error is reported during the analysis phase.

```python
class AttentionMSEAnalysisInterface(ABC):
    @abstractmethod
    def get_attention_module_cls(self) -> str:
        ...

    @abstractmethod
    def get_attention_output_extractor(self) -> Callable[[Union[tuple, torch.Tensor]], torch.Tensor]:
        ...
```

| Method | Purpose |
|------|------|
| `get_attention_module_cls` | Returns the class name string of the attention module to be hooked. |
| `get_attention_output_extractor` | Extracts the tensor used for MSE computation from the `forward` return value. |

### Command Line Example

```bash
msmodelslim analyze attn \
    --model_type DeepSeek-V3 \
    --model_path ${model_path} \
    --metrics mse \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

### Command-line Parameter Description

| Name | Description |
|------|------|
| `attn` | Attention structure sensitivity analysis |
| `--metrics` | Analysis algorithm. When the value is `mse`, this algorithm is used. |

For complete parameters, see [Parameter Description in the Sensitive Layer Analysis Tool Usage Guide](../../feature_guide/sensitive_layer_analysis/usage.md#34-parameter-description).

## FAQs

### Error indicating that `AttentionMSEAnalysisInterface` is not implemented

**Symptom**: When running `analyze attn --metrics mse`, an error is reported indicating that `AttentionMSEAnalysisInterface` is not implemented.

**Solution**: The adapter for the current `model_type` has not been integrated into this analysis path. Use a model type from the supported list, or implement the hook class name and output extraction logic in the adapter according to the interface.
