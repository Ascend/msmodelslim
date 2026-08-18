---
toc_depth: 3
---
# Quantization Sensitive Layer Analysis Tool Guide

## Introduction

`analyze` is the quantization sensitive layer analysis interface in msModelSlim. It analyzes the quantization sensitivity of each layer in a model and helps you identify sensitive layers so that you can optimize them in a targeted way.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../getting_started/install_guide.md).

## Function Overview

### Supported Ascend AI Processors

Currently, this feature supports only Atlas A3 training products, Atlas A3 inference products, Atlas A2 training products, Atlas 800I A2 inference products, and A200I A2 Box heterogeneous subracks. The memory must be more than 1.5 times the model size.

### Description

- **Multi-dimensional analysis**: Supports the `std`, `quantile`, and `kurtosis` metrics for linear layers, the `attention_mse` metric for attention structures, and the `mse_model_wise` metric for model-level output. It can accurately evaluate layer sensitivity from multiple dimensions, including data distribution, robustness, kurtosis characteristics, attention-output differences, and model-output differences.
- **Flexible configuration**: Supports custom calibration datasets in the JSON or JSONL format, layer-name matching, and a rich set of parameters to meet quantization needs in different scenarios.
- **Intelligent output**: Supports printing a Top K list of sensitive layers. The actual number printed may be greater than or equal to the target number, for example when QKV is printed together.

### Precautions

- The Transformers version depends on the model and is unrelated to quantization features.
- The actual number of rollback layers depends on the inference engine implementation, so it may differ from the `topk` setting.
- The default `topk` value is 15. It serves only as a reference rollback value. If the printed layers involve QKV, QKV is printed together.
- For security reasons, `trust_remote_code` defaults to `False`.
- Sensitive layer analysis currently supports only LLMs.

### Command Syntax

```bash
msmodelslim analyze [parameters]
```

### Parameters

#### Mandatory parameters

| Parameter| Type| Default Value| Description| Example Value|
|------|------|--------|------|--------|
| `--model_type` | `str` | - | Model type. Used to specify the model architecture to analyze. See [Detailed Parameter Description](#detailed-parameter-description).| `Qwen2.5-7B-Instruct` |
| `--model_path` | `str` | - | Path to the original model. An absolute path is recommended.| `/path/Qwen2.5-7B-Instruct` |

#### Optional Parameters

| Parameter| Type| Default Value| Description| Example Value|
|------|------|--------|------|--------|
| `--device` | `str` | `npu` | Target device used to run the analysis. Options: `npu`, `cpu`.| `npu` |
| `--patterns` | `List[str]` | `["*"]` | Layer-name patterns to analyze. Wildcards are supported. You can specify multiple patterns separated by spaces. If no value is passed, the default value is used.| `"*linear*"` `"*attention.*"` `"*mlp.*"` |
| `--metrics` | `str` | `"kurtosis"` | Metric used for analysis. Options: `"std"`, `"quantile"`, `"kurtosis"`, `"attention_mse"`, `"mse_model_wise"`.| `"kurtosis"` |
| `--calibration_dataset` | `str` | `"boolq.jsonl"` | Path to the calibration dataset file. JSON and JSONL formats are supported, and the file must end with `.json` or `.jsonl`. The path can be an absolute or a relative path.|`/path/data.jsonl`|
| `--top_k` | `int` | `15` | Number of sensitive layers to output as Top K. This must be an integer greater than 0. The recommended range is 10 to 20.|  `15` |
| `--trust_remote_code` | `bool` | `False` | Whether to trust remote code. You must ensure security yourself. Options: `True`, `False`. If model loading depends on files outside the Transformers library, set `--trust_remote_code` to `True`, for example for DeepSeek-V3 series models.| `False` |
| `-h, --help` | - | - | CLI help information.| - |

### Detailed Parameter Description

#### Supported Model Types

Different analysis metrics support different `model_type` values. Check the corresponding list for the metric you selected.

**Supported `model_type` values for linear-layer metrics (`std` / `quantile` / `kurtosis`)**

The `model_type` values currently supported by the linear-layer metrics are the same as those for ModelslimV1 quantization. For details, see the `[ModelAdapter]` section in [config.ini](https://gitcode.com/Ascend/msmodelslim/blob/master/config/config.ini).

**Supported `model_type` values for the attention-structure metric (`attention_mse`)**

| model_type       |
| ---------------- |
| DeepSeek-V3      |
| DeepSeek-V3-0324 |
| DeepSeek-R1      |
| DeepSeek-R1-0528 |
| DeepSeek-V3.1    |

- **Unsupported `model_type`**: If the input `model_type` is not in the list for the selected metric, the system prints a warning and may fall back to the default model adapter. When you use a specific metric such as `attention_mse`, it may also raise an error if the model adapter for that `model_type` does not implement the required interface.
- **Recommendation**: Use the `model_type` that matches `--metrics` in the list above to ensure correct analysis behavior and compatibility.

#### Path and File Requirements

- `model_path`: The path must exist. An absolute path is recommended. It must contain valid model files.
- **calib_dataset**:
  - JSON format is supported. The JSON file contains a string list. See `anti_prompt.json` in the `lab_calib` directory.
  - JSONL format is supported. Each line contains a JSON object. See the JSONL example files in the `lab_calib` directory, such as `boolq.jsonl`.
  - Relative paths are resolved in the `lab_calib` directory for the configured `calib_dataset`.

#### Algorithm Selection Notes

- `std`: Standard deviation algorithm. It is simple and performs well, so it suits common scenarios.
- `quantile`: Quantile algorithm. It is less sensitive to outliers and suits scenarios that require higher accuracy.
- `kurtosis`: Kurtosis algorithm. It can identify distribution-shape characteristics and suits scenarios that need finer control.
- `attention_mse`: Attention MSE algorithm. It uses the difference between attention outputs before and after quantization and suits scenarios where you need to evaluate attention-module quantization.
- `mse_model_wise`: Model-level MSE algorithm. It uses the difference between the **last decoder output** before and after quantization to measure quantization sensitivity from the perspective of the model as a whole.

### Analysis Algorithm Description

#### std (Standard Deviation)

##### Principle

- Calculate the maximum, minimum, and standard deviation of the activation values.
- Use this formula: `score = max(|max_value|, |min_value|) / std`.
- This reflects the variation and range of the data.

##### Applicable Scenarios

- Recommended for: common quantization scenarios
- **Advantages**:
  - Simple and fast.
  - Sensitive to changes in the data distribution.
  - Intuitively reflects data volatility.
- **Characteristics**:
  - The larger the standard deviation, the smaller the score, which means the layer is less sensitive to quantization.
  - The larger the data range, the larger the score, which means the layer is more sensitive to quantization.

#### quantile (Quantile-based)

##### Principle

- Calculate the first and third quartiles of the activation values.
- Use this formula: `score = 2 * max_abs / 254 / (Q3 - Q1)`.
- Evaluate the distribution with the interquartile range, or IQR.

##### Applicable Scenarios

- **Recommended for**: scenarios where you need to account for the tail of the distribution
- **Advantages**:
  - Not sensitive to outliers.
  - Reflects the robustness of the data distribution.
  - Suitable for scenarios that require higher quantization accuracy.
- **Characteristics**:
  - The larger the IQR, the smaller the score, which means the distribution is more spread out and less sensitive to quantization.
  - The larger the absolute data value, the larger the score, which means the layer is more sensitive to quantization.

#### kurtosis (Kurtosis-based)

##### Principle

- Calculate the kurtosis of the activation values.
- Kurtosis formula: `kurtosis = E[(X-μ)**4] / σ**4 - 3`.
- This reflects the peakedness of the data distribution.

##### Applicable Scenarios

- **Recommended for**: scenarios where you need to accurately identify extreme values in the activation distribution
- **Advantages**:
  - Can identify how sharp the distribution peak is.
  - Sensitive to extreme values.
  - Suitable for quantization scenarios that require fine control.
- **Characteristics**:
  - The larger the kurtosis, the larger the score, which means the distribution is more concentrated and more sensitive to quantization.
  - The closer the kurtosis is to 0, the smaller the score, which means the distribution is closer to normal and less sensitive.

#### attention_mse (Attention MSE)

##### Principle

- Collect the attention-module outputs before quantization, when the weights are floating-point, and after quantization, when the weights are quantized.
- Calculate the MSE between the floating-point and quantized outputs of the same layer.
- Formula: $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float}}^{(i)} - y_{\text{quant}}^{(i)})^2$
- This reflects the sensitivity of the attention module to quantization.

##### Applicable Scenarios

- **Recommended for**: scenarios where you need to quantize attention modules
- **Prerequisites**:
  - The model adapter for the selected `model_type` must inherit from and implement `AttentionMSEAnalysisInterface`.

##### Adaptation Process

```python
class AttentionMSEAnalysisInterface(ABC):
    @abstractmethod
    def get_attention_module_cls(self) -> str:
        ...

    @abstractmethod
    def get_attention_output_extractor(self) -> Callable[[Union[tuple, torch.Tensor]], torch.Tensor]:
        ...
```

- `get_attention_module_cls()`: Returns the module class name, as a string, used to match attention layers. During analysis, the tool registers hooks on modules of this type and collects outputs and computes MSE only for these modules.
- `get_attention_output_extractor()`: Returns an extractor function. Its input is the complete output of the attention module `forward` method, usually a `tuple` or a single `Tensor`. It extracts and returns the tensor used for sensitivity analysis from that output.

##### Characteristics

- Larger values mean the output deviation of the attention layer after quantization is greater, so the layer is more sensitive.

#### mse_model_wise (Model-wise MSE)

##### Principle

- Run a forward pass twice on the same batch of calibration samples, once **before quantization, with floating-point weights**, and once **after quantization**. Collect the model final output, which is the output of the last decoder layer.
- Calculate the MSE from the difference in the model final output. A larger score means the layer has a greater impact on the final output after quantization and is therefore more sensitive.

##### Applicable Scenarios

- **Recommended for**: scenarios where you want to evaluate the quantization impact of each layer from the perspective of the model final output. This does not require model adaptation. The results are sorted by **decoder layer** or block, and it is suitable for **whole-layer rollback** or **whole-block rollback** by structure, for example an entire attention module or an entire MLP.
- **Usage Notes**:
  - The `pattern` parameter directly affects which structures are quantized when you compare the errors before and after quantization for a single Decoder layer. Different pattern selections produce different ranking results.
  - Different models have different forward propagation processes. This method works only when the output of the current Decoder layer can be used directly as the input of the next layer. If the inputs and outputs between layers cannot be aligned, the tool raises an error and reports that the model is unsupported.

### Examples

#### Basic Usage

```bash
# Specify the analysis algorithm and dataset.
# model_path specifies the model path.
# calib_dataset specifies the calibration dataset path.
msmodelslim analyze \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics quantile \
    --calibration_dataset ${calib_dataset} \
    --top_k 20 \
    --device cpu
```

#### Custom Layer Patterns

```bash
# Analyze only attention layers and MLP layers.
msmodelslim analyze \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --patterns "*attention*" "*mlp*" \
    --metrics std
```

#### Complete Configuration Example

```bash
msmodelslim analyze \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --device npu \
    --patterns "*.down_proj*" "*.o_proj*"\
    --metrics kurtosis \
    --calibration_dataset ${calib_dataset} \
    --top_k 15 \
    --trust_remote_code False
```

### Output Description

The tool supports clean YAML-format output, which makes it easy to paste directly into a YAML file.

#### Console Output

Using the complete configuration above as an example, the console output is as follows:

```text
...
msmodelslim.core.analysis_service.pipeline_analysis.service - INFO - ==========ANALYSIS: Starting Layer Analysis==========
msmodelslim.core.analysis_service.pipeline_analysis.service - INFO - Analysis metrics: kurtosis
msmodelslim.core.analysis_service.pipeline_analysis.service - INFO - Layer patterns: ['*.down_proj*', '*.o_proj*']
msmodelslim.core.runner.layer_wise_runner - INFO - Start to handle dataset
msmodelslim.core.runner.layer_wise_runner - INFO - Handle dataset success
msmodelslim.core.runner.layer_wise_runner - INFO - Start to init model
msmodelslim.core.runner.layer_wise_runner - INFO - Init model success
msmodelslim.core.runner.layer_wise_runner - INFO - KV cache requirement: False
msmodelslim.core.runner.layer_wise_runner - INFO - Scheduler 3 unit: [LoadProcessor(device=npu:0, non_blocking=False), UnaryAnalysisProcessor, LoadProcessor(device=meta, non_blocking=False)]
msmodelslim.core.runner.layer_wise_runner - INFO - Run processor LoadProcessor(device=npu:0, non_blocking=False) for "model.layers.0"
msmodelslim.core.runner.layer_wise_runner - INFO - Run processor UnaryAnalysisProcessor for "model.layers.0"
msmodelslim.core.runner.layer_wise_runner - INFO - Run processor LoadProcessor(device=meta, non_blocking=False) for "model.layers.0"
...
msmodelslim.app.analysis.application - INFO - === Layer Analysis Results (kurtosis method) ===
msmodelslim.app.analysis.application - INFO - Patterns analyzed: ['*.down_proj*', '*.o_proj*']
msmodelslim.app.analysis.application - INFO - Total layers analyzed: 128
msmodelslim.app.analysis.application - INFO - Layer Sensitivity Scores (higher score = more sensitive to quantization):
...
msmodelslim.app.analysis.application - INFO - === YAML Format for quantization ===
...
msmodelslim.app.analysis.application - INFO - === End of YAML Format ===
msmodelslim.app.analysis.application - INFO - ===========ANALYSIS COMPLETE===========
```

## FAQ

### Symptom: The Calibration Dataset File Format Is Wrong or Cannot Be Read

**Solution**:

1. Ensure that the file format is a supported JSON or JSONL format.
2. Ensure that each record contains the required fields.
3. Check whether the file path is correct.
4. Ensure that the calibration file has read permission.

### Symptom: What Happens If an Unsupported `model_type` Is Used?

**Solution**:
When the input `model_type` is not in the supported list:

- The system prints a warning that it is using the default model.
- It automatically uses the default model for processing.
- The analysis result may not be optimal.
- **Recommendation**: Prefer the standard `model_type` values in the [Supported Model Types](#supported-model-types) to get the best compatibility and analysis accuracy.

### Symptom: How Do You Apply the Analysis Result to the Quantization Configuration?

**Solution**:

1. Add highly sensitive layers to the quantization blacklist so they are not quantized.
2. Use lower quantization precision for moderately sensitive layers, for example 8-bit instead of 4-bit.
3. Adjust the quantization strategy based on the analysis result to achieve the best balance between accuracy and performance.
