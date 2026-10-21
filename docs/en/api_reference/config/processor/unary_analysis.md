<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisProcessorConfig -->
# unary_analysis Configuration Reference

## 1. Configuration Overview

Unary (non-quantized) sensitivity analysis processor configuration.

| Item | Content |
|------|------|
| Config Class | `UnaryAnalysisProcessorConfig` |
| Source | [processor.py](../../../../../msmodelslim/processor/analysis/unary_operator/processor.py) |

## 2. Parameter List

<h3 id="2-1-unary-analysis">2.1 UnaryAnalysisProcessorConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `unary_analysis` | `unary_analysis` | Processor type, fixed to `unary_analysis`. | None |
| `metrics` | `string` | Optional | `kurtosis` | — | Analysis metric: `quantile` (quantile), `std` (standard deviation), `kurtosis` (kurtosis), `ra_compress` (RA Compress long-sequence compression head screening) | None |
| `patterns` | `list[string]` | Optional | `['*']` | — | List of layer name patterns to analyze; the default `*` matches all | None |
| `metric_params` | `object` | Optional | `{}` | — | Metric-specific hyperparameters, parsed only by the analysis method specified in `metrics`. Currently supported only for `metrics=ra_compress`: `induction_head_ratio` (default 0.14) and `echo_head_ratio` (default 0.01), both with a value range of [0, 1]. | None |

**Configuration Constraints**

- Validates metric-specific hyperparameters; illegal use raises an error directly.

## 3. Complete Configuration Reference

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: unary_analysis
    metrics: kurtosis
    patterns:
    - '*'
    metric_params: {}
```

## 4. Notes

- `metric_params` takes effect only for `metrics=ra_compress`; passing it with other metrics raises an error directly. This field is set only in the configuration, and the CLI does not expose a corresponding parameter; `msmodelslim analyze attn_head` runs with the defaults (`induction_head_ratio=0.14`, `echo_head_ratio=0.01`).
- A larger ratio selects (keeps) more KV heads, which means lower KV Cache compression strength, smaller accuracy risk, and smaller memory savings; a smaller ratio does the opposite. It is recommended to evaluate empirically based on the number of selected heads and the downstream compression ratio.
- The command-line workflow reads the template `msmodelslim/core/analysis_service/pipeline_analysis/pipeline_template/ra_compress.yaml` installed with the package; after modifying this YAML, you must re-run `bash install.sh` in the repository root for the change to take effect.

`ra_compress` overrides the default ratios:

```yaml
process:
- type: unary_analysis
  metrics: ra_compress
  metric_params:
    induction_head_ratio: 0.14
    echo_head_ratio: 0.01
```
