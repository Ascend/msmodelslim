# Auto Tuning Strategies Overview

msModelSlim searches for the optimal quantization configuration through automated strategies. The following table summarizes the currently supported tuning strategies and their main characteristics. For usage, see the [Auto Tuning Usage Guide](../../user_guide/usage_auto_precision_tuning.md); for the configuration protocol, see the [Auto Tuning Configuration Reference Navigation](../../api_reference/config/tuning/README.md).

| Strategy Name | Core Idea | Applicable Scenarios | Details |
| :--- | :--- | :--- | :--- |
| **Standing High** | Combines an outlier strategy and, while the accuracy requirement is met, minimizes the number of fallback layers based on binary search | Requires fine-grained control over the template and the strategy, and requires a complete quantization configuration | [View Details](standing_high/standing_high.md) |
| **Standing High With Experience** | Requires only the quantization type and the structure configuration, and automatically generates the quantization configuration from expert experience | Familiar with the model structure; a complete quantization configuration is not required | [View Details](standing_high_with_experience/standing_high_with_experience.md) |
| **Binary Fallback** | Binary-searches only the minimal fallback prefix; the template is a complete PracticeConfig | The outlier suppression is fixed in the template, so the fallback path must be specified precisely | [View Details](binary_fallback/binary_fallback.md) |

## Strategy Selection Recommendations

- **A complete quantization YAML already exists**: prefer **Standing High** or **Binary Fallback**, and let the tool automatically search for fallback layers on a fixed outlier suppression and quantization template.
- **Only the quantization type and the model structure are known**: you can use **Standing High With Experience**, which automatically generates the quantization configuration from expert experience.
- **Sensitive layer ranking to assist fallback**: you can use the [Sensitive Linear Layer Analysis Usage Guide](../../user_guide/usage_sensitive_linear_analysis.md), the [Layer-wise Sensitive Layer Analysis Usage Guide](../../user_guide/usage_sensitive_layer_analysis.md), and the [Attention Sensitive Layer Analysis Usage Guide](../../user_guide/usage_sensitive_attn_analysis.md), together with the metrics in [Quantization Algorithm Overview - Sensitive Layer Analysis Algorithms](../quantization_algorithms/README.md#3-sensitive-layer-analysis-algorithms).
