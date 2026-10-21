# Layer-Wise MSE Sensitive Layer Analysis Algorithm Terminology Entry

> **Entry Category**: [Sensitive Layer Analysis Algorithms](../README.md#3-sensitive-layer-analysis-algorithms)<br>
> **English Name**: mse_layer_wise<br>
> **Application Domain**: Quantization sensitive layer analysis, model quantization accuracy tuning<br>

---

## 1. Overview

Layer-wise MSE (`mse_layer_wise`) is a quantization sensitivity analysis algorithm at the granularity of a Decoder block. It compares the outputs of designated substructures inside the block before and after quantization and aggregates their local mean squared errors into a whole-block sensitivity score, thereby screening candidates that need whole-layer or whole-block protection. Its core characteristic is a local within-block error perspective, and it complements [MSE Model Wise](../mse_model_wise/term_mse_model_wise.md), which observes the model's final output.

---

## 2. Term Description

Statistical metrics such as [Std](../std/term_std.md) cannot directly reflect the quantization reconstruction error. Layer-wise MSE observes that comparing the output difference (MSE) between the floating-point and quantized forward passes within each Decoder block directly measures the quantization reconstruction error aggregated inside the block, thereby supporting whole-layer or whole-block (such as an entire attention/MLP segment) fallback decisions.

### 2.1 Core Idea

The core idea of Layer-wise MSE is to "measure the sensitivity of a Decoder block by its local quantization reconstruction error": under the same calibration input, compare the floating-point output and the quantized output for the pre-defined substructures to be quantized in the block, then aggregate these local MSEs into a block-level score. A larger score means the block has greater difficulty preserving its local numerical behavior under the current quantization scheme.

The advantage of local MSE is that it maps the analysis object to the error source fairly clearly: a high score for a layer means that, under the current quantization configuration and inputs, the layer itself has difficulty preserving its output.

### 2.2 Working Mechanism

Layer-wise MSE measures the sensitivity of a decoder block (or a defined layer-level unit) to quantization through a local counterfactual experiment: keep the input calibration samples fixed, replace the designated modules within the layer with quantized versions while keeping the corresponding floating-point outputs as a reference, then compute the local output difference. The errors from multiple submodules can be aggregated into a score for that layer, producing a sensitivity ranking from high to low.

This method only observes the immediate error that the quantization perturbation causes at the exit of the current layer, so it localizes well and its computation is relatively controllable. The approximation it implies is that "the better the local output is reconstructed, the less the subsequent model tends to be affected"; however, it does not explicitly multiply by the Jacobian of the subsequent network with respect to that error, so some directions with a small local MSE may still be amplified by later layers.

### 2.3 Mathematical Description

Compute the MSE for each alignable output path:

$$
\text{MSE}_j = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float}}^{(i)} - y_{\text{quant}}^{(i)})^2
$$

Aggregate within the block by taking the mean:

$$
\text{score} = \frac{1}{m} \sum_{j=1}^{m} \text{MSE}_j
$$

- $y_{\text{float}}$: the floating-point forward output of the submodule
- $y_{\text{quant}}$: the quantized forward output of the submodule
- $n$: number of output elements
- $m$: the number of valid submodules in the block (used for the quantization comparison)
- $\text{score}$: the sensitivity score of the Decoder block

If the subsequent network is denoted $g_l$, the change in the final output is approximated to first order as:

$$
\Delta y\approx J_{g_l}\,\delta h_{l+1}.
$$

Layer-wise MSE measures only $\|\delta h_{l+1}\|_2^2/n$, which is equivalent to assuming that all local error directions are equally important; it does not multiply by the downstream Jacobian $J_{g_l}$. This is precisely its mathematical difference from Model-wise MSE. If the output scales differ greatly across layers, additional normalized quantities such as $\|\delta h\|_2/\|h\|_2$ may be considered, but the definition must be kept consistent when ranking.

### 2.4 Key Properties

- **`layer`-scope analysis**: outputs a sensitivity ranking at Decoder-block granularity.
- **Within-block aggregation**: the score for the block is the mean of all valid MSEs in the block.
- **Analysis-scope dependent**: the block-level score depends on which substructures are included in the quantization intervention, so different analysis scopes should produce separate rankings.
- **Strongly local**: better suited to locating layers that are directly hard to quantize, but it does not capture how the subsequent network amplifies the error direction.

Its limitation is that it ignores the sensitive directions of the subsequent network. The MSE is also affected by the output dimension and magnitude: if the output scales differ greatly across layers, the absolute MSE values may not be fully comparable; when aggregating multiple submodules, the number of modules and their scales also affect the layer score.

### 2.5 Use Cases

- Scenarios that compare the sensitivity of layers from the perspective of submodule outputs within a Decoder block, supporting whole-layer or whole-block fallback decisions.
- Scenarios that require a block-level evaluation of the quantization reconstruction error.

### 2.6 Usage Restrictions

- The `quant_modules` parameter determines the set of submodules included in the quantization comparison; different configurations produce different rankings.
- Only for `layer`-scope analysis; not applicable to the `linear`/`attn` scopes.

---

## 3. Related Terms

- [MSE Model Wise](../mse_model_wise/term_mse_model_wise.md): a similar algorithm and another `layer`-scope analysis metric, but it focuses on the global error of the model's final output.
- [Attention MSE](../attention_mse/term_attention_mse.md): a similar algorithm and another MSE-based sensitive layer analysis metric, but used for the `attn` scope.
- [Kurtosis](../kurtosis/term_kurtosis.md): a comparison algorithm and another sensitive layer analysis metric, but used for the `linear` scope.

---

## 4. Reference Documents

1. [Layer-Wise MSE Analysis Configuration Workflow Guide](./usage_mse_layer_wise.md)
