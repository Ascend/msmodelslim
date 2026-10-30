# Attention MSE Sensitive Layer Analysis Algorithm Terminology Entry

> **Entry Category**: [Sensitive Layer Analysis Algorithms](../README.md#3-sensitive-layer-analysis-algorithms)<br>
> **English Name**: attention_mse<br>
> **Application Domain**: Quantization sensitive layer analysis, Attention weight quantization<br>

---

## 1. Overview

Attention MSE (`mse`) is a quantization sensitivity analysis algorithm at the granularity of an attention module. It compares the output of the same Attention module under floating-point weights and under the target quantized weights, and measures the quantization perturbation by the mean squared error, thereby producing a sensitivity ranking. Its core characteristic is that it directly observes the actual output error of the attention subsystem, and it complements layer-level metrics such as [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md).

---

## 2. Term Description

When performing weight quantization on Attention structures or evaluating their sensitivity, it is necessary to directly measure the output drift of the attention subsystem under quantized weights. Attention MSE runs the forward pass with floating-point and quantized weights separately, compares the two sets of tensors at the attention module output, and uses the MSE to characterize how sensitive the module is to weight quantization.

### 2.1 Core Idea

The core idea of Attention MSE is to "directly measure the output drift": for the same calibration sample, run the forward pass with floating-point weights and with quantized weights separately, collect the tensors at the attention module output, and compute the mean squared error between the two outputs. A larger MSE means the attention module is more sensitive to the current quantization scheme.

This metric comes closer to the actual function of attention than simply comparing weight MSE, because it compresses the combined effect of the weight error after the Q/K/V projections, the dot product, and softmax into a single score at the output level.

### 2.2 Working Mechanism

During analysis, the target attention module can be viewed as a controlled intervention target: keep the input sample fixed, change only the weight/operator numerical error introduced by the quantization configuration being evaluated, and then obtain the floating-point output $Y_{fp}$ and the quantized output $Y_q$ separately. The MSE between them is the direct trace this quantization perturbation leaves at the output of the attention subsystem.

This position is highly integrative. Errors in Q/K first affect the dot-product logits and are then redistributed by the softmax nonlinearity into attention probabilities; errors in V or the output projection affect the weighted-sum result. Therefore the attention output MSE does not merely reflect the quantization error of a single weight tensor; it also partly captures how the internal nonlinearity of attention amplifies or suppresses the error. Comparing module by module under the same data and the same baseline yields a relative sensitivity ranking for fallback or precision improvement.

### 2.3 Mathematical Description

Compute the MSE between the floating-point and quantized outputs for the same layer and the same sample:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float}}^{(i)} - y_{\text{quant}}^{(i)})^2
$$

- $y_{\text{float}}$: the attention output using floating-point weights
- $y_{\text{quant}}$: the attention output using quantized weights
- $n$: number of output elements
- $\text{MSE}$: mean squared error, used for sensitivity ranking

A first-order expansion for small perturbations lets the logit error caused by Q/K be written as:

$$
\Delta L \approx \frac{\Delta QK^T+Q\Delta K^T}{\sqrt d}.
$$

If $P=\operatorname{softmax}(L)$, the local Jacobian of one row of the softmax is $J_{\mathrm{sm}}=\operatorname{diag}(p)-pp^T$, so:

$$
\Delta O \approx J_{\mathrm{sm}}\Delta L\,V + P\Delta V.
$$

This approximation shows that Attention MSE is influenced both by the magnitude of the quantization error and by the current shape of the attention distribution: when the logits at several positions are close, a probability reordering may be more pronounced; when the softmax is already highly saturated, perturbations in some directions may instead be compressed.

### 2.4 Key Properties

- **`attn`-scope analysis**: outputs a sensitivity ranking at the granularity of attention modules.
- **Direct measurement**: directly measures the output drift of the attention subsystem under quantized weights.
- **Quantization-configuration aware**: the MSE magnitude is related to the current quantization scheme.
- **Captures the nonlinear propagation inside attention**: the amplification of Q/K errors through the logits and softmax is reflected in the final output MSE.

The limitations stem from the same point: the MSE is affected by the output magnitude, sequence length, mask, input semantics, and the current quantization baseline, so its absolute value has no cross-model universal threshold; it is also only a numerical proxy and is not equivalent to final task quality.

### 2.5 Use Cases

- Scenarios that require weight quantization of Attention structures or evaluation of their sensitivity.
- Scenarios that need a sensitivity ranking at the granularity of attention modules to support fallback decisions.

### 2.6 Usage Restrictions

- Requires comparable outputs of the same Attention module under floating-point weights and the target quantized weights.
- The MSE value depends on the current quantization scheme and the calibration data, so it is only suitable for comparing the relative sensitivity among modules under the same quantization target.

---

## 3. Related Terms

- [Std](../std/term_std.md): a comparison algorithm and another sensitive layer analysis metric, but used for the `linear` scope.
- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md): a similar algorithm and another MSE-based sensitive layer analysis metric, but used for the `layer` scope.
- [MSE Model Wise](../mse_model_wise/term_mse_model_wise.md): a similar algorithm and an analysis metric based on the MSE of the model's final output.

---

## 4. Reference Documents

1. [Attention MSE Analysis Configuration Workflow Guide](./usage_attention_mse.md)
