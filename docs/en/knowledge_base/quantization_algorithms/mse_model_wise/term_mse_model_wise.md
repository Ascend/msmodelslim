# Model-wise MSE Sensitive Layer Analysis Algorithm Terminology Entry

> **Entry Category**: [Sensitive Layer Analysis Algorithms](../README.md#3-sensitive-layer-analysis-algorithms)<br>
> **English Name**: mse_model_wise<br>
> **Application Domain**: Quantization sensitive layer analysis, model quantization accuracy tuning<br>

---

## 1. Overview

Model-wise MSE (`mse_model_wise`) is an algorithm that measures the quantization sensitivity of a single layer from the perspective of the final model output. It introduces the target quantization perturbation layer by layer and then compares the mean squared error of the final hidden state or output before and after the perturbation, in order to judge the global impact of a local error after propagation through the rest of the network; its core characteristics are an end-to-end propagation perspective, layer-by-layer intervention, and complementarity with local MSE metrics.

---

## 2. Term Description

[Layer-wise MSE](../mse_layer_wise/term_mse_layer_wise.md) focuses on the local reconstruction error at a single block's output and cannot directly reflect the effect of that layer's quantization on end-to-end behavior. Model-wise MSE observes that by simulating the real inference path through layer-by-layer chained forward passes (taking the previous layer's output as the next layer's input) and using the MSE of the model's final output as the sensitivity score, the end-to-end quantization impact can be captured more closely.

### 2.1 Core Idea

The core idea of Model-wise MSE is to "measure layer sensitivity by the error after the quantization perturbation has propagated to the end of the model": for the same calibration data, apply a quantization perturbation to the target substructure of one candidate decoder layer at a time, let the error continue through the rest of the network, and then use the MSE of the final output relative to the floating-point baseline as that layer's score.

Model-wise observation is usually closer to the final accuracy because it automatically incorporates how the following layers respond to the direction of the error.

### 2.2 Working Mechanism

Model-wise MSE likewise runs sensitivity experiments with quantization perturbations, but places the observation point at the later, model-level output. After applying a quantization error $\delta h_l$ to candidate layer $l$, the perturbation is allowed to continue through the remaining network, and the final output is compared with the all-floating-point baseline. Under a linearization approximation, the final change can be written as $\delta y\approx J_l\delta h_l$, so the metric captures both the local quantization error and the amplification/suppression effect of the downstream network Jacobian.

Each evaluation usually needs the perturbation to propagate through the full remaining path, so the cost is higher than layer-wise MSE; but it comes closer to answering "if this layer really is quantized, how much will the overall model numerics deviate". Because deep networks are nonlinear, errors from different layers also interact, so a single-layer model-wise score is mainly used for ranking and candidate fallback and cannot simply be summed to predict the final loss when several layers are quantized at once.

### 2.3 Mathematical Description

For each layer, the MSE of the final output:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float, final}}^{(i)} - y_{\text{quant, final}}^{(i)})^2
$$

- $y_{\text{float, final}}$: the model's final output when only that layer's relevant substructure remains in floating point
- $y_{\text{quant, final}}$: the model's final output after that layer's relevant substructure is quantized
- $n$: number of output elements
- $\text{MSE}$: that layer's sensitivity score

A larger score means the layer is more sensitive to quantization (that is, quantizing it has a greater impact on the model's final output).

If the final model is the composite mapping $F=g_l\circ f_l$, then for a quantization perturbation $\delta h_l$ at layer $l$:

$$
\Delta y\approx J_{g_l}\delta h_l,\qquad\|\Delta y\|_2^2\approx\delta h_l^T(J_{g_l}^TJ_{g_l})\delta h_l.
$$

Model-wise MSE therefore implicitly uses a metric matrix $J^TJ$ determined by the downstream network. If several layers produce perturbations $\delta y_i$ at the same time, the total error also contains cross terms $2\langle\delta y_i,\delta y_j\rangle$, which explains why single-layer scores cannot simply be summed to predict the quantization error of the whole model.

### 2.4 Key Properties

- **`layer`-scope analysis**: outputs a sensitivity ranking at Decoder-block granularity.
- **End-to-end perspective**: measures the quantization impact by the model's final output, staying closer to the end-to-end effect.
- **Chained forward pass**: simulates the real inference path with layer-by-layer chained forward passes.
- **Includes downstream propagation**: the final output error implicitly includes the response of all networks after the candidate layer to the quantization perturbation.

The cost is that evaluation is more expensive, and the final output MSE is still only a proxy for the task loss. In autoregressive models, a very small logit difference may change the discrete token during sampling, and conversely there may be a numerical difference without a change in ranking.

### 2.5 Use Cases

- Scenarios that want to assess the quantization sensitivity of each layer from the perspective of the model's final output to support whole-layer or whole-block fallback decisions.
- Scenarios that require end-to-end quantization impact evaluation.

### 2.6 Usage Restrictions

- The number of calibration batches and the sequence length significantly increase the number of forward passes and the intermediate cache, which may cause OOM on large models; control the calibration scale accordingly.
- Some architectures are constrained by chained-forward alignment; when the tensor shapes or semantics between layers cannot be aligned, that layer and the following layers are skipped.
- Only for `layer`-scope analysis; not applicable to the `linear`/`attn` scopes.

---

## 3. Related Terms

- [MSE Layer Wise](../mse_layer_wise/term_mse_layer_wise.md): a similar algorithm and another `layer`-scope analysis metric, but it focuses on the local reconstruction error at a single block's output.
- [Attention MSE](../attention_mse/term_attention_mse.md): a similar algorithm and another MSE-based sensitive layer analysis metric, but used for the `attn` scope.
- [Std](../std/term_std.md): a comparison algorithm and another sensitive layer analysis metric, but used for the `linear` scope.

---

## 4. Reference Documents

1. [Model-wise MSE Analysis Configuration Workflow Guide](./usage_mse_model_wise.md)
