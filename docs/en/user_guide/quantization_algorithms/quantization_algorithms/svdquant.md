# SVDQuant: Post-Training Quantization Algorithm for Diffusion Models Based on Low-Rank Residual Reconstruction

<!-- md-trans-meta sourceCommit=8957c34df2ee97f79b3b71a54d214332f282037e translatedAt=2026-08-19T08:16:39.261Z pushedAt=2026-08-19T08:17:52.370Z -->

## Introduction

SVDQuant is a post-training quantization technique for diffusion models. Through a three-stage pipeline—**outlier migration**, **low-rank decomposition**, and **residual quantization**—it absorbs outliers in weights and activations into low-rank components, thereby alleviating quantization difficulty and improving model performance. Based on the core idea of SVDQuant, this tool implements a three-stage quantization pipeline of `outlier suppression → svd_res → linear_quant`.

## Before You Begin

Install the msModelSlim tool. For details, see [msModelSlim Tool Installation Guide](../../../install_guide/install_guide.md).

## Principles and Implementation

### Principle

**Core idea:**

1. **Outlier migration** (taking Iterative Smooth as an example): migrate outliers in activations to weights through mathematically equivalent transformations, making the activation distribution more uniform and reducing the difficulty of activation quantization.

2. **Low-rank decomposition** (SVDResidual): perform SVD low-rank decomposition on the migrated weights, extracting the main structure of the weights as low-rank components (retained at high precision), while the residual part is more suitable for quantization.

3. **Residual quantization** (Linear Quant): perform low-bit quantization (such as 4-bit) on the residual weights, while the low-rank branch runs at high precision (such as FP16), thereby reducing quantization error while maintaining visual quality.

**Three-stage pipeline:**

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  iter_smooth    │ ──► │    svd_res      │ ──► │  linear_quant   │
│                 │     │                 │     │                 │
│  (Calibration   │     │                 │     │                 │
│data required.)  │     │  (data-free)    │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Stage 1: Outlier Migration**

Diffusion models exhibit significant outliers in their activations, where the extreme values of a few channels compress the quantization precision of the remaining channels. The outlier suppression algorithm "migrates" the outliers in activations into the weights through a mathematically equivalent transformation. Taking Iterative Smooth as an example, its core transformation is:

$$y = x W + b = \underbrace{(x \cdot \text{diag}(s)^{-1})}_{\text{smoothed activation}} \cdot \underbrace{(\text{diag}(s) \cdot W)}_{\text{weight absorbing outliers}} + b$$

where the scaling factor $s$ is jointly computed from the statistics of activations and weights:

$$s = \left(\frac{A_\text{scale}^\alpha}{W_\text{scale}^{1-\alpha}}\right), \quad s \geq s_\text{min}$$

- $A_\text{scale}$: the per-channel absolute maximum of the activations

- $W_\text{scale}$: the maximum value of each weight column

- $\alpha$: the balancing parameter that controls the migration strength (the larger $\alpha$ is, the more outliers are migrated to the weight)

After migration, the activation distribution becomes more uniform (and thus easier to quantize), but the weight, having absorbed the outlier information, becomes more difficult to quantize. This is exactly the problem that the next stage, SVD low-rank decomposition, is designed to solve.

**Stage 2: Low-Rank Residual Decomposition**

After outlier migration, the weight absorbs the outliers from the activations, and its distribution exhibits a clear low-rank structure (outliers are concentrated on a few channels). SVD low-rank decomposition is performed on the migrated weight $W$ to extract the low-rank components:

1. Perform SVD decomposition: $W \approx (U \cdot S) \cdot V^\top$

2. Compute the residual: $R = W - (U \cdot S) \cdot V^\top$

3. Replace the weight with the residual $R$, and retain the low-rank component in parameter form.

The main structure of the weights (including the low-rank components caused by outliers) is extracted into the low-rank branch and runs at high precision (such as FP16); the residual part is more evenly distributed and is therefore more suitable for low-bit quantization.

**Stage 3: Residual Quantization**

The residual weight $R$ and the corresponding activations are quantized at low bit width (such as W4A4 MXFP4). The residual main path runs with 4-bit quantization, while the low-rank bypass runs at high precision; the two are added together to restore the original output precision.

**Mathematical Equivalence of the Three-Stage Collaboration**

Combining the three stages, the computation at inference time is:

$$
\begin{aligned}
\text{out} &= \underbrace{Q(x \cdot \text{diag}(s)^{-1}) \cdot Q(R)}_{\text{quantized residual main path}} + \underbrace{(x \cdot \text{diag}(s)^{-1} \cdot V) \cdot (U \cdot S)^\top}_{\text{high-precision low-rank bypass}} \\
           &\approx x \cdot \text{diag}(s)^{-1} \cdot W + b \\
           &= x W + b
\end{aligned}
$$

Here, $Q(\cdot)$ denotes the quantization operation. When the quantization precision is sufficient, the output is approximately equivalent to that of the original linear layer.

### SVDResidual Principle Detailed Explanation

#### SVD Decomposition Process

Perform low-rank decomposition on the linear layer weight $W \in \mathbb{R}^{\text{out} \times \text{in}}$:

1. Convert the weight to `float32` to ensure numerical stability of the SVD computation:

   - `weight_float = weight.float()`

2. Set the rank: `rank = config.rank`

3. Perform decomposition:

   - `U, S, V = torch.svd_lowrank(weight_float, q=rank)`

   - The return values satisfy $W \approx U \cdot \text{diag}(S) \cdot V^\top$

   - `U [out_dim, rank]`, `S [rank]`, `V [in_dim, rank]`

4. Construct the low-rank parameters:

   - `svd_lowrank_l1 = V[:, :rank].t()`          → $V^\top$, shape `[rank, in_dim]`

   - `svd_lowrank_l2 = U[:, :rank] * S[:rank]`   → $U \cdot S$, shape `[out_dim, rank]`

5. Reconstruct the low-rank approximation:

   - `reconstructed = svd_lowrank_l2 @ svd_lowrank_l1`, i.e., $(U \cdot S) \cdot V^\top$

6. Compute the residual weight:

   - `residual = original_weight - reconstructed`, that is, $R = W - (U \cdot S) \cdot V^\top$

After processing:

- Replace `weight` with `residual`;

- Save `svd_lowrank_l1` and `svd_lowrank_l2` in the corresponding `Linear` module in parameter form;

- Wrap the layer as `SVDResidualWrapper` through Hook IR, and during the forward pass, execute the "residual main path + low-rank bypass" and add them together.

#### Forward Equivalence Relation

Taking a single linear layer as an example, the original linear layer is computed as:

$$y = x W^\top + b$$

After SVD residual decomposition:

- Original weight: $W$

- Low-rank approximation: $W_\text{low} = (U \cdot S) \cdot V^\top$

- Residual weight: $R = W - W_\text{low}$

The forward computation is divided into dual paths:

1. **Main path (residual)**:

   - `residual_out = wrapped_module(x)`, that is, $x R^\top + b$

   - At this point, the internal weight of `wrapped_module` has been replaced with $R$, and the bias remains unchanged.

2. **Bypass (low-rank two-stage linear)**:

   > **Key point**: The mathematical meaning of `F.linear(x, W)` is $x W^\top$ (note the transpose), therefore:

   - `lowrank_hidden = F.linear(x, svd_lowrank_l1, bias=None)`

     - The weight parameter is $V^\top$, and the actual computation is $x (V^\top)^\top = x V$.

   - `lowrank_out = F.linear(lowrank_hidden, svd_lowrank_l2, bias=None)`

     - The weight parameter is $U \cdot S$, and the actual computation is $(x V) (U \cdot S)^\top$

3. **Aggregation**:

   - `out = residual_out + lowrank_out`

   - Mathematical derivation:

$$
\begin{aligned}
\text{out} &= x R^\top + b + (x V)(U \cdot S)^\top \\
           &= x R^\top + b + x V (U \cdot S)^\top \\
           &= x \big(R^\top + V (U \cdot S)^\top\big) + b \\
           &= x \big(R + (U \cdot S) V^\top\big)^\top + b \\
           &= x W^\top + b
\end{aligned}
$$

Therefore, the sum of the outputs of the two paths is equivalent to a linear transformation using the original weight $W$.

#### Processing Scope

An `nn.Linear` module is processed if it meets the following conditions:

- `isinstance(submodule, nn.Linear)`

- The module name matches `include`

- The module name does not match `exclude`

During the `post_run` phase, a warning is issued for patterns that are not matched (and are not `"*"`).

### Implementation

The three-stage pipeline of SVDQuant is completed by three Processors working in sequence:

1. **Outlier migration**: Based on calibration data, collect activation statistics, compute scaling factors, and migrate activation outliers into the weights (taking Iterative Smooth as an example, compute the per-channel scaling factor $s$, multiply the weights by $s$, and divide the input by $s$). For details, see [Iterative Smooth Algorithm](../outlier_suppression_algorithms/iterative_smooth.md).

2. **Low-rank residual decomposition** (`svd_res`): Perform SVD low-rank decomposition $W \approx (U \cdot S) \cdot V^\top$ on the migrated weights, replace the weights with the residual $R = W - (U \cdot S) \cdot V^\top$, and retain the low-rank components $V^\top$ and $U \cdot S$ in parameter form. During the forward pass, two paths run in parallel: the residual main path computes $x R^\top + b$, and the low-rank bypass computes $(x V)(U \cdot S)^\top$; their sum is equivalent to the original linear transformation.

3. **Residual quantization** (`linear_quant`): Perform low-bit quantization on the residual weights and activations. For details, see [Linear Quantization Algorithm](linear_quant.md).

## Applicable Requirements

- **Low-bit quantization**: Suitable for extremely low-bit quantization scenarios such as W4A4, especially for diffusion models.

- **Outlier structure**: The model activations contain significant outliers, and the outliers exhibit a low-rank structure in the weights, making them suitable for absorption through a low-rank branch.

- **Computational resources**: `svd_res` is a data-free algorithm with low computational overhead; `iter_smooth` requires calibration data to collect activation statistics.

- **Usage restrictions**: The target layer must be a standard `torch.nn.Linear`, and the module name must be obtainable through `model.named_modules()`.

## Feature Description

### YAML Configuration Example

The following is a complete three-stage pipeline configuration example for SVDQuant (using Wan2.2 T2V W4A4 quantization as an example):

```yaml
spec:
  process:
    # Stage 1: outlier migration
    - type: "iter_smooth"
      alpha: 0.25                        # Balance parameter that controls the intensity of outlier migration
      include: ["*"]                     # Included layers
      exclude: ["*blocks.0.*"]

    # Stage 2: low-rank residual decomposition
    - type: "svd_res"
      rank: 32                           # Rank of the low-rank decomposition
      include: ["*"]                     # Keep consistent with iter_smooth
      exclude: ["*blocks.0.*"]

    # Stage 3: residual quantization (W4A4 MXFP4)
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp4"
          symmetric: True
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp4"
          symmetric: True
          method: "minmax"
      include: ["*"]
      exclude: ["*blocks.0.*"]
```

>[!NOTE]
>
> The `include`/`exclude` settings of the three stages should remain consistent to ensure that the same group of Linear layers sequentially undergoes outlier migration → low-rank decomposition → quantization.

### YAML Configuration Field Detailed Explanation

SVDQuant involves configuration fields of three processors. Only the exclusive fields of `svd_res` are described here:

| Field Name | Function | Description |
|---------|----------------|------|
| type    | Processor type identifier | Fixed value `"svd_res"`, used to identify an SVD residual processor. |
| rank    | Rank of the low-rank decomposition | An integer greater than 0 that controls the rank of the approximation. The default value is `32`. Due to operator implementation limitations, it is recommended not to exceed 128. |
| include | Layers to include | A list of strings that supports wildcard matching. It matches the full module paths returned by `model.named_modules()`. |
| exclude | Layers to exclude | A list of strings that supports wildcard matching. It is used to explicitly exclude modules that should not undergo SVD decomposition. |

For details about the configuration fields of other processors, see [Iterative Smooth YAML Configuration Fields](../outlier_suppression_algorithms/iterative_smooth.md#) and [Linear Quantization YAML Configuration Field Detailed Explanation](linear_quant.md#yaml-configuration-fields).

## FAQs

### 1. Why is outlier migration performed before SVD?

- Significant outliers exist in the activations of diffusion models, and direct quantization leads to severe accuracy degradation.

- Outlier migration moves outliers from activations to weights, making activations easier to quantize.

- However, after migration the weights become harder to quantize. Fortunately, outliers exhibit a low-rank structure in the weights, which is exactly suitable for extraction via SVD low-rank decomposition.

- This "migration → decomposition" combination is the core innovation of SVDQuant: outliers are first migrated to the weights and then absorbed by the low-rank branch, thereby solving the quantization difficulties of both activations and weights simultaneously.

### 2. How is the rank selected?

- A larger `rank` fits the original weights better, but incurs higher computation and storage overhead for the low-rank branch;

- A smaller `rank` achieves stronger compression, but retains more uncaptured information in the residual, which may affect quantization precision;

- You can perform a grid search or select empirically (such as 8, 16, 32, 64, etc.) based on the model scale and precision requirements.

### 3. Why float32 conversion is performed before SVD?

- For low-precision weights such as `float16` / `bfloat16`, performing SVD directly is prone to numerical instability;

- Converting to `float32` makes the SVD computation more numerically stable;

- After decomposition is complete, the resulting low-rank matrices are converted back to the original `dtype` and `device` to ensure compatibility with the training/inference environment.

### 4. How do I confirm which layers are decomposed?

- Check the warning messages in the logs about unmatched patterns.

- Enumerate the model's `named_modules()` in the code to confirm whether the names match the `include` / `exclude` patterns.

- After running the Processor, check whether the target `Linear` layers have gained the `svd_lowrank_l1` / `svd_lowrank_l2` parameters.

### 5. How does the Smooth alpha parameter affect SVDQuant?

- `alpha` controls the intensity of outlier migration from activations to weights: the larger $\alpha$ is, the more aggressive the migration.

- In the SVDQuant pipeline, `alpha` needs to be coordinated with `rank`: a larger `alpha` moves more outliers into the weights, which may require a larger `rank` to fully absorb these low-rank components.
