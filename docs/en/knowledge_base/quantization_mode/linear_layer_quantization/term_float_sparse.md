# Float Sparse Quantization

> **Entry Category**: Quantization Data Format ([Linear Layer Quantization](README.md))<br>
> **English Name**: Float Sparse<br>
> **Application Domain**: Large language model quantization and compression, inference acceleration<br>
> **Hosting IR Class**: `W16A16sLinear` ([`msmodelslim/ir/w16a16s.py`](../../../../../msmodelslim/ir/w16a16s.py), a `nn.Module` subclass, not `AutoFakeQuantLinear`)<br>
> **Export Type**: `W16A16S`; the secondary-compression deployment form is `W16A16SC`<br>

## 1. Overview

Float sparse quantization (export type `W16A16S`) does not lower the bit width of the weights and activations; instead it applies **numerical sparsity** to the weights: part of the floating-point weight values are set to zero while the remaining elements stay [FP16/BF16](../../quantization_basic/term_fp16_bf16.md). It isolates bit-width quantization error so that a controlled experiment reflects only the sparsity dimension—the forward pass is still a 16bit linear layer computation, and the zero values are skipped by the hardware sparsity operator. `W16A16SC` is a special deployment form of `W16A16S` that performs one more weight compression after the numerical sparsification, for landing on a compression unit.

---

## 2. Term Description

### 2.1 Mode Specification

| Dimension | Value | Description |
|------|------|------|
| Quantization Target | Weight W, Activation A | Neither is quantized; both participate in the computation as FP16/BF16 |
| Bit Width | W 16bit, A 16bit | No bit-width compression |
| Data Type | FP16/BF16 | Floating point; the sparsity comes from zeroing weight values and is not part of the `WxAy` bit-width naming |
| Parameter Acquisition | N/A | No quantization parameters are produced; the weights are numerically sparse tensors produced by upstream sparsification |
| Quantization Granularity | N/A | Sparsity zeroes values by element magnitude, so there is no notion of quantization granularity |
| Symmetry | N/A | No quantization parameters |
| Sparse Form | `W16A16S` / `W16A16SC` | `W16A16S` is the numerical-sparsity export; `W16A16SC` is its secondary-compression deployment form |

### 2.2 Quantization Formula

This mode does not lower the bit width and introduces no quantization error, serving as the sparse quantization baseline; there is no quantization/dequantization formula. Its weights and activations stay 16bit (FP16/BF16), and what it carries are tensors in which some elements are zero after upstream sparsification.

### 2.3 Relationship to Other Modes

- **Relationship to [W8A8 Sparse Quantization](term_w8a8_sparse.md) (different compression path)**:
  - This mode: the weights stay 16bit floating point and part of the values are zeroed, used to evaluate the accuracy loss introduced by zeroing sparsity alone.
  - W8A8 sparse: INT8 quantization is performed first, then the actual value range is narrowed to a subset of the 256 integer levels (codebook narrowing) and compressed a second time to raise the compression ratio, rather than zeroing elements.

- **Relationship to [W8A16 Static Quantization](term_w8a16_static.md) (both are high-accuracy baselines, with different sources of benefit)**:
  - This mode: neither W nor A is quantized, and the benefit comes from numerical sparsity (skipping zero-valued elements).
  - W8A16: the benefit comes from weight bit-width compression (halving the memory traffic at 8bit). The two correspond to two independent compression paths: sparsity and bit width.

- **Relationship to [W8A8 Static Quantization](term_w8a8_static.md) (integer quantization scheme)**:
  - This mode is a zeroing-sparsity baseline with zero quantization error; W8A8 is an integer quantization scheme. The two use different compression paths and should not be understood as a simple stack.

### 2.4 Applicable Scenarios and Limitations

#### 2.4.1 Applicable Scenarios

- **Sparse quantization accuracy baseline**: used as a comparison that is free from bit-width quantization interference when evaluating the accuracy of a sparse scheme.
- **Sparse benefit evaluation**: measures on its own the compute/memory-traffic benefit of numerically sparse weights on the target hardware.
- **Secondary-compression deployment**: when a hardware compression unit is available, `W16A16SC` is exported from `W16A16S`.

#### 2.4.2 Usage Limitations

- **No bit-width compression benefit**: it does not lower the bit width itself and is not used as the final deployment compression scheme.
- **Depends on hardware sparsity support**: the sparsity benefit requires support from the target hardware sparsity operator; otherwise it is no different from ordinary FP16.
- **Non-quantized IR**: it does not belong to the `AutoFakeQuantLinear` family and does not take part in the computation or validation of quantization parameters.

---

## 3. Related Workflows

- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: select this mode via `--quant_type` or a YAML configuration and run quantization.
- *[Quantization Accuracy Tuning Guide](../../../user_guide/process_quantization_precision_tuning.md)*: accuracy degradation caused by this mode can be rolled back layer by layer and tuned through that workflow.

---

## 4. Related Terms

- [Quantization Modes](../README.md): the parent concept; this entry belongs to the [Linear Layer Quantization](README.md) category and is one concrete mode under it.
- [W8A8 Sparse Quantization](term_w8a8_sparse.md): a comparison mode, in which the INT8 codebook is narrowed and then compressed a second time, not zeroed.
- [W8A16 Static Quantization](term_w8a16_static.md): a peer mode, the accuracy baseline along the bit-width dimension.
- [W8A8 Static Quantization](term_w8a8_static.md): a comparison mode that can be combined with sparsity to form W8A8S.
- [Float Sparse (ADMM) Algorithm](../../quantization_algorithms/float_sparse/term_float_sparse.md): a companion term, the algorithm producing the sparse weights carried by this mode.
- [Linear Quantization Algorithm](../../quantization_algorithms/linear_quant/term_linear_quant.md): a companion entry; the code implementation of linear layer quantization.

---

## 5. Reference Documents

1. *[Float Sparse Parameter Configuration Workflow Guide](../../quantization_algorithms/float_sparse/usage_float_sparse.md)*
2. *[Linear Quantization Parameter Configuration Workflow Guide](../../quantization_algorithms/linear_quant/usage_linear_quant.md)*
3. *[Quantization Modes](../README.md)*
