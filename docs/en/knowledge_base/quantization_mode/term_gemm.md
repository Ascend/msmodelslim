# GEMM

> **Entry Category**: Quantization Basic Concept ([Quantization Modes](README.md))<br>
> **English Name**: GEMM (General Matrix Multiply)<br>
> **Application Domain**: Linear layer computation, inference acceleration<br>

## 1. Overview

**GEMM** (General Matrix Multiply) is the matrix multiplication $C=A\times B$, the most common class of basic operation in a Transformer. Besides $Y=X\cdot W$ in [linear layers](linear_layer_quantization/README.md), $QK^{\top}$ and the attention weighting $PV$ in Flash Attention are also matrix multiplications, and all of them reduce to GEMM. Quantization lets GEMM run with integer/low-precision inputs and is the main source of compute benefit from quantization.

---

## 2. Term Description

### 2.1 Definition

GEMM is a general matrix multiplication of the form $C=A\times B$. At least two classes of computation in a Transformer reduce to GEMM: one is the linear layer, such as the Q/K/V projections, the attention output projection, and the $Y=X\cdot W$ of each MLP layer; the other is the matrix multiplication inside attention, such as the $QK^{\top}$ and the attention weighting $PV$ covered by [FA Quantization](fa_quantization/README.md). GEMM is therefore not equivalent to a linear layer; a linear layer is merely one application of GEMM.

### 2.2 Integer GEMM

**Integer GEMM** refers to a matrix multiplication whose inputs are low-precision integers and whose multiply-accumulate is performed in the low-precision integer domain, for example with inputs of [INT8](../dtype_ops/term_int8.md) or [INT4](../dtype_ops/term_int4.md). Once both weights and activations are quantized to integers, the whole computation is integer, and the result is multiplied back by the scale to reconstruct. Integer GEMM can call dedicated low-precision operators and is the main source of compute benefit from quantization.

If only one side is quantized (such as [W8A16 Static Quantization](linear_layer_quantization/term_w8a16_static.md)), the integers must be dequantized back to floating point before a floating-point GEMM, and there is no integer GEMM speedup.

### 2.3 Relationship with Quantization

- The prerequisite is [Quantization and Dequantization](../dtype_ops/term_quantization.md): only when both weights and activations are quantized to integers can the computation enter the integer domain.
- The choice of data type ([INT8](../dtype_ops/term_int8.md), [FP8](../dtype_ops/term_fp8.md), [MXFP8](../dtype_ops/term_mxfp8.md)/[MXFP4](../dtype_ops/term_mxfp4.md)) determines the hardware operators and the accuracy characteristics of integer/low-precision GEMM.

---

## 3. Related Workflows

- [One-Click Quantization Usage Guide](../../user_guide/usage_one_click_quantization.md): the execution entry point for quantization tasks.
- [Quantization Accuracy Tuning Guide](../../user_guide/process_quantization_precision_tuning.md): accuracy problems caused by the data type and the quantization scheme can be diagnosed and tuned through this workflow.

---

## 4. Related Terms

- [Quantization Modes](README.md): the parent concept, the directory this entry belongs to.
- [Quantization and Dequantization](../dtype_ops/term_quantization.md): a companion term, the prerequisite for integer GEMM.
- [Linear Layer Quantization](linear_layer_quantization/README.md): a subordinate concept, the quantization application of GEMM on linear layers.
- [FA Quantization](fa_quantization/README.md): a subordinate concept, the quantization application of GEMM on attention matrix multiplication.
- [W8A8 Static Quantization](linear_layer_quantization/term_w8a8_static.md): a companion mode, the typical implementation of integer GEMM.

---

## 5. Reference Documents

1. *[Quantization Modes](README.md)*
