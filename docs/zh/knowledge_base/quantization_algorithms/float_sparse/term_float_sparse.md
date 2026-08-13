# 浮点稀疏（ADMM）算法词条

> **词条类别**：量化算法
> **英文名称**：Float Sparse
> **英文缩写**：ADMM
> **应用领域**：模型稀疏化、高压缩率部署、推理加速
> **msModelSlim 实现**：`msmodelslim/processor/sparse/float_sparse.py`、`msmodelslim/processor/sparse/admm.py`

---

## 1. 概述

浮点稀疏（Float Sparse）是一种基于 ADMM（交替方向乘子法）的模型稀疏化算法。它对浮点权重进行稀疏化处理，结合 L2 量化保持重要位置的精度，通过硬件压缩单元实现更高的压缩率。其核心特征是：ADMM 优化求解最优稀疏模式、激活统计构建 Hessian、迭代稀疏、精度保护，适用于 Atlas 300I Duo 等推理卡的压缩场景，稀疏化后可配合 [线性量化](../linear_quant/term_linear_quant.md) 进一步压缩。

---

## 2. 词条介绍

现有 W8A8S 量化方法虽然支持权重稀疏量化，但稀疏率较低，在 Atlas 300I Duo 推理卡压缩单元上难以实现理想的压缩效果；且为满足精度要求通常需要回退部分网络层，显著降低推理性能。浮点稀疏直接对浮点权重进行稀疏化，结合硬件压缩单元实现更高的压缩率，在保证精度的同时提升推理性能。

---

## 3. 原理

### 1. 核心思想

浮点稀疏的核心思想是“以 ADMM 求解最优稀疏模式”：通过前向 hook 收集激活统计信息构建 Hessian 矩阵，使用 ADMM 算法迭代求解带约束的优化问题，找到最优的权重稀疏模式；并通过识别重要权重位置、应用 L2 量化保持关键权重精度。

### 2. 数学描述

ADMM 求解带稀疏约束的权重优化问题，主循环迭代更新：

$$
\text{sparse\_weights} = (\text{weights} + \lambda) \cdot \text{mask}
$$

$$
\lambda \leftarrow \lambda + (\text{weights} - \text{sparse\_weights})
$$

$$
\text{weights} \leftarrow H^{-1} \cdot (H \cdot \text{weights} + \rho \cdot (\text{sparse\_weights} - \lambda))
$$

- $W$：权重矩阵
- $\text{mask}$：稀疏掩码
- $\lambda$：拉格朗日乘子
- $H$：Hessian 矩阵，$H = X^T X$（$X$ 为输入激活）
- $\rho$：惩罚参数，初始化为 $\rho_0 = \text{PERCDAMP} \times \operatorname{mean}(\operatorname{diag}(H))$

统计信息收集阶段累积 Hessian 并计算行缩放因子：

$$
H \leftarrow H + X^T X
$$

$$
\text{scaler\_row} \leftarrow \text{scaler\_row} + \frac{\|X_i\|_2^2}{n\_samples}
$$

- $X$：输入激活
- $X_i$：第 $i$ 行激活
- $n\_samples$：样本数

### 3. 关键性质

- **ADMM 优化**：使用交替方向乘子法求解带约束的稀疏优化问题。
- **激活统计驱动**：通过前向 hook 收集激活统计构建 Hessian，自适应调整稀疏策略。
- **精度保护**：使用 L2 量化保持重要位置精度，避免关键权重被过度压缩。
- **迭代稀疏**：通过多次迭代逐步优化稀疏模式，平衡稀疏率与精度。
- **硬件压缩适配**：结合 Atlas 300I Duo 压缩单元实现高压缩率。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[收集激活统计] --> B[构建 Hessian]
    B --> C[ADMM 迭代稀疏]
    C --> D[精度保护]
    D --> E[部署稀疏模块]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/processor/sparse/float_sparse.py` 与 `msmodelslim/processor/sparse/admm.py` 中实现，核心类包括 `AdmmPruner` 与 `FloatSparseProcessor`，通过 `type: "float_sparse"` 处理器使用。

### 2. 处理流程

- **预处理阶段**：安装前向 hook 收集输入激活数据，累积 Hessian 矩阵并计算行缩放因子。
- **ADMM 稀疏化**：归一化 Hessian 与权重，设置初始惩罚参数，执行 ADMM 主循环（投影到稀疏空间、更新拉格朗日乘子、更新权重）。
- **精度保护**：使用量化误差与缩放因子的乘积作为重要性度量，选择 top-k% 重要位置应用 L2 量化。
- **模块部署**：将稀疏化后的模块转换为量化模块。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "float_sparse"
      sparse_ratio: 0.3
      include: ["*"]
      exclude: ["*self_attn*"]
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"float_sparse"`。 |
| sparse_ratio | 稀疏比例 | 取值范围 `0.0~1.0`，默认 `0.3`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配，优先级高于 `include`。 |

---

## 6. 适用场景与限制

### 1. 适用场景

- 需要高压缩率模型部署的场景，尤其是 Atlas 300I Duo 推理卡压缩单元场景。
- W8A8S 稀疏率不足、需要更高压缩率的场景。

### 2. 使用限制

- 生成的权重需要在 Atlas 300I Duo 推理卡上利用硬件压缩单元进行进一步压缩。
- Atlas 300I Duo 不支持 bfloat 数据类型，需将模型 `config.json` 的 `torch_dtype` 修改为 `float16`。
- 仅支持 v1 框架中的逐层量化，仅支持 `nn.Linear` 模块。
- 校准数据 token id 个数需 ≥2048；稀疏比例建议在 0.3 附近逐步调整。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：可集成浮点稀疏作为稀疏化步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可调整稀疏比例。

---

## 8. 关联词条

- [线性量化](../linear_quant/term_linear_quant.md)：配套术语，稀疏化后可配合量化进一步压缩。
- [SVDQuant](../svdquant/term_svdquant.md)：对比算法，同为面向高压缩率的模型压缩方案。
- [SSZ](../ssz/term_ssz.md)：配套术语，同为低比特/低资源模型压缩算法。

---

## 9. 参考资料

1. 《浮点稀疏 使用指南》([./usage_float_sparse.md](./usage_float_sparse.md))
