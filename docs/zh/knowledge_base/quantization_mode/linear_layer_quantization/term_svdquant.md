# SVDQuant 量化

> **词条类别**：量化数据格式（[线性层量化](README.md)）
> **英文名称**：SVDQuant Quantization
> **首次提出**：Li et al., NeurIPS 2024
> **应用领域**：大语言模型量化压缩、低比特量化精度优化
> **承载 IR 类**：`SVDResidualWrapper`（[`msmodelslim/ir/svd_residual.py`](../../../../../msmodelslim/ir/svd_residual.py)，`WrapperIR` 子类，配合 `linear_quant` 对残差分量做低比特量化）

---

## 1. 概述

SVDQuant 是一种面向极低比特场景的[线性层量化](README.md)模式：通过离群值迁移、SVD 低秩分解与残差量化三阶段协同，把权重拆成高精度低秩旁路与低比特残差主通路，推理时双通路相加还原原线性变换。它解决 4bit 等极低比特下激活与权重中离群值导致量化精度严重损失的问题，区别于以 [W4A4 动态量化](term_w4a4_dynamic.md) 为代表的直接整数量化；核心特征是低秩旁路高精度吸收主体结构、残差主通路低比特量化。

---

## 2. 词条介绍

SVDQuant 不是对线性层权重与激活直接做低位宽量化，而是在量化之前先改变权重的数值结构：离群值先被迁移进权重，再经 SVD 分解将主体结构提取到低秩旁路，使剩下的残差权重分布更均匀、更适合低比特量化。低秩旁路以 [FP16/BF16](../../quantization_basic/term_fp16_bf16.md) 高精度运行，残差主通路按 `linear_quant` 配置量化，两者输出相加，在数学上近似未分解前的线性变换。

### 模式规格

| 维度 | 取值 | 说明 |
|------|------|------|
| 量化对象 | 残差主通路的权重 W 与激活 A；低秩旁路不量化 | 离群值迁移后权重被拆为低秩分量（高精度）与残差分量（低比特量化） |
| 位宽 | 残差 W/A 低比特（如 4bit）；低秩旁路 FP16 | 位宽取决于残差线性量化配置 |
| 数据类型 | 混合 | 残差走低比特（如 MXFP4 / INT4），低秩旁路 FP16 |
| 参数获取方式 | 跟随残差线性量化配置（静态 / 动态） | 由 [`linear_quant`](../../quantization_algorithms/linear_quant/term_linear_quant.md) 决定 |
| 量化粒度 | 跟随残差线性量化配置（如 per-block） | 无固定粒度，随配置而定 |
| 对称性 | 跟随残差线性量化配置 | 随配置而定 |

### 量化公式

阶段一（离群值迁移）：用逐通道缩放因子改造权重与激活，使激活离群值迁入权重。阶段二（低秩分解）：对迁移后的权重 $W$ 做 SVD 分解：

$$
W \approx (U \cdot S) \cdot V^\top, \qquad R = W - (U \cdot S) \cdot V^\top
$$

- $W$：离群值迁移后的权重矩阵
- $U$：左奇异向量，形状 $[\text{out\_dim}, \text{rank}]$
- $S$：奇异值，形状 $[\text{rank}]$
- $V$：右奇异向量，形状 $[\text{in\_dim}, \text{rank}]$，$V^\top$ 为其转置
- $R$：残差权重，进入低比特量化

阶段三（推理时双通路计算）：

$$
\text{out} = Q(X \cdot \operatorname{diag}(s)^{-1}) \cdot Q(R) + (X \cdot \operatorname{diag}(s)^{-1} \cdot V) \cdot (U \cdot S)^\top \approx XW + b
$$

- $X$：输入激活
- $s$：离群值迁移的逐通道缩放因子，由激活与权重统计联合计算
- $Q(\cdot)$：低比特量化操作（如 4bit 量化）
- $b$：线性层偏置

### 与其他模式的关系

- **与 [W4A4 动态量化](term_w4a4_dynamic.md)（同为极低比特方案）**：W4A4 直接对全部权重与激活整数量化；SVDQuant 先分解权重、仅量化残差分量，借低秩旁路吸收离群值，降低 4bit 的量化精度风险。二者可理解为"直接量化"与"分解后量化"两条路径。
- **与 [W8A8 静态量化](term_w8a8_static.md)（基础线性量化方案）**：残差主通路的量化即通过 [`linear_quant`](../../quantization_algorithms/linear_quant/term_linear_quant.md) 实现，W8A8 等线性量化模式可作为残差主通路的底层量化方式。
- **与 [W16A16S 量化](term_w16a16s.md)（同为结构级模式）**：W16A16S 以保留高位宽、跳过稀疏零值为收益；SVDQuant 以低秩分量吸收离群值为收益。两者都通过"保留部分高精度分量 + 压缩主体"的结构性思路降低压缩精度风险。

### 适用场景与限制

#### 1. 适用场景

- 极低比特（如 W4A4）且激活 / 权重存在显著离群值的层，尤其适用于扩散模型与含大量 Linear 的模型。
- 需要尽量保留精度的压缩场景：低秩旁路以高精度保留主体结构，残差分量低比特化。

#### 2. 使用限制

- 目标层须为标准 `torch.nn.Linear`，且可通过 `model.named_modules()` 获取模块名。
- 离群值迁移、SVD 分解、残差量化三阶段的 `include` / `exclude` 应保持一致，确保同一组层依次经历三阶段。
- 分解秩 `rank` 受算子实现限制，建议不超过 128。

---

## 3. 关联流程

- 《[一键量化完整指南](../../../user_guide/usage_quick_quantization.md)》：通过 `--quant_type` 或 YAML 配置 SVDQuant 的 `iter_smooth` → `svd_res` → `linear_quant` 三阶段流水线。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可调整分解秩 `rank` 与离群值迁移强度 `alpha`。

---

## 4. 关联词条

- [量化模式](../README.md)：上位概念，本词条属于[线性层量化](README.md)类别，是该类别下的一种结构级模式。
- [线性层量化](README.md)：上位概念，本词条所属类别，SVDQuant 作用于其中的 Linear 层。
- [W4A4 动态量化](term_w4a4_dynamic.md)：对比模式，同属极低比特方案，SVDQuant 以分解方式降低 4bit 精度风险。
- [W8A8 静态量化](term_w8a8_static.md)：配套模式，可作为残差主通路的底层量化方式。
- [W16A16S 量化](term_w16a16s.md)：对比模式，同为"保留高精度分量 + 压缩主体"的结构级模式。
- [FP16/BF16](../../quantization_basic/term_fp16_bf16.md)：配套术语，低秩旁路使用的高精度数据类型。
- 《[SVDQuant 低秩残差量化算法词条](../../quantization_algorithms/svdquant/term_svdquant.md)》：配套术语，算法侧的原理与实现说明。

---

## 5. 参考文档

1. Li M et al. SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models. NeurIPS 2024. https://arxiv.org/abs/2411.05007
2. 《[SVDQuant 使用指南](../../quantization_algorithms/svdquant/usage_svdquant.md)》
3. 《[量化模式](../README.md)》
