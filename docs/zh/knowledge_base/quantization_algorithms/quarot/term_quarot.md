# QuaRot 旋转量化算法词条

> **词条类别**：离群值抑制算法
> **英文名称**：QuaRot
> **英文缩写**：QuaRot
> **首次提出**：Ashkboos et al., NeurIPS 2024
> **应用领域**：大语言模型量化压缩、低比特量化精度优化
> **msModelSlim 实现**：`msmodelslim/processor/quarot/`

---

## 1. 概述

QuaRot（Quantization with Rotation）是一种用于大语言模型量化的离群值抑制算法。它通过对权重和激活值应用精心构造的正交旋转矩阵（如 Hadamard 矩阵），将离群值“分散”到多个通道中，使变换后的激活在每个通道上的最大值尽可能均衡，从而降低量化误差。其核心特征是：正交变换保证计算等价、旋转不改变模型输入输出映射、可配合在线旋转算子进一步提升精度，与 [SmoothQuant](../smooth_quant/term_smooth_quant.md) 同为离群值抑制算法。

---

## 2. 词条介绍

大语言模型的激活值中存在离群值，直接量化时单个通道的极端离群值会迫使缩放因子过大，导致整体量化精度下降。QuaRot 观察到，正交旋转可以重新分布参数、均衡各通道的数值范围，且由于正交矩阵满足 $QQ^T = I$，变换前后模型数学等价，因此可以在不改变模型行为的前提下显著平滑激活分布，为低比特量化奠定基础。

---

## 3. 原理

### 1. 核心思想

QuaRot 的核心思想是“用正交旋转均衡通道分布”：对权重矩阵 $W$ 应用左旋转、对激活矩阵 $X$ 应用右旋转，使得旋转后的激活在每个通道上的最大值尽可能均衡。由于旋转是可逆正交变换，$X' W' = XW$ 严格成立，变换前后的模型计算等价。

### 2. 数学描述

设正交旋转矩阵为 $Q$（满足 $Q^T Q = I$），对权重与激活分别变换：

$$
W' = Q^T W, \quad X' = X Q
$$

变换保持计算等价：

$$
X' W' = (X Q)(Q^T W) = X W
$$

- $W$：权重矩阵
- $X$：激活矩阵
- $Q$：正交旋转矩阵（如 Hadamard 矩阵）
- $I$：单位矩阵

对于包含 RMSNorm 的层间，因 $RMSNorm(X) = RMSNorm(X Q^T) \cdot Q$，计算不变性依然成立。当启用块对角旋转时，$Q$ 为块对角矩阵，每个块的大小由 `block_size` 控制。

### 3. 关键性质

- **计算等价性**：正交旋转不改变矩阵乘法的数学结果，不引入额外推理误差。
- **离群值分散**：旋转将离群值分散到多个通道，均衡各通道最大值。
- **层间不变性**：RMSNorm 的缩放不变性保证跨层旋转后模型输出不变。
- **可块对角化**：支持 `block_size` 控制旋转矩阵的块对角结构，兼顾效果与计算效率。
- **在线旋转可选**：支持 `online` 模式在推理时动态注入旋转算子，进一步提升精度。

---

## 4. 流程示意

> 以下为本算法在 msModelSlim 中的简化流程概览。

```mermaid
flowchart LR
    A[校准数据] --> B[获取旋转映射]
    B --> C[融合 LayerNorm]
    C --> D[执行旋转]
    D --> E[插入量化]
    E --> F[交付量化]
```

---

## 5. 在 msModelSlim 中的实现

### 1. 实现位置

算法在 `msmodelslim/processor/quarot/` 目录下实现，核心处理逻辑位于 `offline_quarot/quarot.py`，通过 `type: "quarot"` 处理器使用。

### 2. 处理流程

- **pre_run 阶段**：从模型适配器获取 LayerNorm 融合映射、mean 融合名称与旋转映射；执行 `_fuse_norm`、`_bake_mean`、`_rotate`；可选导出全局旋转矩阵信息，可选初始化在线旋转。
- **preprocess 阶段**：逐层按 prefix 过滤该层相关的融合映射、bake 名称与旋转命令，执行该层的层融合与旋转；启用在线旋转时执行 `online_rotate_o_proj_input()` 与 `online_rotate_down_proj()`。
- **post_run 阶段**：执行剩余的融合、bake 与旋转操作，清理状态；启用在线旋转时将 HookIR 转换为 WrapperIR。

### 3. 配置示例

> 以下为最小可用的 YAML 配置片段。各字段的详细含义如下表所示。

```yaml
spec:
  process:
    - type: "quarot"
      online: False
      block_size: -1
      max_tp_size: 4
      down_proj_online_layers: []
      export_extra_info: True
```

**字段说明**：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"quarot"`。 |
| online | 在线旋转开关 | 布尔值，`True` 启用在线旋转，默认 `False`。 |
| block_size | 旋转矩阵对角块大小 | `-1` 或大于 0 的 2 的幂，`-1` 表示不进行块对角矩阵处理。 |
| max_tp_size | 最大张量并行大小 | 仅在线旋转时生效，大于 0 且为 2 的幂或等于 1，默认 `4`。 |
| down_proj_online_layers | 使用在线旋转的 down 层 | 层索引列表，指定哪些层的 `down_proj` 使用在线旋转，默认 `[]`。 |
| export_extra_info | 是否导出全局旋转信息 | 默认 `True`，为 `True` 时导出 `optional/quarot.safetensors` 并追加描述字段。 |

### 4. 模型适配接口

模型适配需实现 `QuaRotInterface` 接口，提供以下方法：

- `get_ln_fuse_map()`
- `get_bake_names()`
- `get_rotate_map(block_size)`

启用在线旋转（`online: True`）时还需实现 `LAOSOnlineRotationInterface`，提供以下方法：

- `get_head_dim()`
- `get_num_attention_heads()`
- `get_layer_wise_ov_pair()`
- `get_layer_wise_up_down_pair()`

参考实现：`msmodelslim/model/qwen3/model_adapter.py` 或 `msmodelslim/model/deepseek_v3/model_adapter.py`。

---

## 6. 适用场景与限制

### 1. 适用场景

- 激活值存在显著离群值、需要低比特（如 W4A4）量化的大语言模型场景。
- 作为 [AutoRound](../autoround/term_autoround.md) 等低比特量化算法的前置离群值抑制步骤。

### 2. 使用限制

- 模型必须实现 `QuaRotInterface` 接口；启用在线旋转时还需实现 `LAOSOnlineRotationInterface`。
- 启用在线旋转并以 TP 并行部署时，`tp_size` 需为 2 的幂且小于等于 `max_tp_size`，否则会导致精度异常。
- 在线旋转依赖推理框架支持插入旋转算子，且会一定程度降低推理性能，需权衡精度与性能。

---

## 7. 关联流程

- 《[一键量化 (V1)](../../../user_guide/usage_quick_quantization.md)》：默认集成本算法作为离群值抑制前置步骤。
- 《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》：精度不达标时可考虑启用本算法。

---

## 8. 关联词条

- [Adapt Rotation](../adapt_rotation/term_adapt_rotation.md)：下位概念，在 QuaRot 基础上用数据驱动方式优化旋转矩阵。
- [SmoothQuant](../smooth_quant/term_smooth_quant.md)：对比算法，采用通道级缩放而非正交旋转抑制离群值。
- [AutoRound](../autoround/term_autoround.md)：配套术语，常在本算法处理后接 AutoRound 权重量化。
- [KVCache Quant](../kvcache_quant/term_kvcache_quant.md)：配套术语，可与本算法组合用于长序列推理。
- [MinMax](../minmax/term_minmax.md)：应用对象，旋转后的激活值更易于 MinMax 量化。

---

## 9. 参考资料

1. Ashkboos S et al. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. NeurIPS 2024. https://arxiv.org/abs/2404.00456
2. 《QuaRot 使用指南》([./usage_quarot.md](./usage_quarot.md))
