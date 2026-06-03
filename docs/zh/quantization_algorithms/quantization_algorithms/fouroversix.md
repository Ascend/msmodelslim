# FouroverSix：自适应块缩放权重量化算法说明

## 简介

- **问题**：传统的 mxFP4 量化方法在处理不同分布的数据块时，统一使用固定的缩放因子（通常将最大值缩放到 FP4 的上限 6），可能导致部分块的量化误差较大。
- **目标**：通过自适应选择每个数据块的最优缩放因子，在保持 mxFP4 量化格式的同时，最小化整体量化误差，提升模型精度。
- **论文链接**：https://arxiv.org/abs/2512.02010

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../getting_started/install_guide.md)。

## 原理和实现

### 原理

FouroverSix（4-over-6）算法的核心思想是对每个数据块**自适应选择**最优的缩放方案：

**核心思想：**

1. **分块处理**：将权重矩阵按指定的块大小划分成多个独立的数据块。
2. **双路径评估**：对每个数据块同时尝试两种缩放方案：
   - **方案A（Scale-to-6）**：将块内最大值缩放到 FP4 格式的最大值 6，充分利用动态范围。
   - **方案B（Scale-to-4）**：将块内最大值缩放到较小的值 4，为数据分布提供更多余量。
3. **智能择优**：计算两种方案的均方误差（MSE），选择误差较小的方案作为该块的最终量化方案。
4. **指数舍入**：缩放因子使用 e8m0 格式存储，通过最近邻舍入（含银行家舍入规则）确保精度。

### 实现

算法实现在 [`msmodelslim/core/quantizer/impl/fouroversix.py`](../../../../msmodelslim/core/quantizer/impl/fouroversix.py) 中：

- 实现类：`WeightFouroverSixQuantizer`
- 注册的量化类型：`mxfp4_per_block_sym`

**核心代码逻辑：**

```python
# 方案A: Scale to 6
scale_a = max_per_block / 6.0
scale_E_a = self.__nearest_neighbor_rounding_to_e8m0(scale_a)
# ... 量化、反量化 ...
mse_a = torch.mean((weight_tensor - dequantized_weights_a) ** 2, dim=-1)

# 方案B: Scale to 4
scale_b = max_per_block / 4.0
scale_E_b = self.__nearest_neighbor_rounding_to_e8m0(scale_b)
# ... 量化、反量化 ...
mse_b = torch.mean((weight_tensor - dequantized_weights_b) ** 2, dim=-1)

# 选择 MSE 较小的方案
mask = mse_a <= mse_b
selected_scale = torch.where(mask, scale_E_a, scale_E_b)
reshape_quantized_weights = torch.where(mask, quantized_weights_a, quantized_weights_b)
```

**指数舍入策略（e8m0 格式）：**

- **尾数 > 0.5**：指数加 1
- **尾数 < 0.5**：指数保持不变
- **尾数 == 0.5**：采用银行家舍入规则（偶数进 1，奇数不进）

## 适用要求

- **高精度需求**：适用于对精度要求较高的 mxFP4 量化场景，特别适合处理数据分布不均匀的模型。
- **计算成本**：FouroverSix 算法需要对每个块执行两次量化/反量化操作并计算 MSE，计算量略高于传统的 mxFP4 量化。
- **使用限制**：
  - 仅支持 mxFP4 格式的 per_block 对称量化。
  - 权重必须为 2D 张量。

## 功能介绍

### YAML配置示例

作为 Processor 使用，YAML 配置示例如下：

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_block"    # 量化范围
          dtype: "mxfp4"        # 量化数据类型
          symmetric: true       # 是否对称量化
          method: "fouroversix" # 量化算法-FouroverSix
```

### YAML配置字段详解

#### qconfig.weight (权重量化配置)

| 参数名       | 作用     | 可选值                      | 说明                                            | 默认值                         |
|-----------|--------|--------------------------|-----------------------------------------------|-----------------------------|
| scope     | 量化范围   | `"per_block"`            | per_block: 每个块独立参数                            | `"per_block"`               |
| dtype     | 量化数据类型 | `"mxfp4"`                | mxFP4 格式量化                                    | `"mxfp4"`                   |
| symmetric | 是否对称量化 | `true`, `false`          | true: 对称量化，零点为0<br/>false: 非对称量化，零点可调整        | `true`                      |
| method    | 量化方法   | `"fouroversix"`          | fouroversix: 自适应块缩放量化算法                      | `"fouroversix"`             |

## 技术优势

### 与传统 mxFP4 量化的对比

| 特性 | 传统 mxFP4 量化 | FouroverSix 量化 |
|------|----------------|-----------------|
| 缩放策略 | 固定（最大值→6） | 自适应（6 或 4） |
| 量化误差 | 部分块误差较大 | 每个块最优 |
| 计算复杂度 | O(N) | O(2N) |
| 精度表现 | 基准 | 通常提升 0.5-2.0 个百分点 |

### 适用场景

- **Transformer 模型**：注意力层和前馈网络层的权重分布差异较大，FouroverSix 能自适应处理。
- **多模态模型**：不同模态的数据分布差异显著，自适应缩放能有效提升精度。
- **大模型量化**：模型越大，权重分布越多样化，FouroverSix 的优势越明显。

## FAQ

### FouroverSix 算法名称的由来？

FouroverSix 名称来源于算法的核心思想：对于 FP4 格式（最大值为 6），每个数据块可以选择将最大值缩放到 6（充分利用动态范围）或缩放到 4（提供更多余量）。"4-over-6" 表示在 6 的上限内，智能选择是否使用 4 作为实际的缩放目标。

### 为什么选择 4 作为备选缩放目标？

- FP4 格式的动态范围是 [-6, 6]，选择 4 作为备选可以为数据分布提供约 33% 的余量。
- 经验表明，当数据块内存在离群值或分布不均匀时，使用较小的缩放因子可以显著降低量化误差。
- 4 是 6 的约 2/3，在数学上是一个合理的平衡点。
