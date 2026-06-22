# Ceil_X：自适应除数 MXFP4 权重量化算法说明

## 简介

- **问题**：MXFP4 per-block 量化中，传统方法使用 $s = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$ 计算 shared exponent，缩放后数值范围为 $\max(|x|) / 2^s \in [4, 8)$，而 MXFP4 的表示上限为 6.0，导致 $\,[6, 8)$ 区间内的大值被截断，引入较大量化误差。
- **目标**：通过引入可配置除数 $c$ 配合 `ceil` 操作，将缩放后数值范围压缩至 MXFP4 的可表示区间内，减少大值截断，提升量化精度。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../install_guide/install_guide.md)》。

## 原理和实现

### 原理

Ceil_X 算法针对传统 MXFP4 per-block 量化的大值截断问题，使用 **ceil** + **可配置除数** 重新设计 shared exponent 的计算方式。

**传统 floor 缩放的问题：**

传统方法计算 shared exponent：

$$s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$$

缩放后 block 内最大值的量级为：

$$\frac{\max(|x|)}{2^{s_{\text{floor}}}} = \frac{\max(|x|)}{2^{\lfloor \log_2(\max(|x|)) \rfloor - 2}} = 4 \cdot \frac{\max(|x|)}{2^{\lfloor \log_2(\max(|x|)) \rfloor}} \in [4, 8)$$

由于 $\max(|x|) / 2^{\lfloor \log_2(\max(|x|)) \rfloor} \in [1, 2)$，整体范围落在 $\,[4, 8)$。但 MXFP4 的表示上限为 6.0，$\,[6, 8)$ 区间内的值被截断，产生较大误差。

**Ceil_X 的改进：**

Ceil_X 引入除数 $c$ 并改用 ceil 操作。对于每个 block，缩放因子 $2^s$ 的计算方式为：

$$s = \text{ceil}\left(\log_2\left(\frac{\max(|x|)}{c}\right)\right)$$

缩放后 block 内最大值的量级为：

$$\frac{\max(|x|)}{2^s} = \frac{\max(|x|)}{2^{\text{ceil}(\log_2(\max(|x|)/c))}}$$

当 $\max(|x|) \in (c/2, c]$ 时，$\log_2(\max/c) \in (-1, 0]$，$\text{ceil}=0$，缩放后值为 $\max(|x|) \in (c/2, c]$；当 $\max(|x|) \in (c, 2c)$ 时，$\log_2(\max/c) \in (0, 1)$，$\text{ceil}=1$，缩放后值为 $\max(|x|)/2 \in (c/2, c)$。因此缩放后数值范围被压缩到 $(c/2, c]$。

默认 $c = 7.25$ 时，缩放后数值范围为 $(3.625, 7.25]$。相比传统方法的 $[4, 8)$：

- 下界从 $4$ 降至 $3.625$，为小值保留了更大的表示范围，提升小值量化精度
- $[6, 8)$ 区间仅剩 $(6, 7.25]$ 仍存在轻微截断，截断比例显著减小

**核心思想：**

1. **分块处理**：将权重矩阵沿指定轴按块大小 32 划分成独立的数据块。
2. **Ceil_X 缩放**：对每个数据块计算 shared exponent：
   $$s = \text{ceil}\left(\log_2\left(\frac{\max(|x|)}{c} + \epsilon\right)\right) - e_{\text{max}}$$
   其中 $c$ 是可配置的除数（ceil_x_value），$\epsilon = 9.6 \times 10^{-7}$ 为数值稳定项，$e_{\text{max}} = 2^{e_{\text{bits}}-1} = 2$。
3. **自适应搜索（可选）**：在 $\,[c_{\text{min}}, c_{\text{max}}]$ 范围内以步长 $c_{\text{step}}$ 搜索使 MSE 最小的除数 $c$：
   $$c^* = \arg\min_{c \in [c_{\text{min}}, c_{\text{max}}]} \sum_{\text{blocks}} \|x - \hat{x}(c)\|^2$$

### 实现

算法实现在 [`msmodelslim/core/quantizer/impl/ceil_x.py`](../../../../../msmodelslim/core/quantizer/impl/ceil_x.py) 中：

- 实现类：`MXWeightPerBlockCeilX`
- 注册的量化类型：`mxfp4_per_block_sym`
- 配置模型：`CeilXExtConfig`

**核心代码逻辑：**

```python
# 计算 per-block min/max
self.minmax_block_observer.update(weight_value, sync=False, shared_exp_axes=shared_exp_axes)
min_val, max_val = self.minmax_block_observer.get_min_max()

# 计算 ceil_x shared exponent
shared_exp = ceil(log2(max_val / ceil_x_value + 9.6e-7))
shared_exp = clip(shared_exp, -scale_emax - emax, scale_emax - emax)

# 量化
w_q_storage = quantize(QStorage(FLOAT, weight_value), q_param)
```

**enable_search 搜索策略：**

```python
# 在 [search_min, search_max] 内以 search_step 步长搜索最优 ceil_x_value
candidates = [search_min + i * search_step for i in range(num_steps)]
for value in candidates:
    q_param = ceil_x_qparam(..., ceil_x_value=value)
    recon = dequantize(quantize(weight, q_param), q_param).value
    mse = ((weight - recon) ** 2).mean().item()
    if mse < best_mse:
        best_mse, best_value = mse, value
```

## 适用要求

- **精度提升**：适用于对 mxFP4 量化精度有更高要求的场景，尤其是权重分布范围较大、floor 缩放导致步长过粗的模型层。
- **计算成本**：无搜索模式时计算量与标准 MXFP4 量化一致；启用 enable_search 时增加若干次前向量化评估。

## 功能介绍

### YAML配置示例

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_block"
          dtype: "mxfp4"
          symmetric: true
          method: "ceil_x"
          ext:
            ceil_x_value: 7.25        # 除数，取值范围 [6.0, 12.0]
            enable_search: false      # 是否启用 MSE 搜索
            search_min: 6.0           # 搜索范围下限
            search_max: 12.0          # 搜索范围上限
            search_step: 0.25         # 搜索步长
```

### YAML配置字段详解

#### qconfig.weight.ext (权重量化扩展参数)

| 参数名 | 作用 | 可选值 | 说明 | 默认值 |
|--------|------|--------|------|--------|
| `ceil_x_value` | 除数 $c$ | [6.0, 12.0] | 控制 shared exponent 的收紧程度 | 7.25 |
| `enable_search` | 是否启用 MSE 搜索 | `true`, `false` | 在搜索范围内寻找最优除数 | `false` |
| `search_min` | 搜索范围下限 | [6.0, 12.0] | MSE 搜索的起始值 | 6.0 |
| `search_max` | 搜索范围上限 | [6.0, 12.0] | MSE 搜索的结束值，须大于 search_min | 12.0 |
| `search_step` | 搜索步长 | > 0 | 搜索时的步进间隔 | 0.25 |

## 技术优势

### 与传统 mxFP4 量化的对比

| 特性 | 传统 mxFP4 量化 | Ceil_X 量化 |
|------|----------------|-------------|
| 缩放因子 | $2^{\lfloor \log_2(\max) \rfloor - 2}$ | $2^{\text{ceil}(\log_2(\max / c))}$ |
| 缩放后数值范围 | $[4, 8)$ | $(c/2, c] = (3.625, 7.25]$ |
| 大值截断 | $[6, 8)$ 区间被截断 | 仅 $(6, 7.25]$ 轻微截断，截断比例显著减小 |
| 小值表示范围 | 下界 $4$ | 下界 $3.625$，小值量化精度更高 |
| 自适应搜索 | 不支持 | 可选 MSE 最优搜索 |
| 计算复杂度 | O(N) | O(N)（无搜索）/ O(kN)（搜索时） |

## FAQ

### Ceil_X 名称的由来？

Ceil_X 来源于算法使用的两个核心操作：`ceil`（向上取整）和可配置除数 `x`（ceil_x_value）。与 floor 缩放相比，ceil 操作将缩放后数值范围从 $[4, 8)$ 压缩至 $(c/2, c]$，完全落在 MXFP4 的可表示区间 $[0, 6]$ 内，避免大值截断。

### 默认值 7.25 是如何确定的？

7.25 在 W4A4 MXFP4 对称量化中经多次经验验证得到。在该取值下，ceil 操作使缩放后数值范围从 $[4, 8)$ 压缩至 $(3.625, 7.25]$，完全落在 MXFP4 的表示范围内，消除大值截断误差。

### enable_search 模式搜索的是什么？

搜索在同一层权重上寻找使整体 MSE 最小的全局除数 $c$（非 per-block 独立搜索）。搜索结果存入独立字段，不会修改用户的原始配置值。
