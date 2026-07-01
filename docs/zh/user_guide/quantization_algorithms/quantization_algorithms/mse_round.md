# MSE_Round：Per-Block MSE 最优 MXFP8 权重量化算法说明

## 1. 简介

- **问题**：MXFP8 per-block 量化中，传统 minmax 方法固定使用 $s = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$ 计算 shared exponent。缩放后 block 内最大值的量级为 $\max(|x|) / 2^s \in [2^{e_{\text{max}}}, 2^{e_{\text{max}}+1})$，即 $\,[8, 16)$（$e_{\text{max}} = 8$）。当 block 内最大值接近上界时，缩放结果可能超出 MXFP8 的表示上限（$448$），引发截断并引入较大量化误差。
- **目标**：对每个 block 在 **ceil** 与 **floor** 两档 shared exponent 之间，通过实际量化-反量化 MSE 比较，自适应选择误差更小的缩放方案，提升 MXFP8 权重量化精度。

## 2. 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../install_guide/install_guide.md)》。

## 3. 原理和实现

### 3.1 原理

MSE_Round 算法针对传统 MXFP8 per-block 量化中 floor 缩放可能导致的截断问题，在每个 block 内对 **ceil** 与 **floor** 两档 shared exponent 进行 MSE 比较，选取最优者。

**传统 floor 缩放：**

传统 minmax 方法计算 shared exponent：

$$s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$$

缩放后 block 内最大值的量级为：

$$\frac{\max(|x|)}{2^{s_{\text{floor}}}} = \frac{\max(|x|) \cdot 2^{e_{\text{max}}}}{2^{\lfloor \log_2(\max(|x|)) \rfloor}} \in [2^{e_{\text{max}}}, 2^{e_{\text{max}}+1}) = [8, 16)$$

由于 $\max(|x|) / 2^{\lfloor \log_2(\max(|x|)) \rfloor} \in [1, 2)$，整体范围落在 $\,[8, 16)$。MXFP8（E4M3 格式）的表示上限为 $2^{e_{\text{max}}} \times 1.75 = 448$，当缩放后最大值接近或超过该上限时，大值被截断，产生较大误差。

**MSE_Round 的改进：**

MSE_Round 为每个 block 同时计算 ceil 与 floor 两档候选 shared exponent：

$$s_{\text{ceil}} = \lceil \log_2(\max(|x|)) \rceil - e_{\text{max}}$$

$$s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$$

分别用两档参数完成量化-反量化，计算 block 内 MSE：

$$\text{MSE}_{\text{ceil}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i(s_{\text{ceil}}))^2, \quad \text{MSE}_{\text{floor}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i(s_{\text{floor}}))^2$$

最终 per-block 选择 MSE 更小的 shared exponent：

$$s^* = \begin{cases} s_{\text{ceil}} & \text{if } \text{MSE}_{\text{ceil}} < \text{MSE}_{\text{floor}} \\ s_{\text{floor}} & \text{otherwise} \end{cases}$$

ceil 档缩放后数值范围为 $(2^{e_{\text{max}}-1}, 2^{e_{\text{max}}}] = (4, 8]$，相比 floor 的 $\,[8, 16)$ 压缩了缩放幅度，在 block 最大值接近上界时有效避免大值截断；floor 档则在 block 分布较均匀时可能获得更优的整体 MSE。MSE_Round 通过 per-block 自适应选择，兼顾两种缩放策略的优势。

**核心思想：**

1. **分块处理**：将权重矩阵沿指定轴按块大小 32 划分成独立的数据块。
2. **双候选计算**：对每个 block 基于 $\max(|x|)$ 分别计算 ceil 与 floor 两档 shared exponent，其中 $e_{\text{max}} = 2^{e_{\text{bits}}-1} = 8$（MXFP8 E4M3 格式）。
3. **MSE 择优**：对每档候选执行完整的量化-反量化流程，计算 block 内 MSE，选取 MSE 更小的 shared exponent 作为最终量化参数。

### 3.2 实现

算法实现在 [`msmodelslim/core/quantizer/impl/mse_round.py`](../../../../../msmodelslim/core/quantizer/impl/mse_round.py) 中：

- 实现类：`MXWeightPerBlockMseRound`
- 注册的量化类型：`mxfp8_per_block_sym`

**核心代码逻辑：**

```python
# 计算 per-block max
minmax_block_observer.update(weight_value, sync=False, shared_exp_axes=shared_exp_axes)
_, max_val = minmax_block_observer.get_min_max()

# 计算 ceil / floor 两档候选 shared exponent
log2v = log2(max_val + FP32_MIN_NORMAL * (max_val == 0))
shared_exp_up = ceil(log2v) - emax      # ceil 档
shared_exp_down = floor(log2v) - emax   # floor 档

# 分别量化-反量化，计算 block 内 MSE
dequant_up = dequantize(quantize(weight, q_param_up), q_param_up)
dequant_down = dequantize(quantize(weight, q_param_down), q_param_down)
mse_up = (weight - dequant_up).pow(2).mean(dim=-1, keepdim=True)
mse_down = (weight - dequant_down).pow(2).mean(dim=-1, keepdim=True)

# 选取 MSE 更小的 shared exponent
shared_exp = select_by_mse(mse_up, mse_down, shared_exp_up, shared_exp_down)
```

**MSE 择优策略：**

```python
# 当 ceil 档 MSE 有效且更小时选 ceil，否则选 floor
select_up = valid_up & ((mse_up < mse_down) | ~valid_down)
shared_exp = where(select_up, shared_exp_up, shared_exp_down)
```

当某一候选的 shared exponent 超出 E8M0 表示范围（被标记为 NaN）时，自动回退至另一有效候选。

## 4. 适用要求

- **精度提升**：适用于对 MXFP8 权重量化精度有更高要求的场景，尤其是 block 内最大值分布不均、floor 缩放导致大值截断的模型层。
- **计算成本**：每个 block 需执行两次量化-反量化评估，计算量约为标准 minmax MXFP8 量化的 2 倍，但无需额外超参搜索，开销可控。

## 5. 功能介绍

### 5.1 YAML配置示例

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "mse_round"
```

W8A8 MXFP 混精场景下，可将权重设为 `mse_round`、激活保持 `minmax`，示例如下：

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "mse_round"
      include: ["*"]
```

## 6. 技术优势

### 6.1 与传统 MXFP8 量化的对比

| 特性 | 传统 minmax MXFP8 量化 | MSE_Round 量化 |
|------|----------------------|----------------|
| 缩放因子 | 固定 $2^{\lfloor \log_2(\max) \rfloor - e_{\text{max}}}$ | per-block 在 ceil / floor 两档间 MSE 择优 |
| 缩放后数值范围 | $[8, 16)$ | ceil 档 $(4, 8]$ 或 floor 档 $[8, 16)$，自适应选择 |
| 大值截断 | 接近上界时可能截断 | 优先选取避免截断的 ceil 档 |
| 自适应策略 | 不支持 | per-block MSE 最优选择 |
| 额外超参 | 无 | 无（零配置，开箱即用） |
| 计算复杂度 | O(N) | O(2N) |

### 6.2 与 Ceil_X（MXFP4）的对比

| 特性 | Ceil_X（MXFP4） | MSE_Round（MXFP8） |
|------|----------------|-------------------|
| 目标格式 | MXFP4 | MXFP8 |
| 改进方式 | ceil + 可配置除数 $c$ | per-block ceil/floor MSE 择优 |
| 搜索粒度 | 可选全局 MSE 搜索 $c$ | per-block 两档比较 |
| 典型场景 | W4A4 MXFP4 权重量化 | W8A8 MXFP8 权重量化 |

## 7. FAQ

### 7.1 MSE_Round 名称的由来？

MSE_Round 来源于算法的核心操作：通过 **MSE**（均方误差）比较，在 **ceil** 与 **floor** 两档 shared exponent 之间做出最优 **Round**（取舍）决策。与固定使用 floor 的 minmax 方法相比，MSE_Round 在每个 block 内自适应选择量化误差更小的缩放方案。

### 7.2 为什么 MXFP8 也需要改进缩放策略？

虽然 MXFP8 的表示范围（上限 448）远大于 MXFP4（上限 6.0），但传统 floor 缩放将 block 内最大值映射到 $\,[8, 16)$ 量级，在 block 最大值接近 $2^k$ 上界时，缩放结果仍可能超出 MXFP8 可表示范围并引发截断。MSE_Round 通过 per-block 比较 ceil 与 floor 两档的实际量化误差，在避免截断与保持量化精度之间取得更优平衡。

### 7.3 MSE_Round 与 Ceil_X 有什么区别？

两者分别针对 MXFP8 和 MXFP4 格式优化 shared exponent 计算策略。Ceil_X 引入可配置除数 $c$ 配合 ceil 操作压缩缩放范围，并支持全局 MSE 搜索最优 $c$；MSE_Round 则在每个 block 内对 ceil/floor 两档进行 MSE 比较，无需额外超参，开箱即用。在 Wan2.2 等模型的 W8A8 MXFP 量化实践中，MSE_Round 作为权重量化方法取得了良好的精度表现。

### 7.4 是否支持激活值量化？

当前 MSE_Round 仅注册于 `mxfp8_per_block_sym` 权重量化方案。激活值量化请继续使用 `minmax` 等已有方法。
