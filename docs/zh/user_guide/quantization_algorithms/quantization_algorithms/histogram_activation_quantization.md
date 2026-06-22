# Histogram：直方图激活值量化算法说明

## 简介

- **来源**：修改自PyTorch的相关实现。
- **问题**：传统MinMax量化器容易受到离群值影响，导致量化范围过大，有效比特位利用率低，量化精度损失严重。
- **目标**：通过分析激活值的直方图分布，自动搜索最优的截断区间，过滤离群值，提高量化精度和模型性能。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 原理和实现

### 原理

直方图激活值量化算法的核心思想是通过分析输入张量的分布直方图，自动搜索最优的截断区间（clip_min, clip_max），以避免量化范围过大。

### 实现

- 算法在 [`msmodelslim/core/quantizer/impl/histogram.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quantizer/impl/histogram.py) 和 [`msmodelslim/core/observer/histogram.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/observer/histogram.py) 中实现，处理流程分4步。

1. **直方图统计**：
   - 将输入张量的值域划分为固定数量的bins（默认2048）。
   - 统计每个bin中数值的频次，构建分布直方图。
   - 支持上采样（upsample_rate=16）以减少量化误差。

2. **截断值搜索**：
   - 每次移动固定的百分位数（stepsize=1e-5），逐步调整截断区间。
   - 通过计算量化误差评估候选区间的质量，在量化误差不再减小或越界时停止搜索。

3. **量化误差度量**：
   - **L2范数误差**：默认量化误差，计算量化前后分布的L2范数差异。
   - **KL散度误差**：计算量化前后分布的KL散度，目前精度性能低于L2范数方法，通过一键量化yaml配置时，暂无入口。

4. **量化参数计算**：
   - 以最优截断区间的上下界为max/min，计算并保存scale和zero_point。
   - 执行伪量化操作，返回量化后的张量。

### 核心组件

#### 直方图观察器（HistogramObserver）

```python
class HistogramObserver(TorchHistogramObserver):
    def __init__(self, config: HistogramObserverConfig):
        super().__init__(qscheme=torch.per_tensor_affine)
        self.config = config
        self.clip_min = None
        self.clip_max = None
        self.upsample_rate = 16  # 上采样率，减少量化误差
```

**核心方法实现**

1. **forward方法**：

   ```python
   # 继承自TorchHistogramObserver的forward方法
   # 用于更新直方图统计信息
   # 在update方法中被调用
   ```

2. **update方法**：

   ```python
   def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):
       """
       更新直方图，并进行截断值搜索，保存最佳的量化截断值

       主要步骤：
       1. 输入验证：检查张量有效性，过滤NaN和无穷值
       2. 直方图更新：调用父类forward方法更新直方图统计
       3. 参数搜索：执行非线性参数搜索，找到最优截断区间

       Args:
            x: 输入张量
            sync: 是否同步
            group: 进程组

       Returns:
            None
       """
   ```

3. **内部搜索方法实现**：

   **L2范数搜索**：

   ```python
   def _compute_l2_error(self, start_bin: int, end_bin: int):
       """
       计算量化前后的L2范数误差

       算法原理：
       1. 计算目标bin宽度
       2. 计算源bin到目标bin的映射关系
       3. 将误差分解为起始、中间、结束三个部分
       4. 通过_get_norm方法（显式计算积分）计算各部分L2范数误差
       """
   ```

   **KL散度搜索**：

   ```python
   def _compute_kl_error(self, start_bin: int, end_bin: int):
       """
       计算量化前后的KL散度

       算法原理：
       1. 计算真实分布p_i
       2. 计算量化后的分布q_i
       3. 计算KL散度：KL = sum(p_i * log(p_i / q_i))
       """
   ```

4. **非线性参数搜索**：

   ```python
   def _non_linear_param_search(self) -> tuple[torch.Tensor, torch.Tensor]:
       """
       采用二分搜索策略寻找最优截断区间

       搜索策略：
       1. 初始化：alpha=0.0（下界），beta=1.0（上界）
       2. 迭代搜索：每次移动固定百分位数（stepsize=1e-5）
       3. 边界调整：根据移动步长，决定移动左边界还是右边界：选择单次移动长度更长（即分布更稀疏）的一边
       4. 早停条件：量化误差不再改善或边界越界时停止
       5. 返回结果：最优的start_bin和end_bin对应的截断值
       """
   ```

#### 直方图量化器（ActPerTensorHistogram）

```python
class ActPerTensorHistogram(AutoActQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        histogram_config = HistogramObserverConfig(symmetric=config.symmetric)
        self.histogram_observer = HistogramObserver(histogram_config)
        self.q_param: Optional[QParam] = None
```

**核心方法实现**

1. **forward方法**：

   ```python
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       """
       前向传播方法，执行量化计算
       """
       # 更新直方图观察器，统计输入张量的分布信息
       self.histogram_observer.update(x)
       # 获取基于直方图统计的最佳截断值
       # clip_min: 最小截断值，clip_max: 最大截断值
       clip_min, clip_max = self.histogram_observer.get_clip_bounds()
       # 根据截断值计算量化参数
       self.q_param = calculate_qparam(
           min_val=clip_min,      # 最小截断值
           max_val=clip_max,      # 最大截断值
           q_dtype=QDType(self.config.dtype),
           q_scope=QScope(self.config.scope),
           symmetric=self.config.symmetric,
       )
       # 执行伪量化操作
       return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value.clamp(clip_min, clip_max)
   ```

2. **量化参数管理**：

   ```python
   def get_q_param(self) -> QParam:
       """
       获取计算得到的量化参数
       """
       if self.q_param is None:
          raise SpecError(
                  "No q_param was set",
                  action="Please call forward first"
              )
       return self.q_param
   ```

## 功能介绍

### YAML配置示例

Histogram 作为 [`linear_quant`](linear_quant.md) 处理器中激活值量化方法（`method: "histogram"`）使用，YAML配置示例如下：

```yaml
spec:
  process:
  - type: "linear_quant"
    qconfig:
      act:
        scope: "per_tensor"  # 目前只支持per_tensor
        dtype: "int8"        # 目前只支持int8
        symmetric: false     # 支持对称/非对称量化，取值分别为True/False
        method: "histogram"  # 配置为"histogram", 即启用直方图激活值量化
      weight:
        scope: "per_channel"
        dtype: "int8"
        symmetric: true
        method: "minmax"     # 不支持直方图权重量化，此处不应配置为"histogram"
```

### YAML配置字段详解

#### qconfig.act（激活值量化配置）

以下字段均属于 `linear_quant` 处理器配置，完整的 `linear_quant` 字段说明请参考[线性量化配置字段详解](linear_quant.md#yaml配置字段详解)。

| 参数名 | 作用 | 可选值 | 说明 |
|--------|-------|--------|------|
| scope | 量化范围 | `"per_tensor"` | 直方图量化目前仅支持 per_tensor |
| dtype | 量化数据类型 | `"int8"` | 直方图量化目前仅支持 int8 |
| symmetric | 是否对称量化 | `true`, `false` | true: 对称量化；false: 非对称量化 |
| method | 量化方法 | `"histogram"` | 固定值，启用直方图激活值量化 |

### 算法参数

**HistogramObserverConfig**

目前由量化器自行配置，用户不需要调整。

```python
class HistogramObserverConfig(BaseModel):
    symmetric: bool = False                             # 是否对称量化
    search_method: SearchMethod = SearchMethod.L2_NORM  # 搜索方法
    dtype: QDType = QDType.INT8                         # 量化数据类型
    scope: QScope = QScope.PER_TENSOR                   # 量化范围
```

**搜索方法枚举**

```python
class SearchMethod(str, Enum):
    L2_NORM = "l2_norm"              # L2范数搜索
    KL_DIVERGENCE = "kl_divergence"  # KL散度搜索
```

## FAQ

### 配置错误

**现象**：日志提示中，出现ValidationError。

**可能原因**：

- 在支持激活值量化的场景中将histogram方法错误配置到了weight处。
- 在支持激活值量化的场景中选择了histogram不支持的配置，如int4量化。
- 在不支持激活值量化的场景中配置了histogram方法。

**解决方案**：

- 排查yaml是否配置错误。

```yaml
- type: "linear_quant"
  qconfig:
   act:
     scope: "per_tensor" # 目前只支持per_tensor
     dtype: "int8" # 目前只支持int8
     symmetric: False # 支持对称/非对称量化，取值分别为True/False
     method: "histogram" # 配置为"histogram", 即启用直方图激活值量化
   weight:
     scope: "per_channel"
     dtype: "int8"
     symmetric: True
     method: "minmax" # 不支持直方图权重量化，此处不应配置为"histogram"
```

- 排查对应的quantizer在初始化时是否存在AutoActQuantizer。可以根据 YAML 中 `process` 列表项的 `type` 字段查找对应的量化器名称，在 [`msmodelslim/core/quantizer`](https://gitcode.com/Ascend/msmodelslim/tree/master/msmodelslim/core/quantizer) 中查看对应的代码。

```python
class LinearQuantizer(nn.Module):

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self, config: LinearQConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.input_quantizer = AutoActQuantizer.from_config(config.act)  # 支持激活值量化
        self.weight_quantizer = AutoWeightQuantizer.from_config(config.weight)
        self.weight: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None
        self.q_weight: Optional[QStorage] = None
```
