# GPTQ：权重量化算法说明

## 简介

- **问题**：传统量化方法（如MinMax）在权重分布不均匀时，量化误差较大，影响模型精度。
- **目标**：通过逐列优化方式，将当前列的量化误差在后续未量化列的权重中进行补偿，进而达到最小化整体量化误差，提升量化模型的精度。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 原理和实现

### 原理

通过逐层逐列对权重进行量化，根据量化误差和激活值的hessian矩阵对未量化的权重进行补偿，以此达到整体量化权重误差最小的目的。

**核心思想：**

1. **逐层量化**：对模型每一层独立量化，避免误差积累。
2. **基于二阶信息的误差修正**：利用Hessian矩阵评估权重量化对输出的影响，并动态调整未量化权重以补偿误差。
3. **分块量化**：将权重划分成多个块，减少计算复杂度。
4. **惰性批量更新**：在更新未量化权重时，延迟更新，将多个更新合并执行，减少计算开销。

### 实现

算法实现在 [`msmodelslim/core/quantizer/impl/gptq.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quantizer/impl/gptq.py) 中：

  - `per_channel`实现类：`WeightPerChannelGPTQ`
  - `per_group`实现类：`WeightPerGroupGPTQ`

## 适用要求

- **高精度需求**：适用于对精度要求较高的模型量化场景。
- **计算成本**：GPTQ算法需要计算激活值的hessian矩阵和矩阵分解，且对权重逐列量化，量化速度较慢，计算成本较大。
- **使用限制**：
    - 目前支持int8场景的 per_channel/per_group 对称与非对称量化。
    - 当前暂不支持int4场景的 per_channel/per_group 对称与非对称量化（后续将支持）。
    - 由于GPTQ量化算法依赖模型激活值，MOE模型量化场景要求校准集覆盖所有专家，对校准集要求较高，此场景不推荐使用。
    - per_tensor量化粒度暂不支持。
    - 权重必须为2D张量。

## 功能介绍

### YAML配置示例

作为Processor使用，YAML配置示例如下：

per_channel量化示例

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_channel"   # 量化范围
          dtype: "int8"          # 量化数据类型
          symmetric: true        # 是否对称量化
          method: "gptq"         # 量化算法-GPTQ
          ext: # 可选，扩展参数
            percdamp: 0.01       # 可选，阻尼系数，默认值0.01
            block_size: 128      # 可选，分块大小，默认值128
```

per_group量化示例

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_group"   # 量化范围
          dtype: "int8"        # 量化数据类型
          symmetric: true      # 是否对称量化
          method: "gptq"       # 量化算法-GPTQ
          ext: # 可选，扩展参数
            percdamp: 0.01     # 可选，阻尼系数，默认值0.01
            block_size: 128    # 可选，分块大小，默认值128
            group_size: 128    # 可选，分组大小，默认值128
```

### YAML配置字段详解

#### qconfig.weight (权重量化配置)

| 参数名       | 作用     | 可选值                            | 说明                                            | 默认值                         |
|-----------|--------|--------------------------------|-----------------------------------------------|-----------------------------|
| scope     | 量化范围   | `"per_channel"`, `"per_group"` | per_channel: 每个通道独立参数<br/>per_group: 每个分组独立参数 | `"per_channel"`             |
| dtype     | 量化数据类型 | `"int8"`                       | 8位整数量化                                        | `"int8"`                    |
| symmetric | 是否对称量化 | `true`, `false`                | true: 对称量化，零点为0<br/>false: 非对称量化，零点可调整        | `true`                      |
| method    | 量化方法   | `"gptq"`                       | gptq: gptq权重量化                                | `"gptq"`                    |
| ext       | 扩展配置   | `object`                       | 包含GPTQ特有的配置参数                                 | 见下方 **ext (GPTQ扩展配置)** 详细配置 |

**ext (GPTQ扩展配置)**

**作用**: 配置GPTQ算法特有的参数。

| 参数名        | 作用     | 类型      | 说明                                          | 示例值       |
|------------|--------|---------|---------------------------------------------|-----------|
| percdamp   | 阻尼系数   | `float` | percdamp 用于平滑梯度更新，减少量化引入的噪声对训练的影响           | 默认值`0.01` |
| block_size | 迭代分块大小 | `int`   | 迭代分块大小，必须能被待量化nn.Linear层的out_features维度整除   | 默认值`128`  |
| group_size | 量化分组大小 | `int`   | 量化分组大小，必须能被待量化nn.Linear层的input_features维度整除 | 默认值`128`  |

## FAQ

### GPTQ算法中percdamp、block_size和group_size三个超参数的含义和用途是什么？

- **percdamp**：
  - **含义**：阻尼百分比（damping percentage），用于在逆 Hessian 矩阵计算中加入一个小的对角阻尼，防止数值不稳定。
  - **用途**：GPTQ 在更新权重时需要求解一个线性方程组，该过程涉及 Hessian 矩阵的逆。当 Hessian 矩阵接近奇异时，直接求逆会导致数值爆炸。percdamp
    通过在 Hessian 对角线上添加一个微小量（通常为 max(diag(H)) * percdamp）来改善条件数，使得求逆稳定。
  - **典型值**：0.01（即 1% 的阻尼）。
  - **影响**：过大可能导致精度损失，过小可能引起数值不稳定。
- **block_size**：
  - **含义**：块大小，指 GPTQ 在一次迭代中同时处理的列数（即权重矩阵的列块）。
  - **用途**：GPTQ 是逐层、逐块进行量化的。block_size 决定了每次计算 Hessian
    逆并更新权重的列组大小。较大的块可以提高并行计算效率，但会占用更多显存；较小的块则更精细但可能减慢速度。
  - **典型值**：128。
  - **影响**：block_size 影响量化速度和内存占用，但对最终量化精度影响较小。
- **group_size**：
  - **含义**：分组大小，用于分组量化（group-wise quantization）。
  - **用途**：在 GPTQ 中，权重可以按组共享量化参数（scale 和 zero point）。group_size 指定每个组包含的连续元素个数。例如
    group_size=128 表示每128 个权重使用一组量化参数。较小的 group_size 能更好地适应权重的局部分布，提高量化精度，但会额外存储更多缩放因子，增加模型体积。
  - **典型值**：128。
  - **影响**：group_size 越小，量化精度越高，但模型文件会略大。通常 128 是精度与压缩率的较好平衡点。
