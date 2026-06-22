# 浮点稀疏：基于 ADMM （Alternating Direction Method of Multipliers，交替方向乘子法）的模型稀疏化算法说明

## 简介

- **问题**：现有W8A8S量化方法虽然支持权重稀疏量化，但稀疏率较低，在 Atlas 300I Duo 推理卡压缩单元上难以实现理想的压缩效果。此外，为满足精度要求通常需要回退部分网络层，这显著降低了模型的推理性能。因此，我们提出对浮点权重进行稀疏化处理，结合硬件压缩单元实现更高的压缩率，在保证模型精度的同时显著提升推理性能。
- **目标**：通过ADMM（交替方向乘子法）算法实现模型浮点稀疏化，结合L2量化保持重要位置的精度，在保证模型性能的同时实现高压缩率。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 原理和实现

### 原理

**核心思想：**

1. **ADMM优化**：使用交替方向乘子法求解带约束的优化问题，找到最优的权重稀疏模式。
2. **激活统计**：通过前向hook收集激活统计信息，构建Hessian矩阵。
3. **迭代稀疏**：通过多次迭代逐步优化稀疏模式，平衡稀疏率和模型精度。
4. **精度保护**：使用L2量化保持重要位置的精度，避免关键权重被过度压缩。

**算法流程：**

1. **预处理阶段**：安装前向hook，收集激活统计信息，构建Hessian矩阵。
2. **ADMM稀疏化**：使用ADMM算法求解最优稀疏模式。
3. **迭代优化**：通过多次迭代优化稀疏结果。
4. **精度保护**：识别重要权重位置，应用L2量化保持精度。
5. **模块部署**：将稀疏化后的模块转换为量化模块。

### 实现

#### 代码实现

- 算法在 [msmodelslim/processor/sparse/float_sparse.py](../../../../../msmodelslim/processor/sparse/float_sparse.py) 和 [admm.py](../../../../../msmodelslim/processor/sparse/admm.py) 中实现：

**ADMM稀疏器核心类**

```python
class AdmmPruner:
    def __init__(self, layer: nn.Linear):
        ...

    def add_batch(self, inp: torch.Tensor):
        ...

    def fasterprune(self, sparse_ratio: float):
        ...

    def free(self):
        ...
```

**浮点稀疏处理器**

```python
class FloatSparseProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter):
        ...

    def preprocess(self, request):
        ...

    def postprocess(self, request):
        ...
```

#### 核心算法步骤

1. **统计信息收集**：
   - 安装前向hook收集输入激活数据。
   - 累积Hessian矩阵：`H += X^T * X`。
   - 计算行缩放因子：`scaler_row += ||X_i||_2^2 / n_samples`。

2. **ADMM稀疏化**：
   - 归一化Hessian矩阵和权重。
   - 设置初始惩罚参数：`rho0 = PERCDAMP * mean(diag(H))`。
   - 计算Hessian逆矩阵。
   - 执行ADMM主循环：
     - 投影到稀疏空间：`sparse_weights = (weights + lambda) * mask`。
     - 更新拉格朗日乘子：`lambda += (weights - sparse_weights)`。
     - 更新权重：`weights = H_inv * (H*weights + rho*(sparse_weights - lambda))`。

3. **精度保护**：
   - 使用量化误差和缩放因子的乘积作为重要性度量。
   - 选择top-k%的重要位置保持精度。
   - 应用L2量化：保持重要位置精度，其他位置进行量化。

## 适用要求

- **高压缩需求**：适用于需要高压缩率的模型部署场景。
- **精度敏感**：通过精度保护机制，在压缩的同时保持关键权重精度。
- **计算成本**：ADMM算法需要多次迭代，计算成本较高，速度较慢。
- **内存需求**：需要存储Hessian矩阵和激活统计信息，显存占用较高。
- **使用限制**：
  - 当前算法生成的权重需要在 Atlas 300I Duo 推理卡上利用硬件压缩单元进行进一步压缩，才能有效减小模型大小，降低模型部署时的显存占用，并获得性能提升。
  - 由于 Atlas 300I Duo 推理卡 不支持 bfloat 数据类型，因此对模型进行浮点稀疏时，需要手动将模型路径下的 config.json 中的 `torch_dtype` 字段修改成 float16。
  - 仅支持 `v1 框架` 中的 `逐层量化`。
  - 目前仅支持 `nn.Linear` 模块进行浮点稀疏。
  - 需要校准数据集收集激活统计信息，校准数据的 token id 个数 >= 2048。
  - 稀疏比例建议在 `0.3` 附近逐步调整。

## 功能介绍

### YAML配置示例

作为Processor使用，YAML配置示例如下：

```yaml
spec:
  process:
    - type: "float_sparse"
      sparse_ratio: 0.3          # 稀疏比例，取值范围为 0.0~1.0，默认0.3。
      include: [ "*" ]           # 包含的层，支持通配符。
      exclude: ["*self_attn*"]   # 排除的层，支持通配符。
```

### YAML配置字段详解

| 字段名 | 作用 | 数据类型 | 默认值 | 说明 |
|--------|------|----------|--------|------|
| type | 处理器类型标识 | string | - | 固定值"float_sparse"，用于标识该对象为浮点稀疏量化处理器。 |
| sparse_ratio | 稀疏比例 | float | 0.3 | 稀疏比例，取值范围为 0.0~1.0，默认0.3。 |
| include | 包含的层 | array[string] | ["*"] | 支持通配符匹配，指定要执行浮点稀疏量化的层。 |
| exclude | 排除的层 | array[string] | [] | 支持通配符匹配，优先级高于include。 |

### 算法参数

浮点稀疏算法内部使用以下参数（可通过修改源码调整，[msmodelslim/processor/sparse/admm.py](../../../../../msmodelslim/processor/sparse/admm.py)）：

```python
# ADMM参数
KEEP_BITS = 2                    # 保持精度的位数
KEEP_PROPORTION = 0.02          # 保持精度的比例：2%
PERCDAMP = 0.1                  # 阻尼系数
ITERATIVE_PRUNE = 15            # 迭代稀疏次数
ITERS = 20                      # ADMM最大迭代次数
```

## 性能特点

### 优势

1. **高压缩率**：通过ADMM算法实现高稀疏率，压缩效果显著。
2. **精度保护**：智能识别重要权重位置，避免关键信息丢失。
3. **自适应优化**：基于激活统计信息自行调整稀疏策略。
4. **逐层量化**：支持逐层量化，降低内存占用。

### 局限性

1. **计算开销**：ADMM迭代和Hessian矩阵计算增加模型稀疏时间。
2. **显存占用**：需要存储额外的统计信息和中间结果。
3. **参数调优**：稀疏比例等参数需要根据具体模型调整。

## FAQ

### 浮点稀疏不支持叠加 w8a8 稀疏量化

**现象**：用户尝试对已经进行W8A8S（权重INT8稀疏量化）处理的模型，再应用浮点稀疏算法进行进一步稀疏化。

**解决方案**：浮点稀疏算法（W16A16S）和W8A8S稀疏量化是两种不同的技术路径，不支持叠加使用。多次稀疏化处理会累积精度损失，可能严重影响模型性能。

### 稀疏比例设置过高

**现象**：稀疏比例过高导致模型精度严重下降。

**解决方案**：降低 sparse_ratio 配置参数，建议在 0.3 附近逐步调整。

### 校准数据长度不够，导致求矩阵逆失败

**现象**：处理大模型时出现求矩阵逆失败错误。

**解决方案**：增加校准集中每条数据长度，保证经过 tokenizer 编码后的 token id 数量 >= 2048。

### 校准集数量过多导致显存溢出

**现象**：处理大模型时出现显存溢出错误。

**解决方案**：减少校准集数量，或使用单卡显存更大的机器进行浮点稀疏。
