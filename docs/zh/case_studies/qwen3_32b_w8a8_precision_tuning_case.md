# Qwen3-32B W8A8精度调优案例

## 概述

### 案例目标

**目标**：对Qwen3-32B模型进行W8A8量化，使量化模型相比浮点模型的精度损失控制在可控范围以内。

**初始状态**：使用全静态量化（per-channel/per-tensor）搭配Smooth Quant离群值抑制算法，量化后模型对话出现乱码，无法正常使用。

## 前期准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../getting_started/install_guide.md)。

## 调优过程

### 步骤1：确认精度问题可信

在开始调优前，首先排除环境干扰，确保问题真实存在：

- **推理引擎验证**：浮点模型在目标推理引擎上能正常复现原始精度。
- **测评结果检查**：量化模型测评输出存在明显异常（对话乱码），确认为量化精度问题。
- **确定波动范围**：AIME25评测数据集当前精度损失异常。

### 步骤2：调整离群值抑制算法（首要步骤）

初始配置使用Smooth Quant算法导致对话中出现乱码，按照调优策略，依次尝试不同的离群值抑制算法：

| 离群值抑制算法 | AIME25 精度（%） | 量化时间（s） | 备注 |
|---------------|-----------------|--------------|------|
| Smooth Quant | 对话乱码 | 326 | 初始配置，精度下降明显 |
| Iterative Smooth（对称/alpha:0.5） | 53.33 | 324 | 相比Smooth Quant有改善，但精度仍不足 |
| Iterative Smooth（非对称/alpha:0.5） | 63.33 | 305 | 非对称方案精度提升10%，符合预期 |
| Iterative Smooth（对称/alpha:0.9） | 66.67 | 319 | 调整alpha参数后精度进一步提升 |
| Flex Smooth Quant | 63.33 | 1380 | 精度与Iterative Smooth（非对称/alpha:0.5）相当，但所需时间更长 |

**调优结果**：

综合考虑精度和量化时间，最终选择 **Iterative Smooth（对称/alpha:0.9）** 算法，具体分析如下：

**1. 精度对比分析**：

- Iterative Smooth（对称/alpha:0.9）精度为66.67%，在对称方案中最高。
- 相比Iterative Smooth（对称/alpha:0.5）的53.33%，精度提升13.34个百分点。
- 虽然Iterative Smooth（非对称/alpha:0.5）精度为63.33%，但对称方案（alpha:0.9）精度为66.67%，已超过非对称方案。
- 相比Flex Smooth Quant的63.33%，精度提升3.34个百分点。

**2. 量化时间对比分析**：

- Iterative Smooth（对称/alpha:0.9）量化时间为319秒，与Iterative Smooth（对称/alpha:0.5）的324秒相当。
- 相比Flex Smooth Quant的1380秒，量化时间节省76.9%，效率明显提升。

**最终决策**：

综合以上分析，**Iterative Smooth（对称/alpha:0.9）** 是当前场景下的最佳选择。

### 步骤3：量化算法选择

在确定离群值抑制算法后，优化量化算法配置：

#### 量化方法对比

| 权重量化方法 | 激活量化粒度 | AIME25 精度（%） | 量化时间（s） | 备注 |
|-------------|-------------|----------------|-------------|------|
| minmax | per-tensor（静态量化） | 66.67 |319 | 基础配置（基于步骤2结果） |
| minmax | per-token（动态量化） |80.00 | 289 | 激活使用per-token后精度提升13.33个百分点 |
| ssz | per-tensor（静态量化） | 63.33 | 408 | 权重量化使用ssz方法，但静态量化精度下降 |
| ssz | per-token（动态量化） | 70.00 | 348 | ssz + per-token精度低于minmax+per-token |

#### 调优结果

综合考虑精度和量化时间，最终选择 **minmax + per-token（动态量化）** 配置，具体分析如下：

**1. 精度对比分析**：

- minmax + per-token（动态量化）精度为80.00%，相比minmax + per-tensor（静态量化）的66.67%，精度提升13.33个百分点。
- ssz + per-tensor（静态量化）精度为63.33%，相比minmax + per-tensor（静态量化）配置下降3.34个百分点，说明ssz方法在该INT8量化场景下效果不如minmax方法。
- ssz + per-token（动态量化）精度为70.00%，低于minmax + per-token（80.00%）10个百分点，说明ssz方法在该INT8动态量化场景下也不如minmax方法。

**2. 量化时间对比分析**：

- minmax + per-token量化时间为289秒，相比minmax + per-tensor（319秒）节省30秒（9.4%），量化效率提升。
- ssz + per-token量化时间为348秒，比minmax + per-token多59秒（20.4%），量化效率较低。

**3. 综合对比分析**：

- **精度方面**：minmax + per-token精度最高（80.00%），优于ssz + per-token（70.00%）和所有静态量化方案。
- **量化时间方面**：minmax + per-token量化时间最短（289秒），相比ssz + per-token（348秒）节省59秒（17.0%），相比minmax + per-tensor（319秒）也节省30秒。
- **方法复杂度**：minmax方法实现简单，计算速度快；ssz方法通过迭代搜索，计算更复杂，INT8量化优先选择minmax算法。

**最终决策**：
综合以上分析，**minmax + per-token（动态量化）** 在精度和量化时间两个方面达到最佳平衡。minmax + per-token不仅精度最高（80.00%），比ssz + per-token高10个百分点，而且量化时间最短（289秒），比ssz + per-token节省59秒，实现更简单，是当前场景下的最佳选择。相比步骤2的配置（66.67%），精度提升了13.33个百分点（相对提升20%），为后续调优奠定了良好基础

### 步骤4：校准集调整

步骤3达到80.00%精度后，已满足预设精度要求。为展示完整的调优流程并验证校准集调整的效果，本节在GPQA数据集上进行测试验证。GPQA数据集题目数量更多，能够更清晰地展现不同配置间的精度差异。本节以Iterative Smooth配合静态量化策略作为基准配置。

#### 校准集优化策略

校准集的质量直接影响量化参数的准确性：

| 调整策略 | 具体操作 | 校准集变化 | 优化目的 |
|---------|---------|-----------|---------|
| 初始校准集 | 10条随机样本 | 10条 | 建立基准配置 |
| 增加数据量 | 从10条增加到30条样本 | 10→30条 | 提升量化参数估计的准确性 |
| 匹配应用场景 | 使用中文对话数据替换随机数据 | 30条（中文对话） | 使校准数据更贴近实际应用场景 |
| 平衡数据分布 | 从GPQA、C-Eval、MMLU等多个数据集抽取样本混合 | 30条（多数据集混合） | 提升数据分布的多样性和均衡性 |
| 剔除异常数据 | 移除导致量化精度下降的3条异常样本 | 30→27条 | 减少异常样本对量化参数的干扰 |
| 加入badcase | 加入浮点模型在GPQA上的5个badcase样本 | 27→32条 | 帮助量化模型学习困难样本，提升精度 |

#### 调优过程

将AISBench测评结果中的badcase样本加入量化校准集，重新生成量化权重。具体操作如下：

1.**获取badcase样本**：从AISBench测评结果中提取少量badcase样本。例如，一个badcase样本为：

```text
What is the correct answer to this question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?

Choices:
(A)10^-11 eV
(B)10^-8 eV
(C)10^-9 eV
(D)10^-4 eV
Format your response as follows: "The correct answer is (insert answer here)"
```

2.**格式转换**：

   - **JSONL格式**：参考 `msmodelslim/lab_calib/mix_calib.jsonl`，将文本放在`"inputs_pretokenized"`字段后，格式如下：

     ```json
     {"inputs_pretokenized":"What is the correct answer to this question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\n\nChoices:\n(A)10^-11 eV\n(B)10^-8 eV\n\n(C)10^-9 eV\n(D)10^-4 eV\nFormat your response as follows: \"The correct answer is (insert answer here)\""}
     ```

   - **JSON格式**：参考 `msmodelslim/lab_calib/qwen3_cot_w4a4.json`，直接将文本加入字符串列表中即可。

3.**重新量化**：将调整后的校准集用于量化，重新生成量化权重。

#### 调优结果

| 量化策略 | GPQA 精度（%） | 备注 |
|-------------|---------------|------|
| Iterative Smooth + 静态量化 | 46.97 | 基准配置 |
| Iterative Smooth + 静态量化 + badcase调整校准集 | 55.56 | 相比基准配置精度提升8.59个百分点，说明badcase样本有助于量化模型学习困难样本，提升量化精度 |

### 步骤5：量化回退（备选方案）

量化回退是指将量化敏感层保持为原始浮点精度，以提升量化模型精度。当通过步骤1-4调整后精度仍无法满足要求时，可通过量化回退策略进一步优化。本节在GPQA数据集上验证量化回退的效果，展示完整的调优流程。

#### 使用场景

量化回退适用于以下情况：

- 通过步骤1-4调整后，精度仍无法满足精度要求。
- 需要在精度和性能之间寻求更精细的平衡。
- 某些特定层对量化极度敏感，需要保持高精度。

#### 调优过程

**1. 敏感层分析**

使用msModelSlim提供的敏感层分析工具识别量化敏感层。详细使用方法请参考《[量化敏感层分析使用指南](../feature_guide/sensitive_layer_analysis/usage.md)》。

执行分析命令：

```bash
msmodelslim analyze \
    --model_type Qwen3-32B \
    --model_path ${model_path}
```

根据量化敏感度得分从高到低排序，Top敏感层结果如下：

```text
layers.3.mlp.down_proj
layers.63.mlp.down_proj
layers.2.mlp.down_proj
layers.1.mlp.down_proj
layers.4.mlp.down_proj
layers.6.mlp.down_proj
layers.7.mlp.down_proj
layers.5.mlp.down_proj
layers.0.mlp.down_proj
layers.31.mlp.down_proj
layers.62.mlp.down_proj
layers.5.mlp.gate_proj
layers.5.mlp.up_proj
layers.32.mlp.down_proj
layers.8.mlp.gate_proj
layers.8.mlp.up_proj
layers.6.mlp.gate_proj
layers.6.mlp.up_proj
```

**分析结果**：`mlp.down_proj` 层敏感度排名靠前，是量化难度较大的层类型，应优先考虑回退。

**2. 修改量化配置**

在量化配置YAML中，通过`exclude`字段回退最为敏感的前9层（均为`mlp.down_proj`层）：

```yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "iter_smooth"
      alpha: 0.9
      scale_min: 1e-5
      symmetric: True
      enable_subgraph_type:
        - 'norm-linear'
        - 'linear-linear'
        - 'ov'
        - 'up-down'
      include:
        - "*"
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_tensor"
          dtype: "int8"
          symmetric: false
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"
      include: 
        - "*"
      exclude:
        - 'model.layers.3.mlp.down_proj'
        - 'model.layers.63.mlp.down_proj'
        - 'model.layers.2.mlp.down_proj'
        - 'model.layers.1.mlp.down_proj'
        - 'model.layers.4.mlp.down_proj'
        - 'model.layers.6.mlp.down_proj'
        - 'model.layers.7.mlp.down_proj'
        - 'model.layers.5.mlp.down_proj'
        - 'model.layers.0.mlp.down_proj'
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

**3. 重新生成量化权重**

使用修改后的配置重新进行量化，生成包含回退层的量化模型。

#### 调优结果

| 量化策略 | GPQA 精度（%） | 备注 |
|-------------|---------------|------|
| Iterative Smooth + 静态量化 | 46.97 | 基准配置 |
| Iterative Smooth + 静态量化 + 回退前9层 | 51.51 | 相比基准配置精度提升4.54个百分点，说明回退量化敏感层能有效提升量化精度，但会带来一定的性能开销和模型大小增加 |

## 最终配置总结

### 调优路径回顾

| 步骤 | 关键操作 | AIME25 精度（%） | 精度提升 | 备注 |
|------|---------|-----------------|---------|------|
| 初始状态 | Smooth Quant + minmax + 静态量化 | 乱码 | - | 初始配置，无法正常使用 |
| 步骤2 | Iterative Smooth（对称/alpha:0.9） | 66.67% | +66.67% | 离群值抑制算法优化，解决乱码问题 |
| 步骤3 | minmax + per-token（动态量化） | 80.00% | +13.33% | 激活量化粒度优化，达到精度要求 |

**说明**：步骤3达到80.00%精度后，已满足预设精度要求。步骤4和步骤5在GPQA数据集上进行验证，展示校准集调整和量化回退的调优效果。

### 最终配置

**离群值抑制算法**：Iterative Smooth（对称/alpha:0.9）

**量化配置**：

- **权重量化**：`minmax` 方法，`per_channel` 粒度，`int8` 数据类型，对称量化。
- **激活量化**：`minmax` 方法，`per_token` 粒度（动态量化），`int8` 数据类型，对称量化。

### 调优经验总结

1. **离群值抑制算法是关键**：从Smooth Quant切换到Iterative Smooth（对称/alpha:0.9），精度从乱码提升至66.67%，使量化模型具备可用性。

2. **激活量化粒度影响显著**：从per-tensor（静态量化）切换到per-token（动态量化），精度从66.67%提升至80.00%，提升13.33个百分点（相对提升20%），但需要注意动态量化可能带来一定的推理性能损失。

3. **量化方法选择很重要**：minmax方法在INT8量化场景下表现优于ssz方法，不仅精度更高（80.00% vs 70.00%），而且量化时间更短（289秒 vs 348秒），实现更简单，是当前场景下的最佳选择。

4. **校准集质量影响精度**：在GPQA数据集上验证，通过加入badcase样本优化校准集，精度从46.97%提升至55.56%，提升8.59个百分点，说明场景匹配数据和困难样本对量化精度提升有显著作用。

5. **量化回退是最后手段**：在GPQA数据集上验证，通过回退9个量化敏感层，精度从46.97%提升至51.51%，提升4.54个百分点。量化回退能有效提升精度，但会带来性能开销和模型大小增加，应在其他优化手段无法满足要求时使用。
