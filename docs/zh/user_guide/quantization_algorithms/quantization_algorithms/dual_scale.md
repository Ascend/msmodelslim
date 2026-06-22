# DualScale：w4a4量化方案说明

## 简介

- **背景**：传统的分组量化通常采用单尺度（single-scale）策略：每组K维元素共享一个缩放因子。例如，MXFP4格式中每32个元素使用一个FP8格式的共享指数。然而，激活值中存在结构化的异常通道（outlier
  channels）——某些通道的值平均比其他通道高出数个数量级。单尺度量化难以同时精准表示这些差异巨大的数值范围，导致精度损失。
- **核心思想**：通过两级粒度递进的缩放因子，在保持硬件高效性的同时提升4比特量化的精度。

## 算法原理

计算逻辑与核心公式如下：

### 1.权重激活值 (Activation) 的动态量化与反量化 (Fake Quantization)

#### a) 外层缩放 (Dual Scale)

将输入 X 按照 `x_dual_block_size` 划分，计算每个大块的最大绝对值，得到外层尺度 $S_{dual\_x}$:
$$X_{dualscaled} = \frac{X}{S_{dual\_x}}, \quad S_{dual\_x} = \frac{\max(|X_{block}|)}{{MXFP4\_MAX\_NORMAL}}$$

#### b) 内层量化与反量化 (Inner Quant-Dequant)

将 $X_{dualscaled}$ 进一步按 `x_inner_block_size` 划分，计算内层尺度并转换为目标低比特格式：
$$X_{q\_dq\_inner} = \text{mxfp4\_quantize\_dequantize}(X_{dualscaled}, S_{inner\_x})$$

#### c) 外层反量化 (Dual Scale Dequantization)

$$X_{q\_dq} = X_{q\_dq\_inner} \times S_{dual\_x}$$

### 2. 权重 (Weight) 的静态反量化 (Dequantization)

权重在初始化时已完成了量化存储，前向传播时仅进行两级反量化恢复至高精度：

#### a) 内层反量化 (Inner Dequantization)

根据内层参数 `inner_w_q_param` 恢复基础缩放：
$$W_{dualscaled\_q\_dq} = \text{dequantize}(W_{quantized}, S_{inner\_w})$$

#### b) 外层反量化 (Dual Scale Dequantization)

乘以模型初始化时固化的外层权重尺度 `weight_dual_scale` ($S_{dual\_w}$):
$$W_{q\_dq} = W_{dualscaled\_q\_dq} \times S_{dual\_w}$$

### 3. 矩阵乘法矩阵 (Linear Inverted)

$$\text{Output} = X_{q\_dq} \cdot W_{q\_dq}^T + \text{bias}$$

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../install_guide/install_guide.md)》。

## 适用要求

- **低比特量化**：适合极低比特量化场景中的4比特量化。
- **高精度需求**：在低比特条件下仍能保持较高的模型精度。
- **计算资源**：需要额外的优化过程，计算成本高于简单量化方法。
- **使用限制**：

    - 需要足够的校准数据或训练迭代次数来优化参数，由于涉及到迭代优化，量化时长相对其他方法相对较久。
    - 当前该方案主要面向Qwen3稠密系列模型（如Qwen3-8B/14B/32B）的低比特量化场景，不保证可泛化到其他系列模型。

## 功能介绍

### 昇腾AI处理器支持情况

| 产品系列                               | 支持 |
|------------------------------------|----|
| Atlas 350 加速卡    | ✓  |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品    | ✗  |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | ✗  |
| Atlas 推理系列产品                       | ✗  |

> [!NOTE]
>
>算法实现包含训练过程，对NPU显存有一定的要求，仅支持NPU显存>=64G的设备。

### YAML配置示例

作为Processor使用，YAML配置示例如下：

```yaml
 process:
   - type: "linear_quant"
     qconfig:
       act:
         scope: "dual_scale"
         dtype: "mxfp4"
         symmetric: True
         method: "dualscale"
         ext: {
           dual_block_size: 512
         }
       weight:
         scope: "dual_scale"
         dtype: "mxfp4"
         symmetric: True
         method: "dualscale"
         ext: {
           dual_block_size: 512
         }
```

### YAML配置字段详解

#### qconfig.weight (权重量化配置)

| 参数名             | 作用      | 可选值             | 说明                                     | 默认值           |
|-----------------|---------|-----------------|----------------------------------------|---------------|
| scope           | 量化范围    | `"dual_scale"`  | dual_scale: 双尺度                        | `"per_block"` |
| dtype           | 量化数据类型  | `"mxfp4"`       | mxFP4 格式量化                             | `"mxfp4"`     |
| symmetric       | 是否对称量化  | `true`, `false` | true: 对称量化，零点为0<br/>false: 非对称量化，零点可调整 | `true`        |
| method          | 量化方法    | `"dualscale"`   | dualscale: 二级量化算法                      | `"dualscale"` |
| dual_block_size | block大小 | `"int"`         | dual_block_size: block大小               | `512`         |
