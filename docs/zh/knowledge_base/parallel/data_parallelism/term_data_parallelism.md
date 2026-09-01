# 数据并行 量化术语百科词条

> **词条类别**：其他
>
> **英文名称**：Data Parallelism
>
> **英文缩写**：DP
>
> **中文别名**：分布式数据并行、多卡数据并行
>
> **应用领域**：大模型多卡量化、校准数据并行加速
>
> **msModelSlim 实现**：[msmodelslim/core/runner/dp_layer_wise_runner.py](../../../../../msmodelslim/core/runner/dp_layer_wise_runner.py)

---

<a id="overview"></a>

## 1. 概述

数据并行（Data Parallelism，DP）是多卡量化中的基础并行范式：各 rank 持有同构、同值的模型副本，校准数据按 rank 切分，各卡仅对本分片做前向与统计，并通过集合通信使得各 rank 达到一致的量化状态。它解决的是单卡校准吞吐不足、超大校准集耗时过长的问题。

---

## 2. 背景与动机

大模型训练后量化（Post-Training Quantization, PTQ）依赖校准前向收集激活统计量。模型规模与校准样本量增大后，单卡串行校准成为耗时瓶颈。一条有效的解决途径是对校准数据集进行切分，复制多份模型副本同时进行前向校准。

---

<a id="principle"></a>

## 3. 原理

<a id="1-core-idea"></a>

### 3.1 核心思想

- **模型副本同构**：每个 rank 加载同一套模型结构与权重，避免因拓扑差异导致 rank 间无法对齐。
- **数据分片**：校准集按 rank 划分，前向校准并行执行，从而缩短前向校准的墙钟时间。
- **状态同步**：对量化所需的激活值的最大值、最小值等统计量进行集合通信，从而使得各 rank 的量化状态与单卡量化达到语义一致。

### 3.2 关键性质

- **语义等价性**：在正确地完成集合通信的基础上，多卡量化与单卡量化在语义上等价。

---

## 4. 流程示意

> 详细适配步骤与案例，请参阅《[数据并行机制使用指南](data_parallelism_guide.md)》。

```mermaid
flowchart LR
    A[校准集切分] --> B[各 rank 本地前向]
    B --> C[本地统计量]
    C --> D[集合通信]
    D --> E[一致量化参数写回]
```

---

<a id="scenarios-and-limitations"></a>

## 5. 适用场景与限制

### 5.1 适用场景

- 大规模校准集下的多卡 PTQ。

### 5.2 使用限制

- 单卡显存须至少能容纳单层模型权重。
- 量化中涉及的算法均须已实现分布式支持。

---

## 6. 关联流程

- 《[数据并行机制使用指南](data_parallelism_guide.md)》：说明 DP 完备性支持、统计量同步与适配步骤。
- 《[一键量化使用说明](../../../user_guide/usage_quick_quantization.md)》：通过 `--device_id` 参数触发多卡量化。

---

<a id="related-terms"></a>

## 7. 关联词条

- [专家并行](../expert_parallelism/term_expert_parallelism.md)：配套术语，MoE 模型量化场景下与 DP 叠加实现进一步加速并降低显存。
- [分布式任务调度器](../distributed_task_scheduler/term_distributed_task_scheduler.md)：后续术语，在 DP 基础上对计算密集型算法的执行进行调度优化以进一步提升量化速度。

---

## 8. 参考资料

1. 《[一键量化使用说明](../../../user_guide/usage_quick_quantization.md)》
