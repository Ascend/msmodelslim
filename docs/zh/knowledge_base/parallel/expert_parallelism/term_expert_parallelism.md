# 专家并行

> **词条类别**：其他
>
> **英文名称**：Expert Parallelism
>
> **英文缩写**：EP
>
> **中文别名**：专家并行、MoE 专家分片
>
> **应用领域**：MoE 模型多卡量化
>
> **msModelSlim 实现**：[msmodelslim/utils/distributed/dist_helper.py](../../../../../msmodelslim/utils/distributed/dist_helper.py)

---

<a id="overview"></a>

## 1. 概述

专家并行（Expert Parallelism，EP）是面向 Mixture-of-Experts（MoE）结构的并行策略：将路由专家按 rank 切分，使每个进程只持有一部分专家权重，从而降低单卡显存占用。它解决的是单卡显存不足以容纳 MoE 全量专家的问题。核心特征是同时存在**仅由本 rank 持有的局部模块**与各 rank 共同持有的**共享模块**；量化时必须区分二者——共享模块需跨卡同步，对局部专家则应避免盲目进行集合通信。EP 常与 [DP](../data_parallelism/term_data_parallelism.md) 叠加，并由 `DistHelper` 完成拓扑分类。

---

## 2. 背景与动机

MoE 模型通过稀疏激活多个专家，在不显著增加计算量的前提下大幅扩展模型参数规模以增强模型表达能力。若仍按纯 DP 的并行方式在每张卡复制全部专家，则显存容量受到极大挑战。为应对这一问题，EP 对专家进行切分，每个 rank 上仅持有部分专家权重。

---

<a id="principle"></a>

## 3. 原理

<a id="1-core-idea"></a>

### 3.1 核心思想

- **专家分片**：路由专家集合按 rank 进行切分子集，各 rank 仅加载专家子集。
- **同步边界**：量化所需的统计量 / 参数的集合通信应以共享模块为边界；对仅有局部 rank 持有的路由专家应避免盲目执行集合通信。

---

<a id="scenarios-and-limitations"></a>

## 4. 适用场景与限制

### 4.1 适用场景

- MoE 模型的多卡量化。

### 4.2 使用限制

- 仅适用于 MoE 结构的模型。
- 使用的卡数（除数）须能整除专家数（被除数）。

---

## 5. 关联流程

- 《[专家并行机制使用指南](expert_parallelism_guide.md)》：阐述 EP 下共享 / 局部模块划分及 DistHelper 使用约束。
- 《[一键量化使用说明](../../../user_guide/usage_quick_quantization.md)》：多卡量化入口；MoE 模型在多卡下可能同时启用 DP 与 EP。

---

<a id="related-terms"></a>

## 6. 关联词条

- [数据并行](../data_parallelism/term_data_parallelism.md)：配套术语，提供校准数据分片与共享模块统计归约的基线范式。
- [分布式任务调度器](../distributed_task_scheduler/term_distributed_task_scheduler.md)：配套术语，提交并行任务时须避开含 local_only 依赖的不安全并行。

---

## 7. 参考资料

1. 《[专家并行机制使用指南](expert_parallelism_guide.md)》
