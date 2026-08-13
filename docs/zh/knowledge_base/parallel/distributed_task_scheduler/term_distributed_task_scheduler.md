# 分布式任务调度器

> **词条类别**：其他
>
> **英文名称**：Distributed Task Scheduler
>
> **英文缩写**：DTS
>
> **中文别名**：分布式任务调度
>
> **应用领域**：多卡量化算法子任务并行
>
> **msModelSlim 实现**：[msmodelslim/utils/distributed/task_scheduler/](../../../../../msmodelslim/utils/distributed/task_scheduler/)

---

<a id="overview"></a>

## 1. 概述

分布式任务调度器（Distributed Task Scheduler，DTS）是 msModelSlim 在多卡量化完备性之上引入的调度组件：把 Processor 内可独立执行的算法子任务放入共享队列，由各 rank **分工执行**，并在依赖满足时做跨卡同步。它解决的是仅做 [DP](../data_parallelism/term_data_parallelism.md) 时各卡仍重复跑同一串子任务的算力浪费问题。

---

## 2. 背景与动机

完成多卡完备性支持后，各 rank 已能通过统计归约得到正确的量化结果；但 Processor 内部的子任务链（例如子图平滑中的 T1→T2→T3→T4）通常仍由各 rank 完整重复执行。此时多卡主要缩短校准前向耗时，子任务阶段的墙钟时间仍接近单卡整链耗时。DTS 将可并行子任务从各卡隐式串行执行中分离出来，由不同 rank 分工计算（如一卡执行 T1、另一卡执行 T2），并在依赖满足后同步参数，从而提升计算密集型的量化算法的多卡执行效率。

---

<a id="principle"></a>

## 3. 原理

<a id="1-core-idea"></a>

### 3.1 核心思想

- **任务化**：将 Processor 内逻辑拆成原子化的子任务，并提交到共享的任务队列中。
- **单次执行**：共享队列保证同一逻辑任务只被一个 rank 执行，避免所有 rank 均重复执行相同的任务。

<a id="2-math"></a>

### 3.2 数学描述

#### 3.2.1 任务与依赖

设算法子任务集合为 $\mathcal{T}=\{T_1,\ldots,T_k\}$，依赖关系为偏序 $\prec$。串行基线耗时：

$$
T_{\mathrm{serial}} = \sum_{i=1}^{k} t(T_i)
\tag{1}
$$

其中：

- $T_i$：第 $i$ 个子任务
- $t(T_i)$：执行 $T_i$ 的墙钟耗时
- $T_{\mathrm{serial}}$：单卡执行的总耗时

#### 3.2.2 理想并行耗时

记同步开销为 $T_{\mathrm{sync}}$ 时，DTS 理想耗时可写为：

$$
T_{\mathrm{DTS}} \approx \max_{0 \le r < W} \sum_{T \in \mathcal{T}_r} t(T) + T_{\mathrm{sync}}
\tag{2}
$$

其中：

- $W$：卡数
- $\mathcal{T}_r$：调度器分配给 rank $r$ 的子任务子集（满足依赖偏序）
- $T_{\mathrm{sync}}$：跨 rank 参数 / 缓冲区同步开销

#### 3.2.3 语义等价约束

设 $S_0$ 为初始模型状态，$F_{T}$ 为任务 $T$ 对状态的更新函数。则要求对任意 rank $r$ 的终态 $S_r^{\mathrm{final}}$：

$$
S_r^{\mathrm{final}} = F_{T_k} \circ \cdots \circ F_{T_1}(S_0)
\tag{3}
$$

即与按串行顺序执行完整任务链的结果一致。

<a id="3-properties"></a>

### 3.3 关键性质

- **语义一致性**：分布式任务调度不改变量化算法的子任务之间的偏序关系，因而调度完成后量化状态与单卡串行执行完整任务链保持语义一致。
- **条件性加速**：DTS 并不总是带来正向收益，只有当算法子任务的计算复杂性较高时，任务并行收益大于卡间通信开销才能产生加速效果。

---

<a id="scenarios-and-limitations"></a>

## 4. 适用场景与限制

### 4.1 适用场景

- 包含计算密集型量化算法的多卡量化。

### 4.2 使用限制

- 量化算法需先完成 DP 支持。

---

## 5. 关联流程

- 《[分布式任务调度器使用指南](distributed_task_scheduler_guide.md)》：提供 DTS 的适配流程说明与案例参考。

---

<a id="related-terms"></a>

## 6. 关联词条

- [数据并行](../data_parallelism/term_data_parallelism.md)：前置术语，DTS 构建于 DP 基础之上。
- [专家并行](../expert_parallelism/term_expert_parallelism.md)：配套术语，对于 MoE 模型可与 EP 进行叠加。

---

## 7. 参考资料

1. 《[分布式任务调度器使用指南](distributed_task_scheduler_guide.md)》
