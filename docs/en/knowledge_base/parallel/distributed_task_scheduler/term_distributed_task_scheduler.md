# Distributed Task Scheduler

> **Entry Category**: Other
>
> **English Name**: Distributed Task Scheduler
>
> **Abbreviation**: DTS
>
> **Chinese Aliases**: Distributed Task Scheduling
>
> **Application Domain**: Subtask parallelism of multi-device quantization algorithms
>
> **msModelSlim Implementation**: [msmodelslim/utils/distributed/task_scheduler/](../../../../../msmodelslim/utils/distributed/task_scheduler/)

---

<a id="overview"></a>

## 1. Overview

The distributed task scheduler (Distributed Task Scheduler, DTS) is a scheduling component that msModelSlim introduces on top of the completeness of multi-device quantization: it puts the algorithm subtasks that can be executed independently inside a Processor into a shared queue, has the ranks **execute them cooperatively**, and performs cross-device synchronization once the dependencies are satisfied. It solves the waste of compute that occurs when only [DP](../data_parallelism/term_data_parallelism.md) is used and every device still reruns the same chain of subtasks.

---

## 2. Background and Motivation

After the multi-device completeness support is in place, every rank can already obtain correct quantization results through statistics reduction; however, the subtask chain inside a Processor (for example T1→T2→T3→T4 in subgraph smoothing) is usually still executed in full and repeatedly by every rank. In that case multiple devices mainly shorten the calibration forward pass, and the wall-clock time of the subtask stage is still close to the whole-chain time on a single device. DTS separates the parallelizable subtasks from the implicit serial execution on each device, has different ranks compute them cooperatively (for example one device runs T1 and another runs T2), and synchronizes the parameters once the dependencies are satisfied, thereby improving the multi-device execution efficiency of compute-intensive quantization algorithms.

---

<a id="principle"></a>

## 3. Principles

<a id="1-core-idea"></a>

### 3.1 Core Idea

- **Taskification**: split the logic inside the Processor into atomic subtasks and submit them to a shared task queue.
- **Single execution**: the shared queue guarantees that the same logical task is executed by only one rank, avoiding every rank executing the same task repeatedly.

<a id="2-math"></a>

### 3.2 Mathematical Description

#### 3.2.1 Tasks and Dependencies

Let the set of algorithm subtasks be $\mathcal{T}=\{T_1,\ldots,T_k\}$ and the dependencies be the partial order $\prec$. The serial baseline time is:

$$
T_{\mathrm{serial}} = \sum_{i=1}^{k} t(T_i)
$$

where:

- $T_i$: the $i$-th subtask
- $t(T_i)$: the wall-clock time to execute $T_i$
- $T_{\mathrm{serial}}$: the total execution time on a single device

#### 3.2.2 Ideal Parallel Time

Let the synchronization overhead be $T_{\mathrm{sync}}$; then the ideal DTS time can be written as:

$$
T_{\mathrm{DTS}} \approx \max_{0 \le r < W} \sum_{T \in \mathcal{T}_r} t(T) + T_{\mathrm{sync}}
$$

where:

- $W$: the number of devices
- $\mathcal{T}_r$: the subset of subtasks that the scheduler assigns to rank $r$ (satisfying the dependency partial order)
- $T_{\mathrm{sync}}$: the cross-rank parameter / buffer synchronization overhead

#### 3.2.3 Semantic Equivalence Constraint

Let $S_0$ be the initial model state and $F_{T}$ the update function that task $T$ applies to the state. Then for any rank $r$, the final state $S_r^{\mathrm{final}}$ must satisfy:

$$
S_r^{\mathrm{final}} = F_{T_k} \circ \cdots \circ F_{T_1}(S_0)
$$

that is, it must be consistent with the result of executing the complete task chain in serial order.

<a id="3-properties"></a>

### 3.3 Key Properties

- **Semantic consistency**: distributed task scheduling does not change the partial order among the algorithm's subtasks, so after scheduling the quantization state remains semantically consistent with executing the complete task chain serially on a single device.
- **Conditional speedup**: DTS does not always bring a positive gain; a speedup occurs only when the algorithm's subtasks are computationally complex enough that the task-parallel gain exceeds the inter-device communication overhead.

---

<a id="scenarios-and-limitations"></a>

## 4. Applicable Scenarios and Limitations

### 4.1 Applicable Scenarios

- Multi-device quantization that includes compute-intensive quantization algorithms.

### 4.2 Usage Limitations

- The quantization algorithm must first implement DP support.

---

## 5. Related Workflows

- *[Distributed Task Scheduler Usage Guide](distributed_task_scheduler_guide.md)*: provides the adaptation workflow description and case references for DTS.

---

<a id="related-terms"></a>

## 6. Related Terms

- [Data Parallelism](../data_parallelism/term_data_parallelism.md): a prerequisite term; DTS is built on top of DP.
- [Expert Parallelism](../expert_parallelism/term_expert_parallelism.md): a companion term; for MoE models it can be combined with EP.

---

## 7. Reference Documents

1. *[Distributed Task Scheduler Usage Guide](distributed_task_scheduler_guide.md)*
