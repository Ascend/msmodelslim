# Expert Parallelism

> **Entry Category**: Other
>
> **English Name**: Expert Parallelism
>
> **Abbreviation**: EP
>
> **Chinese Aliases**: Expert Parallelism, MoE Expert Sharding
>
> **Application Domain**: Multi-device quantization of MoE models
>
> **msModelSlim Implementation**: [msmodelslim/utils/distributed/dist_helper.py](../../../../../msmodelslim/utils/distributed/dist_helper.py)

---

<a id="overview"></a>

## 1. Overview

Expert parallelism (Expert Parallelism, EP) is a parallel strategy for Mixture-of-Experts (MoE) structures: it shards the routed experts by rank so that each process holds only part of the expert weights, thereby reducing single-device memory usage. It solves the problem that the memory of a single device is insufficient to hold all the experts of an MoE model. Its core characteristic is that **local modules held only by the current rank** coexist with the **shared modules** held in common by every rank; quantization must distinguish the two—shared modules need cross-device synchronization, while for local experts collective communication must not be performed blindly. EP is often combined with [DP](../data_parallelism/term_data_parallelism.md), and `DistHelper` performs the topology classification.

---

## 2. Term Description

### 2.1 Background and Motivation

An MoE model sparsely activates several experts to greatly expand the model parameter scale and enhance the model's expressive power without significantly increasing the compute. If all experts were still replicated on every device as in pure DP, the memory capacity would be severely challenged. To address this, EP shards the experts so that each rank holds only part of the expert weights.

<a id="principle"></a>

### 2.2 Principles

<a id="1-core-idea"></a>

#### 2.2.1 Core Idea

- **Expert sharding**: the set of routed experts is divided into subsets by rank, and each rank loads only its expert subset.
- **Synchronization boundary**: the collective communication of the statistics / parameters required for quantization should use the shared modules as its boundary, and collective communication must not be performed blindly for routed experts held by only some ranks.

<a id="scenarios-and-limitations"></a>

### 2.3 Applicable Scenarios and Limitations

#### 2.3.1 Applicable Scenarios

- Multi-device quantization of MoE models.

#### 2.3.2 Usage Limitations

- It applies only to models with an MoE structure.
- The number of devices used (the divisor) must divide the number of experts (the dividend).

---

## 3. Related Workflows

- *[Expert Parallelism Usage Guide](expert_parallelism_guide.md)*: explains the shared / local module division under EP and the constraints on using DistHelper.
- *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*: the multi-device quantization entry point.

---

<a id="related-terms"></a>

## 4. Related Terms

- [Data Parallelism](../data_parallelism/term_data_parallelism.md): a prerequisite term; it provides the baseline paradigm of calibration data sharding and statistics reduction for shared modules.
- [Distributed Task Scheduler](../distributed_task_scheduler/term_distributed_task_scheduler.md): a companion term; when submitting parallel tasks, unsafe parallelism that depends on `local_only` must be avoided.

---

## 5. Reference Documents

1. *[Expert Parallelism Usage Guide](expert_parallelism_guide.md)*
