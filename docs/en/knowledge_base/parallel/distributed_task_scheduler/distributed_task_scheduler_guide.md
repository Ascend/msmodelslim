# Distributed Task Scheduler Usage Guide

## 1. Scope

This workflow applies to further parallelizing compute-intensive algorithm tasks on shared modules on top of DP support, thereby improving the efficiency of multi-device quantization.

## 2. Process Relationships and Prerequisites

**Parent process**: complete the algorithm's DP support according to the [Data Parallelism Usage Guide](../data_parallelism/data_parallelism_guide.md).

**Prerequisites**:

- Multi-device quantization runs correctly.
- The semantics of [Expert Parallelism](../expert_parallelism/term_expert_parallelism.md) have been understood: a task that depends on a `local_only` module cannot be split across ranks.

**Follow-up operations**: verify that the DTS result is semantically equivalent to the result without integration, and record the speedup ratio.

## 3. Inputs and Deliverables

| Type | Name | Source or Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Quantization algorithm that already supports DP | `msmodelslim/processor/` | `support_distributed()` returns `True` and the multi-device result is semantically equivalent to the single-device result | The multi-device quantization artifact matches the single-device one |
| Deliverable | Processor code integrated with DTS | `msmodelslim/processor/` | The algorithm can be split into atomic subtasks | Multi-device execution succeeds and the result is semantically equivalent to the result without integration |

## 4. Process Overview

```mermaid
flowchart LR
  A[Confirm the integration prerequisites] --> B[Encapsulate the subtask function]
  B --> C[Configure the synchronization mechanism]
  C --> D[Build the shared task queue]
  D --> E[Execute the scheduling]
```

## 5. Procedure

### Step 1: Confirm the Integration Prerequisites

**Objective**: confirm that the algorithm satisfies the prerequisites for integrating DTS, avoiding deadlocks or semantic deviations after integration.

**Inputs**: the algorithm's multi-device implementation; the subtask logic.

**Action**:

1. **Confirm DP support**: the multi-device quantization artifact is semantically equivalent to the single-device result (if this is not yet done, first complete support according to the [Data Parallelism Usage Guide](../data_parallelism/data_parallelism_guide.md)).
2. **Identify the splittable tasks and check the constraints**: a subtask must be executable independently by any single rank and must produce a consistent result.

**Output**: a list of splittable tasks (each annotated with the dependent module path and whether splitting across ranks is allowed).

**Pass conditions**: every item in the list can be executed independently by a single rank.

### Step 2: Encapsulate the Subtask Function

**Objective**: encapsulate the splittable logic into a Processor method that can be invoked by `submit`.

**Inputs**: the task list from step 1.

**Action**:

1. Encapsulate the task logic as a Processor instance method, keeping the signature simple (for example `def _worker_fn(self, idx) -> None`).
2. **Business parameters may only be passed through `args` / `kwargs` as serializable values**, and must not be the second positional parameter of `submit`. Common patterns:
   - Pass an **integer index**, and look up a task table with identical content on the current rank at execution time. Taking [FlexAWQSSZProcessor](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/processor/anti_outlier/flex_smooth/processor.py) as an example, `submit(fn=self._worker_fn, args=(idx,), ...)` with `adapter_config = self.sorted_configs[idx - 1]` inside `_worker_fn`;
   - Pass a **module name**. Taking [LinearQuant](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/processor/quant/linear.py) as an example, `submit(fn=self._dts_calibrate_forward, args=(name,), ...)`, where `_dts_calibrate_forward` performs `self.model.get_submodule(module_name)` and then runs the local logic.

**Output**: a submittable subtask function.

**Pass conditions**: the subtask can be executed independently on any single rank, and the result is consistent with executing the whole chain.

### Step 3: Configure the Synchronization Mechanism

**Objective**: align the model state that the algorithm's subtasks produce on each rank after the tasks are executed, equivalent to every rank having executed the complete task chain.

**Inputs**: how the task affects the model state (structural change / parameter change / buffer change).

**Action**: choose among the three synchronization levels as needed (priority `sync_fn` > `DTSMixin.distributed_sync` > `default_module_state_sync`; a subtask executes only one of them, and once `sync_fn` is provided the module-level synchronization described below is no longer performed):

| Mechanism | Trigger | Applicable Scenario |
| --- | --- | --- |
| `sync_fn` | `submit(..., sync_fn=...)`, with the signature `(record: TaskExecutionRecord, sync_ctx: TaskSyncContext) -> None` | The task causes a **model structural change** (such as IR replacement), which module-level synchronization cannot satisfy; a custom implementation may use `DTSMixin.broadcast_tensor` for tensor broadcasting |
| `DTSMixin.distributed_sync` | Some submodule class in the dependency-path subtree inherits `DTSMixin` and implements `distributed_sync(record, sync_ctx)`, which is called automatically during scheduling | The model structure is unchanged but the default synchronization is insufficient (for example a derived quantity must be synchronized); all ranks must invoke collective communication in the same order |
| `default_module_state_sync` | Default behavior: takes the rank that executed the task (`record.executor_rank`) as the source and broadcasts its parameters and buffers to all ranks | Most parameter / buffer change scenarios; the implementation first synchronizes parameter names and shape metadata with `broadcast_object_list`, then broadcasts tensor by tensor (under HCCL/NCCL, CPU tensors are automatically staged to the accelerator) |

**Output**: the completed synchronization configuration.

**Pass conditions**: after the scheduling execution, the model state is identical on every rank (compared with the serial execution result without DTS integration).

**Troubleshooting**: when the results are inconsistent, check whether the synchronization policy was chosen correctly (a structural change must use `sync_fn`); the collective communication order inside `sync_fn` must be the same on all ranks, otherwise it hangs.

### Step 4: Build the Shared Task Queue

**Objective**: submit the algorithm's subtasks to the shared task queue.

**Inputs**: the `fn` from step 2; the synchronization policy from step 3; the dependent module paths; `DistHelper` (when `local_only` determination is involved).

**Action**:

1. **Create the scheduler**: inside the algorithm's Processor, create the distributed task scheduler with `with DistributedTaskScheduler(self.model) as scheduler:`.
2. **Identify `dependencies`**: the list of **module paths** involved in this task (for example `[source] + list(targets)` in an outlier suppression subgraph), used to execute the subtasks in waves so as to preserve equivalence with serial execution.
3. **Determine the semantics of `parallel`** (default `True`):
   - `True`: with multiple devices the subtasks are executed cooperatively by different ranks (each subtask usually runs on only one rank); after the scheduling execution finishes, synchronization makes the result equivalent to every rank having executed it;
   - `False`: with multiple devices the subtasks do not enter the shared queue; this applies to scenarios such as a subtask that contains a non-shared module. Taking FlexAWQSSZ as an example, the subtasks of non-fused subgraphs and of non-shared modules must be completed by each rank itself:

     ```python
     is_non_fusion = (m.source is None and m.targets is not None)
     has_non_shared_module = any(
         not self.dist_helper.is_shared(name) for name in module_names
     )   # module_names = [m.source] + list(m.targets) (with None removed)
     scheduler.submit(fn=self._worker_fn, args=(idx,),
                      dependencies=module_names,
                      parallel=not (is_non_fusion or has_non_shared_module))
     ```

4. **Configure the synchronization policy** (optional, see step 3): pass `sync_fn` to customize the task-level synchronization.

**Output**: the scheduler context with all `submit` calls completed.

**Pass conditions**: all `submit` calls raise no exception (the dependency path check passes); before `run()`, the number and the semantics of the tasks submitted by each rank are consistent (`run()` validates internally, see step 5).

**Troubleshooting**: when a dependency path fails to resolve, use `model.named_modules()` to check the full path name; when `submit mismatch across ranks` is reported on multiple devices, check whether each rank generated its tasks from the same data (for example whether the task table `sorted_configs` is identical on every rank).

### Step 5: Execute the Scheduling

**Objective**: trigger DTS to distribute and execute all subtasks and to synchronize by wave.

**Inputs**: the scheduler with the submission completed in step 4.

**Action**:

1. After all ranks complete every `submit` inside the same `with` block, call `scheduler.run()`; it returns a `TaskExecutionRecord` list in the same order as the submissions (containing `executor_rank`, `exec_time_s`, `sync_time_s`, and so on).
2. Understand the execution mechanism (`WaveDTSBackend`):
   - **Semantic consistency check**: `run()` first compares each rank's task count and task semantic hash with `all_gather_object` (`task_semantic_hash` is a sha256 computed by reducing over fn / sync_fn / args / kwargs / dependencies / parallel); on a mismatch it raises `RuntimeError` directly and points out the index of the first inconsistent task, avoiding hidden errors caused by ranks submitting different tasks;
   - **Task distribution**: tasks are preferentially taken from the shared task queue (`DPLayerWiseRunner` creates and injects each rank's cross-process queue before `mp.spawn`; rank 0 puts in task indexes and a `None` terminator, and each rank blocks on `get` with a 300s timeout); without a shared queue it degrades to static round-robin (`idx % world_size == rank`);
   - **Execution protection**: `_collective_op_guard` is mounted while a shared task executes, and any illegal collective communication raises an error directly (a subtask function must not contain collective communication, otherwise it causes a process deadlock);
   - **Performance log**: after `run()` finishes it prints `【DTS】 Summary: ...` (task count, number executed by this rank, exec/sync time, speedup, and queue mode); when `exec_over_sync <= 1` it outputs the hint `DTS not suitable for parallel`.

**Output**: the `TaskExecutionRecord` list (usable for auditing the execution assignment and the time consumed).

**Pass conditions**: `run()` returns normally, with no semantic mismatch / queue timeout / collective interception errors.

**Troubleshooting**: on `queue.get() timeout after 300s`, check whether all ranks entered the same `with` block and called `run()` (an upstream producer is stalled, or the shared queue was not injected); on `submit mismatch`, investigate the consistency of the task generation logic across ranks.

## 6. Acceptance Conditions

- Multi-device quantization completes successfully after the multi-device quantization command is run, with no errors in the log.

## 7. Troubleshooting

- **Dependency path error at submission time**: check the full path with `model.named_modules()`; remove `None` / empty-string entries.
- **Payload not reducible**: `args` / `kwargs` contain non-serializable objects; change them to stable values such as an index / module name / scalar.
- **Process deadlock**: check whether the subtask function contains collective communication and remove it. In the shared queue each subtask is usually executed by only one rank, and the other ranks cannot reach the synchronization point, so a subtask function must not contain collective communication operations.

## 8. Case List

| Case | Brief Description | Link |
| --- | --- | --- |
| FlexAWQSSZ integration | Turning per-subgraph smoothing into tasks: `submit(args=(idx,))` looks up the task table, `parallel=not (is_non_fusion or has_non_shared_module)` (step 4 of this guide contains the complete determination code) | [FlexAWQSSZProcessor](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/processor/anti_outlier/flex_smooth/processor.py) |
| LinearQuant integration | Turning per-module calibration into tasks: `submit(fn=_dts_calibrate_forward, args=(name,), dependencies=[name + ".weight_quantizer"], parallel=True)` | [processor/quant/linear.py](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/processor/quant/linear.py) |
| FlexSmoothQuant integration | In the same family as FlexAWQSSZ, task-based smoothing based on subgraph priority | [processor/anti_outlier/flex_smooth/processor.py](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/processor/anti_outlier/flex_smooth/processor.py) |

## 9. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| Distributed Task Scheduler (DTS) | On top of DP completeness, it schedules algorithm subtasks across devices; each task runs only once and dependencies drive the waves | *[Distributed Task Scheduler](term_distributed_task_scheduler.md)* |
| Data Parallelism (DP) | Each rank holds a complete model replica, the calibration data is split by rank, and the statistics of shared modules are reduced across devices | *[Data Parallelism](../data_parallelism/term_data_parallelism.md)* |
| Expert Parallelism (EP) | Routed experts are sharded by rank, introducing `local_only` modules held only by the current rank (DTS must avoid assigning tasks that depend on them) | *[Expert Parallelism](../expert_parallelism/term_expert_parallelism.md)* |

## 10. Interface Documentation List

| API or Capability | Brief Description | Link |
| --- | --- | --- |
| `DistributedTaskScheduler` | The core DTS class | [scheduler.py](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/utils/distributed/task_scheduler/scheduler.py) |
| `WaveDTSBackend` | The default wave-based backend: dependency prefix / `parallel` semantic conflicts are split into waves automatically, distribution via a shared queue or static round-robin, collective interception, and performance logging | [backend/wave.py](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/utils/distributed/task_scheduler/backend/wave.py) |
| `DTSMixin` / `default_module_state_sync` | Module-level synchronization: `distributed_sync` for custom synchronization; default parameter / buffer broadcasting | [sync.py](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/utils/distributed/task_scheduler/sync.py) |
