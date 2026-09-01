# 分布式任务调度器使用指南

## 1. 适用范围

适用于在 DP 支持的基础上，进一步对共享模块上的计算密集型算法任务做并行加速，从而进一步提升多卡量化效率。

## 2. 流程关系与前置条件

**上级流程**：按[数据并行机制使用指南](../data_parallelism/data_parallelism_guide.md)完成算法的 DP 支持。

**前置条件**：

- 多卡量化可正确跑通。
- 已理解[专家并行](../expert_parallelism/term_expert_parallelism.md)语义：含 `local_only` 模块依赖的任务不能跨 rank 分工。

**后续操作**：验证 DTS 结果与未接入时语义等价、记录加速比。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 已完成 DP 支持的量化算法 | `msmodelslim/processor/` | `support_distributed()` 返回 `True`，多卡结果与单卡语义等价 | 多卡量化产物与单卡比对一致 |
| 交付件 | 接入 DTS 的 Processor 代码 | `msmodelslim/processor/` | 算法可拆分为原子化的子任务 | 多卡跑通，结果与未接入时语义等价 |

## 4. 流程总览

```mermaid
flowchart LR
  A[确认接入前置条件] --> B[封装子任务函数]
  B --> C[配置同步机制]
  C --> D[构建共享任务队列]
  D --> E[执行调度]
```

## 5. 操作步骤

### 步骤 1：确认接入前置条件

**目标**：确认算法满足 DTS 的接入前提，避免接入后出现死锁或语义偏差。

**输入**：算法多卡实现；子任务逻辑。

**操作**：

1. **确认 DP 支持**：多卡量化产物与单卡语义等价（未完成请先按[数据并行机制使用指南](../data_parallelism/data_parallelism_guide.md)完成支持）。
2. **识别可拆分任务并核对约束**：子任务须可被任意单个 rank 独立执行并产生一致结果。

**输出**：可拆分任务清单（每项标注依赖模块路径、是否允许跨 rank 分工）。

**通过条件**：清单中每项均可由单 rank 独立执行。

### 步骤 2：封装子任务函数

**目标**：把可拆分逻辑封装为可由 `submit` 调用的 Processor 方法。

**输入**：步骤 1 的任务清单。

**操作**：

1. 将任务逻辑封装为 Processor 实例方法，签名保持简单（如 `def _worker_fn(self, idx) -> None`）。
2. **业务参数只经 `args` / `kwargs` 传入可序列化值**，不得作为 `submit` 的第二个位置参数。常用模式：
   - 传**整数索引**，执行时回查本 rank 上内容一致的任务表，以[FlexAWQSSZProcessor](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/anti_outlier/flex_smooth/processor.py)）为例，`submit(fn=self._worker_fn, args=(idx,), ...)`，`_worker_fn` 内 `adapter_config = self.sorted_configs[idx - 1]`；
   - 传**模块名**，以[LinearQuant](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/quant/linear.py)为例，`submit(fn=self._dts_calibrate_forward, args=(name,), ...)`，`_dts_calibrate_forward` 内 `self.model.get_submodule(module_name)` 后执行本地逻辑。

**输出**：可提交的子任务函数。

**通过条件**：子任务在任意单 rank 上可独立执行，结果与整链执行一致。

### 步骤 3：配置同步机制

**目标**：使算法子任务在各 rank 产生的模型状态在任务执行后对齐，等价于各 rank 都执行过完整任务链。

**输入**：任务对模型状态的影响方式（结构变化 / 参数变化 / buffer 变化）。

**操作**：按需在三级同步中选择（优先级 `sync_fn` > `DTSMixin.distributed_sync` > `default_module_state_sync`，一个子任务只执行其中一级，提供 `sync_fn` 后不再做下方模块级同步）：

| 机制 | 触发方式 | 适用场景 |
| --- | --- | --- |
| `sync_fn` | `submit(..., sync_fn=...)`，签名 `(record: TaskExecutionRecord, sync_ctx: TaskSyncContext) -> None` | 任务导致**模型结构变化**（如 IR 替换），模块级同步无法满足；自定义实现中可用 `DTSMixin.broadcast_tensor` 做张量广播 |
| `DTSMixin.distributed_sync` | 依赖路径子树中某子模块类继承 `DTSMixin` 并实现 `distributed_sync(record, sync_ctx)`，调度执行时自动调用 | 模型结构未变、但默认同步不满足（如需要同步派生量）；各 rank 调用时集合通信顺序须一致 |
| `default_module_state_sync` | 默认行为：以执行该任务的 rank（`record.executor_rank`）为源，广播其参数与 buffer 到各 rank | 大多数参数 / buffer 变化场景；实现中先 `broadcast_object_list` 同步参数名与形状元数据，再逐张量广播（HCCL/NCCL 下 CPU 张量自动 staging 到加速器） |

**输出**：配置完成的同步策略。

**通过条件**：调度执行后各 rank 模型状态一致（与未接入 DTS 的串行执行结果比对）。

**异常处置**：结果不一致时检查同步策略选择是否正确（结构变化必须用 `sync_fn`）；`sync_fn` 内集合通信顺序须所有 rank 一致，否则挂死。

### 步骤 4：构建共享任务队列

**目标**：将算法子任务提交至共享任务队列。

**输入**：步骤 2 的 `fn`；步骤 3 的同步策略；依赖模块路径；`DistHelper`（含 `local_only` 判定时）。

**操作**：

1. **创建调度器**：在算法 Processor 内部通过 `with DistributedTaskScheduler(self.model) as scheduler:` 创建分布式任务调度器。
2. **识别 `dependencies`**：本任务涉及的**模块路径**列表（如离群值抑制子图中的 `[source] + list(targets)`），用于分波次执行子任务，以保持与串行执行的等价性。
3. **判定 `parallel` 语义**（默认 `True`）：
   - `True`：多卡时子任务由不同 rank 分担执行（每个子任务通常只在一个 rank 上跑），调度执行结束后经同步使结果等价于各 rank 都执行过；
   - `False`：多卡时子任务不进入共享队列，适用于子任务内包含非共享模块等场景。以 FlexAWQSSZ 为例，其中非融合子图和非共享模块的子任务，需由各 rank 各自完成：

     ```python
     is_non_fusion = (m.source is None and m.targets is not None)
     has_non_shared_module = any(
         not self.dist_helper.is_shared(name) for name in module_names
     )   # module_names = [m.source] + list(m.targets)（去 None）
     scheduler.submit(fn=self._worker_fn, args=(idx,),
                      dependencies=module_names,
                      parallel=not (is_non_fusion or has_non_shared_module))
     ```

4. **配置同步策略**（可选，见步骤 3）：传 `sync_fn` 自定义任务级同步。

**输出**：完成全部 `submit` 的调度器上下文。

**通过条件**：全部 `submit` 无异常（依赖路径校验通过）；`run()` 前各 rank 提交的任务数量与语义一致（`run()` 内部会校验，见步骤 5）。

**异常处置**：依赖路径解析失败时，用 `model.named_modules()` 核对全路径名；多卡下报 `submit mismatch across ranks` 时，检查各 rank 是否基于相同数据生成了任务（如任务表 `sorted_configs` 是否各 rank 一致）。

### 步骤 5：执行调度

**目标**：触发 DTS 分发执行全部子任务并按波次同步。

**输入**：步骤 4 提交完毕的调度器。

**操作**：

1. 各 rank 在同一 `with` 块内完成全部 `submit` 后调用 `scheduler.run()`，返回与提交顺序一致的 `TaskExecutionRecord` 列表（含 `executor_rank`、`exec_time_s`、`sync_time_s` 等）。
2. 了解执行机制（`WaveDTSBackend`）：
   - **语义一致性校验**：`run()` 先 `all_gather_object` 比对各 rank 的任务数与任务语义哈希（`task_semantic_hash` 由 fn / sync_fn / args / kwargs / dependencies / parallel 规约后 sha256），不一致直接 `RuntimeError` 并指出首个不一致任务索引——避免各 rank 提交不同任务导致的隐性错误；
   - **任务分发**：优先经共享任务队列抢任务（`DPLayerWiseRunner` 在 `mp.spawn` 前创建并注入各 rank 的跨进程队列，rank 0 放入任务索引与 `None` 终止符，各 rank 阻塞 `get`，超时 300s）；无共享队列时退化为静态轮询（`idx % world_size == rank`）；
   - **执行保护**：共享任务执行期间挂载 `_collective_op_guard`，非法集合通信直接报错（子任务函数内不得含集合通信，否则引起进程死锁）；
   - **性能日志**：`run()` 结束后打印 `【DTS】 Summary: ...`（任务数、本 rank 执行数、exec/sync 耗时、speedup 与队列模式）；`exec_over_sync ≤ 1` 时输出 `DTS not suitable for parallel` 提示。

**输出**：`TaskExecutionRecord` 列表（可用于审计执行分工与耗时）。

**通过条件**：`run()` 正常返回，无语义不匹配 / 队列超时 / collective 拦截报错。

**异常处置**：`queue.get() timeout after 300s` 时检查各 rank 是否都进入同一 `with` 并调用 `run()`（上游 producer 停滞或共享队列未注入）；`submit mismatch` 时排查各 rank 任务生成逻辑的一致性。

## 6. 验收条件

- 执行多卡量化命令后顺利完成多卡量化，日志无报错。

## 7. 异常处置

- **提交时依赖路径报错**：核对 `model.named_modules()` 全路径；删除 `None` / 空串条目。
- **载荷不可规约**：`args` / `kwargs` 含非序列化对象，改为索引 / 模块名 / 标量等稳定值。
- **进程死锁**：检查子任务函数中是否包含集合通信，移除其中的集合通信行为。共享队列中每个子任务通常只由一个 rank 执行，其余 rank 无法到达同步位点，因此子任务函数内不应包含集合通信操作。

## 8. 案例列表

| 案例 | 简述 | 链接 |
| --- | --- | --- |
| FlexAWQSSZ 接入 | 逐子图平滑任务化：`submit(args=(idx,))` 回查任务表，`parallel=not (is_non_fusion or has_non_shared_module)`（本指南步骤 4 含完整判定代码） | [FlexAWQSSZProcessor](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/anti_outlier/flex_smooth/processor.py) |
| LinearQuant 接入 | 逐模块校准任务化：`submit(fn=_dts_calibrate_forward, args=(name,), dependencies=[name + ".weight_quantizer"], parallel=True)` | [processor/quant/linear.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/quant/linear.py) |
| FlexSmoothQuant 接入 | 与 FlexAWQSSZ 同族，基于子图优先级的任务化平滑 | [processor/anti_outlier/flex_smooth/processor.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/anti_outlier/flex_smooth/processor.py) |

## 9. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 分布式任务调度器（DTS） | 在 DP 完备性之上对算法子任务做跨卡分工调度，任务只执行一次、依赖驱动分波 | 《[分布式任务调度器](term_distributed_task_scheduler.md)》 |
| 数据并行（DP） | 各 rank 持有完整模型副本、校准数据按 rank 切分、共享模块统计量跨卡归约 | 《[数据并行](../data_parallelism/term_data_parallelism.md)》 |
| 专家并行（EP） | 路由专家按 rank 分片，引入仅本 rank 持有的 `local_only` 模块（DTS 须避开含其依赖的任务分工） | 《[专家并行](../expert_parallelism/term_expert_parallelism.md)》 |

## 10. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `DistributedTaskScheduler` | DTS 核心类 | [scheduler.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/utils/distributed/task_scheduler/scheduler.py) |
| `WaveDTSBackend` | 默认分波后端：依赖前缀 / `parallel` 语义冲突自动分波，共享队列 / 静态轮询分发，collective 拦截与性能日志 | [backend/wave.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/utils/distributed/task_scheduler/backend/wave.py) |
| `DTSMixin` / `default_module_state_sync` | 模块级同步：`distributed_sync` 自定义同步；默认参数 / buffer 广播 | [sync.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/utils/distributed/task_scheduler/sync.py) |
