# 数据并行机制使用指南

## 1. 适用范围

适用于为量化算法适配数据并行机制。

## 2. 流程关系与前置条件

**上级流程**：算法在单卡场景完成量化验证、模型适配器完成逐层加载适配（实现 `ModelSlimPipelineInterfaceV1` 的逐层接口）。

**前置条件**：

- 具备单机多卡环境。

**后续操作**：多卡量化的模型权重进入部署与精度验证。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 已完成单卡验证的量化算法 | `msmodelslim/processor/`（本仓已有实现或新增实现） | 基类 `support_distributed()` 默认返回 `False`，需按本流程重写 | 单卡量化产物正确，量化流程可完整跑通 |
| 输入 | 模型适配器 | `msmodelslim/model/` | 支持 `layer_wise` 逐层加载（`generate_decoder_layer` 等接口） | `msmodelslim quant` 单卡可命中并逐层加载 |
| 交付件 | 声明并实现同步的处理器代码 | `msmodelslim/processor/`（原路径原地修改） | `support_distributed()` 返回 `True`；Observer / Processor 按本流程实现跨卡同步 | 通过 `DPLayerWiseRunner` 启动时的分布式支持校验（日志无 `UnsupportedError`） |

## 4. 流程总览

```mermaid
flowchart LR
  A[声明处理器分布式支持] --> B[注入 DistHelper 建立模块拓扑分类]
  B --> C[识别需全局一致的统计量]
  C --> D[实现跨卡状态同步]
  D --> E[确认多卡入口与 runner 选择]
```

## 5. 操作步骤

### 步骤 1：声明处理器分布式支持

**目标**：让量化流水线中的每个 Processor 声明自身支持多卡执行。

**输入**：量化配置 `spec.process` 中涉及的全部 Processor 实现。

**操作**：

1. 在 Processor 类中重写 `support_distributed()`（基类 [`BaseSessionProcessor`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/base.py) 默认返回 `False`）：

   ```python
   def support_distributed(self) -> bool:
       return True
   ```

2. 对照[并行机制支持列表](../README.md#3-并行机制支持列表)中的 Processor 表确认 DP 覆盖范围，尚未支持 DP 的 Processor 需按后续步骤完成适配。

**输出**：`support_distributed()` 返回 `True` 的 Processor 实现。

**通过条件**：多卡启动时不再因处理器不支持而报 `UnsupportedError`。

**异常处置**：若日志报不支持的处理器，逐个补齐 `support_distributed()` 重写。

### 步骤 2：注入 DistHelper 建立共享/局部模块分类

**目标**：使算法能判断“哪些模块需要跨卡同步、哪些模块禁止同步”，为统计归约划定安全边界。

**输入**：步骤 1 的 Processor（其 `preprocess` 会操作模型结构）。

**执行前检查**：算法在 `preprocess` 中是否还会插入 Observer、替换 IR 等结构变更。

**操作**：

1. 在 `preprocess` 中、**所有结构变更完成之后**初始化并注入统计组件`DistHelper`（以 FA3 为例，见 [`processor.py`](../../../../../msmodelslim/processor/quant/fa3/processor.py)）：

   ```python
   if dist.is_initialized():
       self.dist_helper = DistHelper(request.module, prefix=request.name)
       for _, submodule in request.module.named_modules(prefix=request.name):
           if not isinstance(submodule, _FA3PerHeadObserver):
               continue
           submodule.set_dist_helper(self.dist_helper)
   ```

   - `DistHelper` 构造时对各 rank 的 `named_modules` 做一次 `all_gather_object`，自动区分共享模块（各 rank 共有）与 `local_only` 模块（仅本 rank 持有，典型为 EP 下的路由专家）。
   - **时序约束**：`DistHelper` 是拓扑快照，构造后不随模型改动刷新；若在初始化之后又插入 Observer / 替换 IR，`is_shared` 将按旧结构返回，导致算法行为异常。因此必须在结构变更全部完成后初始化。

**输出**：注入到统计组件中的 `DistHelper` 实例。

**通过条件**：各 rank 对同一共享模块名调用 `is_shared` 结果一致。

**异常处置**：若多卡结果不一致且怀疑分类过期，检查 `DistHelper` 初始化位置是否晚于全部结构变更；`local_only` 模块（EP 局部专家）被误判为共享并同步，会表现为卡死或结果错误。

### 步骤 3：识别需全局一致的统计量

**目标**：明确本算法中哪些变量必须在各 rank 保持一致，形成待同步清单。

**输入**：算法的校准前向与量化参数计算逻辑。

**操作**：

梳理算法中由校准数据累计得到、并最终决定量化结果的变量，例如：

- **激活统计量**：Observer 累计的 `min` / `max` / `sum`等；
- **量化参数**：由统计量计算的量化参数（scale / zero point）等；
- **平滑/旋转系数**：FlexSmoothQuant 的平滑系数、QuaRot 的旋转矩阵等；
- **校准期缓存**：按 batch 缓存的激活张量（如 `StatKey.TENSOR` 类统计）。

判定原则：各卡基于局部校准数据分片计算得到不同值但需要写入最终模型的量，都需要跨卡一致化。

**输出**：待同步变量清单。

**通过条件**：清单覆盖算法中所有影响最终量化结果、且随校准分片变化的量。

### 步骤 4：实现跨卡状态同步

**目标**：对共享模块的统计量 / 参数做集合通信，使各 rank 得到与持有完整校准集等价的全局量。

**输入**：步骤 3 的待同步变量清单、步骤 2 注入的 `DistHelper`。

**操作**：按算法形态选择以下一种或组合实现，优先复用 [`dist_ops.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/utils/distributed/dist_ops.py) 存量工具：

| 工具函数 | 主要入参 | 功能与典型场景 |
| --- | --- | --- |
| `sync_base_operation` | `tensor`（原地更新）；`op`：`min`/`max`/`sum`/`mean`/`prod`；`group`（可选） | 对各 rank 同形张量做集合通信，结果写回同一 `tensor`。适用于 Observer 内累计的 min/max、channel_max 等统计量对齐 |
| `sync_gather_tensors` | `tensor`；`variable_shapes`；`on_cpu`；`group` | 收集各 rank 的单个张量为长度 `world_size` 的列表。适用于需保留各 rank 分片数据而非合并为单一统计量的场景 |
| `sync_gather_tensor_lists` | `tensor_list`；`on_cpu`；`group` | 收集各 rank 的张量列表并展平为一条大列表。适用于校准阶段按 batch 缓存的激活张量在 `postprocess` 一次性合并 |

- **形态 A：Observer 内同步**（统计量累计在 Observer 中）。
  以 [FA3PerHeadObserver](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/quant/fa3/processor.py)为例, Observer 前向时按共享模块判定传入同步开关 `sync`：

  ```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        samples = x.contiguous().view(x.shape[1], -1)
        sync = self._dist_helper is not None and self._dist_helper.is_shared(self._name)
        self._observer.update(samples, sync=sync)
        return x
  ```

  [RecallWindowObserver](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/observer/recall_window.py)内部更新激活统计量时按同步开关调用 `sync_base_operation` 归约：

  ```python
    if sync and dist.is_initialized():
        sync_base_operation(self._min_values, op="min")
        sync_base_operation(self._max_values, op="max")
  ```

- **形态 B：Processor 内同步**（校准期仅本地缓存激活）。
  以 [FlexSmoothQuant](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/anti_outlier/flex_smooth/processor.py) 为例，hook 仅本地追加激活张量：

  ```python
  # FlexStatsCollector.create_hook：校准前向只做本地采集
  cpu_tensor = tensor.to("cpu").reshape(-1, tensor.shape[-1])
  if StatKey.TENSOR not in module_stats:
      module_stats[StatKey.TENSOR] = [cpu_tensor]
  else:
      module_stats[StatKey.TENSOR].append(cpu_tensor)
  ```

  `postprocess` 中通过调用类方法 `sync_act_stats`对各共享模块的本地张量列表做跨卡合并后再继续算法：

  ```python
  # FlexStatsCollector.sync_act_stats 内部：
  local_tensors = self.act_stats[name][StatKey.TENSOR]
  merged = sync_gather_tensor_lists(local_tensors, on_cpu=on_cpu)
  self.act_stats[name][StatKey.TENSOR] = merged
  ```

**输出**：完成跨卡同步的 Observer / Processor 实现。

**通过条件**：多卡校准结束后，各 rank 上待同步变量的值一致，与单卡全量校准语义等价。

**异常处置**：死锁/挂起时检查是否对 `local_only` 模块发起集合通信；数值不一致时检查 `DistHelper` 时序（步骤 2）与同步位点是否覆盖全部待同步量（步骤 3）。

### 步骤 5：确认多卡量化入口与 runner 选择

**目标**：确认量化将以多卡 DP 方式运行，明确触发条件与生效范围。

**输入**：量化 YAML、目标设备列表。

**操作**：

1. 在量化配置中声明 runner（二选一）：

   ```yaml
   spec:
     runner: "dp_layer_wise"   # 显式指定分布式逐层量化
   # runner: "auto"            # 或由框架按设备数自动选择
   ```

2. 通过命令行指定多卡：

   ```shell
   msmodelslim quant \
     --model_path ${MODEL_PATH} \
     --save_path ${SAVE_PATH} \
     --model_type ${MODEL_TYPE} \
     --quant_type ${QUANT_TYPE} \
     --device npu \
     --device_id 0 1 2 3 4 5 6 7
   ```

   `--device_id 0 1 2 3 4 5 6 7` 指定参与量化的设备索引列表。

3. 确认 runner 实际选择。`modelslim_v1` 的 `_choose_runner_type` 按以下规则决定执行管线（对应 [`quant_service.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quant_service/modelslim_v1/quant_service.py)）：

   | `spec.runner` 取值 | 设备列表 | 选择的 runner |
   | --- | --- | --- |
   | `model_wise` | 任意 | `ModelWiseRunner` |
   | `layer_wise` | 任意 | `LayerWiseRunner` |
   | `dp_layer_wise` | 任意 | `DPLayerWiseRunner` |
   | `auto` | 多卡（> 1） | `DPLayerWiseRunner` |
   | `auto`（或未配置） | 单卡 / 未指定 | `LayerWiseRunner` |

   `DPLayerWiseRunner.run()` 中设备数 ≤ 1 时会回退到单卡 `LayerWiseRunner` 执行并打印告警（[`dp_layer_wise_runner.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/runner/dp_layer_wise_runner.py)）。

**输出**：确认以 `DPLayerWiseRunner` 执行多卡量化的配置（YAML + 命令）。

**通过条件**：日志出现 `Starting distributed execution with N devices`（N ≥ 2），各 rank 打印 `Rank i/N initialized on device ...`。

**异常处置**：若日志显示回退单卡（`Number of devices <= 1, falling back to single-device execution`），检查是否用 `--device_id` 传入了多个索引（如 `--device_id 0 1 2 3 4 5 6 7`）；多模态 VLM 服务不支持多卡（见适用范围），请确认目标服务为 `modelslim_v1`。

## 6. 验收条件

- 执行多卡量化命令后顺利完成多卡量化，日志无报错；

## 7. 案例列表

| 案例 | 简述 | 链接 |
| --- | --- | --- |
| FA3Quant 多卡适配 | Observer 内 `sync_base_operation` 归约 per-head min/max，完备性支持完整示例（本指南步骤 4 含代码片段） | [FA3QuantProcessor](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/quant/fa3/processor.py) |

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 数据并行（DP） | 各 rank 持有完整模型副本、校准数据按 rank 切分、统计量跨卡归约 | 《[数据并行](term_data_parallelism.md)》 |
| 专家并行（EP） | 路由专家按 rank 分片，引入仅本 rank 持有的 `local_only` 模块 | 《[专家并行](../expert_parallelism/term_expert_parallelism.md)》 |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `DistHelper` | 模块拓扑分类工具：`is_shared` / `is_local_only` | [dist_helper.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/utils/distributed/dist_helper.py) |
| `sync_base_operation` 等 | 跨 rank 统计量归约工具函数 | [dist_ops.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/utils/distributed/dist_ops.py) |
| `support_distributed()` | Processor 分布式支持声明（基类默认 `False`） | [processor/base.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/processor/base.py) |
| `ascendv1_saver_distributed` | 分布式保存器（由 `ascendv1_saver` 自动转换） | [ascendv1_distributed.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quant_service/modelslim_v1/save/ascendv1_distributed.py) |
| `--device_id 0 1 ...` | 多卡量化入口配置 | 《[一键量化使用说明](../../../user_guide/usage_quick_quantization.md)》 |

## 10. 产品形态与资源限制

- **限制**：当前仅支持单机多卡；由于多卡量化在各卡上的量化调度流程基本与逐层量化逻辑一致，因而显存要求基本与单卡量化一致。
- **资源**：多卡加速收益受 I/O 读写、算法计算密度、硬件性能与同步频率影响，加速效果不严格随卡数线性增长。
