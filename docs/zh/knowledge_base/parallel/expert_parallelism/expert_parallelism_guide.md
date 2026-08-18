# 专家并行机制使用指南

## 1. 适用范围

适用于 MoE 模型的多卡量化适配，特别是当单卡显存不足以支撑加载全部专家的场景，接入专家并行机制可有效降低显存需求。

## 2. 流程关系与前置条件

**上级流程**：模型适配器已完成逐层加载适配；量化算法已完成[数据并行机制使用指南](../data_parallelism/data_parallelism_guide.md)的支持（含 DistHelper 注入与跨卡同步）。

**前置条件**：

- 模型 / 算法已支持 DP；EP 分片后非本地专家会自动落入 DistHelper 的 `local_only`，同步边界与 DTS 局部任务约束见 DP / [分布式任务调度器](../distributed_task_scheduler/distributed_task_scheduler_guide.md)指南，本流程不再重复；
- 分布式保存由 `DPLayerWiseRunner` 自动将 `ascendv1_saver` 转为 `ascendv1_saver_distributed`，适配侧无需改 YAML。

**后续操作**：量化精度验证。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | MoE 模型适配器 | `msmodelslim/model/`（新增或既有） | 支持逐层加载 | `--model_type` 命中适配器 |
| 输入 | 已完成 DP 完备性适配的量化算法 | `msmodelslim/processor/` | 已按[数据并行机制使用指南](../data_parallelism/data_parallelism_guide.md)实现 DP 支持 | 多卡统计同步正确 |
| 交付件 | EP 化的模型加载/构建代码 | `msmodelslim/model/`（原地修改或新增补丁模块） | 各 rank 仅加载本地专家；单进程下自动退化为全量专家 | 多卡各 rank 上非本地专家模块不存在（或为 `None` 槽位） |

## 4. 流程总览

```mermaid
flowchart LR
  A[确认 EP 适用性与分片约束] --> B[实现专家分片加载]
  B --> C[量化映射仅枚举本地专家]
```

## 5. 操作步骤

### 步骤 1：确认 EP 适用性与分片约束

**目标**：确认模型适合 EP 分片，并满足框架的分片约束。

**输入**：模型 `config.json`（`num_experts` / `n_routed_experts`）。

**操作**：

1. 确认模型为 MoE 结构且路由专家数可被 `world_size` 整除。框架统一的分片工具 [`resolve_expert_ep_range`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/common/utils.py)（`msmodelslim/model/common/utils.py`）行为如下：

   - `world_size ≤ 1`（未初始化分布式 / 单进程）：返回全量范围 `[0, num_experts)`，EP 不生效；
   - `world_size > 1`：按 `n_local = num_experts // world_size` 连续分片，rank `r` 的范围为 `[r * n_local, (r+1) * n_local)`；
   - `num_experts % world_size != 0`：抛出 `SchemaValidateError`（提示专家数必须可被 world size 整除）。

**输出**：模型结构分析文档。

### 步骤 2：实现专家分片加载

**目标**：使各 rank 只加载 / 只持有本地专家，同时保证单进程场景自动退化为全量专家。

**输入**：步骤 1 确认的本地专家范围；模型建模代码（自定义权重目录代码或仓库内建模）。

**操作**：对建模文件中的 MoE 类的 `__init__` 与 `forward` 方法进行重写。

- **形态 A：monkey-patch 建模类**（案例：[Kimi-K3 `ep_patches.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/kimi_k3/ep_patches.py)）：

  1. 定位权重目录中的 MoE 块类（Kimi-K3 为 `KimiSparseMoeBlock`）；
  2. 补丁改造 `__init__` 方法，按 `resolve_expert_ep_range(config.num_experts)` 构建**全长度 `ModuleList`**，非本地专家使用 `None` 填充：

     ```python
     self.experts = nn.ModuleList([
         mlp_cls(config, ...) if start <= i < end else None
         for i in range(config.num_experts)
     ])
     ```

  3. 补丁改造 `forward` 方法，按 `use_dp_ep = dist.is_initialized() and dist.get_world_size() > 1` 分支：多进程时先做 DP token 收集（seq_len `all_gather` + `DistHelper.gather_variable_shapes`），本 rank 只计算本地专家（`moe_infer_local_experts` 遍历 `[start, end)`），对局部输出 `all_reduce` 后再计算 latent 投影与共享专家，最后切回本 rank 序列段；单进程时走全量本地路径；
  4. 在适配器 `init_model` 中调用补丁函数。

- **形态 B：仓库内原生建模**（案例：[DeepSeek-V4 `model.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/deepseek_v4/model.py)）：

  1. 在 MoE 块构造中直接按 `world_size` 只创建本地专家：

     ```python
     self.n_local_experts = args.n_routed_experts // world_size
     self.experts_start_idx = rank * self.n_local_experts
     self.experts_end_idx = self.experts_start_idx + self.n_local_experts

     self.experts = nn.ModuleList(
            [
                Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(self.n_routed_experts)
            ]
        )
     ```

  2. MoE 前向在多进程时做 DP token 收集 + 本地专家计算 + `all_reduce`（与形态 A 相同的通信模式）；
  3. 适配器提供本地专家范围查询函数 `_get_local_expert_range()` 。

**输出**：EP 化加载的模型。

**通过条件**：多卡下 `named_modules` 中仅存在本地专家分片；单进程下可加载全量专家。

**异常处置**：补丁未生效时检查补丁调用时机（需在建模类实例化前完成替换）；加载报权重缺失时核对本地专家范围与分片权重文件。

### 步骤 3：量化映射仅枚举本地专家

**目标**：使所有专家相关的量化映射（子图、LN 融合、旋转）只覆盖本 rank 实际存在的专家，避免 Processor 处理不存在的模块。

**输入**：适配器的三个映射接口；步骤 2 的本地专家范围。

**操作**：

在 `get_adapter_config_for_subgraph()`（IterSmooth 子图）、`get_ln_fuse_map()`（QuaRot LN-Linear 融合）、`get_rotate_map()`（QuaRot 旋转映射）中，**所有专家相关条目都基于本地专家范围生成**，禁止枚举全量专家：

```python
expert_start, expert_end = _get_expert_range(text_cfg)   # 或适配器自有的 _get_local_expert_range()
for expert in range(expert_start, expert_end):
    configs.append(AdapterConfig(
        subgraph_type="up-down",
        mapping=MappingConfig(
            source=f"{moe}.experts.{expert}.w3",
            targets=[f"{moe}.experts.{expert}.w2"],
        ),
    ))
```

参考实现：

- DeepSeek-V4：`get_adapter_config_for_subgraph` / `get_ln_fuse_map` / `get_rotate_map` 全部经 `_get_local_expert_range()` 生成（如 `ffn_norm` 的 targets 仅含本地专家的 `w1`/`w3`，`rot` 的 left/right 仅含本地专家的 `w1`/`w2`/`w3`）；
- Kimi-K3：`quarot.py` 中 `rot_latent` 旋转对仅映射 `_get_expert_range(text_cfg)` 范围内的专家（EP 耦合：非本地专家槽位为 `None`，映射到不存在的模块会直接失败）。

**输出**：仅含本地专家的量化映射。

**通过条件**：多卡量化跑通，无「无法解析模块路径」类报错；各 rank 对同一层专家映射的数量等于本地专家数。

**异常处置**：报模块不存在 / 子模块无法获取时，检查映射是否仍枚举了全量专家（按 `range(num_experts)` 而非本地范围）。

## 6. 验收条件

- 执行多卡量化命令后顺利完成多卡量化，日志无报错。

## 7. 案例列表

| 案例 | 简述 | 链接 |
| --- | --- | --- |
| Kimi-K3 EP 接入 | monkey-patch 权重目录的建模文件 | [ep_patches.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/kimi_k3/ep_patches.py) |
| DeepSeek-V4 EP 接入 | 仓库内原生建模 | [model.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/deepseek_v4/model.py) |

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 专家并行（EP） | 路由专家按 rank 分片，引入仅本 rank 持有的 `local_only` 模块 | 《[专家并行](term_expert_parallelism.md)》 |
| 数据并行（DP） | 各 rank 持有完整模型副本、校准数据按 rank 切分、共享模块统计量跨卡归约 | 《[数据并行](../data_parallelism/term_data_parallelism.md)》 |
| 分布式任务调度器（DTS） | 在 DP 完备性之上对算法子任务做跨卡分工调度，须避开含 `local_only` 依赖的任务 | 《[分布式任务调度器](../distributed_task_scheduler/term_distributed_task_scheduler.md)》 |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `resolve_expert_ep_range` / `_get_expert_range` | 本地专家范围解析：单进程返回全量，多进程按 `world_size` 连续分片，不可整除时报错 | [common/utils.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/common/utils.py) |
| `DistHelper` | 模块拓扑分类：`is_shared` / `is_local_only` / `get_shared_modules_slice` | [dist_helper.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/utils/distributed/dist_helper.py) |
| `DistributedAscendV1Saver` | 分布式保存：`local_only` 独占写出、共享模块分工、rank 0 合并 | [ascendv1_distributed.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quant_service/modelslim_v1/save/ascendv1_distributed.py) |
| `--device npu --device_id 0 1 ...` / `spec.runner` | 多卡量化入口（EP 在分布式初始化后生效） | 《[一键量化使用说明](../../../user_guide/usage_quick_quantization.md)》 |

## 10. 产品形态与资源限制

- **约束**：路由专家数必须可被 `world_size` 整除；仅 MoE 模型适用。
- **资源**：EP 的收益是单卡显存随本地专家数近似线性下降；其代价是 MoE 前向引入 token 收集与 `all_reduce` 通信。
