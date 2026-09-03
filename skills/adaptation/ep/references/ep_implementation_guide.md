# msModelSlim EP 实施指南（真实实现参考）

本指南基于 msModelSlim 仓库中已通过验证的 EP 实现提炼而来，主要参考：

- `msmodelslim/model/common/utils.py` — `resolve_expert_ep_range()` / `_get_expert_range()`
- `<model_family>/ep_patches.py` — monkey-patch 方式 EP 改造
- `<model_family>/model_adapter.py` — 适配器挂载 EP patch 的时机
- `<model_family>/model.py` — 模型内嵌的 DP+EP MoE forward
- `msmodelslim/utils/distributed/dist_helper.py` — `DistHelper`（EP 模块范围跟踪、变长 shape all-gather）

> 说明：以下代码片段是**实现参考**，用于指导 Agent 改造目标模型。
> 具体路径、类名、配置项需按目标模型的实际代码调整。

## 1. 专家分片工具：resolve_expert_ep_range

`common/utils.py` 已实现连续均匀分片（全局 expert id 不变）：

```python
from typing import Tuple
import torch.distributed as dist

def resolve_expert_ep_range(num_experts: int) -> Tuple[int, int, int, int]:
    """Return (ep_size, ep_rank, start, end) for contiguous expert sharding."""
    if num_experts <= 0:
        return 1, 0, 0, 0
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return 1, 0, 0, num_experts

    world_size = dist.get_world_size()
    if num_experts % world_size != 0:
        raise SchemaValidateError(
            f"The total number of experts ({num_experts}) must be divisible by the world size ({world_size})."
        )
    n_local = num_experts // world_size
    rank = dist.get_rank()
    start = rank * n_local
    end = start + n_local
    return world_size, rank, start, end
```

适配器侧统一通过 `_get_expert_range(config)` 获取 `(start, end)`：

```python
from msmodelslim.model.common.utils import _get_expert_range

expert_start, expert_end = _get_expert_range(config)
```

- 单进程 / 未初始化分布式 → 全范围 `[0, num_experts)`。
- `num_experts` 必须能被 world size 整除，否则抛 `SchemaValidateError`。

命名为 expert range 的分片在**所有 rank 上覆盖全部专家**（连续、无重叠）。

## 2. EP 模块范围跟踪：DistHelper

`DistHelper` 通过 `dist.all_gather_object` 在进程间同步各 rank 的模块名集合，
自动区分三类模块：

```text
shared       所有 rank 都有 → 全局一致、仅需一次处理
local_only   仅本 rank 有    → 专家分片后的本地路由专家
all          所有进程的并集
```

典型用法：

```python
from msmodelslim.utils.distributed import DistHelper

# 构建模型后
helper = DistHelper(model)          # 内部做 all_gather_object，需在分布式初始化后调用
local_only = set(helper.local_only_modules())  # 本 rank 独有的专家模块
shared     = set(helper.shared_modules())       # 各 rank 共有模块
```

> EP 改造时要求：路由专家 module 是 `local_only`（只有安装了该专家的 rank 才有），
> gateway / shared expert / attention / norm 等是 `shared`。
> Agent 可借助 DistHelper 校验分片是否正确。

变长 shape all-gather（DP+EP 中 gather 各 rank 不同 seq_len 的 token）：

```python
tensor_list = DistHelper.gather_variable_shapes(local_tensor)  # 返回 list[Tensor]
x = torch.cat(tensor_list, dim=<seq_dim>)
```

## 3. expert 构造：全长度 ModuleList + None 占位

推荐保持全局 expert id 与 ModuleList 下标一致：

```python

# MoE.__init__ 内
ep_size, ep_rank, expert_start, expert_end = resolve_expert_ep_range(num_experts)

self.experts = nn.ModuleList(
    [
        Expert(dim, inter_dim) if expert_start <= i < expert_end else None
        for i in range(num_experts)
    ]
)
```

要点：

- `ModuleList` 长度 = `total_experts`，非本地槽位为 `None`。
- `None` 槽位不占参数、不参与 `named_parameters()`、不注册 quant hook。
- 保持 router / checkpoint / module path / quant mapping 的 expert id 一致。

## 4. 本地专家推理循环（calibration 场景）

本地专家循环示例：

```python
def moe_infer_local_experts(experts, x, topk_ids, topk_weight, expert_start, expert_end):
    """EP-safe routed expert inference.

    Args:
        experts: Full-length ModuleList; non-local entries may be None.
        x: [T, H] token features.
        topk_ids / topk_weight: [T, K] routing from the gate.
    """
    y = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.float32, device=x.device)
    for expert_id in range(expert_start, expert_end):
        expert = experts[expert_id]
        if expert is None:
            continue
        token_idx, top_idx = torch.where(topk_ids == expert_id)
        if token_idx.numel() == 0:
            continue
        expert_out = expert(x[token_idx])
        y[token_idx] += expert_out.to(torch.float32) * topk_weight[token_idx, top_idx].unsqueeze(-1).to(torch.float32)
    return y.type_as(x)
```

> 该循环只遍历 `[expert_start, expert_end)`，天然只访问本地专家。
> 若模型已有 `torch.where` 分布式实现（all-to-all dispatch/combine），优先复用原实现。

## 5. MoE forward：DP+EP 模式完整示例

以下为 DP+EP forward 参考（`torch.no_grad` / inference 场景）：

```python
class MoE(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_dp_ep = dist.is_initialized() and dist.get_world_size() > 1

        if use_dp_ep:
            # 1) 同步各 rank seq_len，计算本 rank token 区间
            seq_len_this_rank = x.size(-2)
            seq_len_tensor = torch.tensor([seq_len_this_rank], dtype=torch.long, device=x.device)
            seq_len_list = [torch.zeros_like(seq_len_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(seq_len_list, seq_len_tensor)
            seq_lens = [int(s.item()) for s in seq_len_list]
            rank = dist.get_rank()
            start_pos = sum(seq_lens[:rank])
            end_pos = start_pos + seq_len_this_rank
            # 2) gather 所有 rank 的 token
            x = torch.cat(DistHelper.gather_variable_shapes(x), dim=1)
        else:
            start_pos, end_pos = 0, x.size(-2)

        shape = x.size()
        x = x.view(-1, self.dim)

        # 3) 全局 router（expert id 空间不变）
        weights, indices = self.gate(x)

        # 4) 只计算本地专家
        y = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.float32, device=x.device)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]

        # shared expert 按原语义（EP 中保持全量或按模型约定）
        y += self.shared_experts(x)

        # 5) 跨 rank 合并专家输出
        if use_dp_ep:
            dist.all_reduce(y)
            # 6) 切回本 rank 原始 token
            return y.type_as(x).view(shape)[:, start_pos:end_pos, :]

        return y.type_as(x).view(shape)
```

闭环要点：

```text
router → all-gather token → 全局 gate → 本地 expert 计算 → all_reduce → 切回本 rank token
```

## 6. 适配器挂载 EP patch 的时机

对于采用 monkey-patch 方式的模型，不改权重目录内的 `modeling_*.py`：

- `ep_patches.py` 提供模型族对应的 patch 函数：

  遍历 `sys.modules` + 权重目录动态模块，对目标 MoE block 重写
  `__init__`（分片构造）、本地专家推理函数和 `forward`（DP+EP 闭环）。

- `model_adapter.py` 在 `init_model` 中调用两次：
  1. `AutoModelForCausalLM.from_config` 之前（确保 MoE 构造用上 patch）；
  2. `from_config` 之后（覆盖动态加载的 modeling 模块）。
- patch 加 `_msmodelslim_ep_patched` 标记，避免重复 patch。

如果目标模型不便于 monkey-patch，
则直接在模型类内部实现分片 + EP forward。

## 7. 层内动态加载（layer-wise load）与 EP 的配合

- `_load_decoder_if_not_exist(model, name, idx)` 动态构造 decoder layer 时，

  使用**当前 rank 的 `[expert_start, expert_end)`** 构造 MoE，保证只物质化本地专家。

- `_get_state_dict(module, prefix)` 按 `module.named_parameters()` 收集权重 key：

  非本地 expert 槽位为 `None`，天然不会出现在 `named_parameters()` 中，
  因此 layer-wise loader 不会读取非本地 expert 权重。

- 若 adapter 依赖 `get_weight_map()`（读 `model.safetensors.index.json`），

  注意 index 中**包含全部专家 key**；读取时必须以本地 expert 的 param name 为白名单过滤。

## 8. EP_CHECK 日志模板

在专家完成实例化 / 权重加载之后打印（统计真实 materialized 数量）：

```python
from msmodelslim.utils.logging import get_logger
logger = get_logger()

actual_local_ids = [
    eid for eid in local_expert_ids
    if self.experts[eid] is not None
]

logger.info(
    "[EP_CHECK] rank=%d ep_rank=%d ep_size=%d layer=%s "
    "total_experts=%d local_experts=%d expert_range=[%d,%d) non_local_experts=%d",
    global_rank, ep_rank, ep_size, layer_id,
    total_experts, len(actual_local_ids),
    expert_start, expert_end,
    total_experts - len(actual_local_ids),
)
```

若采用 compact local expert container，统计容器中实际 materialized 数量并打印
对应 global expert ids。

## 9. EP_ACT_GATE 日志模板（数值门禁）

结构日志 `[EP_CHECK]` 通过后，再输出数值门禁日志（单卡量化 vs 多卡 EP 量化逐层激活余弦相似度）：

```python
logger.info(
    "[EP_ACT_GATE] quant_ref=single_card ep_size=%d anchors=%d min_cos=%.6f worst_norm_dev=%.2e verdict=%s",
    ep_size, len(anchor_names), min_cos, worst_norm_dev, verdict,
)
```

若失败，追加输出最早发散的层：

```python
logger.info("[EP_ACT_GATE] first_diverged_layer=%s cos=%.6f", first_layer, first_cos)
```

数值门禁的完整原理、阈值与参考实现见 `ep_activation_gate.md`（EP Check 7）。

## 10. 通用检查项

1. `num_experts % ep_size == 0`；
2. expert container 中非本地槽位为 `None` / 未 materialize；
3. forward 中所有 `collective`（all_gather / all_reduce）在所有 rank 上**同序执行**；
4. 每个 rank 的 `[EP_CHECK]` 打印的实际专家数 == `total_experts / ep_size`。
