"""
Expert-parallel runtime patch for Qwen3-Omni-MoE thinker blocks.

The patch targets the unfused expert representation in transformers 4.57.3.
It keeps global expert indices stable by using a full-length ModuleList whose
non-local entries are None. This also lets DistHelper classify instantiated
experts as local-only modules for distributed quantization and saving.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from msmodelslim.model.common.utils import resolve_expert_ep_range
from msmodelslim.utils.logging import get_logger


# 校准集是否在所有 rank 上重复（样本数少于卡数）。由 model_adapter.handle_dataset
# 设置，供 ep_forward 判断是否可以跳过 all_gather：各 rank 持有同一份完整样本时，
# gather 只会把相同 token 复制 world_size 份，既浪费算力，也会让激活统计变成
# 单卡的 world_size 倍重复视图。
_CALIBRATION_REPLICATED = False


def set_calibration_replicated(value: bool) -> None:
    """Record whether every rank iterates the same full calibration dataset."""
    global _CALIBRATION_REPLICATED
    _CALIBRATION_REPLICATED = bool(value)


def is_calibration_replicated() -> bool:
    """Return True when the calibration set is duplicated on every rank."""
    return _CALIBRATION_REPLICATED


def _accumulate_expert_range(
    experts: nn.ModuleList,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_start: int,
    expert_end: int,
) -> torch.Tensor:
    """Sum the routed contributions of experts ``[expert_start, expert_end)``.

    The accumulator stays in float32 and experts are added in ascending global id
    order, which makes the result a deterministic function of the inputs.  Casting
    to the activation dtype here would round every EP rank's partial sum to bf16
    before the cross-rank reduction (~2^-9 relative error per rank), a much larger
    error source than the single-card path, which rounds each expert contribution
    to the activation dtype before ``index_add_``.
    """
    hidden_dim = hidden_states.shape[-1]
    output = torch.zeros(
        hidden_states.shape[0],
        hidden_dim,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for expert_id in range(expert_start, expert_end):
        expert = experts[expert_id]
        if expert is None:
            continue
        token_idx, topk_idx = torch.where(selected_experts == expert_id)
        if token_idx.numel() == 0:
            continue
        expert_output = expert(hidden_states[token_idx]).to(torch.float32)
        weights = routing_weights[token_idx, topk_idx].unsqueeze(-1).to(torch.float32)
        output.index_add_(0, token_idx, expert_output * weights)
    return output


def moe_infer_local_experts(
    experts: nn.ModuleList,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_start: int,
    expert_end: int,
) -> torch.Tensor:
    """Compute the partial MoE output owned by one EP rank."""
    return _accumulate_expert_range(experts, hidden_states, selected_experts, routing_weights, expert_start, expert_end)


def _sum_in_rank_order(partials: List[torch.Tensor]) -> torch.Tensor:
    """Add partial sums left-to-right so the float rounding order is fixed."""
    acc = partials[0]
    for part in partials[1:]:
        acc = acc + part
    return acc


def _gather_partials(local_partial: torch.Tensor) -> List[torch.Tensor]:
    """Collect every rank's partial into rank order with one contiguous collective."""
    world_size = dist.get_world_size()
    local_partial = local_partial.contiguous()
    if hasattr(dist, "all_gather_into_tensor"):
        gathered = local_partial.new_empty((world_size, *local_partial.shape))
        dist.all_gather_into_tensor(gathered, local_partial)
        return list(gathered.unbind(0))
    gathered = [torch.empty_like(local_partial) for _ in range(world_size)]
    dist.all_gather(gathered, local_partial)
    return gathered


def _gather_hidden_states(hidden_states: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Gather variable sequence lengths and return this rank's output slice."""
    from msmodelslim.utils.distributed import DistHelper

    local_seq_len = hidden_states.size(1)
    seq_len_tensor = torch.tensor([local_seq_len], dtype=torch.long, device=hidden_states.device)
    seq_len_tensors = [torch.zeros_like(seq_len_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(seq_len_tensors, seq_len_tensor)
    seq_lens = [int(item.item()) for item in seq_len_tensors]
    rank = dist.get_rank()
    start = sum(seq_lens[:rank])
    end = start + local_seq_len
    gathered = torch.cat(DistHelper.gather_variable_shapes(hidden_states), dim=1)
    return gathered, start, end


def _patch_sparse_moe_block(modeling) -> bool:
    """Patch one transformers Qwen3-Omni modeling module."""
    block_cls = getattr(modeling, "Qwen3OmniMoeThinkerTextSparseMoeBlock", None)
    expert_cls = getattr(modeling, "Qwen3OmniMoeThinkerTextMLP", None)
    if not isinstance(block_cls, type) or not isinstance(expert_cls, type):
        return False
    if block_cls.__dict__.get("_msmodelslim_ep_patched", False):
        return False

    # Keep the upstream forward for the single-card path: it casts every expert
    # contribution back to the activation dtype before index_add_, which is the
    # baseline rounding the multi-card float32 partial sum is designed around.
    original_forward = block_cls.forward

    def ep_init(self, config):
        super(block_cls, self).__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        ep_size, ep_rank, expert_start, expert_end = resolve_expert_ep_range(self.num_experts)
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.experts_start_idx = expert_start
        self.experts_end_idx = expert_end
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                expert_cls(config, intermediate_size=config.moe_intermediate_size)
                if expert_start <= expert_id < expert_end
                else None
                for expert_id in range(self.num_experts)
            ]
        )
        if ep_size > 1:
            get_logger().info(
                "Qwen3-Omni MoE EP enabled: rank=%s/%s experts=[%s, %s) of %s",
                ep_rank,
                ep_size,
                expert_start,
                expert_end,
                self.num_experts,
            )

    def ep_forward(self, hidden_states: torch.Tensor):
        if self.training:
            raise NotImplementedError("Training mode is not supported by the Qwen3-Omni EP patch.")

        # Single card keeps the upstream per-expert rounding; float32 accumulation
        # is only needed to combine partial sums across EP ranks.
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return original_forward(self, hidden_states)

        # DP 分片时各 rank 持有不同样本，必须先 all_gather 拼成全量 token，
        # 再让本地专家计算部分和、按 rank 顺序合并、slice 回本 rank 区间。
        #
        # 但当校准集无法无重复分片（例如 calibVideos 只有 1 个样本）时，
        # 框架会让所有 rank 遍历同一份完整校准集。此时各 rank 的 hidden_states
        # 是同一份样本，再 all_gather 只会把同一批 token 重复 world_size 份：
        # 专家激活统计变成单卡的 world_size 倍重复视图，还多做数倍无用计算。
        #
        # 此时跳过 gather，各 rank 直接在本 rank 的 token 上计算本地专家部分和，
        # 再按 rank 顺序合并得到完整 MoE 输出。各 rank 持有完整样本，
        # 专家统计与单卡一致，也不需要猜测重复块边界。
        if not is_calibration_replicated():
            hidden_states, slice_start, slice_end = _gather_hidden_states(hidden_states)
        else:
            slice_start, slice_end = 0, hidden_states.size(1)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flattened = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(flattened)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flattened.dtype)

        local_partial = moe_infer_local_experts(
            self.experts,
            flattened,
            selected_experts,
            routing_weights,
            self.experts_start_idx,
            self.experts_end_idx,
        )
        # Every rank owns exactly one expert group; add the gathered partials
        # in ascending rank order so the float sequence is deterministic.
        output = _sum_in_rank_order(_gather_partials(local_partial))
        output = output.to(hidden_states.dtype)
        output = output.reshape(batch_size, sequence_length, hidden_dim)
        output = output[:, slice_start:slice_end, :]

        # Router logits are diagnostic-only during quantization, but preserve the
        # local-token shape expected by the upstream tuple contract.
        router_logits = router_logits.reshape(batch_size, sequence_length, self.num_experts)
        router_logits = router_logits[:, slice_start:slice_end, :].reshape(-1, self.num_experts)
        return output, router_logits

    block_cls.__init__ = ep_init
    block_cls.forward = ep_forward
    block_cls._msmodelslim_ep_patched = True
    return True


def is_ep_patch_applied(modeling_module: Optional[object] = None) -> bool:
    """Return whether the sparse MoE block already carries the EP patch."""
    if modeling_module is None:
        # pylint: disable-next=no-name-in-module
        from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe

        modeling_module = modeling_qwen3_omni_moe
    block_cls = getattr(modeling_module, "Qwen3OmniMoeThinkerTextSparseMoeBlock", None)
    return bool(isinstance(block_cls, type) and block_cls.__dict__.get("_msmodelslim_ep_patched", False))


def apply_qwen3_omni_moe_ep_patches(modeling_module: Optional[object] = None) -> int:
    """Apply the EP patch idempotently and return the number of patched modules."""
    if modeling_module is None:
        # pylint: disable-next=no-name-in-module
        from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe

        modeling_module = modeling_qwen3_omni_moe
    if _patch_sparse_moe_block(modeling_module):
        get_logger().info(
            "Patched Qwen3-Omni thinker sparse MoE block for Expert Parallelism (world_size=%s, reduce=rank_order).",
            dist.get_world_size() if dist.is_initialized() else 1,
        )
        return 1
    return 0
