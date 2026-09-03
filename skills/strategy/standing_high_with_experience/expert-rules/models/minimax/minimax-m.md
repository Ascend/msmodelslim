# MiniMax-M2.7 / M3 专属经验

> 层级：L3。结构模型类背景同 L2「MoE」「自定义 modeling」。本节记该模型独有坑。

## 1. `block_sparse_moe.experts` 命名（M2.7 独有）

- M2.7 的专家命名用 `*block_sparse_moe.experts*`（块稀疏 MoE），非标准 `*mlp.experts*`；若照搬 L2 常规前缀会匹配不到层。
- 原因：M2.7 采用了自定义 modeling 实现，模块命名约定与标准 transformers 不同，必须取真实 `named_modules` 确认。专家意见可信度：高。

## 2. W8A8 mxfp8 路径（M2.7）

- `quarot(export_extra_info: True)` + `linear_quant`（mxfp8 per_block + minmax），include 仅 `*block_sparse_moe.experts*` 与 `*self_attn*`，其余不量化。
- 原因：MXFP8 per_block 粒度在块内分布更集中，适合该模型稳定低比特；仅量化 experts 与 attention 主体，其他结构保持浮点以保精度。专家意见可信度：中。

## 3. W8A8 int 路径（M3）

- `iter_smooth`（alpha 0.5、scale_min 1e-5、symmetric True、enable `norm-linear`）+ 全局 `w8a8_dynamic`。
- 大范围排除：`*norm*`/`*embed_tokens*`/`*lm_head*`/`*vision_tower*`/`*multi_modal_projector*`/`*patch_merge_mlp*`/`*mlp.down_proj*`/`*shared_experts.down_proj*`/`*mlp.gate`/`*indexer*`/`*index_*`。
- 说明 M3 是 VLM 且含 indexer/跨模态结构，回退面较大，属该模型类「先保 LLM 主体、大量外围排除」的实践。原因：跨模态投影与索引结构连接不同模态空间，量化误差易破坏模态对齐，因此初版先全部排除，仅量化 LLM 主体。专家意见可信度：中。
