# GLM-4-MoE 专属经验

> 层级：L3。结构模型类背景同 L2「MoE」。本节记该模型独有坑。

## 1. 末层 `model.layers.92.*` 全排（MTP 层）

- W8A8 场景：`flex_smooth_quant` 与 group 内各 `linear_quant` 均排除 `model.layers.92.*`（末层整体不量化）。
- **关键原因**：`model.layers.92` 是该模型的 **MTP（Multi-Token Prediction）层**，不是普通 transformer layer。GLM-4-MoE 在基础层之外追加了一层 MTP（见 model_adapter：`total_layers = num_hidden_layers + 1`），MTP 层包含 `embed_tokens`/`enorm`/`hnorm`/`eh_proj`/`shared_head` 等特殊子模块，其计算路径（token 拼接 + 双分支投影 + 共享 head 预测）与普通层不同，量化误差会直接损害最终生成质量，因此整层排除。
- 层号 92 是该规模（GLM-4.7）的实测结论：基础层 0~91 + MTP 层索引 92，**移植到其他规模/版本需先确认基础层数，再推导 MTP 层索引**，不可直接照搬 92。专家意见可信度：中（MTP 结构来源已确认，具体层号随规模变化）。

## 2. W8A8 结构拆分

- `flex_smooth_quant` 仅 enable `norm-linear`；`*self_attn*` 与 `*mlp.experts*` 均走 W8A8 动态（均排除末层 92）。
- `part_file_size: 8`（区别于多数别家默认 4）。专家意见可信度：中。
