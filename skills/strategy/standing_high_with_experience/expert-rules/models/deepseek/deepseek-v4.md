# DeepSeek-V4-Pro / Flash 专属经验

> 层级：L3。结构模型类背景同 L2 的「MoE」「MLA」「自定义 modeling」。本节记该模型类独有、不可迁移的坑。

## 1. 结构命名偏移（该模型类最易踩坑）

- V4 的 FFN 命名用 **`ffn` 而非 `mlp`**：`*.ffn.shared_experts.*`、`*ffn*`；shared experts 写作 `ffn.shared_experts`。
- 若照搬 L2/MoE 的 `*mlp.*` 前缀会导致**匹配不到任何层**，量化配置实际无效。
- 原因：V4 采用了自定义 modeling 实现，模块命名约定与标准 transformers 或 L2 归纳的常规模式不同；匹配不到层时配置静默生效但无实际效果「空量化」，肉眼不易察觉。处理前必须取真实 `named_modules` 确认。专家意见可信度：高。

## 2. `indexer` / `compressor` 投影排除（V4 独有）

- attention 相关低秩/索引压缩投影需排除：`*wo_a`、`*wo_b`、`*compressor.wgate`、`*compressor.wkv`、`*indexer.weights_proj`、`*indexer.compressor.wgate`、`*indexer.compressor.wkv`。
- 原因：这些是 V4 引入的 latent 空间索引/压缩侧投影，连接主注意力路径与辅助索引路径，处在不同表示空间的交界面；量化误差会在此处放大并跨空间传播，该清单为 V4 独有，不可迁移到无 indexer 的模型。专家意见可信度：中。

## 3. `quarot` block_size=32

- V4 实践 `quarot` 显式 `block_size: 32`（V3.2 模型类多为默认）。旋转块尺寸是 QuaRot 精度关键超参之一。
- 原因：V4 的 indexer/compressor 结构引入额外的激活分布多样性，显式指定 block_size=32 可在局部旋转与全局能量集中之间取得平衡；若用默认值可能对 V4 独特的激活模式适应性不足。专家意见可信度：中。

## 4. 格式与结构映射

- W8A8（Flash）：`quarot(32)` + `flex_smooth_quant(norm-linear)`，attn 用 `w8a8_dynamic`，`ffn*` 用 `w8a8_dynamic` 但排除 `*gate`。
- W4A8（Pro）：W4 低比特落在 `*ffn*`（排除 `*gate`、`*shared_experts*`），attn 保持 W8A8；`shared_experts` 单列 W8A8 保持高精度。
  - `flex_awq_ssz`（enable `up-down`）排除 `*.ffn.shared_experts.*`。
  - `flex_smooth_quant`（enable `norm-linear`）排除 `*ffn_norm*`。
- 原因：W8A8 场景下 `gate` 是路由决策层，量化误差会导致 token 路由到错误 expert，必须排除；W4A8 场景下 shared_experts 被所有 routed experts 复用，其误差会被全局放大，因此单独提级 W8A8。专家意见可信度：中。
