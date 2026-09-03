# L3 模型专属专家调优意见经验（models/）

> **定位与阅读规则**
>
> - 本目录是**各 `<vendor>/<model>.md` 个案**，只放**该模型/该模型类独有、不可迁移**的调优坑与优化手段。
> - 通用手段（量化格式/离群值抑制/方法/粒度/校准集）见 **L1**；结构模型类共性（MoE/MLA/混合 attention/VLM/DiT）见 **L2**。L3 只在 L1+L2 基础上补充「只有这个模型才这样」的内容。
> - 每条给出专家原因与专家意见可信度；没有独立经验的模型不建假条目。

## 已收录模型索引

| vendor | 模型/模型类 | 文件 | 独有特征速览 |
|---|---|---|---|
| deepseek | DeepSeek-V3 / R1 / V3.1 / V3.2 | `deepseek/deepseek-v3.md` | MLA 低秩投影排除、fa3_quant per-head、`model.layers.61` 尾部敏感层 |
| deepseek | DeepSeek-V4-Pro / Flash | `deepseek/deepseek-v4.md` | `ffn` 命名（非 mlp）、`indexer`/`compressor` 投影排除、QuaRot block_size=32 |
| glm | GLM-4-MoE | `glm/glm-4-moe.md` | 末层 `model.layers.92.*` 排除 |
| glm | GLM-5.1 | `glm/glm-5.md` | W4A4 mxfp4、`part_file_size: 8` |
| minimax | MiniMax-M2.7 / M3 | `minimax/minimax-m.md` | `block_sparse_moe.experts`、iter_smooth、mxfp8 |

未列出的模型：先走 L1 + L2（按触发信号判定结构模型类），有独立实践 YAML 时再补 L3条目。
