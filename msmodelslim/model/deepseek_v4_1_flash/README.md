# DeepSeek-V4.1-Flash-W8A8

> **ModelAdapter 迁移**：`DeepSeekV41FlashModelAdapter`（本目录的 `model_adapter.py`，经 `config.ini`
> 的 `deepseek_v4_1_flash` 模型族注册）已经把上游手搓量化流程迁移到标准 `modelslim_v1` 管道：源检查点
> decode 为 BF16，norm folding + QuaRot 旋转由标准 `quarot` 处理器（`block_size: 32`）经
> 适配器实现的 `QuaRotInterface` 完成，`linear_quant` 按最佳实践 yaml 的 `include` 选
> bit-width，`ascendv1_saver` 落盘后由 `ascendv1_save_postprocess` 补齐 Engram / 旋转
> serving 合同。最佳实践见 `lab_practice/deepseek_v4_1_flash/`
> （`deepseek_v4_1_flash_w4a8_ascend_cfg.yaml` 与 `README.md`）。作为转换标准 ModelAdapter
> 的一部分，旧 `modelslim_convert` 数据面路径已移除，不再向后兼容既有转换 CLI。

内部测试量化权重：主干与 DSpark 指定 Linear 为 W8A8 Dynamic；ViT/Aligner 和 Engram 为 BF16。

Engram 表直接由原始 FP8 E4M3 + group32 UE8M0 scale 解码为 BF16，不经过 INT8 中转；
以完整张量 `layers.1.engram.embed.weight`、`layers.14.engram.embed.weight` 保存，
无 embedding quant scale、`.parts.*` 或 compressed-tensors 配置。
整体继续使用 `quant_method: ascend`。

当前制作统一走 `lab_practice/deepseek_v4_1_flash/` 的 `modelslim_v1` 配置。主干执行
norm folding 和 QuaRot；DSpark 仅 main_proj 输入适配主干旋转，embed/head 保留源值副本。

Engram 的 K 与 gate 保持原模型基底，gate 输入仍需恢复 `h_rot @ Q.T`。`wkv`
最后 5120 行 V 已折叠 `Q.T @ Wv`，直接输出旋转基底的 V；用 `h_rot + gate * v_rot`
更新 residual，禁止再次对增量乘 Q。具体约定见 `config.json` 的
`engram_rotation_config`。ViT 输出边界已旋转。`optional/quarot.safetensors`
保存 global/query 旋转矩阵。

权重转换与推理适配是独立验收项。当前尚未签核该 DeepSeek-V4.1-Flash 的完整 vLLM/vllm-ascend 推理或整模型任务精度。转换与校验记录位于单独的制作目录，不属于权重加载依赖。

Engram 源格式描述位于 `quant_model_description.json` 的
`optional.embedding_storage`（version 2）：每个 `layers.N.engram.embed` 模块记录
`format: bf16`、`weight_dtype: BF16`、`weight_shape: [rows, 256]` 和
`layout: row_major`，没有 scale。dtype 名称与 safetensors header 一致；原有
`FLOAT` tensor tag 和普通 W8A8 Linear 描述保持不变。

这些字段只描述 checkpoint 实际源格式。`engram_storage`、`engram_load_dtype`
由 vLLM 的运行时配置选择，不写入 checkpoint；BF16 源可直接加载，也可由 loader
分块转为 INT8。Engram hash、projection、gate 和 V 旋转合同保持不变。
