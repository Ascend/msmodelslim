# Gemma4-MoE 量化说明

## 模型介绍

[Gemma 4](https://ai.google.dev/gemma) 是 Google DeepMind 推出的开源多模态模型系列。其中 [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) 为指令微调的 Mixture-of-Experts（MoE）变体：

- **稀疏 MoE 架构**：总参数约 26B，每 token 激活约 4B（A4B）。
- **多模态理解**：原生支持文本与图像输入，生成文本输出。
- **长上下文**：上下文窗口最长可达 256K tokens。
- **3D 融合专家权重**：浮点权重中 `experts.gate_up_proj` / `experts.down_proj` 以 3D 张量存储，量化前需拆分为 per-expert `nn.Linear`。

> [!NOTE]
>
> 本适配器面向 `enable_moe_block=True` 的 MoE 检查点。密集版 Gemma4（如 gemma-4-31B-it）请使用对应 dense 适配器，不要使用 `--model_type gemma-4-26B-A4B-it`。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../docs/zh/install_guide/install_guide.md)》。
- 针对 Gemma4-MoE，transformers 版本需为 5.5.3，安装命令如下：

  ```bash
  pip install transformers==5.5.3
  ```

## Gemma4-MoE 模型当前已验证的量化方法

| 模型 | 原始浮点权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|-------------|---------|----------------|---------|
| gemma-4-26B-A4B-it | [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | W8A8 动态量化（MoE experts） | vLLM Ascend 支持 | [W8A8 动态量化](#gemma-4-26b-a4b-it-w8a8-动态量化) |

**说明：**

- 点击量化命令列中的链接可跳转到对应的具体量化命令。
- 当前实践配置对 `*experts*` 做 W8A8 动态量化（激活 `per_token`、权重 `per_channel`），`vision_tower` / `embed_vision` / `router` 保持浮点。
- 该配置为 data-free 动态量化：量化阶段可不跑前向校准；若后续使用混合/静态 W8A8 配置，仍需提供图像+文本校准样本。

## 校准数据说明

校准数据支持的方式，详见《[dataset 配置说明](../../../docs/zh/user_guide/usage_quick_quantization.md#526-dataset---校准数据集配置)》。

- **当前 W8A8 动态实践（`gemma4_moe_w8a8.yaml`）**：data-free，可不依赖校准前向。
- **混合/静态激活量化**：每条样本需同时提供 `image` 与 `text`；缺项样本暂不支持。

## 生成量化权重

### <span id="gemma-4-26b-a4b-it-w8a8-动态量化">gemma-4-26B-A4B-it W8A8 动态量化</span>

该模型的量化已集成至一键量化，示例参数详见文档《一键量化完整指南》中的“[参数说明](../../../docs/zh/user_guide/usage_quick_quantization.md#32-参数说明)”章节。实践配置见《[gemma4_moe_w8a8.yaml](../../../lab_practice/gemma4_moe/gemma4_moe_w8a8.yaml)》。

```shell
msmodelslim quant \
    --model_path /path/to/gemma-4-26B-A4B-it \
    --save_path /path/to/gemma4_moe_w8a8 \
    --device npu \
    --model_type gemma-4-26B-A4B-it \
    --quant_type w8a8 \
    --trust_remote_code True
```

## 附录

### 相关资源

- 《[一键量化配置协议说明](../../../docs/zh/user_guide/usage_quick_quantization.md#5-量化配置协议详解)》。
- 《[multimodal_vlm_modelslim_v1 量化服务配置详解](../../../docs/zh/user_guide/usage_quick_quantization.md#54-multimodal_vlm_modelslim_v1-配置详解)》。
- 《[实践配置 gemma4_moe_w8a8.yaml](../../../lab_practice/gemma4_moe/gemma4_moe_w8a8.yaml)》。
