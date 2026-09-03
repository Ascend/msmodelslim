# Kimi-K3 量化案例

## 模型介绍

[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) 是月之暗面（Moonshot AI）发布的原生多模态大模型（约 2.8T 总参数），面向图文理解与长上下文推理等场景。语言侧采用 **KDA（线性注意力）与 MLA（多头潜在注意力）混合** 的 Decoder 结构，并配合大规模 **Latent MoE**（含 routed / shared experts）；视觉侧为独立 Vision Encoder，与语言塔通过多模态对齐接入。

msModelSlim 当前适配器仅对 **语言侧 Decoder** 做权重量化（含 MoE 专家与注意力投影等），视觉塔保持浮点；校准需同时覆盖 image 与 text。官方权重部分模块以 MXFP4 压缩发布，加载时会先反量化为 BF16 再进入量化流程。

## 使用前准备

- 安装 msModelSlim，参见《[msModelSlim工具安装指南](../../../docs/zh/install_guide/install_guide.md)》。
- 安装依赖：

```bash
pip install transformers==4.57.6 compressed-tensors==0.13.0
pip install -U fla-core
```

- 推荐多卡量化，单卡亦可，但耗时更长、显存占用更高。
- 校准数据需同时包含 image 与 text。

## Kimi-K3 模型当前已验证的量化方法

| 模型 | 原始权重 | 量化方式 | 推理框架 | 量化命令 |
|------|---------|---------|---------|---------|
| Kimi-K3 | [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | W4A8（INT） | vLLM Ascend | [W4A8](#kimi-k3-w4a8-int) |
| Kimi-K3 | [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | W4A8C8（INT + FA3） | vLLM Ascend | [W4A8C8](#kimi-k3-w4a8c8) |
| Kimi-K3 | [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | W4A8（MXFP） | vLLM Ascend | [W4A8 MXFP](#kimi-k3-w4a8-mxfp) |

> [!note]
>
> 单击量化命令列中的链接可跳转到对应命令。仅 `--quant_type w4a8` 时默认命中 INT 实践；MXFP 需同时指定 `--tags`（如 `Ascend_950`）以匹配对应 `verified_tags`。

## 使用示例

请将 `${model_path}`、`${save_path}` 替换为实际路径。一键量化说明见《[一键量化完整指南](../../../docs/zh/user_guide/usage_quick_quantization.md)》。

### <span id="kimi-k3-w4a8-int">Kimi-K3 W4A8（INT）量化</span>

推荐实践：[kimi_k3_w4a8.yaml](../../../lab_practice/kimi_k3/kimi_k3_w4a8.yaml)。

路由专家与 shared experts 使用 INT4 权重量化，注意力 / Latent / dense MLP 等使用 INT8；前置 QuaRot、FlexAWQ-SSZ 与 FlexSmooth。

```shell
msmodelslim quant \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --device npu \
    --device_id 0 1 2 3 4 5 6 7 \
    --model_type Kimi-K3 \
    --quant_type w4a8 \
    --trust_remote_code True
```

### <span id="kimi-k3-w4a8c8">Kimi-K3 W4A8C8（INT + FA3）量化</span>

推荐实践：[kimi_k3_w4a8c8.yaml](../../../lab_practice/kimi_k3/kimi_k3_w4a8c8.yaml)。

在 W4A8（INT）流程基础上增加 `fa3_quant`（MLA 路径的 FA3 / KV 侧 INT8）。

```shell
msmodelslim quant \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --device npu \
    --device_id 0 1 2 3 4 5 6 7 \
    --model_type Kimi-K3 \
    --quant_type w4a8c8 \
    --trust_remote_code True
```

### <span id="kimi-k3-w4a8-mxfp">Kimi-K3 W4A8（MXFP）量化</span>

推荐实践：[kimi_k3_w4a8_mxfp.yaml](../../../lab_practice/kimi_k3/kimi_k3_w4a8_mxfp.yaml)。

专家 / shared 使用 MXFP4 权重 + MXFP8 激活（`per_block`），注意力等使用 MXFP8；前置 QuaRot 与 FlexSmooth。

与 INT W4A8 共用 `quant_type=w4a8`，须通过 `--tags` 匹配 MXFP 的验证场景（`Ascend_950`）：

```shell
msmodelslim quant \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --device npu \
    --device_id 0 1 2 3 4 5 6 7 \
    --model_type Kimi-K3 \
    --quant_type w4a8 \
    --tags vLLM_Ascend Ascend_950 \
    --trust_remote_code True
```
