# Kimi-K3 量化案例

## 模型介绍

Kimi-K3 是月之暗面（Moonshot AI）研发的原生多模态模型（约 2.8T 参数），采用混合注意力（KDA / MLA）与大规模 Latent MoE。本实践仅对语言侧 Decoder 做 W4A8 混合量化。

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

| 模型 | 原始权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|---------|---------|----------------|---------|
| Kimi-K3 | [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | W4A8 量化 | vLLM Ascend | [W4A8 量化](#Kimi-K3-w4a8) |

> [!note]
>
> 单击量化命令列中的链接可跳转到对应的具体量化命令。

## 使用示例

### <span id="Kimi-K3-w4a8">Kimi-K3 W4A8 量化</span>

该系列模型的量化已集成至[一键量化](../../../docs/zh/user_guide/usage_quick_quantization.md#32-参数说明)。推荐实践配置：[kimi_k3_w4a8.yaml](../../../lab_practice/kimi_k3/kimi_k3_w4a8.yaml)。

请将 `${model_path}`、`${save_path}` 替换为实际路径。

```shell
msmodelslim quant \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --device npu --device_id 0 1 2 3 4 5 6 7 \
    --model_type Kimi-K3 \
    --quant_type w4a8 \
    --trust_remote_code True
```

也可显式指定配置：

```shell
msmodelslim quant \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --device npu --device_id 0 1 2 3 4 5 6 7 \
    --model_type Kimi-K3 \
    --config lab_practice/kimi_k3/kimi_k3_w4a8.yaml \
    --trust_remote_code True
```
