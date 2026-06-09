# Kimi K2.6 量化案例

## 模型介绍

Kimi K2.6 是月之暗面（Moonshot AI）研发的原生多模态模型，推动了长视野编码、编码驱动设计、主动自主执行和基于群体的任务编排等实用能力。主要特征包括：

- **长视野编码**：K2.6在复杂的端到端编码任务上取得了显著改进，能够在编程语言（Rust、Go、Python）及涵盖前端、DevOps和性能优化等领域间进行强有力的泛化。
- **编码驱动设计**：K2.6能够将简单的提示和视觉输入转化为生产界面和轻量级全栈工作流程，以精心设计的美学精准生成结构化布局、互动元素和丰富的动画。
- **提升代理群**：K2.6横向扩展至300个子代理，执行4000个协调步骤，能够动态将任务分解为并行的领域专用子任务，实现从文档到网站再到电子表格的端到端输出，实现一次自主运行。
- **主动与开放编排**：对于自主任务，K2.6在支持持久的全天候后台代理方面表现出强劲的能力，能够主动管理日程、执行代码并协调跨平台操作，无需人工监督。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../docs/zh/getting_started/install_guide.md)》。
- 对于Kimi K2.6系列模型，由于模型参数量较大，请先完成“运行前必检”（[Kimi K2.6运行前必检](#运行前必检)）。
- 由于模型量化（Model Quantization）对显存要求较高，请确保在单卡显存不低于64G的环境下执行。
- 需安装 `compressed-tensors`（用于加载原生量化模型）：

  ```bash
  pip install compressed-tensors==0.13.0
  ```

## Kimi-K2.6 模型当前已验证的量化方法

| 模型 | 原始浮点权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|-------------|---------|----------------|---------|
| Kimi-K2.6 | [Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) | W4A8 量化 | vLLM Ascend | [W4A8 量化](#Kimi-K2.6-w4a8) |

> [!note]
>
> 单击量化命令列中的链接可跳转到对应的具体量化命令。

## 使用示例

### <span id="Kimi-K2.6-w4a8">Kimi-K2.6 W4A8 量化</span>

该系列模型的量化已集成至[一键量化](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#参数说明)。

```shell
msmodelslim quant \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --device npu \
    --model_type Kimi-K2.6 \
    --quant_type w4a8 \
    --trust_remote_code True
```

## 附录

### <span id="运行前必检">运行前必检</span>

Kimi K2.6模型采用混合专家（MoE）架构，参数量较大且存在需要手动适配的点，为了避免浪费时间，还请在运行脚本前，根据以下必检项对相关内容进行更改。

1、昇腾（Ascend）不支持`@torch.compile(dynamic=True)`，运行时需要注释掉权重文件夹中相关modeling文件中的部分代码，具体位置见[**图1 需要注释的代码**](#需要注释的代码)：

**图1 需要注释的代码**<a id="需要注释的代码"></a>

![img_1.png](img_1.png)

2、原始权重文件`modeling_kimi_k25.py`中存在代码错误，具体错误位置见[**图2 原始权重文件**](#原始权重文件)，需检查修改，替换代码参考如下：

```python
self.blocks = nn.ModuleList([
    MoonViTEncoderLayer(
        **block_cfg,
        use_deterministic_attn=getattr(self, "use_deterministic_attn", False))
    for _ in range(num_layers)
])
```

**图2 原始权重文件**<a id="原始权重文件"></a>

![img_2.png](img_2.png)
