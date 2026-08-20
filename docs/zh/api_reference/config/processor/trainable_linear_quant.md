<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.trainable_linear_quant.config.processor_config.TrainableLinearQuantProcessorConfig -->
# trainable_linear_quant 配置说明

## 1. 配置概述

可训练线性量化（TLQ）处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `TrainableLinearQuantProcessorConfig` |
| 源码 | [processor_config.py](../../../../../msmodelslim/processor/trainable_linear_quant/config/processor_config.py) |

## 2. 参数列表

<h3 id="2-1-trainable-linear-quant">2.1 TrainableLinearQuantProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `trainable_linear_quant` | `trainable_linear_quant` | 处理器类型，固定为 `trainable_linear_quant`。 | 无 |
| `operations` | `list[object]` | 可选 | `[MinmaxTuneOpConfig(type='minmax_tune', lr=None), RoundTuneOpConfig(type='round_tune', lr=None)]` | 最少1项 | 可训练量化管线 OP 配置列表；每项含 type，其余字段由各插件定义 | 本页 <a href="#2-2-tlq-op-config">§2.2</a> |
| `strategies` | `list[object]` | 可选 | `[]` | 最少1项 | 量化策略配置列表；未提供时为空列表，不应用量化策略；若显式提供则至少 1 项。 | 本页 <a href="#2-6-tlq-quant-strategy-config">§2.6</a> |
| `train_with_act_quant` | `bool` | 可选 | `false` | — | 块级训练前向是否对激活做伪量化（经 x_kernel）；false 与 autoround 的 train_with_act_quant=False 一致；导出 IR 仍由 qconfig.act 决定，不受此项影响 | 无 |
| `enable_quanted_input` | `bool` | 可选 | `false` | — | 是否将本层量化前向结果作为下一层训练/量化传播的旁路输入（q_input）；不影响浮点 teacher：Runner 层间 datas 始终传递 teacher 输出 | 无 |
| `train_config` | `object` | 可选 | 见嵌套配置默认值 | — | 块级 Trainer 超参：iters、gradient_accumulate_steps、select_best、lr（或 learning_rate）、loss_type；各 OP 可单独配置 lr 覆盖全局值 | 本页 <a href="#2-9-block-train-config">§2.9</a> |

**配置约束**

- 归一化 operations：接受单个 dict 或列表；未提供或格式不合法时回退为默认 minmax_tune + round_tune 管线。

<h3 id="2-2-tlq-op-config">2.2 TLQOpConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 必选 | 无 | — | 算子类型，分派具体 TLQ 算子（如 `minmax_tune`、`round_tune`）。 | 无 |
| `lr` | `float / null` | 可选 | `null` | >0.0 | 该 Op 可训练参数学习率；未指定时使用 train_config.lr | 无 |

**配置约束**

- 无。

**派生类**

- `MinmaxTuneOpConfig`（`type: minmax_tune`） — `MinmaxTuneOpConfig` 是嵌套配置。 本页 <a href="#2-3-minmax-tune">§2.3</a>
- `RoundTuneOpConfig`（`type: round_tune`） — `RoundTuneOpConfig` 是嵌套配置。 本页 <a href="#2-4-round-tune">§2.4</a>
- `TrainableSmoothOpConfig`（`type: trainable_smooth`） — `TrainableSmoothOpConfig` 是嵌套配置。 本页 <a href="#2-5-trainable-smooth">§2.5</a>

<h4 id="2-3-minmax-tune">2.3 MinmaxTuneOpConfig</h4>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `minmax_tune` | `minmax_tune` | 插件类型：minmax_tune | 无 |
| `lr` | `float / null` | 可选 | `null` | >0.0 | — | 无 |

**配置约束**

- 无。

<h4 id="2-4-round-tune">2.4 RoundTuneOpConfig</h4>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `round_tune` | `round_tune` | 插件类型：round_tune | 无 |
| `lr` | `float / null` | 可选 | `null` | >0.0 | — | 无 |

**配置约束**

- 无。

<h4 id="2-5-trainable-smooth">2.5 TrainableSmoothOpConfig</h4>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `trainable_smooth` | `trainable_smooth` | 插件类型：trainable_smooth | 无 |
| `lr` | `float / null` | 可选 | `null` | >0.0 | — | 无 |
| `enable_subgraph_type` | `list[string]` | 可选 | `['norm-linear', 'linear-linear', 'ov', 'up-down', 'non-fusion']` | — | 启用的 Smooth 子图类型，须为 SMOOTH_SUPPORTED_SUBGRAPH_TYPES 子集 | 无 |
| `include` | `list[string] / null` | 可选 | `null` | — | 子图入口 include 通配 | 无 |
| `exclude` | `list[string] / null` | 可选 | `null` | — | 子图入口 exclude 通配 | 无 |

**配置约束**

- 无。

<h3 id="2-6-tlq-quant-strategy-config">2.6 QuantStrategyConfig</h3>

trainable_linear_quant 量化策略：对匹配的线性层应用一组可训练量化配置。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `qconfig` | `object` | 必选 | 无 | — | 激活与权重的量化配置，见《LinearQConfig 配置说明》。 | 本页 <a href="#2-7-linear-qconfig">§2.7</a> |
| `include` | `list[string]` | 可选 | `['*']` | — | 包含的模块名称模式，默认 `*` 匹配全部模块 | 无 |
| `exclude` | `list[string]` | 可选 | `[]` | — | 排除的模块名称模式，优先级高于 `include` | 无 |

**配置约束**

- 校验 qconfig：dtype/method 组合须有对应 TLQ kernel 支持，否则报错。

<h3 id="2-7-linear-qconfig">2.7 LinearQConfig</h3>

线性层（Linear）的量化配置，含激活与权重两路量化。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `act` | `object` | 可选 | `{'dtype': 'float', 'scope': 'per_tensor', 'symmetric': True, 'method': 'none', 'ext': {}}` | — | 激活值的量化配置。默认 `float`（不量化激活），仅对权重做量化。 | 本页 <a href="#2-8-qconfig">§2.8</a> |
| `weight` | `object` | 必选 | 无 | — | 权重的量化配置，必选。 | 本页 <a href="#2-8-qconfig">§2.8</a> |

**配置约束**

- 无。

<h3 id="2-8-qconfig">2.8 QConfig</h3>

描述单个张量（权重或激活）的量化方式。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `dtype` | `string` | 必选 | 无 | `float`、`int8`、`int4`、`mxfp8`、`mxfp4`、`fp8_e4m3` | 量化数据类型，如 `int8`、`int4`、`mxfp8`、`mxfp4`、`fp8_e4m3`；`float` 表示该张量不量化。 | 无 |
| `scope` | `string` | 必选 | 无 | `per_tensor`、`per_channel`、`per_group`、`per_block`、`per_token`、`pd_mix`、`per_head`、`dual_scale` | 量化粒度，即 scale/zero_point 的计算范围：`per_tensor`（整张量一个尺度）、`per_channel`（按通道）、`per_group`/`per_block`（按分组或固定块）、`per_token`（按 token）、`per_head`（按注意力头）、`dual_scale`（双尺度）等；合法取值组合取决于 `dtype` 与量化器实现。 | 无 |
| `symmetric` | `bool` | 必选 | 无 | — | 是否对称量化。对称量化只保存 scale；非对称量化额外保存 zero_point，可用性取决于 `dtype`/`scope` 组合。 | 无 |
| `method` | `string` | 必选 | 无 | — | 量化参数估计算法，如 `minmax`、`mse_round`、`histogram`、`ssz`、`none` 等；可用取值取决于 `dtype`/`scope`/`symmetric` 组合，`none` 表示不估计参数（配合 `float` 使用）。 | 无 |
| `ext` | `object` | 可选 | `{}` | — | 量化器扩展参数，随 `method` 与量化器实现而定（如 gptq 的 `percdamp`/`group_size`）；空对象表示无扩展参数。 | 无 |

**配置约束**

- 无。

<h3 id="2-9-block-train-config">2.9 BlockTrainConfig</h3>

块级训练（block train）超参配置。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `iters` | `int` | 可选 | `50` | ≥0 | 块级训练迭代次数；为 0 时 Trainer 跳过优化 | 无 |
| `gradient_accumulate_steps` | `int` | 可选 | `8` | ≥1 | 梯度累加步数，用于在有限显存下调节等效 batch | 无 |
| `lr` | `float` | 可选 | `0.01` | >0.0 | 全局基础学习率 | 无 |
| `select_best` | `object` | 可选 | 见嵌套配置默认值 | — | 最优 iter 快照策略（按 mode 区分字段：ema / min_loss / last） | 本页 <a href="#2-10-selectbestconfig">§2.10</a> |
| `loss_type` | `string` | 可选 | `l1` | `l1`、`custom_outlier` | 块级训练损失：l1（L1Loss reduction=none）、custom_outlier（0.3*全量 L1 + 0.7*3σ 内区域 L1） | 无 |
| `train_seed` | `int` | 可选 | `42` | — | 块级训练随机种子（用于 sample 打乱与确定性算子） | 无 |

**配置约束**

- 无。

<h3 id="2-10-selectbestconfig">2.10 SelectBestConfig（按 `mode` 分派）</h3>

**派生类**

- `EmaSelectBest`（`mode: ema`） — EMA 滑动平均选最优；支持 early stop。 本页 <a href="#2-11-ema-select-best">§2.11</a>
- `MinLossSelectBest`（`mode: min_loss`） — 当轮 loss 历史最小值选最优；支持 early stop。 本页 <a href="#2-12-min-loss-select-best">§2.12</a>
- `LastSelectBest`（`mode: last`） — 仅保存 iter 0 与最后一轮；无 early stop。 本页 <a href="#2-13-last-select-best">§2.13</a>

<h4 id="2-11-ema-select-best">2.11 EmaSelectBest</h4>

EMA 滑动平均选最优；支持 early stop。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `mode` | `string` | 可选 | `ema` | `ema` | 选取策略：ema | 无 |
| `ema_beta` | `float` | 可选 | `0.7` | >0.0；≤1.0 | best_loss 的 EMA 衰减系数 | 无 |
| `ema_window_size` | `int` | 可选 | `5` | ≥1 | mean_loss 滑动平均窗口长度 | 无 |
| `early_stop_patience` | `int` | 可选 | `-1` | ≥-1 | 连续多少 iter 无更优快照后早停；-1 表示禁用 | 无 |

**配置约束**

- 无。

<h4 id="2-12-min-loss-select-best">2.12 MinLossSelectBest</h4>

当轮 loss 历史最小值选最优；支持 early stop。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `mode` | `string` | 可选 | `min_loss` | `min_loss` | 选取策略：min_loss | 无 |
| `early_stop_patience` | `int` | 可选 | `-1` | ≥-1 | 连续多少 iter 无更优快照后早停；-1 表示禁用 | 无 |

**配置约束**

- 无。

<h4 id="2-13-last-select-best">2.13 LastSelectBest</h4>

仅保存 iter 0 与最后一轮；无 early stop。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `mode` | `string` | 可选 | `last` | `last` | 选取策略：last | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: trainable_linear_quant
    operations:
    - type: minmax_tune
      lr: null
    - type: round_tune
      lr: null
    strategies:
    - qconfig:
        act:
          dtype: float
          scope: per_tensor
          symmetric: true
          method: none
          ext: {}
        weight:
          dtype: int8
          scope: per_channel
          symmetric: true
          method: minmax
          ext: {}
      include:
      - '*'
      exclude: []
    train_with_act_quant: false
    enable_quanted_input: false
    train_config:
      iters: 50
      gradient_accumulate_steps: 8
      lr: 0.01
      select_best:
        mode: ema
        ema_beta: 0.7
        ema_window_size: 5
        early_stop_patience: -1
      loss_type: l1
      train_seed: 42
```
