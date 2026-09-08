# Trainable Linear Quant 参数配置流程指南

## 1. 适用范围

Trainable Linear Quant（TLQ）可训练线性量化算法。TLQ 作为线性层量化处理器，通常写在量化 YAML 的 `spec.process` 中，通过块级训练优化量化参数与舍入，**主要用于 W4A4 等极低比特场景的精度增强**。

本指南面向需要**把 TLQ 配进量化任务**的用户：说明推荐起步配置、关键参数何时调整，以及模型适配要注意什么。

- **适用**：目标为 W4A4 等低比特线性量化，一次标定精度不足，可接受更高量化耗时与显存占用以换取更好重构精度。
- **不适用**：仅需 W8 等较高位宽且一次标定已够用；目标 dtype/scope/method 无对应 TLQ kernel；或设备显存/时长无法支撑块级训练。

> **资源提示**：TLQ 含块级训练，量化耗时与显存显著高于 MinMax / Linear Quant 等一次标定算法；正式跑数前请预留充足 NPU 显存与时间预算。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 量化任务配置 | 用户 YAML 或 `lab_practice` 配方 | 符合 `modelslim_v1`（或对应任务协议），可插入 `spec.process` | 能被量化流程加载 |
| 交付件 | TLQ 配置片段 | 写入上述 YAML 的 `process` 项 | `type: trainable_linear_quant`，字段合法；OP 与适配能力匹配 | 可随完整配置复现量化 |

> 模型适配、校准数据属于运行量化的前置条件，在步骤 1 核对；本表只列本指南直接改动的配置输入与交付件。

## 3. 流程总览

本指南的操作顺序如下（与第 4 章步骤一一对应）：

```mermaid
flowchart LR
    A[确认适配与作用范围] --> B[写入推荐配置]
    B[写入推荐配置] --> C[按需调整参数]
    C[按需调整参数] --> D[验证量化效果]
```

## 4. 操作步骤

### 步骤 1：确认适配与作用范围

**目标**：按拟用的 `operations` 确认目标 `model_type` 适配能力，并明确量化作用范围。

**操作**：

按 OP / 数据路径核对适配要求：

| 适配项 | 接口 / 方法 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| `minmax_tune` + `round_tune` | 无额外子图接口 | 否 | 模型已接入通用量化流水线即可 |
| 块 I/O 定制 | `TLQBlockDataInterface` | 可选 | 未实现时使用默认块数据处理；复杂或非常规块结构须由适配器补齐。接口见 [block_data.py](../../../../../msmodelslim/processor/trainable_linear_quant/data/block_data.py)，从 `msmodelslim.model.interface_hub` 导入 |
| `trainable_smooth` | `get_adapter_config_for_subgraph()` | **启用时必须** | 须返回非空 `List[AdapterConfig]`；可复用 IterSmooth / OASQ 同名方法。接口约定见 [interface.py](../../../../../msmodelslim/processor/trainable_linear_quant/interface.py) |

1. **选择 OP 路径**：按目标按需配置 `operations`；`minmax_tune`、`round_tune`、`trainable_smooth` **均可独立使用**。`minmax_tune` + `round_tune` 联用时侧重可学习量化参数与舍入，效果大体等效于 [AutoRound](../autoround/term_autoround.md)；`trainable_smooth` 侧重离群值抑制，**不可**与前两者联用。
2. **模型适配**：仅使用 `minmax_tune` 和/或 `round_tune` 时，一般无需额外接口；默认块数据处理不适用时补 `TLQBlockDataInterface`。使用 `trainable_smooth` 时还须提供 `get_adapter_config_for_subgraph()`。
3. **使适配生效**：若修改了适配器代码，在仓库根目录重新执行 `bash install.sh`。
4. **作用范围与资源**：确定 `strategies` 的低比特目标与 `include`/`exclude`；确认设备显存与时长足以完成块级训练。若选用 `trainable_smooth`，还须确认子图类型能被适配器覆盖。

完整接入步骤见《[LLM 大模型接入指南](../../model/integrating_models.md)》；子图映射写法可参考《[多模态理解模型接入指南](../../model/integrating_multimodal_understanding_model.md)》附录中 IterSmooth 示例。

**输出**：确认所选 OP 路径对应的适配已满足且安装生效，并给出拟用的低比特策略作用范围。

### 步骤 2：写入推荐配置

**目标**：在量化 YAML 的 `spec.process` 中加入第一版 TLQ 配置，作为后续调参的对照起点。

**操作**：

若目标模型已有 `lab_practice` 配方且含 TLQ，优先复用该片段。否则使用下面的 **W4A4 推荐起步配置**（极低比特常用目标；权重/激活均为 INT4）：

```yaml
spec:
  process:
    - type: "trainable_linear_quant"
      operations:
        - type: "minmax_tune"
        - type: "round_tune"
      strategies:
        - qconfig:
            act:
              dtype: "int4"
              scope: "per_token"
              symmetric: true
              method: "minmax"
            weight:
              dtype: "int4"
              scope: "per_channel"
              symmetric: true
              method: "minmax"
          include: ["*"]
          exclude: []
      train_with_act_quant: false
      enable_quanted_input: false
      train_config:
        iters: 50
        gradient_accumulate_steps: 8
        lr: 0.01
        select_best:
          mode: "ema"
          ema_beta: 0.7
          ema_window_size: 5
          early_stop_patience: -1
        loss_type: "l1"
        train_seed: 42
```

说明：

- `strategies[].qconfig` 应与最终部署的 W4A4 格式一致；权重 `method` 用 `minmax`，由 TLQ OP 在其上继续调参。
- W4A4 场景建议 `train_with_act_quant: true`，使训练前向看到激活伪量化误差。
- 该配置量化耗时与显存明显高于 W8 一次标定；敏感层可再拆混合精度策略回退到更高位宽。

**输出**：已写入 YAML 的 TLQ `process` 项。

### 步骤 3：按需调整参数

**目标**：在起步配置基础上，只调整确有必要的字段。

**操作**：

实现入口：[查看对应实现目录](../../../../../msmodelslim/processor/trainable_linear_quant)

| 配置项 | 含义 | 推荐配置 | 何时调整 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定为 `trainable_linear_quant`。 | 固定。 | 不调。 |
| `operations` | 可训练 OP 管线；`minmax_tune`、`round_tune`、`trainable_smooth` 均可独立使用。`minmax_tune` + `round_tune` 联用大体等效于 AutoRound；`trainable_smooth` 侧重离群值抑制，不可与前两者联用。 | 起步用 `minmax_tune` + `round_tune`。 | 需要可学习舍入与范围调参时用默认两项或其中之一；需要可训练离群抑制时单独配置 `trainable_smooth`（须确认适配器子图能力）。 |
| `strategies` | 混合量化策略；含 `qconfig` 与 `include`/`exclude`。 | W4A4 单一策略起步。 | 精度不足时把敏感模块迁到更高精度策略；改位宽后必须重训。 |
| `strategies[].qconfig.weight` | 权重量化目标；dtype/scope/method 须有 TLQ kernel。 | W4A4：`int4` + `per_channel` + `symmetric` + `minmax`。 | 改 MXFP 或其他格式时确认 kernel 已注册，并重评 `iters`。 |
| `strategies[].qconfig.act` | 激活量化配置。 | W4A4：`int4` + `per_token` + `symmetric` + `minmax`。 | 仅权重量化时可改 `float + none`，并通常将 `train_with_act_quant` 设为 `false`。 |
| `train_with_act_quant` | 训练前向是否伪量化激活。 | W4A4 推荐 `true`。 | 权重-only 或排查激活误差贡献时可设 `false`。 |
| `enable_quanted_input` | 是否把本层量化输出作为下一层旁路输入。 | `false`。 | 仅当需要模拟跨层量化误差传播时开启；开启会进一步增加训练开销。 |
| `train_config.iters` | 块级训练迭代次数；为 0 时跳过优化。 | `50`。 | 先确保策略与 OP 正确，再增减；加大迭代会明显拉长量化时间。 |
| `train_config.lr` | 全局学习率；各 OP 的 `lr` 可覆盖。 | `0.01`。 | 损失震荡略降；几乎不更新略升。一次只改 `lr` 或只改 `iters`。 |
| `train_config.gradient_accumulate_steps` | 梯度累加步数。 | `8`。 | 显存紧张时增大；属资源旋钮，会改变等效更新频率。 |
| `train_config.select_best` | 最优快照：`ema` / `min_loss` / `last`。 | `ema`。 | 需按当轮最小损失选点时改 `min_loss`。 |
| `train_config.loss_type` | `l1` 或 `custom_outlier`。 | `l1`。 | 仅当离群主导且 L1 不足时再试 `custom_outlier`。 |
| `train_config.train_seed` | 训练随机种子。 | `42`。 | 稳定性对比时可换种子。 |

调整顺序建议：先锁定 `strategies` 的位宽与模块范围，再确认 `operations`，最后才动 `iters` / `lr`。每次只改一项。

**输出**：调整后的 TLQ 配置。

### 步骤 4：验证量化效果

**目标**：用含 TLQ 的完整配置完成量化，对比端到端精度，确认是否保留当前配置或继续微调。

**操作**：

1. 保持前置离群抑制等步骤不变，只评估 TLQ 配置变化带来的收益。
2. 对比 **无 TLQ（一次标定）** 与 **推荐起步 W4A4 TLQ** 的端到端指标；若没有达到预期，可以继续按步骤 3 单变量调整。
3. 精度不足时优先收窄低比特策略范围或回退敏感层；显存/时长不足时减少需训练的层或增大 `gradient_accumulate_steps`。

**输出**：可合入任务配置的最终 TLQ 片段，以及相对起步配置的调整记录。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Trainable Linear Quant 可训练线性量化算法 | 算法定义、原理、性质与限制。 | 《[Trainable Linear Quant 可训练线性量化算法 量化术语百科词条](./term_trainable_linear_quant.md)》 |
| LLM 大模型接入指南 | 新模型适配器开发与注册。 | 《[LLM 大模型接入指南](../../model/integrating_models.md)》 |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| trainable_linear_quant 配置说明 | 字段类型、默认值与约束。 | 《[trainable_linear_quant 配置说明](../../../api_reference/config/processor/trainable_linear_quant.md)》 |
| modelslim_v1 配置说明 | 任务级 runner / prior / save / dataset 等。 | 《[modelslim_v1 配置说明](../../../api_reference/config/task/modelslim_v1.md)》 |
