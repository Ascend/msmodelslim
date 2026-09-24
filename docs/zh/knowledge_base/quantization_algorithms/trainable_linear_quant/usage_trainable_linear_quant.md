# Trainable Linear Quant 可训练线性量化算法使用指南

## 1. 适用范围

本指南面向需要使用 [Trainable Linear Quant 可训练线性量化算法](./term_trainable_linear_quant.md) 的用户。Trainable Linear Quant（TLQ，`trainable_linear_quant`）作为线性层量化处理器，通常写在量化 YAML 的 `spec.process` 中，通过块级训练优化量化参数与舍入，主要用于 W4A4 等极低比特场景的精度增强。

适用场景：

- 目标为 W4A4 等低比特线性量化，一次标定精度不足，可接受更高量化耗时与显存占用以换取更好重构精度；
- 需要可学习的量化参数调优（`minmax_tune`）与舍入优化（`round_tune`），或可训练离群抑制（`trainable_smooth`）。

不适用场景：

- 仅需 W8 等较高位宽且一次标定已够用；
- 目标 dtype/scope/method 无对应 TLQ kernel；
- 设备显存/时长无法支撑块级训练。

> **资源提示**：TLQ 含块级训练，量化耗时与显存显著高于 MinMax / Linear Quant 等一次标定算法；正式跑数前请预留充足 NPU 显存与时间预算。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及 `TLQBlockDataInterface` 专用接口（按需） | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为前向张量 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors` 分片 | 导出完整且推理冒烟测试通过 |

## 3. 流程总览

TLQ 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线/算法接口] --> B[确认 OP 路径<br/>与作用范围]
    B --> C[建立推荐配置]
    C --> D[块级训练优化<br/>量化参数与舍入]
    D --> E[验证精度<br/>与资源占用]
    E --> F[收敛策略<br/>并导出量化权重]
```

各阶段的关键细节如下：

- **适配模型流水线/算法接口**：适配器需实现 `PipelineInterface` 基础流水线接口；使用默认块数据处理不足的复杂块结构时，需实现 `TLQBlockDataInterface` 补齐块 I/O 定制。
- **确认 OP 路径与作用范围**：按目标配置 `operations`（`minmax_tune`、`round_tune`、`trainable_smooth` 可独立使用），并确定 `strategies` 的低比特目标与 `include`/`exclude`。
- **建立推荐配置**：以 W4A4 推荐起步配置写入 `spec.process`，作为后续调参的对照起点。
- **块级训练优化量化参数与舍入**：逐 Decoder 块训练，优化可学习量化参数（`minmax_tune`）与舍入方向（`round_tune`），显存与耗时显著高于一次标定算法。
- **验证精度与资源占用**：对比无 TLQ（一次标定）与含 TLQ 的端到端精度，并确认设备显存与时长可支撑。
- **收敛策略并导出量化权重**：精度不足时收窄低比特策略范围或回退敏感层，最终导出量化权重目录。

## 4. 操作步骤

### 步骤 1：适配模型流水线与算法接口

**目标**：量化工具不能直接操作任意结构的模型，它要求每个模型先套一层“适配器”，把模型的各种操作翻译成框架能统一调用的标准方法。本步骤即确认并完成这层适配器与 TLQ 专用接口。**模型适配必须先完成，才能执行 TLQ 算法。**

**操作**：模型适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总导出：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`TLQBlockDataInterface`**（位于 `msmodelslim.processor.trainable_linear_quant.interface`）：TLQ 块 I/O 定制接口（可选）。

```python
from abc import ABC, abstractmethod

import torch

class TLQBlockDataInterface(ABC):
    @abstractmethod
    def extract_block_input_hidden_states(self, value) -> torch.Tensor:
        """从 block 输入相关数据中提取训练前向所需的输入 hidden states。"""
        ...

    @abstractmethod
    def extract_block_output_hidden_states(self, value) -> torch.Tensor:
        """从 block 输出相关数据中提取训练损失所需的目标 hidden states。"""
        ...
```

- **`TLQBlockDataInterface`**：用于 TLQ 块 I/O（hidden extract/inject/loss mask）；未实现时使用默认块数据处理，复杂或非常规块结构须由适配器补齐。
- 仅使用 `minmax_tune` 和/或 `round_tune` 时一般无需额外接口；使用 `trainable_smooth` 时还须提供 `get_adapter_config_for_subgraph()`（可复用 IterSmooth / OASQ 同名方法，接口约定见 `msmodelslim.processor.trainable_linear_quant.interface` 中的 `TLQSubgraphAdapter`）。
- 若修改了适配器代码，需在仓库根目录重新执行 `bash install.sh` 使适配生效。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承快速构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- TLQ 算法适配接口（本步骤重点）----
    TLQBlockDataInterface,  # 算法适配：提取/注入 block hidden states 并提供损失掩码（get_loss_mask）
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface,
                     TLQBlockDataInterface):
    def extract_block_input_hidden_states(self, value):
        # 从 block 输入中提取训练前向所需的 hidden states
        ...

    def extract_block_output_hidden_states(self, value):
        # 从 block 输出中提取训练损失所需的目标 hidden states
        ...
```

之后量化命令中通过 `--model_type MyModelAdapter` 引用该适配器。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方且含 TLQ，优先复用该片段。否则使用下面的 **W4A4 推荐起步配置**（极低比特常用目标；权重/激活均为 INT4）：

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
      train_with_act_quant: true
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

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

每轮只调一个变量，并保持同一校准集和模型版本。

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定 `trainable_linear_quant`。 | 固定。 | 不调。 |
| `operations` | 可训练 OP 管线；`minmax_tune`、`round_tune`、`trainable_smooth` 均可独立使用。`minmax_tune` + `round_tune` 联用大体等效于 AutoRound；`trainable_smooth` 侧重离群值抑制，不可与前两者联用。 | 起步用 `minmax_tune` + `round_tune`。 | 需要可学习离群抑制时单独配置 `trainable_smooth`（须确认适配器子图能力）。 |
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

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：TLQ + W4A4 极低比特量化

##### 配置文件：`tlq_w4a4.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto                    # 单卡自动使用 layer_wise，多卡自动使用 dp_layer_wise
  process:
    - type: trainable_linear_quant  # TLQ 块级训练量化
      operations:
        - type: "minmax_tune"       # 可学习量化参数调优
        - type: "round_tune"        # 可学习舍入优化
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
      train_with_act_quant: true    # 训练前向看到激活伪量化误差
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
  save:
    - type: ascendv1_saver          # 昇腾推理标准保存格式
  dataset: mix_calib.jsonl          # 内置混合校准集
```

##### 执行命令（单卡量化）

```bash
msmodelslim quant \
  --model_path <浮点模型目录> \
  --save_path <量化权重输出目录> \
  --model_type <模型适配器名称> \
  --config ./tlq_w4a4.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Trainable Linear Quant 可训练线性量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《Trainable Linear Quant 可训练线性量化算法 量化术语百科词条》](./term_trainable_linear_quant.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线适配接口（数据预处理、模型加载、模块遍历）。 | [《LLM 量化使用指南·步骤 1》](../../ptq/llm/usage_large_language_model_quantization.md) |
| `TLQBlockDataInterface` | TLQ 块 I/O 模型适配接口，由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| trainable_linear_quant 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《trainable_linear_quant 配置说明》](../../../api_reference/config/processor/trainable_linear_quant.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/quant/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/zh/user_guide/usage_weight_quantization.md) |
