# LAOS 低比特量化算法使用指南

## 1. 适用范围

本指南面向需要使用 [LAOS 低比特量化方案](./term_laos.md) 的用户。LAOS 是 W4A4 低比特**联合**量化方案（面向大语言模型），通过组合 `adapt_rotation`（旋转优化）与 `autoround_quant`（低比特量化）处理器，用于 Qwen3 稠密系列等 LLM 的超低比特量化。

适用场景：

- W4A4 等超低比特量化精度不达标，需要通过数据驱动旋转优化打散激活离群通道；
- 需要在超低比特下通过混合精度策略（W8A8 保护范围 + W4A4 压缩范围）平衡压缩率与精度。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及 `LAOSOnlineRotationInterface`、`QuaRotInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/`（如 `laos_calib.jsonl`）或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为前向张量 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors` 分片 | 导出完整且推理冒烟测试通过 |

## 3. 流程总览

LAOS 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线/算法接口] --> B[Stage1<br/>数据驱动旋转优化]
    B --> C[Stage2<br/>应用旋转]
    C --> D[AutoRound<br/>低比特优化]
    D --> E[按层执行<br/>混合精度策略]
    E --> F[验证 W4A4<br/>精度与部署信息]
```

各阶段的关键细节如下：

- **适配模型流水线/算法接口**：适配器需实现 `PipelineInterface`（基础流水线接口）及 `LAOSOnlineRotationInterface`（LAOS 在线旋转）、`QuaRotInterface`（旋转相关）等接口，提供注意力头维度及各 Decoder 层 O/V、up/down 投影的旋转对等网络结构信息，是后续所有旋转步骤的前提。模型适配必须先完成，才能执行 LAOS 算法。
- **Stage1 数据驱动旋转优化**：`adapt_rotation` 第一阶段根据校准激活学习旋转（`steps: 20`、`quant_dtype: int4`、`layer_type: ["up_proj"]`），使后续低比特量化面对更均衡的通道分布。
- **Stage2 应用旋转**：`adapt_rotation` 第二阶段把 Stage 1 学到的旋转真正应用到模型，并决定哪些旋转离线融合（`online: false`）、哪些需要在线保留。
- **AutoRound 低比特优化**：`autoround_quant` 在旋转后的模型上对量化范围/舍入进行优化（`iters: 400`，开启 minmax 与 round tuning）。
- **按层执行混合精度策略**：通过多个 `strategies` 将 Attention、down_proj 等敏感模块保留在 W8A8 保护范围，对主要 MLP 压缩范围使用 W4A4。
- **验证 W4A4 精度与部署信息**：最终以端到端 W4A4 精度与部署链兼容性验证为准；上游目标（位宽、数据分布、旋转结构）变化后应从 Stage 1 开始整套重跑。

## 4. 操作步骤

### 步骤 1：适配模型流水线与算法接口

**目标**：量化工具不能直接操作任意结构的模型，需要适配器将模型操作转换为标准方法。对于 LAOS，模型适配器需要继承并实现流水线接口及 LAOS 旋转相关接口。模型适配必须先完成，才能执行 LAOS 算法。

**操作**：模型适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总提供：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`LAOSOnlineRotationInterface`**（位于 `msmodelslim.processor.quarot.offline_quarot.quarot_interface`）：LAOS 在线旋转接口，用于描述注意力头维度及各 Decoder 层 O/V、up/down 投影的旋转对。
3. **`QuaRotInterface`**（位于 `msmodelslim.processor.quarot.offline_quarot.quarot_interface`）：离线旋转核心接口，提供旋转映射、Norm 融合与均值融合等网络结构信息。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- 旋转算法适配接口（本步骤重点）----
    LAOSOnlineRotationInterface,  # 算法适配：提供 head_dim、注意力头数与逐层 OV 旋转对（get_layer_wise_ov_pair）
    QuaRotInterface,              # 算法适配：提供 LayerNorm 融合映射（get_ln_fuse_map）与旋转对（get_rotate_map）
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface,
                     LAOSOnlineRotationInterface, QuaRotInterface):
    pass
```

各接口方法说明：

- **`PipelineInterface`** 的标准方法：加载数据（`handle_dataset`）、加载模型（`init_model`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）、控制 KV Cache（`enable_kv_cache`）。量化调度器（Runner）与处理器（Processor）只通过这些标准方法驱动模型，与模型内部结构无关。
- **`LAOSOnlineRotationInterface`** 的核心方法：`get_head_dim`（返回注意力头维度）、`get_num_attention_heads`（返回注意力头数量）、`get_layer_wise_ov_pair`（返回各 Decoder 层 O/V 投影旋转对）、`get_layer_wise_up_down_pair`（返回 up/down 投影旋转对）。
- **`QuaRotInterface`** 的核心方法：`get_rotate_map`（返回旋转映射，包括 pre-run 阶段与 preprocess 阶段的旋转对列表）、`get_ln_fuse_map`（返回 LayerNorm 层与 Linear 层的融合映射）、`get_bake_names`（返回需要均值融合的 Linear 层名称列表）。

之后量化命令中通过 `--model_type MyModelAdapter` 引用该适配器。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。请注意：模型适配必须先完成，才能执行后续量化流程。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方，优先使用已验证配方。

```yaml
spec:
  prior:
    - process:
        - type: "adapt_rotation"
          stage: 1
          steps: 20
          quant_dtype: "int4"
          layer_type: ["up_proj"]
      dataset: "laos_calib.jsonl"

  process:
    - type: "adapt_rotation"
      stage: 2
      online: false
      block_size: -1
      max_tp_size: 1

    - type: "autoround_quant"
      iters: 400
      enable_minmax_tuning: true
      enable_round_tuning: true
      strategies:
        - qconfig:
            act:
              dtype: "int8"
              scope: "per_token"
              symmetric: true
              method: "minmax"
            weight:
              dtype: "int8"
              scope: "per_channel"
              symmetric: true
              method: "autoround"
          include: ["*self_attn*", "*.down_proj"]
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
              method: "autoround"
          include: ["*.up_proj", "*.gate_proj"]

  dataset: "laos_calib.jsonl"
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

LAOS 是“自适应旋转 + AutoRound 混合低比特优化”的串联方案，上游旋转改变了下游权重和激活分布，AutoRound 的位宽/策略又定义了旋转真正要服务的最终目标。核心原则是**上游目标变化会使下游最优点失效**：位宽、校准分布或旋转结构发生改变后，应重新生成旋转并重新执行 AutoRound。

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `adapt_rotation.stage: 1` | 第一阶段根据校准激活学习旋转，使后续低比特量化面对更均衡的通道分布。 | LAOS 必须先执行 Stage 1。 | 最终位宽、旋转范围、校准数据或块大小发生实质变化后，应重新学习旋转。 |
| `adapt_rotation.steps` | Stage 1 的旋转优化步数，收益会逐渐进入平台期。 | 仓库 Qwen3 W4A4 实践使用 `20`。 | 若旋转目标在 20 步附近仍持续明显下降可小幅增加；指标已稳定时继续加步数性价比不高。 |
| `adapt_rotation.quant_dtype` | Stage 1 用来模拟下游激活量化噪声的目标类型，支持 INT4/INT8。 | W4A4 目标用 `int4`；W8A8 目标用 `int8`。 | 位宽变化时必须重新执行 Stage 1。该参数优先级高于 `steps`。 |
| `adapt_rotation.layer_type` | 决定 Stage 1 从哪些投影层收集激活来估计旋转目标。 | Qwen3/LAOS 已验证起点为 `["up_proj"]`。 | 只有确认最终低比特范围扩展到其他旋转路径时才扩大；若模块名没有正确匹配，先修正范围。 |
| `adapt_rotation.stage: 2` | 第二阶段把 Stage 1 学到的旋转真正应用到模型。 | Stage 1 完成后执行。 | Stage 2 不是再次搜索最优旋转，Stage 1 目标变了应重新执行。 |
| `adapt_rotation.online` | 是否保留在线旋转，`false` 尽量离线融合，运行时开销较少。 | 仓库 LAOS/Qwen3 W4A4 实践使用 `false`。 | 入门优先离线；只有部署链明确支持在线旋转且存在必须在线处理的层时才开启。 |
| `adapt_rotation.block_size` | 旋转块大小，`-1` 表示按 hidden dimension 整块旋转。 | 仓库 LAOS 实践使用 `-1`。 | 精度优先保持整块旋转；只有目标算子/并行布局明确要求块化时再改。 |
| `adapt_rotation.max_tp_size` | 在线旋转相关的最大 TP 并行度，是部署兼容参数。 | 仓库 LAOS 实践使用 `1`；在线部署时按实际最大 TP 设置。 | 不要随意放大；部署 TP 变化时再调整。 |
| `autoround_quant.iters` | AutoRound 的优化迭代数。 | 正式复现从 `400` 起步；只验证流程可临时降低。 | 先确认 strategy 范围正确，再判断是否需要增加迭代。 |
| `enable_minmax_tuning` | 是否允许 AutoRound 调整量化截断范围。 | 推荐保持开启。 | 只有做算法消融时关闭。 |
| `enable_round_tuning` | 是否优化权重的舍入方向，是 AutoRound 的关键能力。 | `true`。 | 常规 LAOS 不建议关闭，尤其 W4 权重下。 |
| `strategies[].qconfig` | 每个混合精度策略的激活/权重量化目标。 | 敏感/保护范围用 W8A8，主要压缩范围用 W4A4；具体层范围优先照模型实践。 | W4 精度不足时先把少数敏感层迁移到 W8 strategy，而不是直接把全模型升到 W8。 |
| `strategies[].include/exclude` | 定义 W8A8 与 W4A4 分别覆盖哪些模块。 | 第一选择是复用仓库已验证的层范围。 | 精度不足时优先小范围扩大 W8 保护集合；每次只迁移一组结构相近的层。 |
| `dataset` | 两个数据驱动阶段都依赖校准分布。 | 使用与真实业务同分布的数据；复现配方时使用 `laos_calib.jsonl`。 | 数据集或任务分布明显改变时，建议从 Stage 1 开始整套重跑。 |

#### 参数组合与选择顺序

1. **先确定最终 W/A 目标**：决定哪些模块是 W4A4、哪些需要 W8A8 保护，并让 Adapt Rotation 的 `quant_dtype` 与主目标一致。
2. **再固定校准数据与 Stage 1**：用代表性数据完成旋转，`steps` 只在确认目标和数据正确后再调整。
3. **应用 Stage 2，并按部署约束决定 online/block/TP**：这些参数主要解决落地问题，不要拿它们替代精度调参。
4. **最后优化 AutoRound**：先用仓库实践的 `iters` 和两个 tuning 开关，再通过 strategy 范围做局部 W8 回退。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：LAOS W4A4 联合量化（推荐起点）

##### 配置文件：`laos_w4a4.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  prior:
    - process:
        - type: adapt_rotation       # Stage 1 数据驱动旋转优化
          stage: 1
          steps: 20                  # 旋转优化步数
          quant_dtype: int4          # 与最终激活位宽一致
          layer_type: ["up_proj"]    # 采集激活的投影层
      dataset: laos_calib.jsonl
  process:
    - type: adapt_rotation           # Stage 2 应用旋转
      stage: 2
      online: false                  # 优先离线旋转融合
      block_size: -1                 # 整 hidden_dim 旋转
      max_tp_size: 1
    - type: autoround_quant          # AutoRound 低比特优化
      iters: 400
      enable_minmax_tuning: true
      enable_round_tuning: true
      strategies:
        - qconfig:                   # W8A8 保护范围
            act:
              dtype: int8
              scope: per_token
              symmetric: true
              method: minmax
            weight:
              dtype: int8
              scope: per_channel
              symmetric: true
              method: autoround
          include: ["*self_attn*", "*.down_proj"]
        - qconfig:                   # W4A4 压缩范围
            act:
              dtype: int4
              scope: per_token
              symmetric: true
              method: minmax
            weight:
              dtype: int4
              scope: per_channel
              symmetric: true
              method: autoround
          include: ["*.up_proj", "*.gate_proj"]
  save:
    - type: ascendv1_saver           # 昇腾推理标准保存格式
  dataset: laos_calib.jsonl          # LAOS 校准集
```

##### 执行命令（单卡量化）

```bash
msmodelslim quant \
  --model_path <浮点模型目录> \
  --save_path <量化权重输出目录> \
  --model_type <模型适配器名称> \
  --config ./laos_w4a4.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| LAOS 低比特量化方案 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《LAOS 低比特量化方案 量化术语百科词条》](./term_laos.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` / `LAOSOnlineRotationInterface` / `QuaRotInterface` | 模型流水线与 LAOS 算法适配接口，均由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| adapt_rotation 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《adapt_rotation 配置说明》](../../../api_reference/config/processor/adapt_rotation.md) |
| autoround_quant 配置说明 | LAOS 第二阶段的 AutoRound 参数。 | [《autoround_quant 配置说明》](../../../api_reference/config/processor/autoround_quant.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/quant/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/zh/user_guide/usage_weight_quantization.md) |
