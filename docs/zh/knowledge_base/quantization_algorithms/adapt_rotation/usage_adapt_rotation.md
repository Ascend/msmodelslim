# Adapt Rotation 自适应旋转量化算法使用指南

## 1. 适用范围

本指南面向需要使用 [Adapt Rotation 自适应旋转量化算法](./term_adapt_rotation.md) 的用户。Adapt Rotation 作为离群值抑制算法，在 [QuaRot](../quarot/term_quarot.md) 的基础上通过校准数据驱动优化正交旋转矩阵，用于进一步平滑激活离群值，提升低比特量化精度。

适用场景：

- 低比特（如 W4A4、W8A8 等）量化精度要求极高，基础正交旋转难以完全消除激活长尾离群；
- 需要基于校准样本学习更优的正交旋转变换，使激活与权重分布更加均匀。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface`、`AdaptRotationInterface` 及 QuaRot 相关接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为前向张量 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors` 分片 | 导出完整且推理冒烟测试通过 |

## 3. 流程总览

Adapt Rotation 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型旋转接口] --> B[Stage1<br/>数据驱动优化旋转]
    B --> C[Stage2<br/>应用旋转融合]
    C --> D[衔接低比特量化]
    D --> E[验证精度<br/>与部署信息]
```

各阶段的关键细节如下：

- **适配模型旋转接口**：适配器需实现 `AdaptRotationInterface`、`QuaRotInterface` 与 `OnlineQuaRotInterface`，提供两阶段旋转、网络映射与 Norm 融合支持，是后续算法执行的前置条件。
- **Stage1 数据驱动优化旋转**：在 `prior` 阶段基于校准数据采集指定投影层激活，通过梯度优化最小化模拟量化误差，得到优化旋转矩阵。
- **Stage2 应用旋转融合**：在 `process` 阶段将优化得到的旋转矩阵逐层作用到权重与激活中，支持离线融合或在线旋转。
- **衔接低比特量化**：旋转后激活分布更加平滑均匀，衔接低比特 `linear_quant` 即可大幅提升精度表现。
- **验证精度与部署信息**：评估量化精度并在导出时生成对应的模型描述与权重分片。

## 4. 操作步骤

### 步骤 1：适配模型流水线与旋转接口

**目标**：量化工具不能直接操作任意结构的模型，需要通过适配器将模型操作标准化。Adapt Rotation 包含 Stage1（数据优化）与 Stage2（旋转应用），必须在适配器中实现流水线与旋转接口后才能执行。

**操作**：模型适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总提供：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`AdaptRotationInterface`**（位于 `msmodelslim.processor.adapt_rotation`）：自适应旋转专用接口，定义 Stage1 与 Stage2 所需的模型子图与旋转映射。
3. **`QuaRotInterface`** / **`OnlineQuaRotInterface`**：旋转量化基础接口，提供 LayerNorm 融合映射（`get_ln_fuse_map`）与旋转对（`get_rotate_map`）。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- 旋转算法适配接口（本步骤重点）----
    QuaRotInterface,          # 算法适配：提供 LayerNorm 融合映射（get_ln_fuse_map）与旋转对（get_rotate_map）
    OnlineQuaRotInterface,    # 算法适配：提供在线旋转能力
    AdaptRotationInterface,   # 算法适配：定义 Stage1/Stage2 所需的模型子图与旋转映射
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface,
                     QuaRotInterface, OnlineQuaRotInterface, AdaptRotationInterface):
    pass
```

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。

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
          layer_type: ["up_proj"]
          steps: 20
          quant_dtype: "int4"
          block_size: -1
          max_samples: 2048
      dataset: "boolq.jsonl"

  process:
    - type: "adapt_rotation"
      stage: 2
      online: false
      block_size: -1
      max_tp_size: 1
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

ModelSlim 实现入口：
[查看对应实现目录](../../../../../msmodelslim/processor/adapt_rotation)

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器类型标识，固定为 `adapt_rotation`。 | 固定 `adapt_rotation`。 | 不建议调整。 |
| `stage` | 区分 Stage 1（优化学习）与 Stage 2（应用旋转）。 | 先 `1` 后 `2` 组合执行。 | 两阶段不能省略或交换顺序。 |
| `steps` | Stage 1 的优化迭代步数。 | 推荐先保持 `20`。 | 观察 loss 收敛情况，收敛后继续增加仅增加校准时间。 |
| `quant_dtype` | Stage 1 模拟激活量化误差的目标位宽。 | W4A4 用 `int4`，W8A8 用 `int8`。 | 必须与下游最终激活量化位宽对齐。 |
| `layer_type` | Stage 1 采集激活参与旋转优化的目标投影层。 | 推荐 `["up_proj"]`。 | 只有确认其他投影属于旋转优化路径时才扩展。 |
| `block_size` | 旋转块大小；`-1` 为整块旋转，正值须为 2 的幂。 | 优先 `-1`。 | 块越小跨维度扩散越弱，但更适配特定并行限制。 |
| `max_samples` | Stage 1 每层用于优化的最大样本数。 | 先用 `2048`。 | 样本量过少易过拟合，过多增加耗时。 |
| `online` | Stage 2 是否启用运行时在线旋转。 | 默认 `false`（离线融合）。 | 目标部署链明确支持并需要在线旋转时开启。 |
| `down_proj_online_layers` | Stage 2 指定哪些 Decoder 层的 down_proj 在线旋转。 | 默认 `[]`。 | 仅当特定层无法离线融合时加入。 |
| `max_tp_size` | Stage 2 在线旋转支持的最大 TP 规模。 | 离线保持默认；在线设为实际部署 TP。 | 与部署并行配置一致。 |

#### 参数组合与选择顺序

**最后只调一个主要旋钮**：优化步数 `steps` 保持 20，若指标波动再小幅调整。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：Adapt Rotation + W4A4 离线量化

##### 配置文件：`adapt_rotation_w4a4.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  prior:
    - process:
        - type: adapt_rotation
          stage: 1
          layer_type: ["up_proj"]
          steps: 20
          quant_dtype: int4
          block_size: -1
          max_samples: 2048
      dataset: mix_calib.jsonl
  process:
    - type: adapt_rotation
      stage: 2
      online: false
      block_size: -1
      max_tp_size: 1
    - type: linear_quant
      qconfig:
        act:
          dtype: int4
          scope: per_token
          symmetric: true
          method: minmax
        weight:
          dtype: int4
          scope: per_channel
          symmetric: true
          method: minmax
      include: ["*"]
  save:
    - type: ascendv1_saver
  dataset: mix_calib.jsonl
```

##### 执行命令（单卡量化）

```bash
msmodelslim quant \
  --model_path <浮点模型目录> \
  --save_path <量化权重输出目录> \
  --model_type <模型适配器名称> \
  --config ./adapt_rotation_w4a4.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Adapt Rotation 自适应旋转优化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《Adapt Rotation 自适应旋转优化算法 量化术语百科词条》](./term_adapt_rotation.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线适配接口（数据预处理、模型加载、模块遍历）。 | [《LLM 量化使用指南·步骤 1》](../../ptq/llm/usage_large_language_model_quantization.md) |
| `AdaptRotationInterface` / `QuaRotInterface` | 自适应旋转模型适配接口，由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| adapt_rotation 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《adapt_rotation 配置说明》](../../../api_reference/config/processor/adapt_rotation.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/quant/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
