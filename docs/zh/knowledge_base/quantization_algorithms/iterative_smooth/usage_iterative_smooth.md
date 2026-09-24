# Iterative Smooth 迭代平滑算法使用指南

## 1. 适用范围

本指南面向需要使用 [Iterative Smooth 迭代平滑算法](./term_iterative_smooth.md) 的用户。Iterative Smooth 作为离群值抑制算法，通常作为量化前的预处理步骤，用于将激活离群值迁移到权重，提升低比特量化的精度。

适用场景：

- 激活分布存在显著离群通道，直接进行 W8A8 量化时精度下降；
- 模型具有 `norm-linear`、`linear-linear`、`ov`、`up-down` 等平滑结构，可通过迭代更新平滑尺度将激活动态范围迁移到权重。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及 `IterSmoothInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为前向张量 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors` 分片 | 导出完整且推理冒烟测试通过 |

## 3. 流程总览

Iterative Smooth 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线/算法接口] --> B[确定子图范围]
    B --> C[统计<br/>激活/权重分布]
    C --> D[迭代更新平滑尺度]
    D --> E[融合尺度]
    E --> F[衔接低比特量化]
```

各阶段的关键细节如下：

- **适配模型流水线/算法接口**：适配器需实现 `PipelineInterface`（基础流水线接口）及 `IterSmoothInterface`（迭代平滑专用接口），提供模型的平滑子图结构映射，是后续所有平滑步骤的前提。模型适配必须先完成，才能执行 Iterative Smooth 算法。
- **确定子图范围**：通过 `enable_subgraph_type` 选择 `norm-linear`、`linear-linear`、`ov`、`up-down` 等平滑结构，并通过 `include`/`exclude` 圈定作用模块范围。
- **统计激活/权重分布**：通过校准数据集（推荐 50 条）前向运行，统计各通道的激活与权重幅值分布。
- **迭代更新平滑尺度**：以超参数 `alpha`（默认 `0.9`）控制平滑迁移强度，迭代更新通道级平滑尺度，将激活离群压力等价迁移至权重，`scale_min` 用于避免极小尺度引起数值问题。
- **融合平滑尺度**：将平滑尺度的逆变换吸收融合到前驱的 Norm 结构中，保证数学等价。
- **衔接低比特量化**：平滑完成后张量分布更平缓，衔接 `linear_quant` 等低比特量化处理即可提升精度；建议保持后续量化配置不变，只调整当前算法参数以定位收益。

## 4. 操作步骤

### 步骤 1：适配模型流水线与算法接口

**目标**：量化工具不能直接操作任意结构的模型，需要适配器将模型操作转换为标准方法。对于 Iterative Smooth，模型适配器需要继承并实现流水线接口及迭代平滑接口。模型适配必须先完成，才能执行 Iterative Smooth 算法。

**操作**：模型适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总提供：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`IterSmoothInterface`**：迭代平滑专用接口，由 `msmodelslim.model.interface_hub` 汇总导出，提供模型的平滑子图拓扑配置，描述 Norm 与后续 Linear、OV、up/down 等结构之间的连接关系，供平滑处理器定位待平滑模块。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- Iter Smooth 算法适配接口（本步骤重点）----
    IterSmoothInterface,  # 算法适配：返回 norm-linear/linear-linear/ov/up-down 等平滑子图拓扑
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface, IterSmoothInterface):
    pass
```

各接口方法说明：

- **`PipelineInterface`** 的标准方法：加载数据（`handle_dataset`）、加载模型（`init_model`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）、控制 KV Cache（`enable_kv_cache`）。量化调度器（Runner）与处理器（Processor）只通过这些标准方法驱动模型，与模型内部结构无关。
- **`IterSmoothInterface`** 的核心方法：返回待平滑的子图拓扑配置（子图类型如 `norm-linear`、`linear-linear`、`ov`、`up-down`），平滑处理器据此定位并迭代更新平滑尺度。

之后量化命令中通过 `--model_type MyModelAdapter` 引用该适配器。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。请注意：模型适配必须先完成，才能执行后续量化流程。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方，优先使用已验证配方。

```yaml
spec:
  process:
    - type: "iter_smooth"
      alpha: 0.9
      scale_min: 1e-5
      symmetric: true
      enable_subgraph_type:
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"]
      exclude: []
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

ModelSlim 实现入口：
[查看对应实现目录](../../../../../msmodelslim/processor/anti_outlier/iter_smooth)

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定 `iter_smooth`。 | 固定。 | 不调。 |
| `alpha` | 平滑迁移强度，0~1，默认 `0.9`。越大通常把更多激活离群压力迁移到权重。仓库实践中最常见值是 0.9，少数模型使用 0.5。 | 无模型配方时从默认/主流实践 `0.9` 起步；已有实践直接沿用。 | 激活量化仍受极端值主导时可偏大；若权重侧误差明显增大或低比特权重更敏感，可向 0.5 等更保守方向降低。alpha 不是越大越好，必须和后续 W/A qconfig 一起评估。 |
| `scale_min` | 平滑 scale 的最小下限，默认 `1e-5`，用于避免极小尺度引起下溢或数值放大。 | 保持 `1e-5`。 | 它主要是数值稳定保护，不应作为常规精度旋钮。只有确认某些通道 scale 被下限大量截断且影响结果时才研究调整。 |
| `symmetric` | 决定后续量化假设以及平滑是否需要 shift。当前非对称路径主要支持 `norm-linear`，其他子图会关闭 shift。 | 常规 W8A8 实践优先 `true`。 | 只有明确采用非对称激活/权重量化且子图能力匹配时设 `false`。若想在多种子图上统一平滑，不建议为了少量偏移分布贸然切非对称。 |
| `enable_subgraph_type` | 决定哪些结构应用 IterSmooth。 | 先使用目标模型实践；没有配方时从默认四类开始，并通过匹配/评测确认。 | 如果非对称模式、部署融合或模型结构限制某类子图，应先收窄对应类型；不要通过修改 alpha 来补偿错误的子图匹配。 |
| `include` | 作用范围白名单，使用模块名模式决定哪些匹配到的模块进入当前算法。 | 无模型专用配方时从 `["*"]` 开始；已有 `lab_practice` 时直接沿用其模块范围。 | 先保证范围覆盖预期模块，再看精度。收窄范围时优先按结构族或已知敏感层调整。 |
| `exclude` | 作用范围黑名单，命中后从 `include` 的候选中排除，优先级高于 `include`。 | 默认先保持空列表；只有实践配方、兼容性约束或敏感性结果给出明确证据时再加入。 | 局部精度问题优先通过 `exclude` 做小范围回退，比提高全模型位宽或关闭整个算法更容易保留收益。 |

#### 参数组合与选择顺序

**最后只调一个主要旋钮**：IterSmooth 的主旋钮是 `alpha`；`scale_min` 主要保数值稳定，`symmetric` 与子图能力有关。先确定后续量化是否对称以及子图范围，再调 alpha，并保持同一校准集和评测集。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：Iterative Smooth + W8A8 动态量化（推荐起点）

##### 配置文件：`iter_smooth_w8a8.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  process:
    - type: iter_smooth             # Iterative Smooth 迭代平滑预处理
      alpha: 0.9
      scale_min: 1e-5
      symmetric: true
      enable_subgraph_type:
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"]
    - type: linear_quant            # 衔接低比特量化
      qconfig:
        act:
          dtype: int8
          scope: per_token          # 动态激活量化，精度表现好
          symmetric: true
          method: minmax
        weight:
          dtype: int8
          scope: per_channel
          symmetric: true
          method: minmax
      include: ["*"]
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
  --config ./iter_smooth_w8a8.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Iterative Smooth 迭代平滑算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《Iterative Smooth 迭代平滑算法 量化术语百科词条》](./term_iterative_smooth.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` / `IterSmoothInterface` | 模型流水线与 Iterative Smooth 算法适配接口，均由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| iter_smooth 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《iter_smooth 配置说明》](../../../api_reference/config/processor/iter_smooth.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/zh/user_guide/usage_weight_quantization.md) |
