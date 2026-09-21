# SVDQuant 参数配置流程指南

## 1. 适用范围

SVDQuant 低秩残差量化方案。SVDQuant 通过 `iter_smooth → svd_res → linear_quant` 三阶段流水线，对扩散模型等场景进行低比特量化。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标模型与量化目标 | 待量化模型及部署/评测方案 | 明确目标数值格式或位宽、作用模块、精度/性能目标以及部署约束 | 能说明为什么选择本算法以及它在整体量化方案中的位置 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及 `IterSmoothInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 配置约束与实践基线 | 本算法配置说明、目标模型已有 `lab_practice` 配方（如有） | 字段名、支持组合、作用范围和模型适配与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 输入 | 校准数据（算法需要时） | 任务 `dataset`、校准集或模型实践配方 | 数据应能代表真实输入分布；多阶段算法尽量保持各阶段数据分布一致 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 交付件 | SVDQuant 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

```mermaid
flowchart LR
    A[适配模型<br/>流水线/平滑接口] --> B[IterSmooth<br/>迁移离群值]
    B --> C[SVD<br/>提取低秩分量]
    C --> D[量化残差]
    D --> E[保留低秩补偿]
    E --> F[验证低比特精度]
```

## 4. 操作步骤

### 步骤 1：适配模型流水线与平滑接口

**目标**：SVDQuant 第一阶段由 `iter_smooth` 处理器执行离群值迁移，模型适配器除基础流水线接口外，还需实现迭代平滑专用接口，描述模型的平滑子图拓扑。模型适配必须先完成，才能执行 SVDQuant 三阶段流水线。

**操作**：模型适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总提供：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`IterSmoothInterface`**：迭代平滑专用接口，提供模型的平滑子图拓扑配置，描述 Norm 与后续 Linear、OV、up/down 等结构之间的连接关系，供平滑处理器定位待平滑模块。

适配器的组合继承构建方法与代码示例见《[Iterative Smooth 迭代平滑算法使用指南](../iterative_smooth/usage_iterative_smooth.md)》步骤 1。后续 `svd_res` 与 `linear_quant` 阶段由标准流水线方法驱动，无需额外专用接口。之后量化命令中通过 `--model_type` 引用该适配器。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。请注意：模型适配必须先完成，才能执行后续量化流程。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方，优先使用已验证配方，再参考本节理解每个参数为什么这样选。

```yaml
spec:
  process:
    # 阶段一：离群值迁移
    - type: "iter_smooth"
      alpha: 0.25
      include: ["*"]
      exclude: ["*blocks.0.*"]

    # 阶段二：低秩残差分解
    - type: "svd_res"
      rank: 32
      include: ["*"]
      exclude: ["*blocks.0.*"]

    # 阶段三：残差量化（W4A4 MXFP4）
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp4"
          symmetric: True
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp4"
          symmetric: True
          method: "minmax"
      include: ["*"]
      exclude: ["*blocks.0.*"]
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `iter_smooth.alpha` | 第一阶段先迁移离群值，降低后续低秩残差和低比特主干的数值压力。当前文档扩散模型示例使用 `0.25`，比通用 IterSmooth 默认 0.9 更保守，避免过度把激活困难迁到权重。 | 先沿用模型示例 `0.25`；其他模型优先使用其已验证配方。 | 残差仍被少数离群主导时再调平滑；如果低秩分支需要承担过多信息或权重侧误差上升，减弱平滑。不要同时改 alpha 和 rank，否则难判断收益来源。 |
| `svd_res.type` | 第二阶段处理器标识，固定 `svd_res`。 | 固定。 | 不作为调参。 |
| `svd_res.rank` | 低秩残差分支的秩，默认 `32`。rank 越大，低秩分支可保留更多难量化结构，但会增加额外参数/计算；rank 太小则更多信息落回低比特残差。 | 从 `32` 开始。 | 当残差量化已固定但任务精度不足，并且分析显示低秩分解仍能解释主要误差时增加；资源约束更重要且 32 有裕量时再降低。建议按 32→更高/更低的离散候选比较，而不是连续细调。 |
| `svd_res.include/exclude` | 决定哪些层生成低秩残差分支。它应与第一阶段平滑和第三阶段 residual quant 的层范围协调，否则某些层只经过流水线的一部分。 | 三阶段尽量使用一致的主范围和例外层；示例都排除首个 block。 | 若某层不适合低秩补偿，应在相关阶段一致地排除。范围不一致会让精度变化难以解释，也可能形成未预期的数据路径。 |
| `linear_quant` | 第三阶段对主干/残差按目标低比特格式量化。示例使用 MXFP4 per-block MinMax；它决定最终位宽和部署格式。 | 先固定目标部署 qconfig，再调 rank。 | 如果精度不足，优先增加 rank/调整层范围来判断是否是低秩容量问题；最后才提高残差位宽。直接先提位宽会掩盖 SVD 分解是否有效。 |

#### 参数组合与选择顺序

SVDQuant 推荐严格按 **范围一致性 → 固定最终 residual qconfig → 调 rank → 最后微调平滑 alpha** 的顺序。三阶段同时变化会让误差来源无法归因。每轮只改一个变量，并保持同一校准集和评测集；若调整后没有稳定收益，回到上一个基线。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：SVDQuant W4A4 MXFP4 量化（推荐起点）

##### 配置文件：`svdquant_w4a4_mxfp4.yaml`

```yaml
apiversion: multimodal_sd_modelslim_v1
spec:
  process:
    - type: iter_smooth             # 阶段一：离群值迁移
      alpha: 0.25
      include: ["*"]
      exclude: ["*blocks.0.*"]
    - type: svd_res                 # 阶段二：低秩残差分解
      rank: 32
      include: ["*"]
      exclude: ["*blocks.0.*"]
    - type: linear_quant            # 阶段三：残差量化（W4A4 MXFP4）
      qconfig:
        act:
          dtype: mxfp4
          scope: per_block
          symmetric: true
          method: minmax
        weight:
          dtype: mxfp4
          scope: per_block
          symmetric: true
          method: minmax
      include: ["*"]
      exclude: ["*blocks.0.*"]
  dataset: <扩散模型任务校准集>       # 与训练/推理分布一致的文本或图像 Prompt
  save:
    - type: mindie_format_saver     # MindIE-SD 扩散模型保存格式
      part_file_size: 0
```

##### 执行命令（单卡量化）

```bash
msmodelslim quant \
  --model_path <浮点模型目录> \
  --save_path <量化权重输出目录> \
  --model_type <模型适配器名称> \
  --config ./svdquant_w4a4_mxfp4.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| SVDQuant 低秩残差量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | 《[SVDQuant 低秩残差量化算法 量化术语百科词条](./term_svdquant.md)》 |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` / `IterSmoothInterface` | 模型流水线与 SVDQuant 算法适配接口，均由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| svd_res 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《svd_res 配置说明》](../../../api_reference/config/processor/svd_res.md) |
| iter_smooth 配置说明 | 三阶段流水线中的其他处理器参数。 | [《iter_smooth 配置说明》](../../../api_reference/config/processor/iter_smooth.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
