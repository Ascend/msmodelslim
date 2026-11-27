# 组合处理器 (Group) 使用指南

## 1. 适用范围

本指南面向需要在一键量化流水线中使用组合处理器（`type: group`）的用户。组合处理器是 msModelSlim 中用于实现**精细化量化策略**的 DevKit 组件：将多个处理器（如线性量化、平滑处理等）封装在一个逻辑组内，针对模型的不同层应用差异化的量化配置，并在校准阶段**共享推理过程**。使用效果是：在同一轮校准前向下完成分层差异化量化，既提升策略灵活度，又避免多个独立处理器重复推理带来的校准开销。

适用场景：

- **混合量化策略**：如 Attention 层使用 W8A8 静态量化，MLP 层使用 W8A8 动态量化；
- **敏感层保护**：对精度极度敏感的层（如 `down_proj` 或 `gate`）使用更高比特或特定的平滑算法；
- **多算法组合**：在同一组层上先后应用平滑处理（Smooth）和线性量化（Linear Quant）。

组合处理器本身不实现量化公式；子处理器字段与其独立使用时完全一致。模型是否在官方验证列表中请参考《[大模型支持矩阵](../../model/README.md)》；CLI 基础用法可参阅《[一键量化使用指南](../../../user_guide/usage_one_click_quantization.md)》。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标模型与量化目标 | 待量化模型及部署/评测方案 | 明确混合量化或分层策略：哪些层族用何种处理器/位宽，以及精度与性能目标 | 能说明为什么使用 Group，以及它在整体量化方案中的位置 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface`；`configs` 内子算法若有专用接口需一并实现 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 配置约束与实践基线 | 《[group 配置说明](../../../api_reference/config/processor/group.md)》、目标模型已有 `lab_practice` 配方（如有）、各子处理器配置说明 | 字段名、`configs` 支持组合、作用范围与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 输入 | 校准数据（子处理器需要时） | 任务 `dataset`、校准集或模型实践配方 | 数据应能代表真实输入分布；挂入 Group 的多个需数据子处理器共享同一轮校准推理 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 交付件 | Group 参数配置方案 | 用户量化 YAML 或任务配置中的 `type: group` 段 | `configs` 取值合法、层族范围明确；关键子项说明选择依据 | 可作为后续量化流程的配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

组合处理器的整体使用流程如下：

```mermaid
flowchart LR
  A[确认混合策略<br/>与子处理器] --> B[建立推荐配置]
  B --> C[调整作用范围<br/>并避免重复量化]
  C --> D[编写完整任务<br/>配置并执行]
```

各阶段的关键细节如下：

- **确认混合策略与子处理器**：列出层族与目标方案（如 Attention 用 W8A8 静态、MLP 用 W8A8 动态），并确认每个方案对应的处理器 `type` 与独立使用时的字段要求；有 `lab_practice` 配方时优先沿用。
- **建立推荐配置**：把多个本可独立出现在 `process` 中的处理器挂到同一个 `type: group` 的 `configs` 下，建立可比较基线。挂入 Group 后，需校准数据的子处理器可共享同一轮前向。
- **调整作用范围并避免重复量化**：用各子项的 `include` / `exclude` 圈定层族；同一层被多个子项匹配时会先后处理，通常用 `exclude` 避免无意义的重复量化。
- **编写完整任务配置并执行**：补齐 `dataset`、`save` 等任务级字段后，通过一键量化 CLI 跑通端到端流程，并在异常时优先核对范围重叠与是否误写成多个独立处理器。

实现见 [msmodelslim/core/devkit/group.py](../../../../../msmodelslim/core/devkit/group.py)。

## 4. 操作步骤

### 步骤 1：确认混合策略与子处理器

**目标**：明确要对哪些层族应用何种算法/量化方案，避免无目的地堆叠多个独立处理器。

**操作**：

1. 列出层族与策略，例如：`*self_attn*` → W8A8 静态；`*mlp*` → W8A8 动态并排除 `*gate*`。
2. 确认每个策略对应的处理器 `type`（如 `linear_quant`、`smooth_quant`）及其独立使用时的字段要求。
3. 若目标模型已有 `lab_practice` 配方，优先沿用其中的 `type: group`，以及各子处理器的作用范围。

**输出**：一份层族到处理器 type 与关键参数的清单，可直接映射为 `configs` 列表项。

### 步骤 2：建立推荐配置

**操作**：

下面配置用于建立 **W8A8 静态 + 动态混合量化** 的可比较基线。

```yaml
apiversion: modelslim_v1
# 1. 定义静态量化模板 (Anchor)
default_w8a8_static: &w8a8_static
  act:
    scope: "per_tensor"        # 静态量化
    dtype: "int8"
    symmetric: false
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: true
    method: "minmax"

# 2. 定义动态量化模板 (Anchor)
default_w8a8_dynamic: &w8a8_dynamic
  act:
    scope: "per_token"         # 动态量化
    dtype: "int8"
    symmetric: true
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: true
    method: "minmax"

spec:
  process:
    - type: "group"            # 使用组合处理器
      configs:
        - type: "linear_quant" # 子处理器 1：针对 Attention 层
          qconfig: *w8a8_static
          include: ["*self_attn*"]

        - type: "linear_quant" # 子处理器 2：针对 MLP 层
          qconfig: *w8a8_dynamic
          include: ["*mlp*"]
          exclude: ["*gate*"]  # 排除门控层进行更精细控制
```

`group` 的字段定义（`type`、`configs`）与取值范围以配置文档为准，见《[group 配置说明](../../../api_reference/config/processor/group.md)》的参数列表与完整配置参考。各子处理器的字段与其独立使用时完全一致，按 `type` 查阅对应配置文档即可。

**输出**：一份可复现的 `type: group` 配置，后续范围与参数调整均以此为比较对象。

### 步骤 3：调整作用范围并避免重复量化

**操作**：

| 配置项 | 含义 | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定 `group`。 | 固定。 | 不调。 |
| `configs` | 子处理器列表，按顺序执行。 | 先放需要共享校准 Forward 的前向处理器。 | 列表顺序即执行顺序；不要把互不相关、无需共享推理的步骤硬塞进同一 Group。 |
| 子项 `include` / `exclude` | 决定哪些模块进入该子处理器。 | 按层族模式起步（如 `*self_attn*`、`*mlp*`）。 | 同一层被多个子项匹配时会先后处理；通常不建议对同一层重复量化，建议用 `exclude` 精确避让。 |

**输出**：作用范围无重叠冲突（或重叠是刻意的多阶段处理）的 Group 配置。

### 步骤 4：编写完整任务配置并执行

**目标**：把 Group 配置嵌入完整 `modelslim_v1` 任务（含 `dataset`、`save` 等），通过 CLI 跑通一次量化。

**操作**：

1. 按步骤 2 的示例补齐 `apiversion`、`spec.runner`（如需）、`dataset`、`save`。
2. 执行一键量化，命令行如下所示，其中，`${MODEL_PATH}` 为浮点权重目录，`${SAVE_PATH}` 为量化权重输出目录，`${MODEL_TYPE}` 为模型适配器名称（大小写敏感，与支持矩阵一致），`${CONFIG}` 为含 `type: group` 的 YAML 路径，例如 `./group_mixed_w8a8.yaml`。`--device` 取 `npu` 或 `cpu`；多卡时把 `--device_id 0` 写成 `--device_id 0 1 2 3`

```bash
msmodelslim quant \
  --model_path "${MODEL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --model_type "${MODEL_TYPE}" \
  --config "${CONFIG}" \
  --device npu \
  --device_id 0
```

**输出**：在 `${SAVE_PATH}` 下生成完整量化权重与描述文件。

**异常处置**：

- 若精度异常且日志显示同一模块被多次量化，检查 `configs` 中 `include`/`exclude` 是否重叠。
- 若校准明显变慢，确认本应共享 Forward 的处理器是否仍写成多个独立顶层 `process` 项，而非挂在同一 `group` 下。

## 5. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| group 配置说明 | 字段类型、默认值与完整配置约束。 | 《[group 配置说明](../../../api_reference/config/processor/group.md)》 |
| modelslim_v1 配置说明 | runner、process、save、dataset 等任务级配置。 | 《[modelslim_v1 配置说明](../../../api_reference/config/quant/modelslim_v1.md)》 |
| DevKit 总览 | DevKit 索引与领域边界。 | 《[DevKit](../README.md)》 |
| 量化算法总览 | 可作为 `configs` 子项的算法使用指南。 | 《[量化算法](../../quantization_algorithm/README.md)》 |
| 一键量化使用指南 | CLI 参数与端到端流程。 | 《[一键量化使用指南](../../../user_guide/usage_one_click_quantization.md)》 |
