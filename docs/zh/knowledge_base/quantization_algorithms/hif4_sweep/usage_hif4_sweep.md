# HiF4_Sweep 参数配置流程指南

## 1. 适用范围

HiF4_Sweep E6M2 偏移搜索量化算法。HiF4_Sweep 作为 `linear_quant` 处理器的权重量化方法，通过逐块 E6M2 offset 搜索最小化量化 MSE，提升 HiF4 W4A4 量化精度。激活量化使用标准 minmax 方法。

本指南面向首次配置 HiF4_Sweep 的用户，重点不是展开完整执行命令，而是说明推荐配置为什么适合作为起点、哪些参数真正需要调，以及出现精度或资源问题时应优先改哪一项。这类算法通常直接影响量化尺度、舍入方式、量化粒度或低比特表示，因此参数选择会同时影响精度、压缩率以及部署兼容性。第一次使用时建议先固定目标位宽、校准集和评测方式，只采用本指南给出的推荐起点；确认基线稳定后，再围绕真正影响算法行为的参数逐项调整。

如果目标模型已经有完整且已验证的量化配方，应优先复用该配方；若当前算法与目标模型结构、数值格式或部署后端不兼容，不应通过增大搜索强度或扩大作用范围来绕过支持约束。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标模型与量化目标 | 待量化模型及部署/评测方案 | 明确目标数值格式或位宽、作用模块、精度/性能目标以及部署约束 | 能说明为什么选择本算法以及它在整体量化方案中的位置 |
| 输入 | 配置约束与实践基线 | 本算法配置说明、目标模型已有 `lab_practice` 配方（如有） | 字段名、支持组合、作用范围和模型适配与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 输入 | 校准数据（算法需要时） | 任务 `dataset`、校准集或模型实践配方 | 数据应能代表真实输入分布；多阶段算法尽量保持各阶段数据分布一致 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 交付件 | HiF4_Sweep 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

入门时建议先用一组稳定配置建立基线，再围绕影响最大的参数做单变量调整。下面的流程强调"先选参数、再看效果"，不展开量化命令本身。

```mermaid
flowchart LR
    A[确定 HiF4 W4A4 per-block 方案] --> B[计算 base E6M2 编码]
    B[计算 base E6M2 编码] --> C[逐块搜索 E6M2 offset]
    C[逐块搜索 E6M2 offset] --> D[选择 MSE 最小候选]
    D[选择 MSE 最小候选] --> E[量化权重]
    E[量化权重] --> F[对比误差]
```

实际使用时建议把流程理解为"建立基线—观察结果—单变量调整—再次验证"的闭环。流程图中的前几个节点用于固定量化对象、统计信息或初始参数，后几个节点用于应用算法并检查结果；如果效果不理想，应优先回到最近一次修改的参数，而不是同时更换位宽、粒度、作用范围和算法强度。这样可以明确每次变化的因果关系，也便于把有效配置沉淀为后续模型配方。

## 4. 操作步骤

### 步骤 1：建立推荐基线

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方，优先使用已验证配方，再参考本节理解每个参数为什么这样选。

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"       # 量化范围
          dtype: "hif4"            # 量化数据类型
          symmetric: true          # 是否对称量化
          method: "minmax"         # 激活量化算法-标准 minmax
          ext:
            scale_ratio: 1
        weight:
          scope: "per_block"
          dtype: "hif4"
          symmetric: true
          method: "hif4_sweep"     # 权重量化算法-HiF4_Sweep
          ext:
            w_sweep_lower: -3      # Weight offset 搜索下界
            w_sweep_upper: 4       # Weight offset 搜索上界
```

上面的推荐值用于建立第一版可复现基线，其中最值得关注的配置包括 `scope`, `dtype`, `symmetric`, `method`。推荐值并不表示所有模型都只能使用该组合，而是优先选择仓库默认值、已验证实践或较稳健的中间取值，以降低第一次使用时同时遇到精度和兼容性问题的概率。如果目标模型已经有 `lab_practice` 配方，应优先复用该配方；只有在基线精度、显存或吞吐不满足目标时，再按照下一节的参数说明逐项调整。

**输出**：一份可复现的推荐基线配置，后续所有参数调整均以此为比较对象。

### 步骤 2：选择并调整参数

**操作**：

参数选择建议按三个层次理解：首先确认 `dtype/scope/method` 等**算法支持约束**，这类字段不是任意可调；其次确定 `include/exclude`、子图类型等**作用范围**；最后再调整会改变误差与开销的数值参数。下面的推荐值区分了代码默认值、仓库 `lab_practice` 中已验证的实践值和适合首次使用的推荐起点。如果目标模型已有实践配置，优先沿用实践配置，再根据本节说明做单变量调整。

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `scope` | 当前 HiF4_Sweep 量化器只注册为 HiF4 `per_block` 对称量化。分块搜索是算法定义的一部分。 | 固定 `per_block`。 | 需要其他粒度时应换量化器，而不是修改该字段。 |
| `dtype` | 当前只支持 `hif4`。HiF4 的 block 大小为 64，由 dtype 定义。 | 固定 `hif4`。 | 目标后端不是 HiF4 时不要使用 HiF4_Sweep。 |
| `symmetric` | 当前注册组合为对称量化。 | 固定 `true`。 | 不要改为非对称；当前没有对应注册实现。 |
| `method` | 选择 HiF4_Sweep：每个 HiF4 block 在 E6M2 编码空间搜索最优 offset，按重构 MSE 选最好方案。 | 权重固定 `hif4_sweep`，激活固定 `minmax`。 | 切换 method 就是切换算法。权重改为 `minmax` 可关闭搜索。 |
| `ext.axes` | 指定沿哪些张量维度做 HiF4 block reshape，默认 `-1`（最后一维）。轴选择会改变哪些元素共享一个 block，因此直接影响尺度统计和导出布局。 | 线性权重通常保持默认 `-1`；只有模型/后端布局明确要求其他轴时才修改。 | 不要把 axes 当精度微调参数。改轴之前确认目标权重维度、block 布局和后端一致；否则即使离线误差降低，也可能无法正确部署。 |
| `ext.w_sweep_lower` / `ext.w_sweep_upper` | Weight offset 搜索范围，默认 $[-3, 4]$（8 个候选）。理论已证明最优 offset 必在此区间内。取值范围限制为 $[-3, 4]$，超出范围会报错。 | 保持 $[-3, 4]$。 | 缩小范围可加速量化但可能遗漏最优；扩大范围理论上无收益。若仅需快速验证可设 $[0, 0]$ 等价 minmax。 |

### 参数组合与选择顺序

1. **先锁定部署目标与支持组合**：先确定最终位宽/数值格式，再确认当前量化器实际注册了对应的 `dtype + scope + symmetric + method` 组合；不要为了追求某个参数值而越过支持约束。
2. **再固定作用范围**：使用已有模型配方时，先复用其 `include/exclude` 或子图类型；没有配方时先建立覆盖范围明确的基线，确认算法确实作用到了预期模块。
3. **最后只调一个主要旋钮**：搜索范围等一次只改一项，并保持同一校准集和评测集。若调整后没有稳定收益，回到上一个基线，而不是继续叠加多个变化。

HiF4_Sweep 中真正可调的是 `w_sweep_*` 搜索范围；`dtype/scope/symmetric/method/axes` 更多属于格式与布局约束。精度不足时先确认 Weight sweep 已启用（$[-3, 4]$），再考虑缩小搜索范围加速量化。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 3：完整配置参考与命令行

**操作**：

以下给出 HiF4_Sweep 的完整配置参考和参考命令行，供用户直接使用或在此基础上调整 `w_sweep_lower` / `w_sweep_upper`。

#### 完整配置参考

```yaml
apiversion: modelslim_v1

spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"       # 激活量化范围，固定 per_block
          dtype: "hif4"            # 激活量化数据类型，固定 hif4
          symmetric: true          # 对称量化，固定 true
          method: "minmax"         # 激活量化算法，固定 minmax
          ext:
            scale_ratio: 1
        weight:
          scope: "per_block"       # 权重量化范围，固定 per_block
          dtype: "hif4"            # 权重量化数据类型，固定 hif4
          symmetric: true          # 对称量化，固定 true
          method: "hif4_sweep"     # 权重量化算法，固定 hif4_sweep
          ext:
            w_sweep_lower: -3      # Weight offset 搜索下界，取值范围 [-3, 4]
            w_sweep_upper: 4       # Weight offset 搜索上界，取值范围 [-3, 4]
      include:
        - "*"

  dataset: mix_calib.jsonl
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

#### w_sweep_lower / w_sweep_upper 选择与调整

`w_sweep_lower` 和 `w_sweep_upper` 是 HiF4_Sweep 真正可调的参数，控制 E6M2 offset 搜索范围：

- **默认值 [-3, 4]**：理论已证明最优 offset 必在此区间内，覆盖 8 个候选，推荐首次使用时保持默认。
- **快速验证 [0, 0]**：等价于标准 minmax（offset 恒为 0），用于对比 sweep 收益，搜索开销最小。
- **缩小范围加速**：如 [-1, 2]，减少候选数量加速量化，但可能遗漏最优 offset。根据日志中的 offset 分布（步骤 2 参数表说明）裁剪低频区间。
- **取值约束**：`w_sweep_lower` 和 `w_sweep_upper` 取值范围均为 $[-3, 4]$，超出范围会报 `SchemaValidateError`；`w_sweep_lower` 必须小于等于 `w_sweep_upper`，否则报 `SchemaValidateError`。

调整建议：先跑默认 [-3, 4] 基线，观察日志中 offset 分布；如果分布集中在某个子区间，可尝试缩小范围加速量化；若精度无损失则保留缩小后的范围，否则回退到默认。

#### 参考命令行

配置好 YAML 文件后，通过 modelslim 命令行执行量化：

```bash
modelslim --yaml_path ./w4a4_hif4_sweep.yaml --output_dir ./quant_output
```

更多命令行参数和使用说明请参考[《权重量化使用指南》](../../../user_guide/usage_weight_quantization.md)。

**输出**：一份可复现的完整配置方案和对应的量化命令行，可直接用于后续量化流程。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| HiF4_Sweep E6M2 偏移搜索量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | 《[HiF4_Sweep E6M2 偏移搜索量化算法 量化术语百科词条](./term_hif4_sweep.md)》 |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| linear_quant 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | 《[linear_quant 配置说明](../../../api_reference/config/processor/linear_quant.md)》 |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | 《[modelslim_v1 配置说明](../../../api_reference/config/quant/modelslim_v1.md)》 |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | 《[权重量化使用指南](../../../user_guide/usage_weight_quantization.md)》 |
