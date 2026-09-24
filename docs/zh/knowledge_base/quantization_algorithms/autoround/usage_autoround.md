# AutoRound 低比特量化算法使用指南

## 1. 适用范围

本指南面向需要使用 [AutoRound 低比特量化算法](./term_autoround.md) 的用户。AutoRound 作为权重量化处理器，通过可学习舍入与 SignSGD 优化逐层求解最优量化参数，用于 4bit 等超低比特量化场景。

适用场景：

- W4 等低比特权重量化精度不达标，需要通过可学习舍入与 MinMax 联合优化提升精度；
- 需要按模块混合位宽部署（如大部分模块 W8A8、敏感投影层 W4A4），并通过统一优化流程求解。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标模型与量化目标 | 待量化模型及部署/评测方案 | 明确目标数值格式或位宽、作用模块、精度/性能目标以及部署约束 | 能说明为什么选择本算法以及它在整体量化方案中的位置 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 配置约束与实践基线 | 本算法配置说明、目标模型已有 `lab_practice` 配方（如有） | 字段名、支持组合、作用范围和模型适配与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 输入 | 校准数据（算法需要时） | 任务 `dataset`、校准集或模型实践配方 | 数据应能代表真实输入分布；多阶段算法尽量保持各阶段数据分布一致 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 交付件 | AutoRound 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

AutoRound 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线接口] --> B[确定混合位宽策略]
    B --> C[设置优化轮数<br/>与调优开关]
    C --> D[逐层优化<br/>量化参数与舍入]
    D --> E[应用量化结果]
    E --> F[验证精度<br/>与部署信息]
```

各阶段的关键细节如下：

- **适配模型流水线接口**：适配器需实现 `PipelineInterface`（基础流水线接口），量化调度器（Runner）与处理器（Processor）只通过标准流水线方法驱动模型。模型适配必须先完成，才能执行 AutoRound。
- **确定混合位宽策略**：通过 `strategies` 为不同模块划分不同 `qconfig`（位宽、粒度、对称性），策略边界决定最终哪些层承担低比特误差。
- **设置优化轮数与调优开关**：`iters` 控制优化迭代次数，`enable_minmax_tuning` 与 `enable_round_tuning` 分别控制量化范围与舍入方向的可学习优化。
- **逐层优化量化参数与舍入**：Processor 对目标 Linear 层逐层求解，通过 SignSGD 优化舍入方向与量化范围，缓解低比特下的重构误差。
- **应用量化结果**：优化得到的权重舍入与量化范围被写回模型，完成最终量化权重生成。
- **验证精度与部署信息**：对比量化前后的端到端精度，确认混合位宽策略满足部署约束；最终以实际部署链兼容性验证为准。

## 4. 操作步骤

### 步骤 1：适配模型流水线接口

**目标**：量化工具不能直接操作任意结构的模型，需要适配器将模型操作转换为标准方法。`autoround_quant` 处理器通过标准流水线方法驱动模型，模型适配器实现基础流水线接口即可。模型适配必须先完成，才能执行 AutoRound。

**操作**：模型适配代码需实现以下接口，相关接口由 `msmodelslim.model.interface_hub` 汇总提供：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。量化调度器（Runner）与处理器（Processor）只通过这些标准方法驱动模型，与模型内部结构无关。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface):
    pass
```

之后量化命令中通过 `--model_type MyModelAdapter` 引用该适配器。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。请注意：模型适配必须先完成，才能执行后续量化流程。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方，优先使用已验证配方。

```yaml
spec:
  process:
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
          exclude: ["*.up_proj", "*.gate_proj", "*.o_proj"]
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
          include: ["*.up_proj", "*.gate_proj", "*.o_proj"]
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

ModelSlim 实现入口：
[查看对应实现目录](../../../../../msmodelslim/processor/quant/autoround.py)

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定 `autoround_quant`。 | 固定。 | 不作为调参项。 |
| `iters` | AutoRound 优化迭代次数。代码默认 `10` 偏向快速起跑；仓库 Qwen3 W4A4 实践使用 `400`，说明低比特精度优化往往需要显著更多迭代。 | 正式精度基线推荐从 `400` 起步；只验证流程时可先用较小值。 | 增加迭代仍持续改善时才继续加；结果已稳定则不必追求更大数字。不要把“默认 10”误解为低比特精度最佳值。 |
| `enable_minmax_tuning` | 是否允许优化截断边界/量化范围。开启后算法不仅优化舍入，还能调整 MinMax 范围。 | 推荐 `true`。 | 通常保持开启。只有做消融或明确需要锁定量化范围时才关闭；关闭后若精度下降，优先恢复该开关。 |
| `enable_round_tuning` | 是否优化权重舍入方向，是 AutoRound 区别于普通 MinMax 权重量化的核心机制。 | 推荐 `true`。 | 常规使用不建议关闭；关闭后即使 `iters` 很大，也无法获得完整的 AutoRound 舍入优化收益。 |
| `strategies` | 混合量化策略列表。每条 strategy 有自己的 `qconfig` 和 `include/exclude`，用于让不同模块使用不同位宽/粒度。 | 优先复用模型实践。无专用配方时，先用较高精度覆盖大部分模块，再把明确适合的模块降到目标低比特。 | 策略调整顺序应是：先保护少数敏感模块，再扩大低比特覆盖。若低比特精度不足，先把高敏感层移到高精度策略，不要立即把全模型都升位宽。 |
| `strategies[].qconfig.act` | 每条策略的激活量化配置。激活位宽/粒度决定优化时面对的完整 W/A 误差环境。 | 与最终激活部署格式一致；常见 INT 激活从 `per_token + symmetric + minmax` 起步。 | 改变激活位宽会改变最优权重舍入，应视为整套目标变化并重新优化。 |
| `strategies[].qconfig.weight` | AutoRound 权重量化配置，`method` 应为 `autoround`，位宽/粒度决定权重码本与优化难度。 | 与最终权重格式一致；仓库实践使用 per-channel 对称权重。 | 位宽越低误差压力越大。若从 INT8 改 INT4，应重新跑优化并重新评估 `iters` 与策略范围。 |
| `strategies[].include/exclude` | 给每条混合精度策略划分模块范围。 | 直接沿用已验证模型配方；自定义时让各策略范围清晰、尽量避免模糊重叠。 | 精度不足优先做局部策略迁移：把最敏感的投影层从低比特策略移到高比特策略，再看收益。 |

#### 参数组合与选择顺序

**最后只调一个主要旋钮**：`iters` 不是第一个该调的参数。先确保策略中的最终 W/A 位宽、粒度和模块范围正确，再用 `iters` 控制优化充分程度；否则更多迭代只是在优化错误的目标。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：AutoRound 混合位宽量化

##### 配置文件：`autoround_mixed.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto                    # 单卡自动使用 layer_wise，多卡自动使用 dp_layer_wise
  process:
    - type: autoround_quant       # AutoRound 低比特量化
      iters: 400
      enable_minmax_tuning: true
      enable_round_tuning: true
      strategies:
        - qconfig:
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
          exclude: ["*.up_proj", "*.gate_proj", "*.o_proj"]
        - qconfig:
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
          include: ["*.up_proj", "*.gate_proj", "*.o_proj"]
  save:
    - type: ascendv1_saver        # 昇腾推理标准保存格式
  dataset: mix_calib.jsonl        # 内置混合校准集
```

##### 执行命令（单卡量化）

```bash
msmodelslim quant \
  --model_path <浮点模型目录> \
  --save_path <量化权重输出目录> \
  --model_type <模型类型> \
  --config ./autoround_mixed.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| AutoRound 低比特量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《AutoRound 低比特量化算法 量化术语百科词条》](./term_autoround.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线接口，由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| autoround_quant 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《autoround_quant 配置说明》](../../../api_reference/config/processor/autoround_quant.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/quant/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
