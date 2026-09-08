# OASQ 参数配置流程指南

## 1. 适用范围

OASQ（Outlier-Aware Smooth Quantization）离群感知平滑量化算法。OASQ 作为离群值抑制算法，通常写在量化 YAML 的 `spec.process` 中，作为后续线性量化等步骤之前的预处理。

本指南面向需要**把 OASQ 配进量化任务**的用户：说明推荐起步配置、关键参数何时调整，以及模型适配要注意什么。

- **适用**：已确定后续量化目标（如 W8A8 / W4A8），需要在配置中增加 OASQ 抑制激活通道离群。
- **不适用**：目标模型结构无法提供可融合子图且默认 hook 探测也不适用；或当前只需一次标定量化、无需离群抑制。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 量化任务配置 | 用户 YAML 或 `lab_practice` 配方 | 符合 `modelslim_v1`（或对应任务协议），可插入 `spec.process` | 能被量化流程加载 |
| 交付件 | OASQ 配置片段 | 写入上述 YAML 的 `process` 项 | `type: oasq`，字段合法；作用范围与适配器子图一致 | 可随完整配置复现量化 |

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

**目标**：确认目标 `model_type` 能正确识别 OASQ 子图，并明确要对哪些结构做平滑。

**操作**：

1. **模型适配**：建议适配器继承 `OASQInterface`，实现 `get_adapter_config_for_subgraph()`，返回覆盖目标结构的 `List[AdapterConfig]`。接口定义见 [interface.py](../../../../../msmodelslim/processor/anti_outlier/oasq/interface.py)；从 `msmodelslim.model.interface_hub` 导入。未实现时会回退默认 hook 自动探测，复杂结构可能匹配不准。
2. **使适配生效**：若修改了适配器代码，在仓库根目录重新执行 `bash install.sh`。
3. **作用范围**：确定需要启用的 `enable_subgraph_type`（如 `norm-linear`、`ov`），并确认适配器返回的子图能覆盖这些类型。

完整接入步骤见《[LLM 大模型接入指南](../../model/integrating_models.md)》；子图映射写法可参考《[多模态理解模型接入指南](../../model/integrating_multimodal_understanding_model.md)》附录中 IterSmooth 示例（方法签名相同，接口类改为 `OASQInterface`）。

**输出**：确认适配器已支持 OASQ 子图映射且安装生效，并给出拟启用的子图类型列表。

### 步骤 2：写入推荐配置

**目标**：在量化 YAML 的 `spec.process` 中加入第一版 OASQ 配置，作为后续调参的对照起点。

**操作**：

若目标模型已有 `lab_practice` 配方且含 OASQ，优先复用该片段。否则使用下面的推荐起步配置：

```yaml
spec:
  process:
    - type: "oasq"
      # 入门建议不显式设置 max_iters，使用实现默认值（当前为 8）。
      symmetric: true
      enable_subgraph_type:
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"]
      exclude: []
    # 其后接 linear_quant / trainable_linear_quant 等量化步骤
```

将 OASQ 放在真正量化处理器**之前**。`symmetric` 应与后续权重/激活量化的对称性假设一致。

**输出**：已写入 YAML 的 OASQ `process` 项。

### 步骤 3：按需调整参数

**目标**：在起步配置基础上，只调整确有必要的字段。

**操作**：

实现入口：[查看对应实现目录](../../../../../msmodelslim/processor/anti_outlier/oasq)

| 配置项 | 含义 | 推荐配置 | 何时调整 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定为 `oasq`。 | 固定。 | 不调。 |
| `max_iters` | 离群阈值搜索的最大迭代次数；不设置时用实现默认（当前 `8`），显式设置须 >0。 | 省略，用默认。 | 日志显示迭代耗尽且离群比例难落入目标区间时，再小幅增大。 |
| `symmetric` | 是否按对称假设收集统计；非对称时 `norm-linear` 可启用 shift，其他子图会关闭 shift。 | `true`。 | 仅当后续量化明确非对称且主要依赖 `norm-linear` 时设 `false`。 |
| `enable_subgraph_type` | 应用 OASQ 的子图类型列表。 | 默认四类；有实践配方则沿用。 | 某类结构无效或不兼容时单独去掉该类，不要用 `max_iters` 补偿。 |
| `include` / `exclude` | 模块名通配白名单 / 黑名单；`exclude` 优先。 | `include: ["*"]`，`exclude: []`。 | 局部敏感或不兼容层用 `exclude` 回退；改范围后需重新跑量化对比。 |

调整顺序建议：先固定 `symmetric` 与子图类型，再改 `include`/`exclude`，最后才动 `max_iters`。每次只改一项，并保持后续量化配置与校准集不变。

**输出**：调整后的 OASQ 配置。

### 步骤 4：验证量化效果

**目标**：用含 OASQ 的完整配置完成量化，对比端到端精度，确认是否保留当前配置或继续微调。

**操作**：

1. 保持后续量化步骤（如 `linear_quant`、`trainable_linear_quant`）不变，只评估 OASQ 的收益。
2. 对比 **无 OASQ** 与 **推荐起步 OASQ** 的端到端指标；若没有达到预期，可以继续按步骤 3 单变量调整。
3. 仅少数层异常时优先 `exclude`，避免直接关闭整个 OASQ 或全局升位宽。

**输出**：可合入任务配置的最终 OASQ 片段，以及相对起步配置的调整记录。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| OASQ 离群感知平滑量化算法 | 算法定义、原理、性质与限制。 | 《[OASQ 离群感知平滑量化算法 量化术语百科词条](./term_oasq.md)》 |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| oasq 配置说明 | 字段类型、默认值与约束。 | 《[oasq 配置说明](../../../api_reference/config/processor/oasq.md)》 |
| modelslim_v1 配置说明 | 任务级 runner / dataset / save / dataset 等。 | 《[modelslim_v1 配置说明](../../../api_reference/config/task/modelslim_v1.md)》 |
