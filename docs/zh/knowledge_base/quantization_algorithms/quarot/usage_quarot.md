# QuaRot 参数配置流程指南

## 1. 适用范围

QuaRot 旋转量化算法。QuaRot 作为离群值抑制算法，通常作为量化前的预处理步骤，通过正交旋转平滑激活分布，提升低比特量化的精度。

本指南面向首次配置 QuaRot 的用户，重点不是展开完整执行命令，而是说明推荐配置为什么适合作为起点、哪些参数真正需要调，以及出现精度或资源问题时应优先改哪一项。这类算法的主要作用是先改善待量化张量的数值分布，再交给后续量化步骤处理，本身通常不是最终的量化格式。因此判断配置是否合适时，不仅要看平滑或旋转后的张量范围，还要看与后续量化组合后的端到端精度；建议保持后续量化配置不变，只调整当前算法参数。

如果目标模型已经有完整且已验证的量化配方，应优先复用该配方；若当前算法与目标模型结构、数值格式或部署后端不兼容，不应通过增大搜索强度或扩大作用范围来绕过支持约束。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标模型与量化目标 | 待量化模型及部署/评测方案 | 明确目标数值格式或位宽、作用模块、精度/性能目标以及部署约束 | 能说明为什么选择本算法以及它在整体量化方案中的位置 |
| 输入 | 配置约束与实践基线 | 本算法配置说明、目标模型已有 `lab_practice` 配方（如有） | 字段名、支持组合、作用范围和模型适配与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 输入 | 校准数据（算法需要时） | 任务 `dataset`、校准集或模型实践配方 | 数据应能代表真实输入分布；多阶段算法尽量保持各阶段数据分布一致 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 交付件 | QuaRot 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

入门时建议先用一组稳定配置建立基线，再围绕影响最大的参数做单变量调整。下面的流程强调“先选参数、再看效果”，不展开量化命令本身。

```mermaid
flowchart LR
    A[确定离线/在线旋转方式] --> B[选择旋转块大小]
    B[选择旋转块大小] --> C[应用正交旋转]
    C[应用正交旋转] --> D[衔接低比特量化]
    D[衔接低比特量化] --> E[验证精度与部署信息]
```

实际使用时建议把流程理解为“建立基线—观察结果—单变量调整—再次验证”的闭环。流程图中的前几个节点用于固定量化对象、统计信息或初始参数，后几个节点用于应用算法并检查结果；如果效果不理想，应优先回到最近一次修改的参数，而不是同时更换位宽、粒度、作用范围和算法强度。这样可以明确每次变化的因果关系，也便于把有效配置沉淀为后续模型配方。

## 4. 操作步骤

### 步骤 1：确认目标与约束

**操作**：先固定目标模型、最终量化格式/位宽、作用对象和评测基线，再确认当前版本支持的参数组合。目标模型已有 `lab_practice` 配方时，优先把该配方作为实践基线；没有模型专用配方时，再使用本指南给出的通用推荐起点。若算法依赖校准统计或优化数据，还应在调参前固定代表性数据，避免把数据分布变化误判为参数收益。

**输出**：一份明确的配置目标：目标位宽/格式、处理范围、校准条件、评测基线和部署约束。

### 步骤 2：建立推荐基线

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方，优先使用已验证配方，再参考本节理解每个参数为什么这样选。

```yaml
spec:
  process:
    - type: "quarot"                     # 固定为 `quarot`，用于指定 Processor 类型。
      online: False                      # 控制是否启用在线旋转，默认为 False。
      block_size: -1                     # 旋转矩阵启用块对角矩阵时每个块的大小, 取值范围为-1或2的幂次方，如果大于0必须为2的幂，若为-1，表示不进行块对角矩阵处理
      max_tp_size: 4                     # 最大张量并行大小，默认为4，仅在启用在线旋转时生效，必须大于0且为2的幂
      down_proj_online_layers: [ ]       # 用于指定哪些层的down_proj使用在线旋转，默认为空
      export_extra_info: True            # 是否导出全局旋转矩阵：使用 ascend_v1_saver 时在量化权重路径下生成 optional 目录及 quant_model_description.json 追加字段，默认为 True
```

上面的推荐值用于建立第一版可复现基线，其中最值得关注的配置包括 `type`, `online`, `block_size`, `max_tp_size`。推荐值并不表示所有模型都只能使用该组合，而是优先选择仓库默认值、已验证实践或较稳健的中间取值，以降低第一次使用时同时遇到精度和兼容性问题的概率。如果目标模型已经有 `lab_practice` 配方，应优先复用该配方；只有在基线精度、显存或吞吐不满足目标时，再按照下一节的参数说明逐项调整。

**输出**：一份可复现的推荐基线配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

ModelSlim 实现入口：
[查看对应实现目录](../../../../../msmodelslim/processor/quarot)

参数选择建议按三个层次理解：首先确认 `dtype/scope/method` 等**算法支持约束**，这类字段不是任意可调；其次确定 `include/exclude`、子图类型等**作用范围**；最后再调整会改变误差与开销的数值参数。下面的推荐值区分了代码默认值、仓库 `lab_practice` 中已验证的实践值和适合首次使用的推荐起点。如果目标模型已有实践配置，优先沿用实践配置，再根据本节说明做单变量调整。

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定 `quarot`。 | 固定。 | 不调。 |
| `online` | 是否保留在线旋转。`false` 尽可能离线融合；`true` 引入运行时旋转并影响并行/算子要求。 | 默认/优先 `false`。 | 只有目标部署链明确支持并需要在线旋转时开启。开启后不要只看精度，还要评估推理时延和算子兼容。 |
| `block_size` | 旋转块大小，`-1` 表示整 hidden_dim，正值必须是 2 的幂。仓库实践中不少模型显式使用 `32`，说明某些架构/部署会选择块旋转。 | 没有模型配方时先 `-1`；已有实践用 `32` 等值时直接沿用。 | 块越小旋转越局部，可能降低全维混合离群的能力，但更适配某些并行/硬件约束。不要因为仓库很多模型用 32 就把它当通用精度最优值。 |
| `down_proj_online_layers` | 指定哪些 Decoder 层的 down_proj 使用在线旋转，默认空。 | 默认 `[]`。 | 仅在这些层无法离线融合或专用配方要求时加入。层越多，运行时额外旋转越多。 |
| `max_tp_size` | 在线旋转可支持的最大 TP，必须为 1 或 2 的幂，默认 4。 | 在线模式设置为实际部署最大 TP；离线保持默认。 | 实际部署 TP 改变时才调整。过大不会免费带来精度收益，反而可能改变分块/兼容约束。 |
| `export_extra_info` | 是否导出全局旋转元信息，默认 true，供特定 saver/部署链消费。它不改变离线旋转数学结果，但影响下游能否重建必要信息。 | 若目标 saver/后端需要 QuaRot 元信息则保持 `true`；仓库多个实践显式开启。 | 只有确认下游完全不读取这些字段时才关闭。不要为了减少文件而盲目关掉。 |

### 参数组合与选择顺序

1. **先锁定部署目标与支持组合**：先确定最终位宽/数值格式，再确认当前量化器实际注册了对应的 `dtype + scope + symmetric + method` 组合；不要为了追求某个参数值而越过支持约束。
2. **再固定作用范围**：使用已有模型配方时，先复用其 `include/exclude` 或子图类型；没有配方时先建立覆盖范围明确的基线，确认算法确实作用到了预期模块。
3. **最后只调一个主要旋钮**：数值搜索步数、平滑强度、分组大小等一次只改一项，并保持同一校准集和评测集。若调整后没有稳定收益，回到上一个基线，而不是继续叠加多个变化。

QuaRot 的 `block_size/online/max_tp_size` 主要由部署形态决定。先按模型实践固定这些结构参数，再评估旋转带来的量化收益；不要把部署约束参数当作普通精度超参数。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：根据结果收敛参数方案

**操作**：

调参时建议先记录一份完整基线，包括使用的数据集、量化范围、关键参数和端到端指标。每轮只改变一个变量，并把变化结果与基线直接比较；如果某项调整带来收益，再继续小步搜索其邻近取值。对于只有少数层或模块异常的情况，优先采用局部排除、局部回退或混合精度，而不是直接提高全模型精度配置，这通常更容易保留压缩和性能收益。

- **先跑推荐基线，再调单变量。** 不要同时修改位宽、粒度、算法参数和层范围，否则很难判断精度变化来自哪一项。
- **优先回退局部，而不是整体提高精度。** 如果只有少数层敏感，优先通过 `exclude` 或混合策略保留高精度，通常比整体升位宽更划算。
- **最终以模型实践配置和部署能力为准。** 入门推荐用于建立稳定起点；目标模型已有 `lab_practice` 配方时，应优先复用已验证组合。

**输出**：一份可进入后续量化流程的最终参数方案，并保留相对于推荐基线的调整记录。

## 5. 离线 QuaRot 无损自检

QuaRot通过通用`lossless_check`容器完成处理前后自动自检。容器只包装一个子Processor，
在同一Adapter请求内执行参考前向、一次正式QuaRot处理、处理后前向、Linear语义对齐和指标比较。
比较完成后立即释放激活、Hook和请求级Context，不累积整网hidden states。

自检不是data-free流程。输入必须经过数据集加载、Adapter的`handle_dataset`和
`generate_model_forward`。文本、多模态字段、MTP预处理和模型入参均由对应Adapter构造，
容器不会猜测或生成token、图像、音频或hidden states。

若子Processor未完整实现无损自检协议，或Adapter未实现`generate_model_forward`，
容器会输出WARNING日志并跳过比较，但仍按原流程执行子Processor，不阻断正常处理。
这类结果的`passed`为`null`，不能作为自检通过证据。

### 使用前确认

开始配置前先确认三件事：

1. 模型已有可用的QuaRot适配，能够返回离线旋转、Norm融合和bake mean计划。
2. 模型Adapter实现了`generate_model_forward`，能够使用真实样本生成`BatchProcessRequest`。
3. 准备了能触发待检查Linear的代表性数据。MoE、多模态或条件路由模型需要覆盖相应路由或模态。

容器没有模型名称白名单，但这不等于所有模型无需适配即可运行。缺少前两项时，容器只会WARNING
并继续原Processor，结果为`passed: null`，不能作为自检通过证据。

### 最小可用配置

离线QuaRot使用`quarot`子Processor，且必须配置`online: False`。不要在同一流程中再配置
第二个`quarot`，否则模型权重会被重复处理。

```yaml
spec:
  dataset: test.json
  process:
    - type: "lossless_check"
      include:
        - "model.layers.0.self_attn.{q,k,v,o}_proj"
      exclude: []
      metrics:
        - "numerical_equivalence"
      configs:
        - type: "quarot"
          online: False
          block_size: -1
          export_extra_info: True
```

`dataset`沿用快速量化的标准数据集配置。建议先使用1至2条能触发目标结构的代表性样本；
条件专家模型和多模态模型应补充覆盖必验路由或模态的样本。样本未触发的理论结构会记录为
未观察目标，不会被错误计为通过。

### 容器配置

| 字段名 | 类型 | 说明 | 默认值 |
| --- | --- | --- | --- |
| type | `string` | 外层固定为`lossless_check` | `"lossless_check"` |
| configs | `array[object]` | 恰好一个子Processor；禁止嵌套容器 | 无 |
| include | `array[string]` | 要检查的完整结构名；支持`*`、`?`和brace语法 | `["*"]` |
| exclude | `array[string]` | 从include结果中排除的结构，优先级高于include | `[]` |
| metrics | `array[string\|object]` | 必填且非空；按顺序执行的已安装指标 | 无 |

不需要额外参数的指标可以直接填写名称。内置推荐指标使用以下简写即可：

```yaml
metrics:
  - "numerical_equivalence"
```

需要参数的第三方指标使用对象写法：

| 字段名 | 类型 | 说明 | 默认值 |
| --- | --- | --- | --- |
| name | `string` | 已安装的指标名 | 无 |
| 其他字段 | 由指标插件定义 | 例如阈值、比较方向或专用选项 | 由指标插件定义 |

内置`numerical_equivalence`会根据Tensor dtype与数值尺度推导容差，同时记录逐元素接近性
和分布误差。它是开箱即用的推荐项，不是强制项；不接受手工阈值覆盖，也不会被自动加入配置。
用户可以只选择已安装的第三方指标，也可以把多个指标组合使用。

所有配置指标都是判定项，任一失败都会立即终止。没有`default_metric_policy`或`gating`
开关；省略`metrics`、配置空列表、未知名称或重复名称都会在前向前报配置错误。

`cosine_similarity`只用于说明第三方指标可以采用的名称和形态，生产代码没有内置该指标。
只有先安装了提供`msmodelslim.lossless_metric.plugins`入口点的第三方插件包，下面的配置才可使用：

```yaml
spec:
  dataset: test.json
  process:
    - type: "lossless_check"
      include:
        - "model.layers.0.self_attn.*_proj"
      metrics:
        - name: "cosine_similarity"
          threshold: 0.9999
      configs:
        - type: "quarot"
          online: False
          block_size: -1
```

### 接入第三方指标

第三方指标是一个普通Python包。接入方需要提供配置类、指标实现和入口函数；容器、QuaRot和
模型Adapter都不需要修改。以下示例实现余弦相似度指标：

```python
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import Field

from msmodelslim.processor.lossless.metrics import (
    BaseLosslessMetric,
    LosslessMetricConfig,
    MetricResult,
    MetricViolation,
)


class CosineSimilarityMetricConfig(LosslessMetricConfig):
    name: Literal["cosine_similarity"] = "cosine_similarity"
    threshold: float = Field(ge=-1.0, le=1.0)


class CosineSimilarityMetric(BaseLosslessMetric):
    def evaluate(self, reference, processed, context):
        reference = reference.float().reshape(-1)
        processed = processed.float().reshape(-1)
        if reference.numel() == 0:
            value = 1.0
        else:
            value = F.cosine_similarity(
                reference.unsqueeze(0),
                processed.unsqueeze(0),
                dim=1,
                eps=1e-12,
            ).item()
        passed = value >= self.config.threshold
        violations = () if passed else (
            MetricViolation(
                name=self.name,
                value=value,
                threshold=self.config.threshold,
                comparison=">=",
            ),
        )
        return MetricResult(
            name=self.name,
            passed=passed,
            value=value,
            threshold=self.config.threshold,
            violations=violations,
        )


def get_plugin():
    return CosineSimilarityMetricConfig, CosineSimilarityMetric
```

在插件包的`pyproject.toml`中声明入口点：

```toml
[project.entry-points."msmodelslim.lossless_metric.plugins"]
cosine_similarity = "my_lossless_metrics.cosine:get_plugin"
```

安装该包后，前面的`cosine_similarity` YAML配置即可使用。`name`必须与入口点名称和配置类
的Literal值一致；阈值范围由插件配置类校验；`evaluate`不得修改输入Tensor或模型，并且必须
返回`MetricResult`。返回结果的`name`必须与当前指标名一致，`passed`必须是`bool`，
`observations`必须是字典且不能覆盖容器保留字段，`violations`必须是由`MetricViolation`
组成的元组。未知插件、错误组件类型、缺失阈值和非法阈值会在模型执行前报错；运行时返回
合同不合法则按`metric_execution_error`立即终止。

### 检查范围与判定

容器在Adapter已有请求范围内捕获目标结构的input和output，并按真实调用顺序逐次比较。
同一结构被多次调用时，每次调用使用独立的`invocation`编号。离线QuaRot右旋恢复处理后
input坐标，左旋恢复处理后output坐标；同一目标存在多条旋转命令时按严格逆序恢复。
Norm融合、bake mean、split旋转和全局旋转也会对齐到共同语义空间。

`include/exclude`只限制Linear比较范围，不改变子Processor的处理范围。未命中的表达式会统一告警；
范围内理论目标不存在时报告`structure_not_found`。模型中存在但Adapter请求未暴露，
或两次前向均未执行的结构写入`skipped_targets`。结构只在一次前向执行时，
以`invocation_count_mismatch`立即失败。

容器保证两次前向可比：比较期间临时使用评估态，处理后前向前回放Python、CPU及
已初始化加速器的随机状态，退出时逐模块恢复训练状态。内部前向使用`no_grad`；
成功后随机状态等价于只执行一次正式处理后前向，异常时恢复到请求开始前状态。

指标按YAML顺序执行，常规通过时输出INFO日志。检测到首个指标或结构异常后先写入结果，
再输出ERROR日志并抛出异常，不继续执行后续指标和结构。日志与结果会标明`processor`、`decoder`、`structure`、
`structure_type`、`side`、`invocation`、`path`、`reason`、`metric_name`、
`metric_value`和`threshold`。指标失败时使用四条连续ERROR记录，依次给出结论、失败位置、
`failed_metrics`和`key_metrics`。其中`failed_metrics`逐项标明未达标子指标的实测值、
期望比较关系和阈值，`key_metrics`保留本次判定的其他关键观测值。常见`reason`包括：

- `metric_threshold_exceeded`：指标超过阈值。
- `structure_not_found`：理论目标无法在模型中解析。
- `invocation_count_mismatch`或`invocation_order_mismatch`：处理前后调用轨迹不一致。
- `shape_mismatch`、`dtype_mismatch`或`device_mismatch`：Tensor基础属性变化。
- `non_finite_value`：出现NaN或Inf。
- `alignment_error`：子Processor无法恢复到共同语义空间。
- `protocol_contract_invalid`：Processor三方法或Plan/Context返回数据不符合约定。

运行结果写入Context的`lossless_check.state.result`。结果只保存可序列化摘要，
不保存Tensor、Module、Hook或请求级Context。完整通过时`passed: true`；能力不足或目标未完整覆盖时
`passed: null`并给出`skip_reason`；指标或结构异常时`passed: false`并记录首个异常。

自检实际启用时，不支持分布式执行，也不支持需要KV Cache的子Processor。若因能力不足跳过自检，
容器只代理子Processor原有行为。

### 从配置到复验

1. 先用1至2条代表性样本和较小`include`范围运行，确认日志中出现目标Linear的INFO记录。
2. 查看结果中的`checked_targets`、`skipped_targets`、`missing_targets`、`unmatched_include`和
   `unmatched_exclude`，只有真正执行指标且无异常时才是`passed: true`。
3. 若为`passed: null`，根据`skip_reason`补Adapter前向能力或能触发目标的样本，不要把它当作通过。
4. 若为`passed: false`，从首条ERROR中的processor、decoder、structure、side和metric定位问题。
5. 修复旋转计划或模型适配后，从干净权重重新运行。异常终止不会事务式回滚已经修改的模型。

当前自检只支持离线QuaRot、单子Processor、非分布式、无需KV Cache的场景，并只验收实际执行的
Linear输入输出。在线QuaRot和request最终输出比较不在本轮范围内。

## 6. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| QuaRot 旋转量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | 《[QuaRot 旋转量化算法 量化术语百科词条](./term_quarot.md)》 |

## 7. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| quarot 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | 《[quarot 配置说明](../../../api_reference/config/processor/quarot.md)》 |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | 《[modelslim_v1 配置说明](../../../api_reference/config/task/modelslim_v1.md)》 |
