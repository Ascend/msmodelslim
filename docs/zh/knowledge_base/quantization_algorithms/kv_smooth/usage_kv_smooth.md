# KV Smooth 缓存平滑算法使用指南

## 1. 适用范围

本指南面向需要使用 [KV Smooth 缓存平滑算法](./term_kv_smooth.md) 的用户。KV Smooth 作为离群值抑制算法，通常作为 KV Cache 量化的前置步骤，用于压缩 Key 的动态范围，提升缓存量化的精度。

适用场景：

- KV Cache 量化后精度下降，Key 通道存在显著离群值，动态范围过大；
- 需要将 Key 的动态范围压力重新分配到 Query 侧，使 KV Cache 更容易被低比特表示。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及 `KVSmoothFusedInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为前向张量 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors` 分片 | 导出完整且推理冒烟测试通过 |

## 3. 流程总览

KV Smooth 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线/算法接口] --> B[确定 KV Cache<br/>量化目标]
    B --> C[统计 Key<br/>通道幅值]
    C --> D[计算平滑尺度]
    D --> E[迁移 Key 离群值<br/>到 Query]
    E --> F[衔接<br/>KV Cache 量化]
```

各阶段的关键细节如下：

- **适配模型流水线/算法接口**：适配器需实现 `PipelineInterface`（基础流水线接口）及 `KVSmoothFusedInterface`（KV 平滑专用接口），提供模型 Q/K/V 子图结构信息，是后续所有平滑步骤的前提。模型适配必须先完成，才能执行 KV Smooth 算法。
- **确定 KV Cache 量化目标**：确认 KV Cache 目标位宽与部署后端（如 C8），通过 `include`/`exclude` 圈定作用模块范围。
- **统计 Key 通道幅值**：通过校准数据集（推荐 50 条）前向运行，根据 Key 绝对最大值并按 RoPE 成对维度汇总通道幅值。
- **计算平滑尺度**：使用 `scale = key_abs_max ** smooth_factor`（`smooth_factor` 默认 `1.0`）计算通道级平滑尺度。
- **迁移 Key 离群值到 Query**：K 侧权重/偏置除以 scale，Q 侧乘以对应 scale，从而把 K 的动态范围压力转移到 Q，保证注意力计算数学等价。
- **衔接 KV Cache 量化**：平滑完成后 Key 分布更平缓，衔接 KV Cache 量化（如 C8）即可提升精度；建议保持后续量化配置不变，只调整 `smooth_factor` 以定位收益。

## 4. 操作步骤

### 步骤 1：适配模型流水线与算法接口

**目标**：量化工具不能直接操作任意结构的模型，需要适配器将模型操作转换为标准方法。对于 KV Smooth，模型适配器需要继承并实现流水线接口及 KV 平滑接口。模型适配必须先完成，才能执行 KV Smooth 算法。

**操作**：模型适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总提供：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`KVSmoothFusedInterface`**：KV 平滑专用接口，由 `msmodelslim.model.interface_hub` 汇总导出，提供模型 Q/K/V 子图结构信息，供平滑处理器定位 K 侧权重/偏置与 Q 侧投影的对应关系。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- KV Smooth 算法适配接口（本步骤重点）----
    KVSmoothFusedInterface,  # 算法适配：返回 K/V 平滑融合子图（get_kvcache_smooth_fused_subgraph）及 head_dim、KV 组数等结构信息
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface, KVSmoothFusedInterface):
    pass
```

各接口方法说明：

- **`PipelineInterface`** 的标准方法：加载数据（`handle_dataset`）、加载模型（`init_model`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）、控制 KV Cache（`enable_kv_cache`）。量化调度器（Runner）与处理器（Processor）只通过这些标准方法驱动模型，与模型内部结构无关。
- **`KVSmoothFusedInterface`** 的核心方法：提供模型 Q/K/V 子图结构信息（如 K 侧权重/偏置名称、Q 侧投影名称、RoPE 成对维度等），平滑处理器据此定位并融合平滑尺度。

之后量化命令中通过 `--model_type MyModelAdapter` 引用该适配器。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。请注意：模型适配必须先完成，才能执行后续量化流程。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方，优先使用已验证配方。

```yaml
spec:
  process:
    - type: "kv_smooth"
      smooth_factor: 1.0
      include: ["*"]
      exclude: []
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

ModelSlim 实现入口：
[查看对应实现目录](../../../../../msmodelslim/processor/kv_smooth)

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定为 `kv_smooth`。 | 固定。 | 不调。 |
| `smooth_factor` | KV 平滑强度，必须大于 0，默认 `1.0`。实现根据 Key 绝对最大值并按 RoPE 成对维度汇总，然后使用 `scale = key_abs_max ** smooth_factor`；K 侧权重/偏置除以 scale，Q 侧乘以对应 scale，从而把 K 的动态范围压力转移到 Q。 | 从默认 `1.0` 起步。 | 它不是简单线性倍率：因采用幂函数，原始 max 大于/小于 1 时响应不同。K 离群仍明显、KV Cache 量化主要受 K 范围限制时可在小范围内增大；若 Q 侧幅值/量化误差明显恶化则回调。每次调整都应同时看 QK attention 误差与 KV Cache 量化结果。 |
| `include` | 作用范围白名单，使用模块名模式决定哪些匹配到的模块进入当前算法。 | 无模型专用配方时从 `["*"]` 开始；已有 `lab_practice` 时直接沿用其模块范围。 | 先保证范围覆盖预期模块，再看精度。收窄范围时优先按结构族或已知敏感层调整。 |
| `exclude` | 作用范围黑名单，命中后从 `include` 的候选中排除，优先级高于 `include`。 | 默认先保持空列表；只有实践配方、兼容性约束或敏感性结果给出明确证据时再加入。 | 局部精度问题优先通过 `exclude` 做小范围回退，比提高全模型位宽或关闭整个算法更容易保留收益。 |

#### 参数组合与选择顺序

**最后只调一个主要旋钮**：KV Smooth 的目标是重新分配 Q/K 数值范围，不是单向“越平越好”。先用 `1.0`，再根据 **K 是否仍是瓶颈** 和 **Q 是否被放大过度** 决定方向，并保持同一校准集和评测集。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：KV Smooth + KV Cache C8 量化（推荐起点）

##### 配置文件：`kv_smooth_c8.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  process:
    - type: kv_smooth               # KV Smooth 缓存平滑预处理
      smooth_factor: 1.0            # KV 平滑强度，默认 1.0
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
      kv_cache:
        k_dtype: int8               # KV Cache C8 量化
        v_dtype: int8
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
  --config ./kv_smooth_c8.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| KV Smooth 缓存平滑算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《KV Smooth 缓存平滑算法 量化术语百科词条》](./term_kv_smooth.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` / `KVSmoothFusedInterface` | 模型流水线与 KV Smooth 算法适配接口，均由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| kv_smooth 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《kv_smooth 配置说明》](../../../api_reference/config/processor/kv_smooth.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/quant/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/zh/user_guide/usage_weight_quantization.md) |
