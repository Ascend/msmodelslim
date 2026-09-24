# SmoothQuant 平滑量化算法使用指南

## 1. 适用范围

本指南面向需要使用 [SmoothQuant 平滑量化算法](./term_smooth_quant.md) 的用户。SmoothQuant 作为离群值抑制算法，通常作为量化前的预处理步骤，用于将激活离群值按比例迁移到权重，提升低比特量化的精度。

适用场景：

- 激活分布存在显著离群通道（outliers），导致直接进行 W8A8 量化时精度下降；
- 模型具有标准的 `Norm-Linear` 结构（如 LayerNorm/RMSNorm 后接 Linear），可通过数学等价变换将激活动态范围迁移到权重。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及 `SmoothQuantInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为前向张量 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors` 分片 | 导出完整且推理冒烟测试通过 |

## 3. 流程总览

SmoothQuant 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型平滑接口] --> B[统计<br/>激活与权重幅值]
    B --> C[按 alpha<br/>分配量化难度]
    C --> D[迁移<br/>激活离群值到权重]
    D --> E[融合平滑尺度]
    E --> F[衔接低比特量化]
```

各阶段的关键细节如下：

- **适配模型平滑接口**：适配器需实现 `SmoothQuantInterface`，提供模型的 `Norm-Linear` 子图结构映射，是平滑算法定位目标模块的前提。
- **统计激活与权重幅值**：通过校准数据集（推荐 50 条）前向运行，统计各通道的最大绝对值激活与权重幅值。
- **按 alpha 分配量化难度**：通过超参数 `alpha` 平衡激活与权重的缩放因子。
- **迁移激活离群值到权重**：计算通道级缩放向量，将激活离群幅度等价迁移至后续线性层权重中。
- **融合平滑尺度**：将缩放因子的逆变换吸收融合到前驱的 LayerNorm/RMSNorm 中，保证数学等价。
- **衔接低比特量化**：平滑完成后张量分布更平缓，衔接 `linear_quant` 进行 W8A8 离线量化。

## 4. 操作步骤

### 步骤 1：适配模型流水线与平滑接口

**目标**：量化工具不能直接操作任意结构的模型，需要适配器将模型操作转换为标准方法。对于 SmoothQuant，模型适配器需要继承并实现流水线接口及平滑接口。模型适配必须先完成，才能执行 SmoothQuant 算法。

**操作**：模型适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总提供：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`SmoothQuantInterface`**（位于 `msmodelslim.processor.anti_outlier.smooth_quant.interface`）：平滑量化专用接口。

```python
from abc import ABC, abstractmethod
from typing import List
from msmodelslim.core.graph.adapter_types import AdapterConfig

class SmoothQuantInterface(ABC):
    @abstractmethod
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """返回待平滑的子图拓扑配置，描述 Norm 与后续 Linear 之间的连接关系。"""
        ...
```

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- SmoothQuant 算法适配接口（本步骤重点）----
    SmoothQuantInterface,  # 算法适配：get_adapter_config_for_subgraph 返回 norm-linear 映射拓扑
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface, SmoothQuantInterface):
    def get_adapter_config_for_subgraph(self):
        # 返回模型各层的 norm-linear 映射拓扑
        ...
```

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方，优先使用已验证配方。

```yaml
spec:
  process:
    - type: "smooth_quant"
      alpha: 0.5
      symmetric: true
      include: ["*"]
      exclude: []
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

ModelSlim 实现入口：
[查看对应实现目录](../../../../../msmodelslim/processor/anti_outlier/smooth_quant)

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定 `smooth_quant`。当前处理器主要作用于 `norm-linear` 平滑结构。 | 固定。 | 不调。 |
| `alpha` | SmoothQuant 激活/权重动态范围迁移指数，0~1，默认 `0.5`。alpha 增大通常更强地压缩激活离群，把量化困难转移到权重；减小则更保守地保护权重。 | 从 `0.5` 起步。 | 如果后续激活量化误差明显主导，可适度增大；如果权重低比特误差成为主要瓶颈则减小。每次比较要固定后续 qconfig，否则无法判断 alpha 的真实作用。 |
| `symmetric` | 决定平滑后量化假设；当前实现用它决定是否启用 shift（非对称时可对 norm-linear 做中心平移）。 | 常规 W8A8 对称方案用 `true`。 | 只有后续量化确实采用非对称并且部署链支持 shift/fusion 时才设 false。它应与后续量化方案一致。 |
| `include` | 作用范围白名单，使用模块名模式决定哪些匹配到的模块进入当前算法。 | 无模型专用配方时从 `["*"]` 开始；已有配方时直接沿用其模块范围。 | 先保证范围覆盖预期模块，再看精度。收窄范围时优先按结构族或已知敏感层调整。 |
| `exclude` | 作用范围黑名单，命中后从 `include` 的候选中排除，优先级高于 `include`。适合保护敏感层、首尾层或模型专用不兼容结构。 | 默认先保持空列表；只有实践配方、兼容性约束或敏感性结果给出明确证据时再加入。 | 局部精度问题优先通过 `exclude` 做小范围回退，比提高全模型位宽或关闭整个算法更容易保留收益。 |

#### 参数组合与选择顺序

**最后只调一个主要旋钮**：`alpha` 的方向应由“激活侧还是权重侧更难量化”决定。先用 0.5，在同一 W/A qconfig 下比较误差，再只沿一个方向调整。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：SmoothQuant + W8A8 静态量化

##### 配置文件：`smooth_quant_w8a8.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  process:
    - type: smooth_quant
      alpha: 0.5
      symmetric: true
      include: ["*"]
    - type: linear_quant
      qconfig:
        act:
          dtype: int8
          scope: per_token
          symmetric: true
          method: minmax
        weight:
          dtype: int8
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
  --config ./smooth_quant_w8a8.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| SmoothQuant 平滑量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《SmoothQuant 平滑量化算法 量化术语百科词条》](./term_smooth_quant.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线适配接口（数据预处理、模型加载、模块遍历）。 | [《LLM 量化使用指南·步骤 1》](../../ptq/llm/usage_large_language_model_quantization.md) |
| `SmoothQuantInterface` | SmoothQuant 算法模型适配接口，由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| smooth_quant 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《smooth_quant 配置说明》](../../../api_reference/config/processor/smooth_quant.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/quant/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/zh/user_guide/usage_weight_quantization.md) |
