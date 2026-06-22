# Attention MSE（mse）：敏感层分析算法说明

## 简介

- **概述**：`mse`（均方误差，Mean Squared Error）用于**attn**范围分析：分别使用浮点权重与量化权重执行前向推理，对同一 attention 模块的输出计算均方误差，输出**注意力模块粒度**排序。
- **核心思想**：直接度量注意力子系统在量化权重下的输出漂移；数值越大表示该注意力层对权重量化越敏感。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../install_guide/install_guide.md)》。

## 原理

1. 对同一校准样本，分别使用**浮点权重**与**量化权重**执行前向，在 attention 模块输出处采集张量。
2. 对同一层、同一样本的浮点与量化输出计算 MSE：

    $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float}}^{(i)} - y_{\text{quant}}^{(i)})^2$$

3. **解读**：MSE 越大，该 attention 模块对当前量化配置越敏感。

## 适用要求

- **推荐场景**：需要对 **Attention** 结构做权重量化或评估其敏感度时。
- **模型适配（必选）**：对应 `model_type` 的模型适配器必须实现 `AttentionMSEAnalysisInterface`，提供模块类名与输出提取函数；未实现会在分析阶段报错。
- **model_type**：工具当前仅实现了以下模型的接口适配，其他 `model_type` 会报错或需自行在适配器中实现接口。

| model_type       |
| ---------------- |
| DeepSeek-V3      |
| DeepSeek-V3-0324 |
| DeepSeek-R1      |
| DeepSeek-R1-0528 |
| DeepSeek-V3.1    |

## 功能介绍

### 使用说明

本分析依赖工具在 attention 子模块上挂 hook 并读取其前向输出；不同模型的 attention 类名与 `forward` 返回值形态不一致，无法由框架统一推断。因此须在目标 `model_type` 的**模型适配器**中实现 `AttentionMSEAnalysisInterface`（声明待 hook 的类名、以及如何从 `forward` 返回值中取出用于计算 MSE 的张量），`msmodelslim analyze attn --metrics mse` 方可在该模型上使用。以下为接口约定，未实现或实现与模型结构不一致时会在分析阶段报错。

```python
class AttentionMSEAnalysisInterface(ABC):
    @abstractmethod
    def get_attention_module_cls(self) -> str:
        ...

    @abstractmethod
    def get_attention_output_extractor(self) -> Callable[[Union[tuple, torch.Tensor]], torch.Tensor]:
        ...
```

| 方法 | 作用 |
|------|------|
| `get_attention_module_cls` | 返回待挂 hook 的 attention 模块类名字符串 |
| `get_attention_output_extractor` | 从 `forward` 返回值中取出用于计算 MSE 的张量 |

### 命令行示例

```bash
msmodelslim analyze attn \
    --model_type DeepSeek-V3 \
    --model_path ${model_path} \
    --metrics mse \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --device npu
```

### 命令行参数说明

| 参数 | 说明 |
|------|------|
| `attn` | 注意力结构敏感度分析 |
| `--metrics` | 指定分析算法，取值为 `mse` 时使用本算法 |

完整参数见[敏感层分析工具使用指南参数说明](../../feature_guide/sensitive_layer_analysis/usage.md#34-参数说明)。

## FAQ

### 报错提示未实现 `AttentionMSEAnalysisInterface`？

**现象**: 运行 `analyze attn --metrics mse` 时报错，提示未实现 `AttentionMSEAnalysisInterface`。

**解决方案**: 当前 `model_type` 的适配器未接入该分析路径；请换用支持列表中的模型类型，或在适配器中按接口实现 hook 类名与输出提取逻辑。
