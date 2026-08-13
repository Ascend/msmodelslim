# LLM 模型接入案例——以 DeepSeek V4 为例

## 1. 适用范围

本文档面向需要将**新的 Decoder-only 大语言模型**接入 msModelSlim 量化流程的开发人员。通过分析 DeepSeek V4 模型适配器的完整实现，展示从创建适配器到注册模型、执行量化的全流程。

**目标**：掌握 LLM 模型接入 msModelSlim 的代码开发规范，理解模型适配器（Model Adapter）的设计模式与接口实现方法。

**覆盖流程**：模型适配器创建 → 组件接口实现 → 配置注册 → YAML 量化配置 → 量化验证

**关联流程**：[LLM 量化使用指南](usage_large_language_model_quantization.md)

## 2. 环境与版本

| 项 | 版本或配置 |
| --- | --- |
| 产品形态 | Atlas 800I A2 / Atlas A3（不限定） |
| CANN | 参考安装指南 |
| PyTorch | >= 2.7.1 (推荐使用镜像配置2.10.0) |
| Transformers | == 4.48.2 |
| 其他依赖 | 参考requirements.txt |

**本次前置事实**：

- msModelSlim 工具已安装并可正常执行 `msmodelslim --help`
- 目标模型的浮点权重已下载到本地（HuggingFace 格式）
- 目标模型在已有支持矩阵之外，需要新增适配

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重 | `${MODEL_PATH}` | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | `lab_calib/` 或用户指定 | JSONL 格式，每条含文本 prompt | 至少 128条样本 |
| 交付件 | 模型适配器代码 | `msmodelslim/model/{model_name}/` | Python 源码，含 loader、adapter | 代码审查通过 |
| 交付件 | 量化配置 | `lab_practice/{model_name}/` | YAML 格式 | 配置校验通过 |
| 交付件 | 量化权重 | `${SAVE_PATH}` | 含 `quant_model_description.json` 及 `*.safetensors` | 推理冒烟通过 |

## 4. 操作步骤

### 步骤 1：分析模型结构，确定适配策略

**目标**：分析目标模型的结构特点，确定量化接入策略。

**操作**：

1. **分析模型架构**：查阅目标模型的官方实现，了解其结构组成。Decoder-only LLM 通常由 Embedding、多个 Decoder Layer、Norm 和 LM Head 组成。
2. **确定加载策略**：确认模型是否支持 `from_pretrained` 加载。若支持，可直接使用 `TransformersModel` 基类；若不支持，需自定义模型定义（如 DeepSeek V4 自定义 `Transformer` 类）。
3. **确定量化策略**：根据模型规模和推理场景，确定是否需要逐层量化（Layer-wise）、MTP 支持、异常值抑制算法等。

**分析要点**：

- Decoder Layer 结构：Attention 和 FFN 的子模块命名
- 特殊结构：MTP（Multi-Token Prediction）、MoE（Mixture of Experts）、压缩注意力等
- 权重格式：FP8 权重是否需要反量化

**DeepSeek V4 结构特点**：

- 自定义模型定义（非 Transformers 标准库），继承 `Transformer` 类
- 支持 MTP（Multi-Token Prediction）结构
- 使用压缩注意力（Compressed Attention）机制
- 权重可能为 FP8 格式，需调用 `auto_dequant_state_dict` 反量化

**输出**：模型结构分析文档，确定适配策略

### 步骤 2：创建模型适配器目录与文件

**目标**：创建模型适配器目录结构，包含必要的文件。

**操作**：

在 `msmodelslim/model/` 下创建模型目录，目录名建议与模型系列名一致：

```text
msmodelslim/model/{model_name}/
├── __init__.py          # 导出适配器类，可为空
├── model_adapter.py     # 主适配器文件
├── loader.py            # 加载器文件
├── model.py             # （可选）自定义模型定义
├── mtp_quant_module.py  # （可选）MTP 量化支持
└── convert_fp8_to_bf16.py  # （可选）FP8 权重转换
```

**DeepSeek V4 参考结构**：

```text
msmodelslim/model/deepseek_v4/
├── __init__.py
├── model_adapter.py     # DeepSeekV4ModelAdapter 类
├── loader.py            # DeepseekV4AdapterLoader 类
├── model.py             # Transformer, ModelArgs, Block 等
├── mtp_quant_module.py  # MTP 层量化支持
└── convert_fp8_to_bf16.py  # FP8→BF16 反量化
```

**输出**：模型适配器目录结构创建完成

### 步骤 3：实现 Loader 加载器

**目标**：实现加载器类，注册适配器类的导入路径。

**操作**：

加载器继承 `BaseModelAdapterLoader`，只需指定适配器类的完整导入路径：

```python
# msmodelslim/model/{model_name}/loader.py
from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class {ModelName}AdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.{model_name}.model_adapter:{ModelName}ModelAdapter"
```

**DeepSeek V4 参考**：`msmodelslim/model/deepseek_v4/loader.py:7`

```python
class DeepseekV4AdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.deepseek_v4.model_adapter:DeepSeekV4ModelAdapter"
```

**输出**：加载器文件创建完成

### 步骤 4：定义适配器类，继承所需接口

**目标**：定义模型适配器主类，继承基础适配器和所需组件接口。

**操作**：

适配器类继承规则：

- **必须继承** `BaseModelAdapter`（或其子类如 `TransformersModel`）和 `ModelSlimPipelineInterfaceV1`
- **可选继承**：根据量化算法需求，实现 `IterSmoothInterface`、`FlexSmoothQuantInterface`、`QuaRotInterface` 等

```python
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.interface_hub import (
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    QuaRotInterface,
)
from msmodelslim.utils.logging import logger_setter


@logger_setter("msmodelslim.model.{model_name}")
class {ModelName}ModelAdapter(
    TransformersModel,
    ModelSlimPipelineInterfaceV1,  # 必需：量化调度支持
    IterSmoothInterface,           # 可选：异常值抑制
    FlexSmoothQuantInterface,      # 可选：Flex Smooth
    QuaRotInterface,               # 可选：QuaRot 旋转
):
    ...
```

**DeepSeek V4 参考**：`msmodelslim/model/deepseek_v4/model_adapter.py:52-61`

```python
@logger_setter("msmodelslim.model.deepseek_v4")
class DeepSeekV4ModelAdapter(
    TransformersModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    QuaRotInterface,
    AscendV1SaveInterface,
):
```

**输出**：适配器类骨架定义完成

### 步骤 5：实现核心接口方法

**目标**：实现适配器的核心接口方法，使模型可被量化框架调度。

**操作**：

#### 5.1 `handle_dataset`——处理校准数据集

将校准数据转换为模型可接受的输入格式。对于标准 Transformers 模型，可直接使用基类提供的 `_get_tokenized_data`：

```python
def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
    return self._get_tokenized_data(dataset, device)
```

#### 5.2 `init_model`——初始化模型

**关键策略**：仅加载第一层 Decoder Layer，其余层在后续按需加载，以节省显存。

```python
def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
    # 1. 保存原始层数，临时设置为 1
    origin = self.config.num_hidden_layers
    self.config.num_hidden_layers = 1

    # 2. 加载模型（仅第一层 Decoder Layer）
    with torch.device("cpu"):
        model = Transformer(self.config)

    # 3. 恢复原始层数
    self.config.num_hidden_layers = origin

    # 4. 加载完整权重
    state_dict = get_state_dict(self.model_path, model)
    # 若为 FP8 权重，需反量化
    auto_dequant_state_dict("", state_dict, str(self.model_path))
    model.load_state_dict(state_dict)
    model.eval()

    return model
```

#### 5.3 `generate_model_visit`——生成模型访问序列

按模型结构的拓扑顺序，依次 yield 每个需要量化的模块：

```python
def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
    return generated_decoder_layer_visit_func(
        model, transformer_blocks=self.generate_decoder_layer(model)
    )
```

#### 5.4 `generate_decoder_layer`——按需加载 Decoder Layer

实现逐层加载逻辑，按索引从 safetensors 权重文件中加载：

```python
def generate_decoder_layer(self, model: nn.Module):
    for idx in range(self.config.num_hidden_layers):
        layer_prefix = f"layers.{idx}"
        decoder = self.load_decoder_if_not_exist(model, layer_prefix=layer_prefix, idx=idx)
        yield layer_prefix, decoder

def load_decoder_if_not_exist(self, model, layer_prefix, idx):
    # 检查是否已加载
    try:
        decoder = model.get_submodule(layer_prefix)
        return decoder
    except AttributeError:
        pass

    # 按需创建并加载
    with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
        module_list = model.layers
        template_module = module_list[0]
        decoder = template_module.__class__(layer_id=idx, args=self.config)

        state_dict = get_state_dict(self.model_path, decoder, prefix=layer_prefix)
        decoder.load_state_dict(state_dict)
        decoder.eval()
        module_list.append(decoder)

    return decoder
```

#### 5.5 `generate_model_forward`——生成模型前向传播序列

实现逐层前向传播，通过 hook 机制获取第一层输入后逐层传递：

```python
def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
    # 1. 通过 hook 获取第一层输入
    first_block_input = None

    def break_hook(module, hook_args, hook_kwargs):
        nonlocal first_block_input
        first_block_input = (hook_args, hook_kwargs)
        raise TransformersForwardBreak()

    remove_handler = model.layers[0].register_forward_pre_hook(
        break_hook, with_kwargs=True, prepend=True
    )

    try:
        model(inputs[0] if isinstance(inputs, list) else inputs)
    except TransformersForwardBreak:
        pass
    finally:
        remove_handler.remove()

    # 2. 逐层前向传播
    args, kwargs = first_block_input
    for name, block in self.generate_decoder_layer(model):
        h = yield ProcessRequest(name, block, args, kwargs)
        args = (h, *args[1:])
```

#### 5.6 可选：实现异常值抑制接口

若需要支持 IterSmooth / FlexSmoothQuant 等异常值抑制算法，需实现 `get_adapter_config_for_subgraph`，定义 LayerNorm 与 Linear 层的融合映射关系：

```python
def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
    adapter_config = []
    for layer_idx in range(self.config.num_hidden_layers):
        # Norm-Linear 融合
        input_norm_mapping = MappingConfig(
            source=f"layers.{layer_idx}.attn_norm",
            targets=[f"layers.{layer_idx}.attn.wq_a", f"layers.{layer_idx}.attn.wkv"],
        )
        adapter_config.append(AdapterConfig(
            subgraph_type="norm-linear", mapping=input_norm_mapping
        ))

        # Up-Down 融合（FFN）
        up_down_mapping = MappingConfig(
            source=f"layers.{layer_idx}.ffn.experts.expert.w3",
            targets=[f"layers.{layer_idx}.ffn.experts.expert.w2"],
        )
        adapter_config.append(AdapterConfig(
            subgraph_type="up-down", mapping=up_down_mapping
        ))
    return adapter_config
```

**输出**：适配器接口方法实现完成

### 步骤 6：注册模型到配置

**目标**：在 `config/config.ini` 中注册模型名称和适配器入口点。

**操作**：

在 `config/config.ini` 中添加两项配置：

```ini
[ModelAdapter]
# 在 [ModelAdapter] 部分添加模型名称映射
{model_name} = {ModelName1}, {ModelName2}

[ModelAdapterEntryPoints]
# 在 [ModelAdapterEntryPoints] 部分添加适配器入口
{model_name} = msmodelslim.model.{model_name}.loader:{ModelName}AdapterLoader

[ModelAdapterDependencies]
# 可选：指定 Transformers 版本依赖
{model_name} = {"transformers": "==x.x.x"}
```

**DeepSeek V4 参考**：

```ini
[ModelAdapter]
deepseek_v4 = DeepSeek-V4-Flash, DeepSeek-V4-Pro

[ModelAdapterEntryPoints]
deepseek_v4 = msmodelslim.model.deepseek_v4.loader:DeepseekV4AdapterLoader

[ModelAdapterDependencies]
deepseek_v4 = {"transformers": "==4.48.2"}
```

**输出**：模型注册完成

### 步骤 7：创建 YAML 量化配置文件

**目标**：在 `lab_practice/` 下创建对应的量化配置文件。

**操作**：

在 `lab_practice/{model_name}/` 下创建 YAML 文件，定义量化方案：

```yaml
# lab_practice/{model_name}/{model_name}_{quant_type}.yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: True
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: True
          method: "minmax"
      include:
        - "*"
      exclude:
        - "*gate*"  # 排除 router 等特殊层

  dataset: mix_calib.jsonl
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

**DeepSeek V4 参考**：`lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml`

**输出**：量化配置文件创建完成

### 步骤 8：执行量化验证

**目标**：使用量化命令验证模型适配器工作正常。

**操作**：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config_path ${CONFIG_PATH} \
  --trust_remote_code False
```

**验证要点**：

1. 模型加载正常，无 ImportError
2. 校准数据集加载成功
3. 逐层量化和前向传播无报错
4. 量化权重正常保存，输出目录包含 `quant_model_description.json`

**输出**：量化验证通过

## 5. 结果与经验

### 5.1 关键结果汇总

| 步骤 | 关键操作 | 指标 | 变化 | 备注 |
| --- | --- | --- | --- | --- |
| 适配器创建 | 定义 adapter 类，继承 TransformersModel + PipelineInterfaceV1 | 代码行数 | 基础适配器约300行 | 不含可选接口实现 |
| 模型注册 | config.ini 配置 ModelAdapter + EntryPoints | 配置项 | 新增2项 | 含版本依赖声明 |
| 核心接口实现 | handle_dataset / init_model / generate_model_visit/forward | 接口方法 | 4个核心方法 | 逐层加载策略节省显存 |
| YAML 配置 | 创建量化配置 | 配置项 | 1个YAML文件 | 支持多种量化方案 |

### 5.2 经验总结

1. **逐层加载策略**：对于大模型，在 `init_model` 中仅加载第一层 Decoder Layer，其余层在 `generate_decoder_layer` 中按需加载，可显著降低显存占用。适用边界：所有 Decoder-only 架构，需确保 `__class__` 一致性。
2. **Hook 获取首层输入**：使用 `register_forward_pre_hook` 配合 `TransformersForwardBreak` 中断首次前向传播，可获取第一层 Decoder Layer 的输入参数，适用于所有逐层量化场景。
3. **自定义模型 vs Transformers 标准库**：若模型使用 Transformers 标准库实现，可直接继承 `TransformersModel` 节省工作量；若模型为自定义实现（如 DeepSeek V4），需自行实现 `_load_config`、`get_state_dict` 等底层方法。
4. **接口按需实现**：`ModelSlimPipelineInterfaceV1` 是唯一必需接口，其余算法接口（如 `IterSmoothInterface`、`QuaRotInterface`）仅在需要对应算法时实现，避免过度设计。

## 6. 异常处理

- **模型加载失败**：检查 `config.ini` 中 ModelAdapter 和 ModelAdapterEntryPoints 的 key 是否一致。若不一致，配置不生效。
- **显存不足（OOM）**：确认 `init_model` 中将 `num_hidden_layers` 临时设置为 1；确认 `generate_decoder_layer` 实现了按需加载而非全量加载。此外，若仍遇到显存不足，请确认 `--device npu:0` 指定的 NPU 未被其他任务占用。
- **权重加载错误**：若模型使用 FP8 权重，需在加载后调用反量化函数；检查 `get_state_dict` 的 prefix 是否与权重文件中的 key 一致。
- **校准数据格式错误**：确认 `handle_dataset` 返回的数据格式与模型的 forward 签名匹配。

## 7. 附录

- 参考实现：[DeepSeek V4 模型适配器](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/deepseek_v4/model_adapter.py)
- 参考配置：[DeepSeek V4 Flash W8A8 配置](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml)
- 接口定义：[Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/interface_hub.py)
- 基础指南：[LLM Model Integration Guide (英文)](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/developer_guide/integrating_models.md)
