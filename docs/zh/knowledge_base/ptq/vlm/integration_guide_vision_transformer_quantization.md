# VLM 模型接入案例——以 Qwen3.6 为例

## 1. 适用范围

本文档面向需要将**新的多模态视觉语言模型（VLM）**接入 msModelSlim 量化流程的开发人员。通过分析 Qwen3.6（Qwen3.5 MoE 系列）适配器的完整实现，展示 VLM 模型接入的特殊流程。

**目标**：掌握 VLM 模型接入 msModelSlim 的代码开发规范，理解多模态模型相比纯语言模型的差异点，包括视觉编码器加载、多模态校准数据、视觉-语言特征融合等。

**覆盖流程**：模型适配器创建 → 组件接口实现 → 配置注册 → 多模态校准数据 → YAML 量化配置 → 量化验证

**关联流程**：[VLM 量化使用指南](usage_vision_transformer_quantization.md)

## 2. 环境与版本

| 项 | 版本或配置 |
| --- | --- |
| 产品形态 | Atlas 800I A2 / Atlas A3（不限定） |
| CANN | 参考安装指南 |
| PyTorch | >= 2.7.1 (推荐使用镜像配置2.10.0) |
| Transformers | == 5.2.0 |
| 其他依赖 | safetensors, pyyaml, Pillow |

**本次前置事实**：

- msModelSlim 工具已安装并可正常执行 `msmodelslim --help`
- 目标 VLM 的浮点权重已下载到本地（HuggingFace 格式）
- 校准图像已准备（默认使用 `lab_calib/calibImages/` COCO 图像）

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点 VLM 权重 | `${MODEL_PATH}` | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 多模态校准数据集 | `lab_calib/calibImages/` 或用户指定 | 图像目录 + `index.jsonl`，含图像路径与文本 prompt | 至少 128张图像 |
| 交付件 | 模型适配器代码 | `msmodelslim/model/{model_name}/` | Python 源码，含 loader、adapter、moe_utils | 代码审查通过 |
| 交付件 | 量化配置 | `lab_practice/{model_name}/` | YAML 格式（`apiversion: multimodal_vlm_modelslim_v1`） | 配置校验通过 |
| 交付件 | 量化权重 | `${SAVE_PATH}` | 含 `quant_model_description.json` 及 `*.safetensors` | 推理冒烟通过 |

## 4. 操作步骤

### 步骤 1：分析 VLM 模型结构，确定适配策略

**目标**：分析目标 VLM 的结构特点，确定多模态量化的接入策略。

**操作**：

1. **分析模型架构**：VLM 通常由三部分组成：
   - **视觉编码器（Vision Encoder）**：如 ViT、CLIP，将图像转换为视觉特征
   - **视觉特征投影（Visual Projection）**：如 Merger、PatchMerger，将视觉特征映射到语言模型隐藏空间
   - **语言模型（Language Model）**：标准 Decoder-only LLM，处理融合后的多模态特征

2. **确定加载策略**：
   - **视觉部分全量加载**：所有视觉编码器层一次性加载（显存消耗固定）
   - **语言部分逐层加载**：仅加载第一层 Decoder Layer，其余按需加载

3. **确定特殊结构**：
   - MoE 层：是否需要 3D 融合权重转换
   - DeepStack：是否需要在特定层注入视觉特征
   - MTP（Multi-Token Prediction）：是否支持

**Qwen3.6 结构特点**：

- 视觉编码器 + PatchMerger 投影 + Qwen3.5 语言模型
- 支持 Dense 和 MoE 两种架构
- 使用 `AutoProcessor` 进行多模态预处理
- 视觉特征通过 `masked_scatter` 融合到文本嵌入中

**输出**：模型结构分析文档，确定适配策略

### 步骤 2：创建模型适配器目录与文件

**目标**：创建 VLM 模型适配器目录结构。

**操作**：

在 `msmodelslim/model/` 下创建模型目录：

```text
msmodelslim/model/{model_name}/
├── __init__.py          # 导出适配器类
├── model_adapter.py     # 主适配器文件（继承 VLMBaseModelAdapter）
├── loader.py            # 加载器文件
├── moe_utils.py         # （可选）MoE 融合权重转换工具
└── modeling_custom.py   # （可选）自定义模型定义
```

**Qwen3.5 MoE 参考结构**：

```text
msmodelslim/model/qwen3_5_moe/
├── __init__.py
├── model_adapter.py     # Qwen3_5ModelAdapter 类
├── loader.py            # Qwen3_5MoeAdapterLoader 类
├── moe_utils.py         # MoE 3D 权重转标准 Linear
└── modeling_qwen3_5_mtp.py  # MTP 预测器定义
```

**输出**：模型适配器目录结构创建完成

### 步骤 3：实现 Loader 加载器

**目标**：实现加载器类，注册适配器类的导入路径。

**操作**：

```python
# msmodelslim/model/{model_name}/loader.py
from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class {ModelName}AdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.{model_name}.model_adapter:{ModelName}ModelAdapter"
```

**Qwen3.5 MoE 参考**：`msmodelslim/model/qwen3_5_moe/loader.py:7`

```python
class Qwen3_5MoeAdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.qwen3_5_moe.model_adapter:Qwen3_5ModelAdapter"
```

**输出**：加载器文件创建完成

### 步骤 4：定义适配器类，继承所需接口

**目标**：定义 VLM 模型适配器主类，继承 VLM 基础适配器和量化调度接口。

**操作**：

VLM 适配器与 LLM 适配器的关键区别：

- **继承 `VLMBaseModelAdapter`** 而非 `TransformersModel`，提供多模态通用能力
- 视觉部分全量加载，语言部分逐层加载

```python
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.model.interface_hub import (
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
)
from msmodelslim.utils.logging import logger_setter


@logger_setter()
class {ModelName}ModelAdapter(
    VLMBaseModelAdapter,          # 提供多模态通用能力
    ModelSlimPipelineInterfaceV1,  # 必需：量化调度支持
    IterSmoothInterface,           # 可选：异常值抑制
    FlexSmoothQuantInterface,      # 可选：Flex Smooth
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)
```

**Qwen3.5 MoE 参考**：`msmodelslim/model/qwen3_5_moe/model_adapter.py:79-95`

```python
@logger_setter()
class Qwen3_5ModelAdapter(
    VLMBaseModelAdapter, ModelInfoInterface,
    ModelSlimPipelineInterfaceV1, IterSmoothInterface, FlexSmoothQuantInterface
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)
```

**输出**：适配器类骨架定义完成

### 步骤 5：实现核心接口方法

**目标**：实现 VLM 适配器的核心接口方法，包括多模态数据处理、视觉编码器加载、逐层前向传播等。

**操作**：

#### 5.1 `handle_dataset`——处理多模态校准数据

VLM 的校准数据包含图像和文本，需使用 `AutoProcessor` 进行预处理：

```python
def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
    # 1. 加载 Processor
    from transformers import AutoProcessor
    self._processor = AutoProcessor.from_pretrained(
        self.model_path, trust_remote_code=self.trust_remote_code, local_files_only=True
    )

    # 2. 逐样本处理
    processed_data = []
    for item in dataset:
        image_path = item.image
        text = item.text

        # 构建多模态消息格式
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": text},
            ]
        }]

        # 使用 Processor 进行 tokenize
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )

        # 将 tensor 移至目标设备
        processed_item = self._collect_inputs_to_device(
            inputs, device,
            keys=['input_ids', 'attention_mask', 'pixel_values',
                  'image_grid_thw', 'position_ids', 'cache_position'],
            defaults={'logits_to_keep': 0}
        )
        processed_data.append(processed_item)

    return processed_data
```

#### 5.2 `init_model`——初始化 VLM 模型

**关键策略**：视觉编码器全量加载，语言模型仅加载第一层：

```python
def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
    # 1. 保存原始语言层数，临时设置为 1
    origin_layers = self.config.text_config.num_hidden_layers
    self.config.text_config.num_hidden_layers = 1
    self.config.use_cache = False

    # 2. 加载模型（视觉全量 + 语言仅第一层）
    model = {ModelClass}.from_pretrained(
        self.model_path,
        config=self.config,
        trust_remote_code=self.trust_remote_code,
        torch_dtype="auto",
        local_files_only=True,
        device_map="cpu",
        attn_implementation='eager'
    ).eval()

    # 3. 恢复原始层数
    self.config.text_config.num_hidden_layers = origin_layers
    self.config.text_config._attn_implementation = 'eager'

    return model
```

#### 5.3 `generate_model_visit`——生成模型访问序列

VLM 的访问顺序：先视觉编码器（整体），再逐层语言 Decoder：

```python
def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
    # 1. 视觉编码器作为整体处理
    yield ProcessRequest(name="model.visual", module=model.model.visual, args=(), kwargs={})

    # 2. 语言模型逐层处理
    yield from generated_decoder_layer_visit_func(
        model, transformer_blocks=self.generate_decoder_layer(model)
    )
```

#### 5.4 `generate_decoder_layer`——按需加载语言 Decoder Layer

```python
def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
    num_layers = self.config.text_config.num_hidden_layers
    for layer_idx in range(num_layers):
        name = f"model.language_model.layers.{layer_idx}"
        layer = self._load_decoder_if_not_exist(model, name, layer_idx)
        yield name, layer

    # 若支持 MTP
    if self._has_mtp():
        mtp_layer = self._load_mtp_if_not_loaded(model)
        yield "mtp", mtp_layer

def _load_decoder_if_not_exist(self, model, name, idx):
    with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
        try:
            decoder = model.get_submodule(name)
            try:
                _ = decoder.input_layernorm.weight.device
                return decoder
            except RuntimeError:
                pass
        except AttributeError:
            pass

        # 创建 Decoder Layer 结构并加载权重
        module_list = model.model.language_model.layers
        template_module = module_list[0]
        decoder = template_module.__class__(config=self.config.text_config, layer_idx=idx)

        state_dict = self._get_state_dict(decoder, prefix=name)
        decoder.load_state_dict(state_dict)
        decoder.eval()

        # 若是 MoE 层，转换 3D 权重
        if self._is_moe_layer(idx):
            decoder.mlp = convert_experts_to_mlp(decoder.mlp, self.config.text_config)

        if len(module_list) <= idx:
            module_list.append(decoder)
        else:
            module_list[idx] = decoder

    return decoder
```

#### 5.5 `generate_model_forward`——生成多模态前向传播序列

VLM 前向传播分三个阶段：视觉编码器前向 → 视觉特征融合 → 语言模型逐层前向：

```python
def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
    # 1. 提取校准样本
    sample = inputs[0] if isinstance(inputs, list) else inputs

    # ========== 阶段 1：视觉编码器全量前向 ==========
    pixel_values = sample['pixel_values']
    image_grid_thw = sample['image_grid_thw']

    vision_outputs = yield ProcessRequest(
        name="model.visual",
        module=model.model.visual,
        args=(pixel_values, image_grid_thw),
        kwargs={}
    )
    image_embeds = vision_outputs['pooler_output']

    # ========== 阶段 2：视觉特征融合 ==========
    input_ids = sample['input_ids']
    attention_mask = sample['attention_mask']
    inputs_embeds = model.model.language_model.embed_tokens(input_ids)

    # 使用 masked_scatter 将图像特征替换到文本嵌入中
    image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    image_embeds_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)

    # 构建位置编码和注意力掩码
    cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = self._get_position_ids(model, input_ids, image_grid_thw, attention_mask)
    causal_mask = create_causal_mask(
        config=model.config.text_config, input_embeds=inputs_embeds,
        attention_mask=attention_mask, cache_position=cache_position,
        past_key_values=None, position_ids=position_ids,
    )
    position_embeddings = model.model.language_model.rotary_emb(inputs_embeds, position_ids)

    # ========== 阶段 3：语言模型逐层前向 ==========
    hidden_states = inputs_embeds
    for name, layer in self.generate_decoder_layer(model):
        hidden_states = yield ProcessRequest(
            name=name, module=layer, args=(hidden_states,),
            kwargs={
                'attention_mask': causal_mask,
                'position_ids': position_ids,
                'cache_position': cache_position,
                'position_embeddings': position_embeddings,
                'past_key_values': None,
            },
        )
```

#### 5.6 可选：MoE 3D 权重转换

若模型使用 3D 融合权重存储 MoE 专家（如 Qwen3.5 MoE），需将 3D 权重拆分为标准 Linear 层：

```python
# moe_utils.py
def convert_experts_to_mlp(original_moe_module, config):
    """将 3D 融合 MoE 权重转换为标准 MLP 层"""
    # 1. 解析原始 MoE 模块结构
    # 2. 将 gate_up_proj 拆分为 gate_proj 和 up_proj
    # 3. 将 down_proj 从 3D 权重中提取
    # 4. 返回转换后的标准 MLP 模块
    ...
```

**输出**：适配器接口方法实现完成

### 步骤 6：注册模型到配置

**目标**：在 `config/config.ini` 中注册 VLM 模型名称和适配器入口点。

**操作**：

```ini
[ModelAdapter]
{model_name} = {ModelName1}, {ModelName2}

[ModelAdapterEntryPoints]
{model_name} = msmodelslim.model.{model_name}.loader:{ModelName}AdapterLoader

[ModelAdapterDependencies]
{model_name} = {"transformers": "==x.x.x"}
```

**Qwen3.5 MoE 参考**：

```ini
[ModelAdapter]
qwen3_5_moe = Qwen3.5-397B-A17B, Qwen3.5-27B, Qwen3.5-122B-A10B, Qwen3.5-35B-A3B, Qwen3.6-27B, Qwen3.5-4B

[ModelAdapterEntryPoints]
qwen3_5_moe = msmodelslim.model.qwen3_5_moe.loader:Qwen3_5MoeAdapterLoader

[ModelAdapterDependencies]
qwen3_5_moe = {"transformers": "==5.2.0"}
```

**输出**：模型注册完成

### 步骤 7：创建 YAML 量化配置文件

**目标**：创建 VLM 专用的量化配置文件。

**操作**：

VLM 的 YAML 配置使用 `apiversion: multimodal_vlm_modelslim_v1`，需指定多模态校准数据集：

```yaml
# lab_practice/{model_name}/{model_name}_{quant_type}.yaml
apiversion: multimodal_vlm_modelslim_v1
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
        - "*self_attn*"
        - "*mlp*"
        - "*visual.blocks*"
      exclude:
        - "*linear_fc2*"
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
  dataset: "calibImages"  # 校准图像目录
  default_text: "Describe this image in detail."
```

**Qwen3.6 参考**：`lab_practice/qwen3_5_moe/qwen3_6_dense_w8a8.yaml`

**输出**：量化配置文件创建完成

### 步骤 8：执行量化验证

**目标**：使用量化命令验证 VLM 适配器工作正常。

**操作**：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config ${CONFIG_PATH} \
  --trust_remote_code True
```

**验证要点**：

1. 视觉编码器加载正常，无显存溢出
2. 多模态校准数据加载和预处理正常
3. 视觉特征融合正常（`masked_scatter` 操作）
4. 语言模型逐层量化正常
5. 量化权重正常保存

**输出**：量化验证通过

## 5. 结果与经验

### 5.1 关键结果汇总

| 步骤 | 关键操作 | 指标 | 变化 | 备注 |
| --- | --- | --- | --- | --- |
| 适配器创建 | 继承 VLMBaseModelAdapter + PipelineInterfaceV1 | 代码行数 | 基础适配器约400行 | 含多模态数据处理 |
| 视觉编码器策略 | 全量加载，不逐层拆分 | 显存占用 | 固定视觉编码器显存 | 根据模型大小评估 |
| 语言模型策略 | 逐层加载，按需创建 Decoder Layer | 显存占用 | 随层数线性增长 | 每层处理完可释放 |
| 多模态数据处理 | 使用 AutoProcessor + VlmCalibSample | 数据格式 | 统一多模态输入 | 支持图像目录和 index.jsonl |

### 5.2 经验总结

1. **视觉编码器全量加载**：VLM 的视觉编码器通常包含视觉特征融合逻辑（如 PatchMerger），不适合逐层拆分，应全量加载作为一个整体处理。适用边界：视觉编码器显存占用不超过可用 NPU 显存 50% 的场景。
2. **多模态校准数据格式**：VLM 校准数据使用 `VlmCalibSample` 统一格式（`image` + `text`），`dataset` 字段支持三种配置方式：图像目录、`index.jsonl` 索引文件、图像目录 + `default_text` 组合。
3. **视觉特征融合**：`generate_model_forward` 中需实现视觉特征融合逻辑，通常使用 `masked_scatter` 将图像特征替换到文本嵌入的 `<image>` 占位位置。需理解原始模型 forward 中的融合逻辑，确保一致。
4. **MoE 权重转换**：若模型使用 3D 融合权重存储 MoE 专家（如 `gate_up_proj` 形状为 `[num_experts, 2 * inter_dim, dim]`），需在加载后转换为标准 `nn.Linear` 层，否则量化框架无法逐层处理。

## 6. 异常处理

- **视觉编码器显存不足（OOM）**：降低校准图像分辨率，或减少校准图像数量；考虑使用多卡 DP 模式（`--device npu --device_id 0 1`）。此外，若仍遇到显存不足，请确认 `--device npu --device_id 0` 指定的 NPU 未被其他任务占用。
- **Processor 加载失败**：确认模型路径中包含 `preprocessor_config.json` 或类似的配置文件；对于不使用 Processor 的模型（如 InternVL），需使用 tokenizer 手动预处理。
- **视觉特征融合失败**：检查 `image_token_id` 是否正确；确认 `masked_scatter` 的维度匹配；对于 DeepStack 等特殊结构，需在特定层后注入视觉特征。
- **MoE 权重转换错误**：确认 3D 权重的 shape 和拆分逻辑与原始模型定义一致；检查 `moe_utils.py` 中的转换实现。

## 7. 附录

- 参考实现：[Qwen3.5 MoE 模型适配器](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/qwen3_5_moe/model_adapter.py)
- 参考配置：[Qwen3.6 Dense W8A8 配置](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/qwen3_5_moe/qwen3_6_dense_w8a8.yaml)
- 接口定义：[Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/interface_hub.py)
- 多模态接入指南：[Multimodal Understanding Model Integration Guide (英文)](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/developer_guide/integrating_multimodal_understanding_model.md)
- 校准数据集加载：[VLM Dataset Loader](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/infra/dataset_loader/vlm_dataset_loader.py)
