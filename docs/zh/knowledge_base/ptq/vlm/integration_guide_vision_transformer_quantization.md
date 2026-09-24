# VLM 模型接入量化流程指南

## 1. 适用范围

本指南面向需要将**新的多模态视觉语言模型**（VLM，如 Qwen-VL、InternVL 等新结构）接入 msModelSlim 量化框架的开发者。msModelSlim 采用接口解耦设计，开发者只需通过编写模型适配器（Model Adapter）实现对应接口，即可将自有 VLM 模型打通多模态量化流水线。

适用场景：

- 目标多模态理解模型不在官方支持矩阵中，需要新增 `model_type` 并打通量化链路；
- 已有 `model_type` 需要扩展支持新的量化特性（如离群值平滑、旋转量化或 MoE 混合量化）。

模型是否已在支持列表中请参考[《大模型支持矩阵》](../../model/README.md)；标准多模态理解量化操作请参阅[《VLM 量化使用指南》](usage_vision_transformer_quantization.md)；纯文本 LLM 接入请参阅[《LLM 接入指南》](../llm/integration_guide_large_language_model_quantization.md)；扩散生成模型接入请参阅[《DiT 接入指南》](../dit/integration_guide_diffusion_transformer_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 本地路径或下载仓库 | HuggingFace 格式，含 `config.json` 及分片权重 | 目标 Transformers 版本能正常加载 |
| 输入 | 多模态校准数据集 | 本地路径 | 校准图像目录 + 文本或 `index.jsonl`，推荐 50 条 | 可被适配器 `handle_dataset` 成功解析处理 |
| 交付件 | 模型适配器代码 | `msmodelslim/model/{model_name}/` | Python 源码，含 `loader.py` 与 `model_adapter.py` 等 | 代码审查通过且导入无报错 |
| 交付件 | 模型注册配置 | `config/config.ini` | `[ModelAdapter]`、`[ModelAdapterEntryPoints]` 配置项 | key 保持一致，可通过 `--model_type` 命中 |
| 交付件 | 量化配置文件 | `lab_practice/{model_name}/` | YAML 文件（符合 `multimodal_vlm_modelslim_v1` 规范） | 通过量化配置模式校验 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及分片 | 导出完整且推理冒烟测试通过 |

## 3. 操作步骤

### 步骤 1：创建模型适配器工程目录

**目标**：在框架模型库下建立新多模态模型适配包的目录结构。

**操作**：

在 `msmodelslim/model/` 目录下创建以模型系列命名的专属子目录（如 `qwen3_vl`、`internvl3`）：

```text
msmodelslim/model/{model_name}/
├── __init__.py          # 模块初始化（可为空）
├── loader.py            # 适配器延迟加载器
├── model_adapter.py     # 主适配器类，实现各组件接口
└── moe_utils.py         # （可选）MoE 3D 融合权重转换工具
```

> 常规 VLM 通常仅需 `loader.py` 与 `model_adapter.py` 两个文件。若为 MoE 架构，可增加 `moe_utils.py`；若含 MTP 等自定义结构，可在同级目录下扩展辅助模块。

**输出**：完成适配器目录与源码文件初始化。

### 步骤 2：实现加载器 Loader（延迟加载）

**目标**：为框架插件工厂注册适配器类的类路径，避免启动时预先引入非必需模块。

**操作**：

在 `loader.py` 中继承 [BaseModelAdapterLoader](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/plugin_factory/base_loader.py)，并通过 `ADAPTER_CLASS_PATH` 指定适配器类的导入路径：

```python
# msmodelslim/model/{model_name}/loader.py
from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class {ModelName}AdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.{model_name}.model_adapter:{ModelName}ModelAdapter"
```

**输出**：可被框架工厂动态反射调用的 `loader.py`。

### 步骤 3：定义适配器类并继承组件接口

**目标**：定义模型适配器主类，继承 VLM 基础适配基类与量化算法所需的组件接口。

**操作**：

1. **基础模型类选型**：
   - 推荐继承 [VLMBaseModelAdapter](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/common/vlm_base.py)（已内置 `_load_config`、`_collect_inputs_to_device`、权重分片读取等多模态通用实现）。
   - 若模型结构高度自定义无法复用基类，需继承 [BaseModelAdapter](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/base.py) 并自行实现 config 加载与数据收集逻辑。

2. **核心量化接口**：
   - **[`ModelSlimPipelineInterfaceV1`](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/core/runner/pipeline_interface.py)（必须实现）**：模型流水线核心接口，供调度器驱动多模态量化。

3. **可选算法扩展接口**（均定义于 [Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/interface_hub.py)）：
   - **`IterSmoothInterface` / `FlexSmoothQuantInterface`**：若方案需开启离群值平滑抑制，需继承对应接口并在语言 Decoder 上配置算子映射。
   - **`QuaRotInterface`**：若方案需使用旋转矩阵平滑离群值，需继承此接口（仅作用于语言部分）。

```python
from pathlib import Path

from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
from msmodelslim.utils.logging import logger_setter


@logger_setter("msmodelslim.model.{model_name}")
class {ModelName}ModelAdapter(
    VLMBaseModelAdapter,             # 继承 VLM 基础模型适配类
    ModelSlimPipelineInterfaceV1,    # 必要：量化流水线调度核心接口
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)
```

**输出**：适配器类骨架定义完成。

### 步骤 4：实现流水线核心接口方法

**目标**：实现 [ModelSlimPipelineInterfaceV1](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/core/runner/pipeline_interface.py) 中声明的方法，使调度器（Runner）能够统筹驱动多模态数据处理、加载与逐层量化。

**操作**：

适配器需实现以下核心接口方法，接口签名与职责如下：

```python
def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
    # 加载 Processor 构建多模态 inputs 并通过 self._collect_inputs_to_device 移至设备

def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
    # 实例化模型：视觉编码器全量加载，语言模型仅加载第 1 层以控制显存

def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
    # 拓扑遍历生成器：先 yield "model.visual" 视觉编码器，再遍历语言模型逐层 Decoder

def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
    # 按需逐层实例化并加载语言 Decoder Layer，MoE 架构在此阶段执行 3D 权重转换

def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
    # 校准前向三阶段：视觉编码器前向 -> 视觉特征融合与占位替换 -> 语言 Decoder 逐层前向

def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
    # 控制校准前向是否启用 KVCache，PTQ 校准显式关闭以减少显存开销
```

**实现要点**：

- **视觉全量 + 语言逐层**：VLM 推荐在 `init_model` 中加载完整视觉编码器，但语言模型仅初始化第 1 层（临时设 `num_hidden_layers=1`），其余层通过 `generate_decoder_layer` 在前向过程中按需加载并释放，避免大模型多模态量化时显存溢出。
- **特征融合对齐**：`generate_model_forward` 中的视觉特征提取与融合（如 `masked_scatter` 替换 `<image>` 占位、DeepStack 跨层注入等）必须与原模型官方前向逻辑完全一致。
- **MoE 权重转换**：若模型使用 3D 融合权重存储 MoE 专家，需在加载 Decoder 时转换为标准 Linear 层，以便量化框架逐层处理。

**输出**：完成全部接口实现的适配器代码。

### 步骤 5：在全局配置文件中注册模型

**目标**：在 `config/config.ini` 中注册模型名称和加载器入口，使 CLI 命令行能够通过 `--model_type` 正确路由。

**操作**：

在 `config/config.ini` 中添加配置项（注意：三个段落中的 key 名称必须保持严格一致）：

```ini
[ModelAdapter]
# 将用户常用的模型名称映射到统一 key
{model_alias_key} = {ModelName1}, {ModelName2}

[ModelAdapterEntryPoints]
# 指定对应的加载器类路径
{model_alias_key} = msmodelslim.model.{model_name}.loader:{ModelName}AdapterLoader

[ModelAdapterDependencies]
# （可选）声明模型特定的三方库版本依赖约束
{model_alias_key} = {"transformers": "==4.48.2"}
```

**输出**：完成配置注册，可通过 `--model_type {ModelName}` 调用。

### 步骤 6：编写量化 YAML 方案并执行验证

**目标**：创建适配该 VLM 模型的量化方案 YAML，并通过 CLI 运行端到端量化测试。

**操作**：

1. **编写 YAML 量化方案**（以标准的 W8A8 静态量化为例，视觉投影层需排除）：

   ```yaml
   # lab_practice/{model_name}/{model_name}_w8a8.yaml
   apiversion: multimodal_vlm_modelslim_v1
   spec:
     runner: auto
     process:
       - type: linear_quant
         qconfig:
           act:
             scope: per_tensor
             dtype: int8
             symmetric: false
             method: minmax
           weight:
             scope: per_channel
             dtype: int8
             symmetric: true
             method: minmax
         include: ["*"]
         exclude:
           - "*merger*"                 # 视觉特征投影层排除量化
           - "*linear_fc2"
           - "*deepstack_merger_list*"
           - "*mlp.gate"                # MoE router 排除量化
     save:
       - type: ascendv1_saver
         part_file_size: 4
     dataset: calib_images              # 校准图像目录或 index.jsonl
     default_text: "Describe this image in detail."
   ```

2. **执行 CLI 量化验证**：

   ```bash
   msmodelslim quant \
     --model_path <浮点模型目录> \
     --save_path <量化权重输出目录> \
     --model_type <注册的模型适配器名称> \
     --config <量化配置文件路径> \
     --device npu \
     --device_id 0
   ```

**输出**：量化命令执行成功，输出目录包含 `quant_model_description.json` 与分片权重 `*.safetensors`。

## 4. 案例列表

| 案例 | 简述 | 链接 |
| --- | --- | --- |
| Kimi-K3 新模型接入与量化 | 多模态理解模型新结构接入与 W4A8 量化端到端实践 | [《Kimi-K3 新模型 W4A8 量化案例》](../../../best_practices/kimi_k3_adaptation_case.md) |

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| PTQ | 训练后量化（Post-Training Quantization） | [《训练后量化词条》](../term_ptq.md) |
| VLM 量化 | 多模态视觉语言模型训练后量化 | [《多模态理解模型（VLM）量化词条》](./term_vision_transformer_quantization.md) |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| Interface Hub | 量化机制与算法组件对模型所需接口的集中定义与汇总 | [《Interface Hub》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/interface_hub.py) |
| VLM 量化使用指南 | 标准多模态理解模型量化全流程、配置协议与开箱即用示例 | [《VLM 量化使用指南》](usage_vision_transformer_quantization.md) |
| multimodal_vlm_modelslim_v1 配置说明 | `process`/`save`/`dataset` 等多模态理解配置的字段说明 | [《multimodal_vlm_modelslim_v1 配置说明》](../../../api_reference/config/quant/multimodal_vlm_modelslim_v1.md) |
| 一键量化完整指南 | 一键量化 CLI 入口与完整命令参数说明 | [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md) |
| 量化算法总览 | 离群值平滑抑制与各类量化算法详述 | [《量化算法总览》](../../quantization_algorithms/README.md) |
