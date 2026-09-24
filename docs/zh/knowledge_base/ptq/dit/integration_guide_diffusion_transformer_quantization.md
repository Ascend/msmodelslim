# DiT 模型接入量化流程指南

## 1. 适用范围

本指南面向需要将**新的多模态生成模型**（DiT，Diffusion Transformer，如 FLUX.1、HunyuanVideo、Wan2.1、Wan2.2 等新结构）接入 msModelSlim 量化框架的开发者。msModelSlim 采用接口解耦设计，开发者只需通过编写模型适配器（Model Adapter）实现多模态生成流水线接口，即可打通浮点推理 dump 校准数据与逐层量化调度链路。

适用场景：

- 目标多模态生成模型不在官方支持矩阵中，需要新增 `model_type` 并打通量化链路；
- 已有 `model_type` 需要扩展支持新的量化模式（如从 W8A8 MXFP8 扩展至 W4A4 MXFP4、双专家差异化量化等）。

模型是否已在支持列表中请参考[《大模型支持矩阵》](../../model/README.md)；标准 DiT 量化操作请参阅[《DiT 量化使用指南》](usage_diffusion_transformer_quantization.md)；纯文本 LLM 接入请参阅[《LLM 接入指南》](../llm/integration_guide_large_language_model_quantization.md)；多模态理解模型接入请参阅[《VLM 接入指南》](../vlm/integration_guide_vision_transformer_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点 DiT 权重目录 | 本地路径或下载仓库 | 含模型权重文件及配置文件 | 可被官方推理管线加载 |
| 输入 | 官方推理仓库 | 本地路径，加入 `PYTHONPATH` | 可执行浮点去噪推理 | 官方推理示例可正常运行 |
| 输入 | 校准数据集与任务配置 | YAML 中 `dataset` 与 `inference_config` | 指定任务类型及推理尺寸、步数，推荐 50 条 | 适配器成功解析并完成浮点推理 dump |
| 交付件 | 模型适配器代码 | `msmodelslim/model/{model_name}/` | Python 源码，含 loader、model_adapter 等 | 代码审查通过且导入无报错 |
| 交付件 | 模型注册配置 | `config/config.ini` | `[ModelAdapter]`、`[ModelAdapterEntryPoints]` 等段落 | key 保持一致，可通过 `--model_type` 命中 |
| 交付件 | 量化配置文件 | `lab_practice/{model_name}/` | YAML 文件（符合 `multimodal_sd_modelslim_v1` 规范） | 通过量化配置模式校验 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | MindIE-SD 格式，多专家含各专家子目录 | 导出完整且推理冒烟测试通过 |

## 3. 操作步骤

### 步骤 1：创建模型适配器工程目录

**目标**：在框架模型库下建立新 DiT 模型适配包的目录结构。

**操作**：

在 `msmodelslim/model/` 目录下创建以模型系列命名的专属子目录（如 `wan2_2`、`flux`）：

```text
msmodelslim/model/{model_name}/
├── __init__.py              # 模块初始化（可为空）
├── loader.py                # 适配器延迟加载器
├── model_adapter.py         # 主适配器类
├── base_model_adapter.py    # （可选）多专家模型的基础适配器
├── expert_sub_adapter.py    # （可选）专家子适配器（多专家必须）
└── {task}/                  # （可选）按生成任务拆分的子模块（如 t2v/i2v）
```

> 单网络 DiT 通常仅需 `loader.py` 与 `model_adapter.py`；多专家模型（如 Wan2.2 双专家）需增加 `base_model_adapter.py` 与 `expert_sub_adapter.py`。

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

**目标**：定义模型适配器主类，继承基础适配基类与多模态生成量化接口。

**操作**：

1. **基础适配基类**：DiT 依赖官方推理管线加载，需继承 [BaseModelAdapter](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/base.py)（不继承 `TransformersModel`）。
2. **核心流水线接口**：必须继承 [MultimodalPipelineInterface](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/core/quant_service/multimodal_sd_v1/pipeline_interface.py)（重构路径）。
3. **可选算法扩展接口**：根据量化算法需求继承 `OnlineQuaRotInterface`、`FA3QuantAdapterInterface`、`IterSmoothInterface` 等（均定义于 [Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/interface_hub.py)）。

```python
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.interface_hub import ModelInfoInterface
from msmodelslim.core.quant_service.multimodal_sd_v1.pipeline_interface import (
    MultimodalPipelineInterface,
)
from msmodelslim.utils.logging import logger_setter


@logger_setter("msmodelslim.model.{model_name}")
class {ModelName}ModelAdapter(
    BaseModelAdapter,                    # 基础适配基类
    ModelInfoInterface,                  # 模型信息接口
    MultimodalPipelineInterface,         # 必要：多模态生成流水线核心接口
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.pipeline = None
        self.transformer = None
        self.model_args = None
```

**输出**：适配器类骨架定义完成。

### 步骤 4：实现流水线核心接口方法

**目标**：实现 [MultimodalPipelineInterface](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/core/quant_service/multimodal_sd_v1/pipeline_interface.py) 中声明的方法，支持推理参数校验、运行时配置、浮点推理校准 dump 及逐块调度。

**操作**：

适配器需实现以下核心接口方法，接口签名与职责如下：

```python
def get_inference_config_class(self) -> Type[BaseModel]:
    # 返回推理参数 Pydantic 配置类（定义 size/frame_num/sample_steps 等字段及校验规则）

def configure_runtime(self, inference_config: Any) -> None:
    # 将校验通过的推理参数设置到官方推理管线的 model_args

def inference_dump_calib_data(self, dataset: Any = None, inference_config: Any = None):
    # enable_dump 时运行完整浮点去噪推理，捕获各 Timestep 激活并落盘为校准数据

def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[VlmCalibSample]:
    # 校验并标准化校准数据集样本

def init_model(self, device: DeviceType = DeviceType.NPU) -> Dict[str, nn.Module]:
    # 加载推理管线并返回待量化的模型字典（单网络返回 {'': transformer}，双专家返回各专家子模型）

def prepare_calib_data(self, models, dump_config, save_path, dataset, inference_config) -> Dict[str, Any]:
    # 校准数据准备：enable_dump 时执行浮点推理将激活保存为 pth 缓存，否则直接读取已有缓存

def quantization_context(self) -> AbstractContextManager:
    # 提供量化时的 autocast、no_grad 及设备上下文环境

def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
    # 通过关键字（如 attentionblock）定位 Attention Block 并逐块遍历

def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
    # 逐块前向生成器：通过 pre-hook 截获首个 Block 输入后逐块 yield 驱动前向

def get_expert_adapter(self, expert_name: str) -> PipelineInterface:
    # 多专家模型返回对应的专家子适配器，单专家模型返回 self

def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
    # DiT 为密集前向网络无 KV Cache，通常实现为空操作以满足接口契约
```

**实现要点**：

- **浮点推理 Dump 校准**：DiT 量化校准依赖运行完整浮点去噪推理管线捕获激活特征，`inference_dump_calib_data` 须固定生成随机数种子保证可复现性。
- **逐块调度（LayerWise）**：DiT 模型在 `multimodal_sd_modelslim_v1` 中固定使用 `runner: layer_wise`，通过 hook 截获 Attention Block 输入逐块量化。
- **多专家模型显式继承**：多专家模型（如 Wan2.2 双专家）的专家子适配器必须**显式继承**主适配器所实现的扩展接口（如 `OnlineQuaRotInterface`、`IterSmoothInterface`），不能仅靠主适配器继承和 `__getattr__` 代理。

**输出**：完成全部接口契约实现的适配器代码。

### 步骤 5：在全局配置文件中注册模型

**目标**：在 `config/config.ini` 中注册模型名称和加载器入口，使 CLI 命令行能够通过 `--model_type` 正确路由。

**操作**：

在 `config/config.ini` 中添加配置项（推荐按场景拆分为独立 `model_type`）：

```ini
[ModelAdapter]
# 模型名称映射（支持别名）
{model_name} = {ModelName1}, {ModelName2}
{model_name}_t2v = {ModelName}-T2V-{Size}
{model_name}_i2v = {ModelName}-I2V-{Size}

[ModelAdapterEntryPoints]
# 指定加载器类路径
{model_name} = msmodelslim.model.{model_name}.loader:{ModelName}AdapterLoader
{model_name}_t2v = msmodelslim.model.{model_name}.t2v.loader:{ModelName}T2VAdapterLoader
{model_name}_i2v = msmodelslim.model.{model_name}.i2v.loader:{ModelName}I2VAdapterLoader
```

**输出**：完成配置注册，可通过 `--model_type {ModelName}` 调用。

### 步骤 6：编写量化 YAML 方案并执行验证

**目标**：创建适配该 DiT 模型的量化方案 YAML，并通过 CLI 运行端到端量化测试。

**操作**：

1. **编写 YAML 量化方案**（以标准的 W8A8 MXFP8 静态量化为例）：

   ```yaml
   # lab_practice/{model_name}/{model_name}_w8a8_mxfp8.yaml
   apiversion: multimodal_sd_modelslim_v1
   spec:
     runner: layer_wise                  # DiT 固定使用 layer_wise
     process:
       - type: linear_quant
         qconfig:
           act:
             scope: per_block
             dtype: mxfp8
             symmetric: true
             method: minmax
           weight:
             scope: per_block
             dtype: mxfp8
             symmetric: true
             method: mse_round
         include: ["*"]
     save:
       - type: mindie_format_saver       # 昇腾 MindIE-SD 部署格式
     dataset: {model_name}_{task}
     multimodal_sd_config:
       dump_config:
         enable_dump: true               # 首次量化开启浮点 dump
         capture_mode: "args"
         dump_data_dir: ""
       inference_config:
         size: "1280*720"
         frame_num: 81
         sample_steps: 40
         task: "{task}"
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

**输出**：量化命令执行成功，输出目录包含 MindIE-SD 权重文件（多专家模型包含各专家独立子目录）。

## 4. 案例列表

| 案例 | 简述 | 链接 |
| --- | --- | --- |
| Wan2.2 新模型量化案例 | 双专家 DiT 文生视频模型 W4A4F4 低比特量化端到端实践 | [《Wan2.2 新模型 W4A4F4 量化案例》](../../../best_practices/wan2_2_w4a4f4_new_dit_quantization_case.md) |

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| PTQ | 训练后量化（Post-Training Quantization） | [《训练后量化词条》](../term_ptq.md) |
| DiT 量化 | Diffusion Transformer 扩散生成模型训练后量化 | [《多模态生成模型（DiT）量化词条》](./term_diffusion_transformer_quantization.md) |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| Interface Hub | 量化机制与算法组件对模型所需接口的集中定义与汇总 | [《Interface Hub》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/interface_hub.py) |
| Pipeline 接口 | 多模态生成量化服务的流水线接口定义 | [《pipeline_interface.py》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/core/quant_service/multimodal_sd_v1/pipeline_interface.py) |
| DiT 量化使用指南 | 标准 DiT 模型量化全流程、配置协议与开箱即用示例 | [《DiT 量化使用指南》](usage_diffusion_transformer_quantization.md) |
| multimodal_sd_modelslim_v1 配置说明 | `process`/`save`/`multimodal_sd_config` 等多模态生成配置字段说明 | [《multimodal_sd_modelslim_v1 配置说明》](../../../api_reference/config/task/multimodal_sd_modelslim_v1.md) |
| mindie_format_saver 配置说明 | MindIE-SD 保存格式与多专家落盘说明 | [《mindie_format_saver 配置说明》](../../../api_reference/config/format/mindie_format_saver.md) |
| 一键量化完整指南 | 一键量化 CLI 入口与完整命令参数说明 | [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md) |
| 量化算法总览 | 微缩浮点、在线旋转与各类量化算法详述 | [《量化算法总览》](../../quantization_algorithms/README.md) |
