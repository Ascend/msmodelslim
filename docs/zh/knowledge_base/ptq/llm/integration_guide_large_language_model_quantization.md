# LLM 模型接入量化流程指南

## 1. 适用范围

本指南面向需要将**新的 Decoder-only 大语言模型**（如 LLaMA 变种、Qwen 变种、DeepSeek 等新结构）接入 msModelSlim 量化框架的开发者。msModelSlim 采用接口解耦设计，将量化机制与算法所需的模型能力抽取为通用接口。开发者只需通过编写模型适配器（Model Adapter）实现对应接口，即可将自有模型打通量化流水线。

适用场景：

- 目标 LLM 不在官方支持矩阵中，需要新增 `model_type` 并打通量化链路；
- 已有 `model_type` 需要扩展支持新的量化特性（如离群值平滑、旋转量化或敏感层调优）。

模型是否已在支持列表中请参考[《大模型支持矩阵》](../../model/README.md)；标准模型量化操作请参阅[《LLM 量化使用指南》](usage_large_language_model_quantization.md)；多模态模型接入请参阅[《VLM 接入指南》](../vlm/integration_guide_vision_transformer_quantization.md)或[《DiT 接入指南》](../dit/integration_guide_diffusion_transformer_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 本地路径或下载仓库 | HuggingFace 格式，含 `config.json` 及分片权重 | 目标 Transformers 版本能正常加载 |
| 输入 | 校准数据集 | 本地路径或 `lab_calib/` | JSONL 或 JSON 文本 Prompt，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为张量 |
| 交付件 | 模型适配器代码 | `msmodelslim/model/{model_name}/` | Python 源码，含 `loader.py` 与 `model_adapter.py` | 代码审查通过且导入无报错 |
| 交付件 | 模型注册配置 | `config/config.ini` | `[ModelAdapter]`、`[ModelAdapterEntryPoints]` 配置项 | key 保持一致，可通过 `--model_type` 命中 |
| 交付件 | 量化配置文件 | `lab_practice/{model_name}/` | YAML 文件（符合 `modelslim_v1` 规范） | 通过量化配置模式校验 |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及分片 | 导出完整且推理冒烟测试通过 |

## 3. 操作步骤

### 步骤 1：创建模型适配器工程目录

**目标**：在框架模型库下建立新模型适配包的目录结构。

**操作**：

在 `msmodelslim/model/` 目录下创建以模型系列命名的专属子目录（如 `qwen3`、`deepseek_v4`）：

```text
msmodelslim/model/{model_name}/
├── __init__.py          # 模块初始化（可为空）
├── loader.py            # 适配器延迟加载器
└── model_adapter.py     # 主适配器类，实现各组件接口
```

> 针对基于 HuggingFace Transformers 开发的标准开源模型，通常仅需创建 `loader.py` 与 `model_adapter.py` 两个文件。若模型包含特殊结构（如 FP8 权重反量化脚本、MTP 自定义层），可在同级目录下扩展辅助模块。

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

**目标**：定义模型适配器主类，组合继承基础适配基类与量化算法所需的组件接口。

**操作**：

1. **基础模型类选型**：
   - 若模型基于 HuggingFace Transformers 实现，推荐继承 [TransformersModel](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/common/transformers.py)（已内置分词批处理、基础模型加载与 KVCache 控制的通用实现）。
   - 若模型结构高度自定义，无法基于 Transformers 加载，需直接继承 [BaseModelAdapter](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/base.py)。

2. **核心量化接口**：
   - **[`ModelSlimPipelineInterfaceV1`](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/core/runner/pipeline_interface.py)（必须实现）**：模型流水线核心接口（等价于 `PipelineInterface`），供调度器驱动量化。

3. **可选算法扩展接口**（均定义于 [Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/interface_hub.py)）：
   - **`IterSmoothInterface` / `FlexSmoothQuantInterface`**：若量化方案需开启离群值平滑抑制（如 IterSmooth / FlexSmoothQuant），需继承此接口并声明网络中可平滑的 Norm-Linear、Up-Down 子图算子对。
   - **`QuaRotInterface`**：若方案需使用旋转矩阵平滑离群值，需继承此接口。
   - **`StandingHighWithExperienceInterface`**：使用自动精度调优中的“专家经验摸高策略”时需继承。

```python
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
from msmodelslim.utils.logging import logger_setter


@logger_setter("msmodelslim.model.{model_name}")
class {ModelName}ModelAdapter(
    TransformersModel,               # 继承基础模型适配类
    ModelSlimPipelineInterfaceV1,    # 必要：量化流水线调度核心接口
):
    pass
```

**输出**：适配器类骨架定义完成。

### 步骤 4：实现流水线核心接口方法

**目标**：实现 [ModelSlimPipelineInterfaceV1](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/core/runner/pipeline_interface.py) 中声明的方法，使调度器（Runner）能够统筹驱动模型的数据预处理、加载与逐层量化。

**操作**：

适配器需实现以下 5 个核心接口方法，接口签名如下（标准模型继承 `TransformersModel` 后可直接复用基类默认实现）：

```python
def handle_dataset(self, dataset, device=DeviceType.NPU) -> List[Any]
    # 将原始文本 Prompt 列表分词截断并转为张量，返回 [ {"input_ids": tensor, ...} ]
    # 标准模型直接复用 self._get_tokenized_data(dataset, device)

def init_model(self, device=DeviceType.NPU) -> nn.Module
    # 实例化模型并加载权重，返回模型实例
    # 标准模型直接调用 self._load_model(device)

def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]
    # 网络结构遍历生成器（Data-Free 量化），按拓扑顺序逐层 yield ProcessRequest
    # 标准 Decoder 模型直接调用 generated_decoder_layer_visit_func(model)

def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]
    # 校准前向生成器，单条样本逐层前向以捕获激活分布，逐层 yield 并向下传递
    # 标准模型直接调用 transformers_generated_forward_func(model, inputs)

def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None
    # 控制校准前向是否启用 KVCache
    # 标准模型直接复用 self._enable_kv_cache(model, need_kv_cache)
```

**实现要点**：

- **逐层加载控显存**：大参数量模型推荐在 `init_model` 中仅加载第一层 Decoder Layer（临时设置 `num_hidden_layers=1`），其余层在量化前向时按需加载；若源模型为 FP8 权重，需在此阶段执行反量化。
- **层序严格一致**：`generate_model_visit` 与 `generate_model_forward` 的层访问顺序必须完全严格一致，否则逐层调度会错位。
- **禁用 KVCache**：PTQ 校准仅需 Prefill 特征，`enable_kv_cache` 中显式关闭可大幅减少显存开销。

**输出**：完成全部接口契约实现的适配器代码。

### 步骤 5：在全局配置文件中注册模型

**目标**：在 `config/config.ini` 中注册模型名称和加载器入口，使 CLI 命令行能够通过 `--model_type` 正确路由。

**操作**：

在 `config/config.ini` 中添加配置项（注意：三个段落中的 key 名称必须保持严格一致）：

```ini
[ModelAdapter]
# 将用户常用的模型名称（支持别名列表）映射到统一 key
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

**目标**：创建适配该模型的量化方案 YAML，并通过 CLI 运行端到端量化测试。

**操作**：

1. **编写 YAML 量化方案**（以标准的 W8A8 动态量化为例）：

   ```yaml
   # lab_practice/{model_name}/{model_name}_w8a8_dynamic.yaml
   apiversion: modelslim_v1
   spec:
     runner: auto                    # 单卡自动使用 layer_wise，多卡自动使用 dp_layer_wise
     process:
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
         exclude: ["*gate*"]          # 排除 router 等敏感层
     save:
       - type: ascendv1_saver        # 昇腾推理部署推荐格式
     dataset: mix_calib.jsonl        # 通用校准集
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
| DeepSeek-V4-Pro 新模型接入与量化 | 复杂 MoE+MLA 架构大模型 W4A8 混合量化端到端接入实践 | [《DeepSeek-V4-Pro 新模型 W4A8 量化案例》](../../../best_practices/deepseek_v4_pro_w4a8_new_llm_quantization_case.md) |
| Qwen3-32B 精度调优实战 | 典型开源大模型 W8A8 动态量化与敏感层自动调优案例 | [《Qwen3-32B 模型 W8A8 量化调优案例》](../../../best_practices/qwen3_32b_w8a8_precision_tuning_case.md) |

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| PTQ | 训练后量化（Post-Training Quantization） | [《训练后量化词条》](../term_ptq.md) |
| LLM 量化 | Decoder-only 大语言模型训练后量化 | [《LLM 量化词条》](./term_large_language_model_quantization.md) |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| Interface Hub | 量化机制与算法组件对模型所需接口的集中定义与汇总 | [《Interface Hub》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/interface_hub.py) |
| LLM 量化使用指南 | 标准 LLM 量化全流程、配置协议与开箱即用示例 | [《LLM 量化使用指南》](usage_large_language_model_quantization.md) |
| modelslim_v1 配置说明 | `runner`/`process`/`save`/`dataset` 等任务级配置的字段说明 | [《modelslim_v1 配置说明》](../../../api_reference/config/quant/modelslim_v1.md) |
| 一键量化完整指南 | 一键量化 CLI 入口与完整命令参数说明 | [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md) |
| 自动调优使用说明 | 敏感层分析、回退候选及调优策略说明 | [《自动调优使用说明》](../../../user_guide/usage_auto_precision_tuning.md) |
| 量化算法总览 | 离群值平滑抑制与各类量化算法详述 | [《量化算法总览》](../../quantization_algorithms/README.md) |
