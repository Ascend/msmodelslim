# LLM大模型接入指南

## 简介

本文档面向需要将自有模型接入 msModelSlim 的开发者。
msModelSlim认识到量化机制和算法都有适用范围和局限性，而新的模型结构层出不穷，不存在一劳永逸的模型量化方法。
为了尽可能简化自有模型量化流程，msModelSlim将量化机制和算法生效所依赖的模型条件抽取出来，以接口形式描述。
模型则以模型适配器描述，模型适配器是接口实现的组合，通过实现各式各样的机制和算法等组件的接口打通自有模型的量化过程。

## 概念说明

### 接口

* 接口在量化机制和算法等组件中定义，描述了对应组件对模型的诉求，接口定义和使用请参考对应组件的文档和代码
* 仅当使用到对应组件时才需要实现对应接口
* 接口汇总：[`msmodelslim/model/interface_hub.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/interface_hub.py)

### 模型适配器

* 模型适配器是组件接口实现的组合，描述了模型特性和行为，这些特性和行为服务于具体的组件接口实现
* 结构相似的一组模型可复用一个模型适配器，一个模型适配器可注册多个模型
* `msmodelslim`命令中常见参数`model_type`即对应模型适配器注册的模型名，用于匹配和创建模型适配器

## 模型接入

以下内容将以 [`Qwen3-32B`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/qwen3/model_adapter.py) W8A8动态量化场景（简称“场景示例”）的模型接入为例：

### 新建模型适配器`py`文件

建议放在[`msmodelslim/model/`](https://gitcode.com/Ascend/msmodelslim/tree/master/msmodelslim/model) 下，命名如 `qwen3.py`。

### 理清量化过程涉及的组件，以组件接口组合定义适配器类

模型适配器类必须继承自[`BaseModelAdapter`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/base.py)。

根据经验，W8A8动态量化的精度损失很小，无需搭配离群值抑制算法，也很少需要回退；因此，在场景示例中，我们仅需支持量化调度，无需支持离群值量化、敏感层分析等额外功能。需要接入其他算法可以参考[`算法总览`](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/)

```python
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.utils.logging import logger_setter


@logger_setter()
class Qwen3ModelAdapter(TransformersModel,  # 继承自BaseModelAdapter，基于Transformers模型通用特性和行为简化接口实现
                        ModelSlimPipelineInterfaceV1,  # 必要，服务于量化调度
                        ):
    pass
```

### 实现组件接口方法

实现接口所需的方法，方法描述模型特性和行为，若使用IDE，可通过IDE功能快速创建接口方法，再填入功能代码。

```python
from typing import List, Any, Generator
from torch import nn
from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func, \
    transformers_generated_forward_func
from msmodelslim.utils.logging import logger_setter


@logger_setter()
class Qwen3ModelAdapter(TransformersModel,
                        ModelSlimPipelineInterfaceV1
                        ):
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:  # 描述校准集转化为批量输入
        return self._get_tokenized_data(dataset, device)  # TransformersModel已基于Transformers模型特点给出默认实现

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:  # 描述如何初始化模型
        return self._load_model(device)  # TransformersModel已基于Transformers模型特点给出默认实现

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:  # 描述如何将模型分段，必须与模型结构匹配
        # msmodelslim/model/common/layer_wise_forward.py已给出基于DecoderLayer类分段的默认实现
        yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(self, model: nn.Module, inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:  # 描述如何将模型前向过程分段，必须与模型前向过程匹配
        # msmodelslim/model/common/layer_wise_forward.py已给出基于DecoderLayer类前向过程的默认实现
        yield from transformers_generated_forward_func(model,
                                                       inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:  # 描述是否禁用 KVCache，可减少显存
        return self._enable_kv_cache(model, need_kv_cache)  # TransformersModel已基于Transformers模型特点给出默认实现
```

### 注册模型

在配置文件 [`config.ini`](https://gitcode.com/Ascend/msmodelslim/blob/master/config/config.ini) 中注册模型名称，便于同一系列的模型复用一个模型适配器。

```ini
# 在ModelAdapter中的qwen3系列注册Qwen3-32B模型，qwen3对应下面的Qwen3ModelAdapter模型适配器
[ModelAdapter]
default = default
deepseek_v3 = DeepSeek-V3, DeepSeek-V3-0324, DeepSeek-R1, DeepSeek-R1-0528, DeepSeek-V3.1
deepseek_v3_2 = DeepSeek-V3.2-Exp
qwen2_5 = Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct, Qwen2.5-72B-Instruct, Qwen2.5-Coder-7B-Instruct
qwen3 = Qwen3-8B, Qwen3-14B, Qwen3-32B  # 此处添加
qwen3_moe = Qwen3-30B, Qwen3-235B
qwq = Qwen-QwQ-32B, QwQ-32B
wan2_1 = Wan2_1, Wan2.1
qwen3_next = Qwen3-Next-80B-A3B-Instruct
wan2_2 = Wan2_2, Wan2.2

# 如果添加新模型适配器，需要在ModelAdapterEntryPoints中添加，注意ModelAdapter与ModelAdapterEntryPoints中的key需要保持一致，不然无法生效
[ModelAdapterEntryPoints]
default = msmodelslim.model.default.model_adapter:DefaultModelAdapter
deepseek_v3 = msmodelslim.model.deepseek_v3.model_adapter:DeepSeekV3ModelAdapter
deepseek_v3_2 = msmodelslim.model.deepseek_v3_2.model_adapter:DeepSeekV32ModelAdapter
qwen2_5 = msmodelslim.model.qwen2_5.model_adapter:Qwen25ModelAdapter
qwen3 = msmodelslim.model.qwen3.model_adapter:Qwen3ModelAdapter
qwen3_moe = msmodelslim.model.qwen3_moe.model_adapter:Qwen3MoeModelAdapter
qwq = msmodelslim.model.qwq.model_adapter:QwqModelAdapter
wan2_1 = msmodelslim.model.wan2_1.model_adapter:Wan2Point1Adapter
qwen3_next = msmodelslim.model.qwen3_next.model_adapter:Qwen3NextModelAdapter
wan2_2 = msmodelslim.model.wan2_2.model_adapter:Wan2Point2Adapter
```

## 自动调优与敏感层分析

使用 [自动调优](../feature_guide/auto_precision_tuning/usage.md) 且策略需**自动生成回退候选**时（`standing_high` 始终；`binary_fallback` 在未配置非空 `rollback_candidates` 时；`standing_high_with_experience` 委托 Standing High 执行），模型适配器须实现 **`ModelSlimPipelineInterfaceV1`**：

```python
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
```

该接口与 CLI **`msmodelslim analyze`** 及 `PipelineAnalysisService` 的模型协议相同，需实现 `init_model`、`handle_dataset`、`generate_model_visit`、`generate_model_forward` 等方法。调优策略内由 `PipelineAnalysisService` 调用上述方法，**不会在策略侧预先 `load_model`**。

| 调优策略 | 敏感层分析 | 额外接口 |
|----------|------------|----------|
| `standing_high` | 始终自动运行 | 无 |
| `binary_fallback` | 配置了非空 `rollback_candidates` 时跳过 | 无 |
| `standing_high_with_experience` | 委托 Standing High | **`StandingHighWithExperienceInterface`**（`load_model`，离群值抑制能力探测）；**`ModelSlimPipelineInterfaceV1` 须单独继承** |

详见 [自动调优配置协议说明](../feature_guide/auto_precision_tuning/configuration_protocols.md) 及各策略算法文档。

## 量化自有模型

当完成模型适配器的编写与注册后，即可使用一键量化能力对自有模型进行量化。

### 创建W8A8动态量化Yaml配置文件

```yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "linear_quant" # 线性层量化
      qconfig:
        act: # 激活值量化
          scope: "per_token" # 动态量化
          dtype: "int8" # 8比特整数量化
          symmetric: True # 对称量化
          method: "minmax" # 使用minmax算法
        weight: # 权重量化
          scope: "per_channel" # per_channel量化
          dtype: "int8" # 8比特整数量化
          symmetric: True # 对称量化
          method: "minmax" # 使用minmax算法
      include: [ "*" ] # 全局w8a8动态量化
      exclude: [ "*down_proj*" ] # 回退down_proj层

  save:
    - type: "ascendv1_saver"
      part_file_size: 4 # 每个safetensors权重文件最大4G
```

### 量化自有模型

可通过如下命令完成自有模型量化，请注意`trust_remote_code`为`True`时可能执行浮点模型权重中的代码文件，请确保浮点模型来源安全可靠。其中 `${MODEL_PATH}` 为原始浮点权重路径，`${SAVE_PATH}` 为用户自定义的量化权重保存路径，`${MODEL_TYPE}` 为注册的模型名称，`${CONFIG_PATH}` 为YAML配置文件路径。

```bash
msmodelslim quant --model_path ${MODEL_PATH} \
                  --save_path ${SAVE_PATH} \
                  --device npu \
                  --model_type ${MODEL_TYPE} \
                  --config_path ${CONFIG_PATH} \
                  --trust_remote_code False
```

- 详细用法与参数说明请参阅：[`一键量化使用说明`](../feature_guide/quick_quantization_v1/usage.md)
