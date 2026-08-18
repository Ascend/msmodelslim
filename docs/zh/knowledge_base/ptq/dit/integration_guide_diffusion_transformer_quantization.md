# DiT 模型接入案例——以 Wan2.2 为例

## 1. 适用范围

本文档面向需要将**多模态生成模型（DiT）**接入 msModelSlim 量化流程的开发人员。通过分析 Wan2.2 适配器的完整实现，展示 DiT 模型接入的特殊流程。

**目标**：掌握 DiT 模型接入 msModelSlim 的代码开发规范，理解多模态生成模型与 LLM/VLM 的差异点，包括推理管线加载、校准数据 dump、多专家量化等。

**覆盖流程**：模型适配器创建 → 推理管线加载 → 组件接口实现 → 配置注册 → YAML 量化配置 → 量化验证

**关联流程**：[DiT 量化使用指南](usage_diffusion_transformer_quantization.md)

## 2. 环境与版本

| 项 | 版本或配置 |
| --- | --- |
| 产品形态 | Atlas 800I A2 / Ascend 950（不限定） |
| CANN | 参考安装指南 |
| PyTorch | >= 2.7.1 (推荐使用镜像配置2.10.0) |
| MindIE-SD | 参考安装指南 |
| 其他依赖 | wan（Wan2.2 官方推理仓库）、diffusers、safetensors |

**本次前置事实**：

- msModelSlim 工具已安装并可正常执行 `msmodelslim --help`
- 目标 DiT 模型权重已下载到本地
- Wan2.2 官方推理仓库已安装并加入 PYTHONPATH，可正常执行浮点推理

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点 DiT 权重 | `${MODEL_PATH}` | 含模型权重文件及配置文件 | 可被推理管线加载 |
| 输入 | 推理配置 | YAML `multimodal_sd_config.inference_config` | 含 size/frame_num/sample_steps/task 等 | 与 model_type 匹配 |
| 输入 | 校准样本 | `dataset` 字段指定 | 由适配器处理 | 至少 1条样本 |
| 交付件 | 模型适配器代码 | `msmodelslim/model/{model_name}/` | Python 源码，含 loader、adapter | 代码审查通过 |
| 交付件 | 量化配置 | `lab_practice/{model_name}/` | YAML 格式（`apiversion: multimodal_sd_modelslim_v1`） | 配置校验通过 |
| 交付件 | 量化权重 | `${SAVE_PATH}` | MindIE-SD 格式，多专家含各专家子目录 | 推理冒烟通过 |

## 4. 操作步骤

### 步骤 1：分析 DiT 模型结构，确定适配策略

**目标**：分析目标 DiT 模型的结构特点，确定多模态生成量化的接入策略。

**操作**：

1. **分析模型架构**：DiT（Diffusion Transformer）模型通过去噪过程生成图像或视频，通常包含：
   - **主干 Transformer**：由多个 Attention Block 组成，是量化的核心对象
   - **辅助模块**：VAE、Text Encoder（T5/CLIP）、扩散调度器等
   - **多专家结构**：部分模型（如 Wan2.2）包含 low_noise 和 high_noise 两个专家模型

2. **确定加载策略**：DiT 模型通常依赖官方推理仓库（如 Wan2.2 依赖 `wan` 包），需通过推理管线加载。

3. **确定量化对象**：量化主要针对主干 Transformer 的 Attention Block，辅助模块（VAE、Text Encoder）通常不量化。

**Wan2.2 结构特点**：

- 依赖 `wan` 官方推理仓库，通过 `WanT2V` / `WanI2V` / `WanTI2V` 管线加载
- 支持三种任务：T2V（文生视频）、I2V（图生视频）、TI2V（文图生视频）
- T2V/I2V 为多专家结构（low_noise_model + high_noise_model），TI2V 为单专家结构
- Attention Block 类名包含 `attentionblock` 关键字

**输出**：模型结构分析文档，确定适配策略

### 步骤 2：创建模型适配器目录与文件

**目标**：创建 DiT 模型适配器目录结构。

**操作**：

在 `msmodelslim/model/` 下创建模型目录：

```text
msmodelslim/model/{model_name}/
├── __init__.py
├── model_adapter.py    # 主适配器文件（继承 BaseModelAdapter）
├── loader.py           # 加载器文件
├── base_model_adapter.py  # （可选）基础适配器
├── expert_sub_adapter.py  # （可选）专家子适配器
├── constants.py        # （可选）常量定义
├── t2v/                # （可选）按任务拆分
│   ├── __init__.py
│   ├── loader.py
│   └── model_adapter.py
├── i2v/
│   └── ...
└── ti2v/
    └── ...
```

**Wan2.2 参考结构**：

```text
msmodelslim/model/wan2_2/
├── __init__.py
├── model_adapter.py        # Wan2Point2Adapter 类
├── loader.py               # Wan2_2AdapterLoader 类
├── base_model_adapter.py
├── expert_sub_adapter.py
├── constants.py
├── t2v/
├── i2v/
└── ti2v/
```

**输出**：模型适配器目录结构创建完成

### 步骤 3：实现 Loader 加载器

**目标**：实现加载器类，注册适配器类的导入路径。

**操作**：

```python
# msmodelslim/model/{model_name}/loader.py
from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class {ModelName}AdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.{model_name}.model_adapter:{ModelName}Adapter"
```

**Wan2.2 参考**：`msmodelslim/model/wan2_2/loader.py:7`

```python
class Wan2_2AdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.wan2_2.model_adapter:Wan2Point2Adapter"
```

**输出**：加载器文件创建完成

### 步骤 4：定义适配器类，继承所需接口

**目标**：定义 DiT 模型适配器主类，继承基础适配器和多模态生成量化接口。

**操作**：

DiT 适配器与 LLM/VLM 的关键区别：

- **继承 `BaseModelAdapter`**，不继承 `TransformersModel`
- **使用 `LegacyMultimodalPipelineInterface`**（多模态生成 pipeline 接口）
- 需实现 `run_calib_inference`、`apply_quantization` 等生成模型特有方法

```python
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.interface_hub import (
    ModelInfoInterface,
    LegacyMultimodalPipelineInterface,
    FA3QuantAdapterInterface,
    OnlineQuaRotInterface,
    IterSmoothInterface,
)
from msmodelslim.utils.logging import logger_setter


@logger_setter()
class {ModelName}Adapter(
    BaseModelAdapter,
    ModelInfoInterface,
    LegacyMultimodalPipelineInterface,  # 多模态生成 pipeline 接口
    FA3QuantAdapterInterface,          # FA3 激活量化
    OnlineQuaRotInterface,             # 在线 QuaRot
    IterSmoothInterface,               # 异常值抑制
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.pipeline = None
        self.transformer = None
        self.low_noise_model = None
        self.high_noise_model = None
        self._get_default_model_args()
```

**Wan2.2 参考**：`msmodelslim/model/wan2_2/model_adapter.py:86-103`

```python
@logger_setter()
class Wan2Point2Adapter(
    BaseModelAdapter,
    ModelInfoInterface,
    LegacyMultimodalPipelineInterface,
    FA3QuantAdapterInterface,
    OnlineQuaRotInterface,
    IterSmoothInterface,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.pipeline = None
        self.transformer = None
        self.model_args = None
        self.low_noise_model = None
        self.high_noise_model = None
        self._get_default_model_args()
```

**输出**：适配器类骨架定义完成

### 步骤 5：实现推理管线加载与模型参数配置

**目标**：实现 DiT 模型的推理管线加载和参数配置。

**操作**：

#### 5.1 `set_model_args`——配置模型参数

从 YAML 配置读取推理参数，更新到 `model_args`：

```python
def set_model_args(self, override_model_config: object):
    """将 override_model_config 的属性更新到 model_args"""
    self.model_args.ckpt_dir = self.model_path

    # 校验配置属性合法性
    missing_attrs = []
    for key in override_model_config.keys():
        if not hasattr(self.model_args, key):
            missing_attrs.append(key)
    if missing_attrs:
        raise SchemaValidateError(f"illegal config attributes: {missing_attrs}")

    # 更新配置
    for key in override_model_config.keys():
        setattr(self.model_args, key, override_model_config[key])
```

#### 5.2 `load_pipeline`——加载推理管线

通过官方推理仓库加载 DiT 推理管线：

```python
def load_pipeline(self):
    self._load_pipeline()

def _load_pipeline(self):
    # 1. 检查依赖
    self._check_import_dependency()

    # 2. 根据 task 加载对应推理管线
    if "t2v" in args.task:
        self.wan_t2v = wan.WanT2V(config=cfg, checkpoint_dir=args.ckpt_dir, ...)
        self.low_noise_model = self.wan_t2v.low_noise_model
        self.high_noise_model = self.wan_t2v.high_noise_model
    elif "ti2v" in args.task:
        self.wan_ti2v = wan.WanTI2V(config=cfg, checkpoint_dir=args.ckpt_dir, ...)
        self.transformer = self.wan_ti2v.model
    else:
        self.wan_i2v = wan.WanI2V(config=cfg, checkpoint_dir=args.ckpt_dir, ...)
        self.low_noise_model = self.wan_i2v.low_noise_model
        self.high_noise_model = self.wan_i2v.high_noise_model
```

#### 5.3 `_check_import_dependency`——检查依赖

```python
def _check_import_dependency(self):
    import importlib
    try:
        for mod in ("PIL", "wan", "wan.configs", "mindiesd"):
            importlib.import_module(mod)
    except ImportError as e:
        raise ImportError(
            "Failed to import required components from wan. "
            "Please install the Wan2.2 from Modelers and add the Wan2.2 repository "
            "to PYTHONPATH. e.g. export PYTHONPATH=/path/to/Wan2.2:$PYTHONPATH"
        ) from e
```

**输出**：推理管线加载与参数配置实现完成

### 步骤 6：实现核心接口方法

**目标**：实现 DiT 适配器的核心接口方法，包括校准数据 dump、模型访问序列、前向传播等。

**操作**：

#### 6.1 `run_calib_inference`——运行浮点推理生成校准数据

DiT 量化需要先运行浮点推理管线，生成校准数据：

```python
def run_calib_inference(self):
    """运行浮点推理管线，dump 校准数据"""
    stream = torch.npu.Stream()
    args = self.model_args

    for _ in tqdm(range(1), desc='Dump calib data by float model inference'):
        torch.manual_seed(args.base_seed)
        torch.npu.manual_seed(args.base_seed)

        if "t2v" in args.task:
            self.wan_t2v.generate(args.prompt, size=SIZE_CONFIGS[args.size],
                                  frame_num=args.frame_num, sampling_steps=args.sample_steps, ...)
        elif "ti2v" in args.task:
            self.wan_ti2v.generate(args.prompt, img, size=..., frame_num=..., ...)
        elif "i2v" in args.task:
            self.wan_i2v.generate(args.prompt, img, max_area=..., frame_num=..., ...)

        stream.synchronize()
```

#### 6.2 `init_model`——初始化需要量化的子模型

根据任务类型返回需要量化的子模型列表：

```python
def init_model(self, device: DeviceType = DeviceType.NPU) -> Dict[str, nn.Module]:
    if "ti2v" in self.model_args.task:
        return {'': self.transformer}  # 单专家
    else:
        return {  # 多专家
            'low_noise_model': self.low_noise_model,
            'high_noise_model': self.high_noise_model,
        }
```

#### 6.3 `generate_model_visit`——生成模型访问序列

通过关键字 `attentionblock` 定位 Attention Block 模块：

```python
def generate_model_visit(self, model: torch.nn.Module, transformer_blocks=None):
    return generated_decoder_layer_visit_func_with_keyword(
        model, keyword="attentionblock"
    )
```

#### 6.4 `generate_model_forward`——生成前向传播序列

通过 hook 获取第一个 Attention Block 的输入，逐块前向传播：

```python
def generate_model_forward(self, model, inputs):
    # 1. 获取所有 Attention Block
    transformer_blocks = [
        (name, module) for name, module in model.named_modules()
        if "attentionblock" in module.__class__.__name__.lower()
    ]

    # 2. 通过 hook 获取第一个 Block 的输入
    first_block_input = None

    def break_hook(module, hook_args, hook_kwargs):
        nonlocal first_block_input
        first_block_input = (hook_args, hook_kwargs)
        raise TransformersForwardBreak()

    hooks = [transformer_blocks[0][1].register_forward_pre_hook(
        break_hook, with_kwargs=True
    )]

    try:
        model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
    except TransformersForwardBreak:
        pass
    finally:
        for hook in hooks:
            hook.remove()

    # 3. 逐块前向传播
    current_inputs = first_block_input
    for name, block in transformer_blocks:
        args, kwargs = current_inputs
        outputs = yield ProcessRequest(name, block, args, kwargs)
        current_inputs = ((outputs,), current_inputs[1])
```

#### 6.5 `apply_quantization`——应用量化

DiT 量化时需将非 blocks 模块移至 NPU，blocks 模块留在 CPU 逐层处理：

```python
def apply_quantization(self, process_model_func):
    # 遍历所有子模块
    for name, module in self.transformer.named_modules():
        if not name:
            continue
        if not name.startswith('blocks'):
            module.to('npu')  # 非 blocks 模块移至 NPU
        else:
            module.to('cpu')  # blocks 模块留在 CPU

    with torch.no_grad():
        process_model_func()
```

#### 6.6 可选：实现异常值抑制接口

```python
def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
    adapter_config = []
    for layer_idx in range(self.transformer.num_layers):
        # Self-Attention QKV 的 Norm-Linear 融合
        self_attn_qkv_mapping = MappingConfig(
            targets=[f"blocks.{layer_idx}.self_attn.q",
                     f"blocks.{layer_idx}.self_attn.k",
                     f"blocks.{layer_idx}.self_attn.v"]
        )
        adapter_config.append(AdapterConfig(
            subgraph_type="norm-linear", mapping=self_attn_qkv_mapping
        ))

        # Cross-Attention Q 与 K/V 分开统计
        cross_attn_q_mapping = MappingConfig(
            targets=[f"blocks.{layer_idx}.cross_attn.q"]
        )
        cross_attn_kv_mapping = MappingConfig(
            targets=[f"blocks.{layer_idx}.cross_attn.k",
                     f"blocks.{layer_idx}.cross_attn.v"]
        )
        adapter_config.extend([
            AdapterConfig(subgraph_type="norm-linear", mapping=cross_attn_q_mapping),
            AdapterConfig(subgraph_type="norm-linear", mapping=cross_attn_kv_mapping),
        ])
    return adapter_config
```

**输出**：适配器接口方法实现完成

### 步骤 7：注册模型到配置

**目标**：在 `config/config.ini` 中注册 DiT 模型名称和适配器入口点。

**操作**：

```ini
[ModelAdapter]
{model_name} = {ModelName1}, {ModelName2}

[ModelAdapterEntryPoints]
{model_name} = msmodelslim.model.{model_name}.loader:{ModelName}AdapterLoader
```

**Wan2.2 参考**：

```ini
[ModelAdapter]
wan2_2 = Wan2_2, Wan2.2
wan2_2_t2v = Wan2.2-T2V-A14B
wan2_2_i2v = Wan2.2-I2V-A14B
wan2_2_ti2v = Wan2.2-TI2V-5B

[ModelAdapterEntryPoints]
wan2_2 = msmodelslim.model.wan2_2.loader:Wan2_2AdapterLoader
wan2_2_t2v = msmodelslim.model.wan2_2.t2v.loader:Wan2_2T2VAdapterLoader
wan2_2_i2v = msmodelslim.model.wan2_2.i2v.loader:Wan2_2I2VAdapterLoader
wan2_2_ti2v = msmodelslim.model.wan2_2.ti2v.loader:Wan2_2TI2VAdapterLoader
```

**输出**：模型注册完成

### 步骤 8：创建 YAML 量化配置文件

**目标**：创建 DiT 专用的量化配置文件。

**操作**：

DiT 的 YAML 配置使用 `apiversion: multimodal_sd_modelslim_v1`，需配置推理参数和校准 dump：

```yaml
# lab_practice/{model_name}/{model_name}_{quant_type}_{task}.yaml
apiversion: multimodal_sd_modelslim_v1
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: True
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: True
          method: "mse_round"
      include:
        - "*"
    - type: "fa3_quant"
      qconfig:
        dtype: "fp8_e4m3"
        scope: "per_token"
        symmetric: True
        method: "minmax"
      include:
        - "*self_attn"
      exclude:
        - "*blocks.0.self_attn*"

  dataset: wan2_2_t2v

  save:
    - type: "mindie_format_saver"
      part_file_size: 0

  multimodal_sd_config:
    dump_config:
      enable_dump: False
      capture_mode: "args"
      dump_data_dir: ""
    inference_config:
      size: "1280*720"
      frame_num: 81
      sample_steps: 40
      convert_model_dtype: True
      task: "t2v-A14B"
```

**Wan2.2 参考**：`lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_t2v.yaml`

**输出**：量化配置文件创建完成

### 步骤 9：执行量化验证

**目标**：使用量化命令验证 DiT 适配器工作正常。

**操作**：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config ${CONFIG_PATH}
```

**验证要点**：

1. 推理管线加载正常，无 ImportError
2. 浮点推理生成校准数据正常
3. Attention Block 逐块量化和前向传播无报错
4. 多专家模型各专家子目录均有完整权重
5. 量化权重可被 MindIE-SD 加载

**输出**：量化验证通过

## 5. 结果与经验

### 5.1 关键结果汇总

| 步骤 | 关键操作 | 指标 | 变化 | 备注 |
| --- | --- | --- | --- | --- |
| 适配器创建 | 继承 BaseModelAdapter + LegacyMultimodalPipelineInterface | 代码行数 | 基础适配器约500行 | 含推理管线加载逻辑 |
| 推理管线加载 | 通过 wan 官方仓库加载 WanT2V/I2V/TI2V | 依赖 | 需安装 wan 包 | 需加入 PYTHONPATH |
| 校准数据生成 | 运行浮点推理 dump 校准数据 | 耗时 | 与推理配置相关 | 可设 enable_dump=False 跳过 |
| 多专家量化 | low_noise + high_noise 分别量化 | 输出 | 各专家独立子目录 | TI2V 为单专家 |

### 5.2 经验总结

1. **推理管线依赖**：DiT 模型通常依赖官方推理仓库（如 Wan2.2 依赖 `wan` 包），接入时需将官方仓库加入 PYTHONPATH，且需保证浮点推理可正常运行。适用边界：仅适用于有官方推理实现的开源 DiT 模型。
2. **校准数据 dump**：DiT 量化通过运行浮点推理管线生成校准数据（前向捕获激活分布），耗时较长。纯动态量化场景可设 `enable_dump: False` 跳过浮点推理。
3. **多专家量化**：多专家模型（如 Wan2.2 的 low_noise + high_noise）需分别量化，校准数据按专家分别 dump 为 pth 文件，`init_model` 返回各专家子模型字典。
4. **Attention Block 定位**：DiT 无标准 Decoder Layer 概念，通过 `generated_decoder_layer_visit_func_with_keyword` 按类名关键字（如 `attentionblock`）定位可量化模块，需确保关键字与模型类名匹配。

## 6. 异常处理

- **推理管线加载失败**：确认 Wan2.2 官方仓库已安装并加入 PYTHONPATH；确认 `wan`、`mindiesd` 等依赖可正常导入。
- **浮点推理失败**：检查 `inference_config` 参数是否正确（size、frame_num、sample_steps、task 等）；确认 task 与模型类型匹配。
- **校准数据缺失**：检查 `enable_dump` 是否为 True，`dump_data_dir` 是否有写入权限；确认多专家模型的每个专家均有对应的 `calib_data` key。
- **显存不足**：降低分辨率（`size`）或减少推理步数（`sample_steps`）；减少校准样本数量。此外，若仍遇到显存不足，请确认 `--device npu --device_id 0` 指定的 NPU 未被其他任务占用。
- **MindIE-SD 加载失败**：确认使用 `mindie_format_saver` 保存权重；确认多专家模型各专家子目录权重完整。

## 7. 附录

- 参考实现：[Wan2.2 模型适配器](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/wan2_2/model_adapter.py)
- 参考配置：[Wan2.2 T2V W8A8 MXFP8 配置](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_t2v.yaml)
- 接口定义：[Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/interface_hub.py)
- Wan2.2 官方仓库：[Wan2.2](https://github.com/Wan-Video/Wan2.2)
