# QuaRot 旋转量化算法使用指南

## 1. 适用范围

本指南面向需要使用 [QuaRot 旋转量化算法](./term_quarot.md) 的用户。QuaRot 作为离群值抑制算法，通常作为量化前的预处理步骤，通过正交旋转平滑激活分布，提升低比特量化的精度。

适用场景：

- W8/W4 等低比特量化精度不达标，需要通过正交旋转打散激活离群通道；
- 需要在不改变模型前向输出的前提下，使激活与权重分布更加均匀。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及 QuaRot 相关接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为前向张量 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors` 分片 | 导出完整且推理冒烟测试通过 |

## 3. 流程总览

QuaRot 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型旋转接口] --> B[确定<br/>离线/在线旋转方式]
    B --> C[选择旋转块大小]
    C --> D[应用正交旋转]
    D --> E[衔接低比特量化]
    E --> F[验证精度<br/>与部署信息]
```

各阶段的关键细节如下：

- **适配模型旋转接口**：适配器需实现 `QuaRotInterface`（离线旋转）、`OnlineQuaRotInterface`（在线旋转）等接口，提供旋转映射、Norm 融合与均值融合等网络结构信息，是后续所有旋转步骤的前提。
- **确定离线/在线旋转方式**：优先使用离线旋转（`online: false`），旋转矩阵在量化阶段融合进权重，推理无额外开销；仅当部署链明确支持运行时旋转时才开启在线旋转，并需同步评估推理时延与算子兼容性。旋转矩阵默认由 `QuaRotInterface.get_rotate_command` 自动生成 Walsh-Hadamard 正交矩阵，NPU 可用时自动搬移至当前设备；也可通过 `OnlineQuaRotInterface` 的 `RotationConfig` 提供自定义矩阵。
- **选择旋转块大小**：`block_size: -1` 表示全 hidden_dim 旋转，能最大化打散全维离群；指定 2 的幂次（如 `32`）则分块旋转，用于适配特定并行/硬件约束。
- **应用正交旋转**：旋转前通过 `get_ln_fuse_map` 融合 LayerNorm，通过 `get_bake_names` 对必要 Linear 层做均值融合，保证旋转前后数学等价，再逐层应用正交旋转打散离群通道。
- **衔接低比特量化**：旋转后张量分布更均匀，衔接 `linear_quant` 等低比特量化处理即可提升精度；建议保持后续量化配置不变，只调整旋转参数以定位收益。
- **验证精度与部署信息**：`export_extra_info: true` 时由 saver 导出全局旋转矩阵等元信息，供下游推理链重建旋转，不要随意关闭；最终以端到端精度与部署链兼容性验证为准。

## 4. 操作步骤

### 步骤 1：适配模型流水线接口

**目标**：量化工具不能直接操作任意结构的模型，它要求每个模型先套一层“适配器”，把模型的各种操作翻译成框架能统一调用的标准方法。本步骤即确认并完成这层适配器。模型适配必须先完成，才能执行 QuaRot 算法。

**操作**：适配器需实现基础流水线接口 `PipelineInterface`（位于 `msmodelslim.core.runner.pipeline_interface`，继承自基础属性接口 `IModel`）。该接口定义了框架驱动模型所需的 5 个标准方法：加载数据（`handle_dataset`）、加载模型（`init_model`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）、控制 KV Cache（`enable_kv_cache`）。量化调度器（Runner）与处理器（Processor）只通过这些标准方法驱动模型，与模型内部结构无关。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承快速构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface):
    pass
```

`TransformersModel` 提供通用的 LLM 加载与 Tokenizer 逻辑，框架接口提供标准化驱动能力，拼在一起即构成完整适配器。之后量化命令中通过 `--model_type MyModelAdapter` 引用该适配器。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：适配模型旋转接口

**目标**：在适配器中完成 QuaRot 旋转相关的模型适配，为旋转处理器提供网络结构信息（旋转映射、Norm 融合、均值融合等）。适配完成前无法执行 QuaRot 算法。

**操作**：模型适配器中旋转部分的适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总提供：

1. **`QuaRotInterface`**（位于 `msmodelslim.processor.quarot.offline_quarot.quarot_interface`）：离线旋转核心接口。

   ```python
   class QuaRotInterface:
       QuaRotMode = QuaRotMode
       RotatePair = RotatePair

       @staticmethod
       def get_rotate_command(mode: QuaRotMode,
                  size: int,
                  block_size: int = -1,
                  rot_step: int = 1,
                  eye_step: tuple = (-1,)):
           ...

       @abstractmethod
       def get_ln_fuse_map(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
           ...

       @abstractmethod
       def get_bake_names(self) -> Tuple[List[str], List[str]]:
           ...

       @abstractmethod
       def get_rotate_map(self, block_size: int) -> Tuple[List[RotatePair], List[RotatePair]]:
           ...
   ```

   - **`get_rotate_command`**：调用 `create_rot` 生成正交旋转矩阵（Walsh-Hadamard 等），NPU 可用时自动搬移至当前 NPU 设备。
   - **`get_ln_fuse_map`**：返回 LayerNorm 层与 Linear 层的融合映射（`pre_run_fused_ln` 与 `fused_map` 两个字典），用于旋转前的 Norm 融合。
   - **`get_bake_names`**：返回需要均值融合（mean fusion）的 Linear 层名称列表（`pre_run_bake_names` 与 `bake_names`），模型使用 `nn.LayerNorm` 时通常需要。
   - **`get_rotate_map`**：返回旋转映射，包括 pre-run 阶段（通常为 embedding 层旋转）的 `pre_run_pairs` 与 preprocess 阶段的 `rotate_pairs`，每项为携带左旋/右旋矩阵的 `RotatePair` 列表。
2. **`OnlineQuaRotInterface`**（位于 `msmodelslim.processor.quarot.online_quarot.online_quarot_interface`）：简化在线旋转接口。

   ```python
   class OnlineQuaRotInterface:
       RotationConfig = RotationConfig
       QuaRotMode = QuaRotMode

       @abstractmethod
       def get_online_rotation_configs(self, model: Optional[nn.Module] = None) -> Dict[str, RotationConfig]:
           ...
   ```

   返回模块名（如 `model.layers.0.self_attn.q_rot`）到 `RotationConfig` 的映射；`rotation_type` 支持 `"input"`、`"output"`、`"replace"`、`"offline"`，可提供自定义旋转矩阵或指定 `rotation_size` 自动生成。
3. **`LAOSOnlineRotationInterface`**（位于 `msmodelslim.processor.quarot.offline_quarot.quarot_interface`）：LAOS 在线旋转接口，用于描述注意力头维度及各 Decoder 层 O/V、up/down 投影的旋转对。

   ```python
   class LAOSOnlineRotationInterface:
       @abstractmethod
       def get_head_dim(self):
           ...

       @abstractmethod
       def get_num_attention_heads(self):
           ...

       @abstractmethod
       def get_layer_wise_ov_pair(self, decoder_module):
           ...

       @abstractmethod
       def get_layer_wise_up_down_pair(self, decoder_module):
           ...
   ```

4. **`AdaptRotationInterface`**（位于 `msmodelslim.processor.adapt_rotation`）：用于 AdaptRotation stage1/stage2 两阶段旋转的适配接口，仅在编排 `adapt_rotation` 处理器时需要实现。

   以上接口均通过 `msmodelslim.model.interface_hub` 汇总导出，适配开发时从该模块导入即可。

**适配器实现推荐**：

1. **按旋转场景选择接口**：适配器无需实现所有接口，按实际旋转方式选择即可：
   - **离线旋转（`online: false`，推荐起点）**：实现 `QuaRotInterface`，重点提供 `get_rotate_map`（各 Decoder 层的左旋/右旋映射）、`get_ln_fuse_map`（LayerNorm 与 Linear 融合映射）和 `get_bake_names`（均值融合层）。
   - **在线旋转（`online: true`）**：实现 `OnlineQuaRotInterface`，返回各模块的 `RotationConfig`（`rotation_type` 为 `"input"`/`"output"`/`"replace"`/`"offline"`）。若使用 `down_proj_online_layers` 或 LAOS 特性，还需实现 `LAOSOnlineRotationInterface`。
   - **编排 `adapt_rotation` 处理器时**：额外实现 `AdaptRotationInterface`。

2. **组合继承推荐**：在步骤 1 适配器的基础上，组合继承 QuaRot 相关接口：

   ```python
   from msmodelslim.model.interface_hub import (
       # ---- 旋转算法适配接口（本步骤重点）----
       QuaRotInterface,         # 算法适配：提供 LayerNorm 融合映射（get_ln_fuse_map）与旋转对（get_rotate_map）
       OnlineQuaRotInterface,   # 算法适配：提供在线旋转能力
   )

   class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface,
                        QuaRotInterface, OnlineQuaRotInterface):
       pass
   ```

3. **参考已有适配实现**：仓库内置模型适配器可直接参考，如 `msmodelslim/model/glm_5/quarot.py`、`msmodelslim/model/deepseek_v3/quarot.py`（QuaRot 接口实现），以及 `msmodelslim/model/glm_5/model_adapter.py`（适配器组合方式）。

4. **开发新模型适配代码**：请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 3：选型与编排 QuaRot 量化算法

**目标**：确定 QuaRot 的旋转方式与关键参数，并按需衔接后续低比特量化算法链。

**操作**：

1. **QuaRot 算法介绍**：
   QuaRot 在层间计算中引入正交 Walsh-Hadamard 旋转矩阵，在不改变模型前向输出的前提下打散激活分布中的离群通道，使激活与权重分布更加均匀，通常作为量化前的预处理步骤。

2. **推荐配置**：

   ```yaml
   spec:
     process:
       - type: "quarot"                     # 固定为 `quarot`，用于指定 Processor 类型。
         online: False                      # 控制是否启用在线旋转，默认为 False。
         block_size: -1                     # 旋转矩阵启用块对角矩阵时每个块的大小，取值范围为-1或2的幂次方，如果大于0必须为2的幂，若为-1，表示不进行块对角矩阵处理
         max_tp_size: 4                     # 最大张量并行大小，默认为4，仅在启用在线旋转时生效，必须大于0且为2的幂
         down_proj_online_layers: [ ]       # 用于指定哪些层的down_proj使用在线旋转，默认为空
         export_extra_info: True            # 是否导出全局旋转矩阵：使用 ascend_v1_saver 时在量化权重路径下生成 optional 目录及 quant_model_description.json 追加字段，默认为 True
   ```

3. **参数选择与调整指南**：

   每轮只调一个变量，并保持同一校准集和评测集。

   | 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
   | --- | --- | --- | --- |
   | `type` | 处理器标识，固定 `quarot`。 | 固定。 | 不建议调整。 |
   | `online` | 是否保留在线旋转。`false` 尽可能离线融合；`true` 引入运行时旋转并影响并行/算子要求。 | 默认/优先 `false`。 | 只有目标部署链明确支持并需要在线旋转时开启。开启后不要只看精度，还要评估推理时延和算子兼容。 |
   | `block_size` | 旋转块大小，`-1` 表示整 hidden_dim，正值必须是 2 的幂。仓库实践中不少模型显式使用 `32`。 | 没有模型配方时先 `-1`；已有实践用 `32` 等值时直接沿用。 | 块越小旋转越局部，可能降低全维混合离群的能力，但更适配某些并行/硬件约束。 |
   | `down_proj_online_layers` | 指定哪些 Decoder 层的 down_proj 使用在线旋转，默认空。 | 默认 `[]`。 | 仅在这些层无法离线融合或专用配方要求时加入。层越多，运行时额外旋转越多。 |
   | `max_tp_size` | 在线旋转可支持的最大 TP，必须为 1 或 2 的幂，默认 4。 | 在线模式设置为实际部署最大 TP；离线保持默认。 | 实际部署 TP 改变时才调整。 |
   | `export_extra_info` | 是否导出全局旋转元信息，默认 true，供特定 saver/部署链消费。 | 若目标 saver/后端需要 QuaRot 元信息则保持 `true`。 | 只有确认下游完全不读取这些字段时才关闭。 |

4. **算法编排指南**：
   - **先跑推荐配置，再调单变量。** 不要同时修改位宽、粒度、算法参数和层范围。
   - **优先回退局部，而不是整体提高精度。** 如果只有少数层敏感，优先通过 `exclude` 或混合策略保留高精度。
   - **最终以模型实践配置和部署能力为准。** 目标模型已有 `lab_practice` 配方时，应优先复用已验证组合。
   - QuaRot 之后需衔接低比特量化（如 `linear_quant`），QuaRot 本身通常不是最终的量化格式。

**输出**：在 YAML 中编排完毕的 `process` 算法链。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：QuaRot + W8A8 动态量化（推荐起点）

##### 配置文件：`quarot_w8a8_dynamic.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto                    # 单卡自动使用 layer_wise，多卡自动使用 dp_layer_wise
  process:
    - type: quarot                # QuaRot 离群值抑制预处理
      online: False               # 优先离线旋转融合
      block_size: -1              # -1 表示整 hidden_dim 旋转
      max_tp_size: 4              # 在线旋转时生效的最大 TP
      down_proj_online_layers: [] # 默认无 down_proj 在线旋转层
      export_extra_info: True     # 导出全局旋转矩阵元信息
    - type: linear_quant          # 衔接低比特量化
      qconfig:
        act:
          dtype: int8
          scope: per_token        # 动态激活量化，精度表现好
          symmetric: true
          method: minmax
        weight:
          dtype: int8
          scope: per_channel
          symmetric: true
          method: minmax
      include: ["*"]
  save:
    - type: ascendv1_saver        # 昇腾推理标准保存格式
  dataset: mix_calib.jsonl        # 内置混合校准集
```

##### 执行命令（单卡量化）

```bash
msmodelslim quant \
  --model_path <浮点模型目录> \
  --save_path <量化权重输出目录> \
  --model_type <模型适配器名称> \
  --config_path ./quarot_w8a8_dynamic.yaml \
  --device npu:0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件（含 QuaRot 旋转元信息）。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| QuaRot 旋转量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《QuaRot 旋转量化算法 量化术语百科词条》](./term_quarot.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线适配接口（数据预处理、模型加载、模块遍历）。 | [《LLM 量化使用指南·步骤 1》](../../ptq/llm/usage_large_language_model_quantization.md) |
| `QuaRotInterface` / `OnlineQuaRotInterface` / `LAOSOnlineRotationInterface` / `AdaptRotationInterface` | QuaRot 算法模型适配接口，均由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| quarot 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《quarot 配置说明》](../../../api_reference/config/processor/quarot.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
