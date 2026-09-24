# 多模态理解模型（VLM）量化使用指南

## 1. 适用范围

本指南面向对多模态视觉语言模型（如 Qwen2.5-VL、Qwen3-VL、GLM-4.6V、InternVL3.5 等）执行[训练后量化（PTQ）](../term_ptq.md)的用户。文档提供规范的量化使用流程，系统阐明模型适配接口、多模态校准集配置、保存格式、算法选型与视觉组件排除策略、调度器（Runner）机制，并提供完整的开箱即用示例。

适用场景：

- 将浮点 VLM 权重高效量化为 W8/W4 等低比特格式并导出；
- 对 MoE 架构多模态模型执行混合量化（Dense 层静态量化 + Expert 层动态量化）；
- 为多模态在线推理服务（如基于昇腾 NPU 的 MindIE 或 vLLM-Ascend）准备部署权重。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点 VLM 权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 多模态校准数据集 | 工具内置 `lab_calib/calibImages/` 或用户自定义路径 | 包含图像文件及可选配对 Prompt 的目录或 JSONL，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为多模态输入张量 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `multimodal_vlm_modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors` 分片 | 导出完整且推理冒烟测试通过 |

## 3. 流程总览

VLM 量化的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线接口] --> B[配置多模态<br/>校准数据集]
    B --> C[选择权重保存格式]
    C --> D[选型与编排<br/>量化算法]
    D --> E[选择执行调度器]
    E --> F[编写配置<br/>并执行量化]
```

各阶段的关键细节如下：

前置条件：模型适配：量化前需先完成模型适配，适配器实现 PipelineInterface（基础流水线接口）等接口，详见[《VLM 模型接入量化流程指南》](./integration_guide_vision_transformer_quantization.md)。若目标模型已在官方预验证列表中，无需编写适配代码，直接指定对应 `--model_type` 即可。模型适配必须先完成，才能执行量化流程。

- **适配模型流水线接口**：确保待量化 VLM 模型具备符合 `multimodal_vlm_modelslim_v1` 规范的适配器接口，能够驱动视觉特征提取与文本解码的联合前向。
- **配置多模态校准数据集**：通过 `spec.dataset` 与 `spec.default_text` 指定图文校准数据，推荐 50 条图像样本及配对 Prompt，稳定覆盖通用视觉特征分布。
- **选择权重保存格式**：在 `spec.save` 中声明保存处理器，昇腾部署场景首选 `ascendv1_saver`，务必与部署框架版本对齐。
- **选型与编排量化算法**：通过 `include`/`exclude` 控制量化范围，重点将视觉敏感投影层（如 `*merger*`、`*linear_fc2`）排除在量化之外，抑制跨模态对齐误差。
- **选择执行调度器（Runner）**：在 `spec.runner` 中指定调度策略，推荐 `auto`：单卡自动逐层调度，多卡自动数据并行；不支持 `model_wise`。
- **编写配置并执行量化**：整合上述配置生成完整 YAML，通过 `msmodelslim quant` 命令启动量化，导出量化权重与描述文件。

## 4. 操作步骤

### 步骤 1：适配模型流水线接口（V1 Interface）

**目标**：确保待量化 VLM 模型具备符合 `multimodal_vlm_modelslim_v1` 规范的适配器接口，能够驱动视觉特征提取与文本解码的联合前向。

**操作**：

若目标模型已在官方支持列表中，无需编写适配代码，直接指定对应 `--model_type` 即可；若需接入新 VLM 模型，请参见[《VLM 模型接入量化流程指南》](./integration_guide_vision_transformer_quantization.md)。

该指南详细说明了 `PipelineInterface` 规范、核心方法说明（`handle_dataset`、`init_model`、`generate_model_visit`、`generate_model_forward`、`enable_kv_cache`）以及视觉组件封装与适配器注册开发步骤。

**输出**：已在框架中注册可用的 VLM 适配器（通过 `--model_type` 调用）。

### 步骤 2：配置校准数据集

**目标**：选取代表性多模态样本，准确统计视觉与文本跨模态前向中的激活分布。

**操作**：

1. **配置方式与解析机制**：
   在 YAML 配置中通过 `spec.dataset` 和 `spec.default_text` 指定：

   ```yaml
   spec:
     dataset: calibImages                            # 相对目录名或绝对路径
     default_text: "Describe this image in detail." # 图像样本缺省文本时的默认 Prompt
   ```

   若配置为相对文件名 `calibImages`，框架自动寻址内置的 `lab_calib/calibImages/` 图像目录；若为自定义绝对路径或指定 `index.jsonl`，则直接载入对应数据源。

2. **校准集选择指南与推荐配置**：
   - **确认校准集模态支持范围**：校准集的模态支持与模型类型相关，**全模态模型与非全模态模型的校准集支持区别如下**：
      - **全模态模型（如 Qwen3-Omni）**：样本可包含 `text` 及 `image` / `audio` / `video` 的任意组合，`text` 缺省时使用配置中的 `default_text`。同一量化任务内所有样本的有效模态签名必须一致（同质约束），例如不能混用纯文本与图文样本，也不能混用带音轨与不带音轨的视频。
      - **非全模态模型（如仅支持文本 + 图像的 Qwen3-VL、Qwen3.5 等 VLM）**：适配器仅消费 `text` 和 `image` 字段，包含 `audio` 或 `video` 的样本不被适配器支持。
   - **通用多模态量化（推荐）**：默认采用内置的 `calibImages`。图像涵盖通用自然场景、图表与图文排版，配合默认描述 Prompt 可稳定覆盖通用视觉特征分布。
   - **垂直场景定制**：若模型面向工业缺陷检测、医疗影像问答或自动驾驶场景，应替换为真实业务图像，并在 `default_text` 或索引文件中对齐领域专用指令（如“请分析图中异常区域”）。
   - **校准样本量**：**推荐 50 条**（50 张图像及其配对 Prompt）。多模态前向中视觉编码器计算开销相对纯文本较大，50 条样本能够在充分覆盖特征分布的同时避免显著延长校准耗时。

**输出**：在 YAML 中确认的 `dataset` 与 `default_text` 配置。

### 步骤 3：选择权重保存格式

**目标**：根据下游推理部署框架的要求，选择匹配的量化权重导出格式。

**操作**：

1. **配置接口**：
   在 YAML 的 `spec.save` 列表中声明保存处理器：

   ```yaml
   spec:
     save:
       - type: ascendv1_saver       # 昇腾 NPU 部署推荐格式
   ```

2. **格式选型与文档链接**：
   - **[ascendv1_saver 配置说明](../../../api_reference/config/format/ascendv1_saver.md)**：**昇腾 NPU 部署推荐格式**。适配 MindIE 及 vLLM-Ascend 推理引擎，导出包含多模态模型量化参数的 `quant_model_description.json` 及分片量化权重 `*.safetensors`。
   - **[mindie_format_saver 配置说明](../../../api_reference/config/format/mindie_format_saver.md)**：面向 MindIE-SD/MindIE 生态的统一部署格式。

3. **选型指南**：
   - 昇腾部署场景首选 `ascendv1_saver`。
   - 务必与部署框架版本对齐，保存格式不匹配会导致加载权重报错。

**输出**：在 YAML 中配置完成的 `save` 格式。

### 步骤 4：选型与编排量化算法

**目标**：确定量化位宽与范围，重点通过针对性的模块排除策略保障跨模态对齐精度。

**操作**：

1. **典型量化算法**：
   量化流程通过处理器链（`spec.process`）按声明顺序串行调度。常用算法如下：

   | 算法 | 简述 | 适用场景 | 链接 |
   | --- | --- | --- | --- |
   | MinMax | 统计张量极值直接计算 scale 与 zero-point，开销低 | W8A8 量化的基础估计算法 | [《MinMax》](../../quantization_algorithms/minmax/usage_minmax.md) |
   | SmoothQuant | 将激活离群值按通道缩放迁移到权重 | 降低全网静态量化难度 | [《SmoothQuant》](../../quantization_algorithms/smooth_quant/usage_smooth_quant.md) |
   | QuaRot | 引入正交 Walsh-Hadamard 旋转矩阵打散激活离群特征 | 改善低比特多模态模型的分布偏移 | [《QuaRot》](../../quantization_algorithms/quarot/usage_quarot.md) |
   | KVCache Quant | 对多轮图文对话中的 Key/Value 缓存张量量化 | 多轮图文对话的显存优化 | [《KVCache Quant》](../../quantization_algorithms/kvcache_quant/usage_kvcache_quant.md) |

   > 完整算法列表请参阅[《量化算法总览》](../../quantization_algorithms/README.md)。

2. **算法选型与调优**：
   - **视觉敏感投影层排除（关键策略）**：视觉特征投影层（如 `*merger*`、`*linear_fc2`、`*deepstack_merger_list*` 等）负责将图像 patch 特征映射至文本 token 空间，对数值精度高度敏感。在 `exclude` 列表中显式排除，保留浮点权重，可抑制跨模态对齐误差导致的图文理解崩塌。
   - **W8A8 基础量化（推荐起点）**：权重 `per_channel` 对称量化，激活按精度诉求选择 `per_tensor` 或 `per_token`，配合投影层排除，作为大多数 VLM 的首选基线。
   - **MoE 混合量化**：带 MoE 架构的 VLM 通常对 Shared/Dense 层静态量化、Router/Expert 层动态量化，兼顾吞吐与路由精度。
   - **单变量调参**：优先围绕 `exclude` 调整敏感层排除范围，避免盲目叠加多种算法。

**输出**：在 YAML 中编排完毕的 `process` 算法链与模块过滤规则。

### 步骤 5：选择执行调度器（Runner）

**目标**：结合 VLM 视觉编码器与多卡并行环境，选择最佳调度引擎。

**操作**：

1. **接口原理与调度机制**：
   在 YAML 的 `spec.runner` 中指定（源码位于 `msmodelslim/core/runner/`）：

   ```yaml
   spec:
     runner: auto   # auto | layer_wise | dp_layer_wise
   ```

   - **`LayerWiseRunner`（逐层量化调度器）**：仅在当前网络层前向时加载权重，执行完毕立刻卸载释放。
   - **`DPLayerWiseRunner`（数据并行逐层调度器）**：多卡并行切分多模态校准数据，分布式前向聚合激活统计量，成倍缩减量化耗时。
   - **`auto`（自适应调度，推荐）**：单卡自动调度 `layer_wise`；检测到多设备（`device_indices > 1`）时自动调度 `dp_layer_wise`。
   - **约束说明**：VLM 量化**不支持** `model_wise` 调度器。若配置为 `model_wise`，框架将告警并自动回退为 `layer_wise`。

2. **配置指南**：
   - **推荐默认填写 `runner: auto`**：单卡与多卡环境自动兼容。
   - 多卡并行加速：在命令行传入多个设备（如 `--device npu --device_id 0 1 2 3`），配合 `auto` 即可自动启用多卡分布式逐层调度。

**输出**：YAML 中确认的 `runner` 调度策略。

### 步骤 6：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。有关量化命令参数的完整说明，请参阅[《权重量化使用指南》](../../../user_guide/usage_weight_quantization.md)。

**操作**：

#### 完整示例：VLM W8A8 量化（含视觉投影层排除）

##### 配置文件：`vlm_w8a8.yaml`

```yaml
apiversion: multimodal_vlm_modelslim_v1
spec:
  runner: auto                    # 单卡自动使用 layer_wise，多卡自动使用 dp_layer_wise
  process:
    - type: linear_quant
      qconfig:
        act:
          dtype: int8
          scope: per_tensor       # 激活静态量化
          symmetric: true
          method: minmax
        weight:
          dtype: int8
          scope: per_channel      # 权重通道级量化
          symmetric: true
          method: minmax
      include: ["*"]              # 匹配全网线性算子
      exclude:
        - "*merger*"              # 排除视觉投影层保留浮点
        - "*linear_fc2*"
        - "*deepstack_merger_list*"
  save:
    - type: ascendv1_saver        # 昇腾推理标准保存格式
  dataset: calibImages            # 多模态校准图像目录
  default_text: "Describe this image in detail."
```

##### 执行命令（单卡量化）

```bash
msmodelslim quant \
  --model_path <浮点模型目录> \
  --save_path <量化权重输出目录> \
  --model_type <模型适配器名称> \
  --config ./vlm_w8a8.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| PTQ | 训练后量化（Post-Training Quantization） | [《训练后量化词条》](../term_ptq.md) |
| VLM 量化 | 多模态视觉语言模型训练后量化 | [《VLM 量化词条》](./term_vision_transformer_quantization.md) |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim quant` | 一键量化 CLI 入口与完整命令说明 | [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md) |
| multimodal_vlm_modelslim_v1 配置说明 | `runner`/`process`/`save`/`dataset`/`default_text` 等任务级配置的字段说明 | [《multimodal_vlm_modelslim_v1 配置说明》](../../../api_reference/config/quant/multimodal_vlm_modelslim_v1.md) |
