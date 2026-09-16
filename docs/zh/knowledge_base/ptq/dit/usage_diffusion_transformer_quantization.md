# 多模态生成模型（DiT）量化使用指南

## 1. 适用范围

本指南面向首次对 Diffusion Transformer 架构的多模态生成模型（如 FLUX.1、HunyuanVideo、Wan2.1、Wan2.2、Qwen-Image-Edit 等）执行[训练后量化（PTQ）](../term_ptq.md)的用户。文档提供规范的量化使用流程，系统阐明多模态生成模型的适配接口、校准集与浮点推理配置、保存格式、算法选型、调度器（Runner）机制，并提供完整的 W8A8 量化开箱即用示例。

适用场景：

- 文生图模型（FLUX.1、SD3 等）的 W8A8 MXFP8 量化；
- 文生视频模型（HunyuanVideo、Wan2.2、Wan2.1 等）的低比特量化部署；
- 图像编辑模型（Qwen-Image-Edit 等）的量化部署；
- 为基于昇腾 NPU 的 MindIE-SD 等推理引擎准备量化权重。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点 DiT 模型权重目录 | 模型下载或本地路径 | 含模型权重文件及配置文件 | 可被推理管线正常加载 |
| 输入 | 校准数据集与任务配置 | YAML 中 `dataset` 与 `multimodal_sd_config.inference_config` | 指定任务类型（如 `t2v`、`i2v`）及推理分辨率、步数等参数，校准样本推荐 50 条 | 适配器成功解析并完成浮点推理 dump |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `multimodal_sd_modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | MindIE-SD 格式（含描述文件及分片权重；多专家模型包含各专家子目录） | 导出完整且推理冒烟测试通过 |

## 3. 流程总览

DiT 量化的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线接口] --> B[配置校准数据集<br/>与浮点推理参数]
    B --> C[选择权重保存格式]
    C --> D[选型与编排<br/>量化算法]
    D --> E[选择执行调度器]
    E --> F[编写配置<br/>并执行量化]
```

各阶段的关键细节如下：

前置条件：模型适配：量化前需先完成模型适配，适配器实现 PipelineInterface（基础流水线接口）等接口，详见[《DiT 模型接入量化流程指南》](./integration_guide_diffusion_transformer_quantization.md)。若目标模型已在官方预验证列表中，无需编写适配代码，直接指定对应 `--model_type` 即可。模型适配必须先完成，才能执行量化流程。

- **适配模型流水线接口**：确保待量化模型具备符合 `multimodal_sd_modelslim_v1` 规范的适配器接口，能够驱动浮点推理生成校准数据并被逐层量化调度器统一处理。
- **配置校准数据集与浮点推理参数**：量化校准依赖运行完整浮点去噪推理管线，通过 `dataset` 与 `multimodal_sd_config` 联合配置，推荐 50 条校准样本，分辨率、帧数与去噪步数对齐生产部署场景。
- **选择权重保存格式**：统一采用 `mindie_format_saver` 导出 MindIE-SD 格式，多专家模型自动按专家生成独立子目录。
- **选型与编排量化算法**：推荐以 W8A8 MXFP8（权重 `mse_round` 估计）为生产基线，低比特场景按需叠加 `online_quarot`、`fa3_quant` 等增强算法。
- **选择执行调度器（Runner）**：固定 `layer_wise` 逐层调度，不支持跨卡数据并行（`dp_layer_wise`）与整模型常驻（`model_wise`）。
- **编写配置并执行量化**：整合上述配置生成完整 YAML，通过 `msmodelslim quant` 命令启动量化，导出量化权重与描述文件。

## 4. 操作步骤

### 步骤 1：适配模型流水线接口（V1 Interface）

**目标**：确保待量化模型具备符合 `multimodal_sd_modelslim_v1` 规范的适配器接口，能够驱动浮点推理生成校准数据并被逐层量化调度器统一处理。

**操作**：

若目标模型已在官方支持列表中，无需编写适配代码，直接指定对应 `--model_type` 即可；若需接入新 DiT 模型，请参见[《DiT 模型接入量化流程指南》](./integration_guide_diffusion_transformer_quantization.md)。

该指南详细说明了 `MultimodalPipelineInterface` 规范、核心方法说明（`get_inference_config_class`、`configure_runtime`、`prepare_calib_data`、`inference_dump_calib_data`、`quantization_context`、`get_expert_adapter`）以及推理参数校验、运行态配置、浮点校准数据导出与量化执行上下文的适配器开发步骤。

**输出**：已在框架中注册可用的多模态模型适配器（通过 `--model_type` 调用）。

### 步骤 2：配置校准数据集与浮点推理参数

**目标**：配置代表性校准提示词与浮点推理管线参数，生成能够充分覆盖去噪过程与激活特征的校准数据。

**操作**：

1. **配置机制**：
   与 LLM 直接输入文本张量不同，DiT 模型的量化校准依赖运行**完整浮点去噪推理管线**，捕捉模型在不同 Timestep 下的层间激活张量。在 YAML 中通过 `dataset` 与 `multimodal_sd_config` 联合配置：

   ```yaml
   spec:
     dataset: <task_name>                 # 指定任务校准集标识（如 t2v、i2v 等）
     multimodal_sd_config:
       dump_config:
         enable_dump: true                # 是否在量化前运行浮点推理导出校准数据
         capture_mode: "args"
         dump_data_dir: ""                # 浮点校准数据保存路径，为空时保存在 save_path 下
       inference_config:
         size: "1280*720"                 # 生成图像/视频分辨率
         frame_num: 81                    # 生成帧数（视频模型专用）
         sample_steps: 40                 # 去噪步数
         task: "t2v-A14B"                 # 具体任务规格
   ```

2. **校准集与推理参数选择指南**：
   - **校准样本量**：**推荐 50 条**校准样本（如 50 组代表性文本 Prompt 或图文对）。样本过少难以覆盖多样的生成分布；样本过多会成倍增加浮点推理的耗时与存储开销。
   - **分辨率与帧数设置**：应与生产部署的主流场景保持一致（如视频生成模型常用 `1280*720`，帧数对齐部署标准）。如果分辨率设置偏离实际场景，可能导致局部特征激活分布估计不准。
   - **去噪步数（`sample_steps`）**：建议设为生产推理常用的步数（如 30~50 步）。Timestep 覆盖越完整，去噪轨迹统计越充分。
   - **跳过浮点 dump（纯动态量化场景）**：若执行不依赖离线激活统计的纯动态量化（如 Wan2.1 dynamic），可设置 `enable_dump: false`，跳过耗时的浮点去噪推理，直接进行权重量化。

**输出**：已在 YAML 中配置完成的 `dataset` 与 `multimodal_sd_config`。

### 步骤 3：选择权重保存格式

**目标**：根据下游多模态生成模型推理部署引擎的要求，选择匹配的量化权重导出格式。

**操作**：

1. **配置接口**：
   在 YAML 的 `spec.save` 列表中声明保存处理器：

   ```yaml
   spec:
     save:
       - type: mindie_format_saver        # DiT 模型部署格式
   ```

2. **格式选型与文档链接**：
   - **[mindie_format_saver 配置说明](../../../api_reference/config/format/mindie_format_saver.md)**：**DiT 模型部署核心格式**。面向昇腾 MindIE-SD 推理引擎，导出符合 MindIE 规范的量化描述文件及权重。针对多专家模型（如 Wan2.2），会自动在保存目录下生成 `high_noise_model/`、`low_noise_model/` 等各专家的独立子目录。

3. **选型指南**：
   - 当前多模态生成模型在昇腾平台推理部署时，均统一采用 `mindie_format_saver`。

**输出**：在 YAML 中配置完成的 `save` 格式。

### 步骤 4：选型与编排量化算法

**目标**：确定量化数值格式、位宽及优化算法，针对 DiT 独特的激活分布特征选择适配的量化策略。

**操作**：

1. **典型量化算法**：
   DiT 模型的计算瓶颈主要集中于主干 Transformer 的线性投影与注意力计算。针对生成类模型的常用低比特量化算法如下：

   | 算法 | 简述 | 适用场景 | 链接 |
   | --- | --- | --- | --- |
   | MX 格式量化（linear_quant） | `mxfp8`（E4M3/E5M2）或 `mxfp4` 数据类型，配合 `scope: per_block`（通常每 32 元素共享 scale），保持浮点高动态范围 | 昇腾 NPU 硬件优化的权重与激活量化 | [《linear_quant 配置说明》](../../../api_reference/config/processor/linear_quant.md) |
   | INT8 动态量化（linear_quant） | `dtype: int8`，激活 `per_token` 逐 Token 在线计算 scale（动态）、权重 `per_channel`，无需浮点 dump 校准数据 | 快速验证与轻量部署（如 Wan2.1） | [《linear_quant 配置说明》](../../../api_reference/config/processor/linear_quant.md) |
   | MinMax / MSE | 激活量化用 `minmax` 统计极值；权重量化推荐 `mse_round`，以均方误差最小化搜索舍入 | 提升低位宽下的权重保真度 | [《MinMax》](../../quantization_algorithms/minmax/usage_minmax.md) |
   | 在线正交旋转（QuaRot / online_quarot） | 在注意力自投影层引入正交变换矩阵，打散激活离群值通道 | 极端敏感层或低比特量化时的画质改善 | [《QuaRot》](../../quantization_algorithms/quarot/usage_quarot.md) |
   | FA3 注意力量化（fa3_quant） | 对自注意力 QK^T 与 PV 计算引入 `fp8_e4m3` 或 `mxfp4` 低比特执行 | 配合 W4A4 等低比特方案进一步压缩注意力计算开销 | [《fa3_quant 配置说明》](../../../api_reference/config/processor/fa3_quant.md) |
   | 按专家独立配置（per_expert） | 为敏感专家单独指定更保守的量化策略 | 多专家 DiT 中各专家误差容忍度差异较大时 | — |

   > 更多量化算法介绍请参考[《量化算法总览》](../../quantization_algorithms/README.md)。

2. **算法选型与调优**：
   - **W8A8 INT8 动态量化（轻量起点）**：激活 `per_token`、权重 `per_channel`，`dtype` 均为 `int8`。逐 Token 在线计算 scale，可设 `enable_dump: false` 跳过浮点推理 dump，适合快速验证与资源受限场景（可参考 `lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml`，其中排除了 `*ffn.2*` 等敏感层）。
   - **W8A8 MXFP8 量化（推荐首选）**：权重与激活均 `dtype: mxfp8`、`scope: per_block`，权重估计 `mse_round`、`symmetric: true`。在各大 DiT 模型上画质几乎无损，是推荐的生产基线。
   - **低比特与高精度增强**：追求更高压缩比时配置 `mxfp4`，并在自注意力层按需叠加 `online_quarot` 抑制离群值、`fa3_quant` 压缩注意力计算。
   - **单变量调参**：先以标准 W8A8 MXFP8 为基线；画质轻微劣化时优先检查浮点 dump 校准数据质量，或对局部专家/敏感层配置回退，避免盲目组合多个处理器。

**输出**：在 YAML 中编排完毕的 `process` 及可选的 `per_expert` 算法链。

### 步骤 5：选择执行调度器（Runner）

**目标**：明确量化执行流水线的调度模式与硬件约束。

**操作**：

1. **接口原理与调度机制**：
   在 YAML 的 `spec.runner` 中指定（源码位于 `msmodelslim/core/runner/`）：

   ```yaml
   spec:
     runner: layer_wise                   # DiT 模型固定使用 layer_wise
   ```

   - **`LayerWiseRunner`（逐层量化调度器）**：逐层加载模型参数至 NPU，利用捕获到的激活张量完成当前层的统计与量化，量化完成后立即卸载释放显存。整机显存占用仅受限于单层权重与当前 Batch 的激活大小。
   - **调度约束说明**：在多模态生成模型量化服务（`multimodal_sd_modelslim_v1`）中，由于校准数据需按专家完整 dump，**目前不支持跨卡分布式数据并行（`dp_layer_wise`），也不支持整模型常驻（`model_wise`）**。即使配置了其他 runner 类型或传入多卡，量化服务也会统一转换为单卡 `layer_wise` 模式执行。

2. **配置指南**：
   - **固定配置 `runner: layer_wise`**。
   - 执行前请确保单张 NPU 具备容纳最大单个网络层参数及前向临时激活的显存空间。

**输出**：YAML 中确认的 `runner` 调度策略。

### 步骤 6：编写量化配置并执行命令

**目标**：整合上述步骤编写完整的 YAML 量化配置文件，并通过 CLI 启动量化。有关量化命令参数的完整说明，请参阅[《权重量化使用指南》](../../../user_guide/usage_weight_quantization.md)。

**操作**：

#### 完整示例 1：W8A8 MXFP8 静态量化（推荐起点，生成画质与性能均衡）

##### 配置文件：`dit_w8a8_mxfp8.yaml`

```yaml
apiversion: multimodal_sd_modelslim_v1
spec:
  runner: layer_wise                      # DiT 采用单卡逐层量化
  process:
    - type: linear_quant
      qconfig:
        act:
          dtype: mxfp8                    # 激活采用 MXFP8 格式
          scope: per_block                # 块级粒度统计
          symmetric: true
          method: minmax                  # 极值统计
        weight:
          dtype: mxfp8                    # 权重采用 MXFP8 格式
          scope: per_block
          symmetric: true
          method: mse_round               # 均方误差舍入优化，保真度更高
      include: ["*"]                      # 量化全部主干线性层
  save:
    - type: mindie_format_saver           # 导出 MindIE-SD 格式
  dataset: <task_name>                    # 任务标识（如 t2v 等）
  multimodal_sd_config:
    dump_config:
      enable_dump: true                   # 运行浮点推理生成校准数据
      capture_mode: "args"
      dump_data_dir: ""                   # 默认回退至 save_path
    inference_config:
      size: "1280*720"                    # 生成分辨率
      frame_num: 81                       # 生成帧数
      sample_steps: 40                    # 去噪步数
      task: "t2v-A14B"                    # 任务规格
```

##### 执行命令（单卡量化）

```bash
msmodelslim quant \
  --model_path <浮点模型目录> \
  --save_path <量化权重输出目录> \
  --model_type <模型适配器名称> \
  --config_path ./dit_w8a8_mxfp8.yaml \
  --device npu:0
```

#### 完整示例 2：W8A8 INT8 动态量化（轻量起点，无需浮点 dump）

##### 配置文件：`dit_w8a8_int8_dynamic.yaml`

```yaml
apiversion: multimodal_sd_modelslim_v1
spec:
  runner: layer_wise                      # DiT 采用单卡逐层量化
  process:
    - type: linear_quant
      qconfig:
        act:
          dtype: int8                     # 激活 INT8 量化
          scope: per_token                # 动态激活量化，逐 Token 在线计算 scale
          symmetric: true
          method: minmax
        weight:
          dtype: int8                     # 权重 INT8 量化
          scope: per_channel              # 每通道独立 scale
          symmetric: true
          method: minmax
      include: ["*"]                      # 量化全部主干线性层
  save:
    - type: mindie_format_saver           # 导出 MindIE-SD 格式
      part_file_size: 0
  dataset: <task_name>                    # 任务标识（如 t2v 等）
  multimodal_sd_config:
    dump_config:
      enable_dump: false                  # 纯动态量化不依赖离线激活统计，跳过浮点推理 dump
      capture_mode: "args"
      dump_data_dir: ""
    inference_config:
      size: "1280*720"                    # 生成分辨率
      frame_num: 81                       # 生成帧数
      sample_steps: 40                    # 去噪步数
      task: "t2v-A14B"                    # 任务规格
```

##### 执行命令（单卡量化）

```bash
msmodelslim quant \
  --model_path <浮点模型目录> \
  --save_path <量化权重输出目录> \
  --model_type <模型适配器名称> \
  --config_path ./dit_w8a8_int8_dynamic.yaml \
  --device npu:0
```

**输出**：在指定的 `--save_path` 目录下生成完整的 MindIE-SD 格式量化权重及描述文件（多专家模型自动分专家目录存放）。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| DiT 量化 | Diffusion Transformer 多模态生成模型训练后量化 | [《DiT 量化词条》](./term_diffusion_transformer_quantization.md) |
| PTQ | 训练后量化（Post-Training Quantization） | [《训练后量化词条》](../term_ptq.md) |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim quant` | 一键量化 CLI 入口与完整命令说明 | [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md) |
| multimodal_sd_modelslim_v1 配置说明 | `runner`/`process`/`per_expert`/`save`/`dataset`/`multimodal_sd_config` 等任务级配置的字段说明 | [《multimodal_sd_modelslim_v1 配置说明》](../../../api_reference/config/task/multimodal_sd_modelslim_v1.md) |
| linear_quant 配置说明 | `linear_quant` 处理器及其 `qconfig` 各字段的完整取值说明 | [《linear_quant 配置说明》](../../../api_reference/config/processor/linear_quant.md) |
| 多模态生成模型接入 | DiT 模型接入与适配器开发 | [《多模态生成模型接入》](./integration_guide_diffusion_transformer_quantization.md) |
