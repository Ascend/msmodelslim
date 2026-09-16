# 大语言模型（LLM）量化使用指南

## 1. 适用范围

本指南面向对 Decoder-only 架构大语言模型（如 LLaMA、Qwen、GLM、DeepSeek 等）执行[训练后量化（PTQ）](../term_ptq.md)的用户。文档提供规范的量化使用流程，系统阐明模型适配接口、校准集配置、保存格式、算法选型、调度器（Runner）机制，并提供完整的 W8A8 动态量化与静态量化开箱即用示例。

适用场景：

- 将浮点 LLM 权重高效量化为 W8/W4 等低比特格式并导出；
- 为在线推理服务（如基于昇腾 NPU 的 MindIE 或 vLLM-Ascend）准备部署权重。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 可被适配器 `handle_dataset` 成功编码为前向张量 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors` 分片 | 导出完整且推理冒烟测试通过 |

## 3. 流程总览

LLM 量化的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线接口] --> B[配置校准数据集]
    B --> C[选择权重保存格式]
    C --> D[选型与编排<br/>量化算法]
    D --> E[选择执行调度器]
    E --> F[编写配置<br/>并执行量化]
```

各阶段的关键细节如下：

前置条件：模型适配：量化前需先完成模型适配，适配器实现 PipelineInterface（基础流水线接口）等接口，详见[《LLM 模型接入量化流程指南》](./integration_guide_large_language_model_quantization.md)。若目标模型已在官方预验证列表中，无需编写适配代码，直接指定对应 `--model_type` 即可。模型适配必须先完成，才能执行量化流程。

- **适配模型流水线接口**：确保待量化模型具备符合 `modelslim_v1` 规范的适配器接口，能够被调度器（Runner）与处理器（Processor）统筹驱动。
- **配置校准数据集**：通过 `spec.dataset` 指定校准数据，通用场景推荐内置 `mix_calib.jsonl`，样本量推荐 50 条、序列长度建议 2048~4096，为激活统计提供充分依据。
- **选择权重保存格式**：在 `spec.save` 中声明保存处理器，昇腾 NPU 部署场景推荐 `ascendv1_saver`，务必与部署框架版本对齐。
- **选型与编排量化算法**：在 `spec.process` 中按声明顺序编排算法链，推荐以 W8A8 动态量化（`minmax` 估计）为起点，精度不足时再叠加平滑或旋转算法。
- **选择执行调度器（Runner）**：在 `spec.runner` 中指定调度策略，推荐 `auto`：单卡自动逐层调度以降低显存峰值，多卡自动数据并行加速。
- **编写配置并执行量化**：整合上述配置生成完整 YAML，通过 `msmodelslim quant` 命令启动量化，导出量化权重与描述文件。

## 4. 操作步骤

### 步骤 1：适配模型流水线接口（V1 Interface）

**目标**：确保待量化模型具备符合 `modelslim_v1` 规范的适配器接口，能够被调度器（Runner）与处理器（Processor）统筹驱动。

**操作**：

若目标模型已在官方支持列表中，无需编写适配代码，直接指定对应 `--model_type` 即可；若需接入新模型或扩展量化接口，请参见[《LLM 模型接入量化流程指南》](./integration_guide_large_language_model_quantization.md#步骤-4实现流水线核心接口方法)。

该指南详细说明了 `PipelineInterface` 规范、核心方法说明（`handle_dataset`、`init_model`、`generate_model_visit`、`generate_model_forward`、`enable_kv_cache`）以及适配器注册与开发步骤。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：配置校准数据集

**目标**：选取并配置具有代表性的校准数据集，为激活离群值检测与量化参数估计提供充分统计依据。

**操作**：

1. **配置方式与解析机制**：
   在量化 YAML 配置中通过 `spec.dataset` 指定：

   ```yaml
   spec:
     dataset: mix_calib.jsonl   # 相对文件名或绝对路径
   ```

   底层由 `FileDatasetLoader` 处理：若为相对文件名（如 `mix_calib.jsonl`），自动寻址工程内置的 `lab_calib/` 目录；若为包含路径或绝对路径，则直接载入指定文件。文件格式支持 `.jsonl` 或 `.json`，每条样本为包含文本提示词的字符串对象。

2. **校准集选择指南与推荐配置**：
   - **通用模型量化（推荐）**：默认采用内置的 `mix_calib.jsonl`。该数据集涵盖通用问答、代码、多语言问答及多步数学推理，分布均衡鲁棒。
   - **特定垂直场景微调模型**：若模型专门面向特定领域（如专业医学问答、代码补全、长合同解析），推荐准备业务真实分布数据集。建议采取“80% 业务真实数据 + 20% 通用数据”混合配比，兼顾领域精度与通用推理能力。
   - **校准样本量（条数）**：**推荐 50 条**。样本过少时激活值统计可能不充分，容易遗漏偶发离群值导致参数估计偏差；样本过多时激活特征已基本收敛，但会线性增加校准耗时与显存消耗。
   - **序列长度**：建议在 Tokenizer 阶段将截断长度设为 **2048 ~ 4096**（与部署上下文对齐），保证长上下文激活特征的充分捕获。

**输出**：已确定并配置在 YAML 中的校准数据集路径与样本集。

### 步骤 3：选择权重保存格式

**目标**：根据下游推理部署框架的要求，选择匹配的量化权重组织与导出格式。

**操作**：

1. **配置接口**：
   在 YAML 的 `spec.save` 列表中声明保存处理器：

   ```yaml
   spec:
     save:
       - type: ascendv1_saver       # 保存格式类型
   ```

2. **格式选型与文档链接**：
   - **[ascendv1_saver 配置说明](../../../api_reference/config/format/ascendv1_saver.md)**：**昇腾 NPU 部署推荐格式**。专为昇腾加速库（ATB / Ascend Transformer Boost）与 MindIE 推理引擎设计，生成包含量化元数据描述的 `quant_model_description.json` 及分片量化权重 `*.safetensors`，自动完成硬件指令所要求的权重重排（Pack）与反量化系数预处理。
   - **[mindie_format_saver 配置说明](../../../api_reference/config/format/mindie_format_saver.md)**：面向新版本 MindIE-SD/MindIE 生态的统一部署格式。

3. **选型指南**：
   - 当前在昇腾平台上运行 vLLM-Ascend、ATB-Models 或 MindIE-Service 时，均推荐首选 `ascendv1_saver`。
   - 务必与部署框架版本对齐，保存格式不匹配会导致加载权重报错。

**输出**：在 YAML 中配置完成的 `save` 格式。

### 步骤 4：选型与编排量化算法

**目标**：确定量化位宽、颗粒度，并按需编排抗离群预处理算法链，取得精度与性能的最佳平衡。

**操作**：

1. **典型量化算法**：
   量化流程通过处理器链（`spec.process`）按声明顺序串行调度。常用算法如下：

   | 算法 | 简述 | 适用场景 | 链接 |
   | --- | --- | --- | --- |
   | MinMax | 统计张量最大/最小值直接计算 scale 与 zero-point，开销极低 | W8A8 动静态量化的默认估计算法 | [《MinMax》](../../quantization_algorithms/minmax/usage_minmax.md) |
   | SmoothQuant | 将激活中的离群值按通道缩放迁移到权重，降低激活量化难度 | 恢复 W8A8 静态量化的精度 | [《SmoothQuant》](../../quantization_algorithms/smooth_quant/usage_smooth_quant.md) |
   | QuaRot | 引入正交 Walsh-Hadamard 旋转矩阵，在不改变前向输出的前提下打散激活离群通道 | 激活离群严重、平滑算法收益不足时 | [《QuaRot》](../../quantization_algorithms/quarot/usage_quarot.md) |
   | KVCache Quant | 对 Key/Value 缓存张量做低比特量化 | 长序列多轮对话的显存与吞吐优化 | [《KVCache Quant》](../../quantization_algorithms/kvcache_quant/usage_kvcache_quant.md) |

   > 完整支持的离群值抑制算法、量化优化算法及敏感层分析算法，请参阅[《量化算法总览》](../../quantization_algorithms/README.md)。

2. **算法选型与调优**：
   - **W8A8 动态量化（推荐起点）**：激活 `scope: per_token`、权重 `scope: per_channel`、`symmetric: true`、估计方法 `minmax`。逐 Token 动态计算 scale，对离群点适应性好，精度接近浮点基线，无需叠加平滑算法。
   - **W8A8 静态量化（高吞吐优先）**：激活 `scope: per_tensor`，推理无需在线统计 scale，吞吐更高；精度明显退化时叠加 `smooth_quant` 处理器。
   - **单变量调参**：先跑 `minmax` 基线，指标未达验收条件时再按需引入平滑或旋转算法，避免盲目叠加。

**输出**：在 YAML 中编排完毕的 `process` 算法链。

### 步骤 5：选择执行调度器（Runner）

**目标**：根据可用硬件资源（单卡/多卡）及模型参数规模，选择最优的执行调度引擎，防止显存溢出并最大化量化执行速度。

**操作**：

1. **接口原理与调度机制**：
   在 YAML 的 `spec.runner` 中指定（源码位于 `msmodelslim/core/runner/`）：

   ```yaml
   spec:
     runner: auto   # auto | layer_wise | dp_layer_wise | model_wise
   ```

   - **`LayerWiseRunner`（逐层量化调度器）**：在各网络层前后动态挂载 `LoadProcessor`。仅在执行到当前 Decoder Layer 时将其自 CPU/meta 加载至目标 NPU，处理完成后立刻卸载释放。整机显存峰值仅取决于单层参数与当前 Batch 激活，单卡即可完成 70B 及以上超大模型的量化。
   - **`DPLayerWiseRunner`（数据并行逐层调度器）**：继承自 `LayerWiseRunner`，结合 `torch.distributed` 与 `DistributedSampler` 将校准数据均匀切分到各张卡，多卡并行计算单层前向并同步聚合统计量。
   - **`PPRunner` / `model_wise`（整模型调度器）**：整模型常驻显存，仅适用于参数量极小或单卡显存充裕的开发调试。
   - **`auto`（自适应调度，推荐）**：单卡执行时自动调度 `layer_wise`；检测到多设备（`device_indices > 1`）时自动调度 `dp_layer_wise`。

2. **配置指南**：
   - **推荐默认填写 `runner: auto`**：兼顾单卡稳定性与多卡自动并行加速。
   - **大规模校准提速**：当校准集较大或希望缩短耗时，在命令行传入多卡（如 `--device npu:0,1,2,3`），配合 `auto` 即可自动启用多卡并行。

**输出**：YAML 中确认的 `runner` 调度策略。

### 步骤 6：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。有关量化命令参数的完整说明，请参阅[《权重量化使用指南》](../../../user_guide/usage_weight_quantization.md)。

**操作**：

#### 完整示例 1：W8A8 动态量化（推荐起点，精度与易用性均衡）

##### 配置文件：`w8a8_dynamic.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto                    # 单卡自动使用 layer_wise，多卡自动使用 dp_layer_wise
  process:
    - type: linear_quant
      qconfig:
        act:
          dtype: int8
          scope: per_token        # 动态激活量化（每 Token 独立 scale），精度表现好
          symmetric: true
          method: minmax
        weight:
          dtype: int8
          scope: per_channel      # 权重量化（每通道独立 scale）
          symmetric: true
          method: minmax
      include: ["*"]              # 匹配量化所有线性层
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
  --config_path ./w8a8_dynamic.yaml \
  --device npu:0
```

#### 完整示例 2：W8A8 静态量化（推理吞吐优先）

##### 配置文件：`w8a8_static.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  process:
    - type: linear_quant
      qconfig:
        act:
          dtype: int8
          scope: per_tensor       # 静态激活量化（每层张量共享 scale）
          symmetric: true
          method: minmax
        weight:
          dtype: int8
          scope: per_channel      # 权重量化（每通道独立 scale）
          symmetric: true
          method: minmax
      include: ["*"]
  save:
    - type: ascendv1_saver
  dataset: mix_calib.jsonl
```

##### 执行命令（单卡量化）

```bash
msmodelslim quant \
  --model_path <浮点模型目录> \
  --save_path <量化权重输出目录> \
  --model_type <模型适配器名称> \
  --config_path ./w8a8_static.yaml \
  --device npu:0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| PTQ | 训练后量化（Post-Training Quantization） | [《训练后量化词条》](../term_ptq.md) |
| LLM 量化 | Decoder-only 大语言模型训练后量化 | [《LLM 量化词条》](./term_large_language_model_quantization.md) |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim quant` | 一键量化 CLI 入口与完整命令说明 | [《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md) |
| modelslim_v1 配置说明 | `runner`/`process`/`save`/`dataset` 等任务级配置的字段说明 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
