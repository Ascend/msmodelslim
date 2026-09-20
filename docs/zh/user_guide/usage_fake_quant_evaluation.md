# 伪量化推理使用指南

## 1. 适用范围

本指南面向已通过 msModelSlim 产出量化权重，需要**进行 bad case 等小样本场景的精度验证**的开发者与算法工程师。

**适用场景**：

- 已通过 `msmodelslim quant` 生成量化导出件，需要独立于推理引擎做小样本的精度验证。
- 支持大语言模型（LLM）和多模态理解模型（VLM）。
- 当前仅支持 AscendV1 格式的量化权重。

**不适用场景**：

- 伪量化推理当前不支持多模态生成模型（DiT）。
- 需要进行大规模样本的精度测试，如数据集精度验证。由于伪量化推理使用 torch 小算子进行前向推理，且逐层加载权重进行前向推理，推理性能不适用于大规模样本的精度测试。

## 2. 流程关系与前置条件

**上级流程**：《[量化推理精度异常定位](process_quantization_accuracy_anomaly_locating.md)》。

**前置条件**：

- 已使用一键量化生成量化导出件。
- 已按模型发布页与对应接入指南准备运行依赖：不同模型常要求特定 `transformers` 等版本（与一键量化功能要求一致）。
- 已确定要验证的输入样本（如数据集测试中的 bad case）。

**后续操作**：

- 验证通过：量化权重可进入部署与测评环节。
- 验证不通过：按照《[量化精度调优指南](process_quantization_precision_tuning.md)》调整量化方案后重新量化，再回到本流程复验。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 量化导出件 | 量化权重保存目录 | 目录可读 | 量化导出件完整 |
| 输入 | 推理样本文件 | 用户本地路径 | 扩展名为 `.json` 或 `.jsonl` | `msmodelslim eval` 可解析，样本非空且格式符合对应模型约定 |
| 交付件 | 伪量化推理结果 | 标准输出（日志） | 逐样本显示推理结果 | 样本数与输入一致，输出顺序与输入顺序对应 |

## 4. 流程总览

先确认模型与导出格式在支持范围内，按需实现适配器的推理接口，再准备推理样本，随后执行伪量化推理，最后解读推理结果。

```mermaid
flowchart LR
  scope[确认模型与导出格式支持] --> adapt[确认适配器实现推理接口]
  adapt --> prompt[准备推理样本]
  prompt --> run[执行伪量化推理]
  run --> read[解读推理结果]
```

## 5. 操作步骤

### 步骤 1：确认模型与导出格式支持

**目标**：确认待验证模型属于支持范围，且权重是已支持的量化格式。

**操作**：

1. 确认模型类型：模型已接入 msmodelslim，`model_type` 已注册。
2. 确认量化格式：当前仅支持 **AscendV1** 量化格式（由 `ascendv1_saver` 导出）。
3. 确认量化类型已支持：伪量化推理目前支持以下量化类型。

| 量化类型 | 支持状态 | 说明 |
|----------|----------|---------|
| w8a8 | ✅ 支持 | 支持 INT8、MXFP8 |
| c8 | ✅ 支持 | 支持 FA 量化与 KVCache 量化 |

**输出**：选定的 `${MODEL_TYPE}` 与确认可用的量化导出目录 `${MODEL_PATH}`。

**通过条件**：使用 `--model_type ${MODEL_TYPE}` 启动伪量化推理时，日志显示命中预期模型适配器，且无未知模型或未命中适配器相关报错。

### 步骤 2：确认适配器实现推理接口

**目标**：确保 `${MODEL_TYPE}` 对应的模型适配器实现了 [`FakeQuantInferenceInterface`](../../../msmodelslim/core/infer_engine/interface.py)。

**操作**：

- 确认适配器是否已实现该接口：在 `msmodelslim/model/` 下找到对应模型适配器并查看类定义。
- 若未实现，按照以下说明实现该接口，并在完成后于仓库根目录重新执行 `bash install.sh` 使代码修改生效

  接口方法：

  | 方法 | 职责 |
  | --- | --- |
  | `handle_dataset` | 将推理样本转换为模型输入 |
  | `build_meta_model` | 构建浮点模型结构（不读取权重）|

  实现说明：

  - `handle_dataset` 与一键量化所用的 [`PipelineInterface`](../../../msmodelslim/core/runner/pipeline_interface.py) 中同名方法定义一致，实现一次即可复用。
  - `build_meta_model` 构建浮点模型结构，后续内部引擎自动基于该结构重建量化结构。由于输入并不提供浮点权重，因而在实现该方法时须避免加载权重的模型构建方式。以[`qwen3/model_adapter.py`](../../../msmodelslim/model/qwen3/model_adapter.py)为例：

  ```python
  class Qwen3ModelAdapter(DefaultModelAdapter, FakeQuantInferenceInterface, ...):
      def build_meta_model(self) -> nn.Module:
          try:
            from transformers import AutoModelForCausalLM
          except ImportError as exc:
            raise InvalidModelError(
                "Failed to import AutoModelForCausalLM for fake-quant inference shell",
                action="Please install a transformers version that provides AutoModelForCausalLM.",
            ) from exc
          origin_layers = int(self.config.num_hidden_layers)
          if hasattr(self.config, "use_cache"):
              self.config.use_cache = False
          from accelerate import init_empty_weights

          with init_empty_weights(include_buffers=False):
              model = AutoModelForCausalLM.from_config(
                  self.config,
                  trust_remote_code=self.trust_remote_code,
              )

          get_logger().info(
              "Built Qwen3 fake-quant inference shell: %d decoder layers (meta params, CPU non-persistent buffers)",
              origin_layers,
          )
          return model
  ```

  基于 `accelerate` 库的 `init_empty_weights` 创建模型时，仅构建模型结构而不加载权重，注意指定 `include_buffers=False`。当 `include_buffers=True` 时，buffer 类参数（如 RoPE 的 `inv_freq`）将在 meta 上创建，使得 RoPE 等运行时缓冲变成无效值，表现为推理结果静默异常。由于伪量化推理内部通过多轮 prefill 模拟 decode，实际并不维护 KVCache，因而需指定 `self.config.use_cache = False`。

**输出**：已注册且实现 `FakeQuantInferenceInterface` 的适配器。

**通过条件**：使用该 `--model_type` 启动时能正确命中适配器，无接口不支持相关报错。

### 步骤 3：准备推理样本

**目标**：准备待推理验证的样本文件。

**操作**：

- **LLM**：扩展名为 `.json` 或 `.jsonl`。

  ```json
  [
    "Where is the capital of China?",
    "I want to learn about Deep Learning. Please tell me more about it."
  ]
  ```

- **VLM**：命名须为 `index.json` 或 `index.jsonl`。

  index.json 示例：

  ```json
  [
    {"image": "COCO_train2014_000000577589.jpg", "text": "How many people in the picture? "},
    {"image": "COCO_train2014_000000577592.jpg", "text": "Describe the picture. "}
  ]
  ```

**输出**：推理样本文件 `${PROMPT_FILE}`。

**通过条件**：文件非空，格式与模型类型匹配，扩展名为 `.json` 或 `.jsonl`。

### 步骤 4：执行伪量化推理

**目标**：在目标设备上跑通推理，得到逐样本生成结果。

**执行前检查**：

- 已完成步骤 1～3 的格式、导出件、样本与适配确认。
- 多卡并行仅在 `--device npu` 下通过 `--device_id` 指定多个设备索引开启。

**操作**：

按《[msmodelslim eval 命令行 API](../api_reference/cli/msmodelslim_eval.md)》将变量替换为实际值后执行。多卡时追加多个设备索引，例如：

```bash
msmodelslim eval \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --prompt_file ${PROMPT_FILE} \
  --device npu \
  --device_id 0 1 2 3 \
  --max_new_tokens 5
```

**输出**：标准输出中的逐样本 `token_ids` 与 `generated_text`；多卡时每条结果附带 rank 与 device 归属。

**通过条件**：命令正常结束，样本数与输入一致，结果顺序与输入顺序一一对应。

**审计记录**：实际命令行、`${MODEL_TYPE}`、量化导出件路径与量化配置、样本文件路径、关键日志（含各样本生成结果）。

### 步骤 5：解读推理结果

**目标**：读懂日志输出中的逐样本推理结果，判断伪量化模型的生成内容是否符合预期。

**操作**：

1. 阅读日志输出，确认样本规模与并行方式。单卡与多卡的结果格式如下：

   ```text
   Fake-quant inference results (1 sample(s), 8 device(s)):
   Fake-quant sample[0] (rank=0, device=0) token_ids=[29660, 344, 270, 6102, 294] generated_text=' Beijing is the capital of'
   ```

2. 判断结果是否符合预期：

   - **数量与顺序**：结果条数与 `${PROMPT_FILE}` 中的样本条数一致，`sample` 序号连续递增且内容与输入样本一一对应。
   - **语义**：生成文本应通顺且与输入语义相符。示例中问题为 `Where is the capital of China?`，输出 `' Beijing is the capital of'` 已答出 `Beijing`，句中被截断属预期行为，由 `--max_new_tokens` 控制；需要更完整的回答时调大该参数后重跑。

**输出**：逐样本的 `token_ids` 与 `generated_text`。

**通过条件**：结果条数与顺序同输入样本一致。

## 6. 验收条件

- 量化导出件在目标设备上完成推理，逐样本推理结果可判读。

## 7. 异常处置

| 现象 | 处理方向 |
| --- | --- |
| 提示适配器未实现所需接口 | 按步骤 2 实现 `FakeQuantInferenceInterface` 并重新执行 `bash install.sh` 后重试 |
| 导出件无法加载 | 核对 `--model_path` 是否为完整的量化导出目录，当前仅支持 AscendV1 格式 |
| 首 token 解码错误 | 核对 transformers 版本，与量化要求的版本保持一致 |

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| AscendV1 量化格式 | 当前支持的量化导出格式 | 《[AscendV1](../knowledge_base/quantization_format/ascendv1/term_ascendv1.md)》 |
| 数据并行（DP） | 多卡时按样本切分到各卡并行推理的方式 | 《[数据并行](../knowledge_base/parallel/data_parallelism/term_data_parallelism.md)》 |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim eval` | 伪量化推理验证命令行入口 | 《[msmodelslim eval 命令行 API](../api_reference/cli/msmodelslim_eval.md)》 |
