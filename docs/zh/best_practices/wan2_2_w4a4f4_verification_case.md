# Wan2.2 W4A4F4量化验证案例

## 1. 案例背景

本案例基于 MindIE-SD 推理栈，对 Wan2.2 系列双专家 DiT 文生视频模型进行 W4A4F4 量化验证，对比原始浮点权重与量化后权重的生成质量。验证覆盖从逐层量化、权重导出到推理服务部署与评测的完整流程。

**验证目标**：

- **精度**：相对原始浮点权重，量化后模型生成视频质量损失控制在可接受范围内，无肉眼可识别的严重画质崩坏；且在 VBench-1.0-mini 的 1% 子集上，VBench Quality 与 VBench Semantic 的**绝对得分下降均小于 1 个百分点（pp）**，即 `FP 得分 − 量化得分 < 1 pp`。

**关联流程**：《[多模态生成模型接入指南](../knowledge_base/model/integrating_multimodal_generation_model.md)》、《[一键量化使用说明](../user_guide/usage_quick_quantization.md)》

## 2. 模型相关信息

| 项                 | 内容 |
| ------------------ | ------------------------------------ |
| 模型名称            | Wan2.2（阿里通义万相2.2） |
| model_type         | Wan2.2-T2V-A14B |
| 参数量              | 14B（A14B版本） |
| 模态                | 多模态生成（文生视频） |
| 模型结构            | 双专家DiT（low_noise_model + high_noise_model）+ GQA注意力 + 文本编码器 + VAE |
| 量化类型            | w4a4f4 |
| 开源权重来源         | [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) 等官方开源地址 |
| 量化权重 md5        | low_noise_model: `08e2b4c5c759c46903beaf3c4c4a9d14`；high_noise_model: `e7c17613d71cfb408aeab1fd96a98132` |
| 推理框架与并行 | MindIE-SD；本案例并行策略：DiT/T5 启用 FSDP 切分、序列维度启用 Ulysses Sequence Parallel（SP）、CFG 双卡分离、VAE 启用并行；未启用 Expert Parallel（双专家由框架自动调度）。对比双方使用完全相同的并行约定，具体取值见 §5 步骤 4/5。 |

**验证范围声明**：

- 覆盖：DiT主干网络双专家W4A4量化、文生视频生成质量对比
- 不覆盖：文本编码器量化、VAE量化、语音输出、超过81帧/720P以上分辨率长视频、特殊风格微调版本

## 3. 环境与版本

| 项          | 版本或配置 |
| ----------- | -------------------------------------------------- |
| 产品形态     | Ascend_950（限定：浮点推理限定硬件形态） |
| 推理仓库     | 魔乐社区Wan2.2模型推理仓库，量化时推荐使用版本`git checkout 38fb8eb13a018cca316678930720eafa446c4387`，推理时推荐最新版本，仓库参见《[Wan2.2推理仓](https://modelers.cn/models/MindIE/wan2.2)》|
| CANN        | 9.1.0（与推理仓库版本一致） |
| PyTorch     | 2.9.0（与推理仓库版本一致） |
| TorchNPU    | 2.9.0（与推理仓库版本一致） |
| MindIE-SD   | 3.1.0（与推理仓库版本一致） |
| 评测工具     | [AISBench](https://github.com/AISBench/benchmark) @ `c3aa4512090d02ba048ecaa51e66294a42d49d4a` |
| 其他依赖     | Wan2.2-T2V-A14B 模型权重、[VBench-1.0-mini 评测数据集](https://modelers.cn/datasets/AISBench/VBench-1.0-mini) |

**本次前置事实**：

- Wan2.2官方推理仓浮点模型可正常加载，720P/81帧视频生成质量符合官方预期。
- msModelSlim已正确安装，可识别Wan2.2系列model_type。

## 4. 输入和交付件

| 类型  | 名称                   | 来源或保存位置                  | 格式或约束                      | 验收方式                              |
| --- | -------------------- | ------------------------ | -------------------------- | --------------------------------- |
| 输入  | Wan2.2浮点模型权重       | `${MODEL_PATH}`         | 官方开源格式，含DiT双专家权重、VAE、文本编码器、tokenizer、config.json | 可通过官方推理仓正常加载，720P视频生成无异常 |
| 交付件 | Wan2.2 W4A4F4量化权重 | `${QUANT_PATH}` | msModelSlim量化导出格式，含`quant_model_description.json`、双专家量化权重 | 可被MindIE-SD正常加载，成功生成视频 |
| 输入  | 精度评测集       | `${EVAL_DATA_PATH}`         | VBench标准评测集 | 对比双方使用完全相同的数据和随机种子       |
| 交付件 | 精度报告 | `./outputs/default/{timestamp}/` | 生成视频样例（FP vs 量化对比）、VBench等客观指标结果表 | 满足预设验收阈值，无严重画质问题 |

## 5. 操作步骤（环境与服务）

### 步骤 1：环境准备与路径配置

**目标**：设置环境变量，核对依赖版本，确认模型和数据路径正确。  
**输入**：模型路径、数据路径、输出路径。  
**操作**：配置环境变量，执行版本核对命令。  

```bash
# 设置环境变量（替换为实际路径）
export MODEL_PATH=/path/to/Wan2.2-T2V-A14B
export QUANT_PATH=/path/to/output/wan22_w4a4f4
export EVAL_DATA_PATH=/path/to/eval/final_mini_dataset_0_01  # 评测数据集路径，具体可见下方步骤 3
export OUTPUT_DIR=/path/to/output/wan22_verification
export PYTHONPATH=/path/to/Wan2.2:$PYTHONPATH  # 推理仓库路径

# 核对环境版本（必须在验证环境实际执行）
npu-smi info
python -V
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch_npu; print('TorchNPU:', torch_npu.__version__)"
pip show msmodelslim mindiesd
```

**输出**：环境变量配置完成，所有依赖版本核对记录在案。  
**记录**：CANN版本、PyTorch/TorchNPU版本、msmodelslim版本、NPU型号与驱动。  

### 步骤 2：执行Wan2.2 W4A4F4量化

**目标**：运行msModelSlim量化流程，完成双专家逐层量化，导出量化权重。  
**输入**：浮点模型、校准数据集、量化配置。  
**操作**：使用对应model_type执行W4A4F4量化。  

```bash
# T2V文生视频量化示例
msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${QUANT_PATH} \
    --device npu \
    --model_type Wan2.2-T2V-A14B \
    --quant_type w4a4f4 \
    --trust_remote_code True

```

**输出**：量化权重保存至`${QUANT_PATH}`目录，包含双专家量化权重与描述文件。  
**记录**：量化过程完整日志、量化总时长、各层量化状态。  

### 步骤 3：准备VBench-1.0-mini评测子集

**目标**：从VBench-1.0-mini原始数据集中整理出0.01子集，供推理脚本使用。  
**输入**：已下载的VBench-1.0-mini原始数据集。  
**操作**：目录结构调整与文件重命名。  

VBench-1.0-mini原始目录结构如下：

```text
VBench-1.0-mini/
├── VBench_kmeans_info.json              # 全量 mini，91 条 prompt
├── VBench_kmeans_info_0.01.json         # 11 条 prompt（0.01 子集）
├── VBench_kmeans_info_0.05.json         # 43 条 prompt
├── VBench_kmeans_info_0.10.json         # 同全量 mini
└── README.md
```

> [!NOTE]
>
> 推理脚本 `vbench.py` 中 JSON 文件名写死为 `VBench_kmeans_info.json`，因此 `--vbench_mini_root` 指向的目录下必须存在该名称的文件。

目标是将0.01子集整理为以下结构：

```text
final_mini_dataset_0_01/
└── VBench_kmeans_info.json          # 由 VBench_kmeans_info_0.01.json 改名而来
```

执行以下命令完成整理（使用 `cp` 保留原始数据集不动）：

```bash
mkdir -p final_mini_dataset_0_01
cp VBench-1.0-mini/VBench_kmeans_info_0.01.json \
   final_mini_dataset_0_01/VBench_kmeans_info.json
```

完成后推理传参 `--vbench_mini_root ./final_mini_dataset_0_01` 即可使用0.01子集。该子集包含11条prompt，覆盖11个维度，按每条prompt生成1个视频计算共11个视频。

**输出**：`final_mini_dataset_0_01/` 目录准备完毕，可直接传入 `--vbench_mini_root`。  
**记录**：无。  

### 步骤 4：执行Wan2.2浮点模型VBench评测推理

**目标**：运行浮点模型推理，在VBench-mini数据集上生成评测结果作为精度对比基线，该步骤只生成视频。  
**输入**：浮点模型权重、VBench-mini评测数据集、推理超参配置。  
**操作**：本案例中具体取值为：ulysses_size=2、dit_fsdp + t5_fsdp、cfg_size=2、vae_parallel 启用；FP16 与 W4A4F4 两组使用完全相同的并行约定。其中 `ALGO` 为推理仓用于选择 FA 计算方式的环境变量，需按设备与推理方式取值（取值说明详见推理仓文档）：本案例产品形态为 Ascend 950，浮点基线取 `ALGO=0`，量化侧因使能 attention FP8 取 `ALGO=3`，此为两侧唯一的非模型路径差异。

```bash
# Wan2.2浮点模型推理
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export FAST_LAYERNORM=1

torchrun --nproc_per_node=4 --master_port=23459 vbench.py \
--task t2v-A14B \
--ckpt_dir ${MODEL_PATH} \
--size "1280*720" \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 2 \
--ulysses_size 2 \
--vae_parallel \
--base_seed 0 \
--vbench_mini \
--vbench_mini_root ${EVAL_DATA_PATH} \
--save_path ${OUTPUT_DIR}/vbench_fp_output/ \
--num_samples 1 \
--temporal_flickering_samples 1
```

**输出**：推理正常完成，所有评测视频生成完毕，结果保存至`${OUTPUT_DIR}/vbench_fp_output/`目录，生成视频无画质崩坏。  
**记录**：推理完整运行日志。  

### 步骤 5：执行Wan2.2量化模型VBench评测推理

**目标**：运行W4A4F4量化模型推理，生成对比用评测结果。  
**输入**：W4A4F4量化权重、与浮点相同的VBench-mini评测数据集、W4A4F4的推理超参配置。  
**操作**：仅将 `ALGO` 由 0 改为 3、增加 `--quant_dit_path` 参数指向量化权重路径，其余推理参数与浮点完全相同，执行推理生成评测视频。  

```bash
# Wan2.2量化模型推理与VBench评测命令
export ALGO=3
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export FAST_LAYERNORM=1

torchrun --nproc_per_node=4 --master_port=23459 vbench.py \
--task t2v-A14B \
--ckpt_dir ${MODEL_PATH} \
--quant_dit_path ${QUANT_PATH} \
--size "1280*720" \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 2 \
--ulysses_size 2 \
--vae_parallel \
--base_seed 0 \
--vbench_mini \
--vbench_mini_root ${EVAL_DATA_PATH} \
--save_path ${OUTPUT_DIR}/vbench_quant_output/ \
--num_samples 1 \
--temporal_flickering_samples 1
```

**输出**：推理正常完成，所有评测视频生成完毕，结果保存至`${OUTPUT_DIR}/vbench_quant_output/`目录，生成视频无画质崩坏。  
**记录**：量化推理完整运行日志。  

## 6. 精度测试

> Wan2.2生成模型精度验证建议采用**客观指标+人工主观评测**结合方式，重点关注：画质清晰度、颜色准确性、时序一致性、运动连贯性、物体结构是否崩坏。

### 6.1 测试设计

| 项          | 内容                           |
| ---------- | ---------------------------- |
| 评测工具       | AISBench |
| 对比对象       | Wan2.2 FP16浮点推理生成结果 vs Wan2.2 W4A4F4量化推理生成结果 |
| 数据集与任务     | Vbench-1.0-mini 1%子集 |
| 样本数 / 子集策略 | VBench-1.0-mini 1%子集共11条prompt，每条prompt生成1个视频，共11个视频样本 |
| 指标与方向      | VBench Quality、VBench Semantic |
| 验收阈值       | VBench Quality 与 Semantic 两项的绝对得分下降均 < 1 pp（即 `FP 得分 − 量化得分 < 1 pp`）；逐项判定，任一项超阈值即不通过 |
| 随机性控制      | 固定随机种子seed=0、固定采样步数=40、固定分辨率720P、固定帧数81帧，对比双方所有推理参数完全一致 |

### 6.2 可复现过程

**数据准备**：

VBench-1.0-mini 数据集已在步骤 3 中准备完毕（`final_mini_dataset_0_01/` 目录）。VBench 评测的数据准备、小模型权重缓存下载及 AISBench 环境安装请参考 《[AISBench VBench 评测文档](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/extended_benchmark/lmm_generate/vbench.md)》。

**评测配置要点**：

VBench 指标计算通过 aisbench 的 `VBenchEvalTask` 完成，核心配置项如下（以 `ais_bench/configs/vbench_examples/eval_vbench_standard.py` 为例）。

**注意**：以下取值需直接写入配置文件，配置文件中不解析 shell 环境变量，请填写展开后的**绝对路径**（下表以步骤 1 中 `OUTPUT_DIR=/path/to/output/wan22_verification`、`EVAL_DATA_PATH=/path/to/eval/final_mini_dataset_0_01` 为例，请替换为实际路径）。

| 配置项 | 取值 |
| ------ | ---- |
| `DATA_PATH` | 浮点：`/path/to/output/wan22_verification/vbench_fp_output/`；量化：`/path/to/output/wan22_verification/vbench_quant_output/`（即步骤 4/5 `--save_path` 展开后的实际视频目录） |
| `full_json_dir` | `/path/to/eval/final_mini_dataset_0_01/VBench_kmeans_info.json`（步骤 3 准备的 0.01 子集 JSON，与推理阶段 `--vbench_mini_root` 下的文件一致） |
| `VBENCH_CACHE_DIR` | 小模型权重缓存目录路径 |
| `dimension_list` | 留空即评测全部 16 个维度 |
| `load_ckpt_from_local` | `True`（从本地缓存加载小模型权重） |

**执行命令**（推理阶段已在步骤 4/5 中生成视频，此处执行 VBench 指标计算）：

```bash
# 1) 浮点结果指标计算：先将配置文件中的 DATA_PATH 改为 .../vbench_fp_output/
ais_bench ${AISBENCH_HOME}/ais_bench/configs/vbench_examples/eval_vbench_standard.py \
    --mode eval \
    --max-num-workers 1

# 2) 量化结果指标计算：将同一配置文件中的 DATA_PATH 改为 .../vbench_quant_output/，
#    其余配置项（full_json_dir、dimension_list、VBENCH_CACHE_DIR 等）保持不变，再次执行相同命令
ais_bench ${AISBENCH_HOME}/ais_bench/configs/vbench_examples/eval_vbench_standard.py \
    --mode eval \
    --max-num-workers 1
```

**输出与记录**：

- 结果根目录：`./outputs/default/{timestamp}/`（在执行命令的当前目录下）
- 目录结构：

  ```text
  {timestamp}/
  ├── configs/                # 评测配置快照
  │   └── {timestamp}_*.py
  ├── logs/eval/vbench_eval/  # 各维度评分日志（.out）
  ├── results/vbench_eval/    # 各维度原始评分 JSON
  └── summary/                # 汇总结果（.csv/.md/.txt）
  ```

### 6.3 精度结果

| 数据集 | 样本数 | 基线（FP16）（%） | 量化（W4A4F4）（%） | 得分下降（pp） | 结论 |
| -------------- | ------ | -------------- | -------------- | --------------- | ----------------- |
| VBench Quality | 11 | 85.80 | 86.05 | −0.25 | 通过 |
| VBench Semantic | 11 | 72.01 | 71.06 | +0.95 | 通过 |
| VBench Total | 11 | 83.04 | 83.05 | −0.01 | 通过 |

**精度结论**：在 VBench-1.0-mini 1% 子集（共 11 条 prompt，每条生成 1 个视频）上，Wan2.2-T2V-A14B W4A4F4 量化推理结果的 VBench Quality 与 VBench Semantic 两项指标相对 FP16 浮点基线（85.80% / 72.01%）的绝对得分变化分别为 −0.25 pp 与 +0.95 pp，均满足"绝对得分下降 < 1 pp"的验收阈值；VBench Total 相对变化 −0.01 pp，亦通过验证。综合而言，本案例 W4A4F4 量化在 VBench-1.0-mini 1% 子集上达到预设精度目标，量化通过。

## 7. 结果与经验

### 7.1 关键结果汇总

| 步骤 | 关键操作 | 指标 | 变化 | 备注 |
| ------------------ | -------------------- | -------------------- | ------------------- | ------------------ |
| Wan2.2 A14B量化 | W4A4F4量化 | VBench Quality | −0.25 pp | 满足阈值（< 1 pp） |
| Wan2.2 A14B量化 | W4A4F4量化 | VBench Semantic | +0.95 pp | 满足阈值（< 1 pp） |
| Wan2.2 A14B量化 | W4A4F4量化 | VBench Total | −0.01 pp | 满足阈值（< 1 pp） |

### 7.2 经验总结

1. **常见问题**：量化后出现闪烁/色块时，可尝试调整激活量化方式或选择性回退部分层。
2. **对比测试**：浮点与量化推理必须保持超参完全一致，仅模型路径不同。

## 8. 附录

- 量化配置路径：[W4A4F4量化yaml配置](../../../lab_practice/wan2_2/wan2_2_w4a4f4_mxfp_t2v.yaml)
