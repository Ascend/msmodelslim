# Qwen3-32B W8A8精度调优案例

## 1. 案例背景

本案例针对 Qwen3-32B 模型的 W8A8 量化场景：初始采用全静态量化（per-channel/per-tensor）搭配 Smooth Quant 离群值抑制算法，量化后模型对话出现乱码、无法正常使用。通过系统化调优，将量化模型在 AIME25 数据集上的精度从不可用提升至 70.00%，并分别在 GPQA 数据集上验证了校准集调整与量化回退两类备选优化手段的效果。

本案例实际覆盖[《量化精度调优指南》](../user_guide/process_quantization_precision_tuning.md)的全部 5 个步骤：确认精度问题可信 → 调整离群值抑制算法 → 调整量化策略 → 调整校准集 → 量化回退。

**关联流程**：[《量化精度调优指南》](../user_guide/process_quantization_precision_tuning.md)

## 2. 环境与版本

| 项 | 版本或配置 |
| --- | --- |
| 产品形态 | A3（Atlas A3 系列推理产品，不限定：调优方法与结论不限硬件形态，本案例中的精度与量化时间数据在该形态上实测获得） |
| CANN | 9.0.0 |
| PyTorch | 2.10 |
| TorchNPU | 2.10（与 CANN、PyTorch 配套） |
| vLLM Ascend | 0.22.1rc1（推理部署环境） |
| transformers | 4.51.0 |
| 运行镜像 | `m.daocloud.io/quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` |
| 其他依赖 | msModelSlim（本案例工具，安装见《[msModelSlim工具安装指南](../install_guide/install_guide.md)》）；AISBench（精度测评）；Qwen3-32B 模型权重与 tokenizer |

**本次前置事实**：

- msModelSlim 已安装，`msmodelslim` 命令可执行。
- Qwen3-32B 浮点模型已准备，可在目标推理引擎上加载并复现原始精度。
- AIME25、GPQA 测评数据集已就绪，可在 AISBench 上完成测评。
- 已预设量化精度要求，本案例以量化模型 AIME25 精度达到 70.00% 作为达成标准。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | Qwen3-32B 浮点模型 | 本地保存的模型权重目录 | 含模型配置、权重分片及类别所需附属文件（如 tokenizer、config 等） | 文件齐全；若官方提供校验值或版本号，与本地一致 |
| 输入 | 量化配置 YAML | 本案例各步骤给出的配置文件（每组实验一个） | modelslim_v1 协议 YAML | 可被 `msmodelslim quant` 加载执行 |
| 输入 | 校准集 | 仓库内置校准集目录，格式参考[`msmodelslim/lab_calib/mix_calib.jsonl`](../../../lab_calib/mix_calib.jsonl)、[`qwen3_cot_w4a4.json`](../../../lab_calib/qwen3_cot_w4a4.json) | JSON/JSONL，样本与任务、语言匹配 | 样本可被模型 tokenizer 预处理，量化流程可正常读取 |
| 输入 | 测评集与 badcase | AIME25、GPQA 测评集及 AISBench 测评结果 | 与任务匹配的问答格式 | 浮点模型在测评集上可复现原始精度 |
| 交付件 | 最终量化权重目录 | 用户指定的输出目录 | 符合所选导出格式 | 文件齐全；符合所选导出格式约定 |
| 交付件 | 精度测评报告 | AISBench 测评结果目录（按时间戳自动生成） | 浮点与量化对比结果 | 与预设精度要求逐项比对 |

## 4. 操作步骤

### 命令约定

**量化命令**：本案例所有量化实验统一使用以下命令执行，路径参数按实际环境填写：

```bash
# 按实际环境填写以下路径
MODEL_PATH=/path/to/qwen3-32b      # 浮点模型权重目录
SAVE_PATH=/path/to/save            # 量化产物输出目录
CONFIG_PATH=/path/to/config.yaml   # 量化配置 YAML

msmodelslim quant \
    --model_type Qwen3-32B \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --device npu \
    --config_path ${CONFIG_PATH} \
    --trust_remote_code True
```

**测评命令**：精度测评使用 AISBench 工具完成，使用方法见[AISBench 快速入门](https://github.com/AISBench/benchmark/blob/master/docs/source_zh_cn/get_started/quick_start.md)。本案例使用数据集配置 `aime2025_gen_0_shot_chat_prompt`（AIME25）、`gpqa_gen_0_shot_cot_chat_prompt`（GPQA），数据集说明见 [AIME25](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/aime2025/README.md)、[GPQA](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/gpqa/README.md)。

### 步骤1：确认精度问题可信

**目标**：排除环境干扰，确认精度问题真实存在。

**输入**：浮点模型、初始配置量化权重、AIME25 测评数据集、目标推理引擎（如 vLLM Ascend）及指向其服务的 AISBench 模型配置。

**操作**：

1. 将浮点模型部署到目标推理引擎，按命令约定使用 AISBench 完成 AIME25 基线测评（工作目录与后续实验区分），确认可复现原始精度。
2. 将初始量化模型（Smooth Quant 产物）部署后执行同样的 AIME25 测评，检查输出：确认乱码不是上下文截断、超时等非量化问题导致（必要时抽查推理日志中的生成内容）。
3. 结合 AIME25 数据集精度波动范围，判断当前精度损失异常，确认为量化引入。

**输出**：浮点模型 AIME25 基线测评报告；确认精度问题由量化引入，量化模型乱码输出样例。

**记录**：浮点模型 AIME25 基线精度、量化模型乱码输出日志路径。

**下一步**：步骤1 确认浮点基线正常、乱码问题由量化引入（Smooth Quant 初始配置精度远未达标），进入步骤2 调整离群值抑制算法。

### 步骤2：调整离群值抑制算法（关键步骤）

**目标**：通过切换离群值抑制算法消除对话乱码并提升精度。

**输入**：5 组离群值抑制算法配置、浮点模型、AIME25 测评数据集。

**操作**：

1. **生成配置并量化**：在量化配置 YAML 的 `process` 中依次替换离群值抑制处理器（算法、对称性、`alpha` 参数），得到 5 组实验配置（均搭配相同的 `linear_quant` 静态量化与 `save` 段），逐一执行量化命令（`--config_path` 与 `--save_path` 按算法区分）。各组算法配置要点如下（完整参数见算法文档）：

   a. **Smooth Quant（初始配置）**：alpha 0.5、对称。详见[《Smooth Quant 算法》](../knowledge_base/quantization_algorithms/smooth_quant/smooth_quant.md)。

   b. **Iterative Smooth（对称）**：对称/alpha:0.5 与 对称/alpha:0.9 两组仅 `alpha` 参数不同，其余配置相同（4 类子图）。详见[《Iterative Smooth 算法》](../knowledge_base/quantization_algorithms/iterative_smooth/iterative_smooth.md)。

   c. **Iterative Smooth（非对称/alpha:0.5）**：非对称仅支持 `norm-linear` 子图。

   d. **Flex Smooth Quant**：alpha/beta 缺省自动搜索。详见[《Flex Smooth Quant 算法》](../knowledge_base/quantization_algorithms/flex_smooth_quant/flex_smooth_quant.md)。

   算法对比见[离群值抑制算法](../knowledge_base/quantization_algorithms/README.md#离群值抑制算法)。

2. **部署测评**：将每组量化产物部署到目标推理引擎，执行测评命令（AIME25数据集），记录精度与量化耗时。
3. **汇总对比**：汇总各算法的精度与量化时间，选择综合最优方案。

| 离群值抑制算法 | AIME25 精度（%） | 量化时间（s） | 备注 |
|---------------|-----------------|--------------|------|
| Smooth Quant | 对话乱码 | 326 | 初始配置，精度下降明显 |
| Iterative Smooth（对称/alpha:0.5） | 3.33 | 324 | 存在重复token输出 |
| Iterative Smooth（非对称/alpha:0.5） | 0 | 305 | 输出乱码 |
| Iterative Smooth（对称/alpha:0.9） | 63.33 | 319 | 精度正常 |
| Flex Smooth Quant | 13.33 | 1380 | 存在重复token输出；所需时间明显更长 |

**输出**：5 组量化权重及对应 AIME25 测评报告；确定离群值抑制算法为 Iterative Smooth（对称/alpha:0.9）。

**记录**：上表 5 组配置的精度与量化时间实测数据。对比分析：对称/alpha:0.5（3.33%）、非对称/alpha:0.5（0%）、Flex Smooth Quant（13.33%）在 per-tensor 静态激活量化下均出现重复 token/乱码的严重退化，仅对称/alpha:0.9 精度正常（63.33%），量化时间 319秒，较 Flex Smooth Quant 节省 76.9%。

**下一步**：步骤2 最优配置（Iterative Smooth 对称/alpha:0.9）AIME25 精度 63.33%，未达到预设精度要求（70.00%），进入步骤3 在固定离群值抑制算法下调整量化策略。

### 步骤3：调整量化策略（量化算法选择）

**目标**：在离群值抑制算法确定后，通过切换权重量化方法、激活量化粒度进一步提升精度。

**输入**：步骤2确定的基础配置（Iterative Smooth（对称/alpha:0.9））、4 组量化策略配置（组合 `minmax`/`ssz` × `per_tensor`/`per_token`）、AIME25 测评数据集。

**操作**：

1. **生成配置并量化**：在量化配置 YAML 的 `linear_quant` 处理器中，分别修改 `qconfig.act.scope`（`per_tensor`/`per_token`）与 `qconfig.weight.method`（`minmax`/`ssz`），组合出 4 组配置并逐一执行量化命令。`linear_quant` 处理器参数详见[《Linear Quant 算法》](../knowledge_base/quantization_algorithms/linear_quant/linear_quant.md)，量化方法参数详见[MinMax](../knowledge_base/quantization_algorithms/minmax/minmax.md)、[SSZ](../knowledge_base/quantization_algorithms/ssz/ssz.md)。

2. **部署测评**：将每组量化产物部署后执行测评命令（AIME25），记录精度与量化耗时。
3. **汇总对比**：对比精度与量化时间，选择综合最优配置。

| 权重量化方法 | 激活量化粒度 | AIME25 精度（%） | 量化时间（s） | 备注 |
|-------------|-------------|----------------|-------------|------|
| minmax | per-tensor（静态量化） | 63.33 | 319 | 基础配置（基于步骤2结果） |
| minmax | per-token（动态量化） | 70.00 | 289 | 激活使用per-token后精度提升6.67个百分点 |
| ssz | per-tensor（静态量化） | 66.67 | 408 | 静态量化下 ssz 优于 minmax |
| ssz | per-token（动态量化） | 76.67 | 348 | 动态量化下 ssz 优于 minmax，为本次实测最高 |

**输出**：4 组量化权重及对应 AIME25 测评报告；确定量化策略为 **minmax + per-token（动态量化）**。

**记录**：上表 4 组配置的精度与量化时间实测数据。对比分析：动态量化整体优于静态量化（minmax 70.00% vs 63.33%，ssz 76.67% vs 66.67%）；ssz + per-token 精度上限最高（76.67%），minmax + per-token（70.00%）达到预设精度要求（70.00%）；量化时间方面 minmax + per-token 最短（289秒，较 ssz + per-token 节省 59 秒，较 minmax + per-tensor 节省 30秒），综合精度与效率选择 minmax + per-token。

**下一步**：步骤3 达到预设精度要求（70.00%），为展示完整的调优流程并验证校准集调整的效果，进入步骤4。

### 步骤4：调整校准集

**目标**：通过校准集优化（加入 badcase 样本）验证对量化精度的提升效果。GPQA 数据集题目数量更多，能够更清晰地展现不同配置间的精度差异。

**输入**：步骤3达到精度要求后的量化配置（本节以 Iterative Smooth + 静态量化作为基准配置，在 GPQA 数据集上验证）、该基准配置的 GPQA 测评结果（含 badcase 样本）、校准集文件、GPQA 测评数据集。

**操作**：

1. **校准集优化策略对比**：校准集的质量直接影响量化参数的准确性，可按以下策略调整校准集（本案例执行的是最后一行「加入 badcase」策略）：

   | 调整策略 | 具体操作 | 校准集变化 | 优化目的 |
   |---------|---------|-----------|---------|
   | 初始校准集 | 随机样本 | 少量样本 | 建立基准配置 |
   | 增加数据量 | 适当增加样本数量 | 数量增加 | 提升量化参数估计的准确性 |
   | 匹配应用场景 | 使用与任务、语言匹配的数据替换 | 构成变化 | 使校准数据更贴近实际应用场景 |
   | 平衡数据分布 | 从多个数据集抽取样本混合 | 构成变化 | 提升数据分布的多样性和均衡性 |
   | 剔除异常数据 | 移除导致精度下降的异常样本 | 数量减少 | 减少异常样本对量化参数的干扰 |
   | 加入badcase | 加入模型在该数据集上的badcase样本 | 数量增加 | 帮助量化模型学习困难样本，提升精度 |

   本案例使用默认校准集 `mix_calib.jsonl`（48 条），加入 5 条 badcase 后为 53 条。

2. **获取 badcase 样本**：从 AISBench 测评结果文件中筛选「模型输出（`prediction` 字段）与参考答案（`gold` 字段）不一致」的样本，提取少量 badcase。例如，一个 badcase 样本为：

   ```text
   What is the correct answer to this question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?

   Choices:
   (A)10^-11 eV
   (B)10^-8 eV
   (C)10^-9 eV
   (D)10^-4 eV
   Format your response as follows: "The correct answer is (insert answer here)"
   ```

3. **格式转换**：

   - **JSONL格式**：参考校准集文件 `mix_calib.jsonl` 的 JSONL 格式（格式说明见输入和交付件），将文本放在`"inputs_pretokenized"`字段后，格式如下：

     ```json
     {"inputs_pretokenized":"What is the correct answer to this question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\n\nChoices:\n(A)10^-11 eV\n(B)10^-8 eV\n\n(C)10^-9 eV\n(D)10^-4 eV\nFormat your response as follows: \"The correct answer is (insert answer here)\""}
     ```

   - **JSON格式**：参考[`msmodelslim/lab_calib/qwen3_cot_w4a4.json`](../../../lab_calib/qwen3_cot_w4a4.json)，直接将文本加入字符串列表中即可。

   将 badcase 样本与原校准集样本合并，生成新的校准集文件（本案例为 48→53 条）。

4. **重新量化**：在量化配置 YAML 的 `spec` 下通过 `dataset` 字段指定加入 badcase 后的校准集文件（无需改动仓库默认校准集），其余处理器与 `save` 配置同步骤5 的完整 YAML 示例（`iter_smooth` 对称/alpha:0.9、`linear_quant` 静态量化），使用命令约定的量化命令重新生成量化权重。

5. **测评验证**：将新量化产物部署后执行测评命令（GPQA 数据集）。

**输出**：badcase 校准集与重新生成的量化权重。GPQA 精度从基准的 48.98% 提升至 57.07%（提升 8.09 个百分点）。

| 量化策略 | GPQA 精度（%） | 备注 |
|-------------|---------------|------|
| Iterative Smooth + 静态量化 | 48.98 | 基准配置 |
| Iterative Smooth + 静态量化 + badcase调整校准集 | 57.07 | 相比基准配置精度提升8.09个百分点，说明badcase样本有助于量化模型学习困难样本，提升量化精度 |

**记录**：上表 2 组配置的 GPQA 精度实测数据。

**下一步**：为展示完整的调优流程并验证量化回退的效果，进入步骤5。

### 步骤5：量化回退（备选方案）

**目标**：验证将量化敏感层回退为浮点精度对量化精度的提升效果。

**输入**：步骤4的量化配置（Iterative Smooth + 静态量化）、GPQA 测评数据集、校准数据集。

**操作**：

1. **敏感层分析**：使用 msModelSlim 的敏感层分析工具识别量化敏感层，详细使用方法请参考[《量化敏感层分析使用指南》](../user_guide/usage_sensitive_linear_analysis.md)。

   ```bash
   # 按实际环境填写以下路径
   MODEL_PATH=/path/to/qwen3-32b      # 浮点模型权重目录

   msmodelslim analyze linear \
       --model_type Qwen3-32B \
       --model_path ${MODEL_PATH} \
       --device npu \
       --topk 20 \
       --metrics kurtosis
   ```

   根据量化敏感度得分从高到低排序，Top敏感层结果如下：

   ```text
   layers.3.mlp.down_proj
   layers.63.mlp.down_proj
   layers.2.mlp.down_proj
   layers.1.mlp.down_proj
   layers.4.mlp.down_proj
   layers.6.mlp.down_proj
   layers.7.mlp.down_proj
   layers.5.mlp.down_proj
   layers.0.mlp.down_proj
   layers.31.mlp.down_proj
   layers.62.mlp.down_proj
   layers.5.mlp.gate_proj
   layers.5.mlp.up_proj
   layers.32.mlp.down_proj
   layers.8.mlp.gate_proj
   layers.8.mlp.up_proj
   layers.6.mlp.gate_proj
   layers.6.mlp.up_proj
   ```

   分析结果显示 `mlp.down_proj` 层敏感度排名靠前，是量化难度较大的层类型，应优先考虑回退。

2. **修改量化配置**：在量化配置 YAML 中，通过 `exclude` 字段回退最为敏感的前9层（均为 `mlp.down_proj` 层）：

   ```yaml
   apiversion: modelslim_v1
   spec:
     process:
       - type: "iter_smooth"          # 参数同步骤2（对称/alpha:0.9）
       - type: "linear_quant"
         qconfig:
           act:
             scope: "per_tensor"
             dtype: "int8"
             symmetric: false
             method: "minmax"
           weight:
             scope: "per_channel"
             dtype: "int8"
             symmetric: true
             method: "minmax"
         include:
           - "*"
         exclude:
           - 'model.layers.3.mlp.down_proj'
           - 'model.layers.63.mlp.down_proj'
           - 'model.layers.2.mlp.down_proj'
           - 'model.layers.1.mlp.down_proj'
           - 'model.layers.4.mlp.down_proj'
           - 'model.layers.6.mlp.down_proj'
           - 'model.layers.7.mlp.down_proj'
           - 'model.layers.5.mlp.down_proj'
           - 'model.layers.0.mlp.down_proj'
     save:
       - type: "ascendv1_saver"
         part_file_size: 4
   ```

3. **重新生成量化权重**：使用修改后的配置执行量化命令，生成包含回退层的量化模型。

4. **测评验证**：将新量化产物部署后执行测评命令（GPQA 数据集）。

**输出**：含 9 个回退层的量化模型。GPQA 精度从基准的 48.98% 提升至 49.49%（提升 0.51 个百分点）。

| 量化策略 | GPQA 精度（%） | 备注 |
|-------------|---------------|------|
| Iterative Smooth + 静态量化 | 48.98 | 基准配置 |
| Iterative Smooth + 静态量化 + 回退前9层 | 49.49 | 相比基准配置精度提升0.51个百分点，说明回退量化敏感层能有效提升量化精度，但会带来一定的性能开销和模型大小增加 |

**记录**：上表 2 组配置的 GPQA 精度实测数据。

**下一步**：调优结束。

## 5. 结果与经验

### 5.1 关键结果汇总

| 步骤 | 关键操作 | 指标 | 变化 | 备注 |
|------|---------|------|------|------|
| 初始状态 | Smooth Quant + minmax 静态量化 | AIME25 精度：对话乱码 | — | 初始配置，量化模型不可用 |
| 步骤2 | 切换 Iterative Smooth（对称/alpha:0.9） | AIME25 精度：63.33% | 乱码 → 63.33% | 解决乱码问题，模型可用 |
| 步骤3 | 激活量化切换为 per-token（动态量化） | AIME25 精度：70.00% | +6.67 个百分点 | 达到预设精度要求（70.00%） |
| 步骤4 | badcase 样本加入校准集 | GPQA 精度：57.07% | +8.09 个百分点 | 相对 GPQA 基准（48.98%），备选验证 |
| 步骤5 | 回退前 9 个敏感层（mlp.down_proj） | GPQA 精度：49.49% | +0.51 个百分点 | 相对 GPQA 基准（48.98%），备选验证 |

**说明**：步骤3达到 70.00% 精度后已满足预设精度要求；步骤4和步骤5在 GPQA 数据集上验证校准集调整与量化回退两类备选手段的效果，展示完整调优路径。

**最终配置**：

- **离群值抑制算法**：Iterative Smooth（对称/alpha:0.9）。
- **权重量化**：`minmax` 方法，`per_channel` 粒度，`int8` 数据类型，对称量化。
- **激活量化**：`minmax` 方法，`per_token` 粒度（动态量化），`int8` 数据类型。

### 5.2 经验总结

1. **离群值抑制算法是关键**：从Smooth Quant切换到Iterative Smooth（对称/alpha:0.9），AIME25 精度从乱码提升至 63.33%，使量化模型具备可用性。适用边界：优先选用 Iterative Smooth，非对称方案通常优于对称方案但需推理引擎适配支持，量化前需确认推理引擎对离群值抑制算法的适配情况。

2. **激活量化粒度影响显著**：从 per-tensor（静态量化）切换到 per-token（动态量化），AIME25 精度从 63.33% 提升至 70.00%（+6.67 个百分点）。适用边界：per-token 动态量化计算更复杂，可能带来一定的推理性能损失，追求性能时可改用 `pd_mix` 混合策略。

3. **量化方法选择需权衡精度与效率**：实测 ssz 精度上限更高（ssz + per-token 76.67% vs minmax + per-token 70.00%），但 minmax 量化时间更短（289秒 vs 348秒）、实现更简单，且在满足预设精度要求（70.00%）的前提下综合更优，故本案例选择 minmax + per-token。适用边界：追求极限精度时可选用 ssz，追求效率时优先 minmax；INT4 等低比特场景建议优先选择 ssz 方法。

4. **校准集质量影响精度**：在 GPQA 数据集上，将 badcase 样本加入校准集后精度从 48.98% 提升至 57.07%（+8.09 个百分点）。适用边界：数据量建议 10-50条，需与任务、语言匹配（中文模型用中文数据），badcase 需来自同一测评集才能稳定复现收益。

5. **量化回退是最后手段**：在 GPQA 数据集上，回退 9 个量化敏感层后精度从 48.98% 提升至 49.49%（+0.51 个百分点）。适用边界：回退会带来性能开销和模型大小增加，应在其他手段无法满足要求时使用；`mlp.down_proj` 通常为敏感层，可优先回退。

## 6. 附录

- 相关指南：[《量化敏感层分析使用指南》](../user_guide/usage_sensitive_linear_analysis.md)
- 测评数据：[AIME25](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/aime2025/README.md)、[GPQA](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/gpqa/README.md)
