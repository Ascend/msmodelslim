# 调试模式使用指南

## 简介

调试模式（Debug Mode）是一键量化功能提供的高级特性，用于保存量化过程中的关键信息，帮助开发者和高级用户深入了解量化过程、排查问题或进行算法研究。

启用调试模式后，工具会在量化完成时自动保存量化过程中的上下文信息（Context），包括各个处理器的中间结果、统计数据、张量信息等，便于后续分析和调试。

## 使用方法

### 命令格式

在一键量化命令中添加 `--debug` 参数即可启用调试模式：

```bash
msmodelslim quant --debug [其他参数]
```

### 参数说明

| 参数名称 | 解释 | 是否可选 | 说明 |
|---------|------|---------|------|
| --debug | 启用调试模式 | 可选 | 添加此参数后，量化完成时会自动保存上下文信息到调试目录 |

### 使用示例

#### 示例1：基础调试模式

对 Qwen2.5-7B-Instruct 模型进行 W8A8 量化并启用调试模式：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type Qwen2.5-7B-Instruct \
  --quant_type w8a8 \
  --trust_remote_code True \
  --debug
```

#### 示例2：多卡量化 + 调试模式

使用多卡进行分布式量化并启用调试模式：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu:0,1,2,3 \
  --model_type Qwen2.5-7B-Instruct \
  --quant_type w8a8 \
  --trust_remote_code True \
  --debug
```

#### 示例3：自定义配置 + 调试模式

使用自定义配置文件进行量化并启用调试模式：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config_path ${CONFIG_PATH} \
  --trust_remote_code True \
  --debug
```

## 调试信息输出

### 输出目录结构

启用调试模式后，工具会在量化权重保存路径（`save_path`）下创建 `debug_info` 子目录，用于存储调试信息：

```text
${SAVE_PATH}/
├── config.json                          # 原始模型配置文件
├── generation_config.json               # 原始生成配置文件
├── quant_model_description.json         # 量化权重描述文件
├── quant_model_weight_w8a8.safetensors  # 量化权重文件
├── tokenizer_config.json                # 原始分词器配置文件
├── tokenizer.json                       # 原始分词器词汇表
└── debug_info/                          # 调试信息目录
    ├── debug_info.json                  # 调试信息元数据（JSON格式）
    └── debug_info.safetensors           # 调试信息张量数据（SafeTensors格式）
```

### 输出文件说明

#### debug_info.json

包含量化过程中的非张量数据和张量元数据，采用分命名空间（namespace）的结构组织：

**文件结构**：

```json
{
  "namespace_key_1": {
    "field_1": "value",
    "field_2": 123,
    "tensor_field": {
      "_type": "tensor",
      "_file": "debug_info.safetensors",
      "_key": "tensor_0"
    }
  },
  "namespace_key_2": {
    ...
  }
}
```

**字段说明**：

- **命名空间（namespace）**：每个处理器或模块会创建独立的命名空间，用于隔离不同阶段的调试信息
- **普通字段**：直接存储标量值（整数、浮点数、字符串、布尔值等）
- **张量引用**：对于 PyTorch 张量，存储引用信息：
  - `_type`: 固定值 `"tensor"`，标识这是一个张量引用
  - `_file`: 张量数据所在的文件名（`debug_info.safetensors`）
  - `_key`: 张量在 SafeTensors 文件中的键名

#### debug_info.safetensors

以 SafeTensors 格式存储量化过程中的所有张量数据，包括：

- 量化参数（scale、zero_point 等）
- 统计信息（最小值、最大值、直方图等）
- 中间结果张量
- 其他调试用张量

**特点**：

- **高效存储**：SafeTensors 格式支持快速加载和内存映射
- **跨平台兼容**：可在不同框架和平台间共享
- **安全性**：相比 pickle 格式更安全，避免代码注入风险

### 保存的信息内容

调试模式会保存各个处理器在量化过程中产生的调试信息，具体内容取决于使用的量化配置和处理器类型。典型的调试信息包括：

**量化处理器（linear_quant）**：

- 每层的量化参数（scale、zero_point）
- 激活值统计信息（min、max、histogram）
- 权重统计信息
- 量化前后的误差分析

**离群值抑制处理器（如 iter_smooth、flex_smooth）**：

- 平滑因子（smoothing factors）
- 迭代过程中的中间结果
- 每层的平滑效果评估指标

**其他处理器**：

- 处理器特定的配置参数
- 中间计算结果
- 性能统计信息

## 调试信息的使用

### 加载调试信息

可以使用 Python 脚本加载和分析调试信息：

```python
import json
from safetensors import safe_open

# 加载 JSON 元数据
with open("debug_info/debug_info.json", "r") as f:
    debug_meta = json.load(f)

# 加载 SafeTensors 张量数据
with safe_open("debug_info/debug_info.safetensors", framework="pt") as f:
    # 获取所有张量的键名
    tensor_keys = f.keys()
    
    # 加载特定张量
    for key in tensor_keys:
        tensor = f.get_tensor(key)
        print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
```

### 分析量化效果

通过调试信息可以进行以下分析：

**1. 查看量化参数分布**

```python
# 分析某个命名空间下的量化参数
namespace = debug_meta["linear_quant_namespace"]

# 查看 scale 参数的引用
if "scales" in namespace and namespace["scales"]["_type"] == "tensor":
    scale_key = namespace["scales"]["_key"]
    # 从 SafeTensors 加载实际数据进行分析
```

**2. 对比不同层的统计信息**

```python
# 遍历所有命名空间，收集统计信息
for ns_name, ns_data in debug_meta.items():
    if "layer_stats" in ns_data:
        print(f"Layer: {ns_name}")
        print(f"Stats: {ns_data['layer_stats']}")
```

**3. 排查精度问题**

通过对比量化前后的激活值分布、查看离群值抑制效果等，定位精度下降的原因。

## 注意事项

### 存储空间

- 调试信息可能占用较大的磁盘空间（通常为模型大小的 10%-50%），请确保有足够的存储空间
- 对于超大模型（100B+），调试信息可能达到数十 GB

### 性能影响

- 启用调试模式会略微增加量化时间（通常增加 5%-10%）
- 主要开销来自于调试信息的序列化和写入磁盘

### 安全性

- 调试信息可能包含模型的敏感信息（如量化参数、统计数据等）
- 请妥善保管调试信息文件，避免泄露

### 兼容性

- 调试信息的格式可能随版本更新而变化
- 建议使用相同版本的 msModelSlim 工具加载和分析调试信息

## 适用场景

调试模式适用于以下场景：

### 1. 量化精度调优

当量化后模型精度不理想时，通过调试信息分析：

- 哪些层的量化误差较大
- 离群值抑制算法是否生效
- 量化参数分布是否合理

### 2. 算法研究与开发

研究人员可以通过调试信息：

- 分析不同量化算法的效果
- 对比不同配置的量化结果
- 开发新的量化算法或优化策略

### 3. 问题排查与报告

遇到量化问题时，调试信息可以帮助：

- 快速定位问题所在
- 向技术支持提供详细的诊断信息
- 复现和验证问题

### 4. 模型分析与优化

通过调试信息了解：

- 模型各层的激活值分布特征
- 量化敏感层的识别
- 混合精度量化策略的制定依据

## 常见问题

### Q1: 调试信息占用空间过大怎么办？

A: 可以考虑以下方法：

1. 量化完成后及时分析并删除不需要的调试信息
2. 仅在需要深入分析时启用调试模式
3. 使用压缩工具对调试信息目录进行压缩存档

### Q2: 如何判断是否需要启用调试模式？

A: 以下情况建议启用调试模式：

- 量化后精度明显下降，需要排查原因
- 尝试新的量化配置或算法组合
- 进行算法研究或模型分析
- 向技术支持报告问题时需要提供详细信息

### Q3: 调试信息保存失败怎么办？

A: 可能的原因和解决方法：

1. **磁盘空间不足**：清理磁盘空间或更换保存路径
2. **权限问题**：确保对保存路径有写入权限
3. **路径不存在**：确保 `save_path` 参数指定的路径有效

如果保存失败，工具会输出警告信息但不会中断量化流程，量化权重仍会正常保存。

### Q4: 调试信息可以在不同设备间共享吗？

A: 可以。调试信息使用标准的 JSON 和 SafeTensors 格式，可以在不同设备和平台间共享。但需要注意：

- 确保使用兼容版本的 msModelSlim 工具
- SafeTensors 文件可能较大，传输时注意网络带宽

### Q5: 调试模式是否影响量化结果？

A: 不影响。调试模式仅在量化完成后保存上下文信息，不会改变量化算法的执行逻辑和量化结果。启用或不启用调试模式，生成的量化权重完全一致。

## 相关文档

- 《[一键量化完整指南](usage.md)》
- 《[量化精度调优指南](../../case_studies/quantization_precision_tuning_guide.md)》
- 《[敏感层分析模块](../sensitive_layer_analysis/usage.md)》
