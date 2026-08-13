# 浮点稀疏使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中配置和使用浮点稀疏（Float Sparse）算法。浮点稀疏通过 ADMM 算法对浮点权重进行稀疏化，结合硬件压缩单元实现高压缩率部署。

适用角色：算法工程师、模型部署工程师

适用场景：

- 需要高压缩率模型部署的场景，尤其是 Atlas 300I Duo 推理卡压缩单元场景。
- W8A8S 稀疏率不足、需要更高压缩率的场景。

不适用场景：

- 非 `nn.Linear` 模块的稀疏化目标。
- 非 v1 框架的逐层量化场景。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，确定压缩方案阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认部署环境为 Atlas 300I Duo 推理卡。
- 已准备校准数据集，且校准数据 token id 个数 ≥2048。
- Atlas 300I Duo 不支持 bfloat，需将模型 `config.json` 的 `torch_dtype` 修改为 `float16`。

**后续操作**：稀疏流程执行 → 精度验证 → 部署上线或进入调优。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors`；`torch_dtype` 需为 `float16` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认或用户指定 | JSONL 格式，每条含文本 prompt；token id 个数 ≥2048 | 可被工具加载并完成前向推理 |
| 输入 | 量化 YAML 配置文件 | 用户编写 | 符合 msModelSlim YAML 规范，含 `float_sparse` 处理器配置 | 可通过工具 `--config_path` 参数加载 |
| 交付件 | 稀疏量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors`，已应用浮点稀疏 | 推理冒烟通过 |

## 4. 流程总览

```mermaid
flowchart LR
    A[编写 YAML 配置] --> B[执行量化命令]
    B --> C[收集激活统计]
    C --> D[ADMM 迭代稀疏]
    D --> E[验证量化结果]
```

## 5. 操作步骤

### 步骤 1：修改模型并编写 YAML 配置文件

**目标**：准备模型并编写包含 `float_sparse` 处理器配置的 YAML 文件。

**操作**：

1. 将模型路径下 `config.json` 中的 `torch_dtype` 字段修改为 `float16`。
2. 在 `spec.process` 下配置 `float_sparse` 处理器，指定 `type: "float_sparse"`。
3. 按需配置 `sparse_ratio`（默认 `0.3`）与 `include`/`exclude`。

YAML 配置示例：

```yaml
spec:
  process:
    - type: "float_sparse"
      sparse_ratio: 0.3          # 稀疏比例，取值范围为 0.0~1.0，默认0.3。
      include: ["*"]             # 包含的层，支持通配符。
      exclude: ["*self_attn*"]   # 排除的层，支持通配符。
```

YAML 配置字段详解如下：

| 字段名 | 作用 | 说明 |
| --- | --- | --- |
| type | 处理器类型标识 | 固定为 `"float_sparse"`。 |
| sparse_ratio | 稀疏比例 | 取值范围 `0.0~1.0`，默认 `0.3`。 |
| include | 包含的层 | 字符串列表，支持通配符匹配。 |
| exclude | 排除的层 | 字符串列表，支持通配符匹配，优先级高于 `include`。 |

**输出**：YAML 配置文件 `${CONFIG_PATH}`。

### 步骤 2：执行量化命令

**目标**：使用上一步编写的 YAML 配置文件启动稀疏化流程。

**操作**：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config_path ${CONFIG_PATH} \
  --trust_remote_code True
```

参数说明：

| 参数 | 必选 | 说明 |
| --- | --- | --- |
| `model_path` | 是 | 浮点模型权重路径 |
| `save_path` | 是 | 量化权重保存路径 |
| `device` | 否 | 量化设备，默认 `npu` |
| `model_type` | 是 | 模型名称，与支持矩阵一致 |
| `config_path` | 是 | 步骤 1 编写的 YAML 配置路径 |
| `trust_remote_code` | 否 | 是否信任远程代码，默认 `False` |

执行流程说明：

1. 工具加载 YAML 配置，解析 `float_sparse` 处理器。
2. 预处理阶段安装前向 hook 收集激活统计，构建 Hessian 矩阵。
3. ADMM 迭代求解最优稀疏模式，并通过 L2 量化保护重要位置精度。
4. 保存稀疏量化权重。

**输出**：量化权重目录 `${SAVE_PATH}`，包含量化描述文件与权重分片。

### 步骤 3：验证量化结果

**目标**：确认稀疏权重文件完整且可加载。

**操作**：

1. 检查输出目录是否包含 `quant_model_description.json` 文件。
2. 检查日志确认 ADMM 迭代正常完成。
3. 在 Atlas 300I Duo 推理卡上利用硬件压缩单元加载权重进行冒烟测试。

**输出**：稀疏权重验证通过。

## 6. 验收条件

- 量化权重目录包含 `quant_model_description.json` 及所有必需的 `*.safetensors` 分片文件。
- 日志无求矩阵逆失败或显存溢出告警。
- 稀疏后模型推理精度符合业务要求。

## 7. 异常处置

- **不支持叠加 w8a8 稀疏量化**：浮点稀疏（W16A16S）与 W8A8S 是两种不同技术路径，不支持叠加使用。
- **稀疏比例设置过高**：降低 `sparse_ratio`，建议在 0.3 附近逐步调整。
- **校准数据长度不够**：增加校准集中每条数据长度，保证 token id 个数 ≥2048。
- **校准集数量过多导致显存溢出**：减少校准集数量，或使用单卡显存更大的机器。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 浮点稀疏 | 基于 ADMM 的模型稀疏化算法 | [浮点稀疏词条](./term_float_sparse.md) |
| ADMM | 交替方向乘子法，用于求解带约束的稀疏优化问题 | [浮点稀疏词条](./term_float_sparse.md) |
| sparse_ratio | 稀疏比例，控制稀疏化程度 | [浮点稀疏词条](./term_float_sparse.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `float_sparse` 处理器 | 用于执行浮点稀疏的 Processor 配置 | [浮点稀疏词条](./term_float_sparse.md) |
| `AdmmPruner` | ADMM 稀疏器核心类 | [浮点稀疏词条](./term_float_sparse.md) |
