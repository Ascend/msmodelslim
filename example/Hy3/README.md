# Hy3 量化说明

## 模型介绍

- [腾讯混元 Hy3](https://www.tencent.com/zh-cn/tencent-hunyuan-officially-releases-hy3-advancing-agent-capabilities-and-deeper-product-integration) 于 2026 年正式发布，是混元团队在 Hy3 preview 之后推出的正式版本。相比 preview，Hy3 在智能水平、稳定性与成本效益上均有显著提升，智能水平强于同尺寸模型，并可比肩参数规模为 2～5 倍的旗舰模型。

- Hy3 是一款**快慢思考融合**的大语言模型，采用 **混合专家（MoE）** 架构，总参数量约 295B、激活参数量约 21B，支持 256K 上下文长度。模型在复杂推理、指令遵循、上下文学习、代码生成与 Agent 能力上表现突出，尤其适用于软件开发、办公生产、金融建模、前端设计、游戏制作等生产力场景。

- Hy3 采用 Apache 2.0 开源协议，权重可在 [Hugging Face](https://huggingface.co/tencent/Hy3)、[ModelScope](https://modelscope.cn/models/Tencent-Hunyuan/Hy3) 等平台获取。Transformers 中对应架构为 `hy_v3`（`HYV3ForCausalLM`），每层含 192 路 routed experts（top-8 激活）与 1 个 shared expert，路由采用 Sigmoid 评分与 `expert_bias` 负载均衡，并支持 MTP（Multi-Token Prediction）投机解码分支。

- msModelSlim 已适配 Hy3 面向 Ascend 推理的 **W8A8 一键量化**最佳实践，适配器会在量化过程中自动完成 MoE unstack 与 MTP 模块加载。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../docs/zh/install_guide/install_guide.md)》。
- 请在源码根目录执行 editable 安装，以确保 `lab_practice`、`config` 等数据目录正确链接：

  ```bash
  pip install -e .
  ```

- **transformers** 版本需为 **5.6.0**：

  ```bash
  pip install transformers==5.6.0
  ```

- Hy3 模型参数量大，量化对显存要求较高，请确保在显存充足的 NPU 环境下执行。
- 对于 Hy3 模型，请先完成[运行前必检](#hy3-运行前必检)。
- 如需使用 NPU 多卡量化，请先配置环境变量：

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接 | W8A8 | W8A16 | W4A8 | W4A16 | W4A4 | 稀疏量化 | KV Cache | MTP量化 | 量化命令 |
|---------|---------|----------------|------|-------|------|-------|------|---------|----------|--------|---------|
| **Hy3** | Hy3 | [tencent/Hy3](https://huggingface.co/tencent/Hy3) | ✅ | | | | | | | ✅ | [W8A8](#hy3-w8a8量化) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 量化权重生成

- Hy3 模型的 W8A8 量化已集成至《[一键量化](../../docs/zh/user_guide/usage_quick_quantization.md)》，以下提供 Hy3 模型量化权重生成快速启动命令。
- 指定 `--quant_type w8a8` 且 `--model_type Hy3` 时，工具将自动匹配 [hy3_w8a8.yaml](../../lab_practice/hy3/hy3_w8a8.yaml) 最佳实践配置。

一键量化命令参考[《一键量化使用指南》](../../docs/zh/user_guide/usage_quick_quantization.md)。

## 使用示例

- 请将`${model_path}`和`${save_path}`替换为用户实际路径。
- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)

### Hy3系列

#### <span id="hy3-运行前必检">运行前必检</span>

Hy3 模型较大，且 MoE 结构在量化前需进行 expert unstack 转换，为了避免浪费时间，还请在运行命令前，根据以下必检项对相关内容进行确认。

- 1、需安装 **transformers==5.6.0**（Hy3 的 `hy_v3` 架构自该版本起合入官方库）。
- 2、需在源码根目录执行 **`pip install -e .`**，确保 `msmodelslim/lab_practice/hy3/` 目录可被一键量化正确加载。
- 3、`--model_type` 须精确填写 **`Hy3`**（与 `config.ini` 注册名及最佳实践 YAML 中 `verified_model_types` 一致，区分大小写）。
- 4、Hy3 权重目录需为 HuggingFace 标准格式（含 `config.json`、`model.safetensors` 或分片索引文件）。

安装后自检：

```shell
pip install -e .
ls msmodelslim/lab_practice/hy3/hy3_w8a8.yaml
python -c "from importlib.metadata import entry_points; print('Hy3' in {e.name for e in entry_points().select(group='msmodelslim.model_adapter.plugins')})"
```

#### <span id="hy3-w8a8量化">Hy3 W8A8量化</span>

- 生成 Hy3 模型 W8A8 动态量化权重（权重 per-channel INT8、激活 per-token INT8）

  ```shell
  msmodelslim quant \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --device npu \
    --model_type Hy3 \
    --quant_type w8a8 \
    --trust_remote_code True
  ```

- 其中`${model_path}`为 Hy3 浮点权重路径，`${save_path}`为量化后的权重保存路径。
- 该一键量化命令匹配使用的量化配置文件为[hy3_w8a8.yaml](../../lab_practice/hy3/hy3_w8a8.yaml)，可以在其中查看具体的量化策略。
- 量化结果以 **AscendV1** 格式保存，默认分片大小 4GB。

**量化策略摘要：**

| 配置项 | 说明 |
|--------|------|
| 量化范围 | 除排除项外的全部 Linear 层 |
| 排除层 | `lm_head`、MoE 路由 `*.router.gate`、专家偏置 `*.mlp.expert_bias` |
| 保存格式 | `ascendv1_saver`，`part_file_size: 4` |

**适配器自动处理：**

1. **MTP 模块加载**：检测 checkpoint 中 MTP 权重（`enorm`、`hnorm`、`eh_proj`、`final_layernorm`），存在则自动扩展层数并加载。
2. **MoE Unstack**：逐层 visit 时将 fused `HYV3MoE` 拆分为逐专家 `nn.Linear`，使 routed experts 可被 `linear_quant` 量化。
3. **逐层前向**：标准 Transformers decoder layer 逐层 visit，支持分布式多卡量化。

若自动匹配失败，可显式指定配置文件：

  ```shell
  msmodelslim quant \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --device npu \
    --model_type Hy3 \
    --config_path lab_practice/hy3/hy3_w8a8.yaml \
    --trust_remote_code True
  ```

## FAQ

### Hy3量化QA

- Q：指定 `--quant_type w8a8` 后未匹配到 `hy3_w8a8.yaml`，使用了 `default-w8a8` 怎么办？

- A：请依次检查：（1）`--model_type` 是否为 **`Hy3`**；（2）是否已执行 `pip install -e .`；（3）`msmodelslim/lab_practice/hy3/hy3_w8a8.yaml` 是否存在。也可通过 `--config_path lab_practice/hy3/hy3_w8a8.yaml` 显式指定配置。

- Q：量化时报 transformers 版本不匹配怎么办？

- A：请安装 `transformers==5.6.0`。msModelSlim 在加载适配器时会校验依赖版本，版本不符将给出提示。

- Q：MoE 路由层为什么不量化？

- A：`router.gate` 与 `expert_bias` 直接影响专家选择，量化后易导致路由偏移，因此默认排除。

- Q：是否支持 MTP 层量化？

- A：适配器会自动加载 MTP 模块权重；当前 W8A8 最佳实践对 MTP 专用子模块（`enorm`、`hnorm`、`eh_proj`）采用与主干层一致的动态量化策略，路由相关层保持浮点。
