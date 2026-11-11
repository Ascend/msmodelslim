# C4 故障排查

> 只做“有报错/有 traceback 的根因定位与修复”。
> 无报错但效果差（掉点、乱码、无收益）→ [C6 精度性能](C6-accuracy-performance.md)。
> 量化权重拉起失败 → 先按本文定位，再转 [C5 部署推理](C5-deployment.md)。

## 1. 适用问题

- `ImportError` / `ModuleNotFoundError` / `AttributeError`；
- `config doesn't exist`、路径/权限错误；
- `Get model from pretrained failed`、`state_dict` / `size mismatch`；
- SoC、算子、kernel、`aclnn`、`aicore timeout`；
- OOM / `Killed` / 显存/内存不足；
- 量化保存、权重缺失、量化配置校验失败；
- 服务拉起时的初始化失败（根因确认后转 C5）。

## 2. 不适用（转交）

| 问题 | 转到 |
|---|---|
| 纯“是否支持” | [C2 能力边界/支持判定](C2-feature-support.md) |
| 怎么写出正确命令/YAML | [C3 操作指引](C3-quantization-execution.md) |
| 环境/版本/镜像怎么配 | [C1 安装环境](C1-installation-environment.md) |
| 量化权重部署/引擎版本 | [C5 部署推理](C5-deployment.md) |
| 无报错但掉点/无收益 | [C6 精度性能](C6-accuracy-performance.md) |

## 3. 排查步骤

```text
① 收集：完整命令、traceback 最后50行、模型、硬件、CANN/torch/transformers 版本
② 关键词定位：先用下表匹配“错误串”，不要只看错误码
③ 判断根因类别：安装/路径、模型加载、环境/算子、量化/保存、部署
④ 给最小修复步骤；能自检的先给自检命令
⑤ 未命中：检索 docs/FAQ/代码；仍无 → 声明“资料暂未覆盖”并要材料
```

**必须收集的现场信息**：

```bash
msmodelslim quant ...            # 完整命令
python -V
pip show msmodelslim transformers torch torch_npu
npu-smi info
```

> **脱敏**：请客户提供命令、日志和配置前，先提示删除令牌、Cookie、URL 凭据、私网 IP、内部主机名和用户路径；只保留定位所需的最小日志。

## 4. 错误速查

### 4.1 安装 / 路径

| 关键词 | 根因 | 首查 | 处理 |
|---|---|---|---|
| `msmodelslim/config doesn't exist`、`Code: 204` | 在源码目录运行 / 安装不完整 | `docs/zh/install_guide/install_guide.md` | `bash install.sh` 后离开源码目录执行 |
| `ModuleNotFoundError: msmodelslim...` | 安装不完整 / 源码与安装包混用 | 安装指南 / FAQ 安装 | 卸载重装当前 whl；检查 Python 版本/架构 |
| `No module named 'example.common'` | 未在仓库内运行 / `example` 被遮蔽 | `example/*/README.md` | 在 example 子目录运行或设置 `PYTHONPATH` |
| `unsupported operand type(s) for \|` | Python 3.9 与 PEP 604 注解 | 安装指南版本要求 | 升级 Python 3.10/3.11 |
| `code 507013`、`copy_d2d` | 设备拷贝 / 显存 / 驱动 | 运行 FAQ / 硬件 FAQ | 检查显存/驱动/卡占用；310 换 A2/A3 |

→ 版本/镜像类进一步处理见 [C1 安装环境](C1-installation-environment.md)。

### 4.2 模型加载 / 权重

| 关键词 | 根因 | 首查 | 处理 |
|---|---|---|---|
| `Get model from pretrained failed`、`Code: 201` | transformers 版本 / 权重 / 自定义代码 | 模型 example README | 对齐 transformers；确认来源可信后，仅在模型官方要求时开启 `--trust_remote_code`；校验分片 |
| `cannot import name 'Qwen3_5MoeForConditionalGeneration'` | transformers 过旧 | `config.ini` + Qwen3.6 README | 升级到该模型验证的 transformers 版本 |
| `cannot import name 'causal_conv1d_fn'` | 空/坏 `causal_conv1d` 包 | Qwen3.5 README | `pip uninstall causal_conv1d`；检查同名目录；换目录执行 |
| `Unexpected MTP checkpoint key` | MTP 布局 / `model_type` 不匹配 | 模型 example / 适配器 | 用正确 `model_type`；升级当前 master |
| `size mismatch` / `Error(s) in loading state_dict` | 微调结构 / 权重与 adapter 不一致 | 模型 README / `config.json` | 对比结构；必要时新增适配器 |
| `FileNotFoundError: model-...safetensors` | 权重下载不完整 / index 不一致 | 模型下载说明 | 重新下载 / 校验分片 |
| `does not exist`（如 `e_score_correction_bias`） | buffer 未保存 / adapter 不匹配 | 保存器 / 适配器 | 用正确 adapter；buffer 提升为 Parameter 后重新量化 |
| `No implementation found ... autoround` | 把 AutoRound 写进 `linear_quant` | AutoRound 文档 | 改用 `autoround_quant` 或 `method: ssz/minmax/gptq` |
| `Videos features and video tokens do not match` | transformers 版本 / 视频 processor / metadata 不一致 | 模型 README、视频 FAQ | 对齐 transformers 版本；检查视频处理器与 metadata |

### 4.3 环境 / 算子

| 关键词 | 根因 | 首查 | 处理 |
|---|---|---|---|
| `Unsupported soc version: Ascend910_9382` | CANN/torch_npu 旧、A3 不支持 | 安装指南 / 版本配套 | 升级 CANN/torch_npu；用 A3 官方镜像 |
| `Failed to load the backend extension: torch_npu` | torch_npu 与 torch/CANN 不匹配 | 安装指南 | 安装配套 torch_npu；`source set_env` |
| `aclnnCast failed`、`Parse dynamic kernel config fail` | CANN kernel 包 / 镜像不匹配 | 硬件 FAQ | 检查镜像与硬件版本；`source set_env` |
| `aclnnAmin ... DT_BFLOAT16 not implemented` | 310P 不支持 BF16 | 硬件 FAQ | 换 A2/A3；或 CPU/fp16（需评测） |
| `aicore timeout`、`retCode=0x25` | 镜像/驱动/算子超时 | 硬件 FAQ | 官方镜像；检查卡状态；减小 batch/seq |
| `InplaceIndexAdd` 不支持 | 310P/300I 算子不支持 | 硬件 FAQ | 在 A2/A3 量化；310 只做推理 |
| `NPU out of memory` | 显存不足 | 运行 FAQ | 多卡/更大显存/减少校准 |
| `Killed` | 需区分进程 OOM / 容器 cgroup 限制 / 宿主机 OOM | `faq_runtime.md` | 先看退出码、内存上限、容器事件与 `dmesg`；再按进程/容器/节点层级处置，不要直接清系统缓存或杀其他作业 |

→ 镜像与版本配套见 [C1 安装环境](C1-installation-environment.md)。

### 4.4 量化 / 保存 / 服务拉起

| 关键词 | 根因 | 首查 | 处理 |
|---|---|---|---|
| `SchemaValidateError Code: 203` / `trust_remote_code` | 布尔解析 / 旧版 CLI | CLI 文档 | 用当前 CLI；旧版传 `True` |
| `SecurityError Code: 204` | 路径/权限/源码目录 | 安装指南 | 换目录；检查权限/路径 |
| `Default process group has not been initialized` | 分布式未初始化但代码走 EP | Runner / 模型适配器 | 用支持的 runner；多卡正确初始化 |
| `Engine core initialization failed` | 量化格式 / 推理引擎不匹配 | 部署指南 | 用引擎支持的格式；更新 vLLM-Ascend |
| `W8A8S` 无法拉起 vLLM | vLLM-Ascend 不支持稀疏格式 | 部署指南 | 改普通 W8A8 或联系 vllm-ascend 适配 |
| `ModuleNotFoundError: ...anti_utils` | V0 编译扩展 / Python 版本不匹配 | 安装指南 | 最新仓库 `bash install.sh`；用 V1 |
| `weight_scale` 异常大 | 量化范围 / 旋转 / 读取字段错误 | 算法文档 / 产物描述 | 检查 quant_type 与 scale 语义；回退层 |

→ 引擎与格式匹配见 [C5 部署推理](C5-deployment.md)。


## 5. 未命中处理

1. 在 `docs/zh`、`skills`、`example` 中检索错误关键词；
2. 仍无命中 → 给谨慎结论；同一问题反复出现时按 `docs-management` 补 FAQ；
3. 回答声明“资料暂未覆盖”，并请用户提供：
   - 完整命令；
   - traceback 最后50行；
   - `pip show msmodelslim transformers torch torch_npu`；
   - `npu-smi info` 摘要；
   - 模型 `config.json` 关键字段。

## 6. 回答骨架

```markdown
【结论】根因：<一句话>
【依据】<错误串对应代码/FAQ/文档>
【操作】
1. <检查/命令>
2. <修复/替换>
【适用条件/风险】<版本/镜像/硬件限制>
【下一步】若仍失败，请提供完整 traceback 最后50行 + 完整命令 + 版本信息。
```

## 7. 常见错误

| 错误 | 纠正 |
|---|---|
| 只看错误码就下结论 | 必须看完整 traceback 与命令 |
| 把版本问题说成模型不支持 | 先按 C1 对齐版本，再回 C2 |
| 让用户改框架源码 | 优先官方配置/升级；改源码属于最后手段 |
| 对 OOM 只给“换大卡” | 先区分加载 OOM / 校准 OOM / 系统 OOM |
| 报错未命中仍给确定性结论 | 声明资料未覆盖并给出谨慎结论 |
