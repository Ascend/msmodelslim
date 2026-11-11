# C1 安装环境

> 只处理“环境是否搭对、版本是否匹配、镜像/硬件是否支持”。
> 具体 traceback 的根因定位见 [C4 故障排查](C4-error-troubleshooting.md)。

## 1. 适用问题

- msModelSlim 安装失败、依赖冲突、`pip install` 后命令不可用；
- CANN、torch、torch_npu、transformers、Python 版本不匹配；
- 镜像选择、容器环境、源码目录执行导致的路径问题；
- 昇腾硬件平台差异：A2 / A3 / 310P / 300I DUO / 950；
- SoC、算子、kernel、驱动相关的环境问题；
- 量化任务因显存/内存/资源被杀（环境侧判断，根因排查转 C4）。

## 2. 不适用（转交）

| 问题 | 转到 |
|---|---|
| 某个报错的根因与修复 | [C4 故障排查](C4-error-troubleshooting.md) |
| 模型/量化模式是否支持 | [C2 能力边界/支持判定](C2-feature-support.md) |
| 怎么执行量化命令 | [C3 操作指引](C3-quantization-execution.md) |
| 量化权重能否在 vLLM/MindIE 拉起 | [C5 部署推理](C5-deployment.md) |
| 量化后掉点 | [C6 精度性能](C6-accuracy-performance.md) |

## 3. 必查证据（按顺序）

| 顺序 | 资料 | 用途 |
|---|---|---|
| 1 | `docs/zh/install_guide/install_guide.md` | 在线/离线/源码安装、Python/CANN 要求、源码目录注意事项 |
| 2 | `docs/zh/support/faq_installation.md` | 依赖冲突、pydantic/accelerate、安装类 FAQ |
| 3 | `docs/zh/support/faq_runtime.md` | Killed/OOM/显存/内存 |
| 4 | `docs/zh/support/faq_hardware.md` | 硬件/算子兼容（如为预留页，按 C4 走） |
| 5 | `docs/zh/support/faq_warning.md` | 告警/DeprecationWarning 等 |
| 6 | `config/config.ini` 的 `[ModelAdapterDependencies]` | 模型要求的 transformers 版本 |
| 7 | `example/<family>/README.md` | 该模型实际验证过的依赖与命令 |
| 8 | 现场环境 | `pip show` / `npu-smi info` / `python -V` |

## 4. 判定流程

```text
① 确认用户是“装不上”还是“装上了但跑不对”
② 对齐三层版本：CANN ↔ torch/torch_npu ↔ transformers/Python
③ 优先使用与推理引擎一致的官方镜像，避免宿主环境混装
④ 若问题是“config 不存在/源码目录”，先确认安装后是否离开源码根目录
⑤ 若涉及 310P/300I/950，先确认硬件是否在支持范围，再谈量化
```

## 5. 回答要点

1. **优先推荐官方镜像**：vLLM-Ascend / MindIE 官方镜像已经把 CANN、torch_npu、Python 配好；客户环境里自行混装是最常见的故障源。需要隔离时，用推理引擎镜像创建独立容器，避免与宿主环境互相污染。
2. **transformers 以模型 example 为准**，不要用“最新版本”替代模型验证版本；`config.ini` 的 `[ModelAdapterDependencies]` 是第二依据。
3. **安装后离开源码根目录执行量化**，否则 `config/config.ini` 相对路径会失效。
4. **310/310P 只保证有限能力**：量化主流程面向 910；310 精度不保证，能推理不等于能量化。310P 可先尝试双芯量化（用 `ASCEND_RT_VISIBLE_DEVICES` 指定两张卡），但精度仍不保证。
5. **环境类回答必须给自检命令**，不能只给结论。

### 版本自检命令

```bash
python -V
pip show msmodelslim transformers torch torch_npu
npu-smi info
# CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || echo "check CANN install path"
```

### 镜像与环境建议

```text
优先顺序：
1) 与部署引擎一致的官方镜像（vLLM-Ascend / MindIE）
2) 官方发布机器 + 官方镜像
3) 自建环境：按 install_guide 安装 msModelSlim，再按模型 example 对齐 transformers
```


## 6. 常见错误

| 错误 | 纠正 |
|---|---|
| 直接建议“升级到最新 transformers” | 必须按模型 example / `config.ini` 的验证版本 |
| 把算子错误当模型不支持 | 先按 C1/C4 判断环境，再回 C2 |
| 让用户在源码根目录执行 | 违反红线；安装后换目录 |
| 310P 上承诺量化精度 | 只能说明有限支持，精度不保证 |
| 只给“重装试试” | 必须给版本自检与镜像建议 |

## 7. 回答骨架

```markdown
【结论】环境问题：<版本/镜像/目录/硬件> 不匹配，不是模型不支持。
【依据】<install_guide / faq_installation / 模型 example / 现场版本>
【操作】
1. <自检命令>
2. <对齐版本或更换官方镜像>
3. <重新安装/换目录执行>
【适用条件/风险】<CANN/torch_npu/transformers 版本；310 精度不保证>
【下一步】仍失败请提供 `pip show ...`、`npu-smi info`、完整命令与 traceback；必要时转 installation。
```
