# 路由决策（EP 并行适配）

**Load when:** 用户输入参数已回显并获得认可后、进入环境准备之前，判断本次调优是否需要在量化阶段切入 EP 并行适配。

## 目的

编排层只做「路由」，不展开 EP 的实现细节（MoE 检查、专家分片、权重按 rank 加载、mapping 本地化等细节均属于 `adaptation/ep` Skill）。**调优主流程（环境准备 / 模型准备 / 量化配置调优 / 结果输出）始终留在本 Skill，不整段委派给 EP Skill。**

## 判定规则

在用户输入阶段（`references/user_input.md`）提取到的「设备索引」基础上判定：

| 条件 | 路由去向 |
|------|----------|
| 设备卡号数量 ≥ 2（如 `npu:0,1`、`npu:2,3`、`[0,1,2,3]`） | 主流程留在本 Skill；量化前委派 `adaptation/ep` |
| 设备为单卡（如 `npu:0`、`npu:3`） | 保持本 Skill 普通单卡流程，不委派 EP 适配 |
| 用户明确说明「不用多卡 / 不用 EP / 只用单卡」 | 保持本 Skill 普通单卡流程（即便环境存在多卡） |

## 路由动作

满足多卡条件时：

1. 将已对齐的参数透传给 EP 适配 Skill，至少包括：
   - `model_path` / `model_type`
   - 设备列表与卡数（device_list / device_count）
   - 量化方案（W8A8 / W4A8，用于判断 mapping 本地化项）
   - 工作目录（save_path）
2. 由 EP 适配 Skill 完成「MoE 检查 + EP 就绪检查与适配 + `[EP_CHECK]` 验证」，回传 `EP_ADAPT_RESULT` 与 `requires_ep`。
3. 本 Skill 据回传决定后续调优方式：
   - `requires_ep=true` → **后续调优全程开启 EP 并行**：每一轮量化命令固定使用多卡（`--device npu:0,1,...`），每轮量化日志须含 `[EP_CHECK]`，评测服务保持多卡，中途不得退回单卡 / DP；
   - `requires_ep=false` → 退回本 Skill 普通多卡 / 单卡流程，不涉及专家分片。
4. 透传协议与 subagent 委派仍遵守 `subagent_io_protocol.md`（SUBAGENT_IO v1）。

## 与 MoE 检查的职责边界

- 本 Skill **不在路由阶段**判定模型是否 MoE；该判定由 EP 适配 Skill 的 MoE 模型检查完成。
- EP 适配 Skill 判定为 MoE → 完成 EP 适配并回传 `requires_ep=true`，本 Skill 后续量化走多卡 EP。
- EP 适配 Skill 判定为**非 MoE**（无 routed experts）→ 回传 `requires_ep=false`，本 Skill 走普通多卡 / 单卡量化流程。

## 委派契约

`adaptation/ep` 的 SUBAGENT_IO v1 委派契约（input/output 字段、委派模板与回传示例）已并入模型准备阶段文档：[prepare_model.md](./prepare_model.md)「Agent: adaptation/ep」。本文件只负责**路由判定**（何时触发），不重复定义契约。

## 回显要求

路由决策确定后，向用户简短回显：命中的路由去向（EP 多卡适配 / 普通单卡）及触发原因，再进入环境准备阶段。
