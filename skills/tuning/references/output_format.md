# 结果输出

**Load when:** 量化配置调优完成、进入最终结果整理与回显时。

## 交付物

| 交付物 | 来源/路径 | 说明 |
|--------|-----------|------|
| 达标量化权重 | `{save_path}/round_{N}/quantized` | 最终达标轮次（最优一轮）的量化模型文件 |
| 评测报告 | `{save_path}/evaluate_report.*` | 各数据集精度分数与目标对比（含浮点基线） |
| 调优历史记录 | `{save_path}/history` | `tuning/scripts/history_append.py` 逐轮写入；与编排脚本实际路径一致 |
| 最终 Practice | `{save_path}/practice_round_{N}.yaml` 及入库副本 | 收敛配置；入库见「Practice 入库」 |
| 精度缓存 | `{save_path}/history/accuracy.yaml` | `accuracy_append.py` 写入，供续跑复用 |
| 审计汇总 | `{save_path}/audit/audit_log.md` | 主编排按子任务聚合，人类一次性复盘 |

## 审计汇总

每个子任务回传后，主编排在 `{save_path}/audit/audit_log.{json,md}` **按子任务聚合**：

- `subagent_type` + 时间戳；
- `input`：该子任务的委派输入（SUBAGENT_IO 信封）；
- `commands`：执行者回传的关键节点（含 `command`/`exit_code`/`fix`/`output_path`，原样收录，不得虚构）；
- `deliverables`：产物路径清单；
- `status` + `output` / `error`。

人类打开 `audit_log.md` 即可一次性复盘全流程（谁 / 被给了什么 / 执行了什么命令 / 遇错如何修复 / 交付了什么 / 结果如何）。审计日志与最终交付物一并归档，不得删除。执行者责任与字段规范见 `subagent_io_protocol.md`「执行者关键节点记录与审计汇总」。

## 回显格式

向用户展示（简短，不输出长日志）：

```text
调优完成:
- 达标轮次: round_{N}
- 模型产物路径: {save_path}/round_{N}/quantized
- 各数据集精度 vs 目标: gsm8k 83.5% / 83.0% ✅ ...
- 调优历史: {save_path}/history
- 审计汇总: {save_path}/audit/audit_log.md
- 总耗时: {duration}
- 可复现命令:
  msmodelslim quant --model_path ... --save_path ... --config ... ...
  python skills/evaluation/evaluate/scripts/run_evaluation.py ...
```

## 收敛确认

回显后须获得用户认可，调优流程才算圆满完成：

- 用户无异议 → 执行「Practice 入库」并结束。
- 用户有异议（如目标不合理、想继续调优、换数据集）→ 按反馈回到对应阶段（调整目标重跑、续跑调优、切全集验证等），直至用户满意。

## Practice 入库

收敛确认后，调用 `tuning/scripts/finalize_practice_repo.py` 将最终 Practice 写入实践库（供后续 `msmodelslim quant` 复用）：

```bash
python skills/tuning/scripts/finalize_practice_repo.py \
  --model-type "${MODEL_TYPE}" \
  --model-path "${MODEL_PATH}" \
  --final-practice-path "${save_path}/practice_round_{N}.yaml" \
  [--trust-remote-code]
```

> **执行位置注意（实测 F0-2）**：本脚本 `import msmodelslim`（PluginModelFactory 等）。请在**仓库外**工作目录以脚本**绝对路径**调用；仓库根内执行会命中源码树残缺包（缺 `config/`）报 `SecurityError`。

- 脚本内部校验 `model_adapter` 是否实现 `ModelInfoInterface` 且 practice 仓库支持保存；不支持时返回 `ok: false` + `error_code=PRACTICE_SAVE_SKIPPED`（显式失败，不静默视为成功）。此时执行**替代归档**：保留 `practice_round_{N}.yaml`，在回显与审计中记录未入库原因与可复现配置路径；不阻塞交付。
- 入库成功后可向用户提示：后续 `msmodelslim quant` 在同一 `MSMODELSLIM_CUSTOM_PRACTICE_REPO` 环境指定相同 `quant_type` 即可自动匹配该配置。

## 磁盘清理

- 磁盘中同时**最多存储 2份**完整量化权重：最终达标轮次的权重与当前迭代权重（或上一已知达标轮次）。
- 其余无用权重删除释放空间；**严禁**使用 `rm -rf`，一律使用 `rm -r`，删除前确认路径。
- 最终回显时确认 `{save_path}` 下仅保留必要产物（round 目录、history、report、practice yaml）。
