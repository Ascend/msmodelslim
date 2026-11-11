---
name: update-interface-docs
title: 更新接口文档（量化配置 / CLI）
triggers:

  - "更新接口文档"
  - "更新配置文档"
  - "更新CLI文档"
  - "生成配置文档"
  - "重新生成配置文档"
  - "接口文档生成"
  - "配置文档漂移"

target_dir: docs/zh/api_reference/
create_subdir: false
templates: []
---

# 更新接口文档（量化配置 / CLI）

## 目标形态

保持 `docs/zh/api_reference/` 下接口文档与源码实现一致：

- **量化配置文档**：`docs/zh/api_reference/config/` 下的生成稿由源码注解自动生成，带 `generated-by` 标记，不手工编辑。完整规范和生成器用法见《[配置文档生成工具](../../../tools/config_docgen/README.md)》。
- **CLI 文档**：`docs/zh/api_reference/cli/msmodelslim_{quant,analyze,tune}.md` 为手工维护，按模板 05 编写。

## 命名规则

- 沿用现有文件名，不新建或重命名生成稿。
- 量化配置文档的文件名由生成器按公开 `type` / 类名 slug 决定；顶层任务配置固定为 `modelslim_v1.md`、`multimodal_vlm_modelslim_v1.md`、`multimodal_sd_modelslim_v1.md`、`modelslim_convert.md`、`practice_config.md`。

## 操作步骤

1. **量化配置文档**：修改 Pydantic 模型字段、校验器 docstring 或配置类 docstring 后，运行：

   ```bash
   python -m tools.config_docgen
   ```

   仅检查漂移时运行：

   ```bash
   python -m tools.config_docgen --check
   ```

   预期产物：`docs/zh/api_reference/config/` 下生成稿已重新生成且 `--check` 无 drift。

   > 说明：项目级生成器统一通过 `python -m tools.config_docgen` 调用。

2. **CLI 文档**：对照 `msmodelslim.cli` 实现与 `--help` 修改 `docs/zh/api_reference/cli/*.md`，使用示例子标题按 `N.M` 编号。
3. **校验**：对变更文件按《[量化配置文档校验清单](../../../docs/zh/contributing/development_guide/docs_standards/04_config_document_checklist.md)》或《[命令行 API 文档校验清单](../../../docs/zh/contributing/development_guide/docs_standards/05_cli_api_contract_checklist.md)》检查，并叠加《[公共校验清单](../../../docs/zh/contributing/development_guide/docs_standards/00_common_checklist.md)》。

## 校验依据

- 配置文档规范与生成器用法：《[配置文档生成工具](../../../tools/config_docgen/README.md)》
- 模板与清单：`docs/zh/contributing/development_guide/docs_standards/` 下 04 / 05 模板与校验清单
