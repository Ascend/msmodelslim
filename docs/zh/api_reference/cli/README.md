# msmodelslim 命令行 API 总览

`msmodelslim` 是 MindStudio msModelSlim 的命令行入口，提供量化、敏感层分析、伪量化精度评估与自动调优四类子命令。本页汇总全局参数、子命令清单与各子命令文档入口；各子命令的参数细节、参数关系与使用示例见对应文档。

## 1. 命令格式

```text
msmodelslim <command> [<args>]
```

符号说明：

- `<command>` 为子命令名（位置参数），省略时或传入未知子命令时打印顶层帮助。
- 尖括号内为需替换的值，方括号内为可选参数。
- `-h`/`--help`、`-V`/`--version` 可在顶层使用，也可放在任意子命令后使用（如 `msmodelslim quant --version`）。

## 2. 子命令清单

| 子命令 | 功能 | 文档 |
|--------|------|------|
| `quant` | 一键量化或权重转换，并保存量化权重与描述文件 | 《[msmodelslim quant 命令行 API 文档](msmodelslim_quant.md)》 |
| `analyze` | 量化前敏感层分析，输出敏感度最高的层名列表 | 《[msmodelslim analyze 命令行 API 文档](msmodelslim_analyze.md)》 |
| `eval` | 基于 AscendV1 导出做伪量化模型精度评估 | 《[msmodelslim eval 命令行 API 文档](msmodelslim_eval.md)》 |
| `tune` | 自动调优量化配置以达成目标精度 | 《[msmodelslim tune 命令行 API 文档](msmodelslim_tune.md)》 |

## 3. 全局参数

全局参数在顶层解析，适用于全部子命令：

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--help` | `-h` | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 显示帮助信息后退出。顶层显示子命令清单，子命令后显示该子命令的参数说明。 |
| `--version` | `-V` | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 显示版本信息（版本号、构建信息、仓库地址）后退出。该参数在解析子命令前处理，因此在任意位置传入均生效。 |

## 4. 公共参数

以下参数由各子命令分别注册（`quant`、`analyze`、`eval`、`tune` 均已提供），写在子命令之后：

| 参数 | 别名 | 类型 | 传入形式 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 |
|------|------|------|----------|-----------|--------|----------------|------|
| `--log_level` | 无 | `string` | 单值 | 可选 | `info` | `debug`、`info`、`warning`、`error` | 日志级别。显式传入时优先于 `-v`/`-q`。 |
| `--verbose` | `-v` | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 提高输出详细程度（等价 `--log_level debug`）。 |
| `--quiet` | `-q` | `bool` | 不带值开关 | 可选 | 关闭 | 传入即启用 | 抑制非错误输出（等价 `--log_level error`）。 |

## 5. 使用示例

### 5.1 查看版本与顶层帮助

```bash
msmodelslim --version
msmodelslim -h
```

`-V`/`--version` 输出 `msmodelslim <版本> (<git hash>)` 与构建信息；`-h`/`--help` 输出子命令清单。两者均不依赖 model 与权重路径。

### 5.2 查看某个子命令的帮助

```bash
msmodelslim quant -h
msmodelslim analyze linear -h
```

顶层帮助只列出子命令；子命令的参数说明须在子命令后加 `-h`/`--help` 查看。

## 6. 退出码与异常处理

正常运行则无异常。失败时抛出异常，具体原因见错误日志。`-V`/`--version` 与 `-h`/`--help` 正常结束，不抛出异常。
