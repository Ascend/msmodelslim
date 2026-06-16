# 编码规范

所有提交的代码必须通过本地 pre-commit 检查。本文档说明如何使用 pre-commit 以及各项检查规则。

## 快速开始（必读）

执行以下步骤，确保每次 `git commit` 自动运行检查。

### 1. 安装 pre-commit

```bash
pip install "pre-commit>=4.0.0"
```

推荐使用 Python 3.10 环境。

### 2. 安装 git 钩子

在仓库根目录执行一次：

```bash
pre-commit install
```

此后每次 `git commit` 会自动触发检查。

### 3. 提交前检查

先 `git add` 暂存待提交文件，再执行如下命令：

```bash
pre-commit run
```

该命令仅检查**已暂存**的文件，与 `git commit` 时自动触发的范围一致。若未通过，根据提示修复后重新 `git add` 并再次运行。

若只想检查尚未暂存的指定文件，可使用如下命令：

```bash
pre-commit run --files <文件路径>
```

### 4. 正常提交

```bash
git commit -m "your message"
```

若检查通过，提交成功；若失败，提交被拦截。

---

## 检查项（参考）

以下表格列出 pre-commit 会检查的内容，失败时根据提示修复。

### 通用文件检查

| 检查项（钩子名） | 作用 |
|----------------|------|
| trailing-whitespace | 清除行尾空白 |
| end-of-file-fixer | 确保文件末尾有换行符 |
| check-yaml | YAML 语法校验 |
| check-json | JSON 语法校验 |
| check-added-large-files | 阻止提交超大文件 |
| check-merge-conflict | 检测未解决的合并冲突标记 |
| detect-private-key | 检测私钥或敏感凭证泄露 |
| codespell | 拼写检查（适用于注释、文档等） |

### Python 源码检查

| 检查项 | 作用 |
|--------|------|
| ruff-check / ruff-format | 代码风格检查与自动格式化（行宽 120） |
| pylint | 捕获未定义变量、错误导入等逻辑缺陷 |
| bandit | 安全漏洞扫描（如硬编码密码、危险函数） |
| typos | 拼写检查（覆盖所有源文件，包括 Python 代码） |

---

## 白名单配置

当检查工具误报某词汇为拼写错误时，可将其加入白名单。

>[!NOTE]
>
>白名单变更必须随 PR 提交，并在 PR 描述中说明原因。

- typos 白名单：编辑 `pre-commit/typos.toml`
- codespell 白名单：在 `.pre-commit-config.yaml` 中找到 codespell 钩子的 `-L` 参数，追加误报词汇

---

## 常见问题

### 钩子自动修改文件后提交仍然失败

重新 `git add` 修改后的文件，再执行 `git commit`。

### 修改 mkdocs.yml 时 YAML 校验误报

因 pymdownx 使用了 `!!python/name` 等非标准标签。可临时跳过 YAML 校验（不要用 `--no-verify` 跳过全部检查）：

```bash
SKIP=check-yaml git commit -m "your message"
```

### 误报拼写错误

按「白名单配置」章节处理，将词汇加入对应白名单。

### 首次运行较慢

pre-commit 会下载各钩子环境（如 Ruff、Pylint 等），后续运行会使用缓存。

### 何时使用全量检查

首次安装钩子后，或需排查历史遗留问题时，可执行 `pre-commit run --all-files` 扫描整个仓库。日常提交无需全量检查。
