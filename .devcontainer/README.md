# Dev Container 快速指南（msModelSlim）

## 🛠️ 使用前提

| 方式 | 工具 | 说明 |
| --- | --- | --- |
| 远程服务器（推荐） | VS Code + `Dev Containers` + `Remote - SSH` | Linux 服务器已启用 Docker |
| 本地 PC | VS Code + `Dev Containers` | Docker Desktop（Linux 模式） |

> ⚠️ **安全提示**：默认配置启用了 Host 网络模式及高权限（`--privileged`），请务必在可信环境中使用。

## 🚀 3步开工

1. **打开项目**：在 VS Code 中打开本代码目录。
2. **加载容器**：`F1` → **Dev Containers: Reopen in Container**。
3. **进入开发**：待初始化日志打印 **READY**（见 `/tmp/msmodelslim-post-create.log`）后即可编码、单测、调试与合入。

## ⏱️ 自动化流程（首次）

| 阶段 | 动作 | 持久化方式 |
| --- | --- | --- |
| 1. 环境拉取 | 拉取 SWR 预置镜像并部署 VS Code Server | — |
| 2. 身份与缓存 | `initialize.sh` 生成 `.host-gitconfig`、预建 pip/pre-commit 缓存目录 | 宿主 `~/.cache/{pip,pre-commit}` bind mount |
| 3. 初始化 | `post-create.sh` 建 `.venv`、拷贝数据目录、`pip install .`、装 pre-commit 钩子并**后台预热** Hook 环境 | `/workspace/.venv` 随仓库 bind mount 持久 |

镜像为华为云 SWR 上的 MindStudio 标准构建镜像（`swr.cn-north-4.myhuaweicloud.com/mindstudio-image/mindstudio-build:26.2.0-0801`）。镜像制作细节见《[MindStudio 统一构建镜像制作指南](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/docker_image_build_guide.md)》；无镜像权限的外部贡献者可走《[安装指南](../docs/zh/install_guide/install_guide.md)》源码安装。

## 🔨 编译与单元测试

环境就绪后，通过 VS Code 菜单栏 **`Terminal`** > **`Run Task`** 即可调用预设的自动化任务：

| 任务名称 | 功能说明 |
| :--- | :--- |
| `msmodelslim: UT reduced (modelslim_v1)` | 精简单元测试（等价 `python3 build.py test local`） |
| `msmodelslim: UT full` | 全量单元测试（等价 `python3 build.py test local -e full_ut=true`） |
| `msmodelslim: Lint (pre-commit all files)` | 全量 pre-commit 检查（等价 `pre-commit run --all-files`） |
| `msmodelslim: Build wheel (local)` | 构建 wheel，产物输出至 `artifacts`（等价 `python3 build.py local`） |
| `msmodelslim: Clean build/test artifacts` | 清理工作区内的构建缓存与测试产物 |

*也可直接在终端执行 `python3 build.py [test]` 等命令，其功能与上述 Build/Test 任务一致。*

## ♻️ 环境复原：毁坏无忧

若开发过程中容器环境搞乱或损坏，无需重新搭建：只需按 `F1` 键选择 **Dev Containers: Rebuild Container**，即可瞬间获得一个全新的纯净环境！

## ❓ 常见问题

**修改 `.vscode/settings.json` 后 `git pull` 冲突？**

该文件被 `skip-worktree` 以支持个性化，远端更新时会拒绝覆盖。用别名安全拉取：

```bash
git sync-up
```

（post-create.sh 已注册该别名：临时取消 skip-worktree → `git pull` → 恢复。）

**首次 Lint 很慢？**

pre-commit 首次要下载并初始化各 Hook 环境；post-create 已在后台预热（`pre-commit install-hooks`），缓存落在 `~/.cache/pre-commit`，重建不重复下载。若预热未完成，首次 `pre-commit run` 仍会较慢，属正常。
