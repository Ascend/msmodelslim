# Contributing to Mindstudio ModelSlim

感谢您考虑为Mindstudio ModelSlim贡献力量! 我们欢迎任何形式的贡献——无论是修复错误，功能增强，文档改进，或是任何反馈建议。无论您是经验丰富的开发者，还是第一次参与开源项目，您的帮助都非常宝贵。

您的支持可以有很多种形式：

- 报告问题或意外行为。
- 建议或实现新功能。
- 改进或扩展文档。
- 审阅Pull Request并协助其他贡献者
- 分享推荐：在博客文章、社交媒体中介绍MindStudio ModelSlim，或为仓库项目点个🌟。

我们期待您的参与！

## 1. 寻找可以贡献的Issue

正在寻找新issue的切入点？ 可以查看以下议题：

- [good-first-issue](https://gitcode.com/Ascend/msmodelslim/issues?categorysearch=%255B%257B%22field%22:%22order_by_sort%22,%22value%22:%22created_at_desc%22,%22label%22:%22%E6%9C%80%E8%BF%91%E5%88%9B%E5%BB%BA%22%257D,%257B%22field%22:%22labels%22,%22value%22:%255B%257B%22id%22:22797,%22name%22:%22good-first-issue%22%257D%255D,%22label%22:%22good-first-issue%22%257D%255D&state=all&order_by=created_at&sort=desc&scope=all&page=1)
- [help-wanted](https://gitcode.com/Ascend/msmodelslim/issues?categorysearch=%255B%257B%22field%22:%22order_by_sort%22,%22value%22:%22created_at_desc%22,%22label%22:%22%E6%9C%80%E8%BF%91%E5%88%9B%E5%BB%BA%22%257D,%257B%22field%22:%22labels%22,%22value%22:%255B%257B%22id%22:22796,%22name%22:%22help-wanted%22%257D%255D,%22label%22:%22help-wanted%22%257D%255D&state=all&order_by=created_at&sort=desc&scope=all&page=1)
- 除了上述两个新手友好issue外，我们也提供了其他的[issue模板](https://gitcode.com/Ascend/msmodelslim/tree/master/.gitcode/ISSUE_TEMPLATE)来作为参考。
- 此外，您也可以通过 [RFC](https://gitcode.com/Ascend/msmodelslim/issues?categorysearch=%255B%257B%22field%22:%22order_by_sort%22,%22value%22:%22created_at_desc%22,%22label%22:%22%E6%9C%80%E8%BF%91%E5%88%9B%E5%BB%BA%22%257D,%257B%22field%22:%22labels%22,%22value%22:%255B%257B%22id%22:25328,%22name%22:%22rfc%22%257D%255D,%22label%22:%22rfc%22%257D%255D&state=all&order_by=created_at&sort=desc&scope=all&page=1) 和 [Roadmap](https://gitcode.com/Ascend/msmodelslim/issues?categorysearch=%255B%257B%22field%22:%22labels%22,%22value%22:%255B%257B%22id%22:22807,%22name%22:%22roadmap%22%257D%255D,%22label%22:%22roadmap%22%257D,%257B%22field%22:%22order_by_sort%22,%22value%22:%22created_at_desc%22,%22label%22:%22%E6%8E%92%E5%BA%8F%22%257D%255D&state=all&order_by=created_at&sort=desc&scope=all&page=1)来了解开发计划与规划。

## 2. Pull Requests 与 Code Reviews

感谢您提交 PR！为优化审查流程，请遵循以下指南：

遵循我们的 Pull Request [模板与规范](https://gitcode.com/Ascend/msmodelslim/blob/master/.gitcode/PULL_REQUEST_TEMPLATE.md)

参考开发者文档 《[LLM 大模型接入指南](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/development_guide/integrating_models/)》

参考开发者文档 《[编码规范](../common/coding_standards.md)》。

提交代码前须在本地通过 pre-commit 检查（首次开发请执行 `pre-commit install`，详见《[编码规范](../common/coding_standards.md)》）。

对涉及用户端功能的改动，请同步更新对应的用户和开发者文档

在 CI 工作流中 添加或更新测试；若无需测试，请说明原因

在上述准备工作完成后提交代码，请输入 compile 命令触发机器人编译流水线

流水线编译通过后请联系[仓库管理和维护成员](https://gitcode.com/Ascend/msmodelslim/member)进行检视与合入

Pull Request需要依次集齐如下四个标签即可完成代码合入：

   1. ascend-cla/yes：CLA检查，首次开发时需要完成CLA的签署，完成后每次提交自动获得此标签。
   2. ci-pipeline-passed：CI流水线，在Pull Request流程中评论`compile`触发，若CI流水线检查不通过，则需要根据提示修改后重新提交。
   3. lgtm：由Reviewers提供，Reviewers审核通过后，会在Pull Request流程中评论`/lgtm`触发lgtm标签。
   4. approved：由Committers提供，Committers审核通过后，会在Pull Request流程中评论`/approved`触发approved标签。

   当您的Pull Request集齐四个标签后，您的PR将被合并到主干分支。

### 2.1 Pull Request最佳实践

- 保持PR的大小适中，便于审查
- 一个PR只解决一个问题或实现一个功能
- 及时响应审查意见
- 保持与主分支同步，及时解决冲突

## 3. 构建与测试

在提交PR之后，评论 compile 即可触发流水线PR-pipeline，平台会自动进行编译、构建、代码检查和开发者测试。如有错误，请根据报错自行整改，疑问请咨询[仓库管理和维护成员](https://gitcode.com/Ascend/msmodelslim/member)

### 3.1 PR标题与分类

只有特定类型的PR才会被审核。请在PR标题前添加合适的前缀，以明确PR类型。请使用以下分类之一:

- `[Feature]`: 新功能相关代码。
- `[Bugfix]`: bug修复相关代码。
- `[Doc]`: 文档相关代码。
- `[Test]`: 开发者测试相关代码。

### 3.2 Commit Requirement

为保持commit记录清晰，请确保每个PR仅包含一个commit。
如果您的PR当前包含多个commits，请在提交前使用以下任一方法（包括但不限于）将其合并为单个commit。(尽管GitCode在合并PR时提供了`Squash 合并`的选项, 提前将PR整理为单个简洁的commit仍然被视为最佳实践，并且深受committer们的欢迎。)

#### 3.2.1 方式一：交互式变基（推荐）

- 查看需要合并的最近几个commit（例如最近3个）：

``` bash
git log --oneline -n 3
```

- 启动交互式rebase (将`N`替换为需要合并的commit数量):

``` bash
git rebase -i HEAD~N
```

- 在打开的编辑器中:
    - 保留第一个commit的`pick`。
    - 将其余commit前的`pick`修改为`squash`(或简写为`s`) 。
- 保存并关闭。随后会打开新窗口，供您编写合并后的简洁、有意义的commit信息。
- 强制推送更新后的分支 (仅限于您自己的特性分支):

``` bash
git push --force-with-lease origin your-branch-name
```

#### 3.2.2 方式二：reset + 新建commit

```bash
# 获取最新的待合入的目标分支（如master）
git fetch origin master

# Soft-reset到主干分支--此操作会保存所有修改并回归到暂存区。
git reset --soft origin/master

# 将所有更改提交为一个新的commit
git commit -m "feat: concise description of your change"

# 强制推送以更新PR分支
git push --force-with-lease origin your-branch-name

```

> 提示: 如果您不确定应基于哪个目标分支，请查看仓库的默认分支或咨询Maintainer.
<br/>
> 警告：切勿对共享或受保护的分支执行强制推送。

### 3.3 单元测试指南

如果您需要运行或编写 msModelSlim 单元测试，请参阅详细的《[开发者测试指南](../common/testing_guide.md)》。该指南涵盖了：

- 准备环境与安装依赖。
- **单元测试执行**：运行 modelslim_v1 模块测试。
- **单元测试编写**：目录结构映射、命名规范、场景化测试设计、conftest.py 使用等。

## 4. 感谢

我们感谢您对 MindStudio ModelSlim 的贡献。您的每一份努力，都让这个项目变得更强大、更易用。祝您创造愉快，编程开心！
