# 开发者测试指南

## 简介

本文档旨在指导开发者在 msModelSlim 项目中运行单元测试，帮助您了解测试环境配置及执行流程，确保能够顺利执行测试。

## 环境要求

### 操作系统

msModelSlim 的单元测试目前仅支持 **Linux** 系统，不支持 Windows 或 macOS。请确保您在 Linux 环境下运行测试。

### Python版本

msModelSlim 推荐使用 Python 3.10 环境进行测试，建议使用 conda 创建独立的虚拟环境：

```bash
# 创建Python 3.10环境
conda create -n ut_py310 python=3.10
conda activate ut_py310

# 确认Python版本
python --version  # 应显示 Python 3.10.x
```

### 必须依赖

运行单元测试需要安装以下依赖包：

```bash
pip install pytest
pip install pytest-mock
pip install coverage
pip install torch==2.1.0
pip install easydict==1.13
pip install einops
pip install "pydantic>=2.10.1"
pip install wcmatch
pip install scipy
pip install pygtrie
pip install accelerate
pip install requests
pip install pyyaml
pip install numpy==1.26.4
pip install transformers==4.51.0
```

> [!NOTE]
>
> numpy 建议使用 1.26.4 版本以避免与 torch 2.1.0 产生兼容性问题。

## 单元测试执行

### 执行用例

使用项目根目录下的 `test/run_ut.sh` 脚本执行单元测试，测试 modelslim_v1 相关模块（app、core、ir、infra、processor、utils），使用 `--modelslim_v1` 参数：

```bash
cd test
bash run_ut.sh --modelslim_v1
```

该参数适用于仅涉及核心量化框架的测试场景，可以显著减少测试执行时间。

### 查看帮助信息

```bash
bash run_ut.sh --help
```

### 输出典型测试

运行 `bash run_ut.sh --modelslim_v1` 后，每个模块会显示测试进度和最终结果。输出格式示例如下：

```text
Running modelslim_v1 related test cases...
===== test session starts =====
collected 57 items

cases/app/analysis/test_analysis_app.py .......                       [ 26%]
cases/app/analysis/test_analysis_methods.py ....................      [ 73%]
cases/app/naive_quantization/test_naive_quantization_app.py ......    [100%]

----- generated xml file: /path/to/report/final_app.xml -----
===== 57 passed, 6 warnings in 6.17s =====

===== test session starts =====
collected 406 items

cases/core/context/test_base_context.py .................    [  4%]
...
----- generated xml file: /path/to/report/final_core.xml -----
===== 392 passed, 14 skipped, 4 warnings in 18.30s =====
```

> [!NOTE]
>
> 测试完成后会看到每个模块的统计信息，格式为 `X passed, Y skipped, Z warnings`。

## 单元测试编写

### 目录结构映射

测试用例的目录结构必须与源码目录结构保持一一对应。测试根目录为 `test/cases/`，其下按模块划分子目录，每个子目录对应 `msmodelslim/` 下的同名模块。

映射规则：`msmodelslim/<module>/<sub>/xxx.py` → `test/cases/<module>/<sub>/test_xxx.py`

以 `core` 模块为例：

```text
源码路径                                测试用例路径
msmodelslim/core/                       test/cases/core/
├── observer/                           ├── observer/
│   ├── minmax.py      ───────────────→ │   ├── test_minmax.py
│   └── histogram.py   ───────────────→ │   └── test_histogram.py
├── quantizer/                          ├── quantizer/
│   ├── base.py        ───────────────→ │   ├── test_base.py
│   └── impl/                           │   └── impl/
│       ├── minmax.py  ───────────────→ │       ├── test_minmax.py
│       └── gptq.py    ───────────────→ │       └── test_gptq.py
└── convert/                            └── convert/
    ├── config.py      ───────────────→     ├── test_config.py
    └── router.py      ───────────────→     └── test_router.py
```

### 新增测试用例

当需要为新增或已有源码文件补充测试时，按以下步骤操作：

1. **定位目标目录**：根据源码路径，找到 `test/cases/` 下对应的目录。如果目录不存在，则创建。

   例如，新增源码文件 `msmodelslim/core/quantizer/impl/new_quantizer.py`，则应在 `test/cases/core/quantizer/impl/` 下创建测试文件。

2. **创建测试文件**：文件命名为 `test_<源码模块名>.py`，如 `test_new_quantizer.py`。

3. **编写测试内容**：按照下文的规范编写测试类和测试方法。

### 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 测试文件 | `test_<模块名>.py` | `test_minmax.py`、`test_context_factory.py` |
| 测试类 | `Test<类名>` | `TestW4A4Quantizer`、`TestMsMinMaxObserver` |
| 测试方法 | `test_<object>_<assertion>_when_<condition>` | `test_quantize_raises_error_when_weight_out_of_range` |

测试方法命名采用三段式结构：

| 段落 | 含义 | 示例 |
|------|------|------|
| `object` | 被测试的方法或属性 | `quantize`、`get_config`、`get_min_max` |
| `assertion` | 对结果的断言描述 | `returns_none`、`raises_error`、`equals_expected` |
| `condition` | 触发该行为的前置条件 | `input_is_empty`、`weight_out_of_range`、`not_updated` |

> [!NOTE]
>
> 命名即文档。当测试失败时，方法名应能直接定位是**哪个场景**出了问题，而非仅仅"某个功能挂了"。例如 `test_quantize_raises_error_when_weight_out_of_range` 失败 → 边界处理有问题；`test_quantize_works` 失败 → 只知道"量化有问题"。

### 测试类与文件规范

- 源码中每个类**必须**对应一个测试类，测试类名以 `Test` 为前缀（如 `W4A4Quantizer` → `TestW4A4Quantizer`）。
- 测试类须添加 docstring，说明被测目标。
- 同一文件中可包含多个测试类，按被测类从上到下排列。
- 每个测试方法须添加中文 docstring，采用 **场景/预期** 格式：`"""场景：xxx。预期：yyy。"""`。
- 每个测试文件必须包含标准版权声明头。
- 导入顺序：标准库 → 第三方库 → 项目内部模块，各组之间空一行；项目内部模块使用完整包路径导入。

### 测试范围与场景设计

#### 测试范围

- **必须覆盖**：所有公开方法和属性（外部接口）。
- **建议覆盖**：包含非平凡逻辑的核心内部/私有方法。

#### 场景化测试设计

**一个测试用例 = 一个场景**，而非一个功能一个用例。

对每个方法，应从以下三类场景设计测试：

| 场景类别 | 说明 | 示例 |
|----------|------|------|
| **Normal** | 典型、合法的输入 | 合法 shape 的 tensor |
| **Boundary** | 边界条件、极限值 | 空 tensor、最小/最大值、单元素 |
| **Exception** | 应抛异常的非法输入 | None 输入、错误类型、超范围值 |

#### 实现清单

为类编写测试时，按以下顺序执行：

1. **映射类**：列出所有公开方法和核心内部方法。
2. **识别场景**：对每个方法枚举 Normal / Boundary / Exception 场景。
3. **命名用例**：使用 `test_<object>_<assertion>_when_<condition>` 格式。
4. **编写测试**：每个测试只聚焦一个场景。
5. **验证覆盖**：确保每个公开方法至少有一个 Normal 场景测试；有输入校验或范围约束的方法须补充 Boundary 和 Exception 场景。

### conftest.py 使用

`conftest.py` 用于定义测试 fixtures 和通用 mock 配置，pytest 会自动发现并加载。

#### 层级规则

- `test/cases/core/conftest.py`：core 模块全局配置，所有 core 子目录的测试共享。
- `test/cases/core/<sub>/conftest.py`：子模块专属配置，仅该子目录下的测试可见。

当前 `test/cases/core/conftest.py` 已包含以下通用配置：

- `mock_init_config()`：初始化配置 mock
- `mock_kia_library()`：KIA 库 mock
- `mock_security_library()`：安全校验 mock
- `sample_torch_tensor` fixture：标准 float tensor
- `mock_dataset_loader` fixture：校准数据 loader mock
- `mock_context_factory` fixture：上下文工厂 mock

新增 fixture 时，根据作用范围选择放置层级：

- 多个子模块共享 → 放在 `test/cases/core/conftest.py`
- 仅单个子模块使用 → 放在 `test/cases/core/<sub>/conftest.py`

### 常用测试模式

| 模式 | 用法 | 关键 API |
|------|------|----------|
| 异常断言 | 验证异常类型和消息 | `pytest.raises(ExceptionType, match="...")` |
| 浮点数比较 | 避免浮点精度问题 | `assert val == pytest.approx(expected)` |
| Mock 外部依赖 | 隔离外部依赖 | `unittest.mock.Mock`、`patch`、`MagicMock` |
| 抽象方法验证 | 验证抽象基类方法标记 | `getattr(cls.method, "__isabstractmethod__", False)` |
