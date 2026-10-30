# Developer Testing Guide

## Introduction

This document aims to guide developers in running unit tests in the msModelSlim project, helping you understand the test environment configuration and execution process to ensure that tests can be executed smoothly.

## Environment Requirements

Complete the setup of the build and test environment as described in [msModelSlim Installation Guide — Source Code Installation](../../install_guide/install_guide.md#231-environment-preparation).

> **Note**: The build method of the environment image and the accompanying software versions are maintained by the unified MindStudio image creation guide and are not redefined in this repository.

## Executing Unit Tests

Run the following commands to execute unit tests, which test the modules related to modelslim_v1 (app, core, ir, infra, processor, and utils):

```bash
cd ~/msmodelslim
python3 build.py test
```

### Typical Test Output

The test progress and final result are displayed for each module. The output format is as follows:

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

>[!NOTE]
>
> After the tests are complete, you will see the statistics of each module in the format `X passed, Y skipped, Z warnings`.

## Writing Unit Tests

### Directory Structure Mapping

The directory structure of test cases must correspond one-to-one with the source code directory structure. The test root directory is `test/cases/`, under which subdirectories are divided by module, and each subdirectory corresponds to a module of the same name under `msmodelslim/`.

Mapping rule: `msmodelslim/<module>/<sub>/xxx.py` → `test/cases/<module>/<sub>/test_xxx.py`

Take the `core` module as an example:

```text
Source path                             Test case path
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

### Adding a Test Case

When you need to add tests for a new or existing source file, follow these steps:

1. **Locate the target directory**: Based on the source path, find the corresponding directory under `test/cases/`. If the directory does not exist, create it.

   For example, if you add a source file `msmodelslim/core/quantizer/impl/new_quantizer.py`, you should create the test file under `test/cases/core/quantizer/impl/`.

2. **Create the test file**: Name the file `test_<source module name>.py`, for example, `test_new_quantizer.py`.

3. **Write the test content**: Write the test classes and test methods according to the specifications below.

### Naming Conventions

| Item | Convention | Example |
|------|------|------|
| Test file | `test_<module_name>.py` | `test_minmax.py`, `test_context_factory.py` |
| Test class | `Test<ClassName>` | `TestW4A4Quantizer`, `TestMsMinMaxObserver` |
| Test method | `test_<object>_<assertion>_when_<condition>` | `test_quantize_raises_error_when_weight_out_of_range` |

Test method naming follows a three-part structure:

| Part | Meaning | Example |
|------|------|------|
| `object` | The method or property under test | `quantize`, `get_config`, `get_min_max` |
| `assertion` | The assertion description of the result | `returns_none`, `raises_error`, `equals_expected` |
| `condition` | The precondition that triggers the behavior | `input_is_empty`, `weight_out_of_range`, `not_updated` |

>[!NOTE]
>
> Naming is documentation. When a test fails, the method name should directly identify **which scenario** has a problem, rather than merely "some feature is broken". For example, if `test_quantize_raises_error_when_weight_out_of_range` fails → there is a boundary handling issue; if `test_quantize_works` fails → you only know that "quantization has a problem".

### Test Class and File Specifications

- Each class in the source code **must** correspond to a test class, and the test class name is prefixed with `Test` (for example, `W4A4Quantizer` → `TestW4A4Quantizer`).
- A test class must include a docstring describing the target under test.
- A single file may contain multiple test classes, arranged from top to bottom according to the classes under test.
- Each test method must include a docstring in the **scenario/expectation** format: `"""Scenario: xxx. Expectation: yyy."""`.
- Each test file must contain a standard copyright declaration header.
- Import order: standard library → third-party library → project internal modules, with one blank line between each group; project internal modules must be imported using the full package path.

### Test Scope and Scenario Design

#### Test Scope

- **Must cover**: all public methods and properties (external interfaces).
- **Recommended to cover**: core internal/private methods that contain non-trivial logic.

#### Scenario-based Test Design

**One test case = one scenario**, rather than one test case per function.

For each method, design tests from the following three scenario categories:

| Scenario Category | Description | Example |
|----------|------|------|
| **Normal** | Typical, valid input | A tensor with a valid shape |
| **Boundary** | Boundary conditions and extreme values | An empty tensor, minimum/maximum values, a single element |
| **Exception** | Invalid input that should raise an exception | None input, wrong type, out-of-range value |

#### Implementation Checklist

When writing tests for a class, proceed in the following order:

1. **Map the class**: List all public methods and core internal methods.
2. **Identify scenarios**: Enumerate Normal / Boundary / Exception scenarios for each method.
3. **Name test cases**: Use the `test_<object>_<assertion>_when_<condition>` format.
4. **Write the test**: Each test focuses on only one scenario.
5. **Verify coverage**: Ensure that every public method has at least one Normal scenario test; methods with input validation or range constraints must be supplemented with Boundary and Exception scenarios.

### Using conftest.py

`conftest.py` is used to define test fixtures and common mock configurations, and pytest automatically discovers and loads it.

#### Hierarchy Rules

- `test/cases/core/conftest.py`: global configuration for the core module, shared by tests in all core subdirectories.
- `test/cases/core/<sub>/conftest.py`: submodule-specific configuration, visible only to tests in that subdirectory.

The current `test/cases/core/conftest.py` already contains the following common configurations:

- `mock_init_config()`: initialization configuration mock
- `mock_kia_library()`: KIA library mock
- `mock_security_library()`: security validation mock
- `sample_torch_tensor` fixture: standard float tensor
- `mock_dataset_loader` fixture: calibration data loader mock
- `mock_context_factory` fixture: context factory mock

When adding a new fixture, select the placement level based on its scope:

- Shared by multiple submodules → place it in `test/cases/core/conftest.py`
- Used by only a single submodule → place it in `test/cases/core/<sub>/conftest.py`

### Common Test Patterns

| Pattern | Usage | Critical API |
|------|------|----------|
| Exception assertion | Verify the exception type and message. | `pytest.raises(ExceptionType, match="...")` |
| Floating-point comparison | Avoid floating-point precision issues. | `assert val == pytest.approx(expected)` |
| Mock external dependencies | Isolate external dependencies. | `unittest.mock.Mock`, `patch`, `MagicMock` |
| Abstract method verification | Verify the abstract base class method marker. | `getattr(cls.method, "__isabstractmethod__", False)` |
