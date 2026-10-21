---
title: FAQ - Installation and Dependencies
description: Causes and solutions for installation and dependency problems; does not cover runtime, hardware, accuracy, or deployment problems.
keywords: [installation, pip, dependencies, pydantic, accelerate, version conflict, huggingface_hub]
---

# Installation and Dependencies FAQ

> This section covers common problems related to msModelSlim installation and dependencies. To add a new entry, append it at the end of the document in the order `1.`, `2.`, and keep the four-section structure "Symptom / Cause / Solution / Related Documents" (the subsection numbers within a section run from `x.1` to `x.4`).

## 1. Why Does Installation Report a pydantic Version Conflict?

### 1.1 Symptom

An error message like the following appears during installation or use:

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
check-wheel-contents 0.6.0 requires pydantic~=2.0, but you have pydantic 1.0 which is incompatible.
```

### 1.2 Cause

msModelSlim depends on `pydantic>=2.10.1`. If an older version of pydantic is already installed in the environment, the msModelSlim requirement may conflict with other software packages that require an older version of pydantic.

### 1.3 Solution

Upgrade pydantic to 2.10.1 or later, or uninstall other software packages in the environment that depend on older versions of pydantic, until all version conflicts are resolved.

### 1.4 Related Documents

For the specific installation steps and dependency requirements, see the [msModelSlim Installation Guide](../install_guide/install_guide.md).

## 2. Why Does msModelSlim Installation Fail?

### 2.1 Symptom

An error occurs when the accelerate dependency library is installed automatically, and one of the following two types of error messages appears:

```text
ERROR: Could not find a version that satisfies the requirement puccinialin (from versions: none)
```

Or:

```text
error: subprocess-exited-with-error
```

### 2.2 Cause

- **First type**: some Python 3.8 environments are incompatible with accelerate, causing dependency resolution to fail.
- **Second type**: if the error persists after the environment is upgraded to Python 3.9 or later, the OS version is generally too old, causing the installation of a sub-dependency of `huggingface_hub` to fail.

### 2.3 Solution

For the preceding problems, try the following solutions:

- Upgrade the Python environment to Python 3.9 or later.
- Upgrade the OS version.
- Use the following commands as a workaround:

```bash
pip install "huggingface_hub==0.20.3"
pip install accelerate
```

> Note: `huggingface_hub==0.20.3` is not the version officially recommended by `accelerate`, and may cause other compatibility issues. This workaround is for reference only, and msModelSlim assumes no responsibility for any problems it may cause.

### 2.4 Related Documents

For the specific installation steps and dependency requirements, see the [msModelSlim Installation Guide](../install_guide/install_guide.md).
