---
title: FAQ - 安装与依赖
description: 安装与依赖类问题的原因与解决；不覆盖运行、硬件、精度、部署问题。
keywords: [安装, pip, 依赖, pydantic, accelerate, 版本冲突, huggingface_hub]
---

# 安装与依赖 FAQ

> 本节覆盖 msModelSlim 安装与依赖相关的常见问题。新增条目按 1.x 顺序追加，并保持“问题现象 / 问题原因 / 解决方法 / 关联文档”的结构。

## 1.1 为什么安装时提示pydantic版本冲突？

### 问题现象

安装或使用时出现类似以下报错信息：

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
check-wheel-contents 0.6.0 requires pydantic~=2.0, but you have pydantic 1.0 which is incompatible.
```

### 问题原因

msModelSlim依赖pydantic>=2.10.1，当环境中已安装低版本pydantic时，会与其他依赖低版本pydantic的软件包产生版本冲突。

### 解决方法

升级pydantic至2.10.1及以上版本，或卸载环境中依赖低版本pydantic的其他软件包，直至环境无版本冲突。

### 关联文档

安装的具体步骤与依赖要求请参见《[安装指南](../install_guide/install_guide.md)》。

## 1.2 为什么安装msModelSlim报错？

### 问题现象

安装时自动安装accelerate依赖库报错，出现以下两类错误信息之一：

```text
ERROR: Could not find a version that satisfies the requirement puccinialin (from versions: none)
```

或：

```text
error: subprocess-exited-with-error
```

### 问题原因

- **第一种**：部分Python 3.8环境与accelerate冲突，导致依赖解析失败。
- **第二种**：环境已升级至Python 3.9及以上后仍报错，一般是操作系统版本过低，导致安装`huggingface_hub`的子依赖失败。

### 解决方法

- 针对第一种情况：升级Python环境至Python 3.9及以上版本。
- 针对第二种情况：升级操作系统版本，或通过以下命令规避：

```bash
pip install "huggingface_hub==0.20.3"
pip install accelerate
```

> 注意：`huggingface_hub==0.20.3`非`accelerate`官方推荐版本，可能引发其他兼容性问题，该方案仅供参考，msModelSlim对由此带来的问题不承担相应责任。

### 关联文档

安装的具体步骤与依赖要求请参见《[安装指南](../install_guide/install_guide.md)》。
