---
title: FAQ - 运行与资源
description: 运行与资源类问题的排查与解决；不覆盖安装、硬件、精度、部署问题。
keywords: [Killed, OOM, NPU, 显存, 内存, 资源, 进程, dmesg]
---

# 运行与资源 FAQ

> 本节覆盖 msModelSlim 运行与资源相关的常见问题。新增条目按 `1.`、`2.` 顺序在文末追加，并保持“问题现象 / 问题原因 / 解决方法 / 关联文档”的四段结构（段内小节编号依次为 `x.1`~`x.4`）。

## 1. 为什么我的程序会显示“Killed”并异常退出？

### 1.1 问题现象

在使用msModelSlim工具运行推理量化时，出现类似以下报错信息：

```text
Killed
...
[Error] TBE Subprocess[task_distribute] raise error[], main process disappeared!
...
```

### 1.2 问题原因

一般是进程被其他用户kill或抢占同一个NPU资源，或NPU显存不足、系统内存不足导致。

### 1.3 解决方法

先确认进程没有被其他用户kill或抢占NPU资源。若不存在资源抢占情况，可通过以下命令查看系统日志、监控系统内存并清理内存：

```shell
# dmesg查看被内核终止的进程或显存不足终止的进程
dmesg | grep -A 3 -B 1 -i "killed process\|oom-kill"

# 监控系统内存
watch free -h

# 清理缓存和内存，部分场景可能需要sudo权限
sync && echo 3 > /proc/sys/vm/drop_caches

# 停止所有python进程，部分场景可能需要sudo权限
pkill python
```

### 1.4 关联文档

量化所需的硬件与资源要求请参见《[安装指南](../install_guide/install_guide.md)》。
