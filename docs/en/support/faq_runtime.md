---
title: FAQ - Runtime and Resources
description: Troubleshooting and solutions for runtime and resource problems; does not cover installation, hardware, accuracy, or deployment problems.
keywords: [Killed, OOM, NPU, NPU memory, system memory, resources, process, dmesg]
---

# Runtime and Resources FAQ

> This section covers common problems related to msModelSlim runtime and resources. To add a new entry, append it at the end of the document in the order `1.`, `2.`, and keep the four-section structure "Symptom / Cause / Solution / Related Documents" (the subsection numbers within a section run from `x.1` to `x.4`).

## 1. Why Does the Program Display "Killed" and Exit Abnormally?

### 1.1 Symptom

When you use msModelSlim to perform quantization for inference, you may see an error message like the following:

```text
Killed
...
[Error] TBE Subprocess[task_distribute] raise error[], main process disappeared!
...
```

### 1.2 Cause

In general, this is caused by the process being killed by another user or by another user preempting the same NPU resource, or by insufficient NPU memory or system memory.

### 1.3 Solution

First, confirm that the process has not been killed by another user and that the NPU resource has not been preempted. If there is no resource preemption, you can use the following commands to view system logs, monitor system memory, and free system memory:

```shell
# Use dmesg to view processes terminated by the kernel or by insufficient memory.
dmesg | grep -A 3 -B 1 -i "killed process\|oom-kill"

# Monitor system memory.
watch free -h

# Clear the cache and memory. Some scenarios may require sudo privileges.
sync && echo 3 > /proc/sys/vm/drop_caches

# Stop all Python processes. Some scenarios may require sudo privileges.
pkill python
```

### 1.4 Related Documents

For the hardware and resource requirements for quantization, see the [msModelSlim Installation Guide](../install_guide/install_guide.md).
