# FAQ

## 1. 为什么我的程序会显示“Killed”并异常退出？

在使用msModelSlim工具运行推理量化时，出现类似以下报错信息：

```text
Killed
...
[Error] TBE Subprocess[task_distribute] raise error[], main process disappeared!
...
```

### 解决方法

请先确认你的进程没有被其他用户kill或抢占同一个NPU资源。一般而言，如果不存在其他用户抢占系统资源的情况，那么可能就是NPU显存不足或系统内存不足导致。可通过以下命令查看系统日志、看管系统内存情况、清理系统内存。

```shell
# dmesg查看被内核终止的进程或显存不足终止的进程
dmesg | grep -A 3 -B 1 -i "killed process\|oom-kill"

# 看管系统内存
watch free -h

# 清理缓存和内存，部分场景可能需要sudo权限
sync && echo 3 > /proc/sys/vm/drop_caches

# 停止所有python进程，部分场景可能需要sudo权限
pkill python
```

## 2. 为什么安装时提示pydantic版本冲突？

msModelSlim依赖pydantic>=2.10.1，请确保环境中的pydantic版本满足此要求。

### pip安装时的错误信息

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
check-wheel-contents 0.6.0 requires pydantic~=2.0, but you have pydantic 1.0 which is incompatible.
```

### 解决方法

请尝试升级pydantic或卸载环境中依赖低版本pydantic的其他软件包，直至环境无版本冲突。

## 3. 为什么安装msModelSlim报错？

### 3.1 自动安装accelerate依赖库时报错

msModelSlim依赖accelerate库来支持多卡运行，因此将其写入requirements.txt，在安装时通过pip自动下载accelerate。

第一种已知原因是部分 Python 3.8 环境与accelerate冲突，报错信息如下：

```text
ERROR: Could not find a version that satisfies the requirement puccinialin (from versions: none)
```

此时，可尝试升级Python环境至 Python 3.9 及以上版本

如果您的环境已升级至 Python 3.9 或以上版本，但仍然出现报错，可能是由于os版本过低，导致安装 `huggingface_hub` 的子依赖失败，报错信息如下：

```text
error: subprocess-exited-with-error
```

此时可尝试升级os版本，或通过以下命令进行规避：

```bash
pip install "huggingface_hub==0.20.3"
pip install accelerate
```

请注意，`huggingface_hub==0.20.3`非`accelerate`官方推荐版本，可能会引发其他兼容性问题。因此，该方案仅供参考，`msModelSlim`对由此带来的问题不承担相应责任。

## 4. 为什么量化权重时出现报错**PTA call acl api failed. *** The param dtype not implemented for DT_BFLOAT16, should be in dtype support list [\*\*\*]**

部分Ascend硬件（例如Atlas 300I/300T系列）只支持float16精度推理，如果模型权重采用`bfloat16`精度量化，可能会导致量化失败。

### 解决方法

修改模型权重路径下`config.json`中的`torch_dtype`为`float16`进行量化。

## 5. 为什么在300I/300T系列硬件上量化权重时会报错**RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is InplaceIndexAdd.**

### 问题原因

在300I/300T系列硬件上进行传统量化（V0）时，由于JIT编译模式与该系列硬件存在兼容性问题，导致`InplaceIndexAdd`算子编译失败，从而引发运行时错误。

### 解决方法

在传统量化（V0）模型量化脚本（`msmodelslim/example`路径下）中，添加`torch_npu.npu.set_compile_mode(jit_compile=False)`来禁用JIT编译模式。

**示例代码：**

```python
import torch_npu

# 在量化脚本开头添加以下代码
torch_npu.npu.set_compile_mode(jit_compile=False)

# 然后执行量化操作
# ... 后续的量化代码
```

## 6. 量化结束告警：**sys:1: DeprecationWarning: builtin type swigvarlink has no module attribute**

### 问题原因

旧版 SWIG 生成的第三方库与 Python 3.10+ 不兼容。

### 解决方法

此告警不影响量化结果，升级触发该警告的第三方库即可消除（如 sentencepiece）。
