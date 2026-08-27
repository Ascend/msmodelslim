# FAQ

## 1. Why Does the Program Display "Killed" and Exit Abnormally?

When you use the msModelSlim tool to run inference quantization, you may see an error message like the following:

```text
Killed
...
[Error] TBE Subprocess[task_distribute] raise error[], main process disappeared!
...
```

### Solution

First, ensure that no other user has killed your process or taken the same NPU resource. In general, if no other user is competing for system resources, the cause is likely insufficient NPU memory or system memory. You can use the following commands to check system logs, monitor system memory, and free system memory.

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

## 2. Why Does Installation Report a pydantic Version Conflict?

msModelSlim depends on `pydantic>=2.10.1`. Make sure the version of pydantic in your environment meets this requirement.

### Error Message During `pip` Installation

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
check-wheel-contents 0.6.0 requires pydantic~=2.0, but you have pydantic 1.0 which is incompatible.
```

### Solution

Try upgrading pydantic or uninstalling other packages in the environment that depend on older versions of pydantic until there are no version conflicts.

## 3. Why Does msModelSlim Installation Fail?

### 3.1 Error During Automatic Installation of the `accelerate` Dependency

msModelSlim depends on the `accelerate` library for multi-card execution, so it is listed in `requirements.txt` and pip downloads it automatically during installation.

One known cause is that some Python 3.8 environments conflict with `accelerate`. The error message looks like this:

```text
ERROR: Could not find a version that satisfies the requirement puccinialin (from versions: none)
```

In this case, you can try upgrading the Python environment to Python 3.9 or later.

If you have already upgraded to Python 3.9 or later and the error still appears, the OS version may be too old, which causes installation of a sub-dependency of `huggingface_hub` to fail. The error message looks like this:

```text
error: subprocess-exited-with-error
```

In this case, you can try upgrading the OS version, or use the following commands as a workaround:

```bash
pip install "huggingface_hub==0.20.3"
pip install accelerate
```

Note that `huggingface_hub==0.20.3` is not the version officially recommended by `accelerate`, and it may cause other compatibility issues. Therefore, this workaround is for reference only, and msModelSlim is not responsible for any issues it may cause.

## 4. Why Do I Get an Error When Quantizing Weights: `PTA call acl api failed. *** The param dtype not implemented for DT_BFLOAT16, should be in dtype support list [***]`

Some Ascend hardware, such as the Atlas 300I/300T series, supports only float16 inference. If the model weights use bfloat16 precision for quantization, the quantization process may fail.

### Solution

Set `torch_dtype` to `float16` in `config.json` in the model weights path, then run quantization.

## 5. Why Does Quantizing Weights on 300I/300T Series Hardware Report `RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is InplaceIndexAdd.`

### Cause

When you run traditional quantization (V0) on 300I/300T series hardware, JIT compilation mode is incompatible with this hardware series. As a result, compilation of the `InplaceIndexAdd` operator fails and triggers a runtime error.

### Solution

In the traditional quantization (V0) model quantization scripts in the `msmodelslim/example` directory, add `torch_npu.npu.set_compile_mode(jit_compile=False)` to disable JIT compilation mode.

**Example code:**

```python
import torch_npu

# Add the following code at the beginning of the quantization script:
torch_npu.npu.set_compile_mode(jit_compile=False)

# Then run the quantization operation.
# ... Remaining quantization code
```
