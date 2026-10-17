# -*- coding: UTF-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import os
from enum import IntEnum
from functools import lru_cache

import torch

from msmodelslim.utils.logging import get_logger

#: Functional form: the ``310P`` member name is not a valid Python identifier,
#: so it cannot be declared as a class attribute; ``AscendDeviceType["310P"]``
#: is how the 310P member is reached.
AscendDeviceType = IntEnum(
    "AscendDeviceType",
    {"A2": 0, "A3": 1, "310P": 2, "A5": 3, "Unknown": -1},
)

#: CPU binding modes, mirroring vllm-ascend's binding strategies.
GLOBAL_SLICE = "global_slice"
TOPOLOGY_AFFINITY = "topology_affinity"

#: Device class -> CPU binding mode (vllm-ascend's ``DEVICE_TYPE_TO_BINDING_MODE``).
#: Topology affinity needs ``npu-smi info -t topo``, which this code deliberately
#: does not call, so the A2/310P/A5 branches fall back to the global-slice mode.
DEVICE_TYPE_TO_BINDING_MODE = {
    AscendDeviceType.A2: TOPOLOGY_AFFINITY,
    AscendDeviceType.A3: GLOBAL_SLICE,
    AscendDeviceType["310P"]: GLOBAL_SLICE,
    AscendDeviceType.A5: GLOBAL_SLICE,
    AscendDeviceType.Unknown: GLOBAL_SLICE,
}


def _soc_device_type() -> AscendDeviceType:
    """Map torch_npu's SoC version to an ``AscendDeviceType``.

    ``ascend910b*`` -> A2, ``ascend910_93*`` -> A3, ``310p`` -> 310P; anything
    else (or no torch_npu) -> ``AscendDeviceType.Unknown``.
    """
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "get_soc_version"):
        return AscendDeviceType.Unknown
    try:
        soc = str(npu.get_soc_version()).lower()
    except Exception as exc:  # pylint: disable=broad-except
        get_logger().warning("get_soc_version failed, CPU binding skipped: %s", exc)
        return AscendDeviceType.Unknown
    if "310p" in soc:
        return AscendDeviceType["310P"]
    if soc.startswith("ascend910b"):
        return AscendDeviceType.A2
    if soc.startswith("ascend910_93"):
        return AscendDeviceType.A3
    return AscendDeviceType.Unknown


@lru_cache(maxsize=1)
def get_current_device_type() -> AscendDeviceType:
    """Return the ``AscendDeviceType`` of the current NPU, cached.

    ``torch_npu.get_soc_version()`` is a runtime round-trip and every NPU on a
    node is the same SoC, so the mapped result is memoized on first call and
    reused for the rest of the process.
    """
    return _soc_device_type()


def _allowed_cpus() -> list:
    """This process's allowed CPU ids.

    ``os.sched_getaffinity`` already reflects the cpuset/cgroup limit, so there
    is no need to shell out to parse /proc or cgroup files.
    """
    try:
        return sorted(os.sched_getaffinity(0))
    except OSError:
        return list(range(os.cpu_count() or 1))


def _global_slice_cpus(allowed: list, device_count: int, device_index: int) -> list:
    """Split ``allowed`` into ``device_count`` contiguous segments; take one.

    Mirrors vllm-ascend's GLOBAL_SLICE mode: every NPU gets a non-overlapping,
    contiguous pool regardless of NPU<->NUMA topology (A3's HCCS mesh makes the
    affinity signal weak; the topology-affinity path that other devices would
    use needs npu-smi topo, which we deliberately do not call).
    """
    n = len(allowed)
    if n <= 0 or device_count <= 0:
        return []
    seg = max(1, n // device_count)
    start = device_index * seg
    end = n if device_index == device_count - 1 else min(start + seg, n)
    return allowed[start:end]


def _effective_binding_mode() -> str:
    """Binding mode for the current device, with the topo fallback applied.

    Only the global-slice division is implemented: the topology-affinity mode
    needs ``npu-smi info -t topo`` (deliberately not called), so A2/310P/A5
    fall back to the same contiguous-slice division as A3.
    """
    mode = DEVICE_TYPE_TO_BINDING_MODE[get_current_device_type()]
    return GLOBAL_SLICE if mode == TOPOLOGY_AFFINITY else mode


def bind_npu_cpus():
    """Pin this process to the CPU slice for its current NPU device.

    Called directly by each multi-process worker; it keys on the process's
    current device (``torch.npu.current_device``), not on any distributed
    state -- every rank gets the contiguous, non-overlapping slice that
    matches its own device id, sized by the node's NPU count.

    ``OMP_PROC_BIND`` is forced off.  Left at ``close`` together with
    ``OMP_PLACES=cores`` -- what ``tools/quant-env.sh`` used to export -- libgomp
    collapses the whole process onto a SINGLE core when the first parallel
    region starts, so N ranks all land on the same core.  ``OMP_NUM_THREADS``
    defaults to 32 and overrides whatever the caller's environment set.
    """
    npu = getattr(torch, "npu", None)
    if npu is None:
        return
    try:
        device_index = npu.current_device()
        device_count = npu.device_count()
    except Exception as exc:  # pylint: disable=broad-except
        get_logger().warning("Could not read the current NPU device, CPU binding skipped: %s", exc)
        return

    os.environ['OMP_PROC_BIND'] = 'false'
    os.environ.pop('OMP_PLACES', None)

    allowed = _allowed_cpus()
    if not allowed:
        return
    cpus = _global_slice_cpus(allowed, device_count, device_index)
    if not cpus:
        return
    try:
        os.sched_setaffinity(0, cpus)
    except OSError as exc:
        get_logger().warning("Could not bind to CPUs %s: %s", cpus, exc)
        return
    os.environ['OMP_NUM_THREADS'] = '32'
    os.environ['MKL_NUM_THREADS'] = '32'
    get_logger().info(
        "Bound to %s (%s) slice %s-%s (%d/%d CPUs) for npu:%d, OMP=%s",
        get_current_device_type().name,
        _effective_binding_mode(),
        cpus[0],
        cpus[-1],
        len(cpus),
        len(allowed),
        device_index,
        os.environ['OMP_NUM_THREADS'],
    )


__all__ = [
    "AscendDeviceType",
    "get_current_device_type",
    "bind_npu_cpus",
]
