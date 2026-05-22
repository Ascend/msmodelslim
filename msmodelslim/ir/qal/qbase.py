#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any
from collections import namedtuple
import torch
from msmodelslim.utils.exception import SchemaValidateError


def _get_format_params(elem_format):
    # 定义格式参数表
    format_info = {
        "mxfp4": {"ebits": 2, "mbits": 3, "emax_offset": 0},    # emax = 2^(ebits - 1)
        "mxfp8_e4m3": {"ebits": 4, "mbits": 5, "emax_offset": 0},    # emax = 2^(ebits - 1)
        "mxfp8_e5m2": {"ebits": 5, "mbits": 4, "emax_offset": -1},   # emax = 2^(ebits - 1) - 1
        "mxint8": {"ebits": 0, "mbits": 8, "emax_offset": 0},    # emax = 0
        "mxint4": {"ebits": 0, "mbits": 4, "emax_offset": 0},    # emax = 0
    }

    if elem_format not in format_info:
        raise Exception("Unknown element format %s" % elem_format)

    info = format_info[elem_format]
    ebits = info["ebits"]
    mbits = info["mbits"]
    emax_offset = info["emax_offset"]

    # 计算 emax
    if ebits == 0:
        emax = 0
    else:
        emax = 2 ** (ebits - 1) + emax_offset

    # 计算 max_norm
    if elem_format == "mxfp8_e4m3":
        max_norm = 2 ** emax * 1.75 # (1 + fraction) * 2 ^ (e-bias) fraction = 2^(-1) + 2^(-2)
    else:
        # 注意：当 mbits == 0 时此式可能无效，但当前所有格式 mbits >= 3
        max_norm = 2 ** emax * float(2 ** (mbits - 1) - 1) / (2 ** (mbits - 2))

    return ebits, mbits, emax, max_norm


class QDType(str, Enum):
    FLOAT = "float"

    INT8 = "int8"
    INT4 = "int4"
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    FP8_E4M3 = "fp8_e4m3"

    PLACEHOLDER = "placeholder"

    @property
    def mx_finfo(self):
        if self not in [QDType.MXFP8, QDType.MXFP4]:
            raise SchemaValidateError(f"mx finfo not defined for {self}")
        Finfo = namedtuple(
            'finfo',
            ['block_size', 'scale_bits', 'flush_fp32_subnorms', 'ebits', 'mbits', 'emax', 'max_norm']
        )
        if self == QDType.MXFP8:
            elem_format = "mxfp8_e4m3"
        elif self == QDType.MXFP4:
            elem_format = "mxfp4"
        else:
            raise SchemaValidateError(f"Unknown QDType for finfo: {self}")
        return Finfo(32, 8, False, *_get_format_params(elem_format))


class QScope(str, Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"
    PER_BLOCK = "per_block"
    PER_TOKEN = "per_token"
    PD_MIX = "pd_mix"
    PER_HEAD = "per_head"
    DUAL_SCALE = "dual_scale"

    PLACEHOLDER = "placeholder"


@dataclass(frozen=True)
class QScheme:
    scope: QScope = QScope.PLACEHOLDER
    dtype: QDType = QDType.PLACEHOLDER
    symmetric: bool = True

    def __repr__(self) -> str:
        return f"QScheme(scope={self.scope.value}, dtype={self.dtype.value}, symmetric={self.symmetric})"


@dataclass
class QParam:
    """

    QParam用于描述量化参数，包括量化范围、量化类型、量化对称性等。

    """

    scheme: QScheme
    ext: Any = None

    def __repr__(self) -> str:
        repr_str = f"QParam(scheme={self.scheme})"
        if self.scheme.scope == QScope.PER_GROUP:
            repr_str += f", group_size={self.ext['group_size']}"
        return repr_str


_TORCH_FLOAT_TYPE: torch.dtype = torch.float32


@dataclass
class QStorage:
    """

    QStorage用于描述某一种自定义数据类型的Tensor。

    """

    dtype: QDType
    value: torch.Tensor
    ext: Any = None

    @staticmethod
    @contextmanager
    def set_value_float_type(dtype: torch.dtype):
        global _TORCH_FLOAT_TYPE
        old_dtype = _TORCH_FLOAT_TYPE
        _TORCH_FLOAT_TYPE = dtype
        yield
        _TORCH_FLOAT_TYPE = old_dtype

    @property
    def T(self):
        return self.same_like(self.value.T)

    def same_like(self, new_val: torch.Tensor):
        return QStorage(dtype=self.dtype, value=new_val, ext=self.ext)

    def to(self, dtype: QDType):
        self.dtype = dtype

        if self.dtype == QDType.FLOAT:
            self.value = self.value.to(_TORCH_FLOAT_TYPE)

        if self.dtype in [QDType.INT8, QDType.INT4]:
            self.value = self.value.to(torch.int8)
        
        if self.dtype == QDType.FP8_E4M3:
            self.value = self.value.to(torch.float32)

        return self

    def reshape(self, shape: torch.Size):
        self.value.reshape(shape)
        return self
