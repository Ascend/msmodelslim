#!/usr/bin/env python
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
# pylint: disable=pointless-string-statement

"""
使用方法：
在 msit/msmodelslim 目录下执行
python3 -m coverage run --source=msmodelslim test/fuzz/pytorch_ptq_api/calibrator/fuzz_test.py -atheris_runs=1000
python3 -m coverage report -m --include='msmodelslim/pytorch/llm_ptq/llm_ptq_tools/quant_tools.py'
python3 -m coverage html -d test/fuzz/pytorch_ptq_api/calibrator/htmlcov -i
"""

# 导入必要的库
import sys  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

import atheris  # Google开源的模糊测试框架  # noqa: E402
import torch  # noqa: E402

# 导入需要测试的模块
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config import QuantConfig  # noqa: E402
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_tools import Calibrator  # noqa: E402
from test.fuzz.common.fuzz_utils import (  # noqa: E402
    consume_bool,
    consume_str,
    consume_int,
    consume_pick_list,
    random_all_tensors,
    DummyModel,
)


def random_mix_cfg(fdp):
    """生成随机混合配置
    可能返回：None、量化配置字典或错误类型数据

    Args:
        fdp: FuzzedDataProvider对象
    Returns:
        Union[None, dict, int]: 随机生成的混合配置
    """
    choice = consume_int(fdp, 0, 2)
    if choice == 0:
        return None
    elif choice == 1:
        allowed_types = ['w8a8', 'w8a16', 'w8a8_dynamic', 'float', 'w4a8_dynamic']
        d = {}
        for _ in range(consume_int(fdp, 0, 3)):
            k = consume_str(fdp)
            v = consume_pick_list(fdp, allowed_types + [consume_str(fdp)])
            d[k] = v
        return d
    else:
        return consume_int(fdp, 0, 100)


def random_save_args(fdp):
    """生成模型保存相关的随机参数
    Args:
        fdp: FuzzedDataProvider对象
    Returns:
        tuple: (output_path, safetensors_name, json_name, save_type, part_file_size)
            - output_path: 输出路径
            - safetensors_name: safetensors文件名或None
            - json_name: json文件名或None
            - save_type: 保存类型列表或None
            - part_file_size: 分片大小或None
    """
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, consume_str(fdp))
    safetensors_name = consume_str(fdp) if consume_bool(fdp) else None
    json_name = consume_str(fdp) if consume_bool(fdp) else None
    save_type_choices = [None, ['numpy'], ['safetensor'], ['ascendv1'], ['numpy', 'safetensor']]
    save_type = consume_pick_list(fdp, save_type_choices)
    part_file_size = consume_int(fdp, 1, 4) if consume_bool(fdp) else None
    return output_path, safetensors_name, json_name, save_type, part_file_size


# 主要的模糊测试函数
@atheris.instrument_func
def TestCalibrator(data):
    """
    主要的模糊测试函数，测试Calibrator类的功能
    包括初始化、运行和保存等操作
    """
    fdp = atheris.FuzzedDataProvider(data)
    try:
        # 构造量化配置
        quant_config = QuantConfig(
            w_bit=consume_pick_list(fdp, [2, 4, 8, 16]),
            a_bit=consume_pick_list(fdp, [2, 4, 8, 16]),
            mm_tensor=consume_bool(fdp),
            dev_type=consume_pick_list(fdp, ['cpu', 'npu']),
            w_sym=consume_bool(fdp),
            pr=1.0,
            act_method=consume_pick_list(fdp, [1, 2, 3, 4]),
        )

        # 构造测试模型和校准数据
        input_dim = 1024
        output_dim = 1024
        model = DummyModel(input_dim, output_dim)
        batch = 1
        calib_data = [torch.randn(batch, input_dim)]

        # 生成随机测试参数
        disable_level_choices = ['L0', 'L1', 'L2', consume_str(fdp)]
        disable_level = consume_pick_list(fdp, disable_level_choices)
        all_tensors = random_all_tensors(fdp)
        mix_cfg = random_mix_cfg(fdp)

        # 初始化Calibrator并测试其功能
        calibrator = Calibrator(
            model,
            quant_config,
            calib_data=calib_data,
            disable_level=disable_level,
            all_tensors=all_tensors,
            mix_cfg=mix_cfg,
        )

        # 随机测试run方法
        if consume_bool(fdp):
            calibrator.run()

        # 随机测试save方法
        if consume_bool(fdp):
            output_path, safetensors_name, json_name, save_type, part_file_size = random_save_args(fdp)
            try:
                calibrator.save(output_path, safetensors_name, json_name, save_type, part_file_size)
            except Exception:
                pass  # 忽略保存时的异常

    except Exception:
        pass  # 忽略所有异常，专注于发现崩溃问题


# 主程序入口
if __name__ == "__main__":
    atheris.Setup(sys.argv, TestCalibrator)
    atheris.Fuzz()
