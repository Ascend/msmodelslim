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
python3 -m coverage run --source=msmodelslim test/fuzz/pytorch_ptq_api/anti_outlier/fuzz_test.py -atheris_runs=1000
python3 -m coverage report -m --include='msmodelslim/pytorch/llm_ptq/anti_outlier/anti_outlier.py'
python3 -m coverage html -d test/fuzz/pytorch_ptq_api/anti_outlier/htmlcov -i
"""
# 导入必要的库
import sys  # noqa: E402

import atheris  # Google开源的模糊测试框架  # noqa: E402
import torch  # noqa: E402

# 导入待测试的模块
from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_outlier import AntiOutlier  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier.config import AntiOutlierConfig  # noqa: E402
from test.fuzz.common.fuzz_utils import (  # noqa: E402
    consume_bool,
    consume_str,
    consume_float,
    consume_int,
    consume_pick_list,
    DummyModel,
)


def random_flex_config(fdp):
    """生成随机的flex配置
    可能返回None、字典或错误类型(字符串)用于测试异常情况
    Args:
        fdp: FuzzedDataProvider对象
    Returns:
        Union[None, dict, str]: 随机生成的flex配置
    """
    choice = consume_int(fdp, 0, 1)
    if choice == 0:
        return None
    else:
        d = {}
        for _ in range(consume_int(fdp, 0, 2)):
            k = consume_pick_list(fdp, ['alpha', 'beta', "other"])
            v = consume_float(fdp)
            d[k] = v
        return d


def random_calib_data(fdp, input_dim):
    """生成随机校准数据
    Args:
        fdp: FuzzedDataProvider对象
        input_dim: 输入维度
    Returns:
        list: 包含随机生成的校准数据的列表
    """
    batch = consume_int(fdp, 1, 2)
    return [[torch.randn(batch, input_dim)]]


@atheris.instrument_func
def TestAntiOutlier(data):
    """模糊测试的主要函数
    测试AntiOutlier类的各种输入组合
    Args:
        data: 由模糊测试框架提供的随机数据
    """
    fdp = atheris.FuzzedDataProvider(data)
    try:
        # 生成随机配置
        flex_config = random_flex_config(fdp)
        config = AntiOutlierConfig(
            w_bit=consume_pick_list(fdp, [2, 4, 8, 16]),
            a_bit=consume_pick_list(fdp, [2, 4, 8, 16]),
            anti_method=consume_pick_list(fdp, ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'other']),
            dev_type=consume_pick_list(
                fdp,
                [
                    'cpu',
                    'npu',
                ],
            ),
            flex_config=flex_config,
        )

        # 创建测试模型和数据
        input_dim = 16
        output_dim = 16
        model = DummyModel(input_dim, output_dim)
        calib_data = random_calib_data(fdp, input_dim)
        norm_class_name = consume_str(fdp) if consume_bool(fdp) else None

        # 测试AntiOutlier类
        anti = AntiOutlier(model, calib_data=calib_data, cfg=config, norm_class_name=norm_class_name)
        if consume_bool(fdp):
            try:
                anti.process()
            except Exception:
                pass
    except Exception:
        pass


if __name__ == "__main__":
    # 设置并运行模糊测试
    atheris.Setup(sys.argv, TestAntiOutlier)
    atheris.Fuzz()
