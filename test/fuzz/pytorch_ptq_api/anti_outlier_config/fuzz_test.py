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
python3 -m coverage run --source=msmodelslim test/fuzz/pytorch_ptq_api/anti_outlier_config/fuzz_test.py -atheris_runs=1000
python3 -m coverage report -m --include='msmodelslim/pytorch/llm_ptq/anti_outlier/config.py'
python3 -m coverage html -d test/fuzz/pytorch_ptq_api/anti_outlier_config/htmlcov -i
"""
import sys  # noqa: E402

import atheris  # noqa: E402

from msmodelslim.pytorch.llm_ptq.anti_outlier.config import AntiOutlierConfig  # noqa: E402
from test.fuzz.common.fuzz_utils import consume_bool, consume_str, consume_int, consume_pick_list, consume_float  # noqa: E402


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


@atheris.instrument_func
def TestAntiOutlierConfig(data):
    """
    模糊测试的主要入口函数
    测试AntiOutlierConfig类的各种输入参数组合
    Args:
        data: 由模糊测试引擎提供的随机数据
    """
    fdp = atheris.FuzzedDataProvider(data)
    try:
        # 生成随机的配置参数
        w_bit = (consume_pick_list(fdp, [2, 4, 8, 16]),)  # 权重比特数
        a_bit = (consume_pick_list(fdp, [2, 4, 8, 16]),)  # 激活比特数
        anti_method = consume_pick_list(fdp, ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'other'])  # 异常值处理方法
        dev_type = consume_pick_list(fdp, ['cpu', 'npu'])  # 设备类型
        dev_id = None  # 设备ID
        w_sym = consume_bool(fdp)  # 是否对称量化
        # 随机生成需要禁用的异常值处理名称列表
        disable_anti_names = [consume_str(fdp) for _ in range(consume_int(fdp, 0, 3))] if consume_bool(fdp) else []
        flex_config = random_flex_config(fdp)  # 随机生成flex配置

        # 使用随机生成的参数创建AntiOutlierConfig对象
        AntiOutlierConfig(
            w_bit=w_bit,
            a_bit=a_bit,
            anti_method=anti_method,
            dev_type=dev_type,
            dev_id=dev_id,
            w_sym=w_sym,
            disable_anti_names=disable_anti_names,
            flex_config=flex_config,
        )
    except Exception:
        # 忽略所有异常，继续测试
        pass


if __name__ == "__main__":
    # 设置模糊测试引擎
    atheris.Setup(sys.argv, TestAntiOutlierConfig)
    # 开始模糊测试
    atheris.Fuzz()
