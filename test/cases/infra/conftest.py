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

Pytest config for infra tests.

复用全局 mock 工具，避免在导入 msmodelslim 期间真正初始化配置文件和安全路径检查。
"""

from testing_utils.mock import mock_kia_library, mock_security_library, mock_init_config

# 在模块导入阶段立即生效，防止 init_config / 安全检查触发真实文件访问
mock_init_config()
mock_kia_library()
mock_security_library()


def pytest_configure(config):
    """注册量化任务插件，使 PracticeConfig 全量校验（dispatch）可用。"""
    from msmodelslim.utils.plugin.plugin_utils import register_plugin
    from msmodelslim.core.quant_service.modelslim_v0.quant_config import get_plugin as v0_gp
    from msmodelslim.core.quant_service.modelslim_v1.quant_config import get_plugin as v1_gp
    from msmodelslim.core.quant_service.modelslim_convert.quant_config import get_plugin as cvt_gp
    from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import get_plugin as sd_gp
    from msmodelslim.core.quant_service.multimodal_vlm_v1.quant_config import get_plugin as vlm_gp

    for getter in (v0_gp, v1_gp, cvt_gp, sd_gp, vlm_gp):
        register_plugin(getter)
