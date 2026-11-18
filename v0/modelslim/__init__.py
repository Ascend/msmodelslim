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
工具正式改名为"msModelSlim"，对工具包的三点变化：
1. 调用接口从`import modelslim;`，更改为`import msmodelslim;`
2. 调用接口需要同时兼容之前的包引入方式`import modelslim`，暂时兼容至MindStudio 7.0.RC3的版本
3. 当对尝试使用原来的导入方式的时候，报出`DeprecationWarning`
"""

import sys  # noqa: E402
import warnings  # noqa: E402
import importlib  # noqa: E402

OLD_PACKAGE_NAME = 'modelslim'
NEW_PACKAGE_NAME = 'msmodelslim'

# 动态导入 msmodelslim
msmodelslim = importlib.import_module(NEW_PACKAGE_NAME)


def _recursive_import_and_replace(module_name, redirected_module_name):
    module = importlib.import_module(module_name)
    sys.modules[redirected_module_name] = module
    for attr in dir(module):
        if not attr.startswith('__'):
            submodule_name = f"{module_name}.{attr}"
            redirected_submodule_name = f"{redirected_module_name}.{attr}"
            try:
                submodule = importlib.import_module(submodule_name)
                sys.modules[redirected_submodule_name] = submodule
                _recursive_import_and_replace(submodule_name, redirected_submodule_name)
            except ModuleNotFoundError:
                continue


_recursive_import_and_replace(NEW_PACKAGE_NAME, OLD_PACKAGE_NAME)


def issue_deprecation_warn():
    warnings.warn(
        "Package API `{}` is deprecated and will be removed in future version. Please use `{}` instead.".format(
            OLD_PACKAGE_NAME, NEW_PACKAGE_NAME
        ),
        DeprecationWarning,
        stacklevel=3,
    )

    # 报出DeprecationWarning


issue_deprecation_warn()
