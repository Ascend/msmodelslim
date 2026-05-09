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

import logging
import os
from configparser import ConfigParser

from setuptools import setup, find_packages

config = ConfigParser()
config.read('./config/config.ini')

abs_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(abs_path, "requirements.txt")) as f:
    required = f.read().splitlines()

model_adapter_plugins = []
entry_section = config["ModelAdapterEntryPoints"]

for group, models in config.items("ModelAdapter"):
    model_list = [m.strip() for m in models.split(",")]
    if group in entry_section:
        entry_point = entry_section[group]
    else:
        logging.warning(f"ModelAdapter group '{group}' has no entry point defined in ModelAdapterEntryPoints")
        continue

    for model in model_list:
        model_adapter_plugins.append(f"{model}={entry_point}")

# 从 config.ini 读取插件配置
plugin_entry_points = {}
for section_name in config.sections():
    if section_name.startswith("Plugin:"):
        # 提取插件名称（去掉 "Plugin:" 前缀）
        plugin_name = section_name[7:]  # len("Plugin:") = 7
        # 自动补全为 msmodelslim.xxx.plugins 格式
        plugin_path = f"msmodelslim.{plugin_name}.plugins"
        plugin_list = []
        for plugin_type, plugin_func_path in config.items(section_name):
            plugin_list.append(f"{plugin_type}={plugin_func_path}")
        if plugin_list:
            plugin_entry_points[plugin_path] = plugin_list

__version__ = '26.0.0.alpha01'
whl_version = os.getenv('WHL_VERSION')
if whl_version is not None:
    __version__ = whl_version

setup(
    name='msmodelslim',
    version=__version__,
    description='msModelSlim, MindStudio ModelSlim Tools',
    long_description_content_type='text/markdown',
    url=config.get('URL', 'repository_url'),
    packages=find_packages(exclude=['precision_tool', 'security', ]) + ['msmodelslim.config', 'msmodelslim.lab_calib',
                                                                        'msmodelslim.lab_practice'],
    package_dir={
        'msmodelslim': 'msmodelslim',
        'msmodelslim.config': 'config',
        'msmodelslim.lab_calib': 'lab_calib',
        'msmodelslim.lab_practice': 'lab_practice',
    },
    package_data={
        '': [
            'LICENSE',
            'data.json',
            'README.md',
            '*.txt',
            '*.bat',
            '*.sh',
            '*.cpp',
            '*.h',
            '*.py',
            '*.so',
        ],
        'msmodelslim.config': ['*'],
        'msmodelslim.lab_calib': ['**'],
        'msmodelslim.lab_practice': ['**'],
        'msmodelslim.core.tune_strategy.common.config_builder.expert_experience': ['*.yaml', '*.yml'],
        'msmodelslim.core.analysis_service': ['pipeline_analysis/pipeline_template/*.yaml'],
    },
    data_files=[('', ['requirements.txt'])],
    license='Mulan PSL v2',
    keywords='msmodelslim',
    python_requires='>=3.7',
    install_requires=required,
    entry_points={
        'console_scripts': [
            'msmodelslim=msmodelslim.cli.__main__:main'
        ],
        "msmodelslim.model_adapter.plugins": model_adapter_plugins,
        **plugin_entry_points,  # 从 config.ini 读取的插件配置（含 quant_service、tuning_strategy、evaluation 等）
    },
)
