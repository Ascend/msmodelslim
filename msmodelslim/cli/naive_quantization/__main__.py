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

import os
from pathlib import Path

from msmodelslim.app.naive_quantization import NaiveQuantizationApplication
from msmodelslim.core.quant_service.proxy import QuantServiceProxy, QuantServiceProxyConfig
from msmodelslim.cli.utils import parse_device_string
from msmodelslim.infra.file_dataset_loader import FileDatasetLoader
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VLMDatasetLoader
from msmodelslim.infra.plugin_practice_dirs import discover_plugin_practice_dirs
from msmodelslim.infra.yaml_practice_manager import YamlPracticeManager
from msmodelslim.infra.debug_info_persistence import DebugInfoPersistence
from msmodelslim.infra.yaml_quant_config_exporter import YamlQuantConfigExporter
from msmodelslim.model import PluginModelFactory
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.security.path import get_valid_read_path
from msmodelslim.core.context import ContextFactory


def get_practice_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lab_practice_dir = os.path.abspath(os.path.join(cur_dir, '../../lab_practice'))
    lab_practice_dir = get_valid_read_path(lab_practice_dir, is_dir=True)
    return Path(lab_practice_dir)


def get_dataset_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lab_calib_dir = os.path.abspath(os.path.join(cur_dir, '../../lab_calib'))
    lab_calib_dir = get_valid_read_path(lab_calib_dir, is_dir=True)
    return Path(lab_calib_dir)


def main(args):
    config_dir = get_practice_dir()
    custom_practice_dir = msmodelslim_config.env_vars.custom_practice_repo
    custom_practice_path = Path(custom_practice_dir) if custom_practice_dir else None
    plugin_dirs = discover_plugin_practice_dirs()
    practice_manager = YamlPracticeManager(
        official_config_dir=config_dir,
        custom_config_dir=custom_practice_path,
        third_party_config_dirs=plugin_dirs if plugin_dirs else None,
    )
    dataset_dir = get_dataset_dir()
    dataset_loader = FileDatasetLoader(dataset_dir)
    vlm_dataset_loader = VLMDatasetLoader(dataset_dir)
    device_type, device_index = parse_device_string(args.device)
    if getattr(args, 'device_id', None):
        device_index = list(args.device_id)

    # Create context persistence if debug mode is enabled
    debug_info_persistence = None
    if args.debug:
        debug_info_persistence = DebugInfoPersistence(save_dir=args.save_path)

    quant_service = QuantServiceProxy(
        QuantServiceProxyConfig(apiversion="proxy"),
        dataset_loader,
        vlm_dataset_loader,
        context_factory=ContextFactory(enable_debug=args.debug),
        debug_info_persistence=debug_info_persistence,
    )

    # 创建YAML量化配置导出器
    quant_config_exporter = YamlQuantConfigExporter()

    app = NaiveQuantizationApplication(
        practice_manager=practice_manager,
        quant_service=quant_service,
        model_factory=PluginModelFactory(),
        quant_config_export_infra=quant_config_exporter,
    )

    app.quant(
        model_type=args.model_type,
        model_path=args.model_path,
        save_path=args.save_path,
        device_type=device_type,
        device_index=device_index,
        quant_type=args.quant_type,
        config_path=args.config_path,
        trust_remote_code=args.trust_remote_code,
        tag=getattr(args, 'tag', None),
    )
