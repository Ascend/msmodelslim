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
from datetime import timedelta
from pathlib import Path

from msmodelslim.app.auto_tuning import AutoTuningApplication
from msmodelslim.core.quant_service.proxy import QuantServiceProxy, QuantServiceProxyConfig
from msmodelslim.core.tune_strategy.plugin_factory import PluginTuningStrategyFactory
from msmodelslim.cli.utils import parse_device_string
from msmodelslim.infra.file_dataset_loader import FileDatasetLoader
from msmodelslim.infra.service_oriented_evaluate_service import ServiceOrientedEvaluateService
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VLMDatasetLoader
from msmodelslim.infra.yaml_plan_manager import YamlTuningPlanManager
from msmodelslim.infra.yaml_practice_history_manager import YamlTuningHistoryManager
from msmodelslim.infra.yaml_practice_accuracy_manager import YamlTuningAccuracyManager
from msmodelslim.infra.yaml_practice_manager import YamlPracticeManager
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
    plan_manager = YamlTuningPlanManager()
    practice_dir = get_practice_dir()
    custom_practice_dir = msmodelslim_config.env_vars.custom_practice_repo
    custom_practice_path = Path(custom_practice_dir) if custom_practice_dir else None
    practice_manager = YamlPracticeManager(official_config_dir=practice_dir, custom_config_dir=custom_practice_path)
    dataset_dir = get_dataset_dir()
    dataset_loader = FileDatasetLoader(dataset_dir)
    vlm_dataset_loader = VLMDatasetLoader(dataset_dir)
    evaluation_service = ServiceOrientedEvaluateService()
    quant_service = QuantServiceProxy(
        QuantServiceProxyConfig(apiversion="proxy"),
        dataset_loader,
        vlm_dataset_loader,
        context_factory=ContextFactory(),
    )
    model_factory = PluginModelFactory()
    tuning_history_manager = YamlTuningHistoryManager()
    tuning_accuracy_manager = YamlTuningAccuracyManager()

    strategy_factory = PluginTuningStrategyFactory(dataset_loader)

    app = AutoTuningApplication(
        plan_manager=plan_manager,
        practice_manager=practice_manager,
        evaluation_service=evaluation_service,
        tuning_history_manager=tuning_history_manager,
        tuning_accuracy_manager=tuning_accuracy_manager,
        quantization_service=quant_service,
        model_factory=model_factory,
        strategy_factory=strategy_factory,
    )

    device_type, device_indices = parse_device_string(args.device)
    if getattr(args, 'device_id', None):
        device_indices = list(args.device_id)

    # --timeout is expressed in seconds; convert to timedelta when given as an int.
    timeout = args.timeout
    if isinstance(timeout, int):
        timeout = timedelta(seconds=timeout)

    app.tune(
        model_type=args.model_type,
        model_path=args.model_path,
        save_path=args.save_path,
        plan_id=args.config,
        device=device_type,
        device_indices=device_indices,
        timeout=timeout,
        trust_remote_code=args.trust_remote_code,
    )
