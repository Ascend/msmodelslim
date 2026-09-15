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

import datetime
from pathlib import Path
from typing import Optional, Union, List

from msmodelslim.core.quant_service import IQuantService
from msmodelslim.core.tune_strategy import ITuningStrategyFactory
from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModelFactory, IModel
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.validation.conversion import (
    convert_to_readable_dir,
    convert_to_writable_dir,
    convert_to_timedelta,
    convert_to_bool,
)
from msmodelslim.utils.validation.type import check_element_type, check_type
from .evaluation_service_infra import EvaluateServiceInfra, EvaluateContext
from .model_info_interface import ModelInfoInterface
from .plan_manager_infra import TuningPlanManagerInfra
from .practice_history_infra import TuningHistoryManagerInfra
from .practice_accuracy_infra import TuningAccuracyManagerInfra
from .practice_manager_infra import PracticeManagerInfra

MAX_ITERATION = 30


@logger_setter()
class AutoTuningApplication:
    def __init__(
        self,
        plan_manager: TuningPlanManagerInfra,
        practice_manager: PracticeManagerInfra,
        evaluation_service: EvaluateServiceInfra,
        tuning_history_manager: TuningHistoryManagerInfra,
        tuning_accuracy_manager: TuningAccuracyManagerInfra,
        quantization_service: IQuantService,
        model_factory: IModelFactory,
        strategy_factory: ITuningStrategyFactory,
    ) -> None:
        self.plan_manager = plan_manager
        self.practice_manager = practice_manager
        self.evaluation_service = evaluation_service
        self.quantization_service = quantization_service
        self.model_factory = model_factory
        self.strategy_factory = strategy_factory
        self.tuning_history_manager = tuning_history_manager
        self.tuning_accuracy_manager = tuning_accuracy_manager

    def tune(
        self,
        model_type: str,
        model_path: Union[Path, str],
        save_path: Union[Path, str],
        plan_id: str,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
        timeout: Optional[Union[datetime.timedelta, str]] = None,
        trust_remote_code: bool = False,
    ) -> None:
        """
        Run the auto tuning application.
        Args:
            model_type: str, the type of the model
            model_path: Union[Path, str], the path of the model
            save_path: Union[Path, str], the directory to save the model
            plan_id: str, the id of the tuning plan
            device: DeviceType, the device to run the model
            device_indices: Optional[List[int]], the indices of the devices to run the model
            timeout: Optional[Union[datetime.timedelta, str]], the timeout of the tuning
            trust_remote_code: bool, whether to trust the remote code
        """
        check_type(model_type, str, param_name="model_type")
        model_path = convert_to_readable_dir(model_path, param_name="model_path")
        save_path = convert_to_writable_dir(save_path, param_name="save_path")
        check_type(plan_id, str, param_name="plan_id")
        check_type(device, DeviceType, param_name="device")
        if device_indices is not None:
            check_element_type(device_indices, int, param_name="device_indices")
        if timeout is not None:
            timeout = convert_to_timedelta(timeout, param_name="timeout")
        trust_remote_code = convert_to_bool(trust_remote_code, param_name="trust_remote_code")

        get_logger().info('Auto tuning with following parameters:')
        get_logger().info("model_type: %r", model_type)
        get_logger().info("model_path: %r", model_path)
        get_logger().info("save_path: %r", save_path)
        get_logger().info("plan_id: %r", plan_id)
        get_logger().info("device: %r", device)
        if device_indices:
            get_logger().info("device_indices: %r", device_indices)
        if timeout:
            get_logger().info("timeout: %r", timeout)
        get_logger().info("trust_remote_code: %r", trust_remote_code)

        self._tune(model_type, model_path, save_path, plan_id, device, device_indices, timeout, trust_remote_code)

    def _tune(
        self,
        model_type: str,
        model_path: Path,
        save_path: Path,
        plan_id: str,
        device: DeviceType,
        device_indices: Optional[List[int]],
        timeout: datetime.timedelta,
        trust_remote_code: bool,
    ) -> None:
        # analyse model
        get_logger().info("===========ANALYSE MODEL===========")
        model_adapter = self.model_factory.create(model_type, model_path, trust_remote_code)
        get_logger().info("Using model adapter %r.", model_adapter.__class__.__name__)

        # create plan
        get_logger().info("===========CREATE TUNING PLAN===========")
        plan = self.plan_manager.get_plan_by_id(plan_id)
        get_logger().info("Using plan %r.", plan_id)

        # create strategy
        get_logger().info("===========CREATE TUNING STRATEGY===========")
        strategy = self.strategy_factory.create_strategy(strategy_config=plan.strategy)
        get_logger().info("Using strategy %r.", plan.strategy.type)

        # start tuning
        get_logger().info("===========START TUNING===========")
        datetime_start = datetime.datetime.now()
        allowed_end_time = datetime_start + timeout if timeout is not None else None
        get_logger().info("Start time: %r, timeout: %r", datetime_start, timeout)

        # Check if history exists and can be resumed
        get_logger().info("===========CHECK HISTORY===========")
        history_path = str(save_path / "history")
        history = self.tuning_history_manager.load_history(history_path)

        # Clear history records
        history.clear_records()

        # Load accuracy record
        accuracy_record = self.tuning_accuracy_manager.load_accuracy(history_path)
        accuracy_count = accuracy_record.get_accuracy_count()
        if accuracy_count > 0:
            get_logger().info(
                "Detected %d existing accuracy records in %r. Will attempt to reuse accuracy records.",
                accuracy_count,
                history_path,
            )
        else:
            get_logger().info("No existing accuracy records found in %r. Starting fresh tuning.", history_path)

        # create quant config generator
        practice_generator = strategy.generate_practice(model=model_adapter)
        evaluate_result = None
        practice = None

        # start tuning
        for count in range(MAX_ITERATION):
            # check timeout
            current_time = datetime.datetime.now()
            if allowed_end_time and current_time > allowed_end_time:
                get_logger().warning("Current time: %r exceed allowed end time: %r!", current_time, allowed_end_time)
                get_logger().warning("===========TIMEOUT===========")
                break

            try:
                get_logger().info("===========TRY %r===========", count)
                get_logger().info("Current time: %r", current_time)
                # generate quant config
                practice = practice_generator.send(evaluate_result)
                get_logger().debug("Practice: %r", practice)
                get_logger().info("Generate practice success")

                evaluate_result = None  # reset evaluate_result
                # Path 1: Try to get accuracy from accuracy records
                evaluate_result = accuracy_record.get_accuracy(practice, plan.evaluation)
                if evaluate_result is not None:
                    get_logger().info("Got accuracy from accuracy records for iteration %d", count)
                    for accuracy_unit in evaluate_result.accuracies:
                        get_logger().info(
                            "Accuracy from accuracy records of %r: %r", accuracy_unit.dataset, accuracy_unit.accuracy
                        )

                # Path 2: If not found in accuracy records, quantize and evaluate
                if evaluate_result is None:
                    quant_model_path = save_path / "quant_model"

                    # quantize model
                    self.quantization_service.quantize(
                        practice.extract_quant_config(),
                        model_adapter=model_adapter,
                        save_path=quant_model_path,
                        device=device,
                        device_indices=device_indices,
                    )
                    get_logger().info("Quantize model success")

                    # evaluate model
                    evaluate_result = self.evaluation_service.evaluate(
                        context=EvaluateContext(
                            evaluate_id=str(count),
                            device=device,
                            device_indices=device_indices,
                            working_dir=save_path,
                        ),
                        evaluate_config=plan.evaluation,
                        model_path=quant_model_path,
                    )
                    get_logger().info("Evaluate model success")
                    for accuracy_unit in evaluate_result.accuracies:
                        get_logger().info("Evaluate Accuracy of %r: %r", accuracy_unit.dataset, accuracy_unit.accuracy)

                    # Save accuracy record only after new evaluation
                    accuracy_record.append_accuracy(practice, plan.evaluation, evaluate_result)
                    get_logger().info("Append accuracy success")

                history.append_history(practice, evaluate_result)
                get_logger().info("Append history success")
            except StopIteration:
                get_logger().info("Strategy stop iterating")
                self._save_practice_to_custom_repo(model_adapter, practice)
                get_logger().info("===========SUCCESS===========")
                break
        else:
            get_logger().warning("===========EXCEED MAX TUNING ITERATION: %r===========", MAX_ITERATION)

    def _save_practice_to_custom_repo(self, model_adapter: IModel, practice):
        if not self.practice_manager.is_saving_supported():
            get_logger().warning(
                "Custom Practice Manager is not provided. Final practice will not be saved to Best Practice Repository."
            )
            return
        if not isinstance(model_adapter, ModelInfoInterface):
            get_logger().warning(
                "Model adapter %r does NOT implement ModelInfoInterface. "
                "Final practice will not be saved to Best Practice Repository",
                {model_adapter.__class__.__name__},
            )
            return

        self.practice_manager.save_practice(model_pedigree=model_adapter.get_model_pedigree(), practice=practice)
        get_logger().info("Save practice to repo success")
