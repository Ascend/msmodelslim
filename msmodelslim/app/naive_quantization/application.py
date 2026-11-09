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

import re
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple

from msmodelslim.core.const import DeviceType
from msmodelslim.core.const import QuantType
from msmodelslim.core.practice.interface import Metadata, PracticeConfig, ScenarioTagMatch
from msmodelslim.core.quant_service import IQuantService
from msmodelslim.model import IModelFactory, IModel
from msmodelslim.utils.config import CONFIG_FILE_PATH, msmodelslim_config
from msmodelslim.utils.exception import EnvError, ModelslimError, SchemaValidateError, ToDoError, UnsupportedError
from msmodelslim.utils.exception_decorator import exception_catcher
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.precheck import (
    DISK_SPACE_GROUP,
    NPU_OCCUPANCY_GROUP,
    PATH_PERMISSION_GROUP,
    STRING_FIELD_GROUP,
    PrecheckManager,
    PrecheckReport,
    PrecheckResult,
    PrecheckStatus,
    check_disk_space,
    npu_index_check,
    npu_occupancy_check,
    precheck_rule,
    readable_path_check,
    string_field_check,
    writable_path_check,
)
from msmodelslim.utils.security import yaml_safe_load
from msmodelslim.utils.validation.conversion import convert_to_writable_dir
from .model_info_interface import ModelInfoInterface
from .practice_manager_infra import PracticeManagerInfra, QuantConfigExportInfra

DEFAULT_PEDIGREE = 'default'
DEFAULT_QUANT_TYPE = QuantType.W8A8
QUANT_ARGUMENT_GROUP = "quant.arguments"
# Minimum free-memory ratio; fail occupancy checks below 90% free memory.
NPU_FREE_MEMORY_RATIO_MIN = 0.9


class TipsType(str, Enum):
    """
    Q1_C0_B0:
    含义解读：
    Q：量化方式是否指定 Q1-指定；Q0-未指定
    C：量化方式是否更改 C1-更改；C0-未更改
    B：是否是最佳实践  B1-是最佳实践；B0-非最佳实践
    """

    Q0C0B0 = "Q0_C0_B0"  # 未指定量化方式，未更改量化方式，非最佳实践，未指定量化方式场景不存在量化方式变更场景
    Q0C0B1 = "Q0_C0_B1"  # 未指定量化方式，未更改量化方式，是最佳实践，未指定量化方式场景不存在量化方式变更场景
    Q1C0B0 = "Q1_C0_B0"  # 已指定量化方式，未更改量化方式，非最佳实践
    Q1C0B1 = "Q1_C0_B1"  # 已指定量化方式，未更改量化方式，是最佳实践，原正常匹配最佳实践场景
    Q1C1B0 = "Q1_C1_B0"  # 已指定量化方式，已更改量化方式，非最佳实践
    Q1C1B1 = "Q1_C1_B1"  # 已指定量化方式，已更改量化方式，是最佳实践


def _build_quant_tips(tips_type: TipsType, model_type: str, quant_type: QuantType, config_id: str) -> str:
    """
    Args:
        tips_type: 提示类型
        model_type:模型类型
        quant_type:量化方式
        config_id:最佳实践文件config_id

    Returns: 提示词
    """

    if tips_type == TipsType.Q0C0B0:
        return (
            f"No quant_type or config_path provided. Default quant_type:{DEFAULT_QUANT_TYPE} will be used."
            f"The default practice:{config_id} for {DEFAULT_QUANT_TYPE} will be used."
        )
    elif tips_type == TipsType.Q0C0B1:
        return (
            f"No quant_type or config_path provided. Default quant_type:{DEFAULT_QUANT_TYPE} will be used."
            f"The best practice:{config_id} for {DEFAULT_QUANT_TYPE} will be used."
        )
    elif tips_type == TipsType.Q1C0B0:
        return (
            f"No best practice found for model_type={model_type} and quant_type={quant_type}. "
            f"The default practice:{config_id} for {quant_type} will be used."
        )
    elif tips_type == TipsType.Q1C0B1:
        return ""
    elif tips_type == TipsType.Q1C1B0:
        return (
            f"No best practice found for model_type={model_type} and quant_type={quant_type}. "
            f"The default practice:{config_id} for {DEFAULT_QUANT_TYPE} will be used."
        )
    elif tips_type == TipsType.Q1C1B1:
        return (
            f"No best practice found for model_type={model_type} and quant_type={quant_type}. "
            f"The best practice:{config_id} for {DEFAULT_QUANT_TYPE} will be used."
        )
    else:
        raise UnsupportedError("Get best practice error", action="Please use the correct msmodelslim version.")


def check_model_type_transformers(model_adapter: IModel, config: PracticeConfig):
    if not hasattr(model_adapter, 'get_model_type'):
        return
    model_type = model_adapter.get_model_type()
    if model_type == "transformers" and config.apiversion == "multimodal_vlm_modelslim_v1":
        raise UnsupportedError(
            "VLM quantization is not supported for model_type transformers",
            action="Please use a dedicated model adapter for multimodal models",
        )
    if model_type == "transformers" and config.apiversion == "multimodal_sd_modelslim_v1":
        raise UnsupportedError(
            "DiT quantization is not supported for model_type transformers",
            action="Please use a dedicated model adapter for multimodal models",
        )


@precheck_rule(group=QUANT_ARGUMENT_GROUP, rule_id=QUANT_ARGUMENT_GROUP)
def check_quant_arguments(device_type, quant_type, config_path, trust_remote_code, tag, model_type):
    """Validate quantization-specific argument semantics."""
    if not isinstance(device_type, DeviceType):
        raise SchemaValidateError(
            "device_type must be a DeviceType, but got %s" % type(device_type),
            action="Please use DeviceType.NPU or DeviceType.CPU.",
        )
    if quant_type is not None and config_path is not None:
        raise SchemaValidateError("quant_type and config_path only one can be provided")
    if quant_type is not None and not isinstance(quant_type, QuantType):
        raise SchemaValidateError("quant_type must be a QuantType")

    if not isinstance(trust_remote_code, bool):
        raise SchemaValidateError("trust_remote_code must be a bool")
    if tag is not None and not isinstance(tag, list):
        raise SchemaValidateError("tag must be a list or None, but got %s" % type(tag))

    if model_type is None:
        from msmodelslim.core.quant_service.modelslim_convert.config_detect import is_modelslim_convert_config

        if config_path is None or not is_modelslim_convert_config(config_path):
            raise SchemaValidateError("model_type is required unless config_path uses apiversion modelslim_convert")


def log_skipped_checks(report: PrecheckReport) -> None:
    """Log configuration skips once per group.

    Only a check disabled by configuration is announced here: a rule that cannot
    probe the environment returns normally and stays silent, so the precheck output
    is either this hint or the raised failure. Other applications render the same
    ``PrecheckReport`` in their own way, which is why the framework itself neither
    logs nor raises.
    """
    logger = get_logger()
    for item in dict.fromkeys(result.group for result in report.results if result.config_skipped):
        logger.info("%s skipped; if you want this check, add %s to %s", item, item, CONFIG_FILE_PATH)


def _failure_label(result: PrecheckResult) -> str:
    """Return the short user-facing label of a failed check.

    The group prefix is dropped so ``path.permission.model_path`` is reported as
    ``model_path`` and ``npu.NPU:3`` as ``NPU:3``; items whose id equals their
    group (such as ``disk``) keep the full id.
    """
    prefix = "%s." % result.group if result.group else ""
    if prefix and result.check_id.startswith(prefix):
        return result.check_id[len(prefix) :]
    return result.check_id


def _failure_detail(result: PrecheckResult) -> str:
    """Render the complete exception behind one failed check."""
    error = result.error
    if isinstance(error, ModelslimError):
        # repr carries type, code, message and action, e.g.
        # "[SecurityError] Code: 204, Message: ..., TIP: ..."
        return repr(error)
    detail = result.reason or ("" if error is None else str(error))
    if result.suggestion and result.suggestion != detail:
        detail = "%s, TIP: %s" % (detail, result.suggestion) if detail else "TIP: %s" % result.suggestion
    return detail


def _failure_line(result: PrecheckResult) -> str:
    """Render one failed check as ``label: <complete exception>``."""
    detail = _failure_detail(result)
    label = _failure_label(result)
    return "%s: %s" % (label, detail) if detail else label


def raise_for_report(report: PrecheckReport) -> None:
    """Abort quantization with this application's failure summary.

    A single failure re-raises its own exception so the precise type is kept.
    Multiple failures are reported by one ``EnvError``: the outer layer is
    always ``EnvError`` while the message carries every failed item on its own
    line together with that item's complete exception (type, code, message and
    tip). Failed items stay available on ``precheck_failures`` for programmatic
    callers. This format belongs to the quantization CLI; other applications may
    raise differently.
    """
    failed = [result for result in report.results if result.status == PrecheckStatus.FAIL]
    if not failed:
        return
    if len(failed) == 1 and isinstance(failed[0].error, ModelslimError):
        error = failed[0].error
    else:
        lines = ["Precheck failed"]
        lines.extend(_failure_line(result) for result in failed)
        skippable_groups = list(dict.fromkeys(result.group for result in failed if result.skippable))
        action = ""
        if skippable_groups:
            action = "remove %s from quant_precheck_item in %s to skip" % (
                ", ".join(skippable_groups),
                CONFIG_FILE_PATH,
            )
        error = EnvError("\n".join(lines), action=action)
    error.precheck_failures = tuple(failed)
    raise error


@logger_setter('msmodelslim.app.naive_quantization')
class NaiveQuantizationApplication:
    def __init__(
        self,
        practice_manager: PracticeManagerInfra,
        quant_service: IQuantService,
        model_factory: IModelFactory,
        quant_config_export_infra: Optional[QuantConfigExportInfra] = None,
    ):
        self.practice_manager = practice_manager
        self.quant_service = quant_service
        self.model_factory = model_factory
        self.quant_config_export_infra = quant_config_export_infra

    @staticmethod
    def check_config(
        metadata: Metadata,
        model_type: str,
        quant_type: QuantType,
        scenario_tags: Optional[List[str]] = None,
        is_default=False,
    ) -> ScenarioTagMatch:
        label = metadata.label
        # Parse quant_type parameters
        match_result = re.match(r'^w(\d+)a(\d+)((?:c8|f[48])?)(s?)$', quant_type.value)
        if not match_result:
            raise ValueError(f"Invalid quant_type format: {quant_type.value}")
        w_bit = int(match_result.group(1))
        a_bit = int(match_result.group(2))
        suffix = match_result.group(3) or ''
        fa_bit = {'f4': 4, 'f8': 8}.get(suffix)
        is_sparse = bool(match_result.group(4))

        # Check if the label matches the quantization parameters
        if label.get('w_bit') != w_bit:
            return ScenarioTagMatch.NO_MATCH
        if label.get('a_bit') != a_bit:
            return ScenarioTagMatch.NO_MATCH
        if is_sparse ^ label.get('is_sparse', False):
            return ScenarioTagMatch.NO_MATCH
        if (suffix == 'c8') ^ label.get('kv_cache', False):
            return ScenarioTagMatch.NO_MATCH

        # FA3 精度匹配：fa_bit 精确匹配，若无 fa_bit 则 fa_quant=True 默认 8bit
        label_fa_bit = label.get('fa_bit')
        if label_fa_bit is None:
            label_fa_bit = 8 if label.get('fa_quant', False) else None
        if fa_bit != label_fa_bit:
            return ScenarioTagMatch.NO_MATCH
        if is_default:
            return ScenarioTagMatch.MATCH

        verified_model_types = getattr(metadata, 'verified_model_types', None)
        if verified_model_types:
            if model_type in verified_model_types:
                return ScenarioTagMatch.MATCH
            return ScenarioTagMatch.NO_MATCH

        return metadata.matches_scenario_tags(model_type, scenario_tags)

    def get_best_practice(
        self,
        model_adapter: IModel,
        quant_type: Optional[QuantType] = None,
        config_path: Optional[Path] = None,
        tag: Optional[List[str]] = None,
    ) -> PracticeConfig:
        """
        获取最佳实践匹配规则如下：
        场景1：指定config_path配置文件，直接采用（含全量前校验：metadata + spec 一次报全），忽略quant_type配置
        场景2：未指定config_path和quant_type，将quant_type置为默认quant_type，然后按照场景3处理
        场景3：指定quant_type，查找最佳实践规则如下（优先级从高到低）：
        当前pedigree + 指定quant_type > 默认pedigree + 指定quant_type
        > 当前pedigree + 默认quant_type > 默认pedigree + 默认quant_type
        """
        # Handle explicit config path — 全量前校验（metadata + spec 一次报全）
        if config_path is not None:
            config = PracticeConfig.model_validate(yaml_safe_load(str(config_path)))
            get_logger().info("Naive Quant apply config_path: %s", config_path)
            check_model_type_transformers(model_adapter, config)
            # spec.model_adapter 与 spec 其他字段一起在权重加载前完成校验
            self._apply_model_adapter_spec(
                model_adapter,
                getattr(model_adapter, 'get_model_type', lambda: None)(),
                config.spec,
            )
            return config

        if not isinstance(model_adapter, ModelInfoInterface):
            raise ToDoError(
                f"Model adapter {model_adapter.__class__.__name__} does NOT implement ModelInfoInterface",
                action="Please implement ModelInfoInterface to support get best practice.",
            )

        model_type = model_adapter.get_model_type()
        model_pedigree = model_adapter.get_model_pedigree()

        # Handle unknown model
        if model_pedigree not in self.practice_manager:
            raise ToDoError(
                f"model_pedigree {model_pedigree} does NOT exist",
                action=f"Maybe you need change model_pedigree of model_adapter "
                f"or add {model_pedigree} in lab_practice.",
            )

        (metadata, raw_dict), tips = self.get_config(model_pedigree, model_type, quant_type, tag)

        if tips != "":
            user_input = input(tips + "(Enter y to continue, otherwise it will exit): ").strip().lower()[:3]
            if user_input != 'y':
                raise UnsupportedError(
                    f"No best practice found for model_type={model_type} and quant_type={quant_type}",
                    action="You can specify the quantization configuration through config_path or change quant_type.",
                )
        # 自动匹配场景：选中后立即做全量强校验（metadata + spec 一次报全，先于权重加载）
        config = PracticeConfig.model_validate(raw_dict)
        # spec.model_adapter 与 spec 其他字段一起在权重加载前完成校验
        self._apply_model_adapter_spec(model_adapter, model_type, config.spec)
        return config

    def _apply_model_adapter_spec(self, model_adapter: IModel, model_type: Optional[str], spec) -> None:
        """校验并应用 ``spec.model_adapter``。

        校验发生在 best-practice 预检阶段（与 spec 其他字段同一时机、先于权重加载）：
        ``model_adapter`` 声明为普通 dict 时，若内容不符合适配器 Loader 提供的
        pydantic 配置类，在此一次报出，而不是到流水线中途才暴露。
        """
        adapter_spec = getattr(spec, 'model_adapter', None)
        if not isinstance(adapter_spec, dict) or not adapter_spec:
            return
        if not hasattr(model_adapter, 'apply_adapter_config'):
            raise SchemaValidateError(
                "spec.model_adapter is declared but the model adapter does not accept model-adapter control parameters",
                action="Remove spec.model_adapter, or make the adapter loader "
                "provide a config class via get_adapter_config_class().",
            )
        config_cls = self.model_factory.get_adapter_config_class(model_type)
        if config_cls is None:
            raise SchemaValidateError(
                "spec.model_adapter is declared but the adapter loader does not provide a config class",
                action="Implement get_adapter_config_class() on the adapter loader, or remove spec.model_adapter.",
            )
        adapter_config = config_cls.model_validate(adapter_spec)
        model_adapter.apply_adapter_config(adapter_config)
        get_logger().info(
            "Applied spec.model_adapter to %s: %s",
            model_adapter.__class__.__name__,
            adapter_config,
        )

    def get_config(
        self,
        model_pedigree: str,
        model_type: str,
        quant_type: Optional[QuantType] = None,
        tag: Optional[List[str]] = None,
    ) -> Tuple[Tuple[Metadata, dict], str]:
        has_quant_type = quant_type is not None
        use_quant_type = quant_type if quant_type is not None else DEFAULT_QUANT_TYPE
        standby_configs: List[Tuple[Metadata, dict]] = []
        is_default = model_pedigree == DEFAULT_PEDIGREE

        def _check(
            metadata: Metadata, model_type: str, qt: QuantType, tag: Optional[List[str]] = None, is_default=False
        ):
            return self.check_config(metadata, model_type, qt, tag, is_default)

        def _build_return(metadata: Metadata, raw_dict: dict, tips_type: TipsType, qt: QuantType):
            tips = _build_quant_tips(tips_type, model_type, quant_type, metadata.config_id)
            return (metadata, raw_dict), tips

        # 场景1：【指定量化方式】在模型适配器的最佳实践目录搜索指定量化类型的最佳实践
        for metadata, raw_dict in self.practice_manager.iter_config(model_pedigree):
            result = _check(metadata, model_type, use_quant_type, tag, is_default)
            if result == ScenarioTagMatch.NO_MATCH:
                continue
            if result == ScenarioTagMatch.STANDBY:
                standby_configs.append((metadata, raw_dict))
                continue
            # 默认模型适配器（未知模型）
            if has_quant_type:
                tips_type = TipsType.Q1C0B0 if model_pedigree == DEFAULT_PEDIGREE else TipsType.Q1C0B1
            else:
                tips_type = TipsType.Q0C0B0 if model_pedigree == DEFAULT_PEDIGREE else TipsType.Q0C0B1
            return _build_return(metadata, raw_dict, tips_type, use_quant_type)

        if standby_configs:
            metadata, raw_dict = standby_configs[0]
            tips = (
                f"No config verified for tags {tag}, including device_type and inference_engine. "
                f"Using standby config: {metadata.config_id}. "
            )
            return (metadata, raw_dict), tips

        # 场景2：【指定量化方式】在最佳实践的default目录搜索指定量化类型的最佳实践
        if model_pedigree != DEFAULT_PEDIGREE:
            for metadata, raw_dict in self.practice_manager.iter_config(DEFAULT_PEDIGREE):
                result = _check(metadata, model_type, use_quant_type, tag, is_default=True)
                if result == ScenarioTagMatch.NO_MATCH:
                    continue
                tips_type = TipsType.Q1C0B0 if has_quant_type else TipsType.Q0C0B0
                return _build_return(metadata, raw_dict, tips_type, use_quant_type)

        if use_quant_type == DEFAULT_QUANT_TYPE or not has_quant_type:
            raise UnsupportedError("Get best practice error", action="Please use the correct msmodelslim version.")

        # 场景3：【默认量化方式】在模型适配器的最佳实践目录搜索默认量化类型的最佳实践
        for metadata, raw_dict in self.practice_manager.iter_config(model_pedigree):
            result = _check(metadata, model_type, DEFAULT_QUANT_TYPE, tag, is_default)
            if result == ScenarioTagMatch.NO_MATCH:
                continue
            if result == ScenarioTagMatch.STANDBY:
                standby_configs.append((metadata, raw_dict))
                continue
            tips_type = TipsType.Q1C1B0 if model_pedigree == DEFAULT_PEDIGREE else TipsType.Q1C1B1
            return _build_return(metadata, raw_dict, tips_type, DEFAULT_QUANT_TYPE)

        if standby_configs:
            metadata, raw_dict = standby_configs[0]
            tips = (
                f"No config verified for tags {tag}, including device_type and inference_engine. "
                f"Using standby config: {metadata.config_id}. "
            )
            return (metadata, raw_dict), tips

        # 场景4：【默认量化方式】在最佳实践的default目录搜索默认量化类型的最佳实践
        if model_pedigree != DEFAULT_PEDIGREE:
            for metadata, raw_dict in self.practice_manager.iter_config(DEFAULT_PEDIGREE):
                result = _check(metadata, model_type, DEFAULT_QUANT_TYPE, tag, is_default=True)
                if result == ScenarioTagMatch.NO_MATCH:
                    continue
                return _build_return(metadata, raw_dict, TipsType.Q1C1B0, DEFAULT_QUANT_TYPE)

        raise UnsupportedError("Get best practice error", action="Please use the correct msmodelslim version.")

    @exception_catcher
    def quant(
        self,
        model_type: Optional[str],
        model_path: str,
        save_path: str,
        device_type: DeviceType = DeviceType.NPU,
        device_index: Optional[List[int]] = None,
        quant_type: Optional[QuantType] = None,
        config_path: Optional[str] = None,
        trust_remote_code: bool = False,
        tag: Optional[List[str]] = None,
    ):
        """
        Run the naive quantization application.
        Args:
            model_type: Optional[str], the type of the model; omit when config_path uses apiversion modelslim_convert
            model_path: str, the path of the model
            save_path: str, the path to save the quantized model
            device_type: DeviceType, the type of device (e.g., DeviceType.NPU, DeviceType.CPU)
                        Default: DeviceType.NPU
            device_index: Optional[List[int]], list of device indices to use (e.g., [0, 1, 2, 3])
                         If None, uses single default device
                         Default: None
            quant_type: Optional[QuantType], the quantization type, config_path and quant_type only one can be provided
            config_path: Optional[str], the path to config file, config_path and quant_type only one can be provided
            trust_remote_code: bool, whether to trust the remote code
            tag: Optional[List[str]], e.g. ['vLLM-Ascend','Atlas_A2_Inference'], tags to match configs with verified_tags
        """
        get_logger().info("===========PRECHECK===========")
        configured_items = msmodelslim_config.precheck_items.get("quant_precheck_item", set())
        invalid_items = configured_items - {DISK_SPACE_GROUP, NPU_OCCUPANCY_GROUP}
        if invalid_items:
            get_logger().warning(
                "Ignoring unknown precheck items: %s; supported items are: %s, %s",
                sorted(invalid_items),
                DISK_SPACE_GROUP,
                NPU_OCCUPANCY_GROUP,
            )

        manager = PrecheckManager()
        string_id = f"{STRING_FIELD_GROUP}.config_path"
        path_id = f"{PATH_PERMISSION_GROUP}.config_path"
        quant_argument_values = (device_type, quant_type, config_path, trust_remote_code, tag, model_type)
        for name, value in (("model_path", model_path), ("save_path", save_path)):
            manager.register(string_field_check.bind(value, name, rule_id=f"{STRING_FIELD_GROUP}.{name}"))
        # model_type is optional: it may be omitted when config_path uses apiversion modelslim_convert.
        manager.register(
            string_field_check.bind(
                model_type, "model_type", skip_when_none=True, rule_id=f"{STRING_FIELD_GROUP}.model_type"
            )
        )
        manager.register(string_field_check.bind(config_path, "config_path", skip_when_none=True, rule_id=string_id))
        manager.register(readable_path_check.bind(model_path, rule_id=f"{PATH_PERMISSION_GROUP}.model_path"))
        manager.register(writable_path_check.bind(save_path, rule_id=f"{PATH_PERMISSION_GROUP}.save_path"))
        manager.register(readable_path_check.bind(config_path, file=True, skip_when_none=True, rule_id=path_id))
        manager.register(check_quant_arguments.bind(*quant_argument_values))
        manager.register(check_disk_space.bind(model_path, save_path))
        manager.register(npu_index_check.bind(device_type, device_index))
        # Register occupancy per NPU; npu.index reports invalid indices.
        requested = (device_index or [0]) if device_index is None or isinstance(device_index, list) else []
        for index in dict.fromkeys(
            value for value in requested if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ):
            rule_id = f"{NPU_OCCUPANCY_GROUP}.NPU:{index}"
            manager.register(
                npu_occupancy_check.bind(
                    device_type,
                    index,
                    free_ratio_min=NPU_FREE_MEMORY_RATIO_MIN,
                    rule_id=rule_id,
                )
            )

        report = manager.run(configured_items)
        log_skipped_checks(report)
        raise_for_report(report)

        model_path = Path(model_path).resolve()
        config_path = Path(config_path).resolve() if config_path is not None else None
        save_path = Path(convert_to_writable_dir(save_path))
        tag = tag or None

        # Log parameters
        get_logger().info("quantization with following parameters:")
        get_logger().info("model_type: %s", model_type)
        get_logger().info("model_path: %s", model_path)
        get_logger().info("save_path: %s", save_path)
        get_logger().info("device_type: %s", device_type)
        if device_index is not None and len(device_index) > 1:
            device_list = ','.join(map(str, device_index))
            get_logger().info(
                "using %d devices: %s:%s",
                len(device_index),
                device_type.value,
                device_list,
            )
        elif device_index is not None and len(device_index) == 1:
            get_logger().info("using single device: %s:%s", device_type.value, device_index[0])
        else:
            get_logger().info("using single device (default): %s", device_type.value)
        if quant_type is not None:
            get_logger().info("quant_type: %s", quant_type)
        if config_path is not None:
            get_logger().info("config_path: %s", config_path)
        get_logger().info("trust_remote_code: %s", trust_remote_code)
        if tag:
            get_logger().info("tag: %s", tag)

        self._quant(
            model_type,
            model_path,
            save_path,
            device_type,
            device_index,
            quant_type,
            config_path,
            trust_remote_code,
            tag,
        )

    def _quant(
        self,
        model_type: Optional[str],
        model_path: Path,
        save_path: Path,
        device_type: DeviceType = DeviceType.NPU,
        device_index: Optional[List[int]] = None,
        quant_type: Optional[QuantType] = None,
        config_path: Optional[Path] = None,
        trust_remote_code: bool = False,
        tag: Optional[List[str]] = None,
    ):
        get_logger().info("===========ANALYSE MODEL===========")
        from msmodelslim.core.quant_service.modelslim_convert.config_detect import is_modelslim_convert_config
        from msmodelslim.model.base import BaseModelAdapter

        use_convert_adapter = config_path is not None and is_modelslim_convert_config(config_path)
        if use_convert_adapter:
            model_adapter = BaseModelAdapter(
                model_type=model_type or "convert",
                model_path=model_path,
                trust_remote_code=trust_remote_code,
            )
            get_logger().info("Using BaseModelAdapter for modelslim_convert (no model code load).")
        else:
            model_adapter = self.model_factory.create(model_type, model_path, trust_remote_code)
            get_logger().info("Using model adapter %s.", model_adapter.__class__.__name__)

        get_logger().info("===========GET BEST PRACTICE===========")
        practice_config = self.get_best_practice(
            model_adapter=model_adapter,
            quant_type=quant_type,
            config_path=config_path,
            tag=tag,
        )
        # 使用量化配置导出基础设施导出配置
        export_model_type = model_type or "convert"
        self.quant_config_export_infra.export_quant_config(practice_config, export_model_type, save_path)

        get_logger().info("Get best practice %s success.", practice_config.metadata.config_id)

        get_logger().info("===========QUANTIZE MODEL===========")
        self.quant_service.quantize(
            quant_config=practice_config.extract_quant_config(),
            model_adapter=model_adapter,
            save_path=save_path,
            device=device_type,
            device_indices=device_index,
        )
        get_logger().info("===========SUCCESS===========")
