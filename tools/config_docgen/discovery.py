#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=too-many-lines
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

基础抽取脚本：从 Pydantic 配置模型注解按指定目标生成量化配置接口文档。

提供基础能力（--targets 指定特定对象、--expand-nested 控制是否展开嵌套）；
具体抽取目标与后处理由 generate.py 实现，由 main.py 暴露稳定入口。
"""

from __future__ import annotations

import argparse
import enum
import importlib
import inspect
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, get_origin

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel  # noqa: E402

from tools.config_docgen.catalog import (  # noqa: E402
    APIVERSION_BY_ROOT,
    CATEGORY_DIRS,
    CONFIG_SOURCE_MODULES,
    MANUAL_KEEP,
    QUALNAME_SLUGS,
    SKIP_CLASS_NAMES,
    SLUG_OVERRIDES,
    TARGET_GROUP_KEYWORDS,
)
from tools.config_docgen.examples import (  # noqa: E402
    load_explicit_examples,
    load_practice_config_example,
    load_scenarios,
    load_tuning_examples,
)
from tools.config_docgen.records import ModelRecord  # noqa: E402
from tools.config_docgen.renderer import (  # noqa: E402
    GENERATED_MARKER,
    LEGACY_GENERATED_MARKERS,
    _anchor,
    _doc_path,
    render_markdown,
)
from tools.config_docgen.schema import (  # noqa: E402
    _literal_values,
    annotation_refers_to,
    class_doc_sections,
    collect_nested_models,
    collect_validator_docs,
    extract_fields,
    extractable_example_payload,
    is_internal_type_tag,
    is_pydantic_model,
    public_type_tag,
    raw_type_tag,
    rewrite_example_internal_types,
    slugify_class,
    split_optional,
    unwrap_annotation,
    wrap_example_at_path,
)


def _ensure_package_config_link() -> Optional[Path]:
    """兼容历史调用；项目级生成器不创建 msmodelslim/config 软链接。"""
    return None


class ConfigModuleImportError(RuntimeError):
    """Raised when a known config source module cannot be imported."""


def _import_known_modules() -> None:
    """导入全部已知配置事实源；任一模块失败都必须中止生成。"""
    failures: List[str] = []
    for name in CONFIG_SOURCE_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - a missing fact source must fail closed
            failures.append(f"{name}: {exc}")
    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise ConfigModuleImportError(f"配置模块导入失败，无法生成完整文档：\n{details}")


def _source_rel(model_cls: type) -> str:
    try:
        path = Path(inspect.getfile(model_cls)).resolve()
    except TypeError:
        return ""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _public_processors(auto_cls: type) -> List[type]:
    registry = getattr(auto_cls, "_registry", set())
    result = []
    for cls in registry:
        if cls.__name__ in SKIP_CLASS_NAMES:
            continue
        tag = public_type_tag(cls)
        if tag is None or is_internal_type_tag(tag):
            continue
        if _is_saver_config(cls):
            continue
        result.append(cls)
    return sorted(result, key=lambda c: public_type_tag(c) or c.__name__)


def _is_saver_config(cls: type) -> bool:
    try:
        from msmodelslim.core.quant_service.modelslim_v1.save.saver import AutoSaverBaseConfig
        from msmodelslim.format.base import QuantFormatConfig
    except Exception:
        return "Saver" in cls.__name__ or "Format" in cls.__name__
    return issubclass(cls, AutoSaverBaseConfig) or issubclass(cls, QuantFormatConfig)


def _public_formats() -> List[type]:
    from msmodelslim.format.base import QuantFormatConfig

    result = []
    for cls in QuantFormatConfig.__subclasses__():
        tag = public_type_tag(cls)
        if tag is None or is_internal_type_tag(tag):
            continue
        result.append(cls)
    return sorted(result, key=lambda c: public_type_tag(c) or c.__name__)


def _root_configs() -> List[type]:
    from msmodelslim.core.practice.interface import PracticeConfig
    from msmodelslim.core.quant_service.modelslim_convert.quant_config import ModelslimConvertQuantConfig
    from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1QuantConfig
    from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import (
        MultimodalSDModelslimV1QuantConfig,
    )
    from msmodelslim.core.quant_service.multimodal_vlm_v1.quant_config import (
        MultimodalVLMModelslimV1QuantConfig,
    )

    return [
        ModelslimV1QuantConfig,
        MultimodalVLMModelslimV1QuantConfig,
        MultimodalSDModelslimV1QuantConfig,
        ModelslimConvertQuantConfig,
        PracticeConfig,
    ]


def _extra_convert_models() -> List[type]:
    from msmodelslim.core.quant_service.modelslim_convert.config_mapper import (
        ConvertOpConfig,
        ConvertPreprocessConfig,
        RenamePattern,
        RenamePreprocessConfig,
    )

    return [
        RenamePreprocessConfig,
        ConvertPreprocessConfig,
        RenamePattern,
        ConvertOpConfig,
    ]


def _tlq_op_subclasses() -> List[type]:
    """可训练 TLQ 算子（operations）的派生配置类。"""
    from msmodelslim.processor.trainable_linear_quant.core.ops.base import TLQOpConfig

    return [s for s in TLQOpConfig.__subclasses__() if is_pydantic_model(s)]


def _tuning_plan_config() -> type:
    from msmodelslim.app.auto_tuning.plan_manager_infra import TuningPlanConfig

    return TuningPlanConfig


def _strategy_config_base() -> type:
    from msmodelslim.core.tune_strategy.interface import StrategyConfig

    return StrategyConfig


def _evaluate_service_config_base() -> type:
    from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateServiceConfig

    return EvaluateServiceConfig


def _public_strategies() -> List[type]:
    from msmodelslim.core.tune_strategy.binary_fallback.strategy import BinaryFallbackStrategyConfig
    from msmodelslim.core.tune_strategy.standing_high.strategy import StandingHighStrategyConfig
    from msmodelslim.core.tune_strategy.standing_high_with_experience.strategy import (
        StandingHighWithExperienceStrategyConfig,
    )

    return [
        StandingHighStrategyConfig,
        StandingHighWithExperienceStrategyConfig,
        BinaryFallbackStrategyConfig,
    ]


def _public_evaluations() -> List[type]:
    from msmodelslim.infra.service_oriented_evaluate_service import ServiceOrientedEvaluateServiceConfig

    return [ServiceOrientedEvaluateServiceConfig]


def _precheck_config_base() -> type:
    from msmodelslim.infra.evaluation.precheck.base import BasePrecheckConfig

    return BasePrecheckConfig


def _public_prechecks() -> List[type]:
    base = _precheck_config_base()
    return sorted(
        [cls for cls in base.__subclasses__() if is_pydantic_model(cls) and _should_keep(cls)],
        key=lambda cls: cls.__name__,
    )


def _auto_tuning_targets() -> List[type]:
    return [_tuning_plan_config()] + _public_strategies() + _public_evaluations()


def _should_keep(cls: type) -> bool:
    if not is_pydantic_model(cls):
        return False
    if cls.__name__ in SKIP_CLASS_NAMES or cls.__name__.startswith("_"):
        return False
    # 插件分派基类不作为独立页面（如调优策略/评估服务基类），只用于在上级字段中展开分派索引。
    try:
        from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateServiceConfig
        from msmodelslim.core.tune_strategy.interface import StrategyConfig

        if cls is StrategyConfig or cls is EvaluateServiceConfig:
            return False
    except Exception:  # nosec B110
        pass
    module = getattr(cls, "__module__", "")
    if "test" in module.split("."):
        return False
    tag = raw_type_tag(cls)
    if tag is not None and is_internal_type_tag(tag):
        try:
            from msmodelslim.format.base import QuantFormatConfig
            from msmodelslim.processor.base import AutoProcessorConfig

            if issubclass(cls, QuantFormatConfig) or issubclass(cls, AutoProcessorConfig):
                return False
        except Exception:
            return False
    try:
        from msmodelslim.core.quant_service.modelslim_v1.save.saver import AutoSaverBaseConfig

        if issubclass(cls, AutoSaverBaseConfig):
            return False
    except Exception:  # nosec B110
        pass
    return True


def _collect_seeds() -> List[type]:
    from msmodelslim.processor.base import AutoProcessorConfig

    seeds: List[type] = []
    seeds.extend(_root_configs())
    seeds.extend(_public_processors(AutoProcessorConfig))
    seeds.extend(_public_formats())
    seeds.extend(_extra_convert_models())
    seeds.extend(_auto_tuning_targets())
    seeds.extend(_public_prechecks())
    seeds.extend(_tlq_op_subclasses())
    seen: Set[type] = set()
    ordered: List[type] = []

    def add(cls: type) -> None:
        if cls in seen or not _should_keep(cls):
            return
        seen.add(cls)
        ordered.append(cls)
        for info in cls.model_fields.values():
            for nested in collect_nested_models(info.annotation):
                add(nested)

    for seed in seeds:
        add(seed)
    return ordered


def _category_for(cls: type) -> str:
    from msmodelslim.core.practice.interface import PracticeConfig
    from msmodelslim.core.quant_service.interface import BaseQuantConfig
    from msmodelslim.format.base import QuantFormatConfig
    from msmodelslim.processor.base import AutoProcessorConfig

    # PracticeConfig 是 metadata + task 的最佳实践任务配置，作为独立任务页生成，
    # 用于展示 metadata、task（apiversion + spec）与嵌套配置的完整结构。
    if cls in _auto_tuning_targets():
        return "自动调优"
    if cls is PracticeConfig or issubclass(cls, BaseQuantConfig):
        return "任务配置"
    if cls.__name__.endswith("ServiceConfig"):
        return "服务规格"
    if issubclass(cls, QuantFormatConfig):
        return "保存格式"
    if issubclass(cls, AutoProcessorConfig):
        return "处理器"
    return "嵌套配置"


def _refers_to_saver(annotation: Any) -> bool:
    from msmodelslim.core.quant_service.modelslim_v1.save.saver import AutoSaverBaseConfig
    from msmodelslim.format.base import QuantFormatConfig

    return annotation_refers_to(annotation, QuantFormatConfig) or annotation_refers_to(annotation, AutoSaverBaseConfig)


def _yaml_paths(models: Sequence[type]) -> Tuple[Dict[type, str], Dict[type, type]]:
    from msmodelslim.format.base import QuantFormatConfig
    from msmodelslim.processor.base import AutoProcessorConfig

    paths: Dict[type, str] = {}
    roots: Dict[type, type] = {}
    for cls in _root_configs():
        if cls in models:
            paths[cls] = ""
            roots[cls] = cls
    tuning_plan = _tuning_plan_config()
    if tuning_plan in models:
        paths[tuning_plan] = ""
        roots[tuning_plan] = tuning_plan
    queue: List[Tuple[type, str, type]] = [(cls, path, roots[cls]) for cls, path in paths.items()]
    seen = set(paths)

    def child_path(parent_path: str, field_name: str, is_list: bool) -> str:
        suffix = f"{field_name}[]" if is_list else field_name
        if not parent_path:
            return suffix
        return f"{parent_path}.{suffix}"

    while queue:
        cls, path, root = queue.pop(0)
        for name, info in cls.model_fields.items():
            is_list = "list[" in str(info.annotation).lower() or "List[" in str(info.annotation)
            if _refers_to_saver(info.annotation):
                for fmt in _public_formats():
                    next_path = child_path(path, name, True)
                    if fmt not in seen:
                        seen.add(fmt)
                        paths[fmt] = next_path
                        roots[fmt] = root
                        queue.append((fmt, next_path, root))
            elif annotation_refers_to(info.annotation, AutoProcessorConfig):
                for proc in _public_processors(AutoProcessorConfig):
                    next_path = child_path(path, name, True)
                    if proc not in seen:
                        seen.add(proc)
                        paths[proc] = next_path
                        roots[proc] = root
                        queue.append((proc, next_path, root))
            elif annotation_refers_to(info.annotation, _strategy_config_base()):
                for strategy in _public_strategies():
                    next_path = child_path(path, name, False)
                    if strategy not in seen:
                        seen.add(strategy)
                        paths[strategy] = next_path
                        roots[strategy] = root
                        queue.append((strategy, next_path, root))
            elif annotation_refers_to(info.annotation, _evaluate_service_config_base()):
                for ev in _public_evaluations():
                    next_path = child_path(path, name, False)
                    if ev not in seen:
                        seen.add(ev)
                        paths[ev] = next_path
                        roots[ev] = root
                        queue.append((ev, next_path, root))
            for nested in collect_nested_models(info.annotation):
                if nested is cls or not _should_keep(nested):
                    continue
                if issubclass(nested, AutoProcessorConfig) or issubclass(nested, QuantFormatConfig):
                    continue
                next_path = child_path(path, name, is_list and nested.__name__ != cls.__name__)
                if nested not in seen:
                    seen.add(nested)
                    paths[nested] = next_path
                    roots[nested] = root
                    queue.append((nested, next_path, root))

    path_overrides = {
        "QConfig": "spec.process[].qconfig.weight",
        "LinearQConfig": "spec.process[].qconfig",
        "AdaptRotationStage1ProcessorConfig": "spec.process[]",
        "AdaptRotationStage2ProcessorConfig": "spec.process[]",
        "RenamePreprocessConfig": "spec.preprocess[]",
        "RenamePattern": "spec.preprocess[].patterns[]",
        "ConvertPreprocessConfig": "spec.preprocess[]",
        "ConvertOpConfig": "spec.preprocess[].ops[]",
        "TuningPlanConfig": "",
        "StandingHighStrategyConfig": "strategy",
        "StandingHighWithExperienceStrategyConfig": "strategy",
        "BinaryFallbackStrategyConfig": "strategy",
        "ServiceOrientedEvaluateServiceConfig": "evaluation",
        "MinmaxTuneOpConfig": "spec.process[].operations[]",
        "RoundTuneOpConfig": "spec.process[].operations[]",
        "TrainableSmoothOpConfig": "spec.process[].operations[]",
    }
    convert_names = {
        "RenamePreprocessConfig",
        "RenamePattern",
        "ConvertPreprocessConfig",
        "ConvertOpConfig",
        "LinearConvertConfig",
        "SaveConfig",
        "ParallelSpecConfig",
        "ConvertDefaults",
        "ModelslimConvertServiceConfig",
        "ModelslimConvertQuantConfig",
    }
    tuning_names = {
        "TuningPlanConfig",
        "StandingHighStrategyConfig",
        "StandingHighWithExperienceStrategyConfig",
        "BinaryFallbackStrategyConfig",
        "ServiceOrientedEvaluateServiceConfig",
    }
    convert_root = next(cls for cls in _root_configs() if cls.__name__ == "ModelslimConvertQuantConfig")
    tuning_root = _tuning_plan_config()
    precheck_base = _precheck_config_base()
    for cls in models:
        if cls.__name__ in path_overrides:
            paths[cls] = path_overrides[cls.__name__]
            if cls.__name__ in tuning_names:
                roots[cls] = tuning_root
            else:
                roots[cls] = convert_root if cls.__name__ in convert_names else _root_configs()[0]
            continue
        if issubclass(cls, precheck_base):
            # 预检查规则按 type 分派，所有派生规则共享 evaluation.precheck[] 路径。
            paths[cls] = "evaluation.evaluation.precheck[]"
            roots[cls] = tuning_root
            continue
        if cls in paths:
            continue
        roots[cls] = _root_configs()[0]
        if issubclass(cls, AutoProcessorConfig) and not _is_saver_config(cls):
            paths[cls] = "spec.process[]"
        elif issubclass(cls, QuantFormatConfig):
            paths[cls] = "spec.save[]"
        else:
            paths[cls] = "spec"
    return paths, roots


def _annotation_is_list(annotation: Any) -> bool:
    """字段是否为列表类型（unwrap Annotated/SerializeAsAny 后判断）。"""
    ann, _ = unwrap_annotation(annotation)
    ann, _ = split_optional(ann)
    return get_origin(ann) in (list, List)


def _nested_refs(
    cls: type,
    yaml_path: str,
    processors: Sequence[type],
    formats: Sequence[type],
    slugs: Dict[type, str],
) -> List[Tuple[str, str, str]]:
    from msmodelslim.format.base import QuantFormatConfig
    from msmodelslim.processor.base import AutoProcessorConfig

    refs: List[Tuple[str, str, str]] = []
    convert_models = _extra_convert_models()[:2]
    for name, info in cls.model_fields.items():
        field_path = f"{yaml_path}.{name}" if yaml_path else name
        if cls.__name__ == "AisbenchServerConfig" and name == "precheck":
            item_path = f"{field_path}[]"
            for precheck in _public_prechecks():
                if precheck in slugs:
                    refs.append((item_path, slugs[precheck], "type 分派"))
            continue
        if _refers_to_saver(info.annotation):
            # 单个 saver/format 对象（如 saver 处理器的 format）为自动注入，不列出公开格式；
            # 只有列表形式（如 spec.save[]）才是用户可选的保存格式。
            if _annotation_is_list(info.annotation):
                item_path = field_path if field_path.endswith("[]") else f"{field_path}[]"
                for fmt in formats:
                    if fmt in slugs:
                        refs.append((item_path, slugs[fmt], "type 分派"))
            continue
        if annotation_refers_to(info.annotation, AutoProcessorConfig):
            item_path = (
                field_path if _annotation_is_list(info.annotation) and not field_path.endswith("[]") else field_path
            )
            if _annotation_is_list(info.annotation) and not item_path.endswith("[]"):
                item_path = f"{item_path}[]"
            for proc in processors:
                if proc in slugs:
                    refs.append((item_path, slugs[proc], "type 分派"))
            continue
        if annotation_refers_to(info.annotation, _strategy_config_base()):
            for strategy in _public_strategies():
                if strategy in slugs:
                    refs.append((field_path, slugs[strategy], "type 分派"))
            continue
        if annotation_refers_to(info.annotation, _evaluate_service_config_base()):
            for ev in _public_evaluations():
                if ev in slugs:
                    refs.append((field_path, slugs[ev], "type 分派"))
            continue
        if cls.__name__ == "ModelslimConvertServiceConfig" and name == "preprocess":
            for nested in convert_models:
                refs.append((f"{field_path}[]", slugs[nested], "type 分派"))
            continue
        nested = collect_nested_models(info.annotation)
        # 单一基础类 + 已注册派生类（如 TLQOpConfig 的 operations）：把派生类作为
        # 同一字段路径下的嵌套引用，渲染时分派到基础类块下。
        if len(nested) == 1 and is_pydantic_model(nested[0]):
            subs = [
                s
                for s in nested[0].__subclasses__()
                if is_pydantic_model(s) and _should_keep(s) and s in slugs and s is not cls
            ]
            if subs:
                for s in subs:
                    refs.append((field_path, slugs[s], "嵌套对象"))
        for nested in nested:
            if not _should_keep(nested) or nested is cls or nested not in slugs:
                continue
            if issubclass(nested, AutoProcessorConfig) or issubclass(nested, QuantFormatConfig):
                continue
            refs.append((field_path, slugs[nested], "嵌套对象"))
    return refs


def _common_class_suffix(names: Sequence[str]) -> str:
    """多个类名的公共后缀（用于合成 Union 分派基础类名，如 Ema/MinLoss/LastSelectBest → SelectBest）。"""
    if not names:
        return ""
    suffix = names[0]
    for n in names[1:]:
        i = 0
        while i < min(len(suffix), len(n)) and suffix[-(i + 1)] == n[-(i + 1)]:
            i += 1
        suffix = suffix[-i:] if i else ""
        if not suffix:
            break
    return suffix


def _dispatch_bases_for(cls: type, yaml_path: str) -> Dict[str, str]:
    """识别字段的 type/mode 分派，返回 field_path → 基础类显示名。

    覆盖：保存格式 / 处理器 / 策略 / 评估服务（已知基础类）、Union 分派
    （如 select_best 按 mode 分派）、单一基础类 + 已注册派生类（如 TLQOpConfig）。
    """
    from msmodelslim.processor.base import AutoProcessorConfig

    bases: Dict[str, str] = {}
    for name, info in cls.model_fields.items():
        field_path = f"{yaml_path}.{name}" if yaml_path else name
        ann = info.annotation
        if _refers_to_saver(ann) and _annotation_is_list(ann):
            item_path = field_path if field_path.endswith("[]") else f"{field_path}[]"
            bases[item_path] = "QuantFormatConfig"
            continue
        if annotation_refers_to(ann, AutoProcessorConfig):
            item_path = field_path if field_path.endswith("[]") else field_path
            if _annotation_is_list(ann) and not item_path.endswith("[]"):
                item_path = f"{item_path}[]"
            bases[item_path] = "AutoProcessorConfig"
            continue
        if annotation_refers_to(ann, _strategy_config_base()):
            bases[field_path] = "StrategyConfig"
            continue
        if annotation_refers_to(ann, _evaluate_service_config_base()):
            bases[field_path] = "EvaluateServiceConfig"
            continue
        if cls.__name__ == "ModelslimConvertServiceConfig" and name == "preprocess":
            # 转换侧 preprocess 按 type 分派 rename / convert 两类预处理步骤。
            bases[f"{field_path}[]"] = "PreprocessConfig"
            continue
        if cls.__name__ == "AisbenchServerConfig" and name == "precheck":
            bases[f"{field_path}[]"] = "BasePrecheckConfig"
            continue
        nested = collect_nested_models(ann)
        names = [getattr(n, "__name__", str(n)) for n in nested]
        if len(set(names)) >= 2:
            # Union 分派（如 select_best 按 mode 分派）
            suffix = _common_class_suffix(names)
            bases[field_path] = suffix if suffix.endswith("Config") else f"{suffix}Config"
            continue
        if len(nested) == 1 and is_pydantic_model(nested[0]):
            # 单一基础类 + 已注册派生类（如 TLQOpConfig）
            core = nested[0]
            subs = [s for s in core.__subclasses__() if is_pydantic_model(s) and _should_keep(s)]
            if subs:
                bases[field_path] = core.__name__
    return bases


def _title_for(cls: type) -> str:
    tag = public_type_tag(cls)
    name = tag or SLUG_OVERRIDES.get(cls.__name__) or cls.__name__
    return f"{name} 配置说明"


def _summary_for(cls: type, yaml_path: str, category: str, doc_summary: str = "") -> str:
    """配置概述：只用类 docstring，不再拼接「该配置位于 …」这类位置句。

    YAML 位置由页面标题/展开小节标题给出，概述保持简洁。
    """
    if doc_summary:
        return doc_summary
    return f"`{cls.__name__}` 是{category}。"


def _to_plain(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, BaseModel):
        try:
            dumped = value.model_dump(mode="python")
        except Exception:
            return str(value)
        return _to_plain(dumped)
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if inspect.isclass(value):
        return getattr(value, "__name__", str(value))
    return value


def _dump_yaml(data: Any) -> str:
    try:
        import yaml
    except ImportError:
        return str(data)

    def enum_representer(dumper: Any, value: enum.Enum) -> Any:
        return dumper.represent_data(value.value)

    yaml.SafeDumper.add_multi_representer(enum.Enum, enum_representer)
    return yaml.safe_dump(
        _to_plain(data),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).rstrip()


def _apiversion_of(root_cls: type) -> Optional[str]:
    mapped = APIVERSION_BY_ROOT.get(root_cls.__name__)
    if mapped:
        return mapped
    info = getattr(root_cls, "model_fields", {}).get("apiversion")
    if info is not None:
        literals = _literal_values(info.annotation)
        if literals and len(literals) == 1 and str(literals[0]) != "Unknown":
            return str(literals[0])
    module = inspect.getmodule(root_cls)
    if module is None:
        return None
    service_name = module.__name__.replace("quant_config", "quant_service")
    try:
        service_mod = importlib.import_module(service_name)
    except Exception:
        service_mod = module
    for _, obj in inspect.getmembers(service_mod, inspect.isclass):
        fields = getattr(obj, "model_fields", None) or {}
        api_info = fields.get("apiversion")
        if api_info is None:
            continue
        literals = _literal_values(api_info.annotation)
        if literals and len(literals) == 1 and str(literals[0]) != "Unknown":
            return str(literals[0])
    return None


def _sibling_discriminators(parent_cls: type, child_cls: type) -> Dict[str, Any]:
    extras: Dict[str, Any] = {}
    for name, info in parent_cls.model_fields.items():
        nested = collect_nested_models(info.annotation)
        if child_cls not in nested or len(nested) < 2:
            continue
        index = nested.index(child_cls)
        for other_name, other_info in parent_cls.model_fields.items():
            if other_name == name:
                continue
            literals = _literal_values(other_info.annotation)
            if literals and len(literals) == len(nested):
                extras[other_name] = literals[index]
    return extras


def _host_item_for(
    record: ModelRecord,
    cls: type,
    slug_to_cls: Dict[str, type],
    format_standin: str,
    records_by_slug: Optional[Dict[str, ModelRecord]] = None,
) -> Optional[Dict[str, Any]]:
    if record.yaml_path.startswith("spec.process[].qconfig"):
        return {"type": "linear_quant"}
    records_by_slug = records_by_slug or {}
    seen: Set[str] = set()
    queue = list(record.parents)
    while queue:
        parent_slug, _location, _scene = queue.pop(0)
        if parent_slug in seen:
            continue
        seen.add(parent_slug)
        parent_cls = slug_to_cls.get(parent_slug)
        tag = public_type_tag(parent_cls) if parent_cls is not None else None
        if tag:
            item: Dict[str, Any] = {"type": tag}
            if parent_cls is not None:
                item.update(_sibling_discriminators(parent_cls, cls))
            return rewrite_example_internal_types(item, public_format_type=format_standin)
        parent_record = records_by_slug.get(parent_slug)
        if parent_record:
            queue.extend(parent_record.parents)
    return None


def _looks_complete_example(data: Any) -> bool:
    if not isinstance(data, dict) or "spec" not in data:
        return False
    return data.get("apiversion") not in (None, "", "Unknown")


def _validate_example(root_cls: type, data: Dict[str, Any]) -> bool:
    try:
        root_cls.model_validate(data)
        return True
    except Exception:
        return False


def _inject_required_save(root_cls: type, wrapped: Dict[str, Any], format_standin: str) -> None:
    spec = wrapped.get("spec")
    if not isinstance(spec, dict) or spec.get("save"):
        return
    spec_info = getattr(root_cls, "model_fields", {}).get("spec")
    if spec_info is None:
        return
    try:
        from msmodelslim.core.quant_service.modelslim_v1.save.saver import AutoSaverBaseConfig
    except Exception:
        return
    for spec_cls in collect_nested_models(spec_info.annotation):
        save_info = getattr(spec_cls, "model_fields", {}).get("save")
        if save_info is None:
            continue
        if annotation_refers_to(save_info.annotation, AutoSaverBaseConfig):
            spec["save"] = [{"type": format_standin}]
            return


def _fill_empty_nested_lists(cls: type, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(payload)
    changed = False
    for name, info in getattr(cls, "model_fields", {}).items():
        alias = info.serialization_alias or info.alias or name
        if data.get(alias) != []:
            continue
        models = [item for item in collect_nested_models(info.annotation) if is_pydantic_model(item)]
        if not models:
            continue
        item = extractable_example_payload(models[0])
        if not item:
            continue
        data[alias] = [item]
        changed = True
    return data if changed else payload


def _is_tuning_root(root_cls: type) -> bool:
    """自动调优根（TuningPlanConfig）：无 apiversion、示例形态为 strategy + evaluation。"""
    try:
        return root_cls is _tuning_plan_config()
    except Exception:
        return getattr(root_cls, "__name__", "") == "TuningPlanConfig"


def _tuning_example_valid(data: Dict[str, Any]) -> bool:
    """TuningPlanConfig 校验依赖插件注册表（strategy / precheck 分派），docgen 环境
    未注册插件时会失败；回退为对具体策略/评估类的子树校验。
    """
    try:
        _tuning_plan_config().model_validate(data)
        return True
    except Exception:  # nosec B110
        pass
    strategy = data.get("strategy")
    ev = data.get("evaluation")
    if not isinstance(strategy, dict) or not isinstance(ev, dict):
        return False
    strategy_ok = False
    for s in _public_strategies():
        try:
            s.model_validate(strategy)
            strategy_ok = True
            break
        except Exception:  # nosec B112
            continue
    if not strategy_ok:
        return False
    for e in _public_evaluations():
        try:
            e.model_validate(ev)
            return True
        except Exception:  # nosec B112
            continue
    return False


def _tuning_sub_example(class_name: str) -> Optional[Dict[str, Any]]:
    """按显式类名读取调优子树示例，避免依赖注册表迭代顺序。"""
    example = load_tuning_examples().get(class_name)
    return _to_plain(example) if isinstance(example, dict) else None


def _complete_tuning_example(cls: type, format_standin: str) -> str:
    """自动调优配置没有 apiversion 根，完整示例由显式登记的子树 YAML 组装。

    strategy 页使用自己的 strategy 子树和默认 evaluation；evaluation 页使用默认
    strategy 和自己的 evaluation；TuningPlanConfig 使用两者默认值。
    """
    default_strategy = _tuning_sub_example("StandingHighStrategyConfig")
    default_evaluation = _tuning_sub_example("ServiceOrientedEvaluateServiceConfig")
    if cls.__name__ == _tuning_plan_config().__name__:
        strategy = default_strategy
        ev = default_evaluation
    elif cls in set(_public_strategies()):
        strategy = _tuning_sub_example(cls.__name__)
        ev = default_evaluation
    elif cls in set(_public_evaluations()):
        strategy = default_strategy
        ev = _tuning_sub_example(cls.__name__)
    else:
        return ""
    if not (isinstance(strategy, dict) and isinstance(ev, dict)):
        return ""
    payload = rewrite_example_internal_types(
        {"strategy": strategy, "evaluation": ev}, public_format_type=format_standin
    )
    if not _tuning_example_valid(payload):
        return ""
    return _dump_yaml(payload)


def _complete_example_yaml(
    cls: type,
    record: ModelRecord,
    root_cls: type,
    slug_to_cls: Dict[str, type],
    format_standin: str,
    records_by_slug: Optional[Dict[str, ModelRecord]] = None,
) -> str:
    if _is_tuning_root(root_cls):
        return _complete_tuning_example(cls, format_standin)
    if cls.__name__ == "PracticeConfig":
        wrapped = rewrite_example_internal_types(
            _to_plain(load_practice_config_example()),
            public_format_type=format_standin,
        )
        if _validate_example(cls, wrapped):
            return _dump_yaml(wrapped)
        return ""
    apiversion = _apiversion_of(root_cls)
    if not apiversion:
        return ""
    explicit = load_explicit_examples().get(cls.__name__)
    if explicit is not None:
        wrapped = rewrite_example_internal_types(
            _to_plain(explicit),
            public_format_type=format_standin,
        )
        wrapped["apiversion"] = apiversion
        _inject_required_save(root_cls, wrapped, format_standin)
        if _validate_example(root_cls, wrapped):
            return _dump_yaml(wrapped)
        return ""
    payload = extractable_example_payload(cls)
    if payload is None:
        return ""
    payload = rewrite_example_internal_types(_to_plain(payload), public_format_type=format_standin)
    host_item = _host_item_for(record, cls, slug_to_cls, format_standin, records_by_slug)

    def _wrap(current: Dict[str, Any]) -> Dict[str, Any]:
        if _looks_complete_example(current):
            wrapped = dict(current)
        else:
            wrapped = wrap_example_at_path(
                current,
                record.yaml_path,
                apiversion=apiversion,
                host_item=host_item,
                category=record.category,
            )
        wrapped = rewrite_example_internal_types(_to_plain(wrapped), public_format_type=format_standin)
        wrapped["apiversion"] = apiversion
        _inject_required_save(root_cls, wrapped, format_standin)
        return wrapped

    wrapped = _wrap(payload)
    if not _validate_example(root_cls, wrapped):
        filled = _fill_empty_nested_lists(cls, payload)
        if filled is not payload:
            wrapped = _wrap(filled)
        if not _validate_example(root_cls, wrapped):
            return ""
    return _dump_yaml(wrapped)


def _unique_slugs(models: Sequence[type]) -> Dict[type, str]:
    used: Dict[str, type] = {}
    mapping: Dict[type, str] = {}
    for cls in models:
        qualname = f"{cls.__module__}.{cls.__name__}"
        slug = QUALNAME_SLUGS.get(qualname) or SLUG_OVERRIDES.get(cls.__name__) or slugify_class(cls)
        if slug in used and used[slug] is not cls:
            qualifier = cls.__module__.split(".")[-2]
            slug = f"{qualifier}_{slugify_class(cls)}"
            if slug in used:
                slug = f"{cls.__module__.replace('.', '_')}_{cls.__name__.lower()}"
        used[slug] = cls
        mapping[cls] = slug
    return mapping


def build_records() -> List[ModelRecord]:
    from msmodelslim.processor.base import AutoProcessorConfig

    _import_known_modules()

    models = _collect_seeds()
    processors = _public_processors(AutoProcessorConfig)
    formats = _public_formats()
    slugs = _unique_slugs(models)
    paths, path_roots = _yaml_paths(models)
    records: List[ModelRecord] = []
    pending: List[Tuple[type, ModelRecord]] = []
    format_standin = public_type_tag(formats[0]) if formats else "ascendv1_saver"
    slug_to_cls = {slugs[cls]: cls for cls in models}
    scenarios_by_class = load_scenarios()

    slug_by_class_name: Dict[str, str] = {}
    for cls, slug in slugs.items():
        slug_by_class_name.setdefault(cls.__name__, slug)

    def name_of(nested: Any) -> str:
        if inspect.isclass(nested):
            return slugs.get(nested, nested.__name__)
        return slug_by_class_name.get(str(nested), str(nested))

    for cls in models:
        yaml_path = paths.get(cls, "spec")
        category = _category_for(cls)
        doc = class_doc_sections(cls)
        constraints = list(dict.fromkeys([*doc.constraints, *collect_validator_docs(cls)]))
        notes = list(dict.fromkeys(doc.notes))
        record = ModelRecord(
            qualname=f"{cls.__module__}.{cls.__name__}",
            class_name=cls.__name__,
            slug=slugs[cls],
            title=_title_for(cls),
            summary=_summary_for(cls, yaml_path, category, doc.summary),
            class_summary=doc.summary,
            class_description=list(doc.description),
            source_rel=_source_rel(cls),
            yaml_path=yaml_path or "(根)",
            category=category,
            type_tag=public_type_tag(cls),
            fields=extract_fields(cls, name_of=name_of),
            constraints=constraints,
            nested_refs=_nested_refs(cls, yaml_path, processors, formats, slugs),
            dispatch_bases=_dispatch_bases_for(cls, yaml_path),
            notes=notes,
            scenarios=scenarios_by_class.get(cls.__name__, []),
        )
        records.append(record)
        pending.append((cls, record))

    parents: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for record in records:
        for field_path, nested_slug, relation in record.nested_refs:
            scene = "通过 type 字段分派" if relation == "type 分派" else "作为嵌套对象引用"
            parents[nested_slug].append((record.slug, field_path, scene))
    records_by_slug = {record.slug: record for record in records}
    for cls, record in pending:
        record.parents = parents.get(record.slug, [])
        record.example_yaml = _complete_example_yaml(
            cls,
            record,
            path_roots.get(cls) or _root_configs()[0],
            slug_to_cls,
            format_standin,
            records_by_slug,
        )
    return records


def _has_generated_marker(text: str) -> bool:
    """兼容识别新工具与历史 skill 生成稿，避免旧稿被误判为手写稿。"""
    return GENERATED_MARKER in text or any(marker in text for marker in LEGACY_GENERATED_MARKERS)


def _rel_to_repo(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _is_nested_or_spec(record: "ModelRecord") -> bool:
    """task 页展开规则：spec（服务规格）与嵌套配置都展开进 task 页。"""
    return record.category in ("嵌套配置", "服务规格")


def _spec_merge_links(catalog: Dict[str, "ModelRecord"]) -> Dict[str, Tuple[str, str]]:
    """spec 不再独立成页时，把对 spec 的引用重定向到所属 task 页的 2.2 小节。

    返回 slug → (输出相对路径, #锚点)。
    quant 页块序固定：根块 §2.1，spec 作为首个嵌套块 §2.2。
    """
    links: Dict[str, Tuple[str, str]] = {}
    for record in catalog.values():
        if record.category != "任务配置":
            continue
        for field_path, slug, relation in record.nested_refs:
            target = catalog.get(slug)
            if target is not None and target.category == "服务规格":
                links[slug] = (f"{CATEGORY_DIRS['任务配置']}/{record.slug}.md", f"#{_anchor(f'2.2 {slug}')}")
    return links


def write_docs(
    records: Sequence[ModelRecord],
    output: Path,
    dry_run: bool,
    check: bool,
    catalog: Optional[Dict[str, ModelRecord]] = None,
    expand_nested: bool = False,
    is_expandable: Any = None,
    prune: bool = False,
) -> int:
    """渲染选中的记录到 output 下的分类子目录；catalog 缺省为 records。

    - ``expand_nested``：为 True 时嵌套内部配置展开进上级文档（不再单独成页）。
    - ``prune``：以完整 catalog 的预期页面集为基准，删除 output 下真正过时的生成稿（MANUAL_KEEP
      与手写文档除外）；check 模式下改为报告 stale。
    """
    if catalog is None:
        catalog = {record.slug: record for record in records}
    if not dry_run and not check:
        output.mkdir(parents=True, exist_ok=True)
    drift = 0
    written = 0
    # 文档按类型放入 quant/spec/processor/format/tuning/nested 子目录，源码链接深度 5。
    # quant 页默认把对应 spec 与嵌套配置一并展开，不再单独生成 spec 页。
    link_overrides = _spec_merge_links(catalog) if expand_nested else {}
    targets: Dict[Path, str] = {}
    for record in records:
        record_is_expandable = is_expandable
        if record_is_expandable is None and expand_nested and record.category == "任务配置":
            record_is_expandable = _is_nested_or_spec
        targets[output / _doc_path(record)] = render_markdown(
            record,
            catalog,
            expand_nested=expand_nested,
            is_expandable=record_is_expandable,
            link_depth=5,
            link_overrides=link_overrides,
        )

    # prune 的过期判断必须以完整 catalog 的预期页面集为基准。局部 --targets
    # 只决定本次重写范围，不能把其他仍有效的生成页误判为 stale。
    protected_records = list(catalog.values())
    if expand_nested:
        protected_records = [record for record in protected_records if not _is_nested_or_spec(record)]
    protected_paths = {output / _doc_path(record) for record in protected_records}

    for path, content in targets.items():
        existing = path.read_text(encoding="utf-8") if path.exists() else None
        if existing is not None and not _has_generated_marker(existing) and path.name in MANUAL_KEEP:
            continue
        if check:
            if existing != content:
                print(f"drift: {_rel_to_repo(path)}")
                drift += 1
            continue
        if dry_run:
            print(f"would write {_rel_to_repo(path)}")
            written += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written += 1
        print(f"wrote {_rel_to_repo(path)}")

    if prune:
        # 递归扫描子目录；保留手写文档与非生成稿，其余过时生成稿删除。
        for path in sorted(output.rglob("*.md")):
            if path.name in MANUAL_KEEP or path in protected_paths:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if not _has_generated_marker(text):
                continue
            if check:
                print(f"stale: {_rel_to_repo(path)}")
                drift += 1
                continue
            if dry_run:
                print(f"would prune {_rel_to_repo(path)}")
                continue
            path.unlink()
            print(f"pruned {_rel_to_repo(path)}")
    print(f"{'checked' if check else 'generated'} {len(targets)} files")
    return drift


class UnknownTargetError(ValueError):
    """Raised when one or more --targets selectors match no discovered config."""


def select_records(records: Sequence[ModelRecord], selectors: Optional[Sequence[str]]) -> List[ModelRecord]:
    """按 --targets 选择记录：支持分类关键字、分类名、类名或 slug。

    ``all``（或空）表示全部记录；多个选择器取并集，结果保持 records 顺序。
    """
    if not selectors:
        return list(records)
    cleaned = list(dict.fromkeys(s.strip() for s in selectors if s and s.strip()))
    if not cleaned:
        return list(records)
    unknown = [
        selector
        for selector in cleaned
        if selector.lower() != "all" and not any(_match_target(record, selector) for record in records)
    ]
    if unknown:
        raise UnknownTargetError(f"未识别的 --targets 选择器: {', '.join(unknown)}")
    if any(selector.lower() == "all" for selector in cleaned):
        return list(records)
    selected: List[ModelRecord] = []
    for record in records:
        if any(_match_target(record, s) for s in cleaned):
            selected.append(record)
    return selected


def _match_target(record: ModelRecord, selector: str) -> bool:
    low = selector.lower()
    if low in TARGET_GROUP_KEYWORDS:
        return record.category == TARGET_GROUP_KEYWORDS[low]
    return (
        record.slug == selector
        or record.slug == low
        or record.class_name == selector
        or record.class_name == low
        or record.category == selector
        or record.category == low
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基础抽取脚本：从 Pydantic 注解按指定目标生成量化配置文档")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs/zh/api_reference/config",
        help="文档输出目录",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help=(
            "抽取目标（可多个）：分类关键字（task/root/spec/service/processor/format/save/nested/all）、"
            "分类名（任务配置/服务规格/处理器/保存格式/嵌套配置）、类名或 slug；默认全部"
        ),
    )
    parser.add_argument(
        "--expand-nested",
        dest="expand_nested",
        action="store_true",
        help="将嵌套内部配置展开进上级文档（不再单独成页）",
    )
    parser.add_argument(
        "--no-expand-nested",
        dest="expand_nested",
        action="store_false",
        help="不展开嵌套配置，每个配置类独立成页",
    )
    parser.set_defaults(expand_nested=False)
    parser.add_argument(
        "--prune",
        dest="prune",
        action="store_true",
        help="按完整配置目录删除 output 下不再生成的过时文档",
    )
    parser.add_argument(
        "--no-prune",
        dest="prune",
        action="store_false",
        help="保留历史遗留文档，仅覆盖本次目标文件",
    )
    parser.set_defaults(prune=False)
    parser.add_argument("--dry-run", action="store_true", help="只打印将写入/删除的文件，不落盘")
    parser.add_argument("--check", action="store_true", help="检查已生成文档是否与源码注解一致")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        records = build_records()
    except ConfigModuleImportError as exc:
        print(f"配置文档生成失败：{exc}", file=sys.stderr)
        return 2
    catalog = {record.slug: record for record in records}
    try:
        selected = select_records(records, args.targets)
    except UnknownTargetError as exc:
        print(f"配置文档生成失败：{exc}", file=sys.stderr)
        return 2
    code = write_docs(
        selected,
        args.output,
        args.dry_run,
        args.check,
        catalog=catalog,
        expand_nested=args.expand_nested,
        prune=args.prune,
    )
    if args.check and code:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
