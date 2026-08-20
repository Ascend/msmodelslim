#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""基础抽取脚本：从 Pydantic 配置模型注解按指定目标生成量化配置接口文档。

提供基础能力（--targets 指定特定对象、--expand-nested 控制是否展开嵌套）；
具体抽取目标与后处理由驱动脚本 gen_config_api_docs.py 控制（见其模块 docstring）。
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import enum
import importlib
import inspect
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, get_origin

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel  # noqa: E402

from quant_config_docgen import (  # noqa: E402
    GENERATED_MARKER,
    ModelRecord,
    _anchor,
    _doc_path,
    annotation_refers_to,
    class_summary,
    collect_nested_models,
    collect_validator_docs,
    extract_declared_example,
    extract_fields,
    extractable_example_payload,
    is_internal_type_tag,
    is_pydantic_model,
    public_type_tag,
    raw_type_tag,
    render_markdown,
    rewrite_example_internal_types,
    slugify_class,
    wrap_example_at_path,
    _literal_values,
)

MANUAL_KEEP = {"processor_group.md", "auto_precision_tuning.md"}
NAV_BEGIN = "# BEGIN GENERATED QUANT CONFIG NAV"
NAV_END = "# END GENERATED QUANT CONFIG NAV"

SLUG_OVERRIDES = {
    "ModelslimV1QuantConfig": "modelslim_v1",
    "ModelslimV1ServiceConfig": "modelslim_v1_spec",
    "MultimodalVLMModelslimV1QuantConfig": "multimodal_vlm_modelslim_v1",
    "MultimodalVLMServiceConfig": "multimodal_vlm_modelslim_v1_spec",
    "MultimodalSDModelslimV1QuantConfig": "multimodal_sd_modelslim_v1",
    "MultimodalSDServiceConfig": "multimodal_sd_modelslim_v1_spec",
    "ModelslimConvertQuantConfig": "modelslim_convert",
    "ModelslimConvertServiceConfig": "modelslim_convert_spec",
    "AdaptRotationStage1ProcessorConfig": "adapt_rotation_stage1",
    "AdaptRotationStage2ProcessorConfig": "adapt_rotation_stage2",
    "TuningPlanConfig": "tuning_plan",
    "StandingHighStrategyConfig": "strategy_standing_high",
    "StandingHighWithExperienceStrategyConfig": "strategy_standing_high_with_experience",
    "BinaryFallbackStrategyConfig": "strategy_binary_fallback",
    "ServiceOrientedEvaluateServiceConfig": "evaluation_service_oriented",
}

QUALNAME_SLUGS = {
    "msmodelslim.processor.quant.autoround.QuantStrategyConfig": "autoround_quant_strategy_config",
    "msmodelslim.processor.flat_quant.flat_quant.QuantStrategyConfig": "flatquant_quant_strategy_config",
    "msmodelslim.processor.trainable_linear_quant.config.processor_config.QuantStrategyConfig": "tlq_quant_strategy_config",
}

APIVERSION_BY_ROOT = {
    "ModelslimV1QuantConfig": "modelslim_v1",
    "MultimodalVLMModelslimV1QuantConfig": "multimodal_vlm_modelslim_v1",
    "MultimodalSDModelslimV1QuantConfig": "multimodal_sd_modelslim_v1",
    "ModelslimConvertQuantConfig": "modelslim_convert",
}


# 自动拼装无法得到完整示例的配置类，提供手写、可解析的完整 YAML 参考。
# 键为类名；值为完整任务配置（apiversion + spec），须能被对应根配置 model_validate 通过。
EXPLICIT_EXAMPLES = {
    "RenamePattern": {
        "apiversion": "modelslim_convert",
        "spec": {
            "preprocess": [
                {
                    "type": "rename",
                    "patterns": [
                        {"from": "model.layers.*.mlp.gate_proj.weight", "to": "model.layers.*.mlp.gate.weight"},
                    ],
                },
            ],
        },
    },
    "ConvertOpConfig": {
        "apiversion": "modelslim_convert",
        "spec": {
            "preprocess": [
                {
                    "type": "convert",
                    "source": ["model.layers.*.mlp.gate_up_proj.weight"],
                    "target": ["model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"],
                    "ops": [
                        {"type": "chunk", "dim": 1, "projections": ["gate_proj", "up_proj"]},
                    ],
                },
            ],
        },
    },
    "GroupProcessorConfig": {
        "apiversion": "modelslim_v1",
        "spec": {
            "process": [
                {
                    "type": "group",
                    "configs": [
                        {
                            "type": "linear_quant",
                            "qconfig": {
                                "act": {"dtype": "float", "scope": "per_tensor", "symmetric": True, "method": "none"},
                                "weight": {
                                    "dtype": "int8",
                                    "scope": "per_channel",
                                    "symmetric": True,
                                    "method": "minmax",
                                },
                            },
                        },
                        {"type": "smooth_quant", "alpha": 0.5, "symmetric": True},
                    ],
                },
            ],
        },
    },
    "TLQOpConfig": {
        "apiversion": "modelslim_v1",
        "spec": {
            "process": [
                {
                    "type": "trainable_linear_quant",
                    "operations": [
                        {"type": "minmax_tune", "lr": 0.01},
                        {"type": "round_tune", "lr": 0.005},
                    ],
                    "strategies": [
                        {
                            "qconfig": {
                                "act": {"dtype": "float", "scope": "per_tensor", "symmetric": True, "method": "none"},
                                "weight": {
                                    "dtype": "int4",
                                    "scope": "per_channel",
                                    "symmetric": True,
                                    "method": "minmax",
                                },
                            },
                            "include": ["*"],
                            "exclude": [],
                        },
                    ],
                },
            ],
        },
    },
    "QuantSaveProcessorConfig": {
        "apiversion": "modelslim_v1",
        "spec": {
            "process": [
                {
                    "type": "saver",
                    "format": {"type": "_auto_save"},
                },
            ],
        },
    },
}

SKIP_CLASS_NAMES = {
    "BaseModel",
    "BaseQuantConfig",
    "AutoProcessorConfig",
    "QuantFormatConfig",
    "AutoSaverBaseConfig",
    "TypedConfig",
    "QuantServiceConfig",
}


def _ensure_package_config_link() -> Optional[Path]:
    """开发树中 config.ini 在仓库根目录，包内相对路径期望 msmodelslim/config。"""
    pkg_config = REPO_ROOT / "msmodelslim" / "config"
    repo_config = REPO_ROOT / "config" / "config.ini"
    if pkg_config.exists() or not repo_config.exists():
        return None
    pkg_config.symlink_to(repo_config.parent.resolve(), target_is_directory=True)
    return pkg_config


def _import_known_modules() -> None:
    modules = [
        "msmodelslim.processor",
        "msmodelslim.processor.kv_smooth.processor",
        "msmodelslim.processor.quarot",
        "msmodelslim.format.registry",
        "msmodelslim.core.quant_service.modelslim_v1.quant_config",
        "msmodelslim.core.quant_service.multimodal_vlm_v1.quant_config",
        "msmodelslim.core.quant_service.multimodal_sd_v1.quant_config",
        "msmodelslim.core.quant_service.modelslim_convert.quant_config",
        "msmodelslim.core.quant_service.modelslim_convert.config_mapper",
        # 自动调优配置
        "msmodelslim.app.auto_tuning.plan_manager_infra",
        "msmodelslim.app.auto_tuning.evaluation_service_infra",
        "msmodelslim.core.tune_strategy.interface",
        "msmodelslim.core.tune_strategy.standing_high.strategy",
        "msmodelslim.core.tune_strategy.standing_high_with_experience.strategy",
        "msmodelslim.core.tune_strategy.binary_fallback.strategy",
        "msmodelslim.infra.service_oriented_evaluate_service",
        "msmodelslim.infra.evaluation.aisbench_server",
        "msmodelslim.infra.vllm_ascend_server",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            print(f"skip import {name}: {exc}", file=sys.stderr)


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
    from msmodelslim.core.quant_service.interface import BaseQuantConfig
    from msmodelslim.format.base import QuantFormatConfig
    from msmodelslim.processor.base import AutoProcessorConfig

    # PracticeConfig 是 BaseQuantConfig 子类（任务配置），作为任务基类配置单独成页，
    # 用于展示 apiversion + spec + metadata（Metadata）的完整实践配置结构。
    if cls in _auto_tuning_targets():
        return "自动调优"
    if issubclass(cls, BaseQuantConfig):
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
    for cls in models:
        if cls.__name__ in path_overrides:
            paths[cls] = path_overrides[cls.__name__]
            if cls.__name__ in tuning_names:
                roots[cls] = tuning_root
            else:
                roots[cls] = convert_root if cls.__name__ in convert_names else _root_configs()[0]
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
    from quant_config_docgen import split_optional, unwrap_annotation

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


def _summary_for(cls: type, yaml_path: str, category: str) -> str:
    """配置概述：只用类 docstring，不再拼接「该配置位于 …」这类位置句。

    YAML 位置由页面标题/展开小节标题给出，概述保持简洁。
    """
    doc = class_summary(cls)
    if doc:
        return doc
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


def _tuning_sub_example(getter) -> Optional[Dict[str, Any]]:
    """取策略/评估类里声明的全量子树示例（json_schema_extra.examples）。"""
    for cls in getter():
        declared = extract_declared_example(cls)
        if isinstance(declared, dict):
            return _to_plain(declared)
    return None


def _complete_tuning_example(cls: type, format_standin: str) -> str:
    """自动调优配置没有 apiversion 根，完整示例由类内声明的子树示例组装：
    页面自身类取自己的全量子树，兄弟类型取默认子树（strategy 页配默认 evaluation，
    evaluation 页配默认 strategy，TuningPlanConfig 取两者默认）。
    """
    declared = extract_declared_example(cls)
    if isinstance(cls, type) and _tuning_plan_config().__name__ == cls.__name__:
        strategy, ev = _tuning_sub_example(_public_strategies), _tuning_sub_example(_public_evaluations)
    elif cls in set(_public_strategies()):
        strategy = _to_plain(declared) if isinstance(declared, dict) else None
        ev = _tuning_sub_example(_public_evaluations)
    elif cls in set(_public_evaluations()):
        strategy = _tuning_sub_example(_public_strategies)
        ev = _to_plain(declared) if isinstance(declared, dict) else None
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
    apiversion = _apiversion_of(root_cls)
    if not apiversion:
        return ""
    explicit = EXPLICIT_EXAMPLES.get(cls.__name__)
    if explicit is not None:
        wrapped = dict(explicit)
        wrapped["apiversion"] = apiversion
        _inject_required_save(root_cls, wrapped, format_standin)
        if _validate_example(root_cls, wrapped):
            return _dump_yaml(wrapped)
        return ""
    declared = extract_declared_example(cls)
    if isinstance(declared, str):
        declared = None
    payload = declared if isinstance(declared, dict) else extractable_example_payload(cls)
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
        record = ModelRecord(
            qualname=f"{cls.__module__}.{cls.__name__}",
            class_name=cls.__name__,
            slug=slugs[cls],
            title=_title_for(cls),
            summary=_summary_for(cls, yaml_path, category),
            class_summary=class_summary(cls),
            source_rel=_source_rel(cls),
            yaml_path=yaml_path or "(根)",
            category=category,
            type_tag=public_type_tag(cls),
            fields=extract_fields(cls, name_of=name_of),
            constraints=collect_validator_docs(cls) or [],
            nested_refs=_nested_refs(cls, yaml_path, processors, formats, slugs),
            dispatch_bases=_dispatch_bases_for(cls, yaml_path),
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


def _nav_entries(records: Sequence[ModelRecord]) -> List[str]:
    groups: Dict[str, List[ModelRecord]] = defaultdict(list)
    for record in records:
        groups[record.category].append(record)
    order = ["任务配置", "服务规格", "处理器", "保存格式", "自动调优", "嵌套配置"]
    lines: List[str] = []
    for category in order:
        items = sorted(groups.get(category, []), key=lambda r: r.slug)
        if not items:
            continue
        lines.append(f"          - {category}:")
        for record in items:
            lines.append(f'              - {record.slug}: "zh/api_reference/config/{_doc_path(record)}"')
    lines.append('          - 组合处理器使用说明: "zh/api_reference/config/processor_group.md"')
    lines.append('          - 自动调优配置: "zh/api_reference/config/auto_precision_tuning.md"')
    return lines


def update_mkdocs(mkdocs_path: Path, records: Sequence[ModelRecord], dry_run: bool) -> None:
    text = mkdocs_path.read_text(encoding="utf-8")
    nav_block = "\n".join([f"          {NAV_BEGIN}"] + _nav_entries(records) + [f"          {NAV_END}"])
    if NAV_BEGIN in text and NAV_END in text:
        start = text.index(NAV_BEGIN)
        end = text.index(NAV_END) + len(NAV_END)
        line_start = text.rfind("\n", 0, start) + 1
        updated = text[:line_start] + nav_block + text[end:]
    else:
        needle = '      - 量化配置:\n'
        if needle not in text:
            print("mkdocs.yml 未找到量化配置导航，已跳过更新", file=sys.stderr)
            return
        insert_at = text.index(needle) + len(needle)
        # 删除旧的手工条目，直到传统 V0 段
        rest = text[insert_at:]
        next_section = rest.find("      - 传统 V0 Python API:")
        if next_section == -1:
            print("mkdocs.yml 未找到传统 V0 导航，已跳过更新", file=sys.stderr)
            return
        updated = text[:insert_at] + nav_block + "\n" + rest[next_section:]
    if dry_run:
        return
    mkdocs_path.write_text(updated, encoding="utf-8")


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
    task 页块序固定：根块 §2.1，spec 作为首个嵌套块 §2.2。
    """
    links: Dict[str, Tuple[str, str]] = {}
    for record in catalog.values():
        if record.category != "任务配置":
            continue
        for field_path, slug, relation in record.nested_refs:
            target = catalog.get(slug)
            if target is not None and target.category == "服务规格":
                links[slug] = (f"task/{record.slug}.md", f"#{_anchor(f'2.2 {slug}')}")
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
    - ``prune``：删除 output 下已生成但不在本次目标集中的过时文档（MANUAL_KEEP
      与手写文档除外）；check 模式下改为报告 stale。
    """
    if catalog is None:
        catalog = {record.slug: record for record in records}
    output.mkdir(parents=True, exist_ok=True)
    drift = 0
    written = 0
    # 文档按类型放入 task/spec/processor/format/tuning/nested 子目录，源码链接深度 5。
    # task 页默认把对应 spec 与嵌套配置一并展开，不再单独生成 spec 页。
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

    for path, content in targets.items():
        existing = path.read_text(encoding="utf-8") if path.exists() else None
        if existing is not None and GENERATED_MARKER not in existing and path.name in MANUAL_KEEP:
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
            if path.name in MANUAL_KEEP or path in targets:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if GENERATED_MARKER not in text:
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


# 目标选择关键字 → 配置分类（--targets 基础能力）
TARGET_GROUP_KEYWORDS = {
    "task": "任务配置",
    "root": "任务配置",
    "spec": "服务规格",
    "service": "服务规格",
    "processor": "处理器",
    "format": "保存格式",
    "save": "保存格式",
    "tuning": "自动调优",
    "auto_tuning": "自动调优",
    "nested": "嵌套配置",
}


def select_records(records: Sequence[ModelRecord], selectors: Optional[Sequence[str]]) -> List[ModelRecord]:
    """按 --targets 选择记录：支持分类关键字、分类名、类名或 slug。

    ``all``（或空）表示全部记录；多个选择器取并集，结果保持 records 顺序。
    """
    if not selectors:
        return list(records)
    cleaned = [s.strip() for s in selectors if s and s.strip()]
    if not cleaned or any(s.lower() == "all" for s in cleaned):
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
        action="store_true",
        help="删除 output 下已生成但不在本次目标集中的过时文档",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将写入/删除的文件，不落盘")
    parser.add_argument("--check", action="store_true", help="检查已生成文档是否与源码注解一致")
    parser.add_argument("--update-mkdocs", action="store_true", help="写入 mkdocs.yml 导航（默认不改）")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    created_link = _ensure_package_config_link()
    try:
        records = build_records()
        catalog = {record.slug: record for record in records}
        selected = select_records(records, args.targets)
        code = write_docs(
            selected,
            args.output,
            args.dry_run,
            args.check,
            catalog=catalog,
            expand_nested=args.expand_nested,
            prune=args.prune,
        )
        if args.update_mkdocs and not args.check:
            update_mkdocs(REPO_ROOT / "mkdocs.yml", selected, args.dry_run)
        if args.check and code:
            return 1
        return 0
    finally:
        if created_link is not None and created_link.is_symlink():
            created_link.unlink()


if __name__ == "__main__":
    sys.exit(main())
