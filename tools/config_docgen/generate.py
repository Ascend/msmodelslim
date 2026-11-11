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

项目级量化配置文档生成入口。

从 Pydantic 配置类、`Field` 注解和 validator docstring 抽取内容，生成
`docs/zh/api_reference/config/` 下的配置文档。`--check` 同时检查注解完整性
与仓库生成稿漂移，由 UT 流水线执行一致性门禁。
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    from tools.config_docgen.discovery import (
        ConfigModuleImportError,
        UnknownTargetError,
        build_records,
        select_records,
        write_docs,
    )
    from tools.config_docgen.examples import ExampleLoadError
    from tools.config_docgen.schema import is_internal_validator, iter_field_after_validators, validator_constraint_text
else:
    from .discovery import ConfigModuleImportError, UnknownTargetError, build_records, select_records, write_docs
    from .examples import ExampleLoadError
    from .schema import is_internal_validator, iter_field_after_validators, validator_constraint_text

DEFAULT_TARGETS = ["root", "processor", "format", "tuning"]
REGENERATE_COMMAND = "python -m tools.config_docgen"
REPO_CONFIG_DIR = REPO_ROOT / "config"
PACKAGE_CONFIG_DIR = REPO_ROOT / "msmodelslim" / "config"


@contextmanager
def _repository_config() -> Iterator[None]:
    """为干净 checkout 临时准备 `msmodelslim/config`。

    生产代码从包内 `config/config.ini` 读取插件信息，而 `.gitignore` 不跟踪该目录。
    仓库根 `config/` 是受版本管理的唯一配置来源；仅在包内目录缺失时复制，并在使用
    结束后删除生成器自己创建的副本，避免污染工作区。
    """

    source_config = REPO_CONFIG_DIR / "config.ini"
    if not source_config.is_file():
        raise FileNotFoundError(f"仓库配置缺少 config.ini: {source_config}")
    if PACKAGE_CONFIG_DIR.exists() and not PACKAGE_CONFIG_DIR.is_dir():
        raise NotADirectoryError(f"包内配置路径不是目录: {PACKAGE_CONFIG_DIR}")

    copied = False
    if not PACKAGE_CONFIG_DIR.exists():
        PACKAGE_CONFIG_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPO_CONFIG_DIR, PACKAGE_CONFIG_DIR)
        copied = True
    try:
        target_config = PACKAGE_CONFIG_DIR / "config.ini"
        if not target_config.is_file():
            raise FileNotFoundError(f"包内配置缺少 config.ini: {target_config}")
        yield
    finally:
        if copied:
            shutil.rmtree(PACKAGE_CONFIG_DIR, ignore_errors=True)


@contextmanager
def _windows_path_compat() -> Iterator[None]:
    """临时兼容 Windows 绝对路径。

    生产代码的安全路径检查只接受 POSIX 风格白名单字符，Windows 盘符和反斜杠
    会导致导入 `msmodelslim` 时失败。仅在生成器进程内放行绝对路径，不创建软链接，
    也不改变 Linux 行为。
    """

    if os.name != "nt":
        yield
        return

    from msmodelslim.utils.security import path as security_path

    original = security_path.get_valid_path

    def get_valid_path(path, extensions=None):
        if isinstance(path, str) and os.path.isabs(path):
            return os.path.realpath(path)
        return original(path, extensions)

    security_path.get_valid_path = get_valid_path
    try:
        yield
    finally:
        security_path.get_valid_path = original


def _source_field_info(model_cls: type, output_name: str):
    """按文档字段名（包含 alias）定位源码 FieldInfo。"""
    model_fields = getattr(model_cls, "model_fields", {}) or {}
    info = model_fields.get(output_name)
    if info is not None:
        return info
    for candidate in model_fields.values():
        if output_name in {candidate.alias, candidate.serialization_alias}:
            return candidate
    return None


def audit_records(records: Sequence[object]) -> List[str]:
    """检查所有配置记录是否具备文档所需的最小源码注解。"""

    problems: List[str] = []
    for record in records:
        qualname = getattr(record, "qualname")
        if not str(getattr(record, "class_summary", "")).strip():
            problems.append(f"{qualname}: 缺少配置类概述 docstring")
        module_name, class_name = qualname.rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module_name), class_name)
        for config_field in getattr(record, "fields"):
            field_name = getattr(config_field, "name")
            description = str(getattr(config_field, "description", "")).strip()
            if not description or description == "—":
                problems.append(f"{qualname}.{field_name}: 缺少字段 description")
                continue
            info = _source_field_info(model_cls, field_name)
            if info is None or not str(getattr(info, "description", "") or "").strip():
                problems.append(f"{qualname}.{field_name}: 源码 Field 缺少 description 注解")
                continue
            for metadata, _ in iter_field_after_validators(info):
                func = metadata.func
                if is_internal_validator(func):
                    continue
                if not validator_constraint_text(func):
                    func_name = getattr(func, "__name__", repr(func))
                    problems.append(
                        f"{qualname}.{field_name}: AfterValidator `{func_name}` 缺少约束说明（config_constraint）"
                    )
    return problems


def audit_validator_docs(records: Sequence[object]) -> List[str]:
    """检查配置类自身声明的 Pydantic validator 是否都有约束说明。"""

    problems: List[str] = []
    for record in records:
        qualname = getattr(record, "qualname")
        module_name, class_name = qualname.rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module_name), class_name)
        decorators = getattr(model_cls, "__pydantic_decorators__", None)
        if decorators is None:
            continue
        for group_name in ("field_validators", "model_validators"):
            group = getattr(decorators, group_name, {}) or {}
            for validator_name, wrapped in group.items():
                func = getattr(wrapped, "func", wrapped)
                if is_internal_validator(func):
                    continue
                owner = getattr(func, "__qualname__", "").split(".", 1)[0]
                if owner != class_name:
                    continue
                if not (inspect.getdoc(func) or "").strip():
                    problems.append(f"{qualname}.{validator_name}: validator 缺少约束 docstring")
    return problems


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从源码注解生成量化配置文档")
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
        help=f"覆盖默认抽取目标；未指定时使用 {DEFAULT_TARGETS}",
    )
    parser.add_argument(
        "--expand-nested",
        dest="expand_nested",
        action="store_true",
        help="将嵌套内部配置展开进上级文档（默认）",
    )
    parser.add_argument(
        "--no-expand-nested",
        dest="expand_nested",
        action="store_false",
        help="不展开嵌套配置",
    )
    parser.set_defaults(expand_nested=True)
    parser.add_argument(
        "--prune",
        dest="prune",
        action="store_true",
        help="按完整配置目录清理不再生成的文档",
    )
    parser.add_argument(
        "--no-prune",
        dest="prune",
        action="store_false",
        help="保留历史遗留文档，仅覆盖生成目标内文件",
    )
    parser.set_defaults(prune=None)
    parser.add_argument("--dry-run", action="store_true", help="只打印将写入/删除的文件，不落盘")
    parser.add_argument("--check", action="store_true", help="检查注解完整性与生成稿一致性")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    with _repository_config(), _windows_path_compat():
        try:
            records = build_records()
        except (ConfigModuleImportError, ExampleLoadError) as exc:
            print(f"配置文档生成失败：{exc}", file=sys.stderr)
            return 2
        if not records:
            print("配置文档生成失败：未发现任何配置类", file=sys.stderr)
            return 2
        problems = audit_records(records) + audit_validator_docs(records)
        if problems:
            print("配置注解审计失败：", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

        catalog = {record.slug: record for record in records}
        selectors = args.targets if args.targets is not None else DEFAULT_TARGETS
        prune = args.prune if args.prune is not None else args.targets is None
        try:
            selected = select_records(records, selectors)
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
            prune=prune,
        )
        if args.check and code:
            print(
                "配置文档与源码注解不一致。请参考 `tools/config_docgen/README.md` 设计配置说明，"
                f"并运行 `{REGENERATE_COMMAND}` 重新生成后提交生成稿。",
                file=sys.stderr,
            )
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
