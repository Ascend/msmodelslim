#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""量化配置接口文档驱动脚本：控制具体抽取目标与后处理（模板 05 架构结论）。

基础抽取能力在 gen_quant_config_docs.py（--targets / --expand-nested），
本脚本按《接口文档架构结论》确定目标与后处理：

1. 目标（结论第 6 点）：
   - 任务配置：4 个根配置页面，每个页面展开对应的 spec 结构（task 与 spec 合并，
     不再单独生成 spec/ 页面）；
   - 处理器：processor 配置结构（全部公开处理器）；
   - 自动调优：TuningPlanConfig + 具体策略 + 评估服务（5 个自动调优配置页）；
     auto_precision_tuning.md 为人工维护的总览文档，保持不变；
   - 保存格式：保留独立页面（spec.save[] 引用过去，便于按格式检索）。
2. 展开策略（结论第 5 点）：嵌套内部配置类默认展开进上级文档的
   「嵌套配置明细」小节，不再单独成页，减少跳转。
3. 模板（结论第 7 点）：「引用的配置 / 被引用的配置」两个小节已移除，
   引用信息收敛到参数列表的「引用配置」列（页内锚点或独立页面链接）。
4. 后处理：按类型子目录（task/spec/processor/format/tuning）组织输出、清理历史遗留的
   嵌套配置文档；不再生成 config/README.md 索引，也不再默认改动 mkdocs.yml。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from gen_quant_config_docs import (  # noqa: E402
    _ensure_package_config_link,
    build_records,
    select_records,
    update_mkdocs,
    write_docs,
)

# 结论第 6 点对应的默认抽取目标（分类关键字，见 gen_quant_config_docs.select_records）。
# 注意：spec 不再独立成页，而是随 task 页展开，因此默认目标不包含 spec。
DEFAULT_TARGETS = ["root", "processor", "format", "tuning"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="驱动脚本：按接口文档架构结论生成量化配置接口文档（含目标控制与后处理）"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs/zh/api_reference/config",
        help="文档输出目录",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help=f"覆盖默认抽取目标；默认 {DEFAULT_TARGETS}",
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
        help="不展开嵌套配置：此时嵌套配置不再有独立页面，字段级引用可能指向未生成页面；"
        "如需独立页面请用基础脚本 gen_quant_config_docs.py 生成 nested/all 目标",
    )
    parser.set_defaults(expand_nested=True)
    parser.add_argument(
        "--prune",
        dest="prune",
        action="store_true",
        help="清理历史遗留、不再生成的文档（默认）",
    )
    parser.add_argument(
        "--no-prune",
        dest="prune",
        action="store_false",
        help="保留历史遗留文档，仅覆盖生成目标内文件",
    )
    parser.set_defaults(prune=True)
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
