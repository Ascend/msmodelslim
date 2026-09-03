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

# pylint: disable=too-many-lines

import argparse
import datetime
import subprocess  # nosec B404
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple

from msmodelslim.cli.logo import print_logo
from msmodelslim.core.const import DeviceType, QuantType
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.logging import get_logger, set_logger_level

FAQ_URL = "https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/support/faq.md"
MIND_STUDIO_LOGO = "[Powered by MindStudio]"

_BOOL_TRUE_VALUES = frozenset(('true', 'yes', 'on'))
_BOOL_FALSE_VALUES = frozenset(('false', 'no', 'off'))


def _cli_convert_to_bool(value: str) -> bool:
    """
    CLI-only bool converter for --trust_remote_code.

    Accepts the canonical ``true`` / ``false`` spellings (per the unified CLI
    spec) plus legacy ``True``/``False``, ``yes``/``no`` and ``on``/``off``.
    Raises ValueError so argparse produces a clean error instead of a traceback.
    """
    lowered = value.strip().lower()
    if lowered in _BOOL_TRUE_VALUES:
        return True
    if lowered in _BOOL_FALSE_VALUES:
        return False
    raise ValueError(f"value must be true/false (legacy True/False, yes/no, on/off are also accepted), got {value!r}")


# ---------------------------------------------------------------------------
# Deprecated aliases: legacy option spellings that actually existed in the
# original branch -> the canonical snake_case name. Only real legacy spellings
# are listed (no invented variants); using one triggers a one-time deprecation
# warning and they never appear in --help output.
# ---------------------------------------------------------------------------
DEPRECATED_ALIASES = {
    '--tag': '--tags',
    '--pattern': '--patterns',
    '--topk': '--top_k',
    '--calib_dataset': '--calibration_dataset',
    '--config_path': '--config',
}


def _repo_root() -> Path:
    """Return the repository root (directory containing .git or setup.py)."""
    cur = Path(__file__).resolve().parent
    for parent in cur.parents:
        if (parent / '.git').exists() or (parent / 'setup.py').exists():
            return parent
    return cur


def _load_build_info() -> Tuple[str, str]:
    """Return ``(git_hash, build_date)`` written at pack time, if present."""
    try:
        from msmodelslim._build_info import BUILD_DATE, GIT_HASH  # pylint: disable=no-name-in-module

        return (GIT_HASH or '').strip(), (BUILD_DATE or '').strip()
    except Exception:
        return '', ''


def _get_version() -> str:
    """Return the installed package version, falling back to setup.py then 'unknown'."""
    try:
        from importlib.metadata import version as _pkg_version  # type: ignore

        pkg_version = _pkg_version('msmodelslim')
    except Exception:
        pkg_version = ''
    if pkg_version:
        return pkg_version
    import re as _re

    setup_file = _repo_root() / 'setup.py'
    if setup_file.exists():
        try:
            match = _re.search(r'^\s*__version__\s*=\s*[\'"]([^\'"]+)[\'"]', setup_file.read_text(), _re.MULTILINE)
        except Exception:
            match = None
        if match:
            return match.group(1)
    return 'unknown'


def _get_git_hash() -> str:
    """Return a short git commit hash from the wheel, else from the source checkout."""
    baked_hash, _ = _load_build_info()
    if baked_hash:
        return baked_hash[:12]
    try:
        out = subprocess.check_output(  # nosec B603, B607
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(_repo_root()),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()[:12]
    except Exception:
        return ''


def _get_build_date() -> str:
    """Return the pack-time UTC date, else the package directory mtime."""
    _, baked_date = _load_build_info()
    if baked_date:
        return baked_date
    module_dir = Path(__file__).resolve().parents[1]
    try:
        ts = module_dir.stat().st_mtime
    except OSError:
        return ''
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _print_version() -> None:
    """Print the unified version banner (MindStudio CLI spec section 4.5)."""
    version = _get_version()
    git_hash = _get_git_hash() or 'unknown'
    build_date = _get_build_date()
    print_logo()
    lines = [
        f"msmodelslim {version} ({git_hash})\n",
        "Copyright (C) 2026 Huawei Technologies Co., Ltd.\n",
        "License: Mulan PSL v2.\n",
        "\n",
        "Build Info:\n",
    ]
    if build_date:
        lines.append(f"  Date: {build_date}\n")
    lines.append("  Repo: https://gitcode.com/Ascend/msmodelslim\n")
    sys.stdout.write(''.join(lines))


class _UnifiedHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """
    Help formatter implementing the unified CLI help layout (spec section 4.4):
      * fixed sections: Usage / Description / Required arguments /
        Optional arguments / Examples / Output;
      * shows the single long name and any single-char short options;
      * embeds enum choices directly in the signature as ``{a,b,c}``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._usage_args = None
        self._usage_prefix = None
        self._description_text = None
        self._epilog_text = None

    def add_usage(self, usage, actions, groups, prefix=None):
        self._usage_args = (usage, actions, groups)
        self._usage_prefix = prefix
        super().add_usage(usage, actions, groups, prefix)

    def add_text(self, text):
        if text is None:
            return
        if self._description_text is None:
            self._description_text = text
        else:
            self._epilog_text = text
        super().add_text(text)

    def _metavar_formatter(self, action, default_metavar):
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            result = '{%s}' % ','.join(str(getattr(c, 'value', c)) for c in action.choices)
        else:
            result = default_metavar

        def _format(tuple_size):
            return result if isinstance(result, tuple) else (result,) * tuple_size

        return _format

    def _format_args(self, action, default_metavar):
        """
        Render the value placeholder of an option per spec 4.4.2.2.

        * nargs='*' (space-separated multi-value) -> ``[<NAME> ...]``;
        * nargs='?' (optional single value, e.g. --trust_remote_code) ->
          single ``<NAME>`` without brackets;
        * otherwise delegate to argparse's default rendering.
        """
        if action.nargs == '*':
            name = self._single_metavar(action, default_metavar)
            return '[%s ...]' % name
        if action.nargs == '?':
            return self._single_metavar(action, default_metavar)
        return super()._format_args(action, default_metavar)

    @staticmethod
    def _single_metavar(action, default_metavar):
        """Resolve the single-value placeholder for ``action`` (without brackets)."""
        if action.metavar is not None:
            return str(action.metavar)
        if action.choices is not None:
            return '{%s}' % ','.join(str(getattr(c, 'value', c)) for c in action.choices)
        return str(default_metavar)

    def _format_action_invocation(self, action):
        opts = action.option_strings
        if not opts:
            return ''
        # Show only: single-char short options (e.g. -v) + the long name.
        short_opts = [s for s in opts if s.startswith('-') and not s.startswith('--')]
        long_opts = [s for s in opts if s.startswith('--')]
        display = short_opts + long_opts[:1]
        if not display:
            return ''
        if action.nargs == 0:
            return ', '.join(display)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ', '.join('%s %s' % (s, args_string) for s in display)

    @staticmethod
    def _indent_text(text, indent='  '):
        return '\n'.join(indent + line if line else line for line in text.splitlines())

    def _pos_metavar(self, action):
        """Return the metavar of a positional action (choices -> {a,b,c})."""
        if isinstance(action, argparse._SubParsersAction):
            # 子命令（subparser）在 usage 中显示为 <dest>（如 <subcmd>、<scope>），
            # 不用 {a,b,c} 枚举子命令名。
            return '<%s>' % action.dest
        if action.metavar is not None:
            return str(action.metavar)
        if action.choices is not None:
            return '{%s}' % ','.join(str(c) for c in action.choices)
        return str(action.dest).upper()

    def _long_option(self, action):
        """Return the canonical long option string of an optional action."""
        if not action.option_strings:
            return ''
        long_opts = [s for s in action.option_strings if s.startswith('--')]
        return long_opts[0] if long_opts else action.option_strings[0]

    def _format_action_columns(self, action):
        """
        Split one action into the 3 help columns (spec 4.4.1):
          col1 = short option (e.g. '-c,'), empty when absent;
          col2 = long option + metavar (e.g. '--config <FILE>');
          col3 = description text.
        """
        opts = action.option_strings
        if not opts:
            return '', self._pos_metavar(action), (action.help or '')
        short_opts = [s for s in opts if s.startswith('-') and not s.startswith('--')]
        short_col = (short_opts[0] + ',') if short_opts else ''
        long_name = self._long_option(action)
        if action.nargs == 0:
            long_col = long_name
        else:
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            long_col = '%s %s' % (long_name, args_string)
        return short_col, long_col, (action.help or '')

    def _format_action_section(self, heading, actions):
        """
        Render one argument section with 3-column alignment; column widths are
        the maximum within the section (spec 4.4.1).
        """
        if not actions:
            return ''
        rows = []
        for action in actions:
            if action.help is argparse.SUPPRESS:
                continue
            subactions = list(self._iter_indented_subactions(action))
            if subactions:
                for sub in subactions:
                    # 子命令（subparser）没有 option_strings，直接显示子命令名。
                    name = sub.option_strings[0] if sub.option_strings else sub.dest
                    rows.append(('', name, sub.help or ''))
            else:
                rows.append(self._format_action_columns(action))
        if not rows:
            return ''
        # 过长的占位符/choices 会让第 3 列被推到很右，可读性差；
        # 当第 2 列超宽时，该行把描述换到下一行（缩进到第 3 列起点），
        # 且不参与列宽计算（只按正常行的最宽值对齐）。
        max_col2 = 52
        col1_width = max(len(r[0]) for r in rows)
        normal = [r for r in rows if len(r[1]) <= max_col2] or rows
        col2_width = max(len(r[1]) for r in normal)
        pad = 2 + col1_width + 1 + col2_width + 2
        width = max(80, pad + 40)
        help_width = max(1, width - pad)
        indent = ' ' * pad
        lines = [heading + ':']
        for short_col, long_col, help_text in rows:
            wrapped = textwrap.wrap(
                help_text or '',
                width=help_width,
                break_long_words=False,
                break_on_hyphens=False,
            ) or ['']
            if len(long_col) > max_col2:
                lines.append('  ' + short_col.ljust(col1_width) + ' ' + long_col)
                lines.extend(indent + line for line in wrapped)
                continue
            head = '  ' + short_col.ljust(col1_width) + ' ' + long_col.ljust(col2_width) + '  '
            lines.append(head + wrapped[0])
            lines.extend(indent + line for line in wrapped[1:])
        return '\n'.join(lines) + '\n\n'

    def _format_usage(self, usage, actions, groups, prefix):
        """
        Single-line usage template (spec 4.4.1):
          <tool> <subcmd> <required-args> [options]
        Required arguments are expanded, everything optional collapses to
        ``[options]``.
        """
        if not prefix:
            # argparse 内部推导子命令 prog 时以空串调用，保持默认渲染。
            return super()._format_usage(usage, actions, groups, prefix)
        parts = [self._prog]
        for action in actions:
            if not action.option_strings:
                parts.append(self._pos_metavar(action))
        has_optional = False
        for action in actions:
            if not action.option_strings or not action.required:
                if action.option_strings:
                    has_optional = True
                continue
            long_name = self._long_option(action)
            if action.nargs == 0:
                parts.append(long_name)
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                parts.append('%s %s' % (long_name, args_string))
        if has_optional:
            parts.append('[options]')
        return 'Usage:\n  ' + ' '.join(parts) + '\n\n'

    def format_help(self):
        """Render help in the unified section layout (spec section 4.4.1).

        Section order follows the spec: Description -> Usage -> Required
        arguments -> Optional arguments -> Examples -> Output.
        """
        usage, actions, groups = self._usage_args
        help_text = ''

        if self._description_text:
            help_text += 'Description:\n' + self._indent_text(self._description_text) + '\n\n'

        # argparse 内部会用 format_help() 推导子命令的 prog 前缀（prefix=''），
        # 此时保持默认渲染；只有真实帮助输出（prefix=None）才使用统一标题。
        prefix = 'Usage:\n  ' if self._usage_prefix is None else self._usage_prefix
        help_text += self._format_usage(usage, actions, groups, prefix)

        positionals = [a for a in actions if not a.option_strings]
        required = [a for a in actions if a.option_strings and a.required]
        optional = [a for a in actions if a.option_strings and not a.required]
        for heading, section_actions in (
            ('Positional arguments', positionals),
            ('Required arguments', required),
            ('Optional arguments', optional),
        ):
            help_text += self._format_action_section(heading, section_actions)

        if self._epilog_text:
            help_text += self._epilog_text + '\n'
        return help_text


def _add_log_level_args(parser: argparse.ArgumentParser) -> None:
    """Add unified log-level / verbosity switches (CLI spec section 4.2.3.1)."""
    parser.add_argument(
        '--log_level',
        dest='log_level',
        default=None,
        choices=['debug', 'info', 'warning', 'error'],
        help='Log level [default: info]',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbose',
        action='store_true',
        help='Increase output verbosity (equivalent to --log_level debug)',
    )
    parser.add_argument(
        '-q',
        '--quiet',
        dest='quiet',
        action='store_true',
        help='Suppress non-error output (equivalent to --log_level error)',
    )


def _apply_log_level(args) -> None:
    """
    Resolve the effective log level:
      * --log_level wins if given explicitly;
      * otherwise --verbose/-v (or --debug) -> debug, --quiet/-q -> error;
      * otherwise fall back to the configured env default.
    """
    log_level = getattr(args, 'log_level', None)
    verbose = getattr(args, 'verbose', False)
    quiet = getattr(args, 'quiet', False)
    debug = getattr(args, 'debug', False)

    if log_level is not None:
        set_logger_level(log_level)
    elif verbose or debug:
        set_logger_level('debug')
    elif quiet:
        set_logger_level('error')
    else:
        set_logger_level(msmodelslim_config.env_vars.log_level)


def _parse_timeout(value):
    """
    Parse the --timeout value.
      * Standard form: an integer number of seconds (per the unified CLI spec).
      * Legacy form: a duration string such as '1D', '2H', '3D4H' (handled
        downstream by convert_to_timedelta), kept for backward compatibility.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _warn_deprecated(argv: List[str]) -> None:
    """Emit a one-time deprecation warning for any legacy option in argv."""
    seen = set()
    for token in argv:
        opt = token.split('=', 1)[0]
        if opt in DEPRECATED_ALIASES and opt not in seen:
            seen.add(opt)
            get_logger().warning(
                "Option %r is deprecated; use %r instead. "
                "The old spelling remains supported for backward compatibility "
                "but will be removed in a future release.",
                opt,
                DEPRECATED_ALIASES[opt],
            )


def _normalize_analyze_argv(argv: List[str]) -> List[str]:
    """
    Backward-compatible argv normalization for `msmodelslim analyze`:
      - If user does not specify scope, default to `linear`.
      - Special-case compatibility: when scope is omitted and `--metrics attention_mse` is used,
        run under `attn` scope and treat it as `--metrics mse`, with a warning.

    We intentionally do NOT inject when user asks for help:
      - `msmodelslim analyze -h/--help` should show scopes help, not scope-specific help.
    """
    if not argv or 'analyze' not in argv:
        return argv

    idx = argv.index('analyze')
    tail = argv[idx + 1 :]

    # Any help request should keep `analyze` help clean (do not auto-inject scope).
    if '-h' in tail or '--help' in tail:
        return argv

    # If user already provided a scope, do nothing.
    if tail and tail[0] in ['linear', 'layer', 'attn', 'attn_head']:
        return argv

    if not tail or tail[0].startswith('-'):
        # Legacy compatibility: `--metrics attention_mse` was renamed to `attn --metrics mse`.
        if '--metrics' in tail:
            try:
                metrics_idx = tail.index('--metrics')
                metrics_val = tail[metrics_idx + 1].strip().lower()
            except Exception:
                metrics_val = None
            if metrics_val == 'attention_mse':
                # `attn` scope does not accept `--pattern`; drop it when converting legacy usage.
                if '--pattern' in tail:
                    pat_idx = tail.index('--pattern')
                    drop_end = pat_idx + 1
                    while drop_end < len(tail) and not tail[drop_end].startswith('-'):
                        drop_end += 1
                    dropped = tail[pat_idx:drop_end]
                    get_logger().warning(
                        "Legacy argument %r is ignored when converting to scope 'attn'. "
                        "Attention analysis runs on all attention modules by default.",
                        dropped,
                    )
                    tail = tail[:pat_idx] + tail[drop_end:]
                    argv = argv[: idx + 1] + tail

                get_logger().warning(
                    "Analyze metric 'attention_mse' is deprecated. "
                    "It has been renamed to scope 'attn' with metric 'mse'. "
                    "Please use: `msmodelslim analyze attn --metrics mse ...`"
                )
                new_argv = argv[:]
                # Replace attention_mse -> mse to satisfy argparse choices of `attn`.
                new_argv[idx + 1 + metrics_idx + 1] = 'mse'
                return new_argv[: idx + 1] + ['attn'] + new_argv[idx + 1 :]

        return argv[: idx + 1] + ['linear'] + argv[idx + 1 :]

    return argv


def _normalize_device_argv(argv: List[str]) -> List[str]:
    """
    Backward-compatible normalization for legacy `--device TYPE:IDX,IDX,...`.

    The device option is split into `--device {npu,cpu}` plus `--device_id`.
    Legacy scripts that pass indices inline
    (e.g. `--device npu:0,1`) are translated to the canonical form with a
    one-time deprecation warning. An explicit `--device_id` option is left
    untouched (it takes precedence). Also applies to `analyze` so legacy
    `--device npu:0,1` enables multi-device DP via `--device_id`.
    """
    if not argv or argv[0] not in ('quant', 'tune', 'analyze'):
        return argv
    if not any(arg == '--device' or arg.startswith('--device=') for arg in argv):
        return argv
    if '--device_id' in argv:
        return argv

    warned = False

    def _warn_once(value: str) -> None:
        nonlocal warned
        if not warned:
            get_logger().warning(
                "Option '--device %s' is deprecated; use '--device %s --device_id %s' instead. "
                "The legacy device:index format remains supported for backward compatibility "
                "but will be removed in a future release.",
                value,
                value.split(':', 1)[0],
                value.split(':', 1)[1].replace(',', ' '),
            )
            warned = True

    result = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith('--device='):
            value = arg.split('=', 1)[1]
            if ':' in value:
                _warn_once(value)
                type_str, idx_str = value.split(':', 1)
                result.extend(['--device', type_str, '--device_id'] + idx_str.split(','))
            else:
                result.append(arg)
        elif arg == '--device':
            result.append(arg)
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is not None and ':' in nxt and not nxt.startswith('-'):
                _warn_once(nxt)
                type_str, idx_str = nxt.split(':', 1)
                result.extend([type_str, '--device_id'] + idx_str.split(','))
                i += 1
        else:
            result.append(arg)
        i += 1
    return result


def _is_help_request(argv: List[str]) -> bool:
    """Check if the command line arguments contain help request."""
    return '-h' in argv or '--help' in argv


def _is_version_request(argv: List[str]) -> bool:
    """Check if the command line arguments contain a version request.

    Detects `--version` / `-V` anywhere in argv (including after a subcommand,
    e.g. `msmodelslim quant --version`), so the version banner is printed
    regardless of position and no stale argparse branch is needed.
    """
    return '--version' in argv or '-V' in argv


def main():
    argv = sys.argv[1:]

    # Handle version request before printing the startup logo / parsing.
    if _is_version_request(argv):
        _print_version()
        sys.exit(0)

    # Emit deprecation warnings for legacy spellings before parsing.
    _warn_deprecated(argv)

    set_logger_level(msmodelslim_config.env_vars.log_level)

    # Print logo at startup, except when help is requested
    if not _is_help_request(argv):
        print_logo()

    parser = argparse.ArgumentParser(
        prog='msmodelslim',
        formatter_class=_UnifiedHelpFormatter,
        description=f"MsModelSlim(MindStudio Model-Quantization Tools), "
        f"{MIND_STUDIO_LOGO}.\n"
        "Providing functions such as model quantization and compression "
        "based on Ascend.\n"
        f"For any issue, refer to the FAQ at {FAQ_URL}",
        epilog="Examples:\n"
        "  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu "
        "--model_type Qwen2.5-7B-Instruct --quant_type w8a8 --trust_remote_code True\n"
        "  msmodelslim analyze linear --model_path ${MODEL_PATH} --model_type Qwen2.5-7B-Instruct\n"
        "  msmodelslim tune --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --config ${CONFIG} "
        "--device npu --model_type Qwen3-32B",
    )
    parser.add_argument(
        '--version',
        '-V',
        action='store_true',
        help='Show version information and exit',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ------------------------------------------------------------------
    # Quant command
    # ------------------------------------------------------------------
    quant_parser = subparsers.add_parser(
        'quant',
        help='Model quantization',
        formatter_class=_UnifiedHelpFormatter,
        description='Quantize a model (W4A4/W8A8/etc.) or convert model weights, and save the results.',
        epilog='Examples:\n'
        '  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} '
        '--device npu --model_type Qwen2.5-7B-Instruct --quant_type w8a8 --trust_remote_code True\n'
        '  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} '
        '--device npu --model_type ${MODEL_TYPE} --config ${CONFIG_PATH} --trust_remote_code ${TRUST_REMOTE_CODE}\n'
        'Output:\n'
        '  Quantized model is written to the directory given by --save_path.',
    )
    quant_parser.add_argument(
        '--model_type',
        dest='model_type',
        metavar='<MODEL_TYPE>',
        required=False,
        default=None,
        help="Type of model to quantize (e.g. 'Qwen2.5-7B-Instruct'). "
        "Optional when --config uses apiversion modelslim_convert (weight convert needs only --model_path).",
    )
    quant_parser.add_argument(
        '--model_path',
        dest='model_path',
        metavar='<PATH>',
        required=True,
        type=str,
        help='Path to the original model',
    )
    quant_parser.add_argument(
        '--save_path',
        dest='save_path',
        metavar='<PATH>',
        required=True,
        type=str,
        help='Path to save the quantized model',
    )
    quant_parser.add_argument(
        '--device',
        dest='device',
        type=str,
        default='npu',
        choices=[d.value for d in DeviceType],
        help='Target device type for quantization [default: npu]',
    )
    quant_parser.add_argument(
        '--device_id',
        dest='device_id',
        nargs='*',
        type=int,
        metavar='<ID>',
        default=None,
        help='Device index (integer) to use for quantization, e.g. 0 or 0 1 2 3',
    )
    quant_parser.add_argument(
        '-c',
        '--config',
        '--config_path',
        dest='config_path',
        metavar='<FILE>',
        type=str,
        help='Explicit path to quantization config file',
    )
    quant_parser.add_argument(
        '--quant_type',
        dest='quant_type',
        type=QuantType,
        choices=QuantType,
        help='Type of quantization to apply',
    )
    quant_parser.add_argument(
        '--trust_remote_code',
        dest='trust_remote_code',
        nargs='?',
        const=True,
        type=_cli_convert_to_bool,
        default=False,
        metavar='<BOOL>',
        help='Trust custom code loaded from the model directory '
        "[default: false]. Pass true/false explicitly for backward "
        "compatibility. Please ensure the security of the loaded custom code file.",
    )
    quant_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode for context recording',
    )
    quant_parser.add_argument(
        '--tags',
        '--tag',
        dest='tag',
        nargs='*',
        metavar='<TAG>',
        default=None,
        help="Optional tags to match configs with verified scenario tags (e.g. vLLM-Ascend Atlas_A2_Inference). "
        "User can add multiple tags; matching requires all tags to appear in the same scenario. "
        "If this parameter is specified without a hardware type tag, the current device type is matched automatically.",
    )
    _add_log_level_args(quant_parser)

    # ------------------------------------------------------------------
    # Analyze command
    # ------------------------------------------------------------------
    analysis_parser = subparsers.add_parser(
        'analyze',
        help='Model quantization sensitivity analyze tool',
        formatter_class=_UnifiedHelpFormatter,
        description='Analyze quantization sensitivity by scope: linear | layer | attn.',
        epilog='Examples:\n'
        '  msmodelslim analyze linear --model_path ${MODEL_PATH} --model_type Qwen2.5-7B-Instruct\n'
        '  msmodelslim analyze layer --model_path ${MODEL_PATH} --model_type Qwen2.5-7B-Instruct '
        '--quant_modules model.layers.0.self_attn.*\n'
        'Output:\n'
        '  Analysis results are logged to the console/stdout.',
    )

    analyze_common_parser = argparse.ArgumentParser(add_help=False)
    analyze_common_parser.add_argument(
        '--model_type',
        dest='model_type',
        metavar='<MODEL_TYPE>',
        required=True,
        help="Type of model to analyze (e.g. 'Qwen2.5-7B-Instruct', 'DeepSeek-V3')",
    )
    analyze_common_parser.add_argument(
        '--model_path',
        dest='model_path',
        metavar='<PATH>',
        required=True,
        type=str,
        help='Path to the original model',
    )
    analyze_common_parser.add_argument(
        '--device',
        dest='device',
        type=DeviceType,
        default=DeviceType.NPU,
        choices=DeviceType,
        help='Target device type for Analysis [default: npu]. '
        'For multi-device DP use --device_id 0 1 2 3 (legacy --device npu:0,1 still works).',
    )
    analyze_common_parser.add_argument(
        '--device_id',
        dest='device_id',
        nargs='*',
        type=int,
        metavar='<ID>',
        default=None,
        help='Device indices for analysis; length > 1 enables DPLayerWiseRunner, e.g. 0 1 2 3',
    )
    analyze_common_parser.add_argument(
        '--calibration_dataset',
        '--calib_dataset',
        dest='calib_dataset',
        metavar='<PATH>',
        type=str,
        default='mix_calib.jsonl',
        help='Calibration dataset. LLM: a .json/.jsonl file — the filename under lab_calib '
        '(default mix_calib.jsonl) or a path to that file, not the parent directory. '
        'VLM: a multimodal dataset directory name under lab_calib (e.g. calibImages) '
        'or a path to that directory.',
    )
    analyze_common_parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Path to save result file (YAML for linear/layer/attn). '
        'If not specified, results are printed to console only.',
    )
    analyze_common_parser.add_argument(
        '--top_k',
        '--topk',
        dest='topk',
        metavar='<N>',
        type=int,
        default=15,
        help='Number of top layers to output for disable_names [default: 15]',
    )
    analyze_common_parser.add_argument(
        '--trust_remote_code',
        dest='trust_remote_code',
        nargs='?',
        const=True,
        type=_cli_convert_to_bool,
        default=False,
        metavar='<BOOL>',
        help='Trust custom code loaded from the model directory '
        "[default: false]. Pass true/false explicitly for backward "
        "compatibility. Please ensure the security of the loaded custom code file.",
    )
    _add_log_level_args(analyze_common_parser)

    analysis_subparsers = analysis_parser.add_subparsers(dest='scope', help='Analyze scopes')
    analysis_subparsers.required = True

    analysis_linear_parser = analysis_subparsers.add_parser(
        'linear',
        parents=[analyze_common_parser],
        help='Analyze individual linear layers; use --patterns to filter what gets listed',
        formatter_class=_UnifiedHelpFormatter,
        description='Analyze individual linear layers. Use --patterns to filter what gets listed.',
        epilog='Examples:\n'
        '  msmodelslim analyze linear --model_path ${MODEL_PATH} --model_type Qwen2.5-7B-Instruct\n'
        '  msmodelslim analyze linear --model_path ${MODEL_PATH} --model_type Qwen2.5-7B-Instruct '
        '--metrics std --patterns mlp*',
    )
    analysis_linear_parser.add_argument(
        '--metrics',
        dest='metrics',
        type=str,
        choices=['std', 'quantile', 'kurtosis'],
        default='kurtosis',
        help='Analysis metrics [default: kurtosis]',
    )
    analysis_linear_parser.add_argument(
        '--patterns',
        '--pattern',
        dest='pattern',
        nargs='*',
        metavar='<PATTERN>',
        default=['*'],
        help='Pattern list to filter displayed linear layers [default: ["*"]]',
    )

    analysis_layer_parser = analysis_subparsers.add_parser(
        'layer',
        parents=[analyze_common_parser],
        help='Analyze layer/block as a group; --quant_modules selects modules to include in pipeline config',
        formatter_class=_UnifiedHelpFormatter,
        description='Analyze layer/block as a group. '
        '--quant_modules selects modules to include in the pipeline config.',
        epilog='Examples:\n'
        '  msmodelslim analyze layer --model_path ${MODEL_PATH} --model_type Qwen2.5-7B-Instruct '
        '--metrics mse_model_wise\n'
        '  msmodelslim analyze layer --model_path ${MODEL_PATH} --model_type Qwen2.5-7B-Instruct '
        '--quant_modules model.layers.0.self_attn.*',
    )
    analysis_layer_parser.add_argument(
        '--metrics',
        dest='metrics',
        type=str,
        choices=['mse_model_wise', 'mse_layer_wise'],
        default='mse_layer_wise',
        help='Analysis metrics, e.g. mse_model_wise [default: mse_layer_wise]',
    )
    analysis_layer_parser.add_argument(
        '--quant_modules',
        dest='quant_modules',
        nargs='*',
        metavar='<MODULE>',
        default=['*'],
        help='Quant modules list that maps to pipeline scope [default: ["*"]]',
    )

    analysis_attn_parser = analysis_subparsers.add_parser(
        'attn',
        parents=[analyze_common_parser],
        help='Analyze attention modules with mse metric (scope defaults to all attention modules)',
        formatter_class=_UnifiedHelpFormatter,
        description='Analyze attention modules with the mse metric (scope defaults to all attention modules).',
        epilog='Examples:\n  msmodelslim analyze attn --model_path ${MODEL_PATH} --model_type Qwen2.5-7B-Instruct',
    )
    analysis_attn_parser.add_argument(
        '--metrics',
        dest='metrics',
        type=str,
        choices=['mse'],
        default='mse',
        help='Analysis metrics [default: mse]',
    )

    analysis_attn_head_parser = analysis_subparsers.add_parser(
        'attn_head',
        parents=[analyze_common_parser],
        help='Analyze attention heads with ra_compress metric (induction/echo head selection)',
        formatter_class=_UnifiedHelpFormatter,
        description='Analyze attention heads with the ra_compress metric (induction/echo head selection).',
        epilog='Examples:\n  msmodelslim analyze attn_head --model_path ${MODEL_PATH} --model_type Qwen2.5-7B-Instruct',
    )
    analysis_attn_head_parser.add_argument(
        '--metrics',
        type=str,
        choices=['ra_compress'],
        default='ra_compress',
        help='Analysis metrics [default: ra_compress]',
    )

    # ------------------------------------------------------------------
    # auto tuning command
    # ------------------------------------------------------------------
    tuning_parser = subparsers.add_parser(
        'tune',
        help='Model quantization auto tuning tool',
        formatter_class=_UnifiedHelpFormatter,
        description='Automatically tune quantization configs to satisfy the target accuracy.',
        epilog='Examples:\n'
        '  msmodelslim tune --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --config ${CONFIG} '
        '--device npu --model_type Qwen3-32B\n'
        '  msmodelslim tune --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --config ${CONFIG} '
        '--device npu --device_id 0 --timeout 3600\n'
        'Output:\n'
        '  Tuning results are written to the directory given by --save_path.',
    )
    tuning_parser.add_argument(
        '--model_type',
        dest='model_type',
        metavar='<MODEL_TYPE>',
        type=str,
        default='default',
        help="Type of model to quantize (e.g. 'Qwen2.5-7B-Instruct', 'Qwen3-32B')",
    )
    tuning_parser.add_argument(
        '--model_path',
        dest='model_path',
        metavar='<PATH>',
        required=True,
        type=str,
        help='Path to the original model',
    )
    tuning_parser.add_argument(
        '--save_path',
        dest='save_path',
        metavar='<PATH>',
        required=True,
        type=str,
        help='Path to save tuning results',
    )
    tuning_parser.add_argument(
        '-c',
        '--config',
        dest='config',
        metavar='<FILE>',
        required=True,
        type=str,
        help='Path to tuning config file',
    )
    tuning_parser.add_argument(
        '--device',
        dest='device',
        type=str,
        default='npu',
        choices=[d.value for d in DeviceType],
        help='Target device type for tuning [default: npu]',
    )
    tuning_parser.add_argument(
        '--device_id',
        dest='device_id',
        nargs='*',
        type=int,
        metavar='<ID>',
        default=None,
        help='Device index (integer) to use for tuning, e.g. 0 or 0 1 2 3',
    )
    tuning_parser.add_argument(
        '--timeout',
        dest='timeout',
        type=_parse_timeout,
        metavar='<SECONDS>',
        default=None,
        help='Timeout for tuning, in seconds. '
        'Legacy duration strings such as 1D, 2H, 3D4H are still accepted '
        'for backward compatibility.',
    )
    tuning_parser.add_argument(
        '--trust_remote_code',
        dest='trust_remote_code',
        nargs='?',
        const=True,
        type=_cli_convert_to_bool,
        default=False,
        metavar='<BOOL>',
        help='Trust custom code loaded from the model directory '
        "[default: false]. Pass true/false explicitly for backward "
        "compatibility. Please ensure the security of the loaded custom code file.",
    )
    _add_log_level_args(tuning_parser)

    # 兼容旧设备写法：--device npu:0,1 → --device npu --device_id 0 1
    argv = _normalize_device_argv(argv)
    if argv[:1] == ['analyze']:
        argv = _normalize_analyze_argv(argv)
    args = parser.parse_args(argv)

    # Resolve the effective log level based on the unified log switches.
    _apply_log_level(args)

    if args.command == 'quant':
        from msmodelslim.cli.naive_quantization.__main__ import main as quant_main

        quant_main(args)
    elif args.command == 'analyze':
        from msmodelslim.cli.analysis.__main__ import main as analysis_main

        analysis_main(args)
    elif args.command == 'tune':
        from msmodelslim.cli.auto_tuning.__main__ import main as tuning_main

        tuning_main(args)
    else:
        # 可扩展其他组件
        parser.print_help()


if __name__ == '__main__':
    main()
