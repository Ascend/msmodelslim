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

import argparse
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import List

import msmodelslim  # noqa
from msmodelslim.cli.logo import print_logo
from msmodelslim.core.const import DeviceType, QuantType
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.logging import get_logger, set_logger_level

FAQ_HOME = "gitcode repo: Ascend/msmodelslim, wiki"
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
    """Return the msmodelslim repository root (dir containing .git or setup.py)."""
    cur = Path(__file__).resolve().parent
    for parent in cur.parents:
        if (parent / '.git').exists() or (parent / 'setup.py').exists():
            return parent
    return cur


def _get_version() -> str:
    """Return the installed package version, falling back to 'unknown'."""
    # 1) Prefer the metadata registered by the installed distribution.
    try:
        from importlib.metadata import version as _pkg_version  # type: ignore

        pkg_version = _pkg_version('msmodelslim')
    except Exception:  # pragma: no cover - not installed as a package
        pkg_version = ''
    if pkg_version:
        return pkg_version
    # 2) Fall back to reading `__version__` from the repo setup.py without importing it
    #    (importing setup.py has heavy build side effects we must avoid).
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
    """
    Return the current git commit hash (>= 7 chars) or empty string.

    Uses subprocess only to run ``git rev-parse HEAD`` with a fixed argument
    list, no shell and no user-controlled input (hence the nosec exemptions).
    """
    try:
        out = subprocess.check_output(  # nosec B603, B607
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(_repo_root()),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()[:12]
    except Exception:
        return ''


def _print_version() -> None:
    """Print the unified version banner (MindStudio CLI spec section 4.5)."""
    version = _get_version()
    git_hash = _get_git_hash()
    print_logo()
    sys.stdout.write(
        f"msmodelslim {version} ({git_hash})\n"
        "Copyright (C) 2026 Huawei Technologies Co., Ltd.\n"
        "License: Mulan PSL v2.\n"
    )
    if git_hash:
        sys.stdout.write(
            f"\nBuild Info:\n  GitCommit : {git_hash}\n  Repo      : https://gitcode.com/Ascend/msmodelslim\n"
        )


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

    def _format_action_section(self, heading, actions):
        """Render one argument section (heading + action lines) like argparse does."""
        if not actions:
            return ''
        root = self._Section(self, None, None)
        section = self._Section(self, root, heading)
        for action in actions:
            if action.help is not argparse.SUPPRESS:
                invocations = [self._format_action_invocation(action)]
                for subaction in self._iter_indented_subactions(action):
                    invocations.append(self._format_action_invocation(subaction))
                self._action_max_length = max(
                    self._action_max_length,
                    max(len(invocation) for invocation in invocations) + self._current_indent,
                )
                section.items.append((self._format_action, [action]))
        return section.format_help()

    def format_help(self):
        """Render help in the unified section layout (spec section 4.4.1)."""
        usage, actions, groups = self._usage_args
        # argparse 内部会用 format_help() 推导子命令的 prog 前缀（prefix=''），
        # 此时保持默认渲染；只有真实帮助输出（prefix=None）才使用统一标题。
        prefix = 'Usage:\n  ' if self._usage_prefix is None else self._usage_prefix
        help_text = self._format_usage(usage, actions, groups, prefix)

        if self._description_text:
            help_text += 'Description:\n' + self._indent_text(self._description_text) + '\n\n'

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
    untouched (it takes precedence), and `analyze` is not affected because it
    never supported the colon format.
    """
    if not argv or argv[0] not in ('quant', 'tune'):
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
        f"For any issue, refer FAQ first: {FAQ_HOME}",
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
        description='Quantize a model (W4A4/W8A8/etc.) and save the quantized weights.',
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
        help="Optional tags to match configs with verified scenario tags (e.g. mindie Atlas_A2_Inference, vllm cpu). "
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
        '--quant_modules lm_head\n'
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
        help='Target device type for Analysis [default: npu]',
    )
    analyze_common_parser.add_argument(
        '--calibration_dataset',
        '--calib_dataset',
        dest='calib_dataset',
        metavar='<FILE>',
        type=str,
        default='mix_calib.jsonl',
        help='Calibration dataset file path or filename in lab_calib directory. '
        'Supports .json and .jsonl formats [default: mix_calib.jsonl]',
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
        help='Number of top layers to output for disable_names [default: 15, empirical value, for reference only]',
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
        '  msmodelslim analyze layer --model_path ${MODEL_PATH} --model_type Qwen2.5-7B-Instruct\n'
        '  msmodelslim analyze layer --model_path ${MODEL_PATH} --model_type Qwen2.5-7B-Instruct '
        '--quant_modules lm_head',
    )
    analysis_layer_parser.add_argument(
        '--metrics',
        dest='metrics',
        type=str,
        choices=['mse_model_wise', 'mse_layer_wise'],
        default='mse_layer_wise',
        help='Analysis metrics [default: mse_layer_wise]',
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
    )
    analysis_attn_head_parser.add_argument(
        '--metrics',
        type=str,
        choices=['ra_compress'],
        default='ra_compress',
        help='Analysis metrics: ra_compress (default: ra_compress)',
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
